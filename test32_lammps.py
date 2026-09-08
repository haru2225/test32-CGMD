#!/usr/bin/env python3
"""Run all-atom score MD or Si-bead CGMD in LAMMPS.

LAMMPS performs velocity-Verlet/Langevin integration.  A supported
``fix external pf/callback`` callback evaluates the test32 NequIP denoiser,
converts its displacement prediction to a score, and returns ``k_B*T*score``
as force in LAMMPS ``metal`` units.

The model consumes the whole configuration, so this implementation requires
exactly one MPI rank.  The PyTorch evaluation itself can run on one GPU.
No potential energy or virial is available; use NVT/Langevin dynamics, not
minimization, NPT, or energy-conservation analysis.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from test32_forcefield import (
    FORMAT_NAME,
    KB_EV_PER_K,
    build_model,
    graph_data,
    json_safe,
    torch_load,
)


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "lammps-output")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--temperature-k", type=float, default=None)
    parser.add_argument("--sigma-ref-angstrom", type=float, default=None)
    parser.add_argument("--force-clip", type=float, default=None)
    parser.add_argument("--timestep-ps", type=float, default=None)
    parser.add_argument("--damping-ps", type=float, default=None)
    parser.add_argument("--thermo-every", type=int, default=100)
    parser.add_argument("--dump-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1339)
    parser.add_argument(
        "--lammps-library-name",
        default=None,
        help="Optional suffix for liblammps_NAME.so, for example mpi.",
    )
    args = parser.parse_args()
    for name in ("steps", "thermo_every", "dump_every"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def quoted(path: Path) -> str:
    # LAMMPS accepts a double-quoted path. Reject embedded quotes instead of
    # attempting shell-style escaping inside the LAMMPS command language.
    resolved = str(path.resolve())
    if '"' in resolved:
        raise RuntimeError(f"LAMMPS paths cannot contain a double quote: {resolved}")
    return f'"{resolved}"'


def load_bundle(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch_load(path, map_location="cpu")
    required = {
        "format",
        "kind",
        "architecture",
        "model_state_dict",
        "species_by_id",
        "atomic_numbers_by_id",
        "sigma_ref_angstrom",
        "temperature_k",
        "force_clip_ev_per_angstrom",
        "cell_angstrom",
    }
    if not isinstance(payload, dict) or payload.get("format") != FORMAT_NAME:
        raise RuntimeError(f"Not a {FORMAT_NAME} bundle: {path}")
    missing = required.difference(payload)
    if missing:
        raise RuntimeError(f"Force pencils bundle lacks fields: {sorted(missing)}")
    model = build_model(payload["architecture"], device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


class ExternalScoreForce:
    """Whole-configuration force provider passed to LAMMPS."""

    def __init__(
        self,
        model: torch.nn.Module,
        payload: dict[str, Any],
        box_lengths: np.ndarray,
        device: torch.device,
        temperature_k: float,
        sigma_ref_angstrom: float,
        force_clip: float,
    ):
        self.model = model
        self.device = device
        self.species = torch.as_tensor(payload["species_by_id"], dtype=torch.long, device=device)
        self.atomic_numbers = np.asarray(payload["atomic_numbers_by_id"], dtype=np.int64)
        self.cell = torch.diag(
            torch.as_tensor(box_lengths, dtype=torch.float32, device=device)
        )
        self.cutoff = float(payload["architecture"]["cutoff_angstrom"])
        self.temperature_k = float(temperature_k)
        self.sigma_ref_angstrom = float(sigma_ref_angstrom)
        self.force_clip = float(force_clip)
        self.natoms = len(self.atomic_numbers)
        self.calls = 0
        self.last_max_force = 0.0
        self.last_rms_force = 0.0

    @torch.no_grad()
    def evaluate(self, tags: np.ndarray, positions: np.ndarray) -> np.ndarray:
        if len(tags) != self.natoms:
            raise RuntimeError(
                f"The callback sees {len(tags)}/{self.natoms} atoms. "
                "Run this whole-configuration model with one MPI rank."
            )
        order = np.argsort(np.asarray(tags))
        sorted_tags = np.asarray(tags, dtype=np.int64)[order]
        expected = np.arange(1, self.natoms + 1, dtype=np.int64)
        if not np.array_equal(sorted_tags, expected):
            raise RuntimeError("LAMMPS atom IDs must remain consecutive from 1 to N")

        sorted_positions = np.asarray(positions, dtype=np.float32)[order]
        position_tensor = torch.as_tensor(sorted_positions, dtype=torch.float32, device=self.device)
        data = graph_data(
            position_tensor,
            self.cell,
            self.species,
            self.atomic_numbers,
            self.cutoff,
        )
        predicted_displacement = self.model(data)
        score = -predicted_displacement / (self.sigma_ref_angstrom**2)
        force = KB_EV_PER_K * self.temperature_k * score

        # Translation invariance requires zero total internal force.  The raw
        # vector denoiser does not enforce it exactly, so project out COM force.
        force -= force.mean(dim=0, keepdim=True)
        norms = torch.linalg.vector_norm(force, dim=-1, keepdim=True).clamp_min(1.0e-12)
        force *= torch.clamp(self.force_clip / norms, max=1.0)
        force -= force.mean(dim=0, keepdim=True)
        if not torch.isfinite(force).all():
            raise RuntimeError("The score model returned a non-finite force")

        self.calls += 1
        self.last_max_force = float(torch.linalg.vector_norm(force, dim=-1).max().cpu())
        self.last_rms_force = float(torch.sqrt(torch.mean(force.square())).cpu())
        sorted_force = force.cpu().numpy().astype(np.float64, copy=False)
        local_force = np.empty_like(sorted_force)
        local_force[order] = sorted_force
        return local_force


def external_force_callback(
    caller: ExternalScoreForce,
    ntimestep: int,
    nlocal: int,
    tag: np.ndarray,
    x: np.ndarray,
    fexternal: np.ndarray,
) -> None:
    """Signature required by lammps.set_fix_external_callback()."""
    if nlocal != caller.natoms:
        raise RuntimeError(
            f"fix external callback has nlocal={nlocal}, expected {caller.natoms}; "
            "use exactly one MPI rank"
        )
    fexternal[:, :] = caller.evaluate(tag, x)


def validate_box(payload: dict[str, Any], box_lengths: np.ndarray) -> None:
    training_cell = np.asarray(payload["cell_angstrom"], dtype=np.float64)
    diagonal = np.diag(np.diag(training_cell))
    if not np.allclose(training_cell, diagonal, atol=1.0e-7):
        raise RuntimeError("The force bundle contains a non-orthorhombic box")
    reference = np.diag(training_cell)
    if not np.allclose(box_lengths, reference, rtol=1.0e-5, atol=1.0e-5):
        raise RuntimeError(
            "LAMMPS box does not match the force-field training box: "
            f"LAMMPS={box_lengths.tolist()}, model={reference.tolist()}"
        )


def main() -> int:
    args = parse_args()
    bundle_path = args.bundle.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    model, payload = load_bundle(bundle_path, device)

    if args.data is None:
        suggested = payload.get("suggested_data_file")
        if not suggested:
            raise RuntimeError("Bundle has no suggested_data_file; pass --data")
        data_path = bundle_path.parent / str(suggested)
    else:
        data_path = args.data.expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"LAMMPS data file not found: {data_path}")

    temperature = float(
        payload["temperature_k"] if args.temperature_k is None else args.temperature_k
    )
    sigma_ref = float(
        payload["sigma_ref_angstrom"]
        if args.sigma_ref_angstrom is None
        else args.sigma_ref_angstrom
    )
    force_clip = float(
        payload["force_clip_ev_per_angstrom"]
        if args.force_clip is None
        else args.force_clip
    )
    timestep_ps = float(
        payload.get("suggested_timestep_ps", 0.001)
        if args.timestep_ps is None
        else args.timestep_ps
    )
    damping_ps = float(
        payload.get("suggested_damping_ps", 1.0)
        if args.damping_ps is None
        else args.damping_ps
    )
    for name, value in (
        ("temperature", temperature),
        ("sigma_ref", sigma_ref),
        ("force_clip", force_clip),
        ("timestep", timestep_ps),
        ("damping", damping_ps),
    ):
        if value <= 0:
            raise RuntimeError(f"{name} must be positive, got {value}")

    try:
        from lammps import lammps
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "LAMMPS's Python module/shared library is unavailable. Rebuild the supplied "
            "Singularity image or install a shared-library LAMMPS build and matching "
            "Python module."
        ) from exc

    trajectory_path = output_dir / f"{payload['kind']}.lammpstrj"
    log_path = output_dir / f"{payload['kind']}.log"
    final_data_path = output_dir / f"{payload['kind']}_final.data"
    constructor_args: dict[str, Any] = {
        "cmdargs": ["-log", str(log_path), "-screen", "none"]
    }
    if args.lammps_library_name:
        constructor_args["name"] = args.lammps_library_name
    lmp = lammps(**constructor_args)
    started = time.monotonic()
    try:
        lmp.commands_list(
            [
                "clear",
                "units metal",
                "dimension 3",
                "boundary p p p",
                "atom_style atomic",
                f"read_data {quoted(data_path)}",
                f"pair_style zero {float(payload['architecture']['cutoff_angstrom'])}",
                "pair_coeff * *",
                "neighbor 0.3 bin",
                "neigh_modify every 1 delay 0 check yes",
                "fix scoreforce all external pf/callback 1 1",
                "fix_modify scoreforce energy no virial no",
                "fix integrate all nve",
                (
                    f"fix thermostat all langevin {temperature} {temperature} "
                    f"{damping_ps} {args.seed} zero yes"
                ),
                f"timestep {timestep_ps}",
                f"velocity all create {temperature} {args.seed + 1} mom yes rot yes dist gaussian",
                f"thermo {args.thermo_every}",
                "thermo_style custom step atoms temp ke",
                f"dump trajectory all custom {args.dump_every} {quoted(trajectory_path)} id type x y z",
                "dump_modify trajectory sort id",
            ]
        )
        nprocs = int(lmp.extract_global("nprocs"))
        if nprocs != 1:
            raise RuntimeError(f"This model requires one MPI rank; LAMMPS started with {nprocs}")
        natoms = int(lmp.get_natoms())
        if natoms != int(payload["num_atoms"]):
            raise RuntimeError(
                f"Data file has {natoms} atoms, force bundle expects {payload['num_atoms']}"
            )
        boxlo, boxhi, xy, yz, xz, periodicity, box_change = lmp.extract_box()
        if max(abs(float(xy)), abs(float(yz)), abs(float(xz))) > 1.0e-10:
            raise RuntimeError("Only an orthorhombic LAMMPS box is currently supported")
        box_lengths = np.asarray(boxhi, dtype=np.float64) - np.asarray(boxlo, dtype=np.float64)
        validate_box(payload, box_lengths)
        provider = ExternalScoreForce(
            model=model,
            payload=payload,
            box_lengths=box_lengths,
            device=device,
            temperature_k=temperature,
            sigma_ref_angstrom=sigma_ref,
            force_clip=force_clip,
        )
        lmp.set_fix_external_callback("scoreforce", external_force_callback, provider)
        print(
            f"LAMMPS {payload['kind']}: N={natoms}, device={device}, T={temperature} K, "
            f"sigma_ref={sigma_ref} A, dt={timestep_ps} ps, steps={args.steps}",
            flush=True,
        )
        print(f"WARNING: {payload.get('scientific_caveat', 'experimental score force')}")
        lmp.command(f"run {args.steps}")
        lmp.command(f"write_data {quoted(final_data_path)}")
    finally:
        lmp.close()

    elapsed = time.monotonic() - started
    metrics = {
        "bundle": str(bundle_path),
        "data": str(data_path),
        "kind": payload["kind"],
        "num_atoms": int(payload["num_atoms"]),
        "steps": args.steps,
        "temperature_k": temperature,
        "sigma_ref_angstrom": sigma_ref,
        "force_clip_ev_per_angstrom": force_clip,
        "timestep_ps": timestep_ps,
        "damping_ps": damping_ps,
        "device": str(device),
        "callback_calls": provider.calls,
        "last_max_force_ev_per_angstrom": provider.last_max_force,
        "last_rms_force_ev_per_angstrom": provider.last_rms_force,
        "elapsed_seconds": elapsed,
        "trajectory": str(trajectory_path),
        "log": str(log_path),
        "final_data": str(final_data_path),
        "scientific_caveat": payload.get("scientific_caveat"),
    }
    (output_dir / "run_metrics.json").write_text(
        json.dumps(json_safe(metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Trajectory: {trajectory_path}")
    print(f"Final data: {final_data_path}")
    print(f"Metrics: {output_dir / 'run_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
