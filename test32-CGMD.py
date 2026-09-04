#!/usr/bin/env python3
"""Generate amorphous SiO2 from a trained test32 checkpoint and export CG data.

This is the supercomputer-portable version of ``toy-model/test32-CGMD.py``.
It intentionally does not train a model. It loads the state-dict checkpoint
created by test32.py, runs DM2's noisy-denoising and polish stages, and writes
the Si-only trajectory consumed by the ScoreMD CG pipeline.

All paths are command-line arguments or are resolved relative to the cloned
DM2 repository. Generation restart data is saved periodically, so submitting
the same PBS file again continues an interrupted 12-hour job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Optional

import ase.io
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase import Atoms
from ase.neighborlist import primitive_neighbor_list
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch_geometric.data import Data

if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([slice])

from graphite.nn.basis import bessel
from graphite.nn.models.e3nn_nequip import NequIP


warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.jit._check")

SCRIPT_DIR = Path(__file__).resolve().parent
DM2_ROOT = Path(os.environ.get("DM2_ROOT", SCRIPT_DIR.parents[1])).resolve()
DEFAULT_INPUT = DM2_ROOT / "demo" / "demo_training" / "simu_data" / "sio2_3000_glass_1k_sample0.dat"
DEFAULT_CHECKPOINT = DM2_ROOT / "demo" / "model" / "test32_sio2_glass_nequip.pt"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "test32-CGMD-output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--gen-noisy-steps", type=int, default=2_900)
    parser.add_argument("--gen-polish-steps", type=int, default=100)
    parser.add_argument("--gen-max-sigma", type=float, default=1.0)
    parser.add_argument("--generation-checkpoint-steps", type=int, default=100)
    parser.add_argument(
        "--time-budget-hours",
        type=float,
        default=11.5,
        help="Save restart data and exit before PBS walltime (0 disables the guard).",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--allow-cpu", action="store_true", help="Only for a small local smoke test.")
    parser.add_argument("--force", action="store_true", help="Overwrite a completed run.")
    args = parser.parse_args()

    if args.cutoff <= 0:
        parser.error("--cutoff must be positive")
    if args.gen_noisy_steps < 0 or args.gen_polish_steps < 0:
        parser.error("Generation step counts cannot be negative")
    if args.generation_checkpoint_steps < 1:
        parser.error("--generation-checkpoint-steps must be positive")
    return args


def torch_load(path: Path, map_location: str | torch.device = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def atomic_torch_save(payload, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def capture_rng_state() -> dict:
    state = {"torch": torch.get_rng_state(), "numpy": np.random.get_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[dict]) -> None:
    if not state:
        return
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def deadline_reached(deadline: Optional[float]) -> bool:
    # Leave two minutes for writing restart data and clean Singularity exit.
    return deadline is not None and time.monotonic() >= deadline - 120.0


class InitialEmbedding(nn.Module):
    """Architecture-compatible embedding for test32's state-dict checkpoint."""

    def __init__(self, num_species: int, cutoff: float):
        super().__init__()
        self.embed_node_x = nn.Embedding(num_species, 8)
        self.embed_node_z = nn.Embedding(num_species, 8)
        self.embed_edge = partial(bessel, start=0.0, end=cutoff, num_basis=16)

    def forward(self, data: Data) -> Data:
        data.h_node_x = self.embed_node_x(data.x)
        data.h_node_z = self.embed_node_z(data.x)
        data.h_edge = self.embed_edge(data.edge_attr.norm(dim=-1))
        return data


def build_model(cutoff: float, device: torch.device) -> NequIP:
    return NequIP(
        init_embed=InitialEmbedding(num_species=2, cutoff=cutoff),
        irreps_node_x="8x0e",
        irreps_node_z="8x0e",
        irreps_hidden="64x0e + 32x1e",
        irreps_edge="4x0e + 4x1e + 2x2e",
        irreps_out="1x1e",
        num_convs=3,
        radial_neurons=[16, 64],
        num_neighbors=12,
    ).to(device)


def load_model(checkpoint_path: Path, cutoff: float, device: torch.device) -> NequIP:
    payload = torch_load(checkpoint_path, map_location="cpu")
    # Accept both test32.py's plain state dict and a wrapped restart checkpoint.
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model = build_model(cutoff, device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def graph_on_device(data: Data, cutoff: float, numbers: np.ndarray) -> Data:
    """Build ASE's periodic graph on CPU and transfer only its edges to the GPU."""
    device = data.pos.device
    positions = data.pos.detach().cpu().numpy()
    cell = data.cell.detach().cpu().numpy()
    pbc = data.pbc.detach().cpu().numpy()
    i, j, displacement = primitive_neighbor_list(
        "ijD",
        cutoff=cutoff,
        pbc=pbc,
        cell=cell,
        positions=positions,
        numbers=numbers,
    )
    data.edge_index = torch.from_numpy(np.stack((i, j))).long().to(device)
    data.edge_attr = torch.from_numpy(displacement).float().to(device)
    return data


def save_restart(
    path: Path,
    phase: str,
    noisy_index: int,
    polish_index: int,
    current_positions: torch.Tensor,
    cg_frames: list[np.ndarray],
    args: argparse.Namespace,
) -> None:
    atomic_torch_save(
        {
            "format_version": 1,
            "phase": phase,
            "noisy_index": noisy_index,
            "polish_index": polish_index,
            "current_positions": current_positions.detach().cpu(),
            "cg_frames_angstrom": torch.from_numpy(np.stack(cg_frames)).float(),
            "noisy_steps": args.gen_noisy_steps,
            "polish_steps": args.gen_polish_steps,
            "max_sigma": args.gen_max_sigma,
            "cutoff": args.cutoff,
            "rng_state": capture_rng_state(),
        },
        path,
    )


@torch.no_grad()
def generate(
    args: argparse.Namespace,
    atoms,
    model: nn.Module,
    device: torch.device,
    restart_path: Path,
    deadline: Optional[float],
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    numbers = np.asarray(atoms.numbers)
    si_mask_numpy = numbers == 14
    si_mask_device = torch.tensor(si_mask_numpy, dtype=torch.bool, device=device)
    species = torch.tensor(LabelEncoder().fit_transform(numbers), dtype=torch.long, device=device)
    cell = torch.tensor(np.asarray(atoms.cell), dtype=torch.float32, device=device)
    pbc = torch.tensor(atoms.pbc, dtype=torch.bool, device=device)

    phase = "noisy"
    noisy_index = 0
    polish_index = 0
    current_positions = torch.tensor(atoms.positions, dtype=torch.float32, device=device)
    cg_frames = [np.asarray(atoms.positions[si_mask_numpy], dtype=np.float32)]

    if restart_path.exists():
        restart = torch_load(restart_path)
        expected = (
            args.gen_noisy_steps,
            args.gen_polish_steps,
            args.gen_max_sigma,
            args.cutoff,
        )
        found = (
            restart["noisy_steps"],
            restart["polish_steps"],
            restart["max_sigma"],
            restart["cutoff"],
        )
        if found != expected:
            raise RuntimeError(
                "Restart settings do not match this invocation: "
                f"restart={found}, requested={expected}. "
                "Use the same settings or select a new output directory."
            )
        phase = restart["phase"]
        noisy_index = int(restart["noisy_index"])
        polish_index = int(restart["polish_index"])
        current_positions = restart["current_positions"].to(device)
        cg_frames = [frame.numpy() for frame in restart["cg_frames_angstrom"]]
        restore_rng_state(restart.get("rng_state"))
        print(
            f"Resuming phase={phase}, noisy={noisy_index}/{args.gen_noisy_steps}, "
            f"polish={polish_index}/{args.gen_polish_steps}",
            flush=True,
        )

    sigmas = (
        torch.linspace(args.gen_max_sigma, 0.001, args.gen_noisy_steps, device=device)
        if args.gen_noisy_steps
        else torch.empty(0, device=device)
    )

    def make_data() -> Data:
        return Data(x=species, pos=current_positions, cell=cell, pbc=pbc)

    if phase == "noisy":
        for index in range(noisy_index, args.gen_noisy_steps):
            if deadline_reached(deadline):
                save_restart(
                    restart_path,
                    "noisy",
                    index,
                    polish_index,
                    current_positions,
                    cg_frames,
                    args,
                )
                print("Time budget reached; noisy-generation restart saved.", flush=True)
                return None

            data = graph_on_device(make_data(), args.cutoff, numbers)
            displacement = model(data) + sigmas[index] * torch.randn_like(current_positions)
            current_positions = current_positions - displacement
            cg_frames.append(current_positions[si_mask_device].detach().cpu().numpy())
            noisy_index = index + 1

            if noisy_index % args.generation_checkpoint_steps == 0:
                save_restart(
                    restart_path,
                    "noisy",
                    noisy_index,
                    polish_index,
                    current_positions,
                    cg_frames,
                    args,
                )
                print(f"noisy step {noisy_index}/{args.gen_noisy_steps}", flush=True)

        phase = "polish"
        save_restart(
            restart_path,
            phase,
            noisy_index,
            polish_index,
            current_positions,
            cg_frames,
            args,
        )

    for index in range(polish_index, args.gen_polish_steps):
        if deadline_reached(deadline):
            save_restart(
                restart_path,
                "polish",
                noisy_index,
                index,
                current_positions,
                cg_frames,
                args,
            )
            print("Time budget reached; polish-generation restart saved.", flush=True)
            return None

        data = graph_on_device(make_data(), args.cutoff, numbers)
        current_positions = current_positions - model(data)
        cg_frames.append(current_positions[si_mask_device].detach().cpu().numpy())
        polish_index = index + 1

        if polish_index % args.generation_checkpoint_steps == 0:
            save_restart(
                restart_path,
                "polish",
                noisy_index,
                polish_index,
                current_positions,
                cg_frames,
                args,
            )
            print(f"polish step {polish_index}/{args.gen_polish_steps}", flush=True)

    if restart_path.exists():
        restart_path.unlink()
    return current_positions.detach().cpu().numpy(), np.stack(cg_frames)


def wrap_positions(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    fractional = positions @ np.linalg.inv(cell)
    fractional -= np.floor(fractional)
    return fractional @ cell


def bond_and_angle_stats(positions, numbers, cell) -> tuple[np.ndarray, np.ndarray]:
    si_indices = np.where(numbers == 14)[0]
    o_indices = np.where(numbers == 8)[0]
    displacement = positions[si_indices, None, :] - positions[None, o_indices, :]
    fractional = displacement @ np.linalg.inv(cell)
    fractional -= np.round(fractional)
    displacement = fractional @ cell
    distances = np.linalg.norm(displacement, axis=-1)
    bonds, angles = [], []
    for si_local in range(len(si_indices)):
        nearest = np.argsort(distances[si_local])[:4]
        bonds.extend(distances[si_local, nearest].tolist())
        vectors = displacement[si_local, nearest]
        vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
        for first in range(4):
            for second in range(first + 1, 4):
                cosine = np.clip(np.dot(vectors[first], vectors[second]), -1.0, 1.0)
                angles.append(np.degrees(np.arccos(cosine)))
    return np.asarray(bonds), np.asarray(angles)


def save_histogram(reference, generated, target, xlabel, title, marker) -> None:
    figure, axis = plt.subplots()
    axis.hist(reference, bins=60, density=True, histtype="step", label="reference", linewidth=2)
    axis.hist(generated, bins=60, density=True, histtype="step", label="generated", linewidth=2)
    axis.axvline(marker, color="gray", linestyle=":")
    axis.set(xlabel=xlabel, title=title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(target, dpi=160)
    plt.close(figure)


def save_results(
    output_dir: Path,
    atoms,
    final_positions: np.ndarray,
    cg_frames_angstrom: np.ndarray,
    args: argparse.Namespace,
    checkpoint_sha256: str,
) -> dict:
    numbers = np.asarray(atoms.numbers)
    cell_angstrom = np.asarray(atoms.cell)
    final_wrapped = wrap_positions(final_positions, cell_angstrom)
    final_atoms = Atoms(
        numbers=numbers,
        positions=final_wrapped,
        cell=cell_angstrom,
        pbc=atoms.pbc,
    )
    ase.io.write(output_dir / "final_structure.extxyz", final_atoms)

    wrapped_cg = np.stack([wrap_positions(frame, cell_angstrom) for frame in cg_frames_angstrom])
    cg_positions_nm = (wrapped_cg / 10.0).astype(np.float32)
    cell_nm = (cell_angstrom / 10.0).astype(np.float32)
    np.save(output_dir / "positions.npy", cg_positions_nm)
    np.savez(
        output_dir / "manifest.npz",
        positions=cg_positions_nm,
        cell_nm=cell_nm,
        labels=np.full(cg_positions_nm.shape[0], 4, dtype=np.int32),
        source="DM2 test32 checkpoint generation; Si-only CG trajectory",
        checkpoint_sha256=checkpoint_sha256,
    )

    reference_positions = wrap_positions(np.asarray(atoms.positions), cell_angstrom)
    generated_bonds, generated_angles = bond_and_angle_stats(final_wrapped, numbers, cell_angstrom)
    reference_bonds, reference_angles = bond_and_angle_stats(reference_positions, numbers, cell_angstrom)
    save_histogram(
        reference_bonds,
        generated_bonds,
        output_dir / "bond_comparison.png",
        "Si-O distance (Angstrom)",
        "SiO2 Si-O bond length",
        1.61,
    )
    save_histogram(
        reference_angles,
        generated_angles,
        output_dir / "angle_comparison.png",
        "O-Si-O angle (degree)",
        "SiO2 O-Si-O angle",
        109.47,
    )

    metrics = {
        "num_frames": int(cg_positions_nm.shape[0]),
        "num_cg_particles": int(cg_positions_nm.shape[1]),
        "reference_bond_mean_angstrom": float(reference_bonds.mean()),
        "reference_bond_std_angstrom": float(reference_bonds.std()),
        "generated_bond_mean_angstrom": float(generated_bonds.mean()),
        "generated_bond_std_angstrom": float(generated_bonds.std()),
        "reference_angle_mean_degree": float(reference_angles.mean()),
        "reference_angle_std_degree": float(reference_angles.std()),
        "generated_angle_mean_degree": float(generated_angles.mean()),
        "generated_angle_std_degree": float(generated_angles.std()),
        "gen_noisy_steps": args.gen_noisy_steps,
        "gen_polish_steps": args.gen_polish_steps,
        "cutoff_angstrom": args.cutoff,
        "checkpoint_sha256": checkpoint_sha256,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def requested_configuration(args: argparse.Namespace, checkpoint_sha256: str) -> dict:
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "cutoff_angstrom": args.cutoff,
        "gen_noisy_steps": args.gen_noisy_steps,
        "gen_polish_steps": args.gen_polish_steps,
        "gen_max_sigma": args.gen_max_sigma,
        "seed": args.seed,
    }


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    deadline = started + args.time_budget_hours * 3600.0 if args.time_budget_hours > 0 else None
    checkpoint_path = args.checkpoint.expanduser().resolve()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Copy test32_sio2_glass_nequip.pt to that path or pass --checkpoint PATH."
        )
    if not input_path.is_file():
        raise FileNotFoundError(f"Input structure not found: {input_path}")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable. Submit to a GPU node or use --allow-cpu for a smoke test.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cpu_count = int(os.environ.get("PBS_NCPUS", os.cpu_count() or 1))
    torch.set_num_threads(max(1, cpu_count))

    checkpoint_sha256 = file_sha256(checkpoint_path)
    configuration = requested_configuration(args, checkpoint_sha256)
    completion_path = output_dir / "run_complete.json"
    if completion_path.exists() and not args.force:
        completed = json.loads(completion_path.read_text(encoding="utf-8"))
        if completed.get("configuration") == configuration:
            print(f"This configuration is already complete: {completion_path}")
            return 0
        print("Completed output uses different settings; starting a new generation.", flush=True)

    print(f"DM2 root: {DM2_ROOT}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint SHA-256: {checkpoint_sha256}")
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Generation: {args.gen_noisy_steps} noisy + {args.gen_polish_steps} polish steps; "
        f"cutoff={args.cutoff} Angstrom; budget={args.time_budget_hours}h",
        flush=True,
    )

    model = load_model(checkpoint_path, args.cutoff, device)
    atoms = ase.io.read(input_path, format="lammps-data")
    print(f"Loaded {len(atoms)}-atom reference structure.", flush=True)

    result = generate(
        args,
        atoms,
        model,
        device,
        output_dir / "generation_restart.pt",
        deadline,
    )
    if result is None:
        print("Submit run_test32-CGMD.pbs again with the same settings to continue.")
        return 0

    final_positions, cg_frames = result
    metrics = save_results(
        output_dir,
        atoms,
        final_positions,
        cg_frames,
        args,
        checkpoint_sha256,
    )
    elapsed_hours = (time.monotonic() - started) / 3600.0
    completion = {
        "status": "complete",
        "elapsed_hours_this_submission": elapsed_hours,
        "configuration": configuration,
        "metrics": metrics,
    }
    completion_path.write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Completed in {elapsed_hours:.3f} hours. Results: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
