#!/usr/bin/env python3
"""Build LAMMPS-callable score-force bundles for test32 SiO2.

Two workflows are supported:

``export-aa``
    Wrap the original two-species test32 NequIP denoiser checkpoint with the
    metadata and initial LAMMPS data file required by ``test32_lammps.py``.

``train-cg``
    Train a separate one-species Si-bead denoiser from
    ``test32-CGMD-output/positions.npy`` and export a CG score-force bundle.

Both bundles store a denoising displacement model.  At inference the score is
estimated with Tweedie's identity, ``score = -predicted_dx / sigma**2``, and
the LAMMPS driver applies ``force = k_B * temperature * score`` in metal units.

The test32 checkpoint is sigma-agnostic and the generated CG trajectory is not
an equilibrium trajectory.  These bundles are therefore experimental force
models, not validated thermodynamic potentials.  The limitation is recorded
inside every output bundle and its JSON sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Optional

import ase.io
import numpy as np
import torch
from ase.neighborlist import primitive_neighbor_list
from torch import nn
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parent
KB_EV_PER_K = 8.617333262145e-5
SI_O2_BEAD_MASS_AMU = 60.0843
FORMAT_NAME = "test32-score-forcefield"
FORMAT_VERSION = 1


def find_dm2_root() -> Path:
    configured = os.environ.get("DM2_ROOT")
    candidates = (
        [Path(configured).expanduser()]
        if configured
        else [ROOT / "DM2", ROOT.parent / "DM2", ROOT.parents[1]]
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "src" / "graphite").is_dir():
            return resolved
    searched = "\n  ".join(str(path.resolve()) for path in candidates)
    raise RuntimeError(
        "Could not find DM2/src/graphite. Set DM2_ROOT to the DM2 checkout.\n"
        f"Searched:\n  {searched}"
    )


DM2_ROOT = find_dm2_root()
DM2_SRC = str(DM2_ROOT / "src")
if DM2_SRC not in sys.path:
    sys.path.insert(0, DM2_SRC)

if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([slice])

from graphite.nn.basis import bessel
from graphite.nn.models.e3nn_nequip import NequIP


DEFAULT_AA_INPUT = (
    DM2_ROOT
    / "demo"
    / "demo_training"
    / "simu_data"
    / "sio2_3000_glass_1k_sample0.dat"
)
DEFAULT_AA_CHECKPOINTS = [
    ROOT / "checkpoints" / "test32_sio2_glass_nequip.pt",
    ROOT / "test32_sio2_glass_nequip.pt",
    ROOT.parent / "checkpoints" / "test32_sio2_glass_nequip.pt",
    DM2_ROOT / "demo" / "model" / "test32_sio2_glass_nequip.pt",
]
DEFAULT_AA_CHECKPOINT = next(
    (path for path in DEFAULT_AA_CHECKPOINTS if path.is_file()),
    DEFAULT_AA_CHECKPOINTS[0],
)
DEFAULT_CG_DIR = ROOT / "test32-CGMD-output"
DEFAULT_FORCEFIELD_DIR = ROOT / "forcefields"


class InitialEmbedding(nn.Module):
    """Embedding used by the original test32 model and the CG derivative."""

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


def architecture_metadata(num_species: int, cutoff: float) -> dict[str, Any]:
    return {
        "num_species": int(num_species),
        "cutoff_angstrom": float(cutoff),
        "irreps_node_x": "8x0e",
        "irreps_node_z": "8x0e",
        "irreps_hidden": "64x0e + 32x1e",
        "irreps_edge": "4x0e + 4x1e + 2x2e",
        "irreps_out": "1x1e",
        "num_convs": 3,
        "radial_neurons": [16, 64],
        "num_neighbors": 12,
    }


def build_model(metadata: dict[str, Any], device: torch.device) -> NequIP:
    return NequIP(
        init_embed=InitialEmbedding(
            num_species=int(metadata["num_species"]),
            cutoff=float(metadata["cutoff_angstrom"]),
        ),
        irreps_node_x=metadata["irreps_node_x"],
        irreps_node_z=metadata["irreps_node_z"],
        irreps_hidden=metadata["irreps_hidden"],
        irreps_edge=metadata["irreps_edge"],
        irreps_out=metadata["irreps_out"],
        num_convs=int(metadata["num_convs"]),
        radial_neurons=list(metadata["radial_neurons"]),
        num_neighbors=float(metadata["num_neighbors"]),
    ).to(device)


def choose_device(name: str, allow_cpu: bool) -> torch.device:
    if name == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if device.type != "cuda" and not allow_cpu:
        raise RuntimeError("A CUDA GPU is required; pass --allow-cpu only for a smoke test")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def torch_load(path: Path, map_location: str | torch.device = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def state_dict_from_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    payload = torch_load(path, map_location="cpu")
    if isinstance(payload, dict) and "model" in payload:
        payload = payload["model"]
    if not isinstance(payload, dict) or not payload:
        raise RuntimeError(f"Checkpoint does not contain a model state_dict: {path}")
    return payload


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def write_sidecar(bundle: dict[str, Any], path: Path) -> None:
    public = {key: value for key, value in bundle.items() if key != "model_state_dict"}
    path.write_text(
        json.dumps(json_safe(public), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def cell_matrix_from_manifest(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as manifest:
        if "cell_nm" not in manifest:
            raise RuntimeError(f"manifest has no cell_nm entry: {path}")
        raw = np.asarray(manifest["cell_nm"], dtype=np.float64)
    if raw.shape == (3,):
        cell_nm = np.diag(raw)
    elif raw.shape == (3, 3):
        cell_nm = raw
    else:
        raise RuntimeError(f"cell_nm must have shape (3,) or (3,3), got {raw.shape}")
    if np.linalg.det(cell_nm) <= 0:
        raise RuntimeError("cell_nm must describe a right-handed, non-zero cell")
    return cell_nm * 10.0


def wrap_positions(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    fractional = np.asarray(positions) @ np.linalg.inv(cell)
    fractional -= np.floor(fractional)
    return fractional @ cell


def wrap_positions_torch(positions: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    fractional = positions @ torch.linalg.inv(cell)
    fractional = fractional - torch.floor(fractional)
    return fractional @ cell


def graph_data(
    positions: torch.Tensor,
    cell: torch.Tensor,
    species: torch.Tensor,
    atomic_numbers: np.ndarray,
    cutoff: float,
) -> Data:
    positions_cpu = positions.detach().cpu().numpy()
    cell_cpu = cell.detach().cpu().numpy()
    first, second, displacement = primitive_neighbor_list(
        "ijD",
        cutoff=cutoff,
        pbc=np.ones(3, dtype=bool),
        cell=cell_cpu,
        positions=positions_cpu,
        numbers=atomic_numbers,
    )
    if len(first) == 0:
        raise RuntimeError(f"Periodic graph has no edges at cutoff={cutoff} Angstrom")
    device = positions.device
    return Data(
        x=species,
        pos=positions,
        cell=cell,
        pbc=torch.ones(3, dtype=torch.bool, device=device),
        edge_index=torch.from_numpy(np.stack((first, second))).long().to(device),
        edge_attr=torch.from_numpy(displacement).float().to(device),
    )


def require_orthorhombic(cell: np.ndarray) -> np.ndarray:
    diagonal = np.diag(np.diag(cell))
    if not np.allclose(cell, diagonal, atol=1.0e-7):
        raise RuntimeError("LAMMPS data export currently supports orthorhombic cells only")
    lengths = np.diag(cell)
    if np.any(lengths <= 0):
        raise RuntimeError(f"Invalid cell lengths: {lengths}")
    return lengths


def write_lammps_data(
    path: Path,
    positions: np.ndarray,
    cell: np.ndarray,
    atom_types: np.ndarray,
    masses: list[float],
    title: str,
) -> None:
    lengths = require_orthorhombic(cell)
    wrapped = wrap_positions(positions, cell)
    types = np.asarray(atom_types, dtype=np.int64)
    if types.shape != (len(wrapped),):
        raise RuntimeError("atom_types does not match the position count")
    if types.min() < 1 or types.max() > len(masses):
        raise RuntimeError("atom type IDs are outside the provided mass table")
    lines = [
        title,
        "",
        f"{len(wrapped)} atoms",
        f"{len(masses)} atom types",
        "",
        f"0.0 {lengths[0]:.16g} xlo xhi",
        f"0.0 {lengths[1]:.16g} ylo yhi",
        f"0.0 {lengths[2]:.16g} zlo zhi",
        "",
        "Masses",
        "",
    ]
    lines.extend(f"{index} {mass:.10g}" for index, mass in enumerate(masses, 1))
    lines.extend(["", "Atoms # atomic", ""])
    for atom_id, (atom_type, xyz) in enumerate(zip(types, wrapped), 1):
        lines.append(
            f"{atom_id} {int(atom_type)} "
            f"{xyz[0]:.16g} {xyz[1]:.16g} {xyz[2]:.16g}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def scientific_caveat(kind: str) -> str:
    if kind == "all_atom":
        return (
            "The original test32 model was trained as a sigma-agnostic displacement "
            "denoiser over several noise widths, not as an energy model or a "
            "temperature-conditioned equilibrium score. sigma_ref is therefore a "
            "user-selected approximation; energy and virial are unavailable."
        )
    return (
        "The CG model is trained from a correlated annealed-denoising/generation "
        "trajectory, not an equilibrium canonical MD ensemble. kBT*score is only a "
        "physical potential of mean force for an equilibrium distribution at the "
        "same temperature. Treat this output as experimental; energy and virial are "
        "unavailable."
    )


def base_bundle(
    *,
    kind: str,
    architecture: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    sigma_angstrom: float,
    temperature_k: float,
    force_clip_ev_per_angstrom: float,
    species_by_id: np.ndarray,
    atomic_numbers_by_id: np.ndarray,
    cell_angstrom: np.ndarray,
) -> dict[str, Any]:
    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "kind": kind,
        "model_class": "graphite.nn.models.e3nn_nequip.NequIP",
        "model_state_dict": state_dict,
        "architecture": architecture,
        "prediction": "denoising_displacement_angstrom",
        "score_estimator": "score_A^-1 = -predicted_dx_A / sigma_ref_A^2",
        "force_formula": "force_eV_per_A = k_B_eV_per_K * temperature_K * score_A^-1",
        "k_b_ev_per_k": KB_EV_PER_K,
        "sigma_ref_angstrom": float(sigma_angstrom),
        "temperature_k": float(temperature_k),
        "force_clip_ev_per_angstrom": float(force_clip_ev_per_angstrom),
        "species_by_id": torch.as_tensor(species_by_id, dtype=torch.long),
        "atomic_numbers_by_id": torch.as_tensor(atomic_numbers_by_id, dtype=torch.long),
        "num_atoms": int(len(species_by_id)),
        "cell_angstrom": torch.as_tensor(cell_angstrom, dtype=torch.float64),
        "units": {"length": "angstrom", "energy": "eV", "time": "ps", "mass": "amu"},
        "conservative": False,
        "provides_energy": False,
        "provides_virial": False,
        "scientific_status": "experimental_unvalidated",
        "scientific_caveat": scientific_caveat(kind),
        "created_unix_time": time.time(),
    }


def validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def export_all_atom(args: argparse.Namespace) -> int:
    checkpoint = args.checkpoint.expanduser().resolve()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"test32 checkpoint not found: {checkpoint}")
    if not input_path.is_file():
        raise FileNotFoundError(f"all-atom structure not found: {input_path}")
    validate_positive("sigma_ref_angstrom", args.sigma_ref_angstrom)
    validate_positive("temperature_k", args.temperature_k)
    validate_positive("force_clip", args.force_clip)

    output_dir.mkdir(parents=True, exist_ok=True)
    architecture = architecture_metadata(num_species=2, cutoff=args.cutoff)
    state_dict = state_dict_from_checkpoint(checkpoint)
    # Loading verifies that the supplied state_dict really matches test32.
    model = build_model(architecture, torch.device("cpu"))
    model.load_state_dict(state_dict)

    atoms = ase.io.read(str(input_path), format="lammps-data")
    numbers = np.asarray(atoms.numbers, dtype=np.int64)
    unique, counts = np.unique(numbers, return_counts=True)
    if not np.array_equal(unique, np.asarray([8, 14])):
        raise RuntimeError(f"Expected O/Si atomic numbers [8,14], found {unique.tolist()}")
    species_by_id = np.where(numbers == 8, 0, 1).astype(np.int64)
    atom_types = np.where(numbers == 8, 1, 2).astype(np.int64)
    cell = np.asarray(atoms.cell, dtype=np.float64)
    start_data = output_dir / "aa_start.data"
    write_lammps_data(
        start_data,
        np.asarray(atoms.positions, dtype=np.float64),
        cell,
        atom_types,
        masses=[15.9994, 28.0855],
        title="test32 all-atom SiO2 start structure (types: 1=O, 2=Si)",
    )
    bundle = base_bundle(
        kind="all_atom",
        architecture=architecture,
        state_dict={name: value.detach().cpu() for name, value in state_dict.items()},
        sigma_angstrom=args.sigma_ref_angstrom,
        temperature_k=args.temperature_k,
        force_clip_ev_per_angstrom=args.force_clip,
        species_by_id=species_by_id,
        atomic_numbers_by_id=numbers,
        cell_angstrom=cell,
    )
    bundle.update(
        {
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_sha256": file_sha256(checkpoint),
            "source_structure": str(input_path),
            "species_mapping": {"LAMMPS_type_1_O": 0, "LAMMPS_type_2_Si": 1},
            "composition": {str(int(z)): int(n) for z, n in zip(unique, counts)},
            "suggested_data_file": start_data.name,
            "suggested_timestep_ps": 0.0001,
            "suggested_damping_ps": 0.1,
        }
    )
    bundle_path = output_dir / "aa_score_forcefield.pt"
    atomic_torch_save(bundle, bundle_path)
    write_sidecar(bundle, output_dir / "aa_score_forcefield.json")
    print(f"All-atom bundle: {bundle_path}")
    print(f"LAMMPS start data: {start_data}")
    print(f"Composition: O={counts[0]}, Si={counts[1]}")
    print(f"WARNING: {bundle['scientific_caveat']}")
    return 0


def training_restart_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    losses: list[dict[str, float]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "format_version": 1,
        "step": int(step),
        "model_state_dict": cpu_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "losses": losses,
        "settings": settings,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    return state


def restore_training_rng(payload: dict[str, Any]) -> None:
    torch.set_rng_state(payload["torch_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state" in payload:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])


def validation_loss(
    model: nn.Module,
    positions_angstrom: np.ndarray,
    indices: np.ndarray,
    cell: torch.Tensor,
    species: torch.Tensor,
    numbers: np.ndarray,
    cutoff: float,
    sigma: float,
    device: torch.device,
    seed: int,
    max_frames: int = 4,
) -> float:
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    values = []
    with torch.no_grad():
        for index in indices[:max_frames]:
            clean = torch.as_tensor(positions_angstrom[index], dtype=torch.float32, device=device)
            noise = torch.randn(clean.shape, generator=generator, device=device) * sigma
            noise -= noise.mean(dim=0, keepdim=True)
            noisy = wrap_positions_torch(clean + noise, cell)
            data = graph_data(noisy, cell, species, numbers, cutoff)
            values.append(torch.nn.functional.mse_loss(model(data), noise).item())
    model.train()
    return float(np.mean(values)) if values else float("nan")


def train_cg(args: argparse.Namespace) -> int:
    positions_path = args.positions.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not positions_path.is_file():
        raise FileNotFoundError(f"CG positions not found: {positions_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"CG manifest not found: {manifest_path}")
    validate_positive("sigma_angstrom", args.sigma_angstrom)
    validate_positive("temperature_k", args.temperature_k)
    validate_positive("cutoff", args.cutoff)
    validate_positive("learning_rate", args.learning_rate)
    validate_positive("force_clip", args.force_clip)
    if not 0.0 <= args.frame_start_fraction < 1.0:
        raise ValueError("--frame-start-fraction must be in [0,1)")
    if not 0.0 <= args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be in [0,1)")
    if args.frame_stride < 1 or args.updates < 1 or args.checkpoint_every < 1:
        raise ValueError("frame stride, updates, and checkpoint interval must be positive")

    device = choose_device(args.device, args.allow_cpu)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    positions_nm = np.load(positions_path, mmap_mode="r")
    if positions_nm.ndim != 3 or positions_nm.shape[-1] != 3:
        raise RuntimeError(f"positions.npy must have shape (frames, beads, 3), got {positions_nm.shape}")
    if positions_nm.shape[0] < 2 or positions_nm.shape[1] < 2:
        raise RuntimeError("CG training needs at least two frames and two beads")
    start = int(np.floor(positions_nm.shape[0] * args.frame_start_fraction))
    selected = np.arange(start, positions_nm.shape[0], args.frame_stride, dtype=np.int64)
    if len(selected) < 2:
        raise RuntimeError(
            f"Only {len(selected)} frame selected; lower --frame-start-fraction or --frame-stride"
        )
    positions_angstrom = np.asarray(positions_nm, dtype=np.float32) * 10.0
    cell_np = cell_matrix_from_manifest(manifest_path)
    require_orthorhombic(cell_np)

    split_rng = np.random.default_rng(args.seed)
    shuffled = selected.copy()
    split_rng.shuffle(shuffled)
    if args.validation_fraction > 0 and len(shuffled) > 2:
        n_valid = max(1, int(round(len(shuffled) * args.validation_fraction)))
    else:
        n_valid = 0
    valid_indices = shuffled[:n_valid]
    train_indices = shuffled[n_valid:]
    if len(train_indices) == 0:
        raise RuntimeError("No training frames remain after validation split")

    n_beads = positions_angstrom.shape[1]
    architecture = architecture_metadata(num_species=1, cutoff=args.cutoff)
    model = build_model(architecture, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    cell = torch.as_tensor(cell_np, dtype=torch.float32, device=device)
    species = torch.zeros(n_beads, dtype=torch.long, device=device)
    numbers = np.full(n_beads, 14, dtype=np.int64)
    settings = {
        "positions": str(positions_path),
        "manifest": str(manifest_path),
        "positions_shape": list(positions_nm.shape),
        "selected_first_frame": int(selected[0]),
        "selected_last_frame": int(selected[-1]),
        "selected_frame_count": int(len(selected)),
        "frame_start_fraction": args.frame_start_fraction,
        "frame_stride": args.frame_stride,
        "validation_fraction": args.validation_fraction,
        "sigma_angstrom": args.sigma_angstrom,
        "cutoff_angstrom": args.cutoff,
        "learning_rate": args.learning_rate,
        "updates": args.updates,
        "seed": args.seed,
        "num_beads": n_beads,
        "cell_angstrom": cell_np.tolist(),
    }
    restart_path = output_dir / "cg_training_restart.pt"
    losses: list[dict[str, float]] = []
    first_step = 1
    if restart_path.is_file():
        restart = torch_load(restart_path, map_location="cpu")
        if restart.get("settings") != settings:
            raise RuntimeError(
                f"Training restart settings differ from this invocation: {restart_path}. "
                "Use matching arguments or choose a new output directory."
            )
        model.load_state_dict(restart["model_state_dict"])
        optimizer.load_state_dict(restart["optimizer_state_dict"])
        # Move optimizer tensors restored on CPU to the selected device.
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        first_step = int(restart["step"]) + 1
        losses = list(restart.get("losses", []))
        restore_training_rng(restart)
        print(f"Resuming CG training at update {first_step}/{args.updates}")

    deadline = (
        time.monotonic() + args.time_budget_hours * 3600.0
        if args.time_budget_hours > 0
        else None
    )
    model.train()
    recent: list[float] = []
    for step in range(first_step, args.updates + 1):
        if deadline is not None and time.monotonic() >= deadline - 120.0:
            atomic_torch_save(
                training_restart_payload(model, optimizer, step - 1, losses, settings),
                restart_path,
            )
            print(f"Time budget reached; saved restart at update {step - 1}: {restart_path}")
            return 0

        frame_index = int(train_indices[np.random.randint(len(train_indices))])
        clean = torch.as_tensor(positions_angstrom[frame_index], dtype=torch.float32, device=device)
        noise = torch.randn_like(clean) * args.sigma_angstrom
        noise -= noise.mean(dim=0, keepdim=True)
        noisy = wrap_positions_torch(clean + noise, cell)
        data = graph_data(noisy, cell, species, numbers, args.cutoff)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(data)
        loss = torch.nn.functional.mse_loss(prediction, noise)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at update {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        recent.append(float(loss.detach().cpu()))

        should_log = step == first_step or step % args.log_every == 0 or step == args.updates
        if should_log:
            train_loss = float(np.mean(recent))
            valid_loss = validation_loss(
                model,
                positions_angstrom,
                valid_indices,
                cell,
                species,
                numbers,
                args.cutoff,
                args.sigma_angstrom,
                device,
                seed=args.seed + step,
            )
            record = {"step": float(step), "train_mse_A2": train_loss, "valid_mse_A2": valid_loss}
            losses.append(record)
            recent.clear()
            print(
                f"update {step}/{args.updates} train_mse={train_loss:.6g} "
                f"valid_mse={valid_loss:.6g}",
                flush=True,
            )
        if step % args.checkpoint_every == 0:
            atomic_torch_save(
                training_restart_payload(model, optimizer, step, losses, settings),
                restart_path,
            )

    start_positions = positions_angstrom[selected[-1]]
    start_data = output_dir / "cg_start.data"
    write_lammps_data(
        start_data,
        start_positions,
        cell_np,
        np.ones(n_beads, dtype=np.int64),
        masses=[SI_O2_BEAD_MASS_AMU],
        title="test32 CG SiO2 start structure (one SiO2 formula unit per bead)",
    )
    bundle = base_bundle(
        kind="coarse_grained",
        architecture=architecture,
        state_dict=cpu_state_dict(model),
        sigma_angstrom=args.sigma_angstrom,
        temperature_k=args.temperature_k,
        force_clip_ev_per_angstrom=args.force_clip,
        species_by_id=np.zeros(n_beads, dtype=np.int64),
        atomic_numbers_by_id=numbers,
        cell_angstrom=cell_np,
    )
    bundle.update(
        {
            "source_positions": str(positions_path),
            "source_manifest": str(manifest_path),
            "training": settings,
            "loss_history": losses,
            "bead_definition": "one Si position representing one SiO2 formula unit",
            "bead_mass_amu": SI_O2_BEAD_MASS_AMU,
            "suggested_data_file": start_data.name,
            "suggested_timestep_ps": 0.001,
            "suggested_damping_ps": 1.0,
        }
    )
    bundle_path = output_dir / "cg_score_forcefield.pt"
    atomic_torch_save(bundle, bundle_path)
    write_sidecar(bundle, output_dir / "cg_score_forcefield.json")
    if restart_path.exists():
        restart_path.unlink()
    print(f"CG bundle: {bundle_path}")
    print(f"LAMMPS start data: {start_data}")
    print(f"WARNING: {bundle['scientific_caveat']}")
    return 0


def inspect_bundle(args: argparse.Namespace) -> int:
    bundle_path = args.bundle.expanduser().resolve()
    payload = torch_load(bundle_path, map_location="cpu")
    if payload.get("format") != FORMAT_NAME:
        raise RuntimeError(f"Not a {FORMAT_NAME} bundle: {bundle_path}")
    public = {key: value for key, value in payload.items() if key != "model_state_dict"}
    print(json.dumps(json_safe(public), indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    aa = subparsers.add_parser("export-aa", help="Package the original test32 denoiser for LAMMPS")
    aa.add_argument("--checkpoint", type=Path, default=DEFAULT_AA_CHECKPOINT)
    aa.add_argument("--input", type=Path, default=DEFAULT_AA_INPUT)
    aa.add_argument("--output-dir", type=Path, default=DEFAULT_FORCEFIELD_DIR)
    aa.add_argument("--cutoff", type=float, default=5.0)
    aa.add_argument("--sigma-ref-angstrom", type=float, default=0.1)
    aa.add_argument("--temperature-k", type=float, default=300.0)
    aa.add_argument("--force-clip", type=float, default=10.0, help="Per-atom force cap in eV/Angstrom")
    aa.set_defaults(handler=export_all_atom)

    cg = subparsers.add_parser("train-cg", help="Train and export a Si-bead score force model")
    cg.add_argument("--positions", type=Path, default=DEFAULT_CG_DIR / "positions.npy")
    cg.add_argument("--manifest", type=Path, default=DEFAULT_CG_DIR / "manifest.npz")
    cg.add_argument("--output-dir", type=Path, default=DEFAULT_FORCEFIELD_DIR)
    cg.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    cg.add_argument("--allow-cpu", action="store_true")
    cg.add_argument("--updates", type=int, default=30_000)
    cg.add_argument("--learning-rate", type=float, default=2.0e-4)
    cg.add_argument("--sigma-angstrom", type=float, default=0.1)
    cg.add_argument("--cutoff", type=float, default=6.0)
    cg.add_argument("--temperature-k", type=float, default=300.0)
    cg.add_argument("--force-clip", type=float, default=10.0, help="Per-bead force cap in eV/Angstrom")
    cg.add_argument("--frame-start-fraction", type=float, default=0.8)
    cg.add_argument("--frame-stride", type=int, default=5)
    cg.add_argument("--validation-fraction", type=float, default=0.1)
    cg.add_argument("--checkpoint-every", type=int, default=500)
    cg.add_argument("--log-every", type=int, default=100)
    cg.add_argument("--time-budget-hours", type=float, default=11.5)
    cg.add_argument("--seed", type=int, default=1337)
    cg.set_defaults(handler=train_cg)

    show = subparsers.add_parser("inspect", help="Print force-field metadata without model weights")
    show.add_argument("--bundle", type=Path, required=True)
    show.set_defaults(handler=inspect_bundle)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
