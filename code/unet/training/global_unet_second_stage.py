"""Configurable training and validation runner for the global residual U-Net.
Loads daily Zarr samples, trains with checkpoint/resume support, evaluates
validation subsets and writes metrics, diagnostics and model checkpoints.
The paper's selected run settings are recorded in configs/unet/unet_run_config.json;
command-line defaults also support earlier architecture-screening experiments.
Requires the external training/validation stores and supporting input metadata.
Existing input-identity and code-snapshot checks remain part of the workflow.
Standalone self-tests and training-history plot rendering are omitted.
Training, validation, checkpointing, numerical diagnostics and saved arrays remain."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import signal
import socket
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable


import numpy as np
import torch
import xarray as xr
import zarr
from torch.utils.data import DataLoader, Subset

from dataloader.zarr_dataset import AWIDownscalingZarrDataset
from training.global_unet import GlobalUNet, POLE_AWARE_BOUNDARY_SCHEME


DATA_ROOT = os.environ.get("DATA_ROOT", "data_root")                                                     
REPO_ROOT = os.environ.get("REPO_ROOT", "repo_root")                   
RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "results_root")                                                  


REPOSITORY = Path(f"{REPO_ROOT}")
TRAIN_ZARR = Path(
    f"{DATA_ROOT}/zarr/awi_downscaling_train.zarr"
)
VALID_ZARR = Path(
    f"{DATA_ROOT}/zarr/awi_downscaling_val.zarr"
)
SIDECAR = Path(
    f"{DATA_ROOT}/grids/static_masks/"
    "surface_fractions_hr.nc"
)
NORM_STATS = Path(
    f"{DATA_ROOT}/grids/norm_stats.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    f"{DATA_ROOT}/runs/global_unet_pilots"
)

EXPECTED_HEIGHT = 1280
EXPECTED_WIDTH = 2624
EXPECTED_TRAIN_LENGTH = 10_593
EXPECTED_VALID_LENGTH = 1_095
EXPECTED_SIDECAR_SHA256 = (
    "cfbd3ea57b526eb2acbb0465c831deee"
    "ad568cc9735f1d1300eeb5015e6229a6"
)
EXPECTED_STATIC_ARRAY_SHA256 = (
    "43452d5dbbafaf5c5794f5e3f00ba8f"
    "bebfb2d938a40d7270730c9d5939540dd"
)

EVALUATION_STEPS_DEFAULT = (0, 25, 50, 75, 100)
DIAGNOSTIC_STEPS_DEFAULT = (0, 1, 2, 5, 10, 25, 50, 75, 100)
MAP_STEPS_DEFAULT = (0, 50, 100)

STOP_REQUESTED = False


METRIC_CSV_FIELDS = [
    "kind",
    "step",
    "split",
    "region",
    "date",
    "index",
    "samples_seen",
    "mse_norm",
    "mse_K2",
    "rmse_K",
    "mae_norm",
    "mae_K",
    "bias_norm",
    "bias_K",
    "baseline_mse_norm",
    "baseline_mse_K2",
    "baseline_rmse_K",
    "skill",
    "corr_corr",
    "correction_mean",
    "correction_std",
    "correction_rms",
    "correction_absmax",
    "required_mean",
    "required_std",
    "required_rms",
    "required_absmax",
    "gradient_norm",
    "post_clip_gradient_norm",
    "clip_applied",
    "head_gradient_norm",
    "encoder0_gradient_norm",
    "bottleneck_gradient_norm",
    "decoder_last_gradient_norm",
    "encoder0_activation_std",
    "bottleneck_activation_std",
    "decoder_last_activation_std",
    "learning_rate",
    "weight_decay",
    "update_seconds",
    "compute_seconds",
    "samples_per_second",
    "memory_allocated_GiB",
    "memory_reserved_GiB",
    "finite",
    "seam_abs_mean",
    "seam_abs_median",
    "interior_abs_mean",
    "interior_sample_median",
    "seam_to_interior_mean_ratio",
    "seam_median_percentile",
    "seam_band_mae_K",
    "global_mae_K",
]

TRAIN_METRIC_CSV_FIELDS = [
    "step",
    "epoch",
    "sample_position",
    "mse_norm",
    "mse_K2",
    "rmse_K",
    "mae_norm",
    "mae_K",
    "learning_rate",
    "gradient_norm",
    "post_clip_gradient_norm",
    "clip_applied",
    "head_gradient_norm",
    "correction_mean",
    "correction_std",
    "correction_rms",
    "correction_absmax",
]

VALIDATION_METRIC_CSV_FIELDS = [
    "step",
    "mse_norm",
    "mse_K2",
    "rmse_K",
    "mae_norm",
    "mae_K",
    "baseline_mse_norm",
    "baseline_mse_K2",
    "baseline_rmse_K",
    "skill",
    "corr_corr",
    "correction_mean",
    "correction_std",
    "correction_rms",
    "correction_absmax",
]


def build_optimizer(
    model: torch.nn.Module,
    *,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
) -> tuple[
    torch.optim.Optimizer,
    list[dict[str, Any]],
]:


    decay_parameters: list[torch.nn.Parameter] = []
    no_decay_parameters: list[torch.nn.Parameter] = []
    decay_names: list[str] = []
    no_decay_names: list[str] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if parameter.ndim >= 2:
            decay_parameters.append(parameter)
            decay_names.append(name)
        else:
            no_decay_parameters.append(parameter)
            no_decay_names.append(name)

    all_trainable_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }

    grouped_ids = [
        id(parameter)
        for parameter in (
            decay_parameters
            + no_decay_parameters
        )
    ]

    if len(grouped_ids) != len(set(grouped_ids)):
        raise RuntimeError(
            "An optimizer parameter appears in more than one group"
        )

    if set(grouped_ids) != all_trainable_ids:
        raise RuntimeError(
            "Optimizer groups do not cover every trainable parameter"
        )

    parameter_groups = [
        {
            "params": decay_parameters,
            "weight_decay": weight_decay,
            "group_name": "decay_ndim_ge_2",
        },
        {
            "params": no_decay_parameters,
            "weight_decay": 0.0,
            "group_name": "no_decay_scalar_vector",
        },
    ]

    optimizer_class: type[torch.optim.Optimizer]

    if optimizer_name == "adamw":
        optimizer_class = torch.optim.AdamW
    elif optimizer_name == "adam":
        optimizer_class = torch.optim.Adam
    else:
        raise ValueError(
            f"Unsupported optimizer: {optimizer_name}"
        )

    optimizer = optimizer_class(
        parameter_groups,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        amsgrad=False,
    )

    summaries = [
        {
            "name": "decay_ndim_ge_2",
            "weight_decay": weight_decay,
            "parameter_tensors": len(
                decay_parameters
            ),
            "parameter_elements": sum(
                parameter.numel()
                for parameter in decay_parameters
            ),
            "parameter_names": decay_names,
        },
        {
            "name": "no_decay_scalar_vector",
            "weight_decay": 0.0,
            "parameter_tensors": len(
                no_decay_parameters
            ),
            "parameter_elements": sum(
                parameter.numel()
                for parameter in no_decay_parameters
            ),
            "parameter_names": no_decay_names,
        },
    ]

    return optimizer, summaries


def log(message: str = "") -> None:
    print(message, flush=True)


def request_stop(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    log(
        f"\nReceived signal {signum}; a latest checkpoint will be written "
        "at the next effective-update boundary."
    )


def parse_integer_steps(text: str, *, name: str) -> tuple[int, ...]:
    if isinstance(text, str) and text.strip().lower() in {"", "none", "null", "off", "false"}:
        return tuple()
    values: list[int] = []
    for token in text.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        value = int(stripped)
        if value < 0:
            raise ValueError(f"{name} cannot contain negative values")
        values.append(value)
    if not values:
        raise ValueError(f"{name} cannot be empty")
    return tuple(sorted(set(values)))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a deterministic, resumable architecture-screen training chunk "
            "for the five-channel training/global_unet.py model."
        )
    )
    parser.add_argument("--norm", choices=("none", "group"), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument(
        "--lat-padding",
        choices=("replicate", "pole_aware"),
        default="replicate",
        help=(
            "Convolution latitude-padding mode. 'replicate' is the validated "
            "production default; 'pole_aware' builds 180-degree "
            "longitude-shifted ghost rows across the poles. When the value "
            "is 'replicate' the argument is excluded from the config hash "
            "so legacy runs keep their recorded hash."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "fixed_step", "monitor_plateau", "cosine"),
        default="fixed_step",
        help=(
            "Use a constant LR, fixed-step LR, fixed-validation monitor "
            "plateau schedule, or deterministic post-warm-up cosine decay."
        ),
    )
    parser.add_argument("--warmup-updates", type=int, default=1292)
    parser.add_argument("--lr-drop1-update", type=int, default=14212)
    parser.add_argument("--lr-drop2-update", type=int, default=18088)
    parser.add_argument("--lr-drop1-factor", type=float, default=0.5)
    parser.add_argument("--lr-drop2-factor", type=float, default=0.25)
    parser.add_argument(
        "--plateau-improvement-factor",
        type=float,
        default=0.995,
        help="Validation improvement threshold: improved if current < best * factor.",
    )
    parser.add_argument("--plateau-patience", type=int, default=3)
    parser.add_argument("--plateau-cooldown", type=int, default=3)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-max-drops", type=int, default=5)
    parser.add_argument("--plateau-min-lr", type=float, default=3.125e-6)
    parser.add_argument(
        "--cosine-min-lr",
        type=float,
        default=3.125e-6,
        help=(
            "Cosine schedule only: the LR reached exactly at --max-updates "
            "after decaying from --learning-rate. Inert (and excluded from "
            "the config hash) for every other --lr-schedule."
        ),
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=None,
        help=(
            "Global-norm gradient clipping applied to the accumulated gradient "
            "before optimizer.step(). Absent or 0 disables clipping."
        ),
    )
    parser.add_argument(
        "--stop-updates",
        type=int,
        default=None,
        help=(
            "Stop this chunk at this global optimizer update. "
            "The full planned run remains --max-updates."
        ),
    )
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default="adamw")
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument("--max-updates", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--train-pool-samples", type=int, default=400)
    parser.add_argument("--train-monitor-samples", type=int, default=32)
    parser.add_argument("--validation-samples", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--evaluation-steps",
        default=",".join(str(value) for value in EVALUATION_STEPS_DEFAULT),
    )
    parser.add_argument(
        "--diagnostic-steps",
        default=",".join(str(value) for value in DIAGNOSTIC_STEPS_DEFAULT),
    )
    parser.add_argument(
        "--map-steps",
        default=",".join(str(value) for value in MAP_STEPS_DEFAULT),
    )
    parser.add_argument("--top-errors", type=int, default=5)
    parser.add_argument(
        "--checkpoint-retention",
        choices=("legacy", "lean"),
        default="legacy",
        help=(
            "legacy (default): keep the validated behavior including "
            "epoch_NNN.pt archives. lean: best_val_mse.pt + latest.pt + one "
            "rolling last_completed_chunk.pt recovery checkpoint, no epoch "
            "archive; adds best_checkpoint_metadata.json. lean is hashed "
            "into the scientific config so retention cannot silently change "
            "across resumed chunks; legacy is popped for hash compatibility."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    positive = (
        "base",
        "depth",
        "learning_rate",
        "max_updates",
        "warmup_updates",
        "lr_drop1_update",
        "lr_drop2_update",
        "epochs",
        "accumulation_steps",
        "train_pool_samples",
        "train_monitor_samples",
        "validation_samples",
        "top_errors",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay cannot be negative")
    if args.lr_drop1_factor <= 0 or args.lr_drop2_factor <= 0:
        raise ValueError("LR drop factors must be positive")
    if not (0.0 < args.plateau_improvement_factor < 1.0):
        raise ValueError("--plateau-improvement-factor must lie in (0, 1)")
    if args.plateau_patience <= 0:
        raise ValueError("--plateau-patience must be positive")
    if args.plateau_cooldown < 0:
        raise ValueError("--plateau-cooldown cannot be negative")
    if not (0.0 < args.plateau_factor < 1.0):
        raise ValueError("--plateau-factor must lie in (0, 1)")
    if args.plateau_max_drops < 0:
        raise ValueError("--plateau-max-drops cannot be negative")
    if args.plateau_min_lr <= 0.0:
        raise ValueError("--plateau-min-lr must be positive")
    if args.cosine_min_lr <= 0.0:
        raise ValueError("--cosine-min-lr must be positive")
    if args.lr_schedule == "cosine":
        if args.warmup_updates >= args.max_updates:
            raise ValueError(
                "The cosine schedule requires warmup_updates < max_updates"
            )
        if args.cosine_min_lr > args.learning_rate:
            raise ValueError("--cosine-min-lr cannot exceed --learning-rate")
    if args.gradient_clip_norm is not None:
        if args.gradient_clip_norm < 0.0:
            raise ValueError("--gradient-clip-norm cannot be negative")
        if args.gradient_clip_norm == 0.0:
            args.gradient_clip_norm = None
    if args.stop_updates is None:
        args.stop_updates = args.max_updates
    if args.stop_updates <= 0:
        raise ValueError("--stop-updates must be positive")
    if args.stop_updates > args.max_updates:
        raise ValueError("--stop-updates cannot exceed --max-updates")
    if args.lr_schedule == "fixed_step" and not (0 < args.warmup_updates <= args.lr_drop1_update <= args.lr_drop2_update <= args.max_updates):
        raise ValueError(
            "LR schedule updates must satisfy "
            "0 < warmup <= drop1 <= drop2 <= max_updates"
        )
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")

    args.evaluation_steps = parse_integer_steps(
        args.evaluation_steps,
        name="--evaluation-steps",
    )
    args.diagnostic_steps = parse_integer_steps(
        args.diagnostic_steps,
        name="--diagnostic-steps",
    )
    args.map_steps = parse_integer_steps(
        args.map_steps,
        name="--map-steps",
    )

    for schedule_name in (
        "evaluation_steps",
        "diagnostic_steps",
        "map_steps",
    ):
        schedule = getattr(args, schedule_name)
        if any(value > args.max_updates for value in schedule):
            raise ValueError(
                f"--{schedule_name.replace('_', '-')} contains a step above "
                f"--max-updates={args.max_updates}"
            )

    if 0 not in args.evaluation_steps:
        raise ValueError("--evaluation-steps must include 0")
    if args.max_updates not in args.evaluation_steps:
        raise ValueError("--evaluation-steps must include --max-updates")
    if not set(args.map_steps).issubset(set(args.evaluation_steps)):
        raise ValueError("--map-steps must be a subset of --evaluation-steps")

    used_training_samples = args.max_updates * args.accumulation_steps
    expected_training_samples = args.train_pool_samples * args.epochs
    if args.train_pool_samples % args.accumulation_steps != 0:
        raise ValueError(
            "--train-pool-samples must be divisible by --accumulation-steps"
        )
    if used_training_samples != expected_training_samples:
        raise ValueError(
            "The multi-epoch training sequence must satisfy "
            "max_updates * accumulation_steps == train_pool_samples * epochs; "
            f"observed {used_training_samples} != {expected_training_samples}."
        )
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_fp32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def learning_rate_for_step(
    args: argparse.Namespace,
    step: int,
    scheduler_state: dict[str, Any] | None = None,
) -> float:
    if args.lr_schedule == "constant":
        return float(args.learning_rate)

    if args.lr_schedule == "monitor_plateau":
        if step <= args.warmup_updates:
            return float(args.learning_rate) * float(step) / float(args.warmup_updates)
        if scheduler_state is None:
            return float(args.learning_rate)
        return float(scheduler_state.get("current_learning_rate", args.learning_rate))

    if args.lr_schedule == "cosine":
        if step <= args.warmup_updates:
            return float(args.learning_rate) * float(step) / float(args.warmup_updates)


        progress = (float(step) - float(args.warmup_updates)) / (
            float(args.max_updates) - float(args.warmup_updates)
        )
        progress = min(max(progress, 0.0), 1.0)
        peak = float(args.learning_rate)
        floor_lr = float(args.cosine_min_lr)
        return floor_lr + 0.5 * (peak - floor_lr) * (
            1.0 + math.cos(math.pi * progress)
        )

    if step <= args.warmup_updates:
        return float(args.learning_rate) * float(step) / float(args.warmup_updates)
    if step <= args.lr_drop1_update:
        return float(args.learning_rate)
    if step <= args.lr_drop2_update:
        return float(args.learning_rate) * float(args.lr_drop1_factor)
    return float(args.learning_rate) * float(args.lr_drop2_factor)


def set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)


def initial_plateau_scheduler_state(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "kind": args.lr_schedule,
        "current_learning_rate": float(args.learning_rate),
        "best_monitor_mse": math.inf,
        "best_monitor_step": -1,
        "bad_monitor_count": 0,
        "cooldown_remaining": 0,
        "drops": 0,
    }


def initial_cosine_scheduler_state(args: argparse.Namespace) -> dict[str, Any]:


    return {
        "kind": "cosine",
        "current_learning_rate": float(args.learning_rate),
        "peak_learning_rate": float(args.learning_rate),
        "cosine_min_lr": float(args.cosine_min_lr),
        "warmup_updates": int(args.warmup_updates),
        "cosine_final_update": int(args.max_updates),
        "monitor_lr_control": "disabled",
    }


def initial_scheduler_state(args: argparse.Namespace) -> dict[str, Any]:
    if args.lr_schedule == "cosine":
        return initial_cosine_scheduler_state(args)
    return initial_plateau_scheduler_state(args)


def update_monitor_plateau_scheduler(
    *,
    args: argparse.Namespace,
    scheduler_state: dict[str, Any],
    current_step: int,
    validation_mse: float,
    events_path: Path,
) -> dict[str, Any]:
    if args.lr_schedule != "monitor_plateau":
        return scheduler_state

    previous_best = float(scheduler_state.get("best_monitor_mse", math.inf))
    current_lr = float(scheduler_state.get("current_learning_rate", args.learning_rate))
    improved = (
        not math.isfinite(previous_best)
        or validation_mse < previous_best * float(args.plateau_improvement_factor)
    )

    action = "observe"
    old_lr = current_lr

    if improved:
        scheduler_state["best_monitor_mse"] = float(validation_mse)
        scheduler_state["best_monitor_step"] = int(current_step)
        scheduler_state["bad_monitor_count"] = 0
        action = "improved"
    elif current_step < int(args.warmup_updates):
        action = "warmup_defer"
    elif int(scheduler_state.get("cooldown_remaining", 0)) > 0:
        scheduler_state["cooldown_remaining"] = int(scheduler_state["cooldown_remaining"]) - 1
        action = "cooldown"
    else:
        scheduler_state["bad_monitor_count"] = int(scheduler_state.get("bad_monitor_count", 0)) + 1
        action = "bad"
        if (
            int(scheduler_state["bad_monitor_count"]) >= int(args.plateau_patience)
            and int(scheduler_state.get("drops", 0)) < int(args.plateau_max_drops)
        ):
            new_lr = max(
                float(args.plateau_min_lr),
                current_lr * float(args.plateau_factor),
            )
            if new_lr < current_lr:
                scheduler_state["current_learning_rate"] = float(new_lr)
                scheduler_state["drops"] = int(scheduler_state.get("drops", 0)) + 1
                scheduler_state["cooldown_remaining"] = int(args.plateau_cooldown)
                scheduler_state["bad_monitor_count"] = 0
                action = "drop"

    record = {
        "kind": "LR_PLATEAU",
        "step": int(current_step),
        "validation_mse_norm": float(validation_mse),
        "previous_best_monitor_mse": previous_best,
        "best_monitor_mse": float(scheduler_state.get("best_monitor_mse", math.inf)),
        "best_monitor_step": int(scheduler_state.get("best_monitor_step", -1)),
        "improved": int(bool(improved)),
        "action": action,
        "old_learning_rate": float(old_lr),
        "current_learning_rate": float(scheduler_state.get("current_learning_rate", old_lr)),
        "bad_monitor_count": int(scheduler_state.get("bad_monitor_count", 0)),
        "cooldown_remaining": int(scheduler_state.get("cooldown_remaining", 0)),
        "drops": int(scheduler_state.get("drops", 0)),
        "patience": int(args.plateau_patience),
        "cooldown": int(args.plateau_cooldown),
        "factor": float(args.plateau_factor),
        "min_lr": float(args.plateau_min_lr),
    }
    append_jsonl(events_path, record)
    log(
        "LR_PLATEAU "
        f"step={current_step} action={action} "
        f"val_mse_norm={validation_mse:.12g} "
        f"best={record['best_monitor_mse']:.12g} "
        f"lr={record['current_learning_rate']:.12g} "
        f"bad={record['bad_monitor_count']} "
        f"cooldown={record['cooldown_remaining']} "
        f"drops={record['drops']}"
    )
    return scheduler_state


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def sha256_file(path: Path, *, chunk_bytes: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def snapshot_tree_hash(root: Path) -> dict[str, Any]:


    excluded_directories = {".git", "__pycache__"}
    excluded_suffixes = {".pyc", ".pyo"}
    entries: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in excluded_directories for part in relative.parts):
            continue
        if relative.suffix in excluded_suffixes:
            continue
        entries.append(f"{relative.as_posix()}\t{sha256_file(path)}\n")
    digest = hashlib.sha256("".join(entries).encode()).hexdigest()
    return {
        "root": str(root),
        "tree_sha256": digest,
        "file_count": len(entries),
    }


def resolve_code_snapshot(lat_padding: str) -> dict[str, Any]:


    code_snapshot_root = os.environ.get("AWI_CODE_SNAPSHOT_ROOT", "").strip()
    if code_snapshot_root:
        snapshot_root_path = Path(code_snapshot_root)
        if not snapshot_root_path.is_dir():
            raise FileNotFoundError(
                f"AWI_CODE_SNAPSHOT_ROOT is not a directory: {snapshot_root_path}"
            )
        return snapshot_tree_hash(snapshot_root_path)
    if lat_padding == "pole_aware":
        raise RuntimeError(
            "AWI_CODE_SNAPSHOT_ROOT must point to the immutable per-run code "
            "snapshot for pole-aware runs; refusing to start without it "
            "(fail-closed production gate)"
        )
    return {
        "root": None,
        "tree_sha256": None,
        "file_count": 0,
    }


def verify_resume_snapshot_gate(
    *,
    lat_padding: str,
    current_snapshot: dict[str, Any],
    existing_config: dict[str, Any],
) -> None:


    if lat_padding != "pole_aware":
        return
    current_tree = current_snapshot.get("tree_sha256")
    config_tree = (
        existing_config.get("provenance", {})
        .get("code_snapshot", {})
        .get("tree_sha256")
    )
    if not current_tree or not config_tree or current_tree != config_tree:
        raise ValueError(
            "Code-snapshot resume gate failed for pole-aware run: "
            f"current_tree={current_tree!r} run_config_tree={config_tree!r} "
            "(both must be present and identical)"
        )


def compute_config_hash(config_core: dict[str, Any]) -> str:


    hash_arguments = dict(config_core["arguments"])
    for volatile_key in ("output_root", "run_name", "resume", "self_test", "stop_updates"):
        hash_arguments.pop(volatile_key, None)
    if hash_arguments.get("gradient_clip_norm") is None:


        hash_arguments.pop("gradient_clip_norm", None)
    if hash_arguments.get("lat_padding") == "replicate":
        hash_arguments.pop("lat_padding", None)
    if hash_arguments.get("lr_schedule") != "cosine":
        hash_arguments.pop("cosine_min_lr", None)
    if hash_arguments.get("checkpoint_retention") == "legacy":


        hash_arguments.pop("checkpoint_retention", None)
    hash_material = {
        key: value
        for key, value in config_core.items()
        if key != "created_utc"
    }
    hash_material["arguments"] = hash_arguments
    return sha256_json(hash_material)


def record_checkpoint_hash(
    *,
    checkpoint_path: Path,
    manifest_path: Path,
    kind: str,
    step: int,
    size_bytes: int,
) -> str:


    digest = sha256_file(checkpoint_path)
    sidecar = checkpoint_path.parent / (checkpoint_path.name + ".sha256")
    temporary = sidecar.parent / (sidecar.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(f"{digest}  {checkpoint_path.name}\n")
    os.replace(temporary, sidecar)
    append_jsonl(
        manifest_path,
        {
            "kind": "CHECKPOINT_HASH",
            "checkpoint_kind": kind,
            "file_name": checkpoint_path.name,
            "step": int(step),
            "size_bytes": int(size_bytes),
            "sha256": digest,
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return digest


def promote_last_completed_chunk(
    *,
    latest_path: Path,
    recovery_path: Path,
    manifest_path: Path,
    step: int,
) -> dict[str, Any]:


    started = time.perf_counter()
    latest_sidecar = latest_path.parent / (latest_path.name + ".sha256")
    recorded_latest_digest = latest_sidecar.read_text().split()[0]
    observed_latest_digest = sha256_file(latest_path)
    if recorded_latest_digest != observed_latest_digest:
        raise ValueError(
            "latest.pt failed its integrity re-hash before recovery "
            f"promotion: sidecar={recorded_latest_digest} "
            f"observed={observed_latest_digest}"
        )
    temporary = recovery_path.parent / (recovery_path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        os.link(latest_path, temporary)
        method = "hardlink_rename"
    except OSError:
        with latest_path.open("rb") as source, temporary.open("wb") as sink:
            while True:
                block = source.read(1 << 22)
                if not block:
                    break
                sink.write(block)
            sink.flush()
            os.fsync(sink.fileno())
        method = "atomic_copy"
    os.replace(temporary, recovery_path)
    digest = sha256_file(recovery_path)
    size_bytes = recovery_path.stat().st_size
    promote_seconds = time.perf_counter() - started
    sidecar = recovery_path.parent / (recovery_path.name + ".sha256")
    sidecar_temporary = sidecar.parent / (sidecar.name + ".tmp")
    with sidecar_temporary.open("w", encoding="utf-8") as handle:
        handle.write(f"{digest}  {recovery_path.name}\n")
    os.replace(sidecar_temporary, sidecar)
    record = {
        "kind": "CHECKPOINT_HASH",
        "checkpoint_kind": "last_completed_chunk",
        "file_name": recovery_path.name,
        "step": int(step),
        "size_bytes": int(size_bytes),
        "sha256": digest,
        "promoted_from": latest_path.name,
        "promotion_method": method,
        "promotion_seconds": promote_seconds,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
    }
    append_jsonl(manifest_path, record)
    return record


def copy_and_gate_norm_stats(
    *,
    source_path: Path,
    run_copy_path: Path,
    expected_normalization: dict[str, dict[str, float]],
) -> dict[str, Any]:


    source_bytes = source_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if run_copy_path.exists():
        existing_sha256 = sha256_file(run_copy_path)
        if existing_sha256 != source_sha256:
            raise ValueError(
                "Existing run-directory norm-stats copy does not match the "
                f"canonical file: {existing_sha256} != {source_sha256}"
            )
    else:
        temporary = run_copy_path.parent / (run_copy_path.name + ".tmp")
        with temporary.open("wb") as handle:
            handle.write(source_bytes)
        os.replace(temporary, run_copy_path)
    reparsed = load_normalization_statistics(run_copy_path)
    if reparsed != expected_normalization:
        raise ValueError(
            "Run-directory norm-stats copy does not reproduce the exact "
            "normalization statistics embedded in the run configuration"
        )
    return {
        "source_path": str(source_path),
        "run_copy_path": str(run_copy_path),
        "sha256": source_sha256,
        "exact_equality_gate": "pass",
    }


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def write_json_strict(path: Path, value: Any) -> None:


    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.write("\n")
    os.replace(temporary, path)


def scheduler_summary_fields(
    args: argparse.Namespace,
    scheduler_state: dict[str, Any],
) -> dict[str, Any]:


    if args.lr_schedule == "cosine":
        return {
            "scheduler_lr_drop_count": 0,
            "scheduler_best_monitor_mse": None,
            "scheduler_best_monitor_step": None,
        }
    return {
        "scheduler_lr_drop_count": int(scheduler_state.get("drops", 0)),
        "scheduler_best_monitor_mse": float(
            scheduler_state.get("best_monitor_mse", float("nan"))
        ),
        "scheduler_best_monitor_step": int(
            scheduler_state.get("best_monitor_step", -1)
        ),
    }


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()


def append_metric_csv(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=METRIC_CSV_FIELDS,
            extrasaction="ignore",
        )
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in METRIC_CSV_FIELDS})
        handle.flush()


def append_standard_metric_csv(
    path: Path,
    row: dict[str, Any],
    fieldnames: list[str],
) -> None:

    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})
        handle.flush()


def read_dates(path: Path) -> list[str]:
    root = zarr.open_group(str(path), mode="r")
    values = root["dates"][:]
    result: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            result.append(value.decode("ascii"))
        else:
            result.append(str(value))
    return result


def season_name(date_string: str) -> str:
    month = int(date_string[5:7])
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def balanced_indices(
    dates: list[str],
    count: int,
    seed: int,
    *,
    allowed: set[int] | None = None,
    excluded: set[int] | None = None,
) -> list[int]:
    if count > len(dates):
        raise ValueError("Requested more indices than available dates")
    allowed_set = set(range(len(dates))) if allowed is None else set(allowed)
    excluded_set = set() if excluded is None else set(excluded)
    candidates = allowed_set - excluded_set
    if count > len(candidates):
        raise ValueError(
            f"Requested {count} indices from only {len(candidates)} candidates"
        )

    groups: dict[str, list[int]] = {name: [] for name in ("DJF", "MAM", "JJA", "SON")}
    for index in sorted(candidates):
        groups[season_name(dates[index])].append(index)

    rng = np.random.default_rng(seed)
    base = count // 4
    remainder = count % 4
    selected: list[int] = []
    for position, name in enumerate(("DJF", "MAM", "JJA", "SON")):
        requested = base + (1 if position < remainder else 0)
        group = np.asarray(groups[name], dtype=np.int64)
        if requested > len(group):
            raise ValueError(
                f"Season {name} has {len(group)} candidates but {requested} were requested"
            )
        chosen = rng.choice(group, size=requested, replace=False)
        selected.extend(int(value) for value in chosen)
    selected.sort()
    return selected


def parse_iso_date_prefix(value: str) -> date:

    return date.fromisoformat(str(value)[:10])


def indices_for_calendar_year_every_other(
    dates: list[str],
    *,
    year: int,
    start_offset: int,
) -> list[int]:
    if start_offset not in (0, 1):
        raise ValueError("start_offset must be 0 for Jan-1 parity or 1 for Jan-2 parity")
    parsed = [parse_iso_date_prefix(value) for value in dates]
    year_indices = [index for index, value in enumerate(parsed) if value.year == year]
    if len(year_indices) < 365:
        raise ValueError(
            f"Requested monitor year {year}, but only found {len(year_indices)} days"
        )
    return year_indices[start_offset::2]


def train_monitor_three_year_every_other(
    train_dates: list[str],
    *,
    count: int,
) -> list[int]:


    if count != 548:
        raise ValueError(
            "train_monitor_three_year_every_other is defined only for count=548"
        )

    year_specs = (
        (1980, 0),
        (1994, 1),
        (2008, 0),
    )
    selected: list[int] = []
    for year, start_offset in year_specs:
        selected.extend(
            indices_for_calendar_year_every_other(
                train_dates,
                year=year,
                start_offset=start_offset,
            )
        )

    selected = sorted(selected)
    if len(selected) != count:
        raise RuntimeError(
            f"Expected {count} fixed train-monitor indices, got {len(selected)}"
        )
    if len(set(selected)) != len(selected):
        raise RuntimeError("Fixed train-monitor policy produced duplicate indices")
    selected_dates = [parse_iso_date_prefix(train_dates[index]) for index in selected]
    if not all(1980 <= value.year <= 2008 for value in selected_dates):
        raise RuntimeError(
            "Fixed train-monitor policy selected a date outside 1980-2008"
        )
    return selected


def validation_every_other_indices(
    validation_dates: list[str],
    *,
    count: int,
) -> list[int]:


    selected = list(range(0, len(validation_dates), 2))
    if len(selected) != count:
        raise ValueError(
            f"Every-other validation monitor produced {len(selected)} indices, "
            f"but --validation-samples requested {count}"
        )
    if len(set(selected)) != len(selected):
        raise RuntimeError("Validation every-other policy produced duplicate indices")
    selected_dates = [
        parse_iso_date_prefix(validation_dates[index])
        for index in selected
    ]
    if not all(2009 <= value.year <= 2011 for value in selected_dates):
        raise RuntimeError(
            "Validation every-other policy selected a date outside 2009-2011"
        )
    return selected


def build_index_plan(
    *,
    train_dates: list[str],
    validation_dates: list[str],
    train_pool_samples: int,
    train_monitor_samples: int,
    validation_samples: int,
    sequence_samples: int,
    epochs: int,
    seed: int,
) -> dict[str, list[int]]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if sequence_samples != train_pool_samples * epochs:
        raise ValueError(
            "sequence_samples must equal train_pool_samples * epochs"
        )
    if train_pool_samples > len(train_dates):
        raise ValueError("Training pool does not fit in the training split")
    if train_monitor_samples > train_pool_samples:
        raise ValueError(
            "Train monitor must fit inside the optimization train pool"
        )


    if train_monitor_samples == 548:
        train_monitor = train_monitor_three_year_every_other(
            train_dates,
            count=train_monitor_samples,
        )
    else:
        train_monitor = balanced_indices(
            train_dates,
            train_monitor_samples,
            seed + 3,
        )

    train_monitor_set = set(train_monitor)
    if len(train_monitor_set) != len(train_monitor):
        raise RuntimeError("Train monitor contains duplicate indices")
    if train_monitor_samples and not train_monitor_set <= set(range(len(train_dates))):
        raise RuntimeError("Train monitor contains indices outside training split")

    fill_count = train_pool_samples - len(train_monitor)
    remaining_indices = np.asarray(
        sorted(set(range(len(train_dates))) - train_monitor_set),
        dtype=np.int64,
    )
    if fill_count > len(remaining_indices):
        raise ValueError(
            f"Requested {fill_count} additional optimization days from only "
            f"{len(remaining_indices)} dates outside the train monitor"
        )

    pool_rng = np.random.default_rng(seed)
    additional_pool = (
        []
        if fill_count == 0
        else [
            int(value)
            for value in pool_rng.choice(
                remaining_indices,
                size=fill_count,
                replace=False,
            )
        ]
    )
    train_pool = sorted(train_monitor + additional_pool)

    if validation_samples == len(range(0, len(validation_dates), 2)):
        validation = validation_every_other_indices(
            validation_dates,
            count=validation_samples,
        )
    else:
        validation = balanced_indices(
            validation_dates,
            validation_samples,
            seed + 4,
        )

    plan: dict[str, list[int]] = {
        "train_pool": train_pool,
        "train_monitor": train_monitor,
        "validation": validation,
    }
    training_sequence: list[int] = []
    train_pool_array = np.asarray(train_pool, dtype=np.int64)
    for epoch in range(1, epochs + 1):
        epoch_rng = np.random.default_rng(seed + 1000 + epoch)
        epoch_sequence = epoch_rng.permutation(train_pool_array).astype(int).tolist()
        if len(epoch_sequence) != train_pool_samples:
            raise RuntimeError(f"Epoch {epoch} has an unexpected length")
        if len(set(epoch_sequence)) != train_pool_samples:
            raise RuntimeError(f"Epoch {epoch} contains duplicate indices")
        if set(epoch_sequence) != set(train_pool):
            raise RuntimeError(f"Epoch {epoch} does not match the fixed train pool")
        plan[f"training_epoch_{epoch}"] = epoch_sequence
        training_sequence.extend(epoch_sequence)

    if len(training_sequence) != sequence_samples:
        raise RuntimeError("Training sequence has an unexpected total length")
    if not set(train_monitor) <= set(train_pool):
        raise RuntimeError("Train monitor must be a subset of the optimization pool")
    for epoch in range(1, epochs + 1):
        if not set(train_monitor) <= set(plan[f"training_epoch_{epoch}"]):
            raise RuntimeError(
                f"Train monitor is not fully present in training epoch {epoch}"
            )

    plan["training_sequence"] = training_sequence
    return plan


def load_normalization_statistics(
    path: Path = NORM_STATS,
) -> dict[str, dict[str, float]]:
    stats = read_json(path)
    required = ("t2m_inp", "tisr", "t2m_tgt", "orography")
    result: dict[str, dict[str, float]] = {}
    for name in required:
        if name not in stats:
            raise KeyError(f"Missing normalization statistics for {name}")
        mean = float(stats[name]["mean"])
        std = float(stats[name]["std"])
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            raise ValueError(f"Invalid normalization statistics for {name}")
        result[name] = {"mean": mean, "std": std}
    return result


def load_grid() -> tuple[np.ndarray, np.ndarray, torch.Tensor, dict[str, float]]:
    with xr.open_dataset(SIDECAR) as dataset:
        latitude = np.asarray(dataset["lat"].values, dtype=np.float64)
        longitude = np.asarray(dataset["lon"].values, dtype=np.float64)

    if latitude.shape != (EXPECTED_HEIGHT,):
        raise ValueError(f"Unexpected latitude shape: {latitude.shape}")
    if longitude.shape != (EXPECTED_WIDTH,):
        raise ValueError(f"Unexpected longitude shape: {longitude.shape}")
    if not np.all(np.diff(latitude) > 0):
        raise ValueError("Latitude is not strictly increasing")
    if not np.all(np.diff(longitude) > 0):
        raise ValueError("Longitude is not strictly increasing")

    longitude_spacing = float(np.median(np.diff(longitude)))
    longitude_wrap_gap = float(longitude[0] + 360.0 - longitude[-1])
    if not np.isclose(longitude_spacing, longitude_wrap_gap, rtol=0, atol=1e-4):
        raise ValueError("Longitude grid is not periodic within tolerance")

    weights = np.cos(np.deg2rad(latitude)).astype(np.float32)
    if not np.all(np.isfinite(weights)) or not np.all(weights > 0):
        raise ValueError("Latitude weights are invalid")
    weights /= weights.mean()
    weight_tensor = torch.from_numpy(weights).reshape(1, 1, EXPECTED_HEIGHT, 1)
    geometry = {
        "latitude_minimum": float(latitude.min()),
        "latitude_maximum": float(latitude.max()),
        "latitude_spacing": float(np.median(np.diff(latitude))),
        "longitude_minimum": float(longitude.min()),
        "longitude_maximum": float(longitude.max()),
        "longitude_spacing": longitude_spacing,
        "longitude_wrap_gap": longitude_wrap_gap,
        "latitude_weight_minimum": float(weights.min()),
        "latitude_weight_maximum": float(weights.max()),
        "latitude_weight_mean": float(weights.mean()),
    }
    return latitude, longitude, weight_tensor, geometry


def inspect_static_store(path: Path) -> dict[str, Any]:
    root = zarr.open_group(str(path), mode="r")
    if root.attrs.get("static_inputs_complete") is not True:
        raise ValueError(f"{path}: static_inputs_complete is not true")
    if "static_inputs" not in set(root.array_keys()):
        raise KeyError(f"{path}: static_inputs is missing")
    array = root["static_inputs"]
    if array.shape != (2, EXPECTED_HEIGHT, EXPECTED_WIDTH):
        raise ValueError(f"{path}: unexpected static shape {array.shape}")
    if array.chunks != (1, EXPECTED_HEIGHT, EXPECTED_WIDTH):
        raise ValueError(f"{path}: unexpected static chunks {array.chunks}")
    if array.dtype != np.dtype("float32"):
        raise TypeError(f"{path}: unexpected static dtype {array.dtype}")
    channel_names = list(root.attrs.get("static_input_channel_names", []))
    if channel_names != ["lsm", "cl"]:
        raise ValueError(f"{path}: unexpected static channel names {channel_names}")
    data = np.asarray(array[:], dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f"{path}: static_inputs contains NaN or Inf")
    if float(data.min()) < 0.0 or float(data.max()) > 1.0:
        raise ValueError(f"{path}: static_inputs lies outside [0, 1]")
    observed_hash = sha256_array(data)
    recorded_hash = root.attrs.get("static_inputs_array_c_order_sha256")
    source_hash = root.attrs.get("static_inputs_source_sha256")
    if observed_hash != EXPECTED_STATIC_ARRAY_SHA256:
        raise ValueError(f"{path}: embedded static-array hash mismatch")
    if recorded_hash != observed_hash:
        raise ValueError(f"{path}: recorded static-array hash mismatch")
    if source_hash != EXPECTED_SIDECAR_SHA256:
        raise ValueError(f"{path}: recorded sidecar source hash mismatch")
    return {
        "path": str(path),
        "shape": list(array.shape),
        "chunks": list(array.chunks),
        "dtype": str(array.dtype),
        "compressor": repr(array.compressor),
        "channel_names": channel_names,
        "source_sha256": source_hash,
        "array_sha256": observed_hash,
        "lsm_mean": float(data[0].mean(dtype=np.float64)),
        "cl_mean": float(data[1].mean(dtype=np.float64)),
    }

def make_loader(
    dataset: AWIDownscalingZarrDataset,
    indices: list[int],
    *,
    num_workers: int,
    persistent_workers: bool,
) -> DataLoader:
    arguments: dict[str, Any] = {
        "dataset": Subset(dataset, indices),
        "batch_size": 1,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": False,
    }
    if num_workers > 0:
        arguments["persistent_workers"] = persistent_workers
        arguments["prefetch_factor"] = 1
        arguments["timeout"] = 300
    return DataLoader(**arguments)


class TensorAccumulator:


    def __init__(self) -> None:
        self.sums: torch.Tensor | None = None
        self.correction_absmax: torch.Tensor | None = None
        self.required_absmax: torch.Tensor | None = None

    def update(
        self,
        *,
        prediction: torch.Tensor,
        target: torch.Tensor,
        baseline: torch.Tensor,
        correction: torch.Tensor,
        required: torch.Tensor,
        latitude_weights: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        weight = latitude_weights * mask.to(dtype=torch.float32)
        weight = weight.expand_as(prediction)
        denominator = weight.sum()

        error = prediction.float() - target.float()
        baseline_error = baseline.float() - target.float()
        correction_f = correction.float()
        required_f = required.float()
        values = torch.stack(
            (
                denominator,
                (weight * error.square()).sum(),
                (weight * error.abs()).sum(),
                (weight * error).sum(),
                (weight * baseline_error.square()).sum(),
                (weight * correction_f).sum(),
                (weight * required_f).sum(),
                (weight * correction_f.square()).sum(),
                (weight * required_f.square()).sum(),
                (weight * correction_f * required_f).sum(),
            )
        )
        self.sums = values if self.sums is None else self.sums + values
        expanded_mask = mask.expand_as(correction_f)

        correction_max = correction_f.abs().masked_fill(
            ~expanded_mask,
            float("-inf"),
        ).max()

        required_max = required_f.abs().masked_fill(
            ~expanded_mask,
            float("-inf"),
        ).max()
        self.correction_absmax = (
            correction_max
            if self.correction_absmax is None
            else torch.maximum(self.correction_absmax, correction_max)
        )
        self.required_absmax = (
            required_max
            if self.required_absmax is None
            else torch.maximum(self.required_absmax, required_max)
        )

    def finalize(self, *, target_std: float) -> dict[str, float]:
        if (
            self.sums is None
            or self.correction_absmax is None
            or self.required_absmax is None
        ):
            return {
                key: float("nan")
                for key in (
                    "mse_norm",
                    "mse_K2",
                    "rmse_K",
                    "mae_norm",
                    "mae_K",
                    "bias_norm",
                    "bias_K",
                    "baseline_mse_norm",
                    "baseline_mse_K2",
                    "baseline_rmse_K",
                    "skill",
                    "corr_corr",
                    "correction_mean",
                    "correction_std",
                    "correction_rms",
                    "correction_absmax",
                    "required_mean",
                    "required_std",
                    "required_rms",
                    "required_absmax",
                )
            }

        packed = torch.cat(
            (
                self.sums,
                self.correction_absmax.reshape(1),
                self.required_absmax.reshape(1),
            )
        ).detach().cpu().numpy().astype(np.float64)
        (
            sum_weight,
            sum_error2,
            sum_abs_error,
            sum_error,
            sum_baseline_error2,
            sum_correction,
            sum_required,
            sum_correction2,
            sum_required2,
            sum_cross,
            correction_absmax,
            required_absmax,
        ) = packed.tolist()

        denominator_floor = 1.0e-20

        if (
            not math.isfinite(sum_weight)
            or sum_weight <= denominator_floor
        ):
            return {
                key: float("nan")
                for key in (
                    "mse_norm",
                    "mse_K2",
                    "rmse_K",
                    "mae_norm",
                    "mae_K",
                    "bias_norm",
                    "bias_K",
                    "baseline_mse_norm",
                    "baseline_mse_K2",
                    "baseline_rmse_K",
                    "skill",
                    "corr_corr",
                    "correction_mean",
                    "correction_std",
                    "correction_rms",
                    "correction_absmax",
                    "required_mean",
                    "required_std",
                    "required_rms",
                    "required_absmax",
                )
            }

        mse = sum_error2 / sum_weight
        mae = sum_abs_error / sum_weight
        bias = sum_error / sum_weight
        baseline_mse = sum_baseline_error2 / sum_weight
        correction_mean = sum_correction / sum_weight
        required_mean = sum_required / sum_weight
        correction_second = sum_correction2 / sum_weight
        required_second = sum_required2 / sum_weight
        correction_variance = max(correction_second - correction_mean**2, 0.0)
        required_variance = max(required_second - required_mean**2, 0.0)
        covariance = sum_cross / sum_weight - correction_mean * required_mean
        variance_floor = 1.0e-20
        if correction_variance <= variance_floor or required_variance <= variance_floor:
            correlation = float("nan")
        else:
            correlation = covariance / math.sqrt(
                correction_variance * required_variance
            )

        baseline_floor = 1.0e-20

        if baseline_mse <= baseline_floor:
            skill = float("nan")
        else:
            skill = 1.0 - mse / baseline_mse

        return {
            "mse_norm": mse,
            "mse_K2": mse * target_std**2,
            "rmse_K": math.sqrt(mse) * target_std,
            "mae_norm": mae,
            "mae_K": mae * target_std,
            "bias_norm": bias,
            "bias_K": bias * target_std,
            "baseline_mse_norm": baseline_mse,
            "baseline_mse_K2": baseline_mse * target_std**2,
            "baseline_rmse_K": math.sqrt(baseline_mse) * target_std,
            "skill": skill,
            "corr_corr": correlation,
            "correction_mean": correction_mean,
            "correction_std": math.sqrt(correction_variance),
            "correction_rms": math.sqrt(max(correction_second, 0.0)),
            "correction_absmax": correction_absmax,
            "required_mean": required_mean,
            "required_std": math.sqrt(required_variance),
            "required_rms": math.sqrt(max(required_second, 0.0)),
            "required_absmax": required_absmax,
        }


class SpatialErrorAccumulator:


    def __init__(self) -> None:
        self.count: torch.Tensor | None = None
        self.sum_signed_error: torch.Tensor | None = None
        self.sum_abs_error: torch.Tensor | None = None
        self.sum_error2: torch.Tensor | None = None
        self.sum_baseline_error2: torch.Tensor | None = None

    def update(
        self,
        *,
        prediction: torch.Tensor,
        target: torch.Tensor,
        baseline: torch.Tensor,
    ) -> None:
        error = prediction.float() - target.float()
        baseline_error = baseline.float() - target.float()
        finite = torch.isfinite(error) & torch.isfinite(baseline_error)
        zero = torch.zeros((), dtype=error.dtype, device=error.device)
        masked_error = torch.where(finite, error, zero)
        masked_baseline = torch.where(finite, baseline_error, zero)

        count = finite.to(torch.float32).sum(dim=(0, 1))
        sum_signed = masked_error.sum(dim=(0, 1))
        sum_abs = masked_error.abs().sum(dim=(0, 1))
        sum_error2 = masked_error.square().sum(dim=(0, 1))
        sum_baseline_error2 = masked_baseline.square().sum(dim=(0, 1))
        if self.count is None:
            self.count = count
            self.sum_signed_error = sum_signed
            self.sum_abs_error = sum_abs
            self.sum_error2 = sum_error2
            self.sum_baseline_error2 = sum_baseline_error2
        else:
            self.count = self.count + count
            self.sum_signed_error = self.sum_signed_error + sum_signed
            self.sum_abs_error = self.sum_abs_error + sum_abs
            self.sum_error2 = self.sum_error2 + sum_error2
            self.sum_baseline_error2 = self.sum_baseline_error2 + sum_baseline_error2

    def save(
        self,
        path: Path,
        *,
        latitude: np.ndarray,
        longitude: np.ndarray,
        target_std: float,
        step: int,
        split: str,
    ) -> None:
        if self.count is None:
            return
        count = self.count.detach().cpu().numpy().astype(np.float64)
        sum_signed = self.sum_signed_error.detach().cpu().numpy().astype(np.float64)
        sum_abs = self.sum_abs_error.detach().cpu().numpy().astype(np.float64)
        sum_error2 = self.sum_error2.detach().cpu().numpy().astype(np.float64)
        sum_baseline_error2 = (
            self.sum_baseline_error2.detach().cpu().numpy().astype(np.float64)
        )
        safe_count = np.where(count > 0, count, np.nan)
        model_rmse_K = np.sqrt(sum_error2 / safe_count) * target_std
        model_mae_K = (sum_abs / safe_count) * target_std
        baseline_rmse_K = np.sqrt(sum_baseline_error2 / safe_count) * target_std
        model_bias_K = (sum_signed / safe_count) * target_std
        with np.errstate(divide="ignore", invalid="ignore"):
            skill = np.where(
                sum_baseline_error2 > 0,
                1.0 - sum_error2 / sum_baseline_error2,
                np.nan,
            )
        np.savez_compressed(
            path,
            step=np.int64(step),
            split=np.asarray(split),
            sample_count=count.astype(np.int64),
            latitude=latitude,
            longitude=longitude,
            sum_signed_error=sum_signed,
            sum_abs_error=sum_abs,
            sum_error2=sum_error2,
            sum_baseline_error2=sum_baseline_error2,
            model_rmse_K=model_rmse_K.astype(np.float32),
            model_mae_K=model_mae_K.astype(np.float32),
            baseline_rmse_K=baseline_rmse_K.astype(np.float32),
            model_bias_K=model_bias_K.astype(np.float32),
            skill=skill.astype(np.float32),
        )


class SpectraAccumulator:


    FIELDS = ("target", "baseline", "prediction", "model_error", "baseline_error")

    def __init__(self) -> None:
        self.count: int = 0
        self.sums: dict[str, torch.Tensor] | None = None

    def update(
        self,
        *,
        prediction: torch.Tensor,
        target: torch.Tensor,
        baseline: torch.Tensor,
        latitude_weights: torch.Tensor,
    ) -> None:
        width = target.shape[-1]
        weight = latitude_weights.reshape(-1).to(torch.float32)
        weight_sum = weight.sum()
        fields = {
            "target": target.float(),
            "baseline": baseline.float(),
            "prediction": prediction.float(),
            "model_error": prediction.float() - target.float(),
            "baseline_error": baseline.float() - target.float(),
        }
        powers: dict[str, torch.Tensor] = {}
        for name, field in fields.items():
            coefficients = torch.fft.rfft(field, dim=-1) / width
            power = coefficients.abs().square()
            weighted = (power * weight.view(1, 1, -1, 1)).sum(dim=2) / weight_sum
            powers[name] = weighted.sum(dim=(0, 1))
        self.count += int(target.shape[0])
        if self.sums is None:
            self.sums = powers
        else:
            for name in self.FIELDS:
                self.sums[name] = self.sums[name] + powers[name]

    def save(
        self,
        path: Path,
        *,
        target_std: float,
        step: int,
        split: str,
        width: int,
    ) -> None:
        if self.sums is None or self.count == 0:
            return
        scale = target_std ** 2
        averaged = {
            name: (self.sums[name] / self.count * scale)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
            for name in self.FIELDS
        }
        wavenumber = np.arange(averaged["target"].shape[0])
        np.savez_compressed(
            path,
            step=np.int64(step),
            split=np.asarray(split),
            sample_count=np.int64(self.count),
            grid_width=np.int64(width),
            wavenumber=wavenumber,
            target_power_K2=averaged["target"],
            baseline_power_K2=averaged["baseline"],
            prediction_power_K2=averaged["prediction"],
            model_error_power_K2=averaged["model_error"],
            baseline_error_power_K2=averaged["baseline_error"],
        )


def region_masks(
    inputs: torch.Tensor,
    latitude: torch.Tensor,
    *,
    orography_mean: float,
    orography_std: float,
) -> dict[str, torch.Tensor]:
    lsm = inputs[:, 3:4]
    elevation_m = inputs[:, 2:3].float() * orography_std + orography_mean
    lat = latitude.view(1, 1, EXPECTED_HEIGHT, 1)
    ones = torch.ones_like(lsm, dtype=torch.bool)
    return {
        "global": ones,
        "land": lsm > 0.5,
        "ocean": lsm < 0.5,
        "elevation_gt_1000m": elevation_m > 1000.0,
        "elevation_le_1000m": elevation_m <= 1000.0,
        "tropics": lat.abs() < 23.5,
        "midlat_north": (lat >= 23.5) & (lat < 60.0),
        "midlat_south": (lat <= -23.5) & (lat > -60.0),
        "highlat_north": (lat >= 60.0) & (lat < 80.0),
        "highlat_south": (lat <= -60.0) & (lat > -80.0),
        "arctic_gt_80N": lat >= 80.0,
        "antarctic_lt_80S": lat <= -80.0,
    }


def weighted_mean_2d(values: torch.Tensor, latitude_weights: torch.Tensor) -> float:
    weights = latitude_weights.view(-1)
    flattened = values.reshape(-1)
    if flattened.numel() != weights.numel():
        raise ValueError("weighted_mean_2d expects one value per latitude row")
    return float((flattened * weights).sum().item() / weights.sum().item())


class SeamAccumulator:
    def __init__(self) -> None:
        self.sums: torch.Tensor | None = None
        self.seam_values: list[np.ndarray] = []
        self.sampled_interior_values: list[np.ndarray] = []

    def update(
        self,
        *,
        prediction: torch.Tensor,
        target: torch.Tensor,
        latitude_weights: torch.Tensor,
        interior_columns: torch.Tensor,
    ) -> None:
        pred = prediction[0, 0].float()
        truth = target[0, 0].float()
        seam = (
            (pred[:, 0] - pred[:, -1])
            - (truth[:, 0] - truth[:, -1])
        ).abs()
        interior = (
            (pred[:, 1:] - pred[:, :-1])
            - (truth[:, 1:] - truth[:, :-1])
        ).abs()
        lat_weight = latitude_weights.view(-1)
        absolute_error = (pred - truth).abs()
        seam_band = torch.cat(
            (absolute_error[:, :2], absolute_error[:, -2:]),
            dim=1,
        )
        values = torch.stack(
            (
                (seam * lat_weight).sum(),
                lat_weight.sum(),
                (interior * lat_weight[:, None]).sum(),
                lat_weight.sum() * interior.shape[1],
                (seam_band * lat_weight[:, None]).sum(),
                lat_weight.sum() * seam_band.shape[1],
                (absolute_error * lat_weight[:, None]).sum(),
                lat_weight.sum() * absolute_error.shape[1],
            )
        )
        self.sums = values if self.sums is None else self.sums + values
        self.seam_values.append(seam.detach().cpu().numpy())
        self.sampled_interior_values.append(
            interior.index_select(1, interior_columns)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

    def finalize(self, *, target_std: float) -> dict[str, float]:
        if self.sums is None:
            raise RuntimeError("No seam samples were accumulated")
        packed = self.sums.detach().cpu().numpy().astype(np.float64)
        (
            weighted_seam_sum,
            weighted_seam_weight,
            weighted_interior_sum,
            weighted_interior_weight,
            seam_band_abs_error_sum,
            seam_band_weight,
            global_abs_error_sum,
            global_weight,
        ) = packed.tolist()
        denominator_floor = 1.0e-20

        denominators = (
            weighted_seam_weight,
            weighted_interior_weight,
            seam_band_weight,
            global_weight,
        )

        if any(
            (not math.isfinite(value))
            or value <= denominator_floor
            for value in denominators
        ):
            return {
                key: float("nan")
                for key in (
                    "seam_abs_mean",
                    "seam_abs_median",
                    "interior_abs_mean",
                    "interior_sample_median",
                    "seam_to_interior_mean_ratio",
                    "seam_median_percentile",
                    "seam_band_mae_K",
                    "global_mae_K",
                )
            }

        seam_flat = np.concatenate(self.seam_values)
        interior_flat = np.concatenate(
            self.sampled_interior_values
        )

        seam_mean = (
            weighted_seam_sum
            / weighted_seam_weight
        )
        interior_mean = weighted_interior_sum / weighted_interior_weight
        seam_median = float(np.median(seam_flat))
        interior_median = float(np.median(interior_flat))
        percentile = float(100.0 * np.mean(interior_flat <= seam_median))
        seam_band_mae = (
            seam_band_abs_error_sum
            / seam_band_weight
        )
        global_mae = (
            global_abs_error_sum
            / global_weight
        )

        seam_ratio = (
            float("nan")
            if interior_mean <= denominator_floor
            else seam_mean / interior_mean
        )

        return {
            "seam_abs_mean": seam_mean,
            "seam_abs_median": seam_median,
            "interior_abs_mean": interior_mean,
            "interior_sample_median": interior_median,
            "seam_to_interior_mean_ratio": seam_ratio,
            "seam_median_percentile": percentile,
            "seam_band_mae_K": seam_band_mae * target_std,
            "global_mae_K": global_mae * target_std,
        }


def activation_hook_factory(
    name: str,
    storage: dict[str, dict[str, float]],
):
    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        tensor = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(tensor, torch.Tensor):
            return
        detached = tensor.detach().float()
        storage[name] = {
            "mean": float(detached.mean().item()),
            "std": float(detached.std(unbiased=False).item()),
            "minimum": float(detached.min().item()),
            "maximum": float(detached.max().item()),
            "rms": float(detached.square().mean().sqrt().item()),
            "finite_fraction": float(torch.isfinite(detached).float().mean().item()),
        }

    return hook


def register_stage_hooks(
    model: GlobalUNet,
) -> tuple[dict[str, dict[str, float]], list[Any]]:
    storage: dict[str, dict[str, float]] = {}
    handles: list[Any] = []
    for index, module in enumerate(model.encoder_blocks):
        handles.append(
            module.register_forward_hook(
                activation_hook_factory(f"encoder_{index}", storage)
            )
        )
    handles.append(
        model.bottleneck.register_forward_hook(
            activation_hook_factory("bottleneck", storage)
        )
    )
    for index, module in enumerate(model.decoder_blocks):
        handles.append(
            module.register_forward_hook(
                activation_hook_factory(f"decoder_{index}", storage)
            )
        )
    handles.append(
        model.head.register_forward_hook(activation_hook_factory("head", storage))
    )
    return storage, handles


def parameter_stage(name: str) -> str:
    if name.startswith("encoder_blocks."):
        return "encoder_" + name.split(".")[1]
    if name.startswith("bottleneck."):
        return "bottleneck"
    if name.startswith("decoder_projections."):
        return "decoder_projection_" + name.split(".")[1]
    if name.startswith("decoder_blocks."):
        return "decoder_" + name.split(".")[1]
    if name.startswith("head."):
        return "head"
    return "other"


def gradient_norms_by_stage(model: GlobalUNet) -> dict[str, float]:
    squared: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        if not bool(torch.isfinite(gradient).all()):
            raise ValueError(f"Non-finite gradient in {name}")
        stage = parameter_stage(name)
        squared[stage] = squared.get(stage, 0.0) + float(gradient.square().sum().item())
    return {stage: math.sqrt(value) for stage, value in squared.items()}


def gradient_norms_by_parameter(
    model: GlobalUNet,
) -> dict[str, float]:

    result: dict[str, float] = {}

    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue

        gradient = parameter.grad.detach().float()

        if not bool(torch.isfinite(gradient).all()):
            raise ValueError(
                f"Non-finite gradient in {name}"
            )

        result[name] = math.sqrt(
            float(gradient.square().sum().item())
        )

    return result


def snapshot_parameter_values(model: GlobalUNet) -> dict[str, torch.Tensor]:

    return {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def parameter_kind(name: str) -> str:
    return "biases" if name.endswith(".bias") else "weights"


def parameter_group_stages(stage: str) -> list[str]:


    groups = ["global", stage]
    if stage.startswith("encoder_") and stage != "encoder_stack":
        groups.append("encoder_stack")
    if stage.startswith("decoder_") and not stage.startswith("decoder_projection"):
        groups.append("decoder_stack")
    return groups


def parameter_update_statistics(
    model: GlobalUNet,
    before: dict[str, torch.Tensor],
    *,
    step: int,
    bias_rms_floor: float = 1.0e-12,
) -> list[dict[str, Any]]:

    groups: dict[tuple[str, str], dict[str, float]] = {}
    for name, parameter in model.named_parameters():
        previous = before.get(name)
        if previous is None:
            continue
        current = parameter.detach().float().cpu()
        delta = current - previous
        stage = parameter_stage(name)
        kind = parameter_kind(name)
        for group_stage in parameter_group_stages(stage):
            key = (group_stage, kind)
            entry = groups.setdefault(
                key,
                {
                    "count": 0.0,
                    "parameter_sse_before": 0.0,
                    "parameter_delta_sse": 0.0,
                },
            )
            entry["count"] += float(previous.numel())
            entry["parameter_sse_before"] += float(previous.square().sum().item())
            entry["parameter_delta_sse"] += float(delta.square().sum().item())

    records: list[dict[str, Any]] = []
    for (stage, kind), values in sorted(groups.items()):
        count = int(values["count"])
        if count == 0:
            continue
        parameter_rms_before = math.sqrt(values["parameter_sse_before"] / count)
        parameter_delta_rms = math.sqrt(values["parameter_delta_sse"] / count)
        is_bias_na = kind == "biases" and parameter_rms_before < bias_rms_floor
        if parameter_rms_before <= 0.0 or is_bias_na:
            relative_update_percent = float("nan")
        else:
            relative_update_percent = (
                100.0 * parameter_delta_rms / parameter_rms_before
            )
        record = {
            "kind": "PARAM_UPDATE",
            "step": step,
            "stage": stage,
            "parameter_kind": kind,
            "parameter_count": count,
            "parameter_rms_before": parameter_rms_before,
            "parameter_delta_rms": parameter_delta_rms,
            "relative_update_percent": relative_update_percent,
            "bias_rms_floor": bias_rms_floor,
            "bias_relative_update_percent_is_na": int(is_bias_na),
        }
        if kind == "biases":
            record["absolute_bias_update_rms"] = parameter_delta_rms
        records.append(record)
    return records


def parameter_health_statistics(
    model: GlobalUNet,
    *,
    step: int,
    max_percentile_values_per_tensor: int = 20_000,
) -> list[dict[str, Any]]:

    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        values = parameter.detach().float().cpu().reshape(-1)
        if values.numel() == 0:
            continue
        stage = parameter_stage(name)
        kind = parameter_kind(name)
        abs_values = values.abs()
        if abs_values.numel() <= max_percentile_values_per_tensor:
            sample = abs_values
        else:
            stride = math.ceil(abs_values.numel() / max_percentile_values_per_tensor)
            sample = abs_values[::stride][:max_percentile_values_per_tensor]
        for group_stage in parameter_group_stages(stage):
            entry = groups.setdefault(
                (group_stage, kind),
                {
                    "count": 0,
                    "sse": 0.0,
                    "abs_sum": 0.0,
                    "minimum": math.inf,
                    "maximum": -math.inf,
                    "samples": [],
                },
            )
            entry["count"] += int(values.numel())
            entry["sse"] += float(values.square().sum().item())
            entry["abs_sum"] += float(abs_values.sum().item())
            entry["minimum"] = min(entry["minimum"], float(values.min().item()))
            entry["maximum"] = max(entry["maximum"], float(values.max().item()))
            entry["samples"].append(sample)

    records: list[dict[str, Any]] = []
    for (stage, kind), values in sorted(groups.items()):
        count = int(values["count"])
        if count == 0:
            continue
        samples = torch.cat(values["samples"])
        rms = math.sqrt(float(values["sse"]) / count)
        record = {
            "kind": "PARAM_HEALTH",
            "step": step,
            "stage": stage,
            "parameter_kind": kind,
            "parameter_count": count,
            "parameter_rms": rms,
            "parameter_mean_abs": float(values["abs_sum"]) / count,
            "parameter_median_abs": float(torch.quantile(samples, 0.50).item()),
            "parameter_p05_abs": float(torch.quantile(samples, 0.05).item()),
            "parameter_p95_abs": float(torch.quantile(samples, 0.95).item()),
            "parameter_p99_abs": float(torch.quantile(samples, 0.99).item()),
            "parameter_min": float(values["minimum"]),
            "parameter_max": float(values["maximum"]),
            "percentile_method": "deterministic_strided_sample",
            "percentile_sample_count": int(samples.numel()),
        }
        records.append(record)
    return records


def global_gradient_norm(model: GlobalUNet) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            gradient = parameter.grad.detach().float()
            if not bool(torch.isfinite(gradient).all()):
                raise ValueError("A model gradient contains NaN or Inf")
            total += float(gradient.square().sum().item())
    return math.sqrt(total)


def weighted_batch_metrics(
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
    baseline: torch.Tensor,
    latitude_weights: torch.Tensor,
    target_std: float,
) -> dict[str, float]:
    mask = torch.ones_like(prediction, dtype=torch.bool)
    accumulator = TensorAccumulator()
    correction = prediction - baseline
    required = target - baseline
    accumulator.update(
        prediction=prediction,
        target=target,
        baseline=baseline,
        correction=correction,
        required=required,
        latitude_weights=latitude_weights,
        mask=mask,
    )
    return accumulator.finalize(target_std=target_std)


def save_map_snapshot(
    path: Path,
    *,
    index: int,
    date: str,
    latitude: np.ndarray,
    longitude: np.ndarray,
    inputs: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    baseline: torch.Tensor,
    orography_mean: float,
    orography_std: float,
) -> None:
    target_np = target[0, 0].detach().cpu().numpy().astype(np.float32)
    prediction_np = prediction[0, 0].detach().cpu().numpy().astype(np.float32)
    baseline_np = baseline[0, 0].detach().cpu().numpy().astype(np.float32)
    correction_np = prediction_np - baseline_np
    required_np = target_np - baseline_np
    signed_error = prediction_np - target_np
    baseline_signed_error = baseline_np - target_np
    input_np = inputs[0].detach().cpu().numpy().astype(np.float32)
    elevation = input_np[2] * np.float32(orography_std) + np.float32(orography_mean)
    np.savez_compressed(
        path,
        index=np.int64(index),
        date=np.asarray(date),
        latitude=latitude,
        longitude=longitude,
        baseline=baseline_np,
        target=target_np,
        prediction=prediction_np,
        learned_correction=correction_np,
        required_correction=required_np,
        signed_error=signed_error,
        absolute_error=np.abs(signed_error),
        baseline_absolute_error=np.abs(baseline_signed_error),
        model_error2_minus_baseline_error2=(
            np.square(signed_error) - np.square(baseline_signed_error)
        ),
        lsm=input_np[3],
        cl=input_np[4],
        orography_m=elevation.astype(np.float32),
    )


def update_top_errors(
    candidates: list[dict[str, Any]],
    *,
    count: int,
    dataset_index: int,
    date: str,
    latitude: np.ndarray,
    longitude: np.ndarray,
    inputs: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    baseline: torch.Tensor,
    target_mean: float,
    target_std: float,
    orography_mean: float,
    orography_std: float,
) -> None:
    absolute_error_k = (prediction - target).abs()[0, 0].float() * target_std
    values, flat_indices = torch.topk(absolute_error_k.reshape(-1), k=count)
    pred_k = prediction[0, 0].float() * target_std + target_mean
    target_k = target[0, 0].float() * target_std + target_mean
    baseline_k = baseline[0, 0].float() * target_std + target_mean
    lsm = inputs[0, 3].float()
    cl = inputs[0, 4].float()
    elevation = inputs[0, 2].float() * orography_std + orography_mean
    for value, flat_index in zip(values, flat_indices):
        flat = int(flat_index.item())
        lat_index = flat // EXPECTED_WIDTH
        lon_index = flat % EXPECTED_WIDTH
        candidates.append(
            {
                "error_K": float(value.item()),
                "dataset_index": dataset_index,
                "date": date,
                "lat": float(latitude[lat_index]),
                "lon": float(longitude[lon_index]),
                "pred_K": float(pred_k[lat_index, lon_index].item()),
                "target_K": float(target_k[lat_index, lon_index].item()),
                "baseline_K": float(baseline_k[lat_index, lon_index].item()),
                "lsm": float(lsm[lat_index, lon_index].item()),
                "cl": float(cl[lat_index, lon_index].item()),
                "elevation_m": float(elevation[lat_index, lon_index].item()),
            }
        )
    candidates.sort(key=lambda item: item["error_K"], reverse=True)
    del candidates[count:]


def evaluate(
    *,
    split: str,
    step: int,
    model: GlobalUNet,
    loader: DataLoader,
    indices: list[int],
    dates: list[str],
    latitude_np: np.ndarray,
    longitude_np: np.ndarray,
    latitude_device: torch.Tensor,
    latitude_weights: torch.Tensor,
    device: torch.device,
    target_mean: float,
    target_std: float,
    orography_mean: float,
    orography_std: float,
    top_error_count: int,
    map_path: Path | None,
    events_path: Path,
    metrics_path: Path,
    validation_metrics_path: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    model.eval()
    accumulators: dict[str, TensorAccumulator] = {}
    seam = SeamAccumulator()


    spatial = SpatialErrorAccumulator() if map_path is not None else None
    spectra = SpectraAccumulator() if map_path is not None else None
    interior_columns = torch.linspace(
        0,
        EXPECTED_WIDTH - 2,
        steps=128,
        device=device,
    ).round().to(torch.long).unique()
    top_errors: list[dict[str, Any]] = []
    fixed_prediction: np.ndarray | None = None

    start = time.perf_counter()
    with torch.no_grad():
        for position, (cpu_inputs, cpu_targets) in enumerate(loader):
            dataset_index = indices[position]
            date = dates[dataset_index]
            inputs = cpu_inputs.to(device, non_blocking=True)
            target = cpu_targets.to(device, non_blocking=True)
            prediction = model(inputs)
            baseline = model.baseline_in_target_units(inputs)
            correction = prediction - baseline
            required = target - baseline
            if not bool(torch.isfinite(prediction).all()):
                raise ValueError(f"{split} prediction contains NaN or Inf")

            masks = region_masks(
                inputs,
                latitude_device,
                orography_mean=orography_mean,
                orography_std=orography_std,
            )
            for region, mask in masks.items():
                accumulator = accumulators.setdefault(region, TensorAccumulator())
                accumulator.update(
                    prediction=prediction,
                    target=target,
                    baseline=baseline,
                    correction=correction,
                    required=required,
                    latitude_weights=latitude_weights,
                    mask=mask,
                )

            seam.update(
                prediction=prediction,
                target=target,
                latitude_weights=latitude_weights,
                interior_columns=interior_columns,
            )

            if spatial is not None:
                spatial.update(
                    prediction=prediction,
                    target=target,
                    baseline=baseline,
                )
            if spectra is not None:
                spectra.update(
                    prediction=prediction,
                    target=target,
                    baseline=baseline,
                    latitude_weights=latitude_weights,
                )

            if split == "validation":
                update_top_errors(
                    top_errors,
                    count=top_error_count,
                    dataset_index=dataset_index,
                    date=date,
                    latitude=latitude_np,
                    longitude=longitude_np,
                    inputs=inputs,
                    target=target,
                    prediction=prediction,
                    baseline=baseline,
                    target_mean=target_mean,
                    target_std=target_std,
                    orography_mean=orography_mean,
                    orography_std=orography_std,
                )

            if position == 0:
                fixed_prediction = prediction[0, 0].detach().cpu().numpy().astype(np.float32)
                if map_path is not None:
                    save_map_snapshot(
                        map_path,
                        index=dataset_index,
                        date=date,
                        latitude=latitude_np,
                        longitude=longitude_np,
                        inputs=inputs,
                        target=target,
                        prediction=prediction,
                        baseline=baseline,
                        orography_mean=orography_mean,
                        orography_std=orography_std,
                    )

    elapsed = time.perf_counter() - start
    if spatial is not None and map_path is not None:
        spatial.save(
            map_path.parent / f"spatial_step_{step:05d}.npz",
            latitude=latitude_np,
            longitude=longitude_np,
            target_std=target_std,
            step=step,
            split=split,
        )
    if spectra is not None and map_path is not None:
        spectra.save(
            map_path.parent / f"spectra_step_{step:05d}.npz",
            target_std=target_std,
            step=step,
            split=split,
            width=int(longitude_np.shape[0]),
        )
    regional_results = {
        region: accumulator.finalize(target_std=target_std)
        for region, accumulator in accumulators.items()
    }
    seam_results = seam.finalize(target_std=target_std)

    for region, metrics in regional_results.items():
        record = {
            "kind": "EVAL",
            "step": step,
            "split": split,
            "region": region,
            **metrics,
        }
        append_jsonl(events_path, record)
        append_metric_csv(metrics_path, record)
        log(
            "EVAL "
            f"step={step} split={split} region={region} "
            f"mse_norm={metrics['mse_norm']:.12g} "
            f"mse_K2={metrics['mse_K2']:.9g} "
            f"rmse_K={metrics['rmse_K']:.9g} "
            f"mae_K={metrics['mae_K']:.9g} "
            f"bias_K={metrics['bias_K']:.9g} "
            f"baseline_mse_norm={metrics['baseline_mse_norm']:.12g} "
            f"baseline_mse_K2={metrics['baseline_mse_K2']:.9g} "
            f"skill={metrics['skill']:.9g} "
            f"corr_corr={metrics['corr_corr']:.9g} "
            f"corr_rms={metrics['correction_rms']:.9g} "
            f"required_rms={metrics['required_rms']:.9g}"
        )

    seam_record = {
        "kind": "SEAM",
        "step": step,
        "split": split,
        "region": "seam",
        **seam_results,
    }
    append_jsonl(events_path, seam_record)
    append_metric_csv(metrics_path, seam_record)
    log(
        "SEAM "
        f"step={step} split={split} "
        f"seam_abs_mean={seam_results['seam_abs_mean']:.9g} "
        f"interior_abs_mean={seam_results['interior_abs_mean']:.9g} "
        f"ratio={seam_results['seam_to_interior_mean_ratio']:.9g} "
        f"seam_median_percentile={seam_results['seam_median_percentile']:.6g} "
        f"seam_band_mae_K={seam_results['seam_band_mae_K']:.9g} "
        f"global_mae_K={seam_results['global_mae_K']:.9g}"
    )

    if split == "validation":
        for rank, item in enumerate(top_errors, start=1):
            record = {"kind": "TOPERR", "step": step, "rank": rank, **item}
            append_jsonl(events_path, record)
            log(
                "TOPERR "
                f"step={step} rank={rank} date={item['date']} "
                f"index={item['dataset_index']} lat={item['lat']:.6f} "
                f"lon={item['lon']:.6f} pred_K={item['pred_K']:.6f} "
                f"target_K={item['target_K']:.6f} "
                f"baseline_K={item['baseline_K']:.6f} "
                f"error_K={item['error_K']:.6f} lsm={item['lsm']:.6f} "
                f"cl={item['cl']:.6f} elevation_m={item['elevation_m']:.3f}"
            )

    global_metrics = regional_results["global"]
    summary = {
        "kind": "EVAL_SUMMARY",
        "step": step,
        "split": split,
        "seconds": elapsed,
        "samples": len(indices),
        "global": global_metrics,
        "regions": regional_results,
        "seam": seam_results,
    }
    append_jsonl(events_path, summary)
    if split == "validation":
        append_standard_metric_csv(
            validation_metrics_path,
            {"step": step, **global_metrics},
            VALIDATION_METRIC_CSV_FIELDS,
        )
    log(
        f"EVAL_SUMMARY step={step} split={split} samples={len(indices)} "
        f"seconds={elapsed:.6f} samples_per_second={len(indices) / elapsed:.6f}"
    )
    model.train()
    if fixed_prediction is None:
        raise RuntimeError("Evaluation loader returned no samples")
    return summary, fixed_prediction


def checkpoint_payload(
    *,
    model: GlobalUNet,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    step: int,
    sample_position: int,
    best_validation_mse: float,
    best_validation_step: int,
    index_plan: dict[str, list[int]],
    train_history: list[dict[str, Any]],
    evaluation_history: list[dict[str, Any]],
    diagnostic_history: list[dict[str, Any]],
    scheduler_state: dict[str, Any],
    reload_reference: np.ndarray,
    reload_reference_validation_index: int,
    reload_reference_validation_date: str,
) -> dict[str, Any]:
    reference = np.asarray(reload_reference, dtype=np.float32)
    if reference.size == 0 or not np.isfinite(reference).all():
        raise ValueError("Checkpoint reload reference is empty or non-finite")
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "architecture": model.architecture_dict(),
        "config": config,
        "step": step,
        "sample_position": sample_position,
        "best_validation_mse": best_validation_mse,
        "best_validation_step": best_validation_step,
        "index_plan": index_plan,
        "train_history": train_history,
        "evaluation_history": evaluation_history,
        "diagnostic_history": diagnostic_history,
        "scheduler_state": dict(scheduler_state),
        "reload_reference": reference,
        "reload_reference_validation_index": int(reload_reference_validation_index),
        "reload_reference_validation_date": str(reload_reference_validation_date),
        "reload_reference_shape": list(reference.shape),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all(),
    }


def atomic_save_checkpoint(path: Path, payload: dict[str, Any]) -> int:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path.stat().st_size


def atomic_save_checkpoint_with_hash(
    path: Path,
    payload: dict[str, Any],
    *,
    manifest_path: Path,
    kind: str,
    step: int,
) -> int:


    size_bytes = atomic_save_checkpoint(path, payload)
    record_checkpoint_hash(
        checkpoint_path=path,
        manifest_path=manifest_path,
        kind=kind,
        step=step,
        size_bytes=size_bytes,
    )
    return size_bytes


def restore_checkpoint(
    path: Path,
    *,
    model: GlobalUNet,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    expected_snapshot_tree_sha256: str | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["architecture"] != model.architecture_dict():
        raise ValueError("Checkpoint architecture does not match the current model")
    if checkpoint["config"]["config_hash"] != config["config_hash"]:
        raise ValueError("Checkpoint configuration hash does not match this run")
    if expected_snapshot_tree_sha256 is not None:


        checkpoint_tree = (
            checkpoint.get("config", {})
            .get("provenance", {})
            .get("code_snapshot", {})
            .get("tree_sha256")
        )
        if not checkpoint_tree or checkpoint_tree != expected_snapshot_tree_sha256:
            raise ValueError(
                "Checkpoint code-snapshot tree hash is absent or does not "
                f"match the current snapshot: checkpoint={checkpoint_tree!r} "
                f"current={expected_snapshot_tree_sha256!r}"
            )
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    random.setstate(checkpoint["python_random_state"])
    np.random.set_state(checkpoint["numpy_random_state"])
    torch.set_rng_state(checkpoint["torch_random_state"])
    torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])
    return checkpoint


def fixed_validation_reload_reference(
    *,
    model: GlobalUNet,
    dataset: AWIDownscalingZarrDataset,
    validation_index: int,
    device: torch.device,
) -> np.ndarray:
    cpu_inputs, _ = dataset[validation_index]
    inputs = cpu_inputs.unsqueeze(0).to(device)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            prediction = (
                model(inputs)[0, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
    finally:
        if was_training:
            model.train()
    if not np.isfinite(prediction).all():
        raise ValueError("Reload-reference prediction contains NaN or Inf")
    return prediction


def verify_checkpoint_reload(
    *,
    checkpoint_path: Path,
    model_arguments: dict[str, Any],
    dataset: AWIDownscalingZarrDataset,
    validation_index: int,
    device: torch.device,
) -> dict[str, float | int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "reload_reference" not in checkpoint:
        raise ValueError(f"Checkpoint {checkpoint_path} has no reload_reference")
    reference_index = int(
        checkpoint.get("reload_reference_validation_index", validation_index)
    )
    if reference_index != validation_index:
        raise ValueError(
            "Checkpoint reload reference index does not match the requested "
            f"validation index: {reference_index} != {validation_index}"
        )
    fresh_model = GlobalUNet(**model_arguments).to(device)
    fresh_model.load_state_dict(checkpoint["model_state"])
    fresh_model.eval()
    cpu_inputs, _ = dataset[reference_index]
    inputs = cpu_inputs.unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = fresh_model(inputs)[0, 0].detach().cpu().numpy().astype(np.float32)
    reference = np.asarray(checkpoint["reload_reference"], dtype=np.float32)
    if reference.size == 0 or not np.isfinite(reference).all():
        raise ValueError(f"Checkpoint {checkpoint_path} has an invalid reload_reference")
    if prediction.shape != reference.shape:
        raise ValueError(
            "Checkpoint reload prediction shape does not match stored reference: "
            f"{prediction.shape} != {reference.shape}"
        )
    maximum_difference = float(np.max(np.abs(prediction - reference)))
    passed = int(np.allclose(prediction, reference, rtol=0.0, atol=1.0e-5))
    if not passed:
        raise ValueError(
            f"Checkpoint reload prediction mismatch: max abs difference {maximum_difference}"
        )
    return {
        "checkpoint_reload_pass": passed,
        "checkpoint_reload_max_abs_difference": maximum_difference,
        "checkpoint_reload_step": int(checkpoint["step"]),
        "checkpoint_reload_validation_index": reference_index,
    }


def main() -> None:
    args = parse_arguments()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; submit this program through SLURM")
    for required in (TRAIN_ZARR, VALID_ZARR, SIDECAR, NORM_STATS):
        if not required.exists():
            raise FileNotFoundError(required)

    git_commit = os.environ.get("AWI_GIT_COMMIT", "").strip()
    git_dirty = os.environ.get("AWI_GIT_DIRTY", "").strip()
    if not git_commit:
        raise RuntimeError("AWI_GIT_COMMIT was not supplied by the submission command")
    if git_dirty != "0":
        raise RuntimeError("Pilot jobs require AWI_GIT_DIRTY=0")

    set_seed(args.seed)
    configure_fp32()
    device = torch.device("cuda:0")
    device_properties = torch.cuda.get_device_properties(device)

    normalization = load_normalization_statistics()
    target_mean = normalization["t2m_tgt"]["mean"]
    target_std = normalization["t2m_tgt"]["std"]
    orography_mean = normalization["orography"]["mean"]
    orography_std = normalization["orography"]["std"]
    latitude_np, longitude_np, latitude_weights_cpu, geometry = load_grid()
    latitude_weights = latitude_weights_cpu.to(device)
    latitude_device = torch.from_numpy(latitude_np.astype(np.float32)).to(device)

    embedded_static = {
        "train": inspect_static_store(TRAIN_ZARR),
        "validation": inspect_static_store(VALID_ZARR),
    }
    if (
        embedded_static["train"]["array_sha256"]
        != embedded_static["validation"]["array_sha256"]
    ):
        raise ValueError("Train and validation embedded static arrays differ")

    train_dates = read_dates(TRAIN_ZARR)
    validation_dates = read_dates(VALID_ZARR)
    if len(train_dates) != EXPECTED_TRAIN_LENGTH:
        raise ValueError(f"Unexpected training length: {len(train_dates)}")
    if len(validation_dates) != EXPECTED_VALID_LENGTH:
        raise ValueError(f"Unexpected validation length: {len(validation_dates)}")

    sequence_samples = args.train_pool_samples * args.epochs
    index_plan = build_index_plan(
        train_dates=train_dates,
        validation_dates=validation_dates,
        train_pool_samples=args.train_pool_samples,
        train_monitor_samples=args.train_monitor_samples,
        validation_samples=args.validation_samples,
        sequence_samples=sequence_samples,
        epochs=args.epochs,
        seed=args.seed,
    )
    index_hashes = {name: sha256_json(values) for name, values in index_plan.items()}

    run_name = args.run_name.strip()
    if not run_name:
        job_id = os.environ.get("SLURM_JOB_ID", "nojid")
        run_name = f"norm-{args.norm}_seed-{args.seed}_{utc_timestamp()}_job-{job_id}"
    output_dir = args.output_root / run_name
    checkpoints_dir = output_dir / "checkpoints"
    maps_dir = output_dir / "maps"
    plots_dir = output_dir / "plots"
    config_path = output_dir / "config.json"
    events_path = output_dir / "events.jsonl"
    metrics_path = output_dir / "metrics.csv"
    train_metrics_path = output_dir / "train_metrics.csv"
    validation_metrics_path = output_dir / "validation_metrics.csv"
    latest_path = checkpoints_dir / "latest.pt"
    best_path = checkpoints_dir / "best_val_mse.pt"
    checkpoint_manifest_path = checkpoints_dir / "checkpoint_manifest.jsonl"
    chunk_log_path = output_dir / "run_chunks.jsonl"


    code_snapshot = resolve_code_snapshot(args.lat_padding)

    if output_dir.exists() and not args.resume:
        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory already exists and is non-empty: {output_dir}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    model_arguments = {
        "in_channels": 5,
        "base": args.base,
        "depth": args.depth,
        "norm": args.norm,
        "target_mode": "correction",
        "lat_padding": args.lat_padding,
        "input_t2m_mean": normalization["t2m_inp"]["mean"],
        "input_t2m_std": normalization["t2m_inp"]["std"],
        "target_t2m_mean": target_mean,
        "target_t2m_std": target_std,
    }
    reload_reference_validation_index = index_plan["validation"][0]
    reload_reference_validation_date = validation_dates[reload_reference_validation_index]

    config_core = {
        "git_commit": git_commit,
        "git_dirty": 0,
        "arguments": {
            **vars(args),
            "output_root": str(args.output_root),
            "evaluation_steps": list(args.evaluation_steps),
            "diagnostic_steps": list(args.diagnostic_steps),
            "map_steps": list(args.map_steps),
        },
        "model_arguments": model_arguments,
        "normalization": normalization,
        "embedded_static": embedded_static,
        "geometry": geometry,
        "index_hashes": index_hashes,
        "fixed_map_validation_index": reload_reference_validation_index,
        "fixed_map_validation_date": reload_reference_validation_date,
        "precision": "fp32",
        "tf32": False,
        "physical_batch_size": 1,
        "effective_batch_size": args.accumulation_steps,
        "constant_learning_rate": args.lr_schedule == "constant",
        "learning_rate_schedule": args.lr_schedule,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    if args.lat_padding == "pole_aware":


        config_core["pole_aware_boundary_scheme"] = POLE_AWARE_BOUNDARY_SCHEME
    config_hash = compute_config_hash(config_core)

    norm_stats_provenance = copy_and_gate_norm_stats(
        source_path=NORM_STATS,
        run_copy_path=output_dir / "norm_stats.json",
        expected_normalization=normalization,
    )


    provenance = {
        "code_snapshot": code_snapshot,
        "norm_stats": norm_stats_provenance,
        "checkpoint_manifest_path": str(checkpoint_manifest_path),
        "chunk_log_path": str(chunk_log_path),
    }
    config = {
        **config_core,
        "config_hash": config_hash,
        "provenance": provenance,
    }

    if not args.resume:
        write_json(config_path, config)
        for name, values in index_plan.items():
            write_json(
                output_dir / f"indices_{name}.json",
                {
                    "name": name,
                    "sha256": index_hashes[name],
                    "indices": values,
                    "dates": [
                        (train_dates if name != "validation" else validation_dates)[index]
                        for index in values
                    ],
                },
            )
    else:
        existing_config = read_json(config_path)
        if existing_config.get("config_hash") != config_hash:
            raise ValueError("Existing run configuration does not match requested resume")
        verify_resume_snapshot_gate(
            lat_padding=args.lat_padding,
            current_snapshot=code_snapshot,
            existing_config=existing_config,
        )

    train_dataset = AWIDownscalingZarrDataset(
        TRAIN_ZARR,
        include_static_inputs=True,
    )
    validation_dataset = AWIDownscalingZarrDataset(
        VALID_ZARR,
        include_static_inputs=True,
    )
    if train_dataset.input_shape != (5, EXPECTED_HEIGHT, EXPECTED_WIDTH):
        raise ValueError(f"Unexpected train input shape: {train_dataset.input_shape}")
    if validation_dataset.input_shape != (5, EXPECTED_HEIGHT, EXPECTED_WIDTH):
        raise ValueError(f"Unexpected validation input shape: {validation_dataset.input_shape}")

    model = GlobalUNet(**model_arguments).to(device)
    optimizer, optimizer_summaries = build_optimizer(
        model,
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    step = 0
    sample_position = 0
    best_validation_mse = math.inf
    best_validation_step = -1
    train_history: list[dict[str, Any]] = []
    evaluation_history: list[dict[str, Any]] = []
    diagnostic_history: list[dict[str, Any]] = []
    scheduler_state = initial_scheduler_state(args)
    latest_checkpoint_step: int | None = None

    if args.resume:
        if not latest_path.exists():
            raise FileNotFoundError(f"No latest checkpoint found at {latest_path}")
        checkpoint = restore_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            config=config,
            expected_snapshot_tree_sha256=(
                code_snapshot["tree_sha256"]
                if args.lat_padding == "pole_aware"
                else None
            ),
        )
        step = int(checkpoint["step"])
        sample_position = int(checkpoint["sample_position"])
        best_validation_mse = float(checkpoint["best_validation_mse"])
        best_validation_step = int(checkpoint["best_validation_step"])
        index_plan = {key: list(value) for key, value in checkpoint["index_plan"].items()}
        train_history = list(checkpoint["train_history"])
        evaluation_history = list(checkpoint["evaluation_history"])
        diagnostic_history = list(checkpoint["diagnostic_history"])
        scheduler_state = dict(checkpoint.get("scheduler_state", scheduler_state))
        latest_checkpoint_step = step

    append_jsonl(
        chunk_log_path,
        {
            "kind": "CHUNK_START",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "resume": bool(args.resume),
            "start_step": int(step),
            "start_sample_position": int(sample_position),
            "stop_updates": int(args.stop_updates),
            "max_updates": int(args.max_updates),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "hostname": socket.gethostname(),
            "git_commit": git_commit,
            "config_hash": config_hash,
            "code_snapshot": code_snapshot,
            "norm_stats_sha256": norm_stats_provenance["sha256"],
        },
    )

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    log("=" * 80)
    log("GLOBAL U-NET SECOND-STAGE ARCHITECTURE SCREEN")
    log("=" * 80)
    log("Matched second-stage architecture screen with fixed monitors; PD test remains untouched.")
    log(f"run_directory={output_dir}")
    log(f"run_name={run_name}")
    log(f"git_commit={git_commit}")
    log("git_dirty=0")
    log(f"hostname={socket.gethostname()}")
    log(f"platform={platform.platform()}")
    log(f"python={sys.version.split()[0]}")
    log(f"pytorch={torch.__version__}")
    log(f"cuda_build={torch.version.cuda}")
    log(f"gpu={torch.cuda.get_device_name(device)}")
    log(f"gpu_total_memory_GiB={device_properties.total_memory / (1024**3):.6f}")
    log(f"slurm_job_id={os.environ.get('SLURM_JOB_ID', '')}")
    log(f"norm={args.norm}")
    log("in_channels=5")
    log("target_mode=correction")
    log(f"base={args.base}")
    log(f"depth={args.depth}")
    log(f"lat_padding={args.lat_padding}")
    log(
        "pole_aware_boundary_scheme="
        + (POLE_AWARE_BOUNDARY_SCHEME if args.lat_padding == "pole_aware" else "n/a")
    )
    log(f"checkpoint_retention={args.checkpoint_retention}")
    log("precision=fp32")
    log("tf32=disabled")
    log(f"optimizer={args.optimizer}")
    log(f"learning_rate_peak={args.learning_rate:.12g}")
    log(f"weight_decay={args.weight_decay:.12g}")
    log(f"learning_rate_schedule={args.lr_schedule}")
    log(f"warmup_updates={args.warmup_updates}")
    log(f"lr_drop1_update={args.lr_drop1_update}")
    log(f"lr_drop2_update={args.lr_drop2_update}")
    log(f"lr_drop1_factor={args.lr_drop1_factor:.12g}")
    log(f"lr_drop2_factor={args.lr_drop2_factor:.12g}")
    log(f"plateau_improvement_factor={args.plateau_improvement_factor:.12g}")
    log(f"plateau_patience={args.plateau_patience}")
    log(f"plateau_cooldown={args.plateau_cooldown}")
    log(f"plateau_factor={args.plateau_factor:.12g}")
    log(f"plateau_max_drops={args.plateau_max_drops}")
    log(f"plateau_min_lr={args.plateau_min_lr:.12g}")
    log(f"cosine_min_lr={args.cosine_min_lr:.12g}")
    log(f"gradient_clip_norm={args.gradient_clip_norm}")
    log(f"stop_updates={args.stop_updates}")
    log(f"physical_batch_size=1")
    log(f"accumulation_steps={args.accumulation_steps}")
    log(f"effective_batch_size={args.accumulation_steps}")
    log(f"max_updates={args.max_updates}")
    log(f"epochs={args.epochs}")
    log(f"updates_per_epoch={args.train_pool_samples // args.accumulation_steps}")
    log(f"train_pool_samples={args.train_pool_samples}")
    steps_per_epoch = args.train_pool_samples // args.accumulation_steps
    log(f"steps_per_epoch={steps_per_epoch}")
    log(f"training_sequence_samples={len(index_plan['training_sequence'])}")
    log(f"train_monitor_samples={len(index_plan['train_monitor'])}")
    log(f"validation_samples={len(index_plan['validation'])}")
    log("train_monitor_policy=in_sample_subset_of_train_pool")
    log("validation_monitor_policy=fixed_every_other_validation_day")
    log(f"evaluation_steps={args.evaluation_steps}")
    log(f"diagnostic_steps={args.diagnostic_steps}")
    log(f"map_steps={args.map_steps}")
    log(f"num_workers={args.num_workers}")
    log(f"seed={args.seed}")
    log(f"config_hash={config_hash}")
    log(f"code_snapshot_root={code_snapshot['root']}")
    log(f"code_snapshot_tree_sha256={code_snapshot['tree_sha256']}")
    log(f"code_snapshot_file_count={code_snapshot['file_count']}")
    log(f"norm_stats_run_copy_sha256={norm_stats_provenance['sha256']}")
    for name, digest in index_hashes.items():
        log(f"index_hash_{name}={digest}")
    log(f"architecture={json.dumps(model.architecture_dict(), sort_keys=True)}")
    log(f"total_parameters={total_parameters}")
    log(f"trainable_parameters={trainable_parameters}")
    log(f"normalization={json.dumps(normalization, sort_keys=True)}")
    log(f"embedded_static={json.dumps(embedded_static, sort_keys=True)}")
    log(f"geometry={json.dumps(geometry, sort_keys=True)}")
    for summary in optimizer_summaries:
        log(
            "OPTIMIZER_GROUP "
            f"name={summary['name']} weight_decay={summary['weight_decay']:.12g} "
            f"parameter_tensors={summary['parameter_tensors']} "
            f"parameter_elements={summary['parameter_elements']}"
        )

    train_monitor_loader = make_loader(
        train_dataset,
        index_plan["train_monitor"],
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    validation_loader = make_loader(
        validation_dataset,
        index_plan["validation"],
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    def run_evaluations(current_step: int) -> tuple[dict[str, Any], np.ndarray]:
        map_path = (
            maps_dir / f"step_{current_step:05d}.npz"
            if current_step in args.map_steps
            else None
        )
        train_summary, _ = evaluate(
            split="train_monitor",
            step=current_step,
            model=model,
            loader=train_monitor_loader,
            indices=index_plan["train_monitor"],
            dates=train_dates,
            latitude_np=latitude_np,
            longitude_np=longitude_np,
            latitude_device=latitude_device,
            latitude_weights=latitude_weights,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
            orography_mean=orography_mean,
            orography_std=orography_std,
            top_error_count=args.top_errors,
            map_path=None,
            events_path=events_path,
            metrics_path=metrics_path,
            validation_metrics_path=validation_metrics_path,
        )
        validation_summary, fixed_prediction = evaluate(
            split="validation",
            step=current_step,
            model=model,
            loader=validation_loader,
            indices=index_plan["validation"],
            dates=validation_dates,
            latitude_np=latitude_np,
            longitude_np=longitude_np,
            latitude_device=latitude_device,
            latitude_weights=latitude_weights,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
            orography_mean=orography_mean,
            orography_std=orography_std,
            top_error_count=args.top_errors,
            map_path=map_path,
            events_path=events_path,
            metrics_path=metrics_path,
            validation_metrics_path=validation_metrics_path,
        )
        evaluation_history.extend((train_summary, validation_summary))
        return validation_summary, fixed_prediction

    if step == 0 and not args.resume:

        cpu_inputs, _ = validation_dataset[index_plan["validation"][0]]
        inputs0 = cpu_inputs.unsqueeze(0).to(device)
        activation_storage, handles = register_stage_hooks(model)
        model.eval()
        with torch.no_grad():
            _ = model(inputs0)
        for handle in handles:
            handle.remove()
        model.train()
        for stage, values in activation_storage.items():
            record = {"kind": "ACT", "step": 0, "stage": stage, **values}
            diagnostic_history.append(record)
            append_jsonl(events_path, record)
            log(
                "DIAG_ACT "
                f"step=0 stage={stage} mean={values['mean']:.9g} "
                f"std={values['std']:.9g} min={values['minimum']:.9g} "
                f"max={values['maximum']:.9g} rms={values['rms']:.9g} "
                f"finite_fraction={values['finite_fraction']:.9g}"
            )

        validation_summary, fixed_prediction = run_evaluations(0)
        validation_mse0 = float(validation_summary["global"]["mse_norm"])
        scheduler_state = update_monitor_plateau_scheduler(
            args=args,
            scheduler_state=scheduler_state,
            current_step=0,
            validation_mse=validation_mse0,
            events_path=events_path,
        )
        best_validation_mse = validation_mse0
        best_validation_step = 0
        payload = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            config=config,
            step=0,
            sample_position=0,
            best_validation_mse=best_validation_mse,
            best_validation_step=best_validation_step,
            index_plan=index_plan,
            train_history=train_history,
            evaluation_history=evaluation_history,
            diagnostic_history=diagnostic_history,
            scheduler_state=scheduler_state,
            reload_reference=fixed_prediction,
            reload_reference_validation_index=reload_reference_validation_index,
            reload_reference_validation_date=reload_reference_validation_date,
        )
        best_bytes = atomic_save_checkpoint_with_hash(
            best_path,
            payload,
            manifest_path=checkpoint_manifest_path,
            kind="best_val_mse",
            step=0,
        )
        latest_bytes = atomic_save_checkpoint_with_hash(
            latest_path,
            payload,
            manifest_path=checkpoint_manifest_path,
            kind="latest",
            step=0,
        )
        latest_checkpoint_step = 0
        log(
            f"CKPT step=0 kind=best_val_mse metric={best_validation_mse:.12g} "
            f"path={best_path} bytes={best_bytes} atomic_replace=1"
        )
        log(
            f"CKPT step=0 kind=latest path={latest_path} bytes={latest_bytes} "
            "atomic_replace=1"
        )

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    run_start = time.perf_counter()
    sequence_remaining = index_plan["training_sequence"][sample_position:]
    training_loader = make_loader(
        train_dataset,
        sequence_remaining,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    iterator = iter(training_loader)
    optimizer.zero_grad(set_to_none=True)

    while step < args.stop_updates and not STOP_REQUESTED:
        next_step = step + 1
        current_learning_rate = learning_rate_for_step(args, next_step, scheduler_state)
        set_optimizer_learning_rate(optimizer, current_learning_rate)
        if args.lr_schedule == "cosine":


            scheduler_state["current_learning_rate"] = float(current_learning_rate)
        update_wall_start = time.perf_counter()
        compute_start = torch.cuda.Event(enable_timing=True)
        compute_end = torch.cuda.Event(enable_timing=True)
        compute_start.record()

        training_accumulator = TensorAccumulator()
        activation_storage: dict[str, dict[str, float]] = {}
        hook_handles: list[Any] = []
        last_prediction: torch.Tensor | None = None
        last_target: torch.Tensor | None = None
        last_baseline: torch.Tensor | None = None

        for micro in range(1, args.accumulation_steps + 1):
            try:
                cpu_inputs, cpu_targets = next(iterator)
            except StopIteration as error:
                raise RuntimeError("Training sequence ended before max updates") from error
            inputs = cpu_inputs.to(device, non_blocking=True)
            target = cpu_targets.to(device, non_blocking=True)

            if next_step in args.diagnostic_steps and micro == args.accumulation_steps:
                activation_storage, hook_handles = register_stage_hooks(model)

            prediction = model(inputs)
            baseline = model.baseline_in_target_units(inputs)
            correction = prediction - baseline
            required = target - baseline
            training_accumulator.update(
                prediction=prediction,
                target=target,
                baseline=baseline,
                correction=correction,
                required=required,
                latitude_weights=latitude_weights,
                mask=torch.ones_like(prediction, dtype=torch.bool),
            )
            error2 = (prediction.float() - target.float()).square()
            differentiable_loss = (
                (error2 * latitude_weights).sum()
                / (
                    latitude_weights.sum()
                    * error2.shape[0]
                    * error2.shape[1]
                    * error2.shape[3]
                )
            )
            if not bool(torch.isfinite(differentiable_loss)):
                raise ValueError("Training loss became NaN or Inf")
            (differentiable_loss / args.accumulation_steps).backward()
            sample_position += 1
            last_prediction = prediction
            last_target = target
            last_baseline = baseline

            if next_step <= 3:
                micro_loss = float(differentiable_loss.detach().cpu().item())
                log(
                    "MICRO "
                    f"step={next_step} micro={micro} "
                    f"raw_loss={micro_loss:.12g} "
                    f"divided_loss={micro_loss / args.accumulation_steps:.12g}"
                )

            for handle in hook_handles:
                handle.remove()
            hook_handles = []

        gradient_norm = global_gradient_norm(model)
        stage_gradients = gradient_norms_by_stage(model)
        head_gradient = stage_gradients.get("head", float("nan"))
        bottleneck_gradient = stage_gradients.get("bottleneck", float("nan"))
        encoder0_gradient = stage_gradients.get("encoder_0", float("nan"))
        decoder_last_key = f"decoder_{args.depth - 1}"
        decoder_last_gradient = stage_gradients.get(decoder_last_key, float("nan"))

        if next_step in args.diagnostic_steps:
            parameter_gradients = (
                gradient_norms_by_parameter(model)
            )

            for stage, values in activation_storage.items():
                record = {"kind": "ACT", "step": next_step, "stage": stage, **values}
                diagnostic_history.append(record)
                append_jsonl(events_path, record)
                log(
                    "DIAG_ACT "
                    f"step={next_step} stage={stage} mean={values['mean']:.9g} "
                    f"std={values['std']:.9g} min={values['minimum']:.9g} "
                    f"max={values['maximum']:.9g} rms={values['rms']:.9g} "
                    f"finite_fraction={values['finite_fraction']:.9g}"
                )
            for stage, value in sorted(
                stage_gradients.items()
            ):
                record = {
                    "kind": "GRAD",
                    "step": next_step,
                    "stage": stage,
                    "norm": value,
                }
                diagnostic_history.append(record)
                append_jsonl(events_path, record)

                log(
                    f"DIAG_GRAD step={next_step} "
                    f"stage={stage} norm={value:.12g}"
                )

            for name, value in sorted(
                parameter_gradients.items()
            ):
                record = {
                    "kind": "PARAM_GRAD",
                    "step": next_step,
                    "parameter": name,
                    "norm": value,
                }
                diagnostic_history.append(record)
                append_jsonl(events_path, record)

                log(
                    f"DIAG_PARAM_GRAD "
                    f"step={next_step} "
                    f"parameter={name} "
                    f"norm={value:.12g}"
                )


        post_clip_gradient_norm = gradient_norm
        clip_applied = 0
        if args.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.gradient_clip_norm,
            )
            if gradient_norm > args.gradient_clip_norm:
                clip_applied = 1
                post_clip_gradient_norm = global_gradient_norm(model)

        parameter_before_update = (
            snapshot_parameter_values(model)
            if next_step in args.diagnostic_steps
            else {}
        )
        optimizer.step()
        if parameter_before_update:
            parameter_updates = parameter_update_statistics(
                model,
                parameter_before_update,
                step=next_step,
            )
            parameter_health = parameter_health_statistics(
                model,
                step=next_step,
            )
            for record in parameter_updates:
                diagnostic_history.append(record)
                append_jsonl(events_path, record)
                log(
                    "DIAG_PARAM_UPDATE "
                    f"step={next_step} "
                    f"stage={record['stage']} "
                    f"kind={record['parameter_kind']} "
                    f"count={record['parameter_count']} "
                    f"rms_before={record['parameter_rms_before']:.12g} "
                    f"delta_rms={record['parameter_delta_rms']:.12g} "
                    f"relative_update_percent="
                    f"{record['relative_update_percent']:.12g} "
                    f"bias_na={record['bias_relative_update_percent_is_na']}"
                )
            for record in parameter_health:
                diagnostic_history.append(record)
                append_jsonl(events_path, record)
                log(
                    "DIAG_PARAM_HEALTH "
                    f"step={next_step} "
                    f"stage={record['stage']} "
                    f"kind={record['parameter_kind']} "
                    f"count={record['parameter_count']} "
                    f"rms={record['parameter_rms']:.12g} "
                    f"mean_abs={record['parameter_mean_abs']:.12g} "
                    f"median_abs={record['parameter_median_abs']:.12g} "
                    f"p05_abs={record['parameter_p05_abs']:.12g} "
                    f"p95_abs={record['parameter_p95_abs']:.12g} "
                    f"p99_abs={record['parameter_p99_abs']:.12g}"
                )
        optimizer.zero_grad(set_to_none=True)
        compute_end.record()
        compute_end.synchronize()
        compute_seconds = compute_start.elapsed_time(compute_end) / 1000.0
        step = next_step

        if last_prediction is None or last_target is None or last_baseline is None:
            raise RuntimeError("No training microsteps were executed")

        combined_metrics = training_accumulator.finalize(target_std=target_std)

        update_seconds = time.perf_counter() - update_wall_start
        memory_allocated = torch.cuda.memory_allocated(device) / (1024**3)
        memory_reserved = torch.cuda.memory_reserved(device) / (1024**3)
        samples_seen = sample_position
        activation_encoder0 = activation_storage.get("encoder_0", {}).get("std", float("nan"))
        activation_bottleneck = activation_storage.get("bottleneck", {}).get("std", float("nan"))
        activation_decoder_last = activation_storage.get(
            f"decoder_{args.depth - 1}", {}
        ).get("std", float("nan"))

        step_record = {
            "kind": "STEP",
            "step": step,
            "epoch": ((step - 1) // steps_per_epoch) + 1,
            "split": "train_batch",
            "sample_position": sample_position,
            "samples_seen": samples_seen,
            "mse_norm": combined_metrics["mse_norm"],
            "mse_K2": combined_metrics.get("mse_K2", ""),
            "rmse_K": combined_metrics["rmse_K"],
            "mae_norm": combined_metrics.get("mae_norm", ""),
            "mae_K": combined_metrics.get("mae_K", ""),
            "baseline_mse_norm": combined_metrics["baseline_mse_norm"],
            "skill": combined_metrics["skill"],
            "corr_corr": combined_metrics["corr_corr"],
            "correction_mean": combined_metrics["correction_mean"],
            "correction_std": combined_metrics["correction_std"],
            "correction_rms": combined_metrics["correction_rms"],
            "correction_absmax": combined_metrics["correction_absmax"],
            "required_rms": combined_metrics["required_rms"],
            "gradient_norm": gradient_norm,
            "post_clip_gradient_norm": post_clip_gradient_norm,
            "clip_applied": clip_applied,
            "head_gradient_norm": head_gradient,
            "bottleneck_gradient_norm": bottleneck_gradient,
            "encoder0_gradient_norm": encoder0_gradient,
            "decoder_last_gradient_norm": decoder_last_gradient,
            "encoder0_activation_std": activation_encoder0,
            "bottleneck_activation_std": activation_bottleneck,
            "decoder_last_activation_std": activation_decoder_last,
            "learning_rate": current_learning_rate,
            "weight_decay": args.weight_decay,
            "update_seconds": update_seconds,
            "compute_seconds": compute_seconds,
            "samples_per_second": args.accumulation_steps / update_seconds,
            "memory_allocated_GiB": memory_allocated,
            "memory_reserved_GiB": memory_reserved,
            "finite": 1,
        }
        train_history.append(step_record)
        append_jsonl(events_path, step_record)
        append_metric_csv(metrics_path, step_record)
        append_standard_metric_csv(
            train_metrics_path,
            step_record,
            TRAIN_METRIC_CSV_FIELDS,
        )
        log(
            "STEP "
            f"step={step} samples_seen={samples_seen} "
            f"train_mse_norm={step_record['mse_norm']:.12g} "
            f"train_rmse_K={step_record['rmse_K']:.9g} "
            f"skill_batch={step_record['skill']:.9g} "
            f"corr_corr_batch={step_record['corr_corr']:.9g} "
            f"corr_mean={step_record['correction_mean']:.9g} "
            f"corr_std={step_record['correction_std']:.9g} "
            f"corr_absmax={step_record['correction_absmax']:.9g} "
            f"grad_norm={gradient_norm:.9g} "
            f"post_clip_grad_norm={post_clip_gradient_norm:.9g} "
            f"clip_applied={clip_applied} "
            f"head_grad_norm={head_gradient:.9g} "
            f"bottleneck_grad_norm={bottleneck_gradient:.9g} "
            f"enc0_act_std={activation_encoder0:.9g} "
            f"bottleneck_act_std={activation_bottleneck:.9g} "
            f"dec_last_act_std={activation_decoder_last:.9g} "
            f"lr={current_learning_rate:.9g} wd={args.weight_decay:.9g} "
            f"update_s={update_seconds:.6f} compute_s={compute_seconds:.6f} "
            f"samples_s={step_record['samples_per_second']:.6f} "
            f"mem_alloc_GiB={memory_allocated:.6f} "
            f"mem_reserved_GiB={memory_reserved:.6f} finite=1"
        )

        fixed_prediction_for_checkpoint: np.ndarray | None = None
        if step in args.evaluation_steps:
            validation_summary, fixed_prediction_for_checkpoint = run_evaluations(step)
            validation_mse = float(validation_summary["global"]["mse_norm"])
            scheduler_state = update_monitor_plateau_scheduler(
                args=args,
                scheduler_state=scheduler_state,
                current_step=step,
                validation_mse=validation_mse,
                events_path=events_path,
            )
            if validation_mse < best_validation_mse:
                previous_best = best_validation_mse
                best_validation_mse = validation_mse
                best_validation_step = step
                best_payload = checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    step=step,
                    sample_position=sample_position,
                    best_validation_mse=best_validation_mse,
                    best_validation_step=best_validation_step,
                    index_plan=index_plan,
                    train_history=train_history,
                    evaluation_history=evaluation_history,
                    diagnostic_history=diagnostic_history,
                    scheduler_state=scheduler_state,
                    reload_reference=fixed_prediction_for_checkpoint,
                    reload_reference_validation_index=reload_reference_validation_index,
                    reload_reference_validation_date=reload_reference_validation_date,
                )
                best_bytes = atomic_save_checkpoint_with_hash(
                    best_path,
                    best_payload,
                    manifest_path=checkpoint_manifest_path,
                    kind="best_val_mse",
                    step=step,
                )
                log(
                    f"CKPT step={step} kind=best_val_mse metric={validation_mse:.12g} "
                    f"previous_best={previous_best:.12g} path={best_path} "
                    f"bytes={best_bytes} atomic_replace=1"
                )
                if args.checkpoint_retention == "lean":
                    best_sidecar = best_path.parent / (best_path.name + ".sha256")
                    write_json(
                        output_dir / "best_checkpoint_metadata.json",
                        {
                            "kind": "BEST_CHECKPOINT_METADATA",
                            "selected_monitor_step": int(step),
                            "selected_monitor_mse_norm": float(validation_mse),
                            "selected_monitor_rmse_K": float(
                                validation_summary["global"]["rmse_K"]
                            ),
                            "checkpoint_file_name": best_path.name,
                            "checkpoint_sha256": best_sidecar.read_text().split()[0],
                            "checkpoint_size_bytes": int(best_bytes),
                            "config_hash": config["config_hash"],
                            "architecture": model.architecture_dict(),
                            "pole_aware_boundary_scheme": (
                                POLE_AWARE_BOUNDARY_SCHEME
                                if args.lat_padding == "pole_aware"
                                else None
                            ),
                            "normalization_sha256": norm_stats_provenance["sha256"],
                            "index_hashes": index_hashes,
                            "code_snapshot_tree_sha256": code_snapshot["tree_sha256"],
                            "recorded_utc": datetime.now(timezone.utc).isoformat(),
                        },
                    )

            latest_payload = checkpoint_payload(
                model=model,
                optimizer=optimizer,
                config=config,
                step=step,
                sample_position=sample_position,
                best_validation_mse=best_validation_mse,
                best_validation_step=best_validation_step,
                index_plan=index_plan,
                train_history=train_history,
                evaluation_history=evaluation_history,
                diagnostic_history=diagnostic_history,
                scheduler_state=scheduler_state,
                reload_reference=fixed_prediction_for_checkpoint,
                reload_reference_validation_index=reload_reference_validation_index,
                reload_reference_validation_date=reload_reference_validation_date,
            )
            latest_bytes = atomic_save_checkpoint_with_hash(
                latest_path,
                latest_payload,
                manifest_path=checkpoint_manifest_path,
                kind="latest",
                step=step,
            )
            latest_checkpoint_step = step
            log(
                f"CKPT step={step} kind=latest path={latest_path} "
                f"bytes={latest_bytes} atomic_replace=1"
            )
            if (
                args.checkpoint_retention != "lean"
                and step > 0
                and step % steps_per_epoch == 0
            ):
                epoch_number = step // steps_per_epoch
                epoch_path = checkpoints_dir / f"epoch_{epoch_number:03d}.pt"
                epoch_bytes = atomic_save_checkpoint_with_hash(
                    epoch_path,
                    latest_payload,
                    manifest_path=checkpoint_manifest_path,
                    kind="epoch",
                    step=step,
                )
                log(
                    f"CKPT step={step} kind=epoch epoch={epoch_number} "
                    f"path={epoch_path} bytes={epoch_bytes} atomic_replace=1"
                )

        if STOP_REQUESTED:
            if fixed_prediction_for_checkpoint is None:
                fixed_prediction_for_checkpoint = fixed_validation_reload_reference(
                    model=model,
                    dataset=validation_dataset,
                    validation_index=reload_reference_validation_index,
                    device=device,
                )
            latest_payload = checkpoint_payload(
                model=model,
                optimizer=optimizer,
                config=config,
                step=step,
                sample_position=sample_position,
                best_validation_mse=best_validation_mse,
                best_validation_step=best_validation_step,
                index_plan=index_plan,
                train_history=train_history,
                evaluation_history=evaluation_history,
                diagnostic_history=diagnostic_history,
                scheduler_state=scheduler_state,
                reload_reference=fixed_prediction_for_checkpoint,
                reload_reference_validation_index=reload_reference_validation_index,
                reload_reference_validation_date=reload_reference_validation_date,
            )
            latest_bytes = atomic_save_checkpoint_with_hash(
                latest_path,
                latest_payload,
                manifest_path=checkpoint_manifest_path,
                kind="latest_signal",
                step=step,
            )
            latest_checkpoint_step = step
            log(
                f"CKPT step={step} kind=latest_signal path={latest_path} "
                f"bytes={latest_bytes} atomic_replace=1"
            )
            break

    if step < args.max_updates and not STOP_REQUESTED and latest_checkpoint_step != step:
        fixed_prediction_for_checkpoint = fixed_validation_reload_reference(
            model=model,
            dataset=validation_dataset,
            validation_index=reload_reference_validation_index,
            device=device,
        )
        latest_payload = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            config=config,
            step=step,
            sample_position=sample_position,
            best_validation_mse=best_validation_mse,
            best_validation_step=best_validation_step,
            index_plan=index_plan,
            train_history=train_history,
            evaluation_history=evaluation_history,
            diagnostic_history=diagnostic_history,
            scheduler_state=scheduler_state,
            reload_reference=fixed_prediction_for_checkpoint,
            reload_reference_validation_index=reload_reference_validation_index,
            reload_reference_validation_date=reload_reference_validation_date,
        )
        latest_bytes = atomic_save_checkpoint_with_hash(
            latest_path,
            latest_payload,
            manifest_path=checkpoint_manifest_path,
            kind="latest_stop_updates",
            step=step,
        )
        latest_checkpoint_step = step
        log(
            f"CKPT step={step} kind=latest_stop_updates path={latest_path} "
            f"bytes={latest_bytes} atomic_replace=1"
        )

    torch.cuda.synchronize(device)
    total_wall_seconds = time.perf_counter() - run_start
    peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)

    reload_result = verify_checkpoint_reload(
        checkpoint_path=latest_path,
        model_arguments=model_arguments,
        dataset=validation_dataset,
        validation_index=reload_reference_validation_index,
        device=device,
    )

    validation_records = [
        item for item in evaluation_history if item["split"] == "validation"
    ]
    train_monitor_records = [
        item for item in evaluation_history if item["split"] == "train_monitor"
    ]
    final_validation = validation_records[-1] if validation_records else None
    final_train_monitor = (
        train_monitor_records[-1] if train_monitor_records else None
    )
    mean_update_seconds = float(
        np.mean([item["update_seconds"] for item in train_history])
    ) if train_history else float("nan")
    nonfinite_events = 0


    clip_events = int(
        sum(int(item.get("clip_applied", 0) or 0) for item in train_history)
    )
    clip_fraction = (
        clip_events / len(train_history) if train_history else float("nan")
    )
    final_step_record = train_history[-1] if train_history else {}
    parameter_update_available = any(
        item.get("kind") == "PARAM_UPDATE" for item in diagnostic_history
    )
    parameter_health_available = any(
        item.get("kind") == "PARAM_HEALTH" for item in diagnostic_history
    )

    if STOP_REQUESTED:
        chunk_status = "stopped_by_signal"
    elif step >= args.max_updates:
        chunk_status = "completed_max_updates"
    elif step >= args.stop_updates:
        chunk_status = "completed_stop_updates"
    else:
        chunk_status = "stopped_early"

    if args.checkpoint_retention == "lean" and chunk_status in (
        "completed_stop_updates",
        "completed_max_updates",
    ):


        promotion = promote_last_completed_chunk(
            latest_path=latest_path,
            recovery_path=checkpoints_dir / "last_completed_chunk.pt",
            manifest_path=checkpoint_manifest_path,
            step=step,
        )
        log(
            f"CKPT step={step} kind=last_completed_chunk "
            f"method={promotion['promotion_method']} "
            f"seconds={promotion['promotion_seconds']:.3f} "
            f"bytes={promotion['size_bytes']} "
            f"sha256={promotion['sha256']}"
        )

    final = {
        "chunk_status": chunk_status,
        "kind": "FINAL",
        "run_name": run_name,
        "norm": args.norm,
        "updates_completed": step,
        "target_updates": args.max_updates,
        "stop_updates": args.stop_updates,
        "samples_consumed": sample_position,
        "nonfinite_events": nonfinite_events,
        "best_validation_step": best_validation_step,
        "best_validation_mse_norm": best_validation_mse,
        "final_validation_mse_norm": (
            final_validation["global"]["mse_norm"] if final_validation else float("nan")
        ),
        "final_validation_mae_K": (
            final_validation["global"]["mae_K"] if final_validation else float("nan")
        ),
        "final_validation_bias_K": (
            final_validation["global"]["bias_K"] if final_validation else float("nan")
        ),
        "final_validation_rmse_K": (
            final_validation["global"]["rmse_K"] if final_validation else float("nan")
        ),
        "final_validation_skill": (
            final_validation["global"]["skill"] if final_validation else float("nan")
        ),
        "final_validation_corr_corr": (
            final_validation["global"]["corr_corr"] if final_validation else float("nan")
        ),
        "baseline_validation_rmse_K": (
            final_validation["global"]["baseline_rmse_K"] if final_validation else float("nan")
        ),
        "final_train_monitor_mse_norm": (
            final_train_monitor["global"]["mse_norm"] if final_train_monitor else float("nan")
        ),
        "final_train_monitor_mae_K": (
            final_train_monitor["global"]["mae_K"] if final_train_monitor else float("nan")
        ),
        "final_train_monitor_bias_K": (
            final_train_monitor["global"]["bias_K"] if final_train_monitor else float("nan")
        ),
        "final_train_monitor_rmse_K": (
            final_train_monitor["global"]["rmse_K"] if final_train_monitor else float("nan")
        ),
        "final_train_monitor_skill": (
            final_train_monitor["global"]["skill"] if final_train_monitor else float("nan")
        ),
        "final_train_monitor_corr_corr": (
            final_train_monitor["global"]["corr_corr"] if final_train_monitor else float("nan")
        ),
        "baseline_train_monitor_rmse_K": (
            final_train_monitor["global"]["baseline_rmse_K"] if final_train_monitor else float("nan")
        ),
        "final_learning_rate": float(
            final_step_record.get("learning_rate", float("nan"))
        ),
        "weight_decay": args.weight_decay,
        "gradient_clip_norm": args.gradient_clip_norm,
        "clip_events": clip_events,
        "clip_fraction": clip_fraction,
        "scheduler_state": dict(scheduler_state),
        **scheduler_summary_fields(args, scheduler_state),
        "parameter_update_diagnostics_available": int(parameter_update_available),
        "parameter_health_diagnostics_available": int(parameter_health_available),
        "peak_memory_allocated_GiB": peak_allocated,
        "peak_memory_reserved_GiB": peak_reserved,
        "mean_update_seconds": mean_update_seconds,
        "total_wall_seconds": total_wall_seconds,
        "metrics_csv_path": str(metrics_path),
        "train_metrics_csv_path": str(train_metrics_path),
        "validation_metrics_csv_path": str(validation_metrics_path),
        "events_jsonl_path": str(events_path),
        "plots_dir": str(plots_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "latest_checkpoint_path": str(latest_path),
        "best_checkpoint_path": str(best_path),
        "final_summary_path": str(output_dir / "final_summary.json"),
        **reload_result,
    }
    if args.lr_schedule == "cosine":


        json.dumps(final, sort_keys=True, allow_nan=False)
        append_jsonl(events_path, final)
        write_json_strict(output_dir / "final_summary.json", final)
    else:
        append_jsonl(events_path, final)
        write_json(output_dir / "final_summary.json", final)
    append_jsonl(
        chunk_log_path,
        {
            "kind": "CHUNK_END",
            "ended_utc": datetime.now(timezone.utc).isoformat(),
            "end_step": int(step),
            "end_sample_position": int(sample_position),
            "status": chunk_status,
            "best_validation_step": int(best_validation_step),
            "latest_checkpoint_step": (
                int(latest_checkpoint_step)
                if latest_checkpoint_step is not None
                else None
            ),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        },
    )

    log("=" * 80)
    log(
        "FINAL "
        f"run_name={run_name} norm={args.norm} updates_completed={step} "
        f"target_updates={args.max_updates} stop_updates={args.stop_updates} "
        f"samples_consumed={sample_position} nonfinite_events={nonfinite_events} "
        f"gradient_clip_norm={args.gradient_clip_norm} "
        f"clip_events={clip_events} clip_fraction={clip_fraction:.6g} "
        f"best_val_step={best_validation_step} "
        f"best_val_mse_norm={best_validation_mse:.12g} "
        f"final_val_mse_norm={final['final_validation_mse_norm']:.12g} "
        f"final_val_rmse_K={final['final_validation_rmse_K']:.9g} "
        f"final_val_mae_K={final['final_validation_mae_K']:.9g} "
        f"final_val_bias_K={final['final_validation_bias_K']:.9g} "
        f"baseline_val_rmse_K={final['baseline_validation_rmse_K']:.9g} "
        f"final_val_skill={final['final_validation_skill']:.9g} "
        f"final_val_corr_corr={final['final_validation_corr_corr']:.9g} "
        f"final_train_monitor_rmse_K={final['final_train_monitor_rmse_K']:.9g} "
        f"final_learning_rate={final['final_learning_rate']:.9g} "
        f"weight_decay={args.weight_decay:.9g} "
        f"scheduler_lr_drop_count={final['scheduler_lr_drop_count']} "
        "scheduler_best_monitor_mse="
        + (
            "null"
            if final["scheduler_best_monitor_mse"] is None
            else f"{final['scheduler_best_monitor_mse']:.12g}"
        )
        + " "
        f"parameter_update_diagnostics_available="
        f"{final['parameter_update_diagnostics_available']} "
        f"parameter_health_diagnostics_available="
        f"{final['parameter_health_diagnostics_available']} "
        f"peak_mem_alloc_GiB={peak_allocated:.6f} "
        f"peak_mem_reserved_GiB={peak_reserved:.6f} "
        f"mean_update_seconds={mean_update_seconds:.6f} "
        f"total_wall_seconds={total_wall_seconds:.6f} "
        f"checkpoint_reload_pass={reload_result['checkpoint_reload_pass']} "
        f"checkpoint_reload_max_abs_difference="
        f"{reload_result['checkpoint_reload_max_abs_difference']:.9g}"
    )
    if step == args.max_updates:
        log("FINAL RESULT: GLOBAL U-NET ARCHITECTURE SCREEN TRAINING PASSED")
    else:
        log("FINAL RESULT: PILOT STOPPED EARLY AFTER WRITING LATEST CHECKPOINT")
    log("=" * 80)


if __name__ == "__main__":
    main()
