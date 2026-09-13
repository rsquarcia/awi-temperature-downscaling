"""Evaluate the frozen U-Net under uniform temperature displacements.
Shifts the temperature input channel while keeping the other four predictors
fixed, runs the selected checkpoint and compares the predicted correction with
the unchanged target-minus-bilinear residual. Saves daily weighted errors in
shards for the shared reducer. This script runs inference but never optimizes
the model; it uses the shipped U-Net with the external checkpoint and test
data. Imported module paths and historical training-source identity are checked
separately; no private source snapshot is required."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from uniform_shift_defs import (C, S, DELTAS, I_ZERO, IM_K, IS_K, TM_K, TS_K, N_DAYS,
                        CKPT, CKPT_SHA, CKPT_STEP, PUBLIC_UNET_ROOT,
                        TRAINING_CODE_TREE_SHA, NORM_STATS, NORM_STATS_SHA,
                        shard_days)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    args = ap.parse_args()
    import torch
    import zarr

    root = Path(args.root)
    days = shard_days(args.nshards, args.shard)
    log(f"shard {args.shard}/{args.nshards}: days {days[0]}..{days[-1]} "
        f"({days.size} days) x {DELTAS.size} displacements")


    gates = {}
    sha = C.sha256_file(CKPT)
    gates["checkpoint_sha256"] = sha
    gates["checkpoint_sha256_matches"] = (sha == CKPT_SHA)
    if not gates["checkpoint_sha256_matches"]:
        raise SystemExit(f"CHECKPOINT GATE FAILED: {sha}")
    sys.path.insert(0, str(PUBLIC_UNET_ROOT))
    import importlib
    stage = importlib.import_module("training.global_unet_second_stage")
    unet = importlib.import_module("training.global_unet")
    dataset = importlib.import_module("dataloader.zarr_dataset")
    gates["public_modules"] = {}
    for module in (stage, unet, dataset):
        observed = Path(module.__file__).resolve()
        expected = PUBLIC_UNET_ROOT / (module.__name__.replace(".", "/") + ".py")
        if observed != expected:
            raise SystemExit(f"MODULE PATH GATE FAILED: {observed} != {expected}")
        gates["public_modules"][module.__name__] = str(observed)
    stage.configure_fp32()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    gates["checkpoint_step"] = int(ck.get("step", -1))
    gates["checkpoint_step_matches"] = (gates["checkpoint_step"] == CKPT_STEP)
    cfg = ck["config"]
    gates["historical_training_code_tree_sha256"] = (
        cfg["provenance"]["code_snapshot"]["tree_sha256"])
    if gates["historical_training_code_tree_sha256"] != TRAINING_CODE_TREE_SHA:
        raise SystemExit("TRAINING SOURCE IDENTITY GATE FAILED")
    gates["normalization_sha256"] = C.sha256_file(NORM_STATS)
    if gates["normalization_sha256"] != NORM_STATS_SHA:
        raise SystemExit("NORMALIZATION FILE IDENTITY GATE FAILED")
    margs = cfg.get("model_arguments") or ck["model_arguments"]
    model = unet.GlobalUNet(**margs)
    gates["architecture_matches_checkpoint"] = (
        ck.get("architecture") == model.architecture_dict())
    model.load_state_dict(ck["model_state"], strict=True)
    model.eval()
    gates["normalization_matches_store"] = (
        stage.load_normalization_statistics(NORM_STATS) == cfg["normalization"])
    for k in ("checkpoint_sha256_matches", "checkpoint_step_matches",
              "architecture_matches_checkpoint", "normalization_matches_store"):
        log(f"GATE {k}: {gates[k]}")
    if not all(gates[k] for k in ("checkpoint_sha256_matches",
                                  "checkpoint_step_matches",
                                  "architecture_matches_checkpoint",
                                  "normalization_matches_store")):
        raise SystemExit("IDENTITY GATES FAILED before the displacement run")
    dev = args.device
    model = model.to(dev)
    for p in model.parameters():
        p.requires_grad_(False)


    W2D = torch.from_numpy(S.W2D.astype(np.float64)).to(dev)
    W_SUM = float(S.W_SUM)
    g = zarr.open_group(str(C.PD_ZARR), mode="r")
    data = dataset.AWIDownscalingZarrDataset(C.PD_ZARR, include_static_inputs=True)
    dates = np.asarray(g["dates"][:]).astype("U10")
    if dates.size != N_DAYS:
        raise SystemExit(f"expected {N_DAYS} PD-test days, found {dates.size}")
    gates["n_input_channels"] = int(data.input_shape[0])
    log(f"{gates['n_input_channels']} input channels; only channel 0 "
        f"(normalized bilinear T2M) is displaced")

    shifts = torch.tensor([float(c) / IS_K for c in DELTAS],
                          dtype=torch.float32, device=dev)
    EVAL_ORDER = [I_ZERO] + [j for j in range(DELTAS.size) if j != I_ZERO]

    day_mse = np.zeros((days.size, DELTAS.size), dtype=np.float64)
    pred_change_sq = np.zeros((days.size, DELTAS.size), dtype=np.float64)
    t0 = time.time()
    for i, dix in enumerate(days):
        cpu_inputs, cpu_target = data[dix]
        tgt = cpu_target[0].numpy()
        B = cpu_inputs[0].numpy().astype(np.float64) * IS_K + IM_K
        T = tgt.astype(np.float64) * TS_K + TM_K
        r_true = torch.from_numpy(T - B).to(dev)
        t = cpu_inputs[None].to(dev)
        ch0 = t[0, 0].clone()
        p_zero = None


        for j in EVAL_ORDER:
            t[0, 0] = ch0 + shifts[j]
            with torch.inference_mode():
                p = ((model(t) - model.baseline_in_target_units(t))[0, 0]
                     .to(torch.float64) * TS_K)
            if j == I_ZERO:
                p_zero = p.clone()
            e = p - r_true
            day_mse[i, j] = (e * e * W2D).sum().item() / W_SUM
            pred_change_sq[i, j] = ((p - p_zero) ** 2 * W2D).sum().item() / W_SUM
        del t, ch0, r_true, p_zero
        if (i + 1) % 5 == 0 or i == days.size - 1:
            el = time.time() - t0
            log(f"  {i+1}/{days.size} days ({el:.0f} s, "
                f"{el/(i+1):.2f} s/day, eta {el/(i+1)*(days.size-i-1):.0f} s)")

    np.savez(root / "partials" / f"unet_day_mse_{args.shard:02d}.npz",
             deltas=DELTAS, days=days, day_mse=day_mse,
             pred_change_sq=pred_change_sq, dates=dates[days])
    (root / "outputs" / f"stage_b_gates_{args.shard:02d}.json").write_text(
        json.dumps(gates, indent=2, default=lambda o: float(o)) + "\n")
    log(f"shard {args.shard} complete; RMSE(delta=0) on this shard = "
        f"{np.sqrt(day_mse[:, I_ZERO].mean()):.8f} K")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
