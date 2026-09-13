"""Add explicit normalized and physical-unit metrics to stage-03 CCA outputs.
Some legacy columns labelled K or K2 contain target-normalized quantities.
This script preserves those values, adds normalized aliases and converts RMS
and RMSE with the training target standard deviation, and MSE with its square.
Creates backups before updating the tables and metadata. This common scale
conversion does not change the selected CCA configuration."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import os


DATA_ROOT = os.environ.get("DATA_ROOT", "data_root")                                                     
RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "results_root")                                                  


OUT = Path(f"{RESULTS_ROOT}/cca_final_run/outputs/cca_grid")

TARGET_STD_K = 21.627892139211994

UNIT_PATCH_FULL = {
    "created_note": "Stage 03 CCA grid scores and residuals are in normalized t2m_tgt units. Converted physical-K metrics use t2m_tgt std only.",
    "input_mean_K": 277.8568272051984,
    "input_std_K": 21.581598298286263,
    "selection_unchanged_by_global_scale": True,
    "skill_scale_invariant": True,
    "source": f"{DATA_ROOT}/zarr/awi_downscaling_train.zarr/.zattrs normalization_stats.t2m_tgt",
    "target_mean_K": 277.8518781731243,
    "target_std_K": TARGET_STD_K,
}

UNIT_PATCH_STATUS = {
    "physical_K_columns_added_to_results_and_metadata": True,
    "stage03_metric_columns_with_suffix_K_are_legacy_normalized_values": True,
    "target_std_K": TARGET_STD_K,
}


CSV_APPEND = [
    "train_mse_norm2", "val_mse_norm2", "train_rmse_norm", "val_rmse_norm",
    "train_required_residual_rms_norm", "train_predicted_residual_rms_norm",
    "val_required_residual_rms_norm", "val_predicted_residual_rms_norm",
    "train_rmse_K_physical", "val_rmse_K_physical",
    "train_required_residual_rms_K_physical", "train_predicted_residual_rms_K_physical",
    "val_required_residual_rms_K_physical", "val_predicted_residual_rms_K_physical",
    "train_mse_K2_physical", "val_mse_K2_physical",
]


def derived(col: str, row: dict[str, str]) -> str:
    if col.endswith("_norm2"):
        return row[col.replace("_norm2", "_K2")]
    if col.endswith("_norm"):
        return row[col.replace("_norm", "_K")]
    base = col[: -len("_physical")]
    val = float(row[base])
    scale = TARGET_STD_K ** 2 if base.endswith("_K2") else TARGET_STD_K
    return repr(val * scale)


def patch_metric_dict(d: dict) -> None:

    for key in [k for k in list(d) if isinstance(d[k], (int, float))]:
        val = float(d[key])
        if key.endswith("_mse_K2"):
            d[key.replace("_K2", "_norm2")] = val
            d[key + "_physical"] = val * TARGET_STD_K ** 2
        elif key.endswith("_rmse_K") or key.endswith("_rms_K"):
            d[key.replace("_K", "_norm")] = val
            d[key + "_physical"] = val * TARGET_STD_K


def backup(path: Path) -> None:
    pre = path.with_name(path.name + ".pre_unit_patch")
    if pre.exists():
        raise SystemExit(f"refusing to overwrite existing backup {pre}")
    shutil.copy2(path, pre)


def main() -> int:
    results = OUT / "cca_grid_results.csv"
    with results.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if any(c in fieldnames for c in CSV_APPEND):
        raise SystemExit("results CSV already unit-patched; refusing to patch twice")
    backup(results)
    for row in rows:
        for col in CSV_APPEND:
            row[col] = derived(col, row)
    tmp = results.with_name(results.name + ".tmp_unit_patch")
    with tmp.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames + CSV_APPEND)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(results)
    print(f"patched {results} rows={len(rows)} added_cols={len(CSV_APPEND)}")

    summary_p = OUT / "cca_grid_summary.json"
    summary = json.loads(summary_p.read_text())
    if "unit_patch" in summary:
        raise SystemExit("summary already unit-patched")
    backup(summary_p)
    patch_metric_dict(summary["selected"])
    for rec in summary.get("best_val_rmse_K_top5", []):
        patch_metric_dict(rec)
    summary["unit_patch"] = UNIT_PATCH_FULL
    summary_p.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(f"patched {summary_p}")

    meta_p = OUT / "selected_cca_metadata.json"
    meta = json.loads(meta_p.read_text())
    if "unit_patch" in meta:
        raise SystemExit("metadata already unit-patched")
    backup(meta_p)
    patch_metric_dict(meta["selected"])
    meta["unit_patch"] = UNIT_PATCH_FULL
    meta_p.write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(f"patched {meta_p}")

    status_p = OUT / "stage03_status.json"
    status = json.loads(status_p.read_text())
    if "unit_patch" in status:
        raise SystemExit("status already unit-patched")
    backup(status_p)
    patch_metric_dict(status["selected"])
    status["unit_patch"] = UNIT_PATCH_STATUS
    status_p.write_text(json.dumps(status, indent=1, sort_keys=True) + "\n")
    print(f"patched {status_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
