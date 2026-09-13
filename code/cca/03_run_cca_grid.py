"""Fit and select EOF-CCA mappings in the retained training score space.
Evaluates configured predictor dimensions, target dimensions, canonical ranks
and regularization strengths on validation data, selecting the minimum
area-weighted full-field validation MSE. Test data are not used for selection.
Writes the grid results and selected mapping for subsequent evaluation.
Legacy metric columns need the normalized-to-physical conversion in stage 03b.
Standalone development smoke modes are omitted; production checks remain."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import os
import re
import struct
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cca_utils import (
    IssueLog,
    _np,
    area_weights_from_lat,
    atomic_savez,
    done_matches,
    done_path,
    dtype_is_float32,
    fmt_cell,
    grid_lat_centers,
    is_relative_to,
    load_json,
    npz_member_npy_info,
    output_state,
    parse_npy_header,
    read_npy_header,
    resolve,
    sha256_file,
    unique_tmp,
    utc_now,
    write_csv_atomic,
    write_done,
    write_json_atomic,
    write_text_atomic,
    zip_member_data_offset,
)


RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "results_root")                                                  


HOME_RUN_ROOT = Path(f"{RESULTS_ROOT}/cca_final_run")
ALLOWED_HOME_OUTPUT_ROOT = HOME_RUN_ROOT / "outputs"
DEFAULT_HEAVY_OUTPUT_ROOT = Path(f"{RESULTS_ROOT}/cca_final_run/outputs_heavy")
ALLOWED_SCRATCH_PREFIX = Path(f"{RESULTS_ROOT}")
FORBIDDEN_HOME_PREFIX = Path("/home")
FORBIDDEN_WORK_PREFIX = Path("/work")
DEFAULT_CONFIG = HOME_RUN_ROOT / "config" / "cca_grid_config.json"
STAGE_NAME = "03_run_cca_grid"
SEAL_TOKENS = ("awi_downscaling_test", "sealed", "holdout")
REAL_RUN_ENV = "CCA_STAGE03_ENABLE_REAL_RUN"
HEAVY_PHASES = ("compute-constants", "run-grid")
SMALL_MEMBER_MAX_BYTES = 64 << 20
SCORE_KEYS = ("train_x", "val_x", "train_y", "val_y")

RESULT_COLUMNS = [
    "Kx", "Ky", "r", "ridge_alpha",
    "train_mse_K2", "train_rmse_K", "train_skill_vs_bilinear",
    "val_mse_K2", "val_rmse_K", "val_skill_vs_bilinear",
    "train_correction_corr", "val_correction_corr",
    "train_required_residual_rms_K", "train_predicted_residual_rms_K", "train_amplitude_ratio",
    "val_required_residual_rms_K", "val_predicted_residual_rms_K", "val_amplitude_ratio",
    "first_canonical_corr", "mean_canonical_corr_retained", "min_canonical_corr_retained",
    "fit_status", "notes",
]


def guard_home_output_root(raw: str | Path, log: IssueLog) -> Path:
    out = resolve(Path(raw))
    allowed = resolve(ALLOWED_HOME_OUTPUT_ROOT)
    if out != allowed and not is_relative_to(out, allowed):
        log.fail(f"home output root {out} is outside allowed tree {allowed}")
    else:
        log.pass_(f"home output root confined to {allowed}")
    return out


def guard_heavy_output_root(raw: str | Path, log: IssueLog) -> Path:
    out = resolve(Path(raw))
    scratch = resolve(ALLOWED_SCRATCH_PREFIX)
    if is_relative_to(out, resolve(FORBIDDEN_HOME_PREFIX)):
        log.fail(f"heavy output root must not be under /home: {out}")
    elif is_relative_to(out, resolve(FORBIDDEN_WORK_PREFIX)):
        log.fail(f"heavy output root must not be under /work: {out}")
    elif out != scratch and not is_relative_to(out, scratch):
        log.fail(f"heavy output root {out} is outside allowed scratch tree {scratch}")
    else:
        log.pass_(f"heavy (smoke) output root confined to scratch tree {scratch}")
    return out


def suspicious_path(value: str) -> list[str]:
    lower = value.lower()
    hits = [token for token in SEAL_TOKENS if token in lower]
    parts = [part for part in re.split(r"[/_.\\-]+", lower) if part]
    if "test" in parts:
        hits.append("test")
    return sorted(set(hits))


def reject_suspicious_path(label: str, raw: str | Path, log: IssueLog) -> None:
    hits = suspicious_path(str(raw))
    if hits:
        log.fail(f"{label} contains sealed/test token(s) {hits}: {raw}")


def npz_small_member(path: Path, member: str, max_bytes: int = SMALL_MEMBER_MAX_BYTES):


    np = _np()
    info = npz_member_npy_info(path, member)
    if info["member_bytes"] > max_bytes:
        raise RuntimeError(
            f"refusing to load npz member {member} ({info['member_bytes']} bytes > cap {max_bytes}); "
            "Stage 03 never needs large members"
        )
    with zipfile.ZipFile(path) as archive:
        payload = archive.read(member)
    return np.load(io.BytesIO(payload), allow_pickle=False)


def load_manifest_split(path: Path, split: str, log: IssueLog) -> tuple[list[dict[str, str]], int]:

    if split not in ("train", "val"):
        log.fail(f"invalid split {split!r}; allowed: train/val")
        return [], 0
    if not path.is_file():
        log.fail(f"missing source manifest: {path}")
        return [], 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    other = sorted({row.get("split") for row in rows} - {"train", "val"})
    if other:
        log.fail(f"manifest has unsupported split rows: {other}")
    picked = sorted((row for row in rows if row.get("split") == split), key=lambda row: int(row["start"]))
    pos = 0
    for index, row in enumerate(picked):
        start, end = int(row["start"]), int(row["end"])
        if start != pos or end <= start:
            log.fail(f"{split} manifest rows not contiguous at row {index}: start={start} expected {pos}")
            return picked, 0
        pos = end
    return picked, pos


def build_grid_rows(grid_cfg: dict[str, Any]) -> list[dict[str, Any]]:

    rows: list[dict[str, Any]] = []
    for kx in grid_cfg["kx_values"]:
        for ky in grid_cfg["ky_values"]:
            for r in grid_cfg["r_values"]:
                if r > min(int(kx), int(ky)):
                    continue
                for alpha in grid_cfg["ridge_alpha_values"]:
                    rows.append({"Kx": int(kx), "Ky": int(ky), "r": int(r), "ridge_alpha": float(alpha)})
    rows.sort(key=lambda t: (t["Kx"], t["Ky"], t["r"], t["ridge_alpha"]))
    return rows


def sym_inv_sqrt_and_sqrt(S, ridge_alpha: float, eig_floor_rel: float):


    np = _np()
    S = np.asarray(S, dtype=np.float64)
    k = S.shape[0]
    if ridge_alpha > 0.0:
        S = S + ridge_alpha * np.eye(k)
    evals, evecs = np.linalg.eigh((S + S.T) * 0.5)
    emax = float(evals[-1])
    if not math.isfinite(emax) or emax <= 0.0:
        raise RuntimeError(f"covariance block has non-positive max eigenvalue {emax}")
    kept = evals > emax * eig_floor_rel
    n_dropped = int(k - int(kept.sum()))
    ve = evecs[:, kept]
    lam = evals[kept]
    inv_sqrt = (ve / np.sqrt(lam)) @ ve.T
    sqrt_m = (ve * np.sqrt(lam)) @ ve.T
    diag = {
        "n_dropped": n_dropped,
        "min_eig": float(evals[0]),
        "max_eig": emax,
        "min_kept_eig": float(lam.min()),
    }
    return inv_sqrt, sqrt_m, diag


def compute_score_stats(train_x64, train_y64):

    np = _np()
    n = train_x64.shape[0]
    x_mean = train_x64.mean(axis=0)
    y_mean = train_y64.mean(axis=0)
    xc = train_x64 - x_mean[None, :]
    yc = train_y64 - y_mean[None, :]
    denom = float(n - 1)
    return {
        "n": n,
        "x_mean": x_mean,
        "y_mean": y_mean,
        "sxx": (xc.T @ xc) / denom,
        "sxy": (xc.T @ yc) / denom,
        "syy": (yc.T @ yc) / denom,
    }


class CCAFitCache:


    def __init__(self, stats: dict[str, Any], eig_floor_rel: float) -> None:
        self.stats = stats
        self.eig_floor_rel = eig_floor_rel
        self._x_key: tuple[int, float] | None = None
        self._x_entry: tuple[Any, dict[str, Any]] | None = None
        self._y_entries: dict[tuple[int, float], tuple[Any, Any, dict[str, Any]]] = {}
        self._svd_key: tuple[int, int, float] | None = None
        self._svd_entry: tuple[Any, Any, Any] | None = None

    def x_whitener(self, kx: int, alpha: float) -> tuple[Any, dict[str, Any]]:
        key = (kx, alpha)
        if self._x_key != key:
            isx, _, diag = sym_inv_sqrt_and_sqrt(self.stats["sxx"][:kx, :kx], alpha, self.eig_floor_rel)
            self._x_key, self._x_entry = key, (isx, diag)
            print(f"WHITENER x Kx={kx} alpha={alpha} min_eig={diag['min_eig']:.6e} "
                  f"max_eig={diag['max_eig']:.6e} min_kept_eig={diag['min_kept_eig']:.6e} dropped={diag['n_dropped']}")
        return self._x_entry

    def y_whitener(self, ky: int, alpha: float) -> tuple[Any, Any, dict[str, Any]]:
        key = (ky, alpha)
        entry = self._y_entries.get(key)
        if entry is None:
            isy, sy_sqrt, diag = sym_inv_sqrt_and_sqrt(self.stats["syy"][:ky, :ky], alpha, self.eig_floor_rel)
            entry = (isy, sy_sqrt, diag)
            self._y_entries[key] = entry
            print(f"WHITENER y Ky={ky} alpha={alpha} min_eig={diag['min_eig']:.6e} "
                  f"max_eig={diag['max_eig']:.6e} min_kept_eig={diag['min_kept_eig']:.6e} dropped={diag['n_dropped']}")
        return entry

    def whitened_svd(self, kx: int, ky: int, alpha: float) -> tuple[Any, Any, Any]:
        np = _np()
        key = (kx, ky, alpha)
        if self._svd_key != key:
            isx, _ = self.x_whitener(kx, alpha)
            isy, _, _ = self.y_whitener(ky, alpha)
            m = isx @ self.stats["sxy"][:kx, :ky] @ isy
            self._svd_key, self._svd_entry = key, np.linalg.svd(m, full_matrices=False)
        return self._svd_entry


def fit_cca_row(cache: CCAFitCache, kx: int, ky: int, r: int, ridge_alpha: float) -> dict[str, Any]:


    np = _np()
    isx, dx = cache.x_whitener(kx, ridge_alpha)
    isy, sy_sqrt, dy = cache.y_whitener(ky, ridge_alpha)
    u, s, vt = cache.whitened_svd(kx, ky, ridge_alpha)
    if not np.all(np.isfinite(s)):
        raise RuntimeError(f"non-finite canonical correlations for row Kx={kx} Ky={ky} r={r}")
    if np.any(np.diff(s) > 1e-12):
        raise RuntimeError(f"canonical correlations not sorted descending for row Kx={kx} Ky={ky} r={r}")
    rho_r = np.clip(s[:r], 0.0, None)
    b = (isx @ u[:, :r]) @ (rho_r[:, None] * (vt[:r, :] @ sy_sqrt))
    notes = []
    status = "ok"
    if dx["n_dropped"] or dy["n_dropped"]:
        status = f"ok_dropped_eigs(sxx:{dx['n_dropped']},syy:{dy['n_dropped']})"
    if float(s[0]) > 1.0 + 1e-6:
        notes.append(f"first_canonical_corr_gt_1:{float(s[0]):.6g}")
    return {
        "B": b,
        "rho_full": s,
        "rho_retained": rho_r,
        "sxx_diag": dx,
        "syy_diag": dy,
        "fit_status": status,
        "fit_notes": notes,
    }


def split_correction_metrics(
    xc_k, y_true_k, b, y_mean_k, *, c_centered: float, sse_raw: float, n_samples: int, n_pixels: int,
) -> dict[str, float]:


    np = _np()
    yhat = xc_k @ b + y_mean_k[None, :]
    pred_sq = float(np.sum(yhat * yhat))
    cross = float(np.sum(yhat * y_true_k))
    true_sq = float(np.sum(y_true_k * y_true_k))
    sse = c_centered + pred_sq - 2.0 * cross
    if sse < 0.0:
        if sse < -1e-9 * max(c_centered, 1.0):
            raise RuntimeError(f"negative SSE {sse} beyond numerical tolerance (c_centered={c_centered})")
        sse = 0.0
    denom = float(n_samples) * float(n_pixels)
    mse = sse / denom
    mse_bilinear = sse_raw / denom
    required_rms = math.sqrt(c_centered / denom)
    predicted_rms = math.sqrt(pred_sq / denom)
    corr = cross / math.sqrt(pred_sq * true_sq) if pred_sq > 0.0 and true_sq > 0.0 else float("nan")
    return {
        "mse_K2": mse,
        "rmse_K": math.sqrt(mse),
        "skill_vs_bilinear": 1.0 - mse / mse_bilinear,
        "mse_bilinear_K2": mse_bilinear,
        "correction_corr": corr,
        "required_residual_rms_K": required_rms,
        "predicted_residual_rms_K": predicted_rms,
        "amplitude_ratio": predicted_rms / required_rms if required_rms > 0.0 else float("nan"),
        "sse": sse,
        "pred_sq": pred_sq,
        "cross": cross,
        "true_sq_retained": true_sq,
    }


def evaluate_grid(
    rows: list[dict[str, Any]],
    stats: dict[str, Any],
    train_x64, train_y64, val_x64, val_y64,
    constants: dict[str, float],
    *,
    n_pixels: int,
    eig_floor_rel: float,
    keep_fit_for: tuple[int, int, int, float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:

    np = _np()
    xc_tr = train_x64 - stats["x_mean"][None, :]
    xc_va = val_x64 - stats["x_mean"][None, :]
    n_tr, n_va = train_x64.shape[0], val_x64.shape[0]
    cache = CCAFitCache(stats, eig_floor_rel)
    out: list[dict[str, Any]] = []
    kept_fit: dict[str, Any] | None = None
    for row in rows:
        kx, ky, r, alpha = row["Kx"], row["Ky"], row["r"], row["ridge_alpha"]
        record: dict[str, Any] = dict(row)
        try:
            fit = fit_cca_row(cache, kx, ky, r, alpha)
            tr = split_correction_metrics(
                xc_tr[:, :kx], train_y64[:, :ky], fit["B"], stats["y_mean"][:ky],
                c_centered=constants["train_centered_weighted_sse"],
                sse_raw=constants["train_raw_weighted_sse"],
                n_samples=n_tr, n_pixels=n_pixels,
            )
            va = split_correction_metrics(
                xc_va[:, :kx], val_y64[:, :ky], fit["B"], stats["y_mean"][:ky],
                c_centered=constants["val_centered_weighted_sse"],
                sse_raw=constants["val_raw_weighted_sse"],
                n_samples=n_va, n_pixels=n_pixels,
            )
            rho_r = fit["rho_retained"]
            record.update({
                "train_mse_K2": tr["mse_K2"], "train_rmse_K": tr["rmse_K"],
                "train_skill_vs_bilinear": tr["skill_vs_bilinear"],
                "val_mse_K2": va["mse_K2"], "val_rmse_K": va["rmse_K"],
                "val_skill_vs_bilinear": va["skill_vs_bilinear"],
                "train_correction_corr": tr["correction_corr"], "val_correction_corr": va["correction_corr"],
                "train_required_residual_rms_K": tr["required_residual_rms_K"],
                "train_predicted_residual_rms_K": tr["predicted_residual_rms_K"],
                "train_amplitude_ratio": tr["amplitude_ratio"],
                "val_required_residual_rms_K": va["required_residual_rms_K"],
                "val_predicted_residual_rms_K": va["predicted_residual_rms_K"],
                "val_amplitude_ratio": va["amplitude_ratio"],
                "first_canonical_corr": float(rho_r[0]) if len(rho_r) else float("nan"),
                "mean_canonical_corr_retained": float(np.mean(rho_r)) if len(rho_r) else float("nan"),
                "min_canonical_corr_retained": float(rho_r[-1]) if len(rho_r) else float("nan"),
                "fit_status": fit["fit_status"],
                "notes": ";".join(fit["fit_notes"]),
            })
            if keep_fit_for is not None and (kx, ky, r, alpha) == keep_fit_for:
                kept_fit = fit
        except Exception as exc:
            for col in RESULT_COLUMNS:
                record.setdefault(col, float("nan"))
            record["fit_status"] = "failed"
            record["notes"] = f"{type(exc).__name__}:{exc}"
        out.append(record)
    return out, kept_fit


def select_best_row(records: list[dict[str, Any]]) -> dict[str, Any]:


    eligible = [
        rec for rec in records
        if str(rec.get("fit_status", "")).startswith("ok") and math.isfinite(float(rec["val_mse_K2"]))
    ]
    if not eligible:
        raise RuntimeError("no eligible grid rows to select from (all failed or non-finite)")
    return min(eligible, key=lambda rec: (float(rec["val_mse_K2"]), rec["r"], rec["Kx"], rec["Ky"]))


def profiled_kx_ky(records: list[dict[str, Any]]) -> list[dict[str, Any]]:

    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for rec in records:
        if not str(rec.get("fit_status", "")).startswith("ok"):
            continue
        groups.setdefault((rec["Kx"], rec["Ky"]), []).append(rec)
    out = []
    for (kx, ky), recs in sorted(groups.items()):
        best = min(recs, key=lambda rec: (float(rec["val_mse_K2"]), rec["r"]))
        out.append({
            "Kx": kx, "Ky": ky, "best_r": best["r"], "ridge_alpha": best["ridge_alpha"],
            "val_mse_K2": best["val_mse_K2"], "val_rmse_K": best["val_rmse_K"],
            "val_skill_vs_bilinear": best["val_skill_vs_bilinear"],
            "train_rmse_K": best["train_rmse_K"], "n_r_tested": len(recs),
        })
    return out


def sensitivity_slices(records: list[dict[str, Any]], selected: dict[str, Any]) -> list[dict[str, Any]]:

    kx, ky, r, alpha = selected["Kx"], selected["Ky"], selected["r"], selected["ridge_alpha"]
    out = []
    for slice_name, match in (
        ("vary_Kx", lambda rec: rec["Ky"] == ky and rec["r"] == r and rec["ridge_alpha"] == alpha),
        ("vary_Ky", lambda rec: rec["Kx"] == kx and rec["r"] == r and rec["ridge_alpha"] == alpha),
        ("vary_r", lambda rec: rec["Kx"] == kx and rec["Ky"] == ky and rec["ridge_alpha"] == alpha),
    ):
        for rec in records:
            if match(rec):
                out.append({"slice": slice_name, **{col: rec[col] for col in RESULT_COLUMNS}})
    return out


def accumulate_weighted_norms(block, mean_flat, w_flat) -> tuple[float, float]:


    np = _np()
    d = np.asarray(block, dtype=np.float64)
    raw = float(np.einsum("np,p->", d * d, w_flat))
    c = d - mean_flat[None, :]
    centered = float(np.einsum("np,p->", c * c, w_flat))
    return raw, centered


def compute_val_residual_constants(
    manifest_rows: list[dict[str, str]],
    mean_flat, w_flat,
    height: int, width: int,
    *,
    day_batch: int,
    expected_days: int,
) -> dict[str, Any]:

    np = _np()
    pixels = height * width
    raw_total = 0.0
    centered_total = 0.0
    days = 0
    shard_stats = []
    for row in manifest_rows:
        path = Path(row["residual_path"])
        hits = suspicious_path(str(path))
        if hits:
            raise RuntimeError(f"refusing sealed/test-token residual path: {path} {hits}")
        n = int(row["end"]) - int(row["start"])
        arr = np.load(path, mmap_mode="r")
        if arr.shape != (n, height, width):
            raise RuntimeError(f"residual shard shape mismatch {path}: {arr.shape} expected {(n, height, width)}")
        shard_raw = 0.0
        shard_centered = 0.0
        for b0 in range(0, n, day_batch):
            b1 = min(n, b0 + day_batch)
            block = np.asarray(arr[b0:b1], dtype=np.float64).reshape(b1 - b0, pixels)
            r_part, c_part = accumulate_weighted_norms(block, mean_flat, w_flat)
            shard_raw += r_part
            shard_centered += c_part
        raw_total += shard_raw
        centered_total += shard_centered
        days += n
        shard_stats.append({"path": str(path), "n_days": n, "raw_sse": shard_raw, "centered_sse": shard_centered})
    if days != expected_days:
        raise RuntimeError(f"validation day count {days} != expected {expected_days}")
    if not (math.isfinite(raw_total) and math.isfinite(centered_total) and raw_total > 0.0 and centered_total > 0.0):
        raise RuntimeError(f"non-finite/non-positive validation norms raw={raw_total} centered={centered_total}")
    return {
        "val_raw_weighted_sse": raw_total,
        "val_centered_weighted_sse": centered_total,
        "n_val_days": days,
        "shards": shard_stats,
    }


def train_constants_from_basis(y_basis_npz: Path, height: int, width: int, n_train: int) -> dict[str, float]:


    np = _np()
    tss = float(np.asarray(npz_small_member(y_basis_npz, "total_weighted_sum_squares.npy")))
    mean_field = np.asarray(npz_small_member(y_basis_npz, "mean_field.npy"), dtype=np.float64)
    if mean_field.shape != (height, width):
        raise RuntimeError(f"y basis mean_field shape {mean_field.shape} expected {(height, width)}")
    w, _ = area_weights_from_lat(grid_lat_centers(height))
    mean_norm2_w = float(np.einsum("hw,h->", mean_field * mean_field, w))
    return {
        "train_centered_weighted_sse": tss,
        "train_mean_weighted_norm2": mean_norm2_w,
        "train_raw_weighted_sse": tss + float(n_train) * mean_norm2_w,
    }


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    write_text_atomic(path, text)


def check_scores(config: dict[str, Any], log: IssueLog) -> dict[str, dict[str, Any]]:


    out: dict[str, dict[str, Any]] = {}
    expected_shapes = {
        "train_x": (int(config["expected_train_days"]), int(config["kx_max"])),
        "val_x": (int(config["expected_val_days"]), int(config["kx_max"])),
        "train_y": (int(config["expected_train_days"]), int(config["ky_max"])),
        "val_y": (int(config["expected_val_days"]), int(config["ky_max"])),
    }
    for key in SCORE_KEYS:
        entry = config["score_inputs"][key]
        npy = resolve(Path(entry["path"]))
        done = resolve(Path(entry["done_json"]))
        reject_suspicious_path(f"score_inputs.{key}.path", npy, log)
        if not npy.is_file():
            log.fail(f"missing score file: {npy}")
            continue
        if not done.is_file():
            log.fail(f"missing score done JSON: {done}")
            continue
        done_sha = sha256_file(done)
        if done_sha != entry["expected_done_sha256"]:
            log.fail(f"score done JSON sha256 mismatch for {key}: {done_sha} != pinned {entry['expected_done_sha256']}")
            continue
        payload = load_json(done)
        ident = payload.get("ident") or {}
        if ident != entry["expected_ident"]:
            log.fail(f"score done identity does not match pinned expected_ident for {key}: {done}")
            continue
        header = read_npy_header(npy)
        if header["shape"] != expected_shapes[key]:
            log.fail(f"{key} npy header shape {header['shape']} expected {expected_shapes[key]}")
            continue
        if not dtype_is_float32(header["descr"]) or header["fortran_order"]:
            log.fail(f"{key} npy dtype/order unexpected: {header['descr']} fortran={header['fortran_order']}")
            continue
        if int(payload.get("bytes", -1)) != npy.stat().st_size:
            log.fail(f"{key} npy size {npy.stat().st_size} != done-recorded {payload.get('bytes')}")
            continue
        log.pass_(f"{key}: pinned done ident + sha256 match; header {header['shape']} float32")
        out[key] = {
            "path": npy,
            "done_json": done,
            "done_sha256": done_sha,
            "ident": ident,
            "file_sha256_recorded": payload.get("sha256"),
            "shape": header["shape"],
        }
    return out


def check_y_basis(config: dict[str, Any], log: IssueLog) -> dict[str, Any] | None:

    height, width = config["expected_grid_shape"]
    entry = config["y_basis"]
    npz_path = resolve(Path(entry["npz"]))
    done_json = resolve(Path(entry["done_json"]))
    if not npz_path.is_file() or not done_json.is_file():
        log.fail(f"missing y basis npz or done JSON: {npz_path} / {done_json}")
        return None
    done_sha = sha256_file(done_json)
    if done_sha != entry["expected_done_sha256"]:
        log.fail(f"y basis done sha256 {done_sha} != pinned {entry['expected_done_sha256']}")
        return None
    if npz_path.stat().st_size != int(entry["expected_bytes"]):
        log.fail(f"y basis npz size {npz_path.stat().st_size} != pinned {entry['expected_bytes']}")
        return None
    try:
        tss_info = npz_member_npy_info(npz_path, "total_weighted_sum_squares.npy")
        mean_info = npz_member_npy_info(npz_path, "mean_field.npy")
    except Exception as exc:
        log.fail(f"y basis member inspection failed: {exc}")
        return None
    if mean_info["shape"] != (height, width):
        log.fail(f"y basis mean_field header shape {mean_info['shape']} expected {(height, width)}")
        return None
    np = _np()
    tss = float(np.asarray(npz_small_member(npz_path, "total_weighted_sum_squares.npy")))
    expected_tss = float(entry["expected_train_centered_weighted_sse"])
    if not math.isclose(tss, expected_tss, rel_tol=1e-9):
        log.fail(f"y basis total_weighted_sum_squares {tss!r} != pinned {expected_tss!r}")
        return None
    log.pass_(f"y basis pinned done sha256 + size match; total_weighted_sum_squares={tss:.6f}")
    return {"npz_path": npz_path, "done_json": done_json, "done_sha256": done_sha,
            "train_centered_weighted_sse": tss, "tss_info": tss_info}


def check_manifest(config: dict[str, Any], log: IssueLog) -> tuple[list[dict[str, str]], str]:
    manifest = resolve(Path(config["source_manifest"]))
    if not manifest.is_file():
        log.fail(f"missing source manifest: {manifest}")
        return [], ""
    manifest_sha = sha256_file(manifest)
    if manifest_sha != config["expected_source_manifest_sha256"]:
        log.fail(f"source manifest sha256 {manifest_sha} != pinned {config['expected_source_manifest_sha256']}")
        return [], manifest_sha
    rows, n_days = load_manifest_split(manifest, "val", log)
    if n_days != int(config["expected_val_days"]):
        log.fail(f"val manifest days {n_days} != expected {config['expected_val_days']}")
    if len(rows) != int(config["expected_val_shards"]):
        log.fail(f"val manifest shards {len(rows)} != expected {config['expected_val_shards']}")
    missing = 0
    for row in rows:
        path = Path(row.get("residual_path", ""))
        reject_suspicious_path("val residual shard", path, log)
        if not path.is_file():
            missing += 1
    if missing:
        log.fail(f"{missing} val residual shard file(s) missing")
    else:
        log.pass_(f"manifest sha256 pinned-match; val: {len(rows)} shards / {n_days} days; residual paths present + token-clean")
    return rows, manifest_sha


def check_grid(config: dict[str, Any], log: IssueLog) -> list[dict[str, Any]]:
    rows = build_grid_rows(config["grid"])
    expected = int(config["expected_valid_rows"])
    if len(rows) != expected:
        log.fail(f"valid grid rows {len(rows)} != expected {expected}")
    else:
        log.pass_(f"grid: {len(rows)} valid rows (r <= min(Kx,Ky)) == expected {expected}")
    for row in rows:
        if row["r"] > min(row["Kx"], row["Ky"]):
            log.fail(f"grid row violates r <= min(Kx,Ky): {row}")
    height, width = config["expected_grid_shape"]
    if height * width != int(config["expected_n_pixels"]):
        log.fail(f"n_pixels {height * width} != expected_n_pixels {config['expected_n_pixels']}")
    return rows


def check_config(config: dict[str, Any], args, log: IssueLog) -> dict[str, Any]:


    if config.get("stage") != STAGE_NAME:
        log.fail(f"config stage {config.get('stage')!r} != {STAGE_NAME!r}")
    home_root = guard_home_output_root(args.output_root, log)
    heavy_root = guard_heavy_output_root(args.heavy_output_root, log)
    scores = check_scores(config, log)
    basis = check_y_basis(config, log)
    manifest_rows, manifest_sha = check_manifest(config, log)
    grid_rows = check_grid(config, log)
    for key, path in config["intended_outputs"].items():
        if path is None:
            continue
        out = resolve(Path(path))
        if not is_relative_to(out, resolve(ALLOWED_HOME_OUTPUT_ROOT)):
            log.fail(f"intended output {key} is outside allowed home outputs tree: {out}")
        if out.exists():
            log.warn(f"intended output already exists (resume/overwrite protocol will decide): {out}")
    if len(scores) == len(SCORE_KEYS) and basis is not None:
        train_manifest_shas = {scores[k]["ident"].get("source_manifest_sha256") for k in scores}
        if train_manifest_shas != {config["expected_source_manifest_sha256"]}:
            log.fail(f"score idents disagree with pinned source manifest sha: {train_manifest_shas}")
        else:
            log.pass_("all four score idents carry the pinned source manifest sha256 (same split definition)")
    return {
        "home_root": home_root,
        "heavy_root": heavy_root,
        "scores": scores,
        "basis": basis,
        "manifest_rows": manifest_rows,
        "manifest_sha": manifest_sha,
        "grid_rows": grid_rows,
    }


def print_issues(log: IssueLog) -> None:
    for msg in log.passes:
        print(f"PASS {msg}")
    for msg in log.warnings:
        print(f"WARN {msg}")
    for msg in log.failures:
        print(f"FAIL {msg}")
    print(f"SUMMARY pass={len(log.passes)} warn={len(log.warnings)} fail={len(log.failures)}")


def print_plan(config: dict[str, Any], state: dict[str, Any], config_sha: str) -> None:
    print("PLAN stage=03_run_cca_grid")
    print(f"PLAN config_sha256={config_sha}")
    grid = config["grid"]
    print(f"PLAN grid Kx={grid['kx_values']} Ky={grid['ky_values']} r={grid['r_values']} alpha={grid['ridge_alpha_values']}")
    print(f"PLAN valid_rows={len(state['grid_rows'])}")
    print("PLAN heavy phase order: 1) compute-constants (singleton)  2) run-grid (singleton)")
    for key, path in config["intended_outputs"].items():
        print(f"PLAN output {key}: {path}")
    env_set = os.environ.get(REAL_RUN_ENV) == "1"
    print(f"PLAN gates: production_execution_approved={config.get('production_execution_approved')} {REAL_RUN_ENV}_set={env_set}")


def gate_heavy(config: dict[str, Any], phase: str) -> None:
    if not config.get("production_execution_approved"):
        raise SystemExit(
            f"REFUSING heavy phase {phase}: config production_execution_approved is not true. "
            "This is gate 1 of 2; do not flip it without explicit approval."
        )
    if os.environ.get(REAL_RUN_ENV) != "1":
        raise SystemExit(
            f"REFUSING heavy phase {phase}: {REAL_RUN_ENV}=1 is not set. "
            "This is gate 2 of 2; export it only in an approved production shell."
        )


def constants_ident(config: dict[str, Any], config_sha: str, state: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": STAGE_NAME,
        "phase": "compute-constants",
        "mode": "production",
        "config_sha256": config_sha,
        "source_manifest_sha256": state["manifest_sha"],
        "y_basis_done_sha256": state["basis"]["done_sha256"],
        "grid_shape": list(config["expected_grid_shape"]),
        "n_pixels": int(config["expected_n_pixels"]),
        "n_train": int(config["expected_train_days"]),
        "n_val": int(config["expected_val_days"]),
    }


def run_compute_constants(config: dict[str, Any], args, log: IssueLog, state: dict[str, Any], config_sha: str) -> None:
    gate_heavy(config, "compute-constants")
    np = _np()
    height, width = config["expected_grid_shape"]
    out_json = resolve(Path(config["intended_outputs"]["metric_constants_json"]))
    ident = constants_ident(config, config_sha, state)
    decision = output_state(out_json, ident, resume=args.resume, overwrite=args.overwrite)
    if decision == "skip":
        print(f"COMPUTE_CONSTANTS_SKIPPED {out_json} (done identity matches)")
        return
    if decision == "refuse":
        raise SystemExit(f"REFUSING to overwrite existing {out_json} without matching done identity; pass --overwrite to replace")

    train_consts = train_constants_from_basis(state["basis"]["npz_path"], height, width, int(config["expected_train_days"]))
    w, _ = area_weights_from_lat(grid_lat_centers(height))
    w_flat = np.repeat(w[:, None], width, axis=1).reshape(-1)
    mean_field = np.asarray(npz_small_member(state["basis"]["npz_path"], "mean_field.npy"), dtype=np.float64)
    mean_flat = mean_field.reshape(-1)
    val = compute_val_residual_constants(
        state["manifest_rows"], mean_flat, w_flat, height, width,
        day_batch=int(config["algorithm"]["val_norm_day_batch"]),
        expected_days=int(config["expected_val_days"]),
    )
    payload = {
        "ident": ident,
        "created_utc": utc_now(),
        "constants": {
            **train_consts,
            "val_raw_weighted_sse": val["val_raw_weighted_sse"],
            "val_centered_weighted_sse": val["val_centered_weighted_sse"],
        },
        "n_val_days": val["n_val_days"],
        "val_shard_stats": val["shards"],
        "definitions": {
            "train_centered_weighted_sse": "Stage 01 y basis member total_weighted_sum_squares: sum over train days/pixels of (residual - train_residual_mean)^2 * area_weight",
            "train_raw_weighted_sse": "train_centered + n_train * ||train_residual_mean||_w^2 (exact; centered train residuals sum to zero)",
            "val_centered_weighted_sse": "exact sum over val days/pixels of (residual - train_residual_mean)^2 * area_weight, streamed from val residual shards",
            "val_raw_weighted_sse": "exact sum over val days/pixels of residual^2 * area_weight (bilinear baseline error)",
            "area_weight": "cos(lat_center)/mean(cos(lat_center)), grid-formula lat centers, mean weight = 1",
        },
    }
    write_json_atomic(out_json, payload)
    write_done(done_path(out_json), ident, {"sha256": sha256_file(out_json)})
    print(f"COMPUTE_CONSTANTS_DONE {out_json}")
    for key, value in payload["constants"].items():
        print(f"CONSTANT {key}={value:.6f}")


def grid_ident(config: dict[str, Any], config_sha: str, state: dict[str, Any], constants_sha: str) -> dict[str, Any]:
    return {
        "stage": STAGE_NAME,
        "phase": "run-grid",
        "mode": "production",
        "config_sha256": config_sha,
        "score_done_sha256s": {key: state["scores"][key]["done_sha256"] for key in SCORE_KEYS},
        "y_basis_done_sha256": state["basis"]["done_sha256"],
        "metric_constants_sha256": constants_sha,
        "grid": {k: list(v) for k, v in config["grid"].items()},
        "valid_rows": len(state["grid_rows"]),
        "n_train": int(config["expected_train_days"]),
        "n_val": int(config["expected_val_days"]),
        "kx_max": int(config["kx_max"]),
        "ky_max": int(config["ky_max"]),
        "grid_shape": list(config["expected_grid_shape"]),
        "n_pixels": int(config["expected_n_pixels"]),
    }


def load_score_matrix(entry: dict[str, Any], verify_sha: bool = True):
    np = _np()
    if verify_sha and entry.get("file_sha256_recorded"):
        actual = sha256_file(entry["path"])
        if actual != entry["file_sha256_recorded"]:
            raise RuntimeError(f"score file sha256 mismatch for {entry['path']}: {actual} != done-recorded {entry['file_sha256_recorded']}")
    return np.asarray(np.load(entry["path"]), dtype=np.float64)


def run_grid_phase(config: dict[str, Any], args, log: IssueLog, state: dict[str, Any], config_sha: str) -> None:
    gate_heavy(config, "run-grid")
    np = _np()
    outputs = {key: resolve(Path(path)) for key, path in config["intended_outputs"].items() if path is not None}
    constants_json = outputs["metric_constants_json"]
    if not constants_json.is_file():
        raise SystemExit(f"run-grid requires metric constants; run compute-constants first: {constants_json}")
    expected_cident = constants_ident(config, config_sha, state)
    if not done_matches(done_path(constants_json), expected_cident):
        raise SystemExit(f"metric constants done identity mismatch (stale or tampered): {done_path(constants_json)}")
    constants_payload = load_json(constants_json)
    constants = {k: float(v) for k, v in constants_payload["constants"].items()}
    constants_sha = sha256_file(constants_json)

    ident = grid_ident(config, config_sha, state, constants_sha)
    results_csv = outputs["grid_results_csv"]
    decision = output_state(results_csv, ident, resume=args.resume, overwrite=args.overwrite)
    if decision == "skip":
        print(f"RUN_GRID_SKIPPED {results_csv} (done identity matches)")
        return
    if decision == "refuse":
        raise SystemExit(f"REFUSING to overwrite existing {results_csv} without matching done identity; pass --overwrite to replace")

    scores64 = {key: load_score_matrix(state["scores"][key]) for key in SCORE_KEYS}
    n_pixels = int(config["expected_n_pixels"])


    for split, c_key in (("train", "train_centered_weighted_sse"), ("val", "val_centered_weighted_sse")):
        score_sq = float(np.sum(scores64[f"{split}_y"] ** 2))
        c_val = constants[c_key]
        if score_sq > c_val * (1.0 + 1e-6):
            raise SystemExit(f"{split} retained Y score energy {score_sq} exceeds centered SSE constant {c_val}; constants/scores inconsistent")
        print(f"CHECK {split} retained_score_energy/centered_sse = {score_sq / c_val:.6f} (truncation keeps the remainder)")

    stats = compute_score_stats(scores64["train_x"], scores64["train_y"])
    eig_floor_rel = float(config["algorithm"]["eig_floor_rel"])
    records, _ = evaluate_grid(
        state["grid_rows"], stats,
        scores64["train_x"], scores64["train_y"], scores64["val_x"], scores64["val_y"],
        constants, n_pixels=n_pixels, eig_floor_rel=eig_floor_rel,
    )
    n_failed = sum(1 for rec in records if rec["fit_status"] == "failed")
    if n_failed:
        log.warn(f"{n_failed} grid row(s) failed to fit; excluded from selection")
    selected = select_best_row(records)
    sel_key = (selected["Kx"], selected["Ky"], selected["r"], selected["ridge_alpha"])
    print(f"SELECTED Kx={sel_key[0]} Ky={sel_key[1]} r={sel_key[2]} alpha={sel_key[3]} "
          f"val_rmse_K={selected['val_rmse_K']:.6f} val_skill={selected['val_skill_vs_bilinear']:.6f}")


    _, sel_fit = evaluate_grid(
        [dict(Kx=sel_key[0], Ky=sel_key[1], r=sel_key[2], ridge_alpha=sel_key[3])], stats,
        scores64["train_x"], scores64["train_y"], scores64["val_x"], scores64["val_y"],
        constants, n_pixels=n_pixels, eig_floor_rel=eig_floor_rel, keep_fit_for=sel_key,
    )
    if sel_fit is None:
        raise SystemExit("selected-row refit unexpectedly failed")

    write_csv_atomic(results_csv, RESULT_COLUMNS, records)
    write_jsonl_atomic(outputs["grid_results_jsonl"], records)
    write_csv_atomic(outputs["profiled_kx_ky_csv"],
                     ["Kx", "Ky", "best_r", "ridge_alpha", "val_mse_K2", "val_rmse_K",
                      "val_skill_vs_bilinear", "train_rmse_K", "n_r_tested"],
                     profiled_kx_ky(records))
    write_csv_atomic(outputs["sensitivity_slices_csv"], ["slice"] + RESULT_COLUMNS,
                     sensitivity_slices(records, selected))
    spectrum_rows = [
        {"mode_index": i + 1, "canonical_correlation": float(v), "retained": i < sel_key[2]}
        for i, v in enumerate(np.asarray(sel_fit["rho_full"]))
    ]
    write_csv_atomic(outputs["canonical_spectrum_csv"],
                     ["mode_index", "canonical_correlation", "retained"], spectrum_rows)

    kx, ky, r, alpha = sel_key
    atomic_savez(
        outputs["selected_model_npz"],
        Kx=np.int64(kx), Ky=np.int64(ky), r=np.int64(r), ridge_alpha=np.float64(alpha),
        x_score_mean=stats["x_mean"][:kx],
        y_score_mean=stats["y_mean"][:ky],
        B=np.asarray(sel_fit["B"], dtype=np.float64),
        canonical_correlations_full=np.asarray(sel_fit["rho_full"], dtype=np.float64),
        canonical_correlations_retained=np.asarray(sel_fit["rho_retained"], dtype=np.float64),
    )
    model_meta = {
        "ident": ident,
        "created_utc": utc_now(),
        "selected": {col: selected[col] for col in RESULT_COLUMNS},
        "prediction_formula": "yhat_scores = (x_scores[:, :Kx] - x_score_mean) @ B + y_score_mean; predicted field = bilinear + train_residual_mean + yhat_scores @ y_eofs_weighted[:Ky] / sqrt_weight",
        "fit_diagnostics": {"sxx": sel_fit["sxx_diag"], "syy": sel_fit["syy_diag"]},
        "metric_constants": constants,
        "selection_rule": "min val_mse_K2; tie-breakers: smaller r, then Kx, then Ky",
        "caveat": "validation-selected; validation metrics here are model-selection metrics, not unbiased final claims; sealed test untouched",
    }
    write_json_atomic(outputs["selected_metadata_json"], model_meta)

    summary = {
        "ident": ident,
        "created_utc": utc_now(),
        "n_rows": len(records),
        "n_failed": n_failed,
        "selected": {"Kx": kx, "Ky": ky, "r": r, "ridge_alpha": alpha,
                     "val_rmse_K": selected["val_rmse_K"], "val_skill_vs_bilinear": selected["val_skill_vs_bilinear"]},
        "best_val_rmse_K_top5": sorted(
            ({"Kx": rec["Kx"], "Ky": rec["Ky"], "r": rec["r"], "val_rmse_K": rec["val_rmse_K"]}
             for rec in records if str(rec["fit_status"]).startswith("ok")),
            key=lambda rec: rec["val_rmse_K"])[:5],
        "metric_constants": constants,
    }
    write_json_atomic(outputs["grid_summary_json"], summary)
    write_done(done_path(results_csv), ident, {
        "sha256": sha256_file(results_csv),
        "n_rows": len(records),
        "n_failed": n_failed,
        "selected": summary["selected"],
    })
    status_path = outputs["stage_status_json"]
    status = load_json(status_path) if status_path.is_file() else {}
    status.update({
        "stage": STAGE_NAME,
        "grid": {"status": "complete", "created_utc": utc_now(), "n_rows": len(records), "n_failed": n_failed},
        "selected": summary["selected"],
        "config_sha256": config_sha,
    })
    write_json_atomic(status_path, status)
    print(f"RUN_GRID_DONE rows={len(records)} failed={n_failed} -> {results_csv}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("plan",) + HEAVY_PHASES, default="plan")
    parser.add_argument("--dry-run", action="store_true", help="alias for --phase plan")
    parser.add_argument("--output-root", type=Path, default=ALLOWED_HOME_OUTPUT_ROOT)
    parser.add_argument("--heavy-output-root", type=Path, default=DEFAULT_HEAVY_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    phase = "plan" if args.dry_run else args.phase

    log = IssueLog()
    config_path = resolve(args.config)
    if not config_path.is_file():
        print(f"FAIL missing config {config_path}")
        return 2
    config = load_json(config_path)
    config_sha = sha256_file(config_path)

    state = check_config(config, args, log)
    if phase == "plan":
        print_plan(config, state, config_sha)
        print_issues(log)
        return 0 if log.ok() else 2

    if not log.ok():
        print_issues(log)
        print(f"REFUSING phase {phase}: preflight failures above")
        return 2

    if phase == 'compute-constants':
        run_compute_constants(config, args, log, state, config_sha)
    elif phase == 'run-grid':
        run_grid_phase(config, args, log, state, config_sha)
    print_issues(log)
    return 0 if log.ok() else 2


if __name__ == "__main__":
    sys.exit(main())
