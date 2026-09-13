"""Evaluate the selected CCA mapping on the full validation period.
Predicts residual EOF scores from the saved validation predictor scores,
reconstructs corrections on the high-resolution grid and compares them with
the target and bilinear baseline. Processes chunks and reduces their global,
regional, daily and spatial statistics without fitting a new model.
Requires the selected mapping, fixed EOF bases and external validation fields.
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
    npz_member_memmap,
    npz_member_npy_info,
    npz_small_member,
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
DEFAULT_CONFIG = HOME_RUN_ROOT / "config" / "cca_eval_config.json"
STAGE_NAME = "04_evaluate_selected_cca"
SEAL_TOKENS = ("awi_downscaling_test", "sealed", "holdout")
REAL_RUN_ENV = "CCA_STAGE04_ENABLE_REAL_RUN"
HEAVY_PHASES = ("eval-chunk", "reduce-eval")
SMALL_MEMBER_MAX_BYTES = 64 << 20


ACC_KEYS = ("err2_cca", "abs_cca", "err_cca", "err2_base", "abs_base", "err_base", "rh2", "cross")
SPECTRA_FIELDS = ("target", "baseline", "prediction", "model_error", "baseline_error")

REGION_ORDER = [
    "global", "land", "ocean",
    "elevation_gt_1000m", "elevation_le_1000m",
    "tropics", "midlat_north", "midlat_south",
    "highlat_north", "highlat_south",
    "arctic_gt_80N", "antarctic_lt_80S",
]

DAILY_COLUMNS = [
    "day_index", "shard_task_id", "day_in_shard",
    "baseline_rmse_K", "cca_rmse_K",
    "baseline_mae_K", "cca_mae_K",
    "baseline_bias_K", "cca_bias_K",
    "mse_skill_vs_bilinear",
    "baseline_rmse_norm", "cca_rmse_norm",
    "correction_corr",
]

REGIONAL_COLUMNS = [

    "region", "area_weighted_rmse_K", "area_weighted_skill",
    "area_weighted_bias_K", "area_weighted_mae_K",

    "baseline_rmse_K", "baseline_mae_K", "baseline_bias_K",
    "rmse_improvement_K", "correction_corr", "amplitude_ratio",
    "n_pixels", "weight_fraction",
    "cca_rmse_norm", "baseline_rmse_norm",
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
        log.pass_(f"heavy output root confined to scratch tree {scratch}")
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


def load_manifest_split(path: Path, split: str, log: IssueLog) -> tuple[list[dict[str, str]], int]:

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


def region_masks(lat_deg, lsm, oro_m) -> dict[str, Any]:


    np = _np()
    height, width = lsm.shape
    lat = np.broadcast_to(np.asarray(lat_deg, dtype=np.float64)[:, None], (height, width))
    ones = np.ones((height, width), dtype=bool)
    return {
        "global": ones,
        "land": lsm > 0.5,
        "ocean": lsm < 0.5,
        "elevation_gt_1000m": oro_m > 1000.0,
        "elevation_le_1000m": oro_m <= 1000.0,
        "tropics": np.abs(lat) < 23.5,
        "midlat_north": (lat >= 23.5) & (lat < 60.0),
        "midlat_south": (lat <= -23.5) & (lat > -60.0),
        "highlat_north": (lat >= 60.0) & (lat < 80.0),
        "highlat_south": (lat <= -60.0) & (lat > -80.0),
        "arctic_gt_80N": lat >= 80.0,
        "antarctic_lt_80S": lat <= -80.0,
    }


def evaluate_days_dense(
    *,
    yhat64,
    base_arr,
    resid_arr,
    eofs,
    ky: int,
    mean_field64,
    w,
    sqrt_w,
    block_rows: int,
    spectra_enabled: bool,
) -> dict[str, Any]:


    np = _np()
    n, height, width = base_arr.shape
    n_freq = width // 2 + 1
    day = {key: np.zeros(n, dtype=np.float64) for key in ACC_KEYS}
    maps = {key: np.zeros((height, width), dtype=np.float64) for key in ACC_KEYS}
    power = np.zeros((len(SPECTRA_FIELDS), n_freq), dtype=np.float64) if spectra_enabled else None
    for r0 in range(0, height, block_rows):
        r1 = min(height, r0 + block_rows)
        rows = r1 - r0
        e_blk = np.asarray(eofs[:ky, r0:r1, :], dtype=np.float64).reshape(ky, rows * width)
        resid_true = np.asarray(resid_arr[:, r0:r1, :], dtype=np.float64)
        base = np.asarray(base_arr[:, r0:r1, :], dtype=np.float64)
        centered_hat = (yhat64 @ e_blk).reshape(n, rows, width)
        resid_hat = mean_field64[r0:r1][None, :, :] + centered_hat / sqrt_w[r0:r1][None, :, None]
        del centered_hat, e_blk
        err_cca = resid_hat - resid_true
        w_blk = w[r0:r1]

        def accumulate(key: str, arr) -> None:
            day[key] += np.einsum("nrw,r->n", arr, w_blk)
            maps[key][r0:r1, :] += arr.sum(axis=0)

        accumulate("err2_cca", err_cca * err_cca)
        accumulate("abs_cca", np.abs(err_cca))
        accumulate("err_cca", err_cca)
        accumulate("err2_base", resid_true * resid_true)                               
        accumulate("abs_base", np.abs(resid_true))
        accumulate("err_base", -resid_true)
        accumulate("rh2", resid_hat * resid_hat)
        accumulate("cross", resid_hat * resid_true)

        if spectra_enabled:
            fields = {
                "target": base + resid_true,
                "baseline": base,
                "prediction": base + resid_hat,
                "model_error": err_cca,
                "baseline_error": -resid_true,
            }
            for index, name in enumerate(SPECTRA_FIELDS):
                coeff = np.fft.rfft(fields[name], axis=-1) / width
                spec = np.abs(coeff) ** 2
                power[index] += np.einsum("nrf,r->f", spec, w_blk)
            del fields
        del resid_true, base, resid_hat, err_cca
    for key in ACC_KEYS:
        if not np.all(np.isfinite(day[key])) or not np.all(np.isfinite(maps[key])):
            raise RuntimeError(f"non-finite accumulator {key}")
    if spectra_enabled and not np.all(np.isfinite(power)):
        raise RuntimeError("non-finite spectra accumulator")
    return {
        "n_days": n,
        "day": day,
        "maps": maps,
        "power": power,
        "weight_row_sum": float(w.sum()),
        "n_freq": n_freq,
    }


def predict_yhat_scores(x_scores64, model) -> Any:

    np = _np()
    kx = int(model["Kx"])
    xc = np.asarray(x_scores64[:, :kx], dtype=np.float64) - model["x_score_mean"][None, :]
    return xc @ model["B"] + model["y_score_mean"][None, :]


def check_selected_model(config: dict[str, Any], log: IssueLog) -> dict[str, Any] | None:


    np = _np()
    entry = config["selected_model"]
    npz_path = resolve(Path(entry["npz"]))
    reject_suspicious_path("selected_model.npz", npz_path, log)
    if not npz_path.is_file():
        log.fail(f"missing selected model npz: {npz_path}")
        return None
    actual_sha = sha256_file(npz_path)
    if actual_sha != entry["expected_sha256"]:
        log.fail(f"selected model npz sha256 {actual_sha} != pinned {entry['expected_sha256']} (stale/tampered model refused)")
        return None
    expected = entry["expected_selected"]
    members = {}
    for name in ("Kx", "Ky", "r", "ridge_alpha", "x_score_mean", "y_score_mean", "B"):
        members[name] = npz_small_member(npz_path, name + ".npy")
    kx, ky = int(members["Kx"]), int(members["Ky"])
    ok = (
        kx == int(expected["Kx"]) and ky == int(expected["Ky"])
        and int(members["r"]) == int(expected["r"])
        and float(members["ridge_alpha"]) == float(expected["ridge_alpha"])
        and members["x_score_mean"].shape == (kx,)
        and members["y_score_mean"].shape == (ky,)
        and members["B"].shape == (kx, ky)
        and members["B"].dtype == np.float64
    )
    if not ok:
        log.fail(
            f"selected model members mismatch: Kx={kx} Ky={ky} r={int(members['r'])} "
            f"alpha={float(members['ridge_alpha'])} B={members['B'].shape} expected {expected}"
        )
        return None
    if not (np.all(np.isfinite(members["B"])) and np.all(np.isfinite(members["x_score_mean"])) and np.all(np.isfinite(members["y_score_mean"]))):
        log.fail("selected model contains non-finite values")
        return None
    log.pass_(f"selected model pinned sha256 match; Kx={kx} Ky={ky} r={int(members['r'])} alpha={float(members['ridge_alpha'])}; B {members['B'].shape} float64 finite")
    return {
        "path": npz_path,
        "sha256": actual_sha,
        "Kx": kx,
        "Ky": ky,
        "r": int(members["r"]),
        "ridge_alpha": float(members["ridge_alpha"]),
        "x_score_mean": np.asarray(members["x_score_mean"], dtype=np.float64),
        "y_score_mean": np.asarray(members["y_score_mean"], dtype=np.float64),
        "B": np.asarray(members["B"], dtype=np.float64),
    }


def check_stage03_references(config: dict[str, Any], log: IssueLog) -> dict[str, Any] | None:


    s3 = config["stage03"]
    out: dict[str, Any] = {}
    for key, pin_key in (("selected_metadata_json", "expected_selected_metadata_sha256"),
                         ("grid_summary_json", "expected_grid_summary_sha256"),
                         ("metric_constants_json", "expected_metric_constants_sha256")):
        path = resolve(Path(s3[key]))
        if not path.is_file():
            log.fail(f"missing Stage 03 reference {key}: {path}")
            return None
        sha = sha256_file(path)
        if sha != s3[pin_key]:
            log.fail(f"Stage 03 {key} sha256 {sha} != pinned {s3[pin_key]} (stale/tampered reference refused)")
            return None
        out[key] = {"path": path, "sha256": sha, "payload": load_json(path)}
    meta = out["selected_metadata_json"]["payload"]
    sel = meta.get("selected") or {}
    expected_sel = s3["expected_selected_val_metrics"]
    for key, expected_value in expected_sel.items():
        actual = sel.get(key)
        if actual is None or not math.isclose(float(actual), float(expected_value), rel_tol=0.0, abs_tol=0.0):
            log.fail(f"selected metadata {key}={actual!r} != config-pinned {expected_value!r}")
            return None
    unit_patch = meta.get("unit_patch") or {}
    norm = config["normalization"]
    for key in ("target_std_K", "target_mean_K", "input_std_K", "input_mean_K"):
        if float(unit_patch.get(key, float("nan"))) != float(norm[key]):
            log.fail(f"normalization {key} mismatch: metadata unit_patch {unit_patch.get(key)!r} != config {norm[key]!r}")
            return None
    constants = {k: float(v) for k, v in out["metric_constants_json"]["payload"]["constants"].items()}
    for key, expected_value in s3["expected_constants"].items():
        if constants.get(key) != float(expected_value):
            log.fail(f"metric constant {key}={constants.get(key)!r} != config-pinned {expected_value!r}")
            return None
    if meta.get("ident", {}).get("metric_constants_sha256") != out["metric_constants_json"]["sha256"]:
        log.fail("selected metadata ident.metric_constants_sha256 does not match the pinned metric_constants.json")
        return None
    log.pass_("Stage 03 metadata/summary/constants pinned sha256 match; selected metrics, unit patch and constants agree with config pins")
    out["constants"] = constants
    out["selected_val_metrics"] = {k: float(sel[k]) for k in expected_sel}
    return out


def check_y_basis(config: dict[str, Any], log: IssueLog) -> dict[str, Any] | None:


    np = _np()
    height, width = config["expected_grid_shape"]
    ky_max = int(config["ky_max"])
    entry = config["y_basis"]
    npz_path = resolve(Path(entry["npz"]))
    done_json = resolve(Path(entry["done_json"]))
    reject_suspicious_path("y_basis.npz", npz_path, log)
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
        eofs_info = npz_member_npy_info(npz_path, "eofs_weighted.npy")
        mean_info = npz_member_npy_info(npz_path, "mean_field.npy")
    except Exception as exc:
        log.fail(f"y basis member inspection failed: {exc}")
        return None
    if eofs_info["shape"] != (ky_max, height, width) or not dtype_is_float32(eofs_info["descr"]) or eofs_info["fortran_order"]:
        log.fail(f"eofs_weighted header {eofs_info['shape']} {eofs_info['descr']} expected {(ky_max, height, width)} float32 C-order")
        return None
    if mean_info["shape"] != (height, width) or not dtype_is_float32(mean_info["descr"]):
        log.fail(f"mean_field header {mean_info['shape']} {mean_info['descr']} expected {(height, width)} float32")
        return None
    lat_member = np.asarray(npz_small_member(npz_path, "latitude.npy"), dtype=np.float64)
    lat_dev = float(np.max(np.abs(lat_member - grid_lat_centers(height))))
    if lat_dev > 1e-4:
        log.fail(f"y basis latitude deviates from grid formula centers by {lat_dev:.3e} deg (> 1e-4)")
        return None
    log.pass_(f"y basis pinned done sha256 + size match; eofs_weighted {eofs_info['shape']} float32 ZIP_STORED memmap-able; latitude matches formula (max dev {lat_dev:.2e} deg)")
    return {"npz_path": npz_path, "done_json": done_json, "done_sha256": done_sha}


def check_scores(config: dict[str, Any], log: IssueLog) -> dict[str, dict[str, Any]]:

    out: dict[str, dict[str, Any]] = {}
    expected_shapes = {
        "val_x": (int(config["expected_val_days"]), int(config["kx_max"])),
        "val_y": (int(config["expected_val_days"]), int(config["ky_max"])),
    }
    for key in ("val_x", "val_y"):
        entry = config["score_inputs"][key]
        npy = resolve(Path(entry["path"]))
        done = resolve(Path(entry["done_json"]))
        reject_suspicious_path(f"score_inputs.{key}.path", npy, log)
        if not npy.is_file() or not done.is_file():
            log.fail(f"missing score file or done JSON for {key}: {npy}")
            continue
        done_sha = sha256_file(done)
        if done_sha != entry["expected_done_sha256"]:
            log.fail(f"score done JSON sha256 mismatch for {key}: {done_sha} != pinned {entry['expected_done_sha256']}")
            continue
        payload = load_json(done)
        if (payload.get("ident") or {}) != entry["expected_ident"]:
            log.fail(f"score done identity does not match pinned expected_ident for {key}: {done}")
            continue
        header = read_npy_header(npy)
        if header["shape"] != expected_shapes[key] or not dtype_is_float32(header["descr"]) or header["fortran_order"]:
            log.fail(f"{key} npy header {header['shape']} {header['descr']} expected {expected_shapes[key]} float32 C-order")
            continue
        if int(payload.get("bytes", -1)) != npy.stat().st_size:
            log.fail(f"{key} npy size {npy.stat().st_size} != done-recorded {payload.get('bytes')}")
            continue
        log.pass_(f"{key}: pinned done ident + sha256 match; header {header['shape']} float32")
        out[key] = {
            "path": npy,
            "done_sha256": done_sha,
            "file_sha256_recorded": payload.get("sha256"),
            "shape": header["shape"],
        }
    return out


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
    height, width = config["expected_grid_shape"]
    bad = 0
    for row in rows:
        n = int(row["end"]) - int(row["start"])
        for column in ("baseline_path", "residual_path"):
            path = Path(row.get(column, ""))
            reject_suspicious_path(f"val {column}", path, log)
            if not path.is_file():
                log.fail(f"missing val shard file: {path}")
                bad += 1
                continue
            header = read_npy_header(path)
            if header["shape"] != (n, height, width) or not dtype_is_float32(header["descr"]) or header["fortran_order"]:
                log.fail(f"val shard header {header['shape']} {header['descr']} expected {(n, height, width)} float32: {path}")
                bad += 1
    if not bad:
        log.pass_(f"manifest sha256 pinned-match; val: {len(rows)} shards / {n_days} days; baseline+residual headers verified, token-clean")
    return rows, manifest_sha


def check_statics(config: dict[str, Any], log: IssueLog) -> dict[str, Any] | None:

    np = _np()
    height, width = config["expected_grid_shape"]
    statics = config["statics"]
    sf_path = resolve(Path(statics["surface_fractions_nc"]["path"]))
    oro_path = resolve(Path(statics["orography_nc"]["path"]))
    for label, path, pin in (
        ("surface_fractions_nc", sf_path, statics["surface_fractions_nc"]["expected_sha256"]),
        ("orography_nc", oro_path, statics["orography_nc"]["expected_sha256"]),
    ):
        reject_suspicious_path(label, path, log)
        if not path.is_file():
            log.fail(f"missing static source {label}: {path}")
            return None
        sha = sha256_file(path)
        if sha != pin:
            log.fail(f"{label} sha256 {sha} != pinned {pin}")
            return None
    try:
        from netCDF4 import Dataset
    except Exception as exc:
        log.fail(f"netCDF4 unavailable for statics: {exc}")
        return None
    with Dataset(sf_path, "r") as ds:
        for var in (statics["surface_fractions_nc"]["lsm_var"], statics["surface_fractions_nc"]["cl_var"]):
            if var not in ds.variables or ds.variables[var].shape != (height, width):
                log.fail(f"surface fractions variable {var} missing or wrong shape")
                return None
        lat = np.asarray(ds.variables[statics["surface_fractions_nc"]["lat_var"]][:], dtype=np.float64)
        lon = np.asarray(ds.variables[statics["surface_fractions_nc"]["lon_var"]][:], dtype=np.float64)
    with Dataset(oro_path, "r") as ds:
        var = statics["orography_nc"]["var"]
        if var not in ds.variables or tuple(ds.variables[var].shape[-2:]) != (height, width):
            log.fail(f"orography variable {var} missing or wrong shape")
            return None
    lat_dev = float(np.max(np.abs(lat - grid_lat_centers(height))))
    if lat.shape != (height,) or lon.shape != (width,) or lat_dev > 1e-4:
        log.fail(f"statics lat/lon mismatch: lat {lat.shape} dev {lat_dev:.3e} deg, lon {lon.shape}")
        return None
    log.pass_(f"statics pinned sha256 match; lat matches grid formula (max dev {lat_dev:.2e} deg); lsm/cl/orography shapes verified")
    return {
        "surface_fractions_path": sf_path,
        "surface_fractions_sha256": statics["surface_fractions_nc"]["expected_sha256"],
        "orography_path": oro_path,
        "orography_sha256": statics["orography_nc"]["expected_sha256"],
        "lat": lat,
        "lon": lon,
    }


def load_statics_fields(config: dict[str, Any], statics_state: dict[str, Any]) -> dict[str, Any]:

    np = _np()
    from netCDF4 import Dataset

    statics = config["statics"]
    with Dataset(statics_state["surface_fractions_path"], "r") as ds:
        lsm = np.asarray(ds.variables[statics["surface_fractions_nc"]["lsm_var"]][:], dtype=np.float32)
        cl = np.asarray(ds.variables[statics["surface_fractions_nc"]["cl_var"]][:], dtype=np.float32)
    with Dataset(statics_state["orography_path"], "r") as ds:
        oro = np.asarray(ds.variables[statics["orography_nc"]["var"]][:], dtype=np.float32)
    oro_m = oro.reshape(oro.shape[-2], oro.shape[-1])
    return {"lsm": lsm, "cl": cl, "oro_m": oro_m}


def build_chunk_plan(manifest_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_index": index,
            "start": int(row["start"]),
            "end": int(row["end"]),
            "n_days": int(row["end"]) - int(row["start"]),
            "task_id": row.get("task_id", str(index)),
            "baseline_path": row["baseline_path"],
            "residual_path": row["residual_path"],
        }
        for index, row in enumerate(manifest_rows)
    ]


def check_config(config: dict[str, Any], args, log: IssueLog) -> dict[str, Any]:


    if config.get("stage") != STAGE_NAME:
        log.fail(f"config stage {config.get('stage')!r} != {STAGE_NAME!r}")
    home_root = guard_home_output_root(args.output_root, log)
    heavy_root = guard_heavy_output_root(args.heavy_output_root, log)
    height, width = config["expected_grid_shape"]
    if height * width != int(config["expected_n_pixels"]):
        log.fail(f"n_pixels {height * width} != expected_n_pixels {config['expected_n_pixels']}")
    model = check_selected_model(config, log)
    stage03 = check_stage03_references(config, log)
    basis = check_y_basis(config, log)
    scores = check_scores(config, log)
    manifest_rows, manifest_sha = check_manifest(config, log)
    statics = check_statics(config, log)
    chunks = build_chunk_plan(manifest_rows) if manifest_rows else []
    if model is not None and scores.get("val_x") is not None:
        if int(model["Kx"]) > int(config["kx_max"]) or int(model["Ky"]) > int(config["ky_max"]):
            log.fail(f"selected Kx/Ky {model['Kx']}/{model['Ky']} exceed score/basis caps {config['kx_max']}/{config['ky_max']}")
    for key, path in config["intended_outputs"].items():
        out = resolve(Path(path))
        if not is_relative_to(out, resolve(ALLOWED_HOME_OUTPUT_ROOT)):
            log.fail(f"intended home output {key} outside allowed tree: {out}")
        if out.exists():
            log.warn(f"intended output already exists (resume/overwrite protocol will decide): {out}")
    for key, path in config["intended_heavy_outputs"].items():
        out = resolve(Path(path))
        if is_relative_to(out, resolve(FORBIDDEN_HOME_PREFIX)) or is_relative_to(out, resolve(FORBIDDEN_WORK_PREFIX)):
            log.fail(f"intended heavy output {key} not on scratch: {out}")
        if out.exists() and not out.is_dir():
            log.warn(f"intended heavy output already exists (resume/overwrite protocol will decide): {out}")
    return {
        "home_root": home_root,
        "heavy_root": heavy_root,
        "model": model,
        "stage03": stage03,
        "basis": basis,
        "scores": scores,
        "manifest_rows": manifest_rows,
        "manifest_sha": manifest_sha,
        "statics": statics,
        "chunks": chunks,
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
    print(f"PLAN stage={STAGE_NAME}")
    print(f"PLAN config_sha256={config_sha}")
    if state["model"] is not None:
        m = state["model"]
        print(f"PLAN selected model Kx={m['Kx']} Ky={m['Ky']} r={m['r']} alpha={m['ridge_alpha']}")
    print(f"PLAN chunks={len(state['chunks'])} (one per validation shard)")
    for chunk in state["chunks"]:
        print(f"PLAN chunk {chunk['chunk_index']}: days [{chunk['start']}, {chunk['end']})")
    print("PLAN heavy phase order: 1) eval-chunk (array over chunk indices)  2) reduce-eval (singleton --array=0-0)")
    for key, path in config["intended_outputs"].items():
        print(f"PLAN home output {key}: {path}")
    for key, path in config["intended_heavy_outputs"].items():
        print(f"PLAN heavy output {key}: {path}")
    spectra = config["spectra"]["enabled"]
    print(f"PLAN spectra enabled={spectra} (computed inside eval-chunk; reduce writes the spectra NPZ)" if spectra
          else "PLAN spectra DISABLED by config; reduce will record the deferral in stage04_status.json")
    env_set = os.environ.get(REAL_RUN_ENV) == "1"
    print(f"PLAN gates: production_execution_approved={config.get('production_execution_approved')} {REAL_RUN_ENV}_set={env_set}")
    print("PLAN units: shards/scores/EOFs normalized t2m_tgt units; error_K = error_norm * "
          f"{config['normalization']['target_std_K']}")
    print("DRY_RUN no outputs written; no shard/score/eofs_weighted data loaded (headers, hashes, small members only)")


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


def base_ident(config: dict[str, Any], config_sha: str, state: dict[str, Any]) -> dict[str, Any]:
    model = state["model"]
    stage03 = state["stage03"]
    return {
        "stage": STAGE_NAME,
        "mode": "production",
        "config_sha256": config_sha,
        "selected_model_path": str(model["path"]),
        "selected_model_sha256": model["sha256"],
        "selected_metadata_sha256": stage03["selected_metadata_json"]["sha256"],
        "grid_summary_sha256": stage03["grid_summary_json"]["sha256"],
        "metric_constants_sha256": stage03["metric_constants_json"]["sha256"],
        "y_basis_done_sha256": state["basis"]["done_sha256"],
        "val_x_done_sha256": state["scores"]["val_x"]["done_sha256"],
        "val_y_done_sha256": state["scores"]["val_y"]["done_sha256"],
        "source_manifest_sha256": state["manifest_sha"],
        "normalization": dict(config["normalization"]),
        "selected": {"Kx": model["Kx"], "Ky": model["Ky"], "r": model["r"], "ridge_alpha": model["ridge_alpha"]},
        "target_std_K": float(config["normalization"]["target_std_K"]),
        "n_val": int(config["expected_val_days"]),
        "grid_shape": list(config["expected_grid_shape"]),
        "n_pixels": int(config["expected_n_pixels"]),
        "spectra_enabled": bool(config["spectra"]["enabled"]),
    }


def chunk_ident(config: dict[str, Any], config_sha: str, state: dict[str, Any], chunk: dict[str, Any]) -> dict[str, Any]:
    height, width = config["expected_grid_shape"]
    ident = base_ident(config, config_sha, state)
    ident.update({
        "phase": "eval-chunk",
        "chunk_index": int(chunk["chunk_index"]),
        "day_start": int(chunk["start"]),
        "day_end": int(chunk["end"]),
        "output_shapes": {
            "day_arrays": [int(chunk["n_days"])],
            "maps": [height, width],
            "power_sums": [len(SPECTRA_FIELDS), width // 2 + 1] if config["spectra"]["enabled"] else None,
        },
    })
    return ident


def reduce_ident(config: dict[str, Any], config_sha: str, state: dict[str, Any], chunk_done_sha256s: dict[str, str]) -> dict[str, Any]:
    height, width = config["expected_grid_shape"]
    ident = base_ident(config, config_sha, state)
    ident.update({
        "phase": "reduce-eval",
        "n_chunks": len(state["chunks"]),
        "chunk_done_sha256s": chunk_done_sha256s,
        "statics_sha256s": {
            "surface_fractions_nc": state["statics"]["surface_fractions_sha256"],
            "orography_nc": state["statics"]["orography_sha256"],
        },
        "output_shapes": {"maps": [height, width], "n_days_total": int(config["expected_val_days"])},
    })
    return ident


def chunk_output_path(config: dict[str, Any], chunk: dict[str, Any]) -> Path:
    root = resolve(Path(config["intended_heavy_outputs"]["chunk_accumulators_dir"]))
    return root / f"chunk_{chunk['start']:05d}_{chunk['end']:05d}.npz"


def load_val_x_scores(state: dict[str, Any]):

    np = _np()
    entry = state["scores"]["val_x"]
    recorded = entry.get("file_sha256_recorded")
    if recorded:
        actual = sha256_file(entry["path"])
        if actual != recorded:
            raise RuntimeError(f"val_x score file sha256 mismatch: {actual} != done-recorded {recorded}")
    return np.asarray(np.load(entry["path"], allow_pickle=False), dtype=np.float64)


def run_eval_chunk(config: dict[str, Any], args, state: dict[str, Any], config_sha: str) -> None:
    gate_heavy(config, "eval-chunk")
    np = _np()
    if args.chunk_index is None:
        raise SystemExit("eval-chunk requires --chunk-index (from SLURM_ARRAY_TASK_ID)")
    chunks = state["chunks"]
    if not (0 <= args.chunk_index < len(chunks)):
        raise SystemExit(f"--chunk-index {args.chunk_index} out of range [0, {len(chunks)})")
    chunk = chunks[args.chunk_index]
    out_npz = chunk_output_path(config, chunk)
    ident = chunk_ident(config, config_sha, state, chunk)
    decision = output_state(out_npz, ident, resume=args.resume, overwrite=args.overwrite)
    if decision == "skip":
        print(f"EVAL_CHUNK_SKIPPED {out_npz} (done identity matches)")
        return
    if decision == "refuse":
        raise SystemExit(f"REFUSING to overwrite existing {out_npz} without matching done identity; pass --overwrite to replace")

    height, width = config["expected_grid_shape"]
    model = state["model"]
    start, end, n = chunk["start"], chunk["end"], chunk["n_days"]
    print(f"EVAL_CHUNK chunk={chunk['chunk_index']} days=[{start},{end}) n={n}", flush=True)

    x_scores = load_val_x_scores(state)
    yhat = predict_yhat_scores(x_scores[start:end], model)
    del x_scores
    mean_field = np.asarray(npz_small_member(state["basis"]["npz_path"], "mean_field.npy"), dtype=np.float64)
    w, sqrt_w = area_weights_from_lat(grid_lat_centers(height))
    eofs = npz_member_memmap(state["basis"]["npz_path"], "eofs_weighted.npy")
    base_arr = np.load(chunk["baseline_path"], mmap_mode="r")
    resid_arr = np.load(chunk["residual_path"], mmap_mode="r")
    if base_arr.shape != (n, height, width) or resid_arr.shape != (n, height, width):
        raise SystemExit(f"shard shape mismatch: baseline {base_arr.shape} residual {resid_arr.shape} expected {(n, height, width)}")

    t0 = time.time()
    acc = evaluate_days_dense(
        yhat64=yhat, base_arr=base_arr, resid_arr=resid_arr, eofs=eofs,
        ky=model["Ky"], mean_field64=mean_field, w=w, sqrt_w=sqrt_w,
        block_rows=int(config["algorithm"]["block_rows"]),
        spectra_enabled=bool(config["spectra"]["enabled"]),
    )
    print(f"EVAL_CHUNK dense pass done in {time.time() - t0:.1f}s", flush=True)

    arrays: dict[str, Any] = {
        "day_start": np.int64(start),
        "day_end": np.int64(end),
        "n_days": np.int64(n),
        "weight_row_sum": np.float64(acc["weight_row_sum"]),
    }
    for key in ACC_KEYS:
        arrays[f"day_wsum_{key}"] = acc["day"][key]
        arrays[f"map_sum_{key}"] = acc["maps"][key]
    if acc["power"] is not None:
        arrays["power_sums"] = acc["power"]
        arrays["spectra_fields"] = np.asarray(SPECTRA_FIELDS)
    atomic_savez(out_npz, **arrays)
    n_pixels = int(config["expected_n_pixels"])
    chunk_rmse_norm = math.sqrt(float(acc["day"]["err2_cca"].sum()) / (n * n_pixels))
    write_done(done_path(out_npz), ident, {
        "sha256": sha256_file(out_npz),
        "bytes": out_npz.stat().st_size,
        "chunk_cca_rmse_norm": chunk_rmse_norm,
        "chunk_baseline_rmse_norm": math.sqrt(float(acc["day"]["err2_base"].sum()) / (n * n_pixels)),
    })
    print(f"EVAL_CHUNK_DONE {out_npz} chunk_cca_rmse_norm={chunk_rmse_norm:.9f}")


def combine_chunks(config: dict[str, Any], config_sha: str, state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:

    np = _np()
    height, width = config["expected_grid_shape"]
    chunk_done_shas: dict[str, str] = {}
    day: dict[str, list] = {key: [] for key in ACC_KEYS}
    maps = {key: np.zeros((height, width), dtype=np.float64) for key in ACC_KEYS}
    power = None
    weight_row_sum = None
    total_days = 0
    for chunk in state["chunks"]:
        out_npz = chunk_output_path(config, chunk)
        ident = chunk_ident(config, config_sha, state, chunk)
        dpath = done_path(out_npz)
        if not out_npz.is_file() or not done_matches(dpath, ident):
            raise SystemExit(f"reduce-eval requires chunk {chunk['chunk_index']} with matching done identity: {out_npz}")
        payload = load_json(dpath)
        if payload.get("sha256") != sha256_file(out_npz):
            raise SystemExit(f"chunk accumulator sha256 mismatch (tampered/partial): {out_npz}")
        chunk_done_shas[str(chunk["chunk_index"])] = sha256_file(dpath)
        with np.load(out_npz, allow_pickle=False) as data:
            if int(data["day_start"]) != chunk["start"] or int(data["day_end"]) != chunk["end"]:
                raise SystemExit(f"chunk day range mismatch in {out_npz}")
            for key in ACC_KEYS:
                day[key].append(np.asarray(data[f"day_wsum_{key}"], dtype=np.float64))
                maps[key] += np.asarray(data[f"map_sum_{key}"], dtype=np.float64)
            if config["spectra"]["enabled"]:
                p = np.asarray(data["power_sums"], dtype=np.float64)
                power = p if power is None else power + p
            weight_row_sum = float(data["weight_row_sum"])
        total_days += chunk["n_days"]
    if total_days != int(config["expected_val_days"]):
        raise SystemExit(f"combined chunk days {total_days} != expected {config['expected_val_days']}")
    combined = {
        "day": {key: np.concatenate(day[key]) for key in ACC_KEYS},
        "maps": maps,
        "power": power,
        "weight_row_sum": weight_row_sum,
        "n_days": total_days,
    }
    return combined, chunk_done_shas


def global_metrics_from_combined(combined: dict[str, Any], *, n_pixels: int, std: float) -> dict[str, Any]:


    day = combined["day"]
    n = combined["n_days"]
    denom = float(n) * float(n_pixels)
    totals = {key: float(day[key].sum()) for key in ACC_KEYS}
    out: dict[str, Any] = {"n_days": n, "n_pixels": n_pixels, "weighted_denominator": denom}
    for tag, e2, ab, er in (("cca", "err2_cca", "abs_cca", "err_cca"), ("baseline", "err2_base", "abs_base", "err_base")):
        mse = totals[e2] / denom
        bias = totals[er] / denom
        out[f"{tag}_mse_norm2"] = mse
        out[f"{tag}_rmse_norm"] = math.sqrt(mse)
        out[f"{tag}_mae_norm"] = totals[ab] / denom
        out[f"{tag}_bias_norm"] = bias
        out[f"{tag}_error_std_norm"] = math.sqrt(max(mse - bias * bias, 0.0))
        out[f"{tag}_mse_K2"] = mse * std * std
        out[f"{tag}_rmse_K"] = math.sqrt(mse) * std
        out[f"{tag}_mae_K"] = out[f"{tag}_mae_norm"] * std
        out[f"{tag}_bias_K"] = bias * std
        out[f"{tag}_error_std_K"] = out[f"{tag}_error_std_norm"] * std
    out["skill_vs_bilinear_MSE"] = 1.0 - out["cca_mse_norm2"] / out["baseline_mse_norm2"]
    rt2, rh2, cross = totals["err2_base"], totals["rh2"], totals["cross"]
    out["target_residual_rms_norm"] = math.sqrt(rt2 / denom)
    out["predicted_residual_rms_norm"] = math.sqrt(rh2 / denom)
    out["target_residual_rms_K"] = out["target_residual_rms_norm"] * std
    out["predicted_residual_rms_K"] = out["predicted_residual_rms_norm"] * std
    out["correction_corr"] = cross / math.sqrt(rh2 * rt2) if rh2 > 0.0 and rt2 > 0.0 else float("nan")
    out["amplitude_ratio"] = math.sqrt(rh2 / rt2) if rt2 > 0.0 else float("nan")
    return out


def consistency_checks(config: dict[str, Any], state: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:

    tol = config["consistency"]
    constants = state["stage03"]["constants"]
    sel = state["stage03"]["selected_val_metrics"]
    std = float(config["normalization"]["target_std_K"])
    n_pixels = int(config["expected_n_pixels"])
    n_val = int(config["expected_val_days"])
    denom = float(n_val) * float(n_pixels)

    def check(name: str, expected: float, actual: float, rel_tol: float, note: str) -> dict[str, Any]:
        rel_dev = abs(actual - expected) / abs(expected) if expected != 0.0 else abs(actual)
        return {
            "name": name, "expected": expected, "actual": actual,
            "rel_dev": rel_dev, "rel_tol": rel_tol, "pass": bool(rel_dev <= rel_tol),
            "note": note,
        }

    expected_baseline_rmse_norm = math.sqrt(constants["val_raw_weighted_sse"] / denom)
    checks = [
        check("baseline_rmse_norm_vs_metric_constants", expected_baseline_rmse_norm,
              metrics["baseline_rmse_norm"], float(tol["baseline_rmse_rel_tol"]),
              "dense bilinear-baseline RMSE (normalized) vs sqrt(val_raw_weighted_sse/(n*P)) from Stage 03 metric_constants.json"),
        check("cca_rmse_norm_vs_stage03_selected", float(sel["val_rmse_norm"]),
              metrics["cca_rmse_norm"], float(tol["cca_rmse_rel_tol"]),
              "dense selected-CCA RMSE (normalized) vs Stage 03 score-space val_rmse_norm (exact orthonormal SSE identity)"),
        check("cca_rmse_K_vs_stage03_selected", float(sel["val_rmse_K_physical"]),
              metrics["cca_rmse_K"], float(tol["cca_rmse_rel_tol"]),
              "dense selected-CCA RMSE (physical K) vs Stage 03 val_rmse_K_physical"),
        check("skill_vs_stage03_selected", float(sel["val_skill_vs_bilinear"]),
              metrics["skill_vs_bilinear_MSE"], float(tol["skill_rel_tol"]),
              "dense MSE skill vs Stage 03 val_skill_vs_bilinear (scale-invariant)"),
        check("stage03_physical_equals_norm_times_std", float(sel["val_rmse_norm"]) * std,
              float(sel["val_rmse_K_physical"]), float(tol["physical_conversion_rel_tol"]),
              "Stage 03 unit patch self-consistency: val_rmse_K_physical == val_rmse_norm * target_std_K"),
        check("dense_physical_equals_norm_times_std", metrics["cca_rmse_norm"] * std,
              metrics["cca_rmse_K"], float(tol["physical_conversion_rel_tol"]),
              "Stage 04 conversion self-consistency: rmse_K == rmse_norm * target_std_K"),
    ]
    return {
        "created_utc": utc_now(),
        "reconstruction_formula": (
            "residual_hat_norm = y_basis_mean_field + (yhat_scores @ eofs_weighted[:Ky].reshape(Ky, P)) / sqrt_weight; "
            "yhat_scores = (val_x_scores[:, :Kx] - x_score_mean) @ B + y_score_mean"
        ),
        "sealed_test_untouched": True,
        "checks": checks,
        "all_pass": all(c["pass"] for c in checks),
        "max_rel_dev": max(c["rel_dev"] for c in checks),
    }


def regional_rows_from_maps(combined, masks, w, *, n_days: int, std: float) -> list[dict[str, Any]]:


    np = _np()
    maps = combined["maps"]
    height, width = maps["err2_cca"].shape
    w2d = np.broadcast_to(np.asarray(w, dtype=np.float64)[:, None], (height, width))
    rows = []
    for region in REGION_ORDER:
        mask = masks[region]
        wm = w2d * mask
        wsum = float(wm.sum())
        n_pix = int(mask.sum())
        if wsum <= 0.0:
            rows.append({"region": region, "n_pixels": n_pix, "weight_fraction": 0.0})
            continue
        denom = float(n_days) * wsum

        def wtot(key: str) -> float:
            return float((wm * maps[key]).sum())

        mse_cca = wtot("err2_cca") / denom
        mse_base = wtot("err2_base") / denom
        rt2, rh2, cross = wtot("err2_base"), wtot("rh2"), wtot("cross")
        rows.append({
            "region": region,
            "area_weighted_rmse_K": math.sqrt(mse_cca) * std,
            "area_weighted_skill": 1.0 - mse_cca / mse_base if mse_base > 0.0 else float("nan"),
            "area_weighted_bias_K": wtot("err_cca") / denom * std,
            "area_weighted_mae_K": wtot("abs_cca") / denom * std,
            "baseline_rmse_K": math.sqrt(mse_base) * std,
            "baseline_mae_K": wtot("abs_base") / denom * std,
            "baseline_bias_K": wtot("err_base") / denom * std,
            "rmse_improvement_K": (math.sqrt(mse_base) - math.sqrt(mse_cca)) * std,
            "correction_corr": cross / math.sqrt(rh2 * rt2) if rh2 > 0.0 and rt2 > 0.0 else float("nan"),
            "amplitude_ratio": math.sqrt(rh2 / rt2) if rt2 > 0.0 else float("nan"),
            "n_pixels": n_pix,
            "weight_fraction": wsum / float(height * width),
            "cca_rmse_norm": math.sqrt(mse_cca),
            "baseline_rmse_norm": math.sqrt(mse_base),
        })
    return rows


def daily_rows_from_combined(combined, chunks, *, n_pixels: int, std: float) -> list[dict[str, Any]]:
    day = combined["day"]
    shard_of_day: list[tuple[str, int]] = []
    for chunk in chunks:
        for i in range(chunk["n_days"]):
            shard_of_day.append((str(chunk["task_id"]), i))
    rows = []
    for d in range(combined["n_days"]):
        p = float(n_pixels)
        mse_cca = float(day["err2_cca"][d]) / p
        mse_base = float(day["err2_base"][d]) / p
        rh2, rt2, cross = float(day["rh2"][d]), float(day["err2_base"][d]), float(day["cross"][d])
        rows.append({
            "day_index": d,
            "shard_task_id": shard_of_day[d][0],
            "day_in_shard": shard_of_day[d][1],
            "baseline_rmse_K": math.sqrt(mse_base) * std,
            "cca_rmse_K": math.sqrt(mse_cca) * std,
            "baseline_mae_K": float(day["abs_base"][d]) / p * std,
            "cca_mae_K": float(day["abs_cca"][d]) / p * std,
            "baseline_bias_K": float(day["err_base"][d]) / p * std,
            "cca_bias_K": float(day["err_cca"][d]) / p * std,
            "mse_skill_vs_bilinear": 1.0 - mse_cca / mse_base if mse_base > 0.0 else float("nan"),
            "baseline_rmse_norm": math.sqrt(mse_base),
            "cca_rmse_norm": math.sqrt(mse_cca),
            "correction_corr": cross / math.sqrt(rh2 * rt2) if rh2 > 0.0 and rt2 > 0.0 else float("nan"),
        })
    return rows


def run_reduce_eval(config: dict[str, Any], args, state: dict[str, Any], config_sha: str) -> None:
    gate_heavy(config, "reduce-eval")
    np = _np()
    height, width = config["expected_grid_shape"]
    n_pixels = int(config["expected_n_pixels"])
    std = float(config["normalization"]["target_std_K"])
    outputs = {key: resolve(Path(path)) for key, path in config["intended_outputs"].items()}
    heavy_outputs = {key: resolve(Path(path)) for key, path in config["intended_heavy_outputs"].items()}

    combined, chunk_done_shas = combine_chunks(config, config_sha, state)
    ident = reduce_ident(config, config_sha, state, chunk_done_shas)
    primary = outputs["selected_cca_validation_metrics_json"]
    decision = output_state(primary, ident, resume=args.resume, overwrite=args.overwrite)
    if decision == "skip":
        print(f"REDUCE_EVAL_SKIPPED {primary} (done identity matches)")
        return
    if decision == "refuse":
        raise SystemExit(f"REFUSING to overwrite existing {primary} without matching done identity; pass --overwrite to replace")

    metrics = global_metrics_from_combined(combined, n_pixels=n_pixels, std=std)


    w, _ = area_weights_from_lat(grid_lat_centers(height))
    w2d = np.broadcast_to(w[:, None], (height, width))
    for key in ("err2_cca", "err2_base"):
        from_maps = float((w2d * combined["maps"][key]).sum())
        from_days = float(combined["day"][key].sum())
        rel = abs(from_maps - from_days) / from_days
        if rel > 1e-9:
            raise SystemExit(f"internal accumulator mismatch for {key}: maps vs days rel dev {rel:.3e}")

    consistency = consistency_checks(config, state, metrics)
    write_json_atomic(outputs["dense_vs_stage03_consistency_json"], {"ident": ident, **consistency})
    for chk in consistency["checks"]:
        status = "PASS" if chk["pass"] else "FAIL"
        print(f"CONSISTENCY {status} {chk['name']} expected={chk['expected']:.12g} actual={chk['actual']:.12g} rel_dev={chk['rel_dev']:.3e} tol={chk['rel_tol']:.1e}")
    if not consistency["all_pass"]:
        write_json_atomic(outputs["stage04_status_json"], {
            "stage": STAGE_NAME, "status": "FAILED_CONSISTENCY", "created_utc": utc_now(),
            "config_sha256": config_sha,
            "detail": "dense metrics do not match Stage 03 within tolerance; see dense_vs_stage03_consistency.json",
        })
        raise SystemExit("REDUCE_EVAL FAILED: dense-vs-Stage03 consistency violated; final outputs NOT written")

    statics_fields = load_statics_fields(config, state["statics"])
    masks = region_masks(state["statics"]["lat"], statics_fields["lsm"], statics_fields["oro_m"])
    n_days = combined["n_days"]
    maps = combined["maps"]


    inv_n = 1.0 / float(n_days)
    spatial_path = heavy_outputs["spatial_maps_npz"]
    spatial_arrays = {
        "lat": np.asarray(state["statics"]["lat"], dtype=np.float64),
        "lon": np.asarray(state["statics"]["lon"], dtype=np.float64),
        "lsm": statics_fields["lsm"],
        "cl": statics_fields["cl"],
        "orography": statics_fields["oro_m"],
        "n_days": np.int64(n_days),

        "baseline_rmse_K": (np.sqrt(maps["err2_base"] * inv_n) * std).astype(np.float32),
        "cca_rmse_K": (np.sqrt(maps["err2_cca"] * inv_n) * std).astype(np.float32),
        "baseline_mae_K": (maps["abs_base"] * inv_n * std).astype(np.float32),
        "cca_mae_K": (maps["abs_cca"] * inv_n * std).astype(np.float32),
        "baseline_bias_K": (maps["err_base"] * inv_n * std).astype(np.float32),
        "cca_bias_K": (maps["err_cca"] * inv_n * std).astype(np.float32),
        "baseline_error_mean_K": (maps["err_base"] * inv_n * std).astype(np.float32),
        "cca_error_mean_K": (maps["err_cca"] * inv_n * std).astype(np.float32),
        "mse_skill_vs_bilinear": (1.0 - maps["err2_cca"] / np.maximum(maps["err2_base"], 1e-300)).astype(np.float32),
        "rmse_improvement_K": ((np.sqrt(maps["err2_base"] * inv_n) - np.sqrt(maps["err2_cca"] * inv_n)) * std).astype(np.float32),
        "target_residual_rms_K": (np.sqrt(maps["err2_base"] * inv_n) * std).astype(np.float32),
        "predicted_residual_rms_K": (np.sqrt(maps["rh2"] * inv_n) * std).astype(np.float32),

        "baseline_error_sumsq_K2": maps["err2_base"] * (std * std),
        "cca_error_sumsq_K2": maps["err2_cca"] * (std * std),
        "baseline_error_sum_K": maps["err_base"] * std,
        "cca_error_sum_K": maps["err_cca"] * std,
        "units_note": np.asarray(
            "error_K = error_norm * target_std_K; baseline error == -residual so baseline_rmse_K == target_residual_rms_K; "
            "bias maps == error mean maps (kept under both required names)"
        ),
        "target_std_K": np.float64(std),
    }
    atomic_savez(spatial_path, **spatial_arrays)
    write_done(done_path(spatial_path), ident, {"sha256": sha256_file(spatial_path), "bytes": spatial_path.stat().st_size,
                                                "kind": "spatial_maps"})
    print(f"REDUCE_EVAL spatial maps -> {spatial_path}")


    spectra_enabled = bool(config["spectra"]["enabled"])
    spectra_path = heavy_outputs["spectra_npz"]
    if spectra_enabled:
        power = combined["power"]
        weight_row_sum = combined["weight_row_sum"]
        norm_factor = 1.0 / (float(n_days) * weight_row_sum)
        power_mean = power * norm_factor
        power_k2 = power_mean * (std * std)
        idx = {name: i for i, name in enumerate(SPECTRA_FIELDS)}
        base_err = power_k2[idx["baseline_error"]]
        model_err = power_k2[idx["model_error"]]
        spectral_skill = np.where(base_err > 0.0, 1.0 - model_err / np.maximum(base_err, 1e-300), np.nan)
        atomic_savez(
            spectra_path,
            wavenumber=np.arange(width // 2 + 1, dtype=np.int64),
            target_power_K2=power_k2[idx["target"]],
            baseline_power_K2=power_k2[idx["baseline"]],
            prediction_power_K2=power_k2[idx["prediction"]],
            model_error_power_K2=power_k2[idx["model_error"]],
            baseline_error_power_K2=power_k2[idx["baseline_error"]],
            spectral_skill_vs_bilinear=spectral_skill,
            sample_count=np.int64(n_days),
            grid_width=np.int64(width),
            weight_row_sum=np.float64(weight_row_sum),
            target_std_K=np.float64(std),
            convention=np.asarray(
                "U-Net SpectraAccumulator convention: coeff = rfft(field, axis=lon)/width; power = |coeff|^2; "
                "cosine-latitude weighted mean over rows (weights cos(lat)/mean, ratio-normalized); mean over validation days; "
                "x target_std_K^2 for K^2. Fields are normalized t2m_tgt anomalies, so the k=0 bin is the power of the "
                "normalized anomaly zonal mean (same DC convention as the U-Net artifacts)."
            ),
        )
        write_done(done_path(spectra_path), ident, {"sha256": sha256_file(spectra_path), "bytes": spectra_path.stat().st_size,
                                                    "kind": "spectra"})
        print(f"REDUCE_EVAL spectra -> {spectra_path}")
    else:
        print("REDUCE_EVAL spectra DISABLED by config; deferral recorded in stage04_status.json")


    regional_rows = regional_rows_from_maps(combined, masks, w, n_days=n_days, std=std)
    write_csv_atomic(outputs["regional_performance_summary_csv"], REGIONAL_COLUMNS, regional_rows)

    daily_rows = daily_rows_from_combined(combined, state["chunks"], n_pixels=n_pixels, std=std)
    write_csv_atomic(outputs["time_series_daily_metrics_csv"], DAILY_COLUMNS, daily_rows)

    model = state["model"]
    model_label = f"selected CCA (Kx={model['Kx']}, Ky={model['Ky']}, r={model['r']}, alpha={model['ridge_alpha']:g})"
    summary_columns = ["model_or_reference", "step", "area_weighted_mse_norm", "area_weighted_rmse_K",
                       "area_weighted_mae_K", "area_weighted_bias_K", "area_weighted_skill",
                       "area_weighted_corr_corr", "correction_corr"]
    write_csv_atomic(outputs["validation_summary_csv"], summary_columns, [
        {"model_or_reference": "bilinear baseline", "step": "reference",
         "area_weighted_mse_norm": metrics["baseline_mse_norm2"], "area_weighted_rmse_K": metrics["baseline_rmse_K"],
         "area_weighted_mae_K": metrics["baseline_mae_K"], "area_weighted_bias_K": metrics["baseline_bias_K"],
         "area_weighted_skill": 0.0, "area_weighted_corr_corr": "NA", "correction_corr": "NA"},
        {"model_or_reference": model_label, "step": "selected",
         "area_weighted_mse_norm": metrics["cca_mse_norm2"], "area_weighted_rmse_K": metrics["cca_rmse_K"],
         "area_weighted_mae_K": metrics["cca_mae_K"], "area_weighted_bias_K": metrics["cca_bias_K"],
         "area_weighted_skill": metrics["skill_vs_bilinear_MSE"],
         "area_weighted_corr_corr": metrics["correction_corr"], "correction_corr": metrics["correction_corr"]},
    ])

    key_rows = [
        ("Stage", STAGE_NAME),
        ("Model", model_label),
        ("Validation days", n_days),
        ("Grid", f"{height}x{width}"),
        ("Validation CCA RMSE (K)", f"{metrics['cca_rmse_K']:.6f}"),
        ("Validation CCA MSE (K^2)", f"{metrics['cca_mse_K2']:.8e}"),
        ("Validation CCA MAE (K)", f"{metrics['cca_mae_K']:.6f}"),
        ("Validation CCA bias (K)", f"{metrics['cca_bias_K']:.6e}"),
        ("Validation CCA error std (K)", f"{metrics['cca_error_std_K']:.6f}"),
        ("Baseline (bilinear) RMSE (K)", f"{metrics['baseline_rmse_K']:.6f}"),
        ("Baseline (bilinear) MAE (K)", f"{metrics['baseline_mae_K']:.6f}"),
        ("Skill vs bilinear (MSE)", f"{metrics['skill_vs_bilinear_MSE']:.6f}"),
        ("RMSE improvement (K)", f"{metrics['baseline_rmse_K'] - metrics['cca_rmse_K']:.6f}"),
        ("Correction correlation", f"{metrics['correction_corr']:.6f}"),
        ("Amplitude ratio (pred/required residual RMS)", f"{metrics['amplitude_ratio']:.6f}"),
        ("Validation CCA RMSE (normalized)", f"{metrics['cca_rmse_norm']:.9f}"),
        ("Baseline RMSE (normalized)", f"{metrics['baseline_rmse_norm']:.9f}"),
        ("target_std_K (norm->K scale)", f"{std!r}"),
        ("Dense-vs-Stage03 max rel dev", f"{consistency['max_rel_dev']:.3e}"),
        ("Caveat", "validation-selected model; these are selection-split metrics, not sealed-test claims"),
    ]
    write_csv_atomic(outputs["key_metrics_csv"], ["metric", "value"], [{"metric": k, "value": v} for k, v in key_rows])

    metrics_payload = {
        "ident": ident,
        "created_utc": utc_now(),
        "units": {
            "normalized": "t2m_tgt normalized units (anomaly / target_std_K); suffix _norm / _norm2",
            "physical": "Kelvin via error_K = error_norm * target_std_K; suffix _K / _K2",
            "target_std_K": std,
        },
        "global_metrics": metrics,
        "consistency": consistency,
        "regional_summary": regional_rows,
        "spectra": {"enabled": spectra_enabled, "path": str(spectra_path) if spectra_enabled else None},
        "spatial_maps_path": str(spatial_path),
        "caveat": "validation-selected model evaluated on the same validation split; sealed test untouched",
    }
    write_json_atomic(primary, metrics_payload)
    write_done(done_path(primary), ident, {"sha256": sha256_file(primary)})

    write_json_atomic(outputs["normalization_audit_json"], {
        "ident": ident,
        "created_utc": utc_now(),
        "normalization_stats": dict(config["normalization"]),
        "conversion": "error_K = error_norm * target_std_K; MSE_K2 = MSE_norm2 * target_std_K^2; skill/correlations scale-invariant",
        "audit": {
            "baseline_rmse_norm_dense": metrics["baseline_rmse_norm"],
            "baseline_rmse_K_dense": metrics["baseline_rmse_K"],
            "expected_baseline_rmse_norm_from_stage03_constants": math.sqrt(
                state["stage03"]["constants"]["val_raw_weighted_sse"] / (float(n_days) * float(n_pixels))),
            "target_residual_rms_norm": metrics["target_residual_rms_norm"],
            "predicted_residual_rms_norm": metrics["predicted_residual_rms_norm"],
            "shard_units_note": "validation residual day RMS ~0.036 and baseline day RMS ~1.0 in normalized units "
                                "(scale audit from the run brief); dense values above confirm the normalized convention",
        },
    })

    heavy_manifest_entries = []
    for kind, path in (("spatial_maps", spatial_path), ("spectra", spectra_path if spectra_enabled else None)):
        if path is None:
            continue
        heavy_manifest_entries.append({
            "kind": kind, "path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path), "done_json": str(done_path(path)),
        })
    for chunk in state["chunks"]:
        cpath = chunk_output_path(config, chunk)
        heavy_manifest_entries.append({
            "kind": "chunk_accumulator", "path": str(cpath), "bytes": cpath.stat().st_size,
            "chunk_index": chunk["chunk_index"], "done_json": str(done_path(cpath)),
        })
    write_json_atomic(outputs["heavy_artifact_manifest_json"], {
        "ident": ident,
        "created_utc": utc_now(),
        "entries": heavy_manifest_entries,
        "scratch_lifetime_warning": "scratch has a ~14-day lifetime since last access and is not archival; "
                                    "preserve spatial/spectra NPZs before they expire if needed for the report",
    })

    write_json_atomic(outputs["stage04_status_json"], {
        "stage": STAGE_NAME,
        "status": "complete",
        "created_utc": utc_now(),
        "config_sha256": config_sha,
        "chunks_reduced": len(state["chunks"]),
        "n_val_days": n_days,
        "consistency_all_pass": True,
        "consistency_max_rel_dev": consistency["max_rel_dev"],
        "spectra": {"enabled": spectra_enabled,
                    "note": "computed in eval-chunk, reduced here" if spectra_enabled
                    else "DEFERRED by config spectra.enabled=false; no spectra NPZ was written"},
        "selected": {"Kx": model["Kx"], "Ky": model["Ky"], "r": model["r"], "ridge_alpha": model["ridge_alpha"]},
        "headline": {
            "cca_rmse_K": metrics["cca_rmse_K"],
            "baseline_rmse_K": metrics["baseline_rmse_K"],
            "skill_vs_bilinear_MSE": metrics["skill_vs_bilinear_MSE"],
        },
    })
    print(f"REDUCE_EVAL_DONE cca_rmse_K={metrics['cca_rmse_K']:.6f} baseline_rmse_K={metrics['baseline_rmse_K']:.6f} "
          f"skill={metrics['skill_vs_bilinear_MSE']:.6f} -> {primary}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("plan",) + HEAVY_PHASES, default="plan")
    parser.add_argument("--dry-run", action="store_true", help="alias for --phase plan")
    parser.add_argument("--chunk-index", type=int, default=None, help="validation chunk index for eval-chunk (SLURM_ARRAY_TASK_ID)")
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

    if phase == 'eval-chunk':
        run_eval_chunk(config, args, state, config_sha)
    elif phase == 'reduce-eval':
        run_reduce_eval(config, args, state, config_sha)
    print_issues(log)
    return 0 if log.ok() else 2


if __name__ == "__main__":
    sys.exit(main())
