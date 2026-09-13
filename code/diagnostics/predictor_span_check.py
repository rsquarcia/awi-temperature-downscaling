"""Inspect whether climate-shift patterns are represented in the frozen predictor span.
Uses retained fields, model partials and EOF bases to project and reconstruct
shift patterns, check numerical consistency and relate them to saved error
components. Writes diagnostic tables and summaries without running inference,
fitting a new basis, recalibrating a model or bootstrapping new intervals.
Requires the external predictor and mechanism-run artifacts."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import platform
import struct
import sys
import time
import zipfile
from pathlib import Path

import numpy as np


DATA_ROOT = os.environ.get("DATA_ROOT", "data_root")                                                     
RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "results_root")                                                  


ROOT_EXPECTED = Path(
    f"{RESULTS_ROOT}/"
    "focused_mechanism_triage_20260806T140607Z"
)
MECH = Path(
    f"{RESULTS_ROOT}/"
    "ssp585_pd_mh_full_mechanism_20260802T095350Z"
)
UNIFORM_RUN = Path(
    f"{RESULTS_ROOT}/uniform_shift_rmse_20260805T131319Z"
)
CCA_HOME = Path(
    f"{RESULTS_ROOT}/cca_final_run"
)
CCA_HEAVY = Path(
    f"{RESULTS_ROOT}/cca_final_run/"
    "outputs_heavy"
)
CCA_MODEL = CCA_HOME / "outputs/cca_grid/selected_cca_model.npz"
X_BASIS = CCA_HEAVY / "eof_bases/x_input_eof_basis_maxK10592.npz"
Y_BASIS = CCA_HEAVY / "eof_bases/y_residual_eof_basis_maxK1024.npz"
X_BLOCKS = CCA_HEAVY / "eof_bases/stage01_work/x_input/eof_blocks"
NORM_STATS = Path(f"{DATA_ROOT}/grids/norm_stats.json")
W_HR2LR = Path(f"{DATA_ROOT}/grids/weights_hr2lr.nc")
W_LR2HR = Path(f"{DATA_ROOT}/grids/weights_lr2hr.nc")
ACCEPTED_BVF = Path(
    f"{RESULTS_ROOT}/transfer_mechanism_audit_20260805T114036Z/"
    "angle_audit_and_decomposition.json"
)
PAPER_RENDERER = Path(
    f"{RESULTS_ROOT}/paper_figures_final_review_20260803T142934Z/"
    "make_transfer_figures.py"
)
CARTOPY_DIR = Path(
    f"{RESULTS_ROOT}/full_validation_tools_20260712/"
    "cartopy_data_20260712T200905Z"
)

H, W = 1280, 2624
NPIX = H * W
KX, KY = 10592, 512
N_DAYS = 1096
TS = 21.627892139211994
TM = 277.8518781731243
IS = 21.581598298286263
IM = 277.8568272051984
CLIMATES = ("pd", "mh", "ssp")
POP_FOR_CLIMATE = {"pd": "test", "mh": "mh", "ssp": "ssp"}
CLIMATE_LABEL = {
    "pd": "PD test",
    "mh": "Mid-Holocene",
    "ssp": "SSP5-8.5",
}
ZARRS = {
    "pd": Path(f"{DATA_ROOT}/zarr/awi_downscaling_test.zarr"),
    "mh": Path(
        f"{RESULTS_ROOT}/mh_preprocess_20260728T173229Z/"
        "results/mh_2076_2078.zarr"
    ),
    "ssp": Path(
        f"{RESULTS_ROOT}/n43_ssp585_final_20260729T130103Z/"
        "results/n43_ssp585_2096_2098.zarr"
    ),
}
PRED_ROOTS = {
    "pd": Path(
        f"{RESULTS_ROOT}/"
        "paper_pipeline_pd_test_r2_20260728T130323Z"
    ),
    "mh": Path(
        f"{RESULTS_ROOT}/mh_full_run_20260728T180741Z"
    ),
    "ssp": Path(
        f"{RESULTS_ROOT}/ssp585_full_run_20260729T130103Z"
    ),
}
X_SCORE_PATHS = {
    "pd": PRED_ROOTS["pd"] / "scores/test_x_scores_maxK10592.npy",
    "mh": PRED_ROOTS["mh"] / "scores/mh_x_scores_maxK10592.npy",
    "ssp": PRED_ROOTS["ssp"] / "scores/ssp585_x_scores_maxK10592.npy",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_dump(path: Path, obj) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(obj, indent=2, sort_keys=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("x", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def npz_member_info(path: Path, member: str):
    if not member.endswith(".npy"):
        member += ".npy"
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise RuntimeError(f"{path}:{member} is not ZIP_STORED")
        with path.open("rb") as fh:
            fh.seek(info.header_offset)
            header = fh.read(30)
            name_len, extra_len = struct.unpack("<HH", header[26:30])
            payload = info.header_offset + 30 + name_len + extra_len
            fh.seek(payload)
            if fh.read(6) != b"\x93NUMPY":
                raise RuntimeError(f"bad NPY magic in {path}:{member}")
            major, minor = fh.read(2)
            if major == 1:
                hlen = struct.unpack("<H", fh.read(2))[0]
            elif major in (2, 3):
                hlen = struct.unpack("<I", fh.read(4))[0]
            else:
                raise RuntimeError(f"unsupported NPY {major}.{minor}")
            meta = ast.literal_eval(fh.read(hlen).decode("latin1"))
            data_offset = fh.tell()
    return member, meta, data_offset


def npz_member_memmap(path: Path, member: str) -> np.memmap:
    _, meta, offset = npz_member_info(path, member)
    order = "F" if meta["fortran_order"] else "C"
    return np.memmap(path, mode="r", dtype=np.dtype(meta["descr"]),
                     shape=tuple(meta["shape"]), offset=offset, order=order)


def npz_small(path: Path, member: str) -> np.ndarray:
    if not member.endswith(".npy"):
        member += ".npy"
    with zipfile.ZipFile(path) as zf, zf.open(member) as fh:
        return np.load(fh, allow_pickle=False)


LAT = -90.0 + (np.arange(H, dtype=np.float64) + 0.5) * (180.0 / H)
LON = (np.arange(W, dtype=np.float64) + 0.5) * (360.0 / W)
W_ROW = np.cos(np.deg2rad(LAT))
W_ROW /= W_ROW.mean()
SQRT_W_ROW = np.sqrt(W_ROW)
W_FLAT = np.repeat(W_ROW, W)
SQRT_W_FLAT = np.repeat(SQRT_W_ROW, W)
W_SUM = float(W_FLAT.sum())


def wmean(field: np.ndarray) -> float:
    return float(np.dot(np.asarray(field, dtype=np.float64).reshape(-1),
                        W_FLAT) / W_SUM)


def wmeansq(field: np.ndarray) -> float:
    a = np.asarray(field, dtype=np.float64).reshape(-1)
    return float(np.dot(a * a, W_FLAT) / W_SUM)


def wrms(field: np.ndarray) -> float:
    return math.sqrt(max(wmeansq(field), 0.0))


def wstd(field: np.ndarray) -> float:
    a = np.asarray(field, dtype=np.float64)
    mu = wmean(a)
    return math.sqrt(max(wmeansq(a - mu), 0.0))


def wcorr(field_a: np.ndarray, field_b: np.ndarray) -> float:
    a = np.asarray(field_a, dtype=np.float64).reshape(-1)
    b = np.asarray(field_b, dtype=np.float64).reshape(-1)
    ac = a - np.dot(a, W_FLAT) / W_SUM
    bc = b - np.dot(b, W_FLAT) / W_SUM
    cov = float(np.dot(ac * bc, W_FLAT) / W_SUM)
    va = float(np.dot(ac * ac, W_FLAT) / W_SUM)
    vb = float(np.dot(bc * bc, W_FLAT) / W_SUM)
    return cov / math.sqrt(va * vb) if va > 0 and vb > 0 else float("nan")


def quantile_summary(a: np.ndarray, prefix: str) -> dict:
    x = np.asarray(a, dtype=np.float64)
    return {
        f"{prefix}_mean": float(x.mean()),
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_p95": float(np.percentile(x, 95)),
        f"{prefix}_maximum": float(x.max()),
    }


def load_model_partials_and_gate() -> tuple[dict, dict]:

    accepted = json.loads(ACCEPTED_BVF.read_text(encoding="utf-8"))
    out = {}
    gate = {}
    for climate in CLIMATES:
        pop = POP_FOR_CLIMATE[climate]
        parts = [np.load(MECH / "partials" / f"model_{pop}_{i:02d}.npz",
                         allow_pickle=False) for i in range(5)]
        order = np.argsort([int(z["lo"]) for z in parts])
        parts = [parts[i] for i in order]
        day_index = np.concatenate([z["day_index"] for z in parts])
        if not np.array_equal(day_index, np.arange(N_DAYS)):
            raise RuntimeError(f"{pop}: retained model partial day cover failed")
        st = np.concatenate([z["s_true"] for z in parts]).astype(np.float64)
        sc = np.concatenate([z["s_cca"] for z in parts]).astype(np.float64)
        su = np.concatenate([z["s_unet"] for z in parts]).astype(np.float64)
        daily = np.concatenate([z["daily"] for z in parts], axis=0)
        quantities = list(parts[0]["quantities"].astype(str))
        regions = list(parts[0]["regions"].astype(str))
        q = {name: i for i, name in enumerate(quantities)}
        gi = regions.index("global")
        weight = float(daily[:, gi, q["w"]].sum())
        f = lambda name: float(daily[:, gi, q[name]].sum() / weight)
        coef_scale = TS / math.sqrt(W_SUM)
        ec = (sc - st) * coef_scale
        eu = (su - st) * coef_scale
        b_c = float(np.dot(ec.mean(axis=0), ec.mean(axis=0)))
        v_c = float(np.mean(np.einsum("ij,ij->i",
                                     ec - ec.mean(axis=0),
                                     ec - ec.mean(axis=0))))
        ref = accepted["components"][pop]
        g = {
            "n_days": int(st.shape[0]),
            "accepted_CCA_B_K2": float(ref["B_K2"]),
            "reproduced_CCA_B_K2": b_c,
            "CCA_B_abs_deviation_K2": abs(b_c - float(ref["B_K2"])),
            "accepted_CCA_V_K2": float(ref["V_K2"]),
            "reproduced_CCA_V_K2": v_c,
            "CCA_V_abs_deviation_K2": abs(v_c - float(ref["V_K2"])),
            "accepted_CCA_F_K2": float(ref["F_K2"]),
            "reproduced_CCA_F_K2": f("Qe_cca2"),
            "CCA_F_abs_deviation_K2": abs(
                f("Qe_cca2") - float(ref["F_K2"])),
            "accepted_UNet_inside_K2": f("Pe_unet2"),
            "accepted_UNet_outside_K2": f("Qe_unet2"),
            "CCA_total_MSE_K2": f("e_cca2"),
            "UNet_total_MSE_K2": f("e_unet2"),
            "CCA_target_split_closure_K2": (
                f("e_cca2") - f("Pe_cca2") - f("Qe_cca2")),
            "UNet_target_split_closure_K2": (
                f("e_unet2") - f("Pe_unet2") - f("Qe_unet2")),
        }
        if max(g["CCA_B_abs_deviation_K2"], g["CCA_V_abs_deviation_K2"],
               g["CCA_F_abs_deviation_K2"]) > 1e-9:
            raise RuntimeError(f"{climate}: accepted CCA B/V/F gate failed: {g}")
        out[climate] = {
            "st": st, "sc": sc, "su": su, "ec": ec, "eu": eu,
            "accepted": {
                "cca_total": f("e_cca2"),
                "cca_inside": f("Pe_cca2"),
                "cca_outside": f("Qe_cca2"),
                "unet_total": f("e_unet2"),
                "unet_inside": f("Pe_unet2"),
                "unet_outside": f("Qe_unet2"),
            },
        }
        gate[climate] = g
        for z in parts:
            z.close()
        log(f"GATE target decomposition {climate}: "
            f"CCA B={b_c:.10g} V={v_c:.10g} F={f('Qe_cca2'):.10g}")
    return out, gate


def baseline_target_norm(input_norm: np.ndarray) -> np.ndarray:

    scale = IS / TS
    offset = (IM - TM) / TS
    return (np.asarray(input_norm, dtype=np.float64) * scale + offset).astype(
        np.float32)


def scan_predictors(x_mean: np.ndarray) -> tuple[dict, dict]:

    import zarr

    raw = {}
    means = {}
    mean_flat = np.asarray(x_mean, dtype=np.float32).reshape(-1)
    sw32 = SQRT_W_FLAT.astype(np.float32)
    for climate in CLIMATES:
        group = zarr.open_group(str(ZARRS[climate]), mode="r")
        if group["inputs"].shape[0] != N_DAYS:
            raise RuntimeError(f"{climate}: expected {N_DAYS} days")
        totals = np.zeros(N_DAYS, dtype=np.float64)
        acc = np.zeros(NPIX, dtype=np.float64)
        first_a = None
        for day in range(N_DAYS):
            base = baseline_target_norm(group["inputs"][day, 0]).reshape(-1)
            acc += base.astype(np.float64)
            a32 = (base - mean_flat) * sw32
            totals[day] = float(np.dot(a32.astype(np.float64),
                                       a32.astype(np.float64)))
            if climate == "pd" and day == 0:
                first_a = a32.copy()
            if (day + 1) % 100 == 0 or day == N_DAYS - 1:
                log(f"predictor scan {climate}: {day + 1}/{N_DAYS}")
        means[climate] = (acc / N_DAYS).reshape(H, W)
        raw[climate] = {"total_weighted_norm2": totals}
        if first_a is not None:
            raw[climate]["first_weighted_anomaly32"] = first_a
    return raw, means


def basis_block_paths() -> list[Path]:
    paths = sorted(X_BLOCKS.glob("eof_block_*.npy"))
    if len(paths) != 52:
        raise RuntimeError(f"expected 52 predictor EOF blocks, found {len(paths)}")
    npx = 0
    for path in paths:
        a = np.load(path, mmap_mode="r")
        if a.shape[0] != KX or a.dtype != np.float32:
            raise RuntimeError(f"bad EOF block {path}: {a.shape} {a.dtype}")
        npx += a.shape[1]
        del a
    if npx != NPIX:
        raise RuntimeError(f"EOF block pixel cover {npx} != {NPIX}")
    return paths


def project_vectors(paths: list[Path], vectors_w: np.ndarray) -> np.ndarray:

    coef = np.zeros((vectors_w.shape[0], KX), dtype=np.float64)
    p0 = 0
    for ib, path in enumerate(paths):
        e = np.load(path, mmap_mode="r")
        nb = e.shape[1]
        for c0 in range(0, nb, 8192):
            c1 = min(c0 + 8192, nb)
            eb = np.asarray(e[:, c0:c1], dtype=np.float64)
            vb = np.asarray(vectors_w[:, p0 + c0:p0 + c1], dtype=np.float64)
            coef += vb @ eb.T
            del eb, vb
        p0 += nb
        log(f"predictor EOF projection block {ib + 1}/{len(paths)}")
        del e
    return coef


def reconstruct_and_reproject(paths: list[Path], coef: np.ndarray):

    fields = np.empty((coef.shape[0], NPIX), dtype=np.float64)
    repro = np.zeros_like(coef)
    p0 = 0
    for ib, path in enumerate(paths):
        e = np.load(path, mmap_mode="r")
        nb = e.shape[1]
        for c0 in range(0, nb, 8192):
            c1 = min(c0 + 8192, nb)
            eb = np.asarray(e[:, c0:c1], dtype=np.float64)
            rec_w = coef @ eb
            sl = slice(p0 + c0, p0 + c1)
            fields[:, sl] = rec_w / SQRT_W_FLAT[sl][None, :] * TS
            repro += rec_w @ eb.T
            del eb, rec_w
        p0 += nb
        log(f"predictor EOF reconstruction block {ib + 1}/{len(paths)}")
        del e
    return fields.reshape(coef.shape[0], H, W), repro


def direct_projection_gate(pd_anom_w32: np.ndarray, coef: np.ndarray,
                           fields: np.ndarray, original_pd_anom_K: np.ndarray,
                           repro: np.ndarray) -> dict:
    accepted = np.load(X_SCORE_PATHS["pd"], mmap_mode="r")[0].astype(np.float64)
    calc = coef[0]
    diff32 = calc.astype(np.float32).astype(np.float64) - accepted
    pfield = fields[0]
    qfield = original_pd_anom_K - pfield
    total = wmeansq(original_pd_anom_K)
    par = wmeansq(pfield)
    out = wmeansq(qfield)
    cross = 2.0 * wmean(pfield * qfield)
    gate = {
        "population": "PD test",
        "day_index": 0,
        "accepted_score_dtype": "float32",
        "calculation_accumulator_dtype": "float64",
        "score_max_abs_deviation_after_float32_cast": float(np.abs(diff32).max()),
        "score_rms_deviation_after_float32_cast": float(
            np.sqrt(np.mean(diff32 * diff32))),
        "score_relative_l2_deviation": float(
            np.linalg.norm(diff32) / max(np.linalg.norm(accepted), 1e-30)),
        "reprojected_coefficient_relative_l2_deviation": float(
            np.linalg.norm(repro[0] - coef[0]) /
            max(np.linalg.norm(coef[0]), 1e-30)),
        "direct_total_MSE_K2": total,
        "direct_parallel_MSE_K2": par,
        "direct_outside_MSE_K2": out,
        "direct_cross_K2": cross,
        "direct_pythagorean_closure_K2": total - par - out,
        "direct_pythagorean_closure_including_cross_K2": (
            total - par - out - cross),
    }
    if (gate["score_max_abs_deviation_after_float32_cast"] > 5e-4 or
            gate["score_relative_l2_deviation"] > 2e-7):
        raise RuntimeError(f"accepted predictor score gate failed: {gate}")
    log("GATE accepted PD predictor score: max abs "
        f"{gate['score_max_abs_deviation_after_float32_cast']:.3e}, rel L2 "
        f"{gate['score_relative_l2_deviation']:.3e}")
    return gate


def daily_predictor_results(raw: dict) -> tuple[list[dict], dict]:
    rows = []
    daily = {}
    for climate in CLIMATES:
        scores = np.load(X_SCORE_PATHS[climate], mmap_mode="r")
        if scores.shape != (N_DAYS, KX):
            raise RuntimeError(f"{climate}: unexpected score shape {scores.shape}")
        total_n = raw[climate]["total_weighted_norm2"]
        captured_n = np.einsum("ij,ij->i",
                               np.asarray(scores, dtype=np.float64),
                               np.asarray(scores, dtype=np.float64))
        outside_n = total_n - captured_n
        if outside_n.min() < -1e-5 * total_n.max():
            raise RuntimeError(f"{climate}: materially negative outside energy")
        outside_n = np.maximum(outside_n, 0.0)
        total_mse = total_n / NPIX * TS ** 2
        outside_mse = outside_n / NPIX * TS ** 2
        outside_rms = np.sqrt(outside_mse)
        outside_frac = outside_n / total_n
        closure = total_mse - (captured_n / NPIX * TS ** 2) - outside_mse
        row = {
            "record": f"{CLIMATE_LABEL[climate]} daily population",
            "record_type": "daily_population",
            "climate": climate,
            "n_days": N_DAYS,
            "total_MSE_K2": float(total_mse.mean()),
            "total_RMS_K": float(math.sqrt(total_mse.mean())),
            "outside_MSE_K2": float(outside_mse.mean()),
            "outside_RMS_K": float(math.sqrt(outside_mse.mean())),
            "outside_fraction": float(outside_mse.mean() / total_mse.mean()),
            "captured_fraction": float(1 - outside_mse.mean() / total_mse.mean()),
            **quantile_summary(outside_rms, "daily_outside_RMS_K"),
            **quantile_summary(outside_frac, "daily_outside_fraction"),
            "pythagorean_closure_max_abs_K2": float(np.abs(closure).max()),
            "pythagorean_closure_max_rel": float(
                np.max(np.abs(closure) / np.maximum(total_mse, 1e-30))),
            "minimum_raw_outside_weighted_norm2": float(
                (total_n - captured_n).min()),
        }
        rows.append(row)
        daily[climate] = {
            "outside_RMS_K": outside_rms,
            "outside_fraction": outside_frac,
            "total_MSE_K2": total_mse,
            "captured_MSE_K2": captured_n / NPIX * TS ** 2,
            "outside_MSE_K2": outside_mse,
        }
        log(f"Part A daily {climate}: outside RMS {row['outside_RMS_K']:.6g} K")
    return rows, daily


def field_projection_row(record: str, rtype: str, field: np.ndarray,
                         parallel: np.ndarray) -> dict:
    outside = field - parallel
    total_mse = wmeansq(field)
    outside_mse = wmeansq(outside)
    par_mse = wmeansq(parallel)
    closure = total_mse - par_mse - outside_mse
    q = np.percentile(parallel, [1, 5, 50, 95, 99])
    return {
        "record": record,
        "record_type": rtype,
        "climate": "",
        "n_days": N_DAYS if "mean displacement" in record else "",
        "total_MSE_K2": total_mse,
        "total_RMS_K": math.sqrt(max(total_mse, 0.0)),
        "outside_MSE_K2": outside_mse,
        "outside_RMS_K": math.sqrt(max(outside_mse, 0.0)),
        "outside_fraction": outside_mse / total_mse,
        "captured_fraction": 1.0 - outside_mse / total_mse,
        "parallel_area_weighted_mean_K": wmean(parallel),
        "parallel_area_weighted_spatial_std_K": wstd(parallel),
        "area_weighted_spatial_correlation": wcorr(field, parallel),
        "parallel_minimum_K": float(parallel.min()),
        "parallel_maximum_K": float(parallel.max()),
        "parallel_grid_p01_K": float(q[0]),
        "parallel_grid_p05_K": float(q[1]),
        "parallel_grid_p50_K": float(q[2]),
        "parallel_grid_p95_K": float(q[3]),
        "parallel_grid_p99_K": float(q[4]),
        "maximum_absolute_Q_field_K": float(np.abs(outside).max()),
        "pythagorean_closure_abs_K2": abs(closure),
        "pythagorean_closure_rel": abs(closure) / max(total_mse, 1e-30),
    }


def reconstruct_y_maps(score_rows: np.ndarray) -> np.ndarray:
    e = npz_member_memmap(Y_BASIS, "eofs_weighted.npy")
    out = np.empty((score_rows.shape[0], NPIX), dtype=np.float64)
    ef = e[:KY].reshape(KY, -1)
    for p0 in range(0, NPIX, 131072):
        p1 = min(p0 + 131072, NPIX)
        eb = np.asarray(ef[:, p0:p1], dtype=np.float64)
        out[:, p0:p1] = (score_rows @ eb) / SQRT_W_FLAT[p0:p1] * TS
        del eb
    del e
    return out.reshape(score_rows.shape[0], H, W)


def cca_uniform_consistency(coef_u: np.ndarray, repro_u: np.ndarray) -> dict:
    model = np.load(CCA_MODEL, allow_pickle=False)
    if (int(model["Kx"]) != KX or int(model["Ky"]) != KY or
            int(model["r"]) != 512 or float(model["ridge_alpha"]) != 0.0):
        raise RuntimeError("frozen CCA configuration gate failed")
    bmat = np.asarray(model["B"], dtype=np.float64)
    response_u = coef_u @ bmat
    response_pu = repro_u @ bmat
    response_qu = (coef_u - repro_u) @ bmat
    maps = reconstruct_y_maps(np.stack([
        response_u, response_u - response_pu, response_qu]))
    accepted_u = np.load(UNIFORM_RUN / "partials/cca_day_mse.npz",
                         allow_pickle=False)["u"].astype(np.float64)
    accepted_map = np.load(
        MECH / "outputs/cca_invariance_sensitivity_maps.npz",
        allow_pickle=False)["cca_sensitivity_ssp_B_uniform_1K"].astype(np.float64)
    section_l = json.loads((MECH / "outputs/section_l_cca.json").read_text())
    remap = section_l["representability"]["ssp_B_uniform_1K"]
    result = {
        "response_u_rms_K": float(np.linalg.norm(response_u) * TS /
                                   math.sqrt(NPIX)),
        "accepted_uniform_response_rms_K": float(
            np.linalg.norm(accepted_u) * TS / math.sqrt(NPIX)),
        "response_score_max_abs_deviation_vs_accepted": float(
            np.abs(response_u - accepted_u).max()),
        "response_score_relative_l2_deviation_vs_accepted": float(
            np.linalg.norm(response_u - accepted_u) /
            max(np.linalg.norm(accepted_u), 1e-30)),
        "response_map_max_abs_deviation_vs_retained_K": float(
            np.abs(maps[0] - accepted_map).max()),
        "response_map_global_RMS_deviation_vs_retained_K": wrms(
            maps[0] - accepted_map),
        "CCA_response_u_minus_response_Pu_max_abs_K": float(
            np.abs(maps[1]).max()),
        "CCA_response_u_minus_response_Pu_global_RMS_K": wrms(maps[1]),
        "CCA_response_Qu_max_abs_K": float(np.abs(maps[2]).max()),
        "CCA_response_Qu_global_RMS_K": wrms(maps[2]),
        "physical_remap_uniform_abs_error_RMS_K": float(
            remap["representability_abs_error_rms_K"]),
        "physical_remap_uniform_relative_error": float(
            remap["representability_rel_error"]),
        "physical_remap_uniform_idempotence_relative_error": float(
            remap["idempotence_rel_error"]),
        "note": (
            "Physical conservative HR->LR plus bilinear LR->HR remapping and "
            "predictor-EOF projection are different operators. The retained "
            "remap preserves 1 K to floating precision; the EOF result is "
            "reported independently."
        ),
    }
    model.close()
    return result


def prediction_mean_errors() -> dict:

    import zarr

    out = {}
    for climate in CLIMATES:
        root = PRED_ROOTS[climate]
        group = zarr.open_group(str(ZARRS[climate]), mode="r")
        sum_c = np.zeros(NPIX, dtype=np.float64)
        sum_u = np.zeros(NPIX, dtype=np.float64)
        count = 0
        for shard in range(20):
            done = json.loads((root / "predictions/b2" /
                               f"shard_{shard:02d}/shard_done.json").read_text())
            lo, hi = int(done["lo"]), int(done["hi"])
            up = np.load(root / "predictions/b2" / f"shard_{shard:02d}" /
                         "b2_prediction_norm.npy", mmap_mode="r")
            cp = np.load(root / "predictions/cca" / f"shard_{shard:02d}" /
                         "cca_prediction_norm.npy", mmap_mode="r")
            if up.shape[0] != hi - lo or cp.shape[0] != hi - lo:
                raise RuntimeError(f"{climate} shard {shard}: prediction shape")
            for k, day in enumerate(range(lo, hi)):
                target = np.asarray(group["targets"][day, 0],
                                    dtype=np.float32).reshape(-1)
                sum_u += np.asarray(up[k], dtype=np.float64).reshape(-1) - target
                sum_c += np.asarray(cp[k], dtype=np.float64).reshape(-1) - target
                count += 1
            del up, cp
            log(f"retained mean error {climate}: shard {shard + 1}/20")
        if count != N_DAYS:
            raise RuntimeError(f"{climate}: retained prediction count {count}")
        out[climate] = {
            "cca_mean_error_K": (sum_c / count * TS).reshape(H, W),
            "unet_mean_error_K": (sum_u / count * TS).reshape(H, W),
        }
    return out


def four_way(model_data: dict, mean_errors: dict) -> tuple[list[dict], dict]:
    rows_by = {}
    gates = {}
    for climate in CLIMATES:
        d = model_data[climate]
        for model in ("CCA", "U-Net"):
            ecoef = d["ec"] if model == "CCA" else d["eu"]
            mu = ecoef.mean(axis=0)
            b_in = float(np.dot(mu, mu))
            centered = ecoef - mu
            v_in = float(np.mean(np.einsum("ij,ij->i", centered, centered)))
            acc = d["accepted"]
            inside_ref = acc["cca_inside"] if model == "CCA" else acc["unet_inside"]
            outside_ref = acc["cca_outside"] if model == "CCA" else acc["unet_outside"]
            total_ref = acc["cca_total"] if model == "CCA" else acc["unet_total"]
            key = "cca_mean_error_K" if model == "CCA" else "unet_mean_error_K"
            b_total = wmeansq(mean_errors[climate][key])
            b_out = b_total - b_in
            v_out = outside_ref - b_out
            row = {
                "model": model,
                "climate": CLIMATE_LABEL[climate],
                "climate_key": climate,
                "n_days": N_DAYS,
                "total_MSE": total_ref,
                "B_in": b_in,
                "V_in": v_in,
                "B_out": b_out,
                "V_out": v_out,
                "closure": total_ref - b_in - v_in - b_out - v_out,
                "inside_reproduction_deviation": b_in + v_in - inside_ref,
                "outside_reproduction_deviation": b_out + v_out - outside_ref,
                "persistent_total_B": b_total,
            }
            rows_by[(model, climate)] = row
            gates[f"{model}_{climate}"] = {
                "inside_reference_K2": inside_ref,
                "inside_four_way_sum_K2": b_in + v_in,
                "inside_abs_deviation_K2": abs(b_in + v_in - inside_ref),
                "outside_reference_K2": outside_ref,
                "outside_four_way_sum_K2": b_out + v_out,
                "outside_abs_deviation_K2": abs(b_out + v_out - outside_ref),
                "total_closure_abs_K2": abs(row["closure"]),
            }
            if (abs(b_in + v_in - inside_ref) > 1e-9 or
                    abs(b_out + v_out - outside_ref) > 1e-12 or
                    abs(row["closure"]) > 1e-9):
                raise RuntimeError(f"four-way gate failed: {model} {climate} {row}")
    components = ("total_MSE", "B_in", "V_in", "B_out", "V_out")
    for model in ("CCA", "U-Net"):
        base = rows_by[(model, "pd")]
        for climate in CLIMATES:
            row = rows_by[(model, climate)]
            for key in components:
                row[f"change_vs_PD_{key}"] = row[key] - base[key]
    for climate in CLIMATES:
        cca = rows_by[("CCA", climate)]
        unet = rows_by[("U-Net", climate)]
        pd_cca = rows_by[("CCA", "pd")]
        pd_unet = rows_by[("U-Net", "pd")]
        for key in components:
            gap = cca[key] - unet[key]
            dgap = gap - (pd_cca[key] - pd_unet[key])
            for model in ("CCA", "U-Net"):
                rows_by[(model, climate)][f"gap_CCA_minus_UNet_{key}"] = gap
                rows_by[(model, climate)][f"change_vs_PD_gap_{key}"] = dgap
    rows = [rows_by[(model, climate)] for model in ("CCA", "U-Net")
            for climate in CLIMATES]
    return rows, gates


def plot_map(ax, field: np.ndarray, title: str, cmap: str, vmin=None, vmax=None):
    import cartopy
    cartopy.config["data_dir"] = str(CARTOPY_DIR)
    import cartopy.crs as ccrs

    stride = 4
    dat = field[::stride, ::stride]
    lon = LON[::stride]
    lat = LAT[::stride]
    dat = np.concatenate([dat, dat[:, :1]], axis=1)
    lon = np.concatenate([lon, [lon[0] + 360.0]])
    pm = ax.pcolormesh(lon, lat, dat, transform=ccrs.PlateCarree(),
                       shading="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                       rasterized=True)
    ax.coastlines(resolution="50m", linewidth=0.35, color="0.2")
    ax.set_global()
    ax.set_title(title, fontsize=8)
    return pm


def make_figures(ret: Path, predictor_rows: list[dict], daily: dict,
                 uniform_parallel: np.ndarray, four_rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy
    cartopy.config["data_dir"] = str(CARTOPY_DIR)
    import cartopy.crs as ccrs

    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.6, "savefig.bbox": "tight",
    })
    pop_rows = [r for r in predictor_rows if r["record_type"] == "daily_population"]
    disp_rows = [r for r in predictor_rows if r["record_type"] in
                 ("mean_displacement", "uniform_displacement")]
    fig = plt.figure(figsize=(11.2, 3.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 1.25, 2.0])
    ax = fig.add_subplot(gs[0, 0])
    ax.bar([r["climate"].upper() for r in pop_rows],
           [r["outside_RMS_K"] for r in pop_rows], color="#4477AA")
    ax.set_ylabel("Outside-span RMS (K)")
    ax.set_title("(a) Daily populations", loc="left")
    ax.grid(axis="y", alpha=0.25)
    ax = fig.add_subplot(gs[0, 1])
    labels = ["PD", "MH", "SSP", "MH−PD\nmean", "SSP−PD\nmean", "1 K"]
    vals = [r["outside_fraction"] for r in pop_rows + disp_rows]
    ax.bar(np.arange(len(vals)), vals, color=["#4477AA"] * 3 + ["#66CCEE"] * 2 + ["#CC6677"])
    ax.set_xticks(np.arange(len(vals)), labels, rotation=25, ha="right")
    ax.set_ylabel("Squared fraction outside span")
    ax.set_title("(b) Span loss", loc="left")
    ax.axhline(0, color="0.2", linewidth=0.6)
    ax.grid(axis="y", alpha=0.25)
    ax = fig.add_subplot(gs[0, 2], projection=ccrs.EqualEarth(central_longitude=0))
    dev = uniform_parallel - 1.0
    lim = float(np.percentile(np.abs(dev), 99.5))
    pm = plot_map(ax, dev, r"(c) $P_x(1\,K)-1\,K$", "RdBu_r", -lim, lim)
    fig.colorbar(pm, ax=ax, orientation="horizontal", pad=0.04, shrink=0.78,
                 label="K")
    for ext in ("png", "pdf"):
        path = ret / f"predictor_span_diagnostics.{ext}"
        if path.exists():
            raise RuntimeError(f"refusing to overwrite {path}")
        fig.savefig(path, dpi=260 if ext == "png" else None)
    plt.close(fig)

    fig = plt.figure(figsize=(10.2, 3.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2)
    lim1 = float(np.percentile(np.abs(uniform_parallel), 99.5))
    ax = fig.add_subplot(gs[0, 0], projection=ccrs.EqualEarth(central_longitude=0))
    pm = plot_map(ax, uniform_parallel, r"(a) $P_x(1\,K)$", "viridis",
                  0.0, max(lim1, 1.0))
    fig.colorbar(pm, ax=ax, orientation="horizontal", pad=0.04, shrink=0.75,
                 label="K")
    ax = fig.add_subplot(gs[0, 1], projection=ccrs.EqualEarth(central_longitude=0))
    dev = uniform_parallel - 1.0
    lim = float(np.percentile(np.abs(dev), 99.5))
    pm = plot_map(ax, dev, r"(b) $P_x(1\,K)-1\,K$", "RdBu_r", -lim, lim)
    fig.colorbar(pm, ax=ax, orientation="horizontal", pad=0.04, shrink=0.75,
                 label="K")
    for ext in ("png", "pdf"):
        path = ret / f"uniform_projection_maps.{ext}"
        if path.exists():
            raise RuntimeError(f"refusing to overwrite {path}")
        fig.savefig(path, dpi=260 if ext == "png" else None)
    plt.close(fig)

    lookup = {(r["model"], r["climate_key"]): r for r in four_rows}
    comp = ("B_in", "V_in", "B_out", "V_out")
    colors = ["#56B4E9", "#E69F00", "#CC79A7", "#009E73"]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.5), sharey=False,
                             constrained_layout=True)
    for i, model in enumerate(("CCA", "U-Net")):
        for j, climate in enumerate(("mh", "ssp")):
            ax = axes[i, j]
            vals = [lookup[(model, climate)][f"change_vs_PD_{k}"] for k in comp]
            ax.bar(np.arange(4), vals, color=colors)
            ax.axhline(0, color="0.15", linewidth=0.8)
            ax.set_xticks(np.arange(4), [r"$B_{in}$", r"$V_{in}$",
                                         r"$B_{out}$", r"$V_{out}$"])
            ax.set_ylabel("Change from PD (K$^2$)")
            ax.set_title(f"{model}: {CLIMATE_LABEL[climate]} − PD")
            ax.grid(axis="y", alpha=0.25)
    for ext in ("png", "pdf"):
        path = ret / f"four_way_error_decomposition.{ext}"
        if path.exists():
            raise RuntimeError(f"refusing to overwrite {path}")
        fig.savefig(path, dpi=260 if ext == "png" else None)
    plt.close(fig)


def fnum(x, digits=6):
    return f"{float(x):.{digits}g}"


def dominant_text(four_rows: list[dict], model: str, climate: str) -> str:
    row = next(r for r in four_rows
               if r["model"] == model and r["climate_key"] == climate)
    comp = ("B_in", "V_in", "B_out", "V_out")
    changes = {k: row[f"change_vs_PD_{k}"] for k in comp}
    key = max(changes, key=lambda k: abs(changes[k]))
    return f"{key} ({changes[key]:+.6g} K^2)"


def make_report(ret: Path, predictor_rows: list[dict], four_rows: list[dict],
                score_gate: dict, target_gates: dict, four_gates: dict,
                uniform_consistency: dict, provenance: dict) -> None:
    p = {r["record"]: r for r in predictor_rows}
    f = {(r["model"], r["climate_key"]): r for r in four_rows}
    u = p["Uniform 1 K displacement"]
    mh_d = p["MH mean displacement"]
    ssp_d = p["SSP5-8.5 mean displacement"]
    fidelity_high = (u["outside_fraction"] < 1e-3 and
                     mh_d["outside_fraction"] < 1e-3 and
                     ssp_d["outside_fraction"] < 1e-3)

    def change(model, climate, comp):
        return f[(model, climate)][f"change_vs_PD_{comp}"]

    def primary(a, b):
        return "persistent" if abs(a) > abs(b) else "time-varying"

    mh_cca_out = primary(change("CCA", "mh", "B_out"),
                         change("CCA", "mh", "V_out"))
    mh_unet_out = primary(change("U-Net", "mh", "B_out"),
                          change("U-Net", "mh", "V_out"))
    ssp_unet_out = primary(change("U-Net", "ssp", "B_out"),
                           change("U-Net", "ssp", "V_out"))
    ssp_unet_in = primary(change("U-Net", "ssp", "B_in"),
                          change("U-Net", "ssp", "V_in"))
    interpretation = (
        "The predictor span captures the tested mean displacements and uniform "
        "shift closely; this strengthens the distinction between coordinate "
        "extrapolation within the training span and information discarded by "
        "projection."
        if fidelity_high else
        "Non-negligible predictor-span loss is present, so any interpretation "
        "that treats predictor projection as lossless is weakened; this does "
        "not by itself identify a model mechanism."
    )
    intervention = (
        "Unnecessary for this paper-level triage because the tested transfer "
        "displacements are already faithfully represented."
        if fidelity_high else
        "Scientifically worthwhile as a single controlled intervention because "
        "Part A found material information outside the predictor span."
    )
    lines = f"""# Focused mechanism triage

## 1. Plain-language summary

This point-estimate diagnostic used the accepted 1096-day PD-test, Mid-Holocene, and SSP5-8.5 populations. No model was rerun. The uniform 1 K field has **{u['outside_RMS_K']:.6g} K** RMS outside the predictor span and a captured squared fraction of **{u['captured_fraction']:.8f}**. Daily outside-span RMS is **{p['PD test daily population']['outside_RMS_K']:.6g} K (PD)**, **{p['Mid-Holocene daily population']['outside_RMS_K']:.6g} K (MH)**, and **{p['SSP5-8.5 daily population']['outside_RMS_K']:.6g} K (SSP5-8.5)**.

For the four-way error split, the dominant MH outside-space increase is **{mh_cca_out} for CCA** and **{mh_unet_out} for the U-Net**. The SSP5-8.5 U-Net outside-space improvement is primarily **{ssp_unet_out}**, while its inside-space deterioration is primarily **{ssp_unet_in}**. The decomposition adds temporal detail but does not support an architectural causal claim.

## 2. Exact sources and conventions

- Frozen CCA: `{CCA_MODEL}`; verified Kx=10592, Ky=512, rank=512, ridge=0.
- Predictor and target EOFs: `{X_BASIS}` and `{Y_BASIS}`.
- Predictor field: bilinear T2M converted from input-normalized values to target-normalized units as `input * (INPUT_STD/TARGET_STD) + (INPUT_MEAN-TARGET_MEAN)/TARGET_STD`, stored/treated as float32 by the accepted projection, then scaled to kelvin by `TARGET_STD={TS}`.
- Predictor centering: frozen PD-training `mean_field`; no evaluation mean is subtracted. Target errors use the frozen 512-mode cosine-latitude-weighted target projector.
- Area weight: `cos(latitude cell centre) / mean(cos(latitude cell centre))`; stored weighted EOF rows are not weighted a second time.
- All three climates retain 29 February and contain exactly 1096 days.
- The PD day-0 predictor-score reproduction has maximum float32-score deviation {score_gate['score_max_abs_deviation_after_float32_cast']:.3e} and relative L2 deviation {score_gate['score_relative_l2_deviation']:.3e}.
- The accepted CCA B/V/F decomposition was reproduced before the new diagnostics; maximum component deviation is {max(max(g['CCA_B_abs_deviation_K2'], g['CCA_V_abs_deviation_K2'], g['CCA_F_abs_deviation_K2']) for g in target_gates.values()):.3e} K².

## 3. Predictor-span results

| population/field | total RMS K | outside RMS K | outside squared fraction | captured squared fraction |
|---|---:|---:|---:|---:|
| PD daily population | {p['PD test daily population']['total_RMS_K']:.6f} | {p['PD test daily population']['outside_RMS_K']:.6f} | {p['PD test daily population']['outside_fraction']:.6g} | {p['PD test daily population']['captured_fraction']:.6g} |
| MH daily population | {p['Mid-Holocene daily population']['total_RMS_K']:.6f} | {p['Mid-Holocene daily population']['outside_RMS_K']:.6f} | {p['Mid-Holocene daily population']['outside_fraction']:.6g} | {p['Mid-Holocene daily population']['captured_fraction']:.6g} |
| SSP5-8.5 daily population | {p['SSP5-8.5 daily population']['total_RMS_K']:.6f} | {p['SSP5-8.5 daily population']['outside_RMS_K']:.6f} | {p['SSP5-8.5 daily population']['outside_fraction']:.6g} | {p['SSP5-8.5 daily population']['captured_fraction']:.6g} |
| MH − PD mean displacement | {mh_d['total_RMS_K']:.6f} | {mh_d['outside_RMS_K']:.6f} | {mh_d['outside_fraction']:.6g} | {mh_d['captured_fraction']:.6g} |
| SSP − PD mean displacement | {ssp_d['total_RMS_K']:.6f} | {ssp_d['outside_RMS_K']:.6f} | {ssp_d['outside_fraction']:.6g} | {ssp_d['captured_fraction']:.6g} |

For MH, `P_x d` has area mean {mh_d['parallel_area_weighted_mean_K']:.6f} K, spatial standard deviation {mh_d['parallel_area_weighted_spatial_std_K']:.6f} K, and weighted spatial correlation {mh_d['area_weighted_spatial_correlation']:.8f} with `d`. The SSP values are {ssp_d['parallel_area_weighted_mean_K']:.6f} K, {ssp_d['parallel_area_weighted_spatial_std_K']:.6f} K, and {ssp_d['area_weighted_spatial_correlation']:.8f}.

## 4. Uniform-shift representation results

`P_x(1 K)` has area-weighted mean {u['parallel_area_weighted_mean_K']:.8f} K, spatial standard deviation {u['parallel_area_weighted_spatial_std_K']:.8f} K, range [{u['parallel_minimum_K']:.8f}, {u['parallel_maximum_K']:.8f}] K, grid-cell p01/p05/p50/p95/p99 of {u['parallel_grid_p01_K']:.8f}/{u['parallel_grid_p05_K']:.8f}/{u['parallel_grid_p50_K']:.8f}/{u['parallel_grid_p95_K']:.8f}/{u['parallel_grid_p99_K']:.8f} K, and maximum `|Q_x u|` {u['maximum_absolute_Q_field_K']:.8f} K.

The CCA correction response is {uniform_consistency['response_u_rms_K']:.10f} K RMS per 1 K versus the accepted {uniform_consistency['accepted_uniform_response_rms_K']:.10f} K. `CCA(u)-CCA(P_xu)` is at most {uniform_consistency['CCA_response_u_minus_response_Pu_max_abs_K']:.3e} K with global RMS {uniform_consistency['CCA_response_u_minus_response_Pu_global_RMS_K']:.3e} K; `CCA(Q_xu)` is at most {uniform_consistency['CCA_response_Qu_max_abs_K']:.3e} K with global RMS {uniform_consistency['CCA_response_Qu_global_RMS_K']:.3e} K.

Physical remapping is separate: the accepted conservative-coarsening/bilinear-return operator preserves 1 K with RMS error {uniform_consistency['physical_remap_uniform_abs_error_RMS_K']:.3e} K. The EOF-span figures above test training-span representability instead.

## 5. Four-component CCA/U-Net results

| model | climate | total MSE | B_in | V_in | B_out | V_out | closure |
|---|---|---:|---:|---:|---:|---:|---:|
"""
    for row in four_rows:
        lines += (f"| {row['model']} | {row['climate']} | {row['total_MSE']:.9f} | "
                  f"{row['B_in']:.9f} | {row['V_in']:.9f} | "
                  f"{row['B_out']:.9f} | {row['V_out']:.9f} | "
                  f"{row['closure']:.2e} |\n")
    lines += f"""

Changes are in `four_way_error_decomposition.csv`. Factually:

1. MH outside-space increase: CCA is {mh_cca_out} (ΔB_out={change('CCA','mh','B_out'):+.6g}, ΔV_out={change('CCA','mh','V_out'):+.6g} K²); U-Net is {mh_unet_out} (ΔB_out={change('U-Net','mh','B_out'):+.6g}, ΔV_out={change('U-Net','mh','V_out'):+.6g} K²).
2. SSP5-8.5 U-Net outside improvement is {ssp_unet_out} (ΔB_out={change('U-Net','ssp','B_out'):+.6g}, ΔV_out={change('U-Net','ssp','V_out'):+.6g} K²).
3. SSP5-8.5 U-Net inside deterioration is {ssp_unet_in} (ΔB_in={change('U-Net','ssp','B_in'):+.6g}, ΔV_in={change('U-Net','ssp','V_in'):+.6g} K²).
4. The CCA B/V/F result is exactly reproduced; the four-way split adds detail about persistence versus variability. It does not independently identify architecture or causation.

## 6. What is exact arithmetic versus interpretation

The B/V identities, inside/outside identities, CCA-minus-U-Net gaps, and changes are deterministic arithmetic on accepted retained predictions/scores and targets. Closures are limited only by float32 EOF storage and retained-field accumulation. “Primarily persistent/time-varying” means the larger absolute point-estimate component change. Claims about explanatory value, model mechanism, or manuscript priority are interpretations and are not implied by the identities.

## 7. Effect on accepted interpretation

{interpretation} The accepted result that SSP CCA mapping deterioration is dominated by persistent coefficient bias is reproduced, not weakened. The U-Net four-way split is additional descriptive detail; no architectural causal claim is made.

## 8. Decision table

| candidate follow-up | evidence from this triage | expected scientific value | compute cost | required new inference | recommendation deferred to manuscript review |
|---|---|---|---|---|---|
| Predictor-span fidelity in paper | Daily outside RMS PD/MH/SSP = {p['PD test daily population']['outside_RMS_K']:.4g}/{p['Mid-Holocene daily population']['outside_RMS_K']:.4g}/{p['SSP5-8.5 daily population']['outside_RMS_K']:.4g} K; uniform captured fraction {u['captured_fraction']:.6g} | Distinguishes span loss from displacement within retained coordinates | Completed; one CPU job | None | Defer wording/placement to manuscript review |
| Four-way error split in paper | Dominant changes: MH CCA {dominant_text(four_rows,'CCA','mh')}, MH U-Net {dominant_text(four_rows,'U-Net','mh')}; SSP CCA {dominant_text(four_rows,'CCA','ssp')}, SSP U-Net {dominant_text(four_rows,'U-Net','ssp')} | Adds persistence/variability detail to accepted inside/outside split | Completed from retained artifacts | None | Defer inclusion to manuscript review |
| Projected-input U-Net intervention | {intervention} | Controlled sensitivity to discarded predictor components only; not a causal representation study | One projected-input construction plus one frozen 1096-day inference pass | Yes, one frozen U-Net pass | Deliberately deferred |

## 9. Feasibility of the deferred projected-input U-Net intervention

It would require the frozen predictor mean/basis, retained per-day predictor scores (or exact reprojection), the three original zarr inputs with non-T2M channels unchanged, the adopted checkpoint and archived inference code, and retained targets/baselines. A practical implementation would reconstruct about 14.7 GB of float32 projected T2M for one 1096-day climate, reading the 142 GB predictor basis once in blocks, then run one frozen U-Net inference job. Peak host memory can remain below roughly 20–40 GB with block/day streaming; GPU memory and inference cost match one accepted U-Net evaluation pass. One new frozen inference job is scientifically sufficient for a chosen population. It is worthwhile if Part A shows material or climate-dependent outside-span RMS/fraction; it is unnecessary if daily and displacement loss are negligible and the uniform field is faithfully captured. This intervention was not performed.

## 10. Provenance and closure checks

- Predictor direct-field Pythagorean closure on reproduced PD day 0: {score_gate['direct_pythagorean_closure_K2']:.3e} K²; cross-aware closure {score_gate['direct_pythagorean_closure_including_cross_K2']:.3e} K².
- Maximum four-way total closure: {max(abs(r['closure']) for r in four_rows):.3e} K².
- Maximum accepted inside/outside reproduction deviation: {max(max(g['inside_abs_deviation_K2'], g['outside_abs_deviation_K2']) for g in four_gates.values()):.3e} K².
- Figures use the paper's Equal Earth projection (`central_longitude=0`) with a 4× display-only decimation; all statistics use the full grid.
- Machine-readable paths, hashes, conventions, gates, environment, and declarations are in `provenance.json`; output hashes are in `MANIFEST.sha256`.
"""
    path = ret / "TRIAGE_REPORT.md"
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.write_text(lines, encoding="utf-8")


def make_readme(ret: Path, work_root: Path) -> None:
    text = f"""# Focused mechanism triage return package

Work root: `{work_root}`

This package contains exact point-estimate predictor-span and four-way target-error diagnostics for the accepted 1096-day PD, Mid-Holocene, and SSP5-8.5 populations.

- No model was rerun.
- No EOF was refitted.
- No bootstrap was run.
- No accepted file was modified.
- No projected-input U-Net intervention was performed.

`TRIAGE_REPORT.md` is the concise review document. CSV files contain full-precision summary values, PNG/PDF files are diagnostic figures, and `provenance.json` records sources, conventions, gates, and closures.
"""
    path = ret / "README.md"
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.write_text(text, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    if root != ROOT_EXPECTED.resolve():
        raise SystemExit(f"root gate failed: {root} != {ROOT_EXPECTED}")
    ret = root / "RETURN_TO_CHAT"
    arrays = root / "arrays"
    if not ret.is_dir() or not arrays.is_dir():
        raise SystemExit("required new output directories are missing")
    required_outputs = [
        "TRIAGE_REPORT.md", "predictor_span_summary.csv",
        "four_way_error_decomposition.csv", "predictor_span_diagnostics.png",
        "predictor_span_diagnostics.pdf", "four_way_error_decomposition.png",
        "four_way_error_decomposition.pdf", "provenance.json", "README.md",
        "MANIFEST.sha256", "uniform_projection_maps.png",
        "uniform_projection_maps.pdf",
    ]
    existing = [str(ret / p) for p in required_outputs if (ret / p).exists()]
    if existing:
        raise SystemExit(f"refusing to reuse outputs: {existing}")
    start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


    model_data, target_gates = load_model_partials_and_gate()

    x_mean = np.asarray(npz_small(X_BASIS, "mean_field.npy"),
                        dtype=np.float32)
    x_lat = np.asarray(npz_small(X_BASIS, "latitude.npy"), dtype=np.float64)
    if x_mean.shape != (H, W) or np.max(np.abs(x_lat - LAT)) > 1e-4:
        raise RuntimeError("predictor basis grid/mean gate failed")


    raw, means = scan_predictors(x_mean)
    d_mh_norm = means["mh"] - means["pd"]
    d_ssp_norm = means["ssp"] - means["pd"]
    vectors_w = np.stack([
        raw["pd"]["first_weighted_anomaly32"].astype(np.float64),
        d_mh_norm.reshape(-1) * SQRT_W_FLAT,
        d_ssp_norm.reshape(-1) * SQRT_W_FLAT,
        (np.ones(NPIX, dtype=np.float64) / TS) * SQRT_W_FLAT,
    ])
    paths = basis_block_paths()
    coef = project_vectors(paths, vectors_w)
    reconstructed, repro = reconstruct_and_reproject(paths, coef)
    original_pd = (baseline_target_norm(
        __import__("zarr").open_group(str(ZARRS["pd"]), mode="r")["inputs"][0, 0]
    ) - x_mean) * TS
    score_gate = direct_projection_gate(
        raw["pd"]["first_weighted_anomaly32"], coef, reconstructed,
        original_pd, repro)

    predictor_rows, daily = daily_predictor_results(raw)
    d_mh_K = d_mh_norm * TS
    d_ssp_K = d_ssp_norm * TS
    predictor_rows.append(field_projection_row(
        "MH mean displacement", "mean_displacement", d_mh_K,
        reconstructed[1]))
    predictor_rows.append(field_projection_row(
        "SSP5-8.5 mean displacement", "mean_displacement", d_ssp_K,
        reconstructed[2]))
    predictor_rows.append(field_projection_row(
        "Uniform 1 K displacement", "uniform_displacement",
        np.ones((H, W), dtype=np.float64), reconstructed[3]))

    uniform_consistency = cca_uniform_consistency(coef[3], repro[3])

    daily_path = arrays / "predictor_daily_distributions.npz"
    if daily_path.exists():
        raise RuntimeError(f"refusing to overwrite {daily_path}")
    np.savez_compressed(daily_path, **{
        f"{climate}__{key}": value for climate in CLIMATES
        for key, value in daily[climate].items()
    })
    maps_path = arrays / "predictor_projection_maps.npz"
    if maps_path.exists():
        raise RuntimeError(f"refusing to overwrite {maps_path}")
    np.savez_compressed(
        maps_path, lat=LAT, lon=LON,
        mh_displacement_K=d_mh_K.astype(np.float32),
        mh_parallel_K=reconstructed[1].astype(np.float32),
        ssp_displacement_K=d_ssp_K.astype(np.float32),
        ssp_parallel_K=reconstructed[2].astype(np.float32),
        uniform_parallel_K=reconstructed[3].astype(np.float32),
        uniform_Q_K=(1.0 - reconstructed[3]).astype(np.float32),
        uniform_parallel_minus_1K=(reconstructed[3] - 1.0).astype(np.float32),
    )

    mean_errors = prediction_mean_errors()
    four_rows, four_gates = four_way(model_data, mean_errors)

    write_csv(ret / "predictor_span_summary.csv", predictor_rows)
    write_csv(ret / "four_way_error_decomposition.csv", four_rows)
    make_figures(ret, predictor_rows, daily, reconstructed[3], four_rows)

    accepted_manifest = json.loads((MECH / "manifest.json").read_text())
    source_records = {
        "frozen_CCA_model": {"path": str(CCA_MODEL), "sha256": sha256(CCA_MODEL)},
        "predictor_EOF_basis": {
            "path": str(X_BASIS),
            "bytes": X_BASIS.stat().st_size,
            "sha256_from_accepted_manifest": accepted_manifest["inputs"]["x_basis"]["sha256"],
        },
        "target_EOF_basis": {
            "path": str(Y_BASIS),
            "bytes": Y_BASIS.stat().st_size,
            "sha256_from_accepted_manifest": accepted_manifest["inputs"]["y_basis"]["sha256"],
        },
        "normalization": {"path": str(NORM_STATS), "sha256": sha256(NORM_STATS)},
        "weights_hr2lr": {
            "path": str(W_HR2LR),
            "sha256_from_accepted_manifest": accepted_manifest["inputs"]["weights_hr2lr"]["sha256"],
        },
        "weights_lr2hr": {
            "path": str(W_LR2HR),
            "sha256_from_accepted_manifest": accepted_manifest["inputs"]["weights_lr2hr"]["sha256"],
        },
        "accepted_mechanism_manifest": {
            "path": str(MECH / "manifest.json"),
            "sha256": sha256(MECH / "manifest.json"),
        },
        "accepted_uniform_manifest": {
            "path": str(UNIFORM_RUN / "MANIFEST.json"),
            "sha256": sha256(UNIFORM_RUN / "MANIFEST.json"),
        },
        "accepted_BVF_audit": {"path": str(ACCEPTED_BVF),
                                "sha256": sha256(ACCEPTED_BVF)},
        "paper_renderer": {"path": str(PAPER_RENDERER),
                           "sha256": sha256(PAPER_RENDERER)},
    }
    for climate in CLIMATES:
        source_records[f"{climate}_zarr"] = {
            "path": str(ZARRS[climate]), "n_days": N_DAYS, "mode": "read-only"}
        source_records[f"{climate}_predictor_scores"] = {
            "path": str(X_SCORE_PATHS[climate]),
            "sha256": sha256(X_SCORE_PATHS[climate])}
        source_records[f"{climate}_retained_predictions"] = {
            "path": str(PRED_ROOTS[climate] / "predictions"),
            "mode": "read-only", "models": ["CCA", "U-Net"]}

    provenance = {
        "kind": "FOCUSED_MECHANISM_TRIAGE",
        "work_root": str(root),
        "started_utc": start,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "declarations": {
            "models_rerun": False,
            "eofs_refitted_or_recomputed": False,
            "normalization_or_centering_changed": False,
            "bootstrap_run": False,
            "accepted_files_modified": False,
            "projected_input_unet_intervention_performed": False,
            "new_climates_or_windows": False,
        },
        "population": "native calendar, 1096 days each, 29 February retained",
        "conventions": {
            "predictor_units": "target-normalized T2M; physical K scale TARGET_STD",
            "input_to_predictor": (
                "t2m_inp_norm*(INPUT_STD/TARGET_STD) + "
                "(INPUT_MEAN-TARGET_MEAN)/TARGET_STD; float32 accepted field"),
            "predictor_centering": "frozen PD-training predictor mean_field",
            "target_centering": "frozen PD-training residual mean_field; cancels in errors",
            "weight": "cos(cell-centre latitude)/mean(cos(cell-centre latitude))",
            "stored_EOF_orientation": "mode,row × latitude × longitude, weighted-space rows",
            "projection": "((field-mean)*sqrt(weight)) @ eofs_weighted.T",
            "reconstruction": "scores @ eofs_weighted / sqrt(weight)",
            "map_projection": "EqualEarth central_longitude=0; display stride 4",
        },
        "dimensions": {"height": H, "width": W, "pixels": NPIX,
                       "Kx": KX, "Ky": KY},
        "normalization_K": {"input_mean": IM, "input_std": IS,
                            "target_mean": TM, "target_std": TS},
        "sources": source_records,
        "gates": {
            "predictor_score_and_direct_closure": score_gate,
            "accepted_target_decomposition": target_gates,
            "four_way": four_gates,
            "uniform_CCA_consistency": uniform_consistency,
        },
        "exact_arithmetic": (
            "Point estimates are deterministic sums, means, projections, and "
            "orthogonal bias/variability identities on accepted arrays."),
        "interpretation_boundary": (
            "Dominance labels compare absolute point-estimate component changes; "
            "they are descriptive and not architectural or causal claims."),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "missing_retained_artifacts": [],
        "internal_arrays": [str(daily_path), str(maps_path)],
    }
    make_readme(ret, root)
    make_report(ret, predictor_rows, four_rows, score_gate, target_gates,
                four_gates, uniform_consistency, provenance)
    json_dump(ret / "provenance.json", provenance)

    manifest_path = ret / "MANIFEST.sha256"
    if manifest_path.exists():
        raise RuntimeError(f"refusing to overwrite {manifest_path}")
    files = sorted(p for p in ret.iterdir() if p.is_file() and p != manifest_path)
    manifest_path.write_text("".join(f"{sha256(p)}  {p.name}\n" for p in files),
                             encoding="utf-8")

    pp = {r["record"]: r for r in predictor_rows}
    print("\nFINAL TERMINAL SUMMARY")
    print(f"1. WORK_ROOT: {root}")
    print("2. Part A completed: YES")
    print("3. Part B completed from retained artifacts: YES")
    print("4. Uniform 1 K outside-span RMS / captured fraction: "
          f"{pp['Uniform 1 K displacement']['outside_RMS_K']:.10g} K / "
          f"{pp['Uniform 1 K displacement']['captured_fraction']:.10g}")
    print("5. PD/MH/SSP daily outside-span RMS: "
          f"{pp['PD test daily population']['outside_RMS_K']:.10g} / "
          f"{pp['Mid-Holocene daily population']['outside_RMS_K']:.10g} / "
          f"{pp['SSP5-8.5 daily population']['outside_RMS_K']:.10g} K")
    print("6. Dominant four-way changes:")
    for model in ("CCA", "U-Net"):
        print(f"   {model} MH: {dominant_text(four_rows, model, 'mh')}; "
              f"SSP: {dominant_text(four_rows, model, 'ssp')}")
    print("7. RETURN_TO_CHAT paths:")
    for name in required_outputs:
        print(f"   {ret / name}")
    print("8. Missing retained artifact: NONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
