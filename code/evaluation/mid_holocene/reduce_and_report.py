"""Calculate mid-Holocene metrics from retained predictions and targets.
Keeps the original shard accumulation, regional/seasonal RMSE, distribution
histograms and spectra, with the original read-only present-day references.
Figure rendering and exploratory review reports are omitted. Historical
intermediate skill columns use MSE ratios; use the SSP5-8.5 rmse-skill command
for the final paper's RMSE-ratio skill and intervals. External inputs are required."""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from downscaling_numerics import (
    GaussLegendreSHTPlan,
    area_weights_from_lat,
    bootstrap_quantities,
    circular_block_bootstrap_indices,
    cubic_latitude_interpolation,
    derive_spherical_ell_max,
    exact_cell_area_row_weights,
    one_sided_zonal_power,
    region_masks,
    season_name,
    unet_pipeline_row_weights,
    zonal_row_weighted_power,
)

DATA_ROOT = os.environ.get("DATA_ROOT", "data_root")                                                     

RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "results_root")                                                  

FORBIDDEN_SUBSTRINGS = (

    "best_val_mse.pt", "selected_cca_model", "y_residual_eof_basis",
    "x_input_eof_basis", "val_x_scores", "checkpoint",

    "awi_downscaling_train", "awi_downscaling_val",
)


def guard(path) -> Path:
    p = Path(path)
    low = str(p).lower()
    for token in FORBIDDEN_SUBSTRINGS:
        if token in low:
            raise SystemExit(f"BLOCKED: artifact-only program may not open {p} "
                             f"(matched {token!r})")
    return p


def assert_artifact_only() -> None:
    for module in ("torch", "training.global_unet",
                   "training.global_unet_second_stage"):
        if module in sys.modules:
            raise SystemExit(f"BLOCKED: {module} is imported; this program is "
                             f"artifact-only")


TARGET_STD_K = 21.627892139211994

TARGET_MEAN_K = 277.8518781731243

ELL_MAX = 426

K_MAX = 437

MH_LABEL = "MH (2076–2078)"                                              


def set_labels_from_meta(meta: dict) -> None:

    global MH_LABEL
    first, last = meta["dates"][0][:4], meta["dates"][-1][:4]
    MH_LABEL = f"MH ({first}–{last})"


REGIONS = ["global", "land", "ocean", "elevation_gt_1000m", "elevation_le_1000m",
           "tropics", "midlat_north", "midlat_south", "highlat_north",
           "highlat_south", "arctic_gt_80N", "antarctic_lt_80S"]

PERIODS = ["annual", "DJF", "MAM", "JJA", "SON"]

T1_REGIONS = ["global", "land", "ocean", "elevation_gt_1000m",
              "arctic_gt_80N", "antarctic_lt_80S"]

HR_STATICS = Path(f"{DATA_ROOT}/grids/static_masks/"
                  "surface_fractions_hr.nc")

ORO_NC = Path(f"{DATA_ROOT}/grids/oro_hr.nc")

PD_ROOT = Path(f"{RESULTS_ROOT}/"
               "paper_pipeline_pd_test_r2_20260728T130323Z")

PD_F4_NPZ = PD_ROOT / "plot_sources" / "F4__PD_global_rmse_maps_source.npz"

PD_F7_ERR_SPH = PD_ROOT / "plot_sources" / "F7__prediction_error_spherical.csv"

PD_F7_ERR_ZON = PD_ROOT / "plot_sources" / "F7__prediction_error_zonal.csv"

PD_F7_ABS_SPH = PD_ROOT / "plot_sources" / "F7__absolute_field_spherical.csv"

PD_F7_ABS_ZON = PD_ROOT / "plot_sources" / "F7__absolute_field_zonal.csv"

PD_REGIONAL_JSON = PD_ROOT / "reports" / "regional_period_rmse.json"

CANONICAL_PD_TEST_STORE = Path(f"{DATA_ROOT}/zarr/"
                               "awi_downscaling_test.zarr")

NORM_T2M_INP_MEAN = 277.8568272051984

NORM_T2M_INP_STD = 21.581598298286263

TISR_MEAN_JM2 = 3223042.5108223786

TISR_STD_JM2 = 1786193.5906606768

F11_T2M_EDGES = np.linspace(TARGET_MEAN_K - 4.0 * TARGET_STD_K,
                            TARGET_MEAN_K + 4.0 * TARGET_STD_K, 61)

F11_TISR_EDGES = np.linspace(0.0, TISR_MEAN_JM2 + 4.0 * TISR_STD_JM2, 61)

T0 = time.time()

OUT: Path = Path(".")

SRC: Path = Path(".")

FIG: Path = Path(".")

PREV: Path = Path(".")

CAP: Path = Path(".")

PROV: Path = Path(".")

REP: Path = Path(".")

TAB: Path = Path(".")


def log(message: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {message}", flush=True)


def set_roots(output_root: Path, sources: Path | None) -> None:
    global OUT, SRC, FIG, PREV, CAP, PROV, REP, TAB
    OUT = guard(output_root)
    SRC = guard(sources) if sources else OUT / "plot_sources"
    FIG = OUT / "figures"
    PREV = OUT / "previews"
    CAP = OUT / "captions"
    PROV = OUT / "provenance"
    REP = OUT / "reports"
    TAB = OUT / "tables"
    for d in (FIG, PREV, CAP, PROV, REP, TAB, OUT / "plot_sources"):
        d.mkdir(parents=True, exist_ok=True)


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jdump(obj, path) -> Path:
    def default(o):
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"not JSON serializable: {type(o).__name__}")
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=default) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_csv(path):
    with open(guard(path), newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path, header, rows) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    return Path(path)


def lon180(lon):
    return ((np.asarray(lon, dtype=np.float64) + 180.0) % 360.0) - 180.0


BOOTSTRAP = {"block_length_days": 60, "n_resamples": 10000, "seed": 20260715,
             "ci_percentiles": [2.5, 97.5],
             "scheme": "paired circular moving-block bootstrap"}


def grid_lat_centers(height: int) -> np.ndarray:
    return -90.0 + (np.arange(height, dtype=np.float64) + 0.5) * (180.0 / height)


def load_statics(height: int, width: int):
    import netCDF4
    with netCDF4.Dataset(guard(HR_STATICS), "r") as ds:
        lsm = np.asarray(ds.variables["lsm"][:], dtype=np.float64)
        lat_file = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        lon_file = np.asarray(ds.variables["lon"][:], dtype=np.float64)
    with netCDF4.Dataset(guard(ORO_NC), "r") as ds:
        oro = np.asarray(ds.variables["var129"][:], dtype=np.float64)
    lsm = lsm.reshape(lsm.shape[-2], lsm.shape[-1])
    oro_m = oro.reshape(oro.shape[-2], oro.shape[-1])
    if lsm.shape != (height, width) or oro_m.shape != (height, width):
        raise SystemExit(f"statics shape mismatch {lsm.shape}/{oro_m.shape}")
    return lsm, oro_m, lat_file, lon_file


def baseline_from_inputs(input_t2m_norm, tgt_mean: float, tgt_std: float):


    scale = NORM_T2M_INP_STD / tgt_std
    offset = (NORM_T2M_INP_MEAN - tgt_mean) / tgt_std
    return np.asarray(input_t2m_norm, dtype=np.float64) * scale + offset


def point_estimates(daily: dict, std: float) -> dict:

    idx = np.arange(daily["w_c"].size, dtype=np.int64)[None, :]
    return {k: float(v[0]) for k, v in bootstrap_quantities(daily, idx, std).items()}


def read_retention(retention_root: Path) -> dict:
    meta = json.loads(guard(retention_root / "retention_metadata.json").read_text())
    if meta.get("kind") != "PAPER_PIPELINE_RETENTION":
        raise SystemExit(f"unexpected retention metadata kind {meta.get('kind')}")
    return meta


def open_shard(retention_root: Path, meta: dict, shard: int) -> dict:

    part = meta["shards"][shard]
    out = {"lo": part["lo"], "hi": part["hi"], "n_days": part["n_days"],
           "dates": meta["dates"][part["lo"]:part["hi"]],
           "store_indices": meta["store_indices"][part["lo"]:part["hi"]]}
    base = retention_root / "predictions"
    for name, rel in (
            ("b2_prediction_norm", f"b2/shard_{shard:02d}/b2_prediction_norm.npy"),
            ("b2_baseline_norm", f"b2/shard_{shard:02d}/b2_baseline_norm.npy"),
            ("cca_prediction_norm",
             f"cca/shard_{shard:02d}/cca_prediction_norm.npy")):
        path = guard(base / rel)
        if not path.exists():
            raise SystemExit(f"missing retained array for shard {shard}: {path}")
        arr = np.load(path, mmap_mode="r")
        if arr.shape[0] != out["n_days"]:
            raise SystemExit(f"{path}: {arr.shape[0]} days, expected "
                             f"{out['n_days']}")
        out[name] = arr
    if str(out["b2_prediction_norm"].dtype) != "float32":
        raise SystemExit("B2 native float32 precision not preserved")
    if str(out["cca_prediction_norm"].dtype) != "float64":
        raise SystemExit("CCA native float64 precision not preserved")
    return out


SPECTRA_ABS_FIELDS = ["HR target", "Bilinear", "CCA", "U-Net"]

SPECTRA_ERR_FIELDS = ["bilinear", "cca", "unet"]

SUM_KEYS = ["w", "e2", "ea", "e1", "bil_e2", "bil_ea", "bil_e1"]

F11_VARIABLES = ["t2m_K", "tisr_Jm2"]


def cmd_prepare_shard(args) -> int:


    import zarr

    set_roots(Path(args.output_root), None)
    retention_root = guard(args.retention_root)
    meta = read_retention(retention_root)
    s = args.shard_index
    if meta["n_shards"] != args.n_shards:
        raise SystemExit(f"shard count mismatch {meta['n_shards']} vs "
                         f"{args.n_shards}")
    set_labels_from_meta(meta)
    sh = open_shard(retention_root, meta, s)
    dates = sh["dates"]
    store_idx = sh["store_indices"]
    n = sh["n_days"]
    height = meta["grid"]["height"]
    width = meta["grid"]["width"]
    std = float(meta["normalization"]["t2m_tgt_std_K"])

    partial_dir = OUT / "partials_complete"
    partial_dir.mkdir(parents=True, exist_ok=True)
    out_path = partial_dir / f"partial_shard_{s:02d}.npz"


    lsm, oro_m, lat_file, lon_file = load_statics(height, width)
    masks = region_masks(lat_file, lsm, oro_m)
    if list(masks) != REGIONS:
        raise SystemExit("region order changed")
    w_c, _ = area_weights_from_lat(grid_lat_centers(height))
    w_u = unet_pipeline_row_weights(lat_file)
    mask_flat = np.stack([np.asarray(masks[r], dtype=np.float64).ravel()
                          for r in REGIONS])
    wm_c = mask_flat * np.broadcast_to(w_c[:, None], (height, width)).ravel()[None, :]
    wm_u = mask_flat * np.broadcast_to(w_u[:, None], (height, width)).ravel()[None, :]
    del mask_flat


    group = zarr.open_group(str(guard(meta["input_store"])), mode="r")
    tgt_mean = float(meta["normalization"]["t2m_tgt_mean_K"])


    pd_group = zarr.open_group(str(guard(CANONICAL_PD_TEST_STORE)), mode="r")
    if pd_group["inputs"].shape[0] != len(meta["dates"]):
        raise SystemExit("PD-test store day count does not match MH retention; "
                         "positional F11 pairing is invalid")


    n_reg, n_per = len(REGIONS), len(PERIODS)
    acc = {f"{pipe}_{k}": np.zeros((n_per, n_reg), dtype=np.float64)
           for pipe in ("cca", "unet") for k in SUM_KEYS}
    daily = {k: np.zeros(n, dtype=np.float64)
             for k in ("w_c", "bil_e2_c", "cca_e2", "w_u", "bil_e2_u", "unet_e2")}
    period_days = np.zeros(n_per, dtype=np.int64)
    map_acc = {k: np.zeros((height, width), dtype=np.float64)
               for k in ("bilinear_e2", "cca_e2", "unet_e2")}

    plan = GaussLegendreSHTPlan(grid_lat_centers(height),
                                np.sort(lon180(lon_file)))
    ell_max = plan.ell_max
    sph_abs = np.zeros((len(SPECTRA_ABS_FIELDS), ell_max + 1), dtype=np.float64)
    sph_err = np.zeros((len(SPECTRA_ERR_FIELDS), ell_max + 1), dtype=np.float64)
    zon_abs = np.zeros((len(SPECTRA_ABS_FIELDS), K_MAX + 1), dtype=np.float64)
    zon_err = np.zeros((len(SPECTRA_ERR_FIELDS), K_MAX + 1), dtype=np.float64)
    zon_row_weight_sum = 0.0


    whist = np.ascontiguousarray(
        np.broadcast_to(w_c[:, None], (height, width))).ravel()
    hist = {f"{state}_{var}": np.zeros(60, dtype=np.float64)
            for state in ("mh", "pd") for var in F11_VARIABLES}
    hist_out = {f"{state}_{var}_{side}": 0.0
                for state in ("mh", "pd") for var in F11_VARIABLES
                for side in ("under_w", "over_w")}
    log(f"shard {s:02d}: {n} days {dates[0]}..{dates[-1]}, ell_max={ell_max}")

    def add_hist(state: str, var: str, values_K: np.ndarray, edges) -> None:
        v = values_K.ravel()
        c, _ = np.histogram(v, bins=edges, weights=whist)
        hist[f"{state}_{var}"] += c
        hist_out[f"{state}_{var}_under_w"] += float(whist[v < edges[0]].sum())
        hist_out[f"{state}_{var}_over_w"] += float(whist[v > edges[-1]].sum())

    for k in range(n):
        date = dates[k]
        si = store_idx[k]
        period_ids = (0, PERIODS.index(season_name(date)))

        b2_pred = np.asarray(sh["b2_prediction_norm"][k], dtype=np.float64)
        b2_base = np.asarray(sh["b2_baseline_norm"][k], dtype=np.float64)
        cca_pred = np.asarray(sh["cca_prediction_norm"][k], dtype=np.float64)
        tgt_u = np.asarray(group["targets"][si, 0], dtype=np.float64)


        inp_norm = np.asarray(group["inputs"][si, 0], dtype=np.float64)
        base_c = baseline_from_inputs(inp_norm, tgt_mean, float(std))

        err = {"cca": (cca_pred - tgt_u, base_c - tgt_u, wm_c, "cca"),
               "unet": (b2_pred - tgt_u, b2_base - tgt_u, wm_u, "unet")}
        for pipe, (e, be, wm, _tag) in err.items():
            flat_e, flat_be = e.ravel(), be.ravel()
            sums = {
                "w": wm.sum(axis=1),
                "e2": wm @ (flat_e * flat_e),
                "ea": wm @ np.abs(flat_e),
                "e1": wm @ flat_e,
                "bil_e2": wm @ (flat_be * flat_be),
                "bil_ea": wm @ np.abs(flat_be),
                "bil_e1": wm @ flat_be,
            }
            for key, vec in sums.items():
                for p in period_ids:
                    acc[f"{pipe}_{key}"][p] += vec
        for p in period_ids:
            period_days[p] += 1

        gi = REGIONS.index("global")
        daily["w_c"][k] = float(wm_c[gi].sum())
        daily["cca_e2"][k] = float(wm_c[gi] @ (err["cca"][0].ravel() ** 2))
        daily["bil_e2_c"][k] = float(wm_c[gi] @ (err["cca"][1].ravel() ** 2))
        daily["w_u"][k] = float(wm_u[gi].sum())
        daily["unet_e2"][k] = float(wm_u[gi] @ (err["unet"][0].ravel() ** 2))
        daily["bil_e2_u"][k] = float(wm_u[gi] @ (err["unet"][1].ravel() ** 2))

        map_acc["cca_e2"] += err["cca"][0] ** 2
        map_acc["bilinear_e2"] += err["cca"][1] ** 2
        map_acc["unet_e2"] += err["unet"][0] ** 2


        order = np.argsort(lon180(lon_file))
        abs_fields = np.stack([tgt_u[:, order], b2_base[:, order],
                               cca_pred[:, order], b2_pred[:, order]])
        sph_abs += plan.transform_c_ell(abs_fields)
        err_fields = np.stack([err["unet"][1][:, order],
                               err["cca"][0][:, order],
                               err["unet"][0][:, order]])
        sph_err += plan.transform_c_ell(err_fields)
        zon_abs += zonal_row_weighted_power(abs_fields, w_c, K_MAX)
        zon_err += zonal_row_weighted_power(err_fields, w_c, K_MAX)
        zon_row_weight_sum += float(w_c.sum())


        add_hist("mh", "t2m_K",
                 inp_norm * NORM_T2M_INP_STD + NORM_T2M_INP_MEAN, F11_T2M_EDGES)
        mh_tisr = (np.asarray(group["inputs"][si, 1], dtype=np.float64)
                   * TISR_STD_JM2 + TISR_MEAN_JM2)
        add_hist("mh", "tisr_Jm2", mh_tisr, F11_TISR_EDGES)
        pd_t2m = (np.asarray(pd_group["inputs"][si, 0], dtype=np.float64)
                  * NORM_T2M_INP_STD + NORM_T2M_INP_MEAN)
        add_hist("pd", "t2m_K", pd_t2m, F11_T2M_EDGES)
        pd_tisr = (np.asarray(pd_group["inputs"][si, 1], dtype=np.float64)
                   * TISR_STD_JM2 + TISR_MEAN_JM2)
        add_hist("pd", "tisr_Jm2", pd_tisr, F11_TISR_EDGES)

        if (k + 1) % 10 == 0 or k == n - 1:
            log(f"shard {s:02d}: {k + 1}/{n} days")

    payload = {f"acc_{k}": v for k, v in acc.items()}
    payload.update({f"daily_{k}": v for k, v in daily.items()})
    payload.update({f"map_{k}": v for k, v in map_acc.items()})
    payload.update({f"hist_{k}": v for k, v in hist.items()})
    payload.update({f"histmeta_{k}": np.float64(v) for k, v in hist_out.items()})
    payload.update({
        "shard": np.int64(s), "lo": np.int64(sh["lo"]), "hi": np.int64(sh["hi"]),
        "n_days": np.int64(n), "dates": np.asarray(dates),
        "period_days": period_days, "regions": np.asarray(REGIONS),
        "periods": np.asarray(PERIODS), "sum_keys": np.asarray(SUM_KEYS),
        "ell": np.arange(ell_max + 1, dtype=np.int64),
        "spherical_absolute_c_ell_sum_norm2": sph_abs,
        "spherical_error_c_ell_sum_norm2": sph_err,
        "spherical_absolute_fields": np.asarray(SPECTRA_ABS_FIELDS),
        "spherical_error_fields": np.asarray(SPECTRA_ERR_FIELDS),
        "k": np.arange(K_MAX + 1, dtype=np.int64),
        "zonal_absolute_power_sum_norm2": zon_abs,
        "zonal_error_power_sum_norm2": zon_err,
        "zonal_row_weight_sum": np.float64(zon_row_weight_sum),
        "f11_t2m_edges": F11_T2M_EDGES,
        "f11_tisr_edges": F11_TISR_EDGES,
        "target_std_K": np.float64(std),
    })
    tmp = out_path.with_suffix(".writing.npz")
    np.savez(tmp, **payload)
    os.replace(tmp, out_path)
    log(f"shard {s:02d}: partial written -> {out_path}")
    print(json.dumps({"status": "PASS", "mode": "prepare-shard", "shard": s,
                      "n_days": n, "partial": str(out_path)}))
    return 0


def pd_regional_mse_K2() -> dict:

    payload = json.loads(guard(PD_REGIONAL_JSON).read_text())
    if payload["regions"] != REGIONS or payload["periods"] != PERIODS:
        raise SystemExit("PD regional report region/period order changed")
    out = {}
    for key, entry in payload["regional_period_rmse_K"].items():
        region, period = key.split("|")
        mse_cca = float(entry["cca"]) ** 2
        mse_unet = float(entry["unet"]) ** 2

        stored = float(entry["skill_vs_cca"])
        derived = 1.0 - mse_unet / mse_cca
        if abs(stored - derived) > 1e-9:
            raise SystemExit(f"PD skill inconsistency at {key}: "
                             f"{stored} vs {derived}")
        out[(region, period)] = {"mse_cca_K2": mse_cca, "mse_unet_K2": mse_unet}
    return out


def cmd_reduce(args) -> int:
    set_roots(Path(args.output_root), None)
    retention_root = guard(args.retention_root)
    meta = read_retention(retention_root)
    set_labels_from_meta(meta)
    n_shards = meta["n_shards"]
    height, width = meta["grid"]["height"], meta["grid"]["width"]
    std = float(meta["normalization"]["t2m_tgt_std_K"])


    missing = []
    for s in range(n_shards):
        for rel in (f"predictions/cca/shard_{s:02d}/cca_prediction_norm.npy",
                    f"predictions/b2/shard_{s:02d}/b2_prediction_norm.npy",
                    f"predictions/b2/shard_{s:02d}/b2_baseline_norm.npy"):
            if not (retention_root / rel).exists():
                missing.append(str(retention_root / rel))
        if not (OUT / "partials_complete" / f"partial_shard_{s:02d}.npz").exists():
            missing.append(str(OUT / "partials_complete"
                               / f"partial_shard_{s:02d}.npz"))
    if missing:
        raise SystemExit("REDUCE FAILED — missing required inputs:\n  "
                         + "\n  ".join(missing))

    n_reg, n_per = len(REGIONS), len(PERIODS)
    acc = {f"{pipe}_{k}": np.zeros((n_per, n_reg), dtype=np.float64)
           for pipe in ("cca", "unet") for k in SUM_KEYS}
    maps = {k: np.zeros((height, width), dtype=np.float64)
            for k in ("bilinear_e2", "cca_e2", "unet_e2")}
    daily_parts = {k: [] for k in ("w_c", "bil_e2_c", "cca_e2",
                                   "w_u", "bil_e2_u", "unet_e2")}
    hist = {f"{state}_{var}": np.zeros(60, dtype=np.float64)
            for state in ("mh", "pd") for var in F11_VARIABLES}
    hist_out = {f"{state}_{var}_{side}": 0.0
                for state in ("mh", "pd") for var in F11_VARIABLES
                for side in ("under_w", "over_w")}
    covered, period_days, sph_abs, sph_err, ell = [], None, None, None, None
    zon_abs = zon_err = kk = None
    zon_row_weight_sum = 0.0

    for s in range(n_shards):
        with np.load(OUT / "partials_complete" / f"partial_shard_{s:02d}.npz",
                     allow_pickle=False) as d:
            if int(d["shard"]) != s:
                raise SystemExit(f"partial {s} carries shard id {int(d['shard'])}")
            for key in acc:
                acc[key] += d[f"acc_{key}"]
            for key in maps:
                maps[key] += d[f"map_{key}"]
            for key in daily_parts:
                daily_parts[key].append(d[f"daily_{key}"])
            for key in hist:
                hist[key] += d[f"hist_{key}"]
            for key in hist_out:
                hist_out[key] += float(d[f"histmeta_{key}"])
            if not (np.array_equal(d["f11_t2m_edges"], F11_T2M_EDGES)
                    and np.array_equal(d["f11_tisr_edges"], F11_TISR_EDGES)):
                raise SystemExit(f"partial {s}: F11 bin edges differ from the "
                                 f"predeclared policy")
            covered.extend([str(x) for x in d["dates"]])
            period_days = (d["period_days"].copy() if period_days is None
                           else period_days + d["period_days"])
            sph_abs = (d["spherical_absolute_c_ell_sum_norm2"].copy()
                       if sph_abs is None
                       else sph_abs + d["spherical_absolute_c_ell_sum_norm2"])
            sph_err = (d["spherical_error_c_ell_sum_norm2"].copy() if sph_err is None
                       else sph_err + d["spherical_error_c_ell_sum_norm2"])
            ell = d["ell"] if ell is None else ell
            zon_abs = (d["zonal_absolute_power_sum_norm2"].copy() if zon_abs is None
                       else zon_abs + d["zonal_absolute_power_sum_norm2"])
            zon_err = (d["zonal_error_power_sum_norm2"].copy() if zon_err is None
                       else zon_err + d["zonal_error_power_sum_norm2"])
            zon_row_weight_sum += float(d["zonal_row_weight_sum"])
            kk = d["k"] if kk is None else kk


    if covered != meta["dates"]:
        raise SystemExit(f"date coverage mismatch: {len(covered)} covered vs "
                         f"{len(meta['dates'])} retained")
    if len(set(covered)) != len(covered):
        raise SystemExit("duplicated dates across shards")
    n_days = len(covered)
    daily = {k: np.concatenate(v) for k, v in daily_parts.items()}
    log(f"reduced {n_shards} shards, {n_days} days, coverage exact")


    point = point_estimates(daily, std)
    rng = np.random.default_rng(BOOTSTRAP["seed"])
    idx = circular_block_bootstrap_indices(n_days, BOOTSTRAP["block_length_days"],
                                           BOOTSTRAP["n_resamples"], rng)
    draws = bootstrap_quantities(daily, idx, std)
    lo_p, hi_p = BOOTSTRAP["ci_percentiles"]
    ci = {k: (float(np.percentile(v, lo_p)), float(np.percentile(v, hi_p)))
          for k, v in draws.items()}
    log("bootstrap complete")


    def rmse(pipe, key, p, r):
        return math.sqrt(acc[f"{pipe}_{key}"][p, r] / acc[f"{pipe}_w"][p, r]) * std

    regional = {}
    for ri, reg in enumerate(REGIONS):
        for pi, per in enumerate(PERIODS):
            mse_cca = acc["cca_e2"][pi, ri] / acc["cca_w"][pi, ri]
            mse_unet = acc["unet_e2"][pi, ri] / acc["unet_w"][pi, ri]
            regional[(reg, per)] = {
                "bilinear": rmse("cca", "bil_e2", pi, ri),
                "cca": rmse("cca", "e2", pi, ri),
                "unet": rmse("unet", "e2", pi, ri),
                "mse_cca_K2": mse_cca * std ** 2,
                "mse_unet_K2": mse_unet * std ** 2,
                "delta_rmse_K": (math.sqrt(mse_cca) - math.sqrt(mse_unet)) * std,
                "skill_vs_cca": 1.0 - mse_unet / mse_cca,
            }


    write_csv(SRC / "T2__MH_regional_rmse_source.csv",
              ["region", "method", "rmse_K"],
              [[r, m, f"{regional[(r, 'annual')][m]:.6f}"]
               for r in T1_REGIONS for m in ("bilinear", "cca", "unet")])
    order = [("bilinear_rmse_K", "Bilinear", "RMSE (K)", "—", "lower is better"),
             ("cca_rmse_K", "CCA", "RMSE (K)", "—", "lower is better"),
             ("unet_rmse_K", "U-Net", "RMSE (K)", "—", "lower is better"),
             ("cca_skill_vs_bilinear", "CCA", "MSE skill vs bilinear (—)",
              "Bilinear", "higher is better; 1−MSE_CCA/MSE_bil"),
             ("unet_skill_vs_bilinear", "U-Net", "MSE skill vs bilinear (—)",
              "Bilinear", "higher is better; 1−MSE_UNet/MSE_bil"),
             ("delta_rmse_K", "CCA−U-Net", "ΔRMSE (K)", "CCA",
              "ΔRMSE=RMSE_CCA−RMSE_UNet; positive = U-Net better"),
             ("skill_vs_cca", "U-Net", "MSE skill vs CCA (—)", "CCA",
              "1−MSE_UNet/MSE_CCA; positive = U-Net better")]
    t2_csv = write_csv(
        SRC / "T2__MH_global_metrics.csv",
        ["quantity", "method_or_pair", "estimate", "ci_lo_95", "ci_hi_95",
         "units", "reference_method", "sign_interpretation", "bootstrap_scheme",
         "block_length_days", "n_resamples"],
        [[q, meth, repr(point[q]), repr(ci[q][0]), repr(ci[q][1]), unit, ref,
          sign, BOOTSTRAP["scheme"], BOOTSTRAP["block_length_days"],
          BOOTSTRAP["n_resamples"]] for q, meth, unit, ref, sign in order])


    lat_g = np.load(retention_root / meta["grid"]["coords"]["hr_lat"])
    lon_g = np.load(retention_root / meta["grid"]["coords"]["hr_lon"])
    np.savez_compressed(
        SRC / "F8__MH_global_rmse_maps_source.npz",
        latitude=lat_g, longitude=lon_g,
        bilinear_rmse_K=np.sqrt(maps["bilinear_e2"] / n_days) * std,
        cca_rmse_K=np.sqrt(maps["cca_e2"] / n_days) * std,
        unet_rmse_K=np.sqrt(maps["unet_e2"] / n_days) * std,
        delta_rmse_K=(np.sqrt(maps["cca_e2"] / n_days)
                      - np.sqrt(maps["unet_e2"] / n_days)) * std)


    pd_reg = pd_regional_mse_K2()
    write_csv(SRC / "F9__MH_PD_shared_sufficient_statistics.csv",
              ["region", "period", "pd_mse_cca_K2", "pd_mse_unet_K2",
               "mh_mse_cca_K2", "mh_mse_unet_K2"],
              [[r, p, repr(pd_reg[(r, p)]["mse_cca_K2"]),
                repr(pd_reg[(r, p)]["mse_unet_K2"]),
                repr(float(regional[(r, p)]["mse_cca_K2"])),
                repr(float(regional[(r, p)]["mse_unet_K2"]))]
               for r in REGIONS for p in PERIODS])


    ell = np.asarray(ell)
    keep = (ell >= 1) & (ell <= ELL_MAX)
    mh_abs_sph = {name: sph_abs[fi] / n_days * std ** 2
                  for fi, name in enumerate(SPECTRA_ABS_FIELDS)}
    mh_err_sph = {name: sph_err[fi] / n_days * std ** 2
                  for fi, name in enumerate(SPECTRA_ERR_FIELDS)}
    kk = np.asarray(kk)
    kkeep = (kk >= 1) & (kk <= K_MAX)
    zon_denom = zon_row_weight_sum * float(width)
    mh_abs_zon = {name: zon_abs[fi] / zon_denom * std ** 2
                  for fi, name in enumerate(SPECTRA_ABS_FIELDS)}
    mh_err_zon = {name: zon_err[fi] / zon_denom * std ** 2
                  for fi, name in enumerate(SPECTRA_ERR_FIELDS)}


    pd_err_sph = load_csv(PD_F7_ERR_SPH)
    pd_err_zon = load_csv(PD_F7_ERR_ZON)
    pd_abs_sph = load_csv(PD_F7_ABS_SPH)
    pd_abs_zon = load_csv(PD_F7_ABS_ZON)

    err_rows = []
    for m in SPECTRA_ERR_FIELDS:
        pd_by_ell = {int(r["ell"]): float(r[f"{m}_error_K2"]) for r in pd_err_sph}
        for e in ell[keep]:
            err_rows.append(["spherical", m, int(e),
                             f"{pd_by_ell[int(e)]:.10e}",
                             f"{mh_err_sph[m][int(e)]:.10e}"])
        pd_by_k = {int(r["k"]): float(r[f"{m}_error_K2"]) for r in pd_err_zon}
        for k_ in kk[kkeep]:
            err_rows.append(["zonal", m, int(k_),
                             f"{pd_by_k[int(k_)]:.10e}",
                             f"{mh_err_zon[m][int(k_)]:.10e}"])
    write_csv(SRC / "F10__MH_prediction_error_source.csv",
              ["spectrum", "method", "index", "pd_power_K2", "mh_power_K2"],
              err_rows)

    abs_rows = []
    for name in SPECTRA_ABS_FIELDS:
        pd_by_ell = {int(r["ell"]): float(r["C_ell_K2"])
                     for r in pd_abs_sph if r["series"] == name}
        for e in ell[keep]:
            abs_rows.append(["spherical", name, int(e),
                             f"{pd_by_ell[int(e)]:.10e}",
                             f"{mh_abs_sph[name][int(e)]:.10e}"])
        pd_by_k = {int(r["k"]): float(r["power_K2"])
                   for r in pd_abs_zon if r["series"] == name}
        for k_ in kk[kkeep]:
            abs_rows.append(["zonal", name, int(k_),
                             f"{pd_by_k[int(k_)]:.10e}",
                             f"{mh_abs_zon[name][int(k_)]:.10e}"])
    write_csv(SRC / "F10__MH_absolute_field_source.csv",
              ["spectrum", "series", "index", "pd_power_K2", "mh_power_K2"],
              abs_rows)


    f11_rows = []
    fractions = {}
    for var, edges in (("t2m_K", F11_T2M_EDGES), ("tisr_Jm2", F11_TISR_EDGES)):
        centers = 0.5 * (edges[:-1] + edges[1:])
        pd_c = hist[f"pd_{var}"]
        mh_c = hist[f"mh_{var}"]
        if pd_c.sum() <= 0 or mh_c.sum() <= 0:
            raise SystemExit(f"empty F11 histogram for {var}")
        pd_f = pd_c / pd_c.sum()
        mh_f = mh_c / mh_c.sum()
        fractions[var] = (pd_f, mh_f)
        for c, pf, mf in zip(centers, pd_f, mh_f):
            f11_rows.append([var, f"{c:.6f}", f"{pf:.8f}", f"{mf:.8f}",
                             f"{mf - pf:.8f}"])
    write_csv(SRC / "F11__MH_PD_distribution_shift_source.csv",
              ["variable", "bin_center", "pd_fraction", "mh_fraction",
               "mh_minus_pd"], f11_rows)
    jdump({"kind": "F11_BIN_POLICY", "created_utc": utc(),
           "policy": "approved predeclared bins from frozen PD normalization "
                     "statistics only; never chosen from MH values",
           "t2m_edges_K": F11_T2M_EDGES,
           "t2m_edge_rule": "linspace(t2m_tgt_mean - 4*std, t2m_tgt_mean + "
                            "4*std, 61) from frozen norm_stats",
           "tisr_edges_Jm2": F11_TISR_EDGES,
           "tisr_edge_rule": "linspace(0, tisr_mean + 4*tisr_std, 61) from "
                             "frozen norm_stats",
           "weights": "frozen float64 mean-one cosine (CCA convention)",
           "fields": {"t2m_K": "store input channel 0 (bilinear input T2M), "
                               "denormalized with frozen t2m_inp stats",
                      "tisr_Jm2": "store input channel 1, denormalized with "
                                  "frozen tisr stats; J m-2"},
           "pd_reference_store": str(CANONICAL_PD_TEST_STORE),
           "weighted_outside_bins": hist_out},
          REP / "F11__bin_policy.json")

    pd_reference_hashes = {
        str(p): sha256(p) for p in (PD_F4_NPZ, PD_F7_ERR_SPH, PD_F7_ERR_ZON,
                                    PD_F7_ABS_SPH, PD_F7_ABS_ZON,
                                    PD_REGIONAL_JSON)}

    reduced = {
        "kind": "REDUCE_REPORT", "created_utc": utc(),
        "population": MH_LABEL,
        "n_shards": n_shards, "n_days": n_days,
        "date_coverage_exact_once": True,
        "first_date": covered[0], "last_date": covered[-1],
        "period_days": {p: int(period_days[i]) for i, p in enumerate(PERIODS)},
        "target_std_K": std,
        "headline_point_estimates": point,
        "headline_ci_95": {k: list(v) for k, v in ci.items()},
        "bootstrap": BOOTSTRAP,
        "weighted_denominators": {
            "cca_pipeline_total": float(acc["cca_w"][0, REGIONS.index("global")]),
            "unet_pipeline_total": float(acc["unet_w"][0, REGIONS.index("global")])},
        "spherical_ell_max_accumulated": int(ell[-1]),
        "spherical_display_range": [1, ELL_MAX],
        "zonal_k_display_range": [1, K_MAX],
        "zonal_row_weight_denominator": zon_row_weight_sum,
        "zonal_denominator_row_weight_times_nlon": zon_denom,
        "zonal_convention": "one-sided rfft(norm='ortho') after row-mean "
                            "removal, interior bins doubled; latitude-weighted "
                            "sum divided by n_days*sum(row weights)*nlon "
                            "(Parseval), then x target_std^2",
        "mh_sources_recomputed_from_this_run": [
            "T2__MH_regional_rmse_source.csv", "T2__MH_global_metrics.csv",
            "F8__MH_global_rmse_maps_source.npz",
            "F9__MH_PD_shared_sufficient_statistics.csv",
            "F10__MH_prediction_error_source.csv",
            "F10__MH_absolute_field_source.csv",
            "F11__MH_PD_distribution_shift_source.csv"],
        "pd_reference_read_only": pd_reference_hashes,
        "no_mh_derived_statistics": True,
        "plot_sources_dir": str(SRC),
        "t2_csv": str(t2_csv),
    }
    jdump(reduced, REP / "reduce_report.json")
    jdump({"regions": REGIONS, "periods": PERIODS,
           "regional_period_rmse_K": {f"{r}|{p}": regional[(r, p)]
                                      for r in REGIONS for p in PERIODS}},
          REP / "regional_period_rmse.json")
    print(json.dumps({"status": "PASS", "mode": "reduce", "n_days": n_days,
                      "headline": point}, indent=2))
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    ps = sub.add_parser("prepare-shard", help="accumulate one shard of numerical sufficient statistics")
    ps.add_argument("--retention-root", required=True)
    ps.add_argument("--output-root", required=True)
    ps.add_argument("--shard-index", type=int, required=True)
    ps.add_argument("--n-shards", type=int, default=20)
    ps.set_defaults(func=cmd_prepare_shard)
    rd = sub.add_parser("reduce", help="combine the original numerical partials and write result tables")
    rd.add_argument("--retention-root", required=True)
    rd.add_argument("--output-root", required=True)
    rd.set_defaults(func=cmd_reduce)

    return p.parse_args(argv)


def main(argv=None) -> int:
    assert_artifact_only()
    args = parse_args(argv)
    rc = args.func(args)
    assert_artifact_only()
    return rc


if __name__ == "__main__":
    sys.exit(main())
