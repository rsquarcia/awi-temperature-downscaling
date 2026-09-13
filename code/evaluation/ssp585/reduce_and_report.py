"""Calculate SSP5-8.5 metrics and final three-climate RMSE-skill statistics.
Keeps the original shard reduction, target histograms, spatial-transfer
helpers and final paired circular moving-block bootstrap. The rmse-skill
command exports the final three-climate estimates and confidence intervals
from retained daily statistics. Figure rendering and review-package assembly
are omitted. Existing data, frozen settings and input checks are preserved;
this script requires external retained artifacts and does not run models."""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os

DATA_ROOT = os.environ.get("DATA_ROOT", "data_root")

RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "results_root")

import shutil
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

SKILL_DEFINITION = "S^RMSE_{M|R} = 1 - RMSE_M / RMSE_R"

SKILL_DEFINITION_LONG = (
    "Every quantity called 'skill' is the RMSE-ratio skill "
    "S^RMSE_{M|R} = 1 - RMSE_M / RMSE_R.  The squared-error form "
    "1 - MSE_M / MSE_R is NOT used anywhere.  For regional, seasonal, global "
    "or masked aggregates each RMSE is formed first from the correctly "
    "weighted aggregate squared-error sum and its denominator, and only then "
    "is the RMSE ratio taken; local skill scores are never averaged and no "
    "value is derived from a rounded published number.  Confidence intervals "
    "recompute the RMSE-based statistic inside every bootstrap replicate "
    "(paired 60-day circular moving-block bootstrap, 10,000 replicates, seed "
    "20260715, fixed climate draw order PD, MH, SSP5-8.5); no endpoint of an "
    "older MSE-skill interval is transformed.")

TRANSFER_LIMIT = 0.3

TRANSFER_TICKS = [-0.3, -0.15, 0.0, 0.15, 0.3]

TRANSFER_SCALE_RULE = ("exact symmetric dimensionless limits -0.3 .. +0.3, "
                       "identical for F8 and F12, centred exactly at zero")

TRANSFER_CMAP = "RdBu"

TRANSFER_P = 0.99

SSP585_LABEL = "SSP5-8.5 (2096–2098)"                                        

FIG_SCENARIO_LABEL = "SSP5-8.5"                               


def set_labels_from_meta(meta: dict) -> None:

    global SSP585_LABEL
    first, last = meta["dates"][0][:4], meta["dates"][-1][:4]
    SSP585_LABEL = f"{FIG_SCENARIO_LABEL} ({first}–{last})"


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

MH_ROOT = Path(f"{RESULTS_ROOT}/mh_full_run_20260728T180741Z")

MH_F8_NPZ = MH_ROOT / "plot_sources" / "F8__MH_global_rmse_maps_source.npz"

MH_F9_CSV = (MH_ROOT / "plot_sources"
             / "F9__MH_PD_shared_sufficient_statistics.csv")

CANONICAL_MH_STORE = Path(f"{DATA_ROOT}/mh_preprocess_20260728T173229Z/"
                          "results/mh_2076_2078.zarr")

CLIMATES = ["pd", "mh", "ssp585"]

NORM_T2M_INP_MEAN = 277.8568272051984

NORM_T2M_INP_STD = 21.581598298286263

TISR_MEAN_JM2 = 3223042.5108223786

TISR_STD_JM2 = 1786193.5906606768

F11_T2M_LO, F11_T2M_HI = 100.0, 400.0

F11_T2M_NBINS_FINE = 30000

F11_T2M_AGGREGATE = 10

F11_T2M_NBINS_DISPLAY = F11_T2M_NBINS_FINE // F11_T2M_AGGREGATE

F11_TISR_LO = 0.0

F11_TISR_HI = TISR_MEAN_JM2 + 4.0 * TISR_STD_JM2

F11_TISR_NBINS = 3000

F11_TARGET_HIST_NPZ = "hr_target_t2m_fine_histograms.npz"

F11_TARGET_HIST_CSV = "revised_F11__PD_MH_SSP585_hr_target_t2m_source.csv"

F11_TARGET_HIST_REPORT = "hr_target_t2m_histogram_report.json"

F11_TARGET_DENORM = ("field_K = TARGET_MEAN_K + field_norm * TARGET_STD_K "
                     f"with the frozen TARGET statistics "
                     f"mean={TARGET_MEAN_K!r} K, std={TARGET_STD_K!r} K "
                     f"(NOT the input statistics "
                     f"mean={NORM_T2M_INP_MEAN!r} K, "
                     f"std={NORM_T2M_INP_STD!r} K)")

F11_T2M_FINE_EDGES = np.linspace(F11_T2M_LO, F11_T2M_HI,
                                 F11_T2M_NBINS_FINE + 1)

F11_T2M_DISPLAY_EDGES = np.linspace(F11_T2M_LO, F11_T2M_HI,
                                    F11_T2M_NBINS_DISPLAY + 1)

F11_TISR_EDGES = np.linspace(F11_TISR_LO, F11_TISR_HI, F11_TISR_NBINS + 1)

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


def assert_reference_roots_read_only() -> None:

    out = OUT.resolve()
    for root in (PD_ROOT, MH_ROOT):
        resolved = root.resolve()
        if out == resolved or resolved in out.parents or out in resolved.parents:
            raise SystemExit(f"BLOCKED: output root {out} overlaps the accepted "
                             f"read-only reference root {resolved}")


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
    path = assert_writable(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=default) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_csv(path):
    with open(guard(path), newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


READONLY_ROOTS: list = []


def assert_writable(path) -> Path:

    p = Path(path).resolve()
    for root in READONLY_ROOTS:
        r = Path(root).resolve()
        if p == r or r in p.parents:
            raise SystemExit(f"BLOCKED: refusing to write {p} inside the "
                             f"read-only accepted root {r}")
    return Path(path)


def write_csv(path, header, rows) -> Path:
    assert_writable(path)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    return Path(path)


def lon180(lon):
    return ((np.asarray(lon, dtype=np.float64) + 180.0) % 360.0) - 180.0


def source_path(name) -> Path:
    p = guard(SRC / name)
    if not p.exists():
        raise SystemExit(
            f"missing plot source {p}.\nRun the reduce mode first, or point "
            f"--sources at a directory that already holds it.")
    return p


def read_only_reference(path) -> Path:
    p = guard(path)
    if not p.exists():
        raise SystemExit(f"missing accepted read-only reference artifact: {p}")
    return p


BOOTSTRAP = {"block_length_days": 60, "n_resamples": 10000, "seed": 20260715,
             "ci_percentiles": [2.5, 97.5],
             "scheme": "paired circular moving-block bootstrap"}

CLIMATE_DRAW_ORDER = ["PD", "MH", "SSP5-8.5"]


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


def bootstrap_quantities(daily: dict, idx: np.ndarray, std: float) -> dict:

    def rs(x):
        return x[idx].sum(axis=1)

    sw_c = rs(daily["w_c"])
    sw_u = rs(daily["w_u"])
    mse_bil_c = rs(daily["bil_e2_c"]) / sw_c
    mse_cca = rs(daily["cca_e2"]) / sw_c
    mse_bil_u = rs(daily["bil_e2_u"]) / sw_u
    mse_unet = rs(daily["unet_e2"]) / sw_u
    rmse_bil_c = np.sqrt(mse_bil_c) * std
    rmse_bil_u = np.sqrt(mse_bil_u) * std
    rmse_cca = np.sqrt(mse_cca) * std
    rmse_unet = np.sqrt(mse_unet) * std
    return {
        "bilinear_rmse_K": rmse_bil_c,
        "bilinear_rmse_unet_pipeline_K": rmse_bil_u,
        "cca_rmse_K": rmse_cca,
        "unet_rmse_K": rmse_unet,


        "cca_skill_vs_bilinear": 1.0 - mse_cca / mse_bil_c,
        "unet_skill_vs_bilinear": 1.0 - mse_unet / mse_bil_u,
        "delta_rmse_K": rmse_cca - rmse_unet,
        "skill_vs_cca": 1.0 - mse_unet / mse_cca,


        "cca_rmse_skill_vs_bilinear": 1.0 - rmse_cca / rmse_bil_c,
        "unet_rmse_skill_vs_bilinear": 1.0 - rmse_unet / rmse_bil_u,
        "rmse_skill_vs_cca": 1.0 - rmse_unet / rmse_cca,
    }


def point_estimates(daily: dict, std: float) -> dict:

    idx = np.arange(daily["w_c"].size, dtype=np.int64)[None, :]
    return {k: float(v[0]) for k, v in bootstrap_quantities(daily, idx, std).items()}


def uniform_weighted_histogram(values, lo: float, hi: float, nbins: int,
                               weights) -> tuple[np.ndarray, float, float]:
    v = np.asarray(values, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    if v.size != w.size:
        raise SystemExit("histogram values and weights differ in size")
    if not np.isfinite(v).all():
        raise SystemExit("non-finite value in a histogram input field")
    width = (hi - lo) / float(nbins)
    idx = np.floor((v - lo) / width).astype(np.int64)
    idx[v == hi] = nbins - 1
    below = idx < 0
    above = idx >= nbins
    inside = ~(below | above)
    counts = np.bincount(idx[inside], weights=w[inside], minlength=nbins)
    return (np.asarray(counts, dtype=np.float64),
            float(w[below].sum()), float(w[above].sum()))


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


def daily_from_partials(retention_root: Path,
                        partials_root: Path | None = None
                        ) -> tuple[dict, float, list[str]]:


    retention_root = guard(retention_root)
    partials_root = guard(partials_root or retention_root)
    meta = read_retention(retention_root)
    keys = ("w_c", "bil_e2_c", "cca_e2", "w_u", "bil_e2_u", "unet_e2")
    parts = {k: [] for k in keys}
    dates: list[str] = []
    std = None
    for s in range(meta["n_shards"]):
        p = guard(partials_root / "partials_complete" / f"partial_shard_{s:02d}.npz")
        if not p.exists():
            raise SystemExit(f"missing preserved partial: {p}")
        with np.load(p, allow_pickle=False) as d:
            if int(d["shard"]) != s:
                raise SystemExit(f"{p}: carries shard id {int(d['shard'])}")
            for k in keys:
                parts[k].append(d[f"daily_{k}"])
            dates.extend(str(x) for x in d["dates"])
            std = float(d["target_std_K"])
    if dates != meta["dates"]:
        raise SystemExit(f"{retention_root}: partial date coverage does not "
                         f"match the retained metadata")
    daily = {k: np.concatenate(v) for k, v in parts.items()}
    if any(v.size != len(dates) for v in daily.values()):
        raise SystemExit(f"{retention_root}: daily series length mismatch")
    return daily, std, dates


def ci_of(draws: np.ndarray) -> tuple[float, float]:
    lo_p, hi_p = BOOTSTRAP["ci_percentiles"]
    return (float(np.percentile(draws, lo_p)), float(np.percentile(draws, hi_p)))


SPECTRA_ABS_FIELDS = ["HR target", "Bilinear", "CCA", "U-Net"]

SPECTRA_ERR_FIELDS = ["bilinear", "cca", "unet"]

SUM_KEYS = ["w", "e2", "ea", "e1", "bil_e2", "bil_ea", "bil_e1"]

F11_VARIABLES = ["t2m_K", "tisr_Jm2"]


def cmd_prepare_shard(args) -> int:


    import zarr

    set_roots(Path(args.output_root), None)
    assert_reference_roots_read_only()
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


    pd_group = zarr.open_group(str(read_only_reference(CANONICAL_PD_TEST_STORE)),
                               mode="r")
    mh_group = zarr.open_group(str(read_only_reference(CANONICAL_MH_STORE)),
                               mode="r")
    for name, grp in (("PD-test", pd_group), ("MH", mh_group)):
        if grp["inputs"].shape[0] != len(meta["dates"]):
            raise SystemExit(f"{name} store day count "
                             f"{grp['inputs'].shape[0]} does not match the "
                             f"SSP5-8.5 retention ({len(meta['dates'])}); "
                             f"positional revised-F11 pairing is invalid")


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
    hist = {}
    for climate in CLIMATES:
        hist[f"{climate}_t2m_fine"] = np.zeros(F11_T2M_NBINS_FINE,
                                               dtype=np.float64)
        hist[f"{climate}_tisr"] = np.zeros(F11_TISR_NBINS, dtype=np.float64)
    hist_out = {f"{climate}_{var}_{side}": 0.0
                for climate in CLIMATES for var in F11_VARIABLES
                for side in ("under_w", "over_w")}
    log(f"shard {s:02d}: {n} days {dates[0]}..{dates[-1]}, ell_max={ell_max}")

    def add_t2m(climate: str, values_K: np.ndarray) -> None:
        counts, under, over = uniform_weighted_histogram(
            values_K, F11_T2M_LO, F11_T2M_HI, F11_T2M_NBINS_FINE, whist)
        hist[f"{climate}_t2m_fine"] += counts
        hist_out[f"{climate}_t2m_K_under_w"] += under
        hist_out[f"{climate}_t2m_K_over_w"] += over

    def add_tisr(climate: str, values_Jm2: np.ndarray) -> None:
        counts, under, over = uniform_weighted_histogram(
            values_Jm2, F11_TISR_LO, F11_TISR_HI, F11_TISR_NBINS, whist)
        hist[f"{climate}_tisr"] += counts
        hist_out[f"{climate}_tisr_Jm2_under_w"] += under
        hist_out[f"{climate}_tisr_Jm2_over_w"] += over

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


        add_t2m("ssp585", inp_norm * NORM_T2M_INP_STD + NORM_T2M_INP_MEAN)
        add_tisr("ssp585", np.asarray(group["inputs"][si, 1], dtype=np.float64)
                 * TISR_STD_JM2 + TISR_MEAN_JM2)
        add_t2m("pd", np.asarray(pd_group["inputs"][si, 0], dtype=np.float64)
                * NORM_T2M_INP_STD + NORM_T2M_INP_MEAN)
        add_tisr("pd", np.asarray(pd_group["inputs"][si, 1], dtype=np.float64)
                 * TISR_STD_JM2 + TISR_MEAN_JM2)
        add_t2m("mh", np.asarray(mh_group["inputs"][si, 0], dtype=np.float64)
                * NORM_T2M_INP_STD + NORM_T2M_INP_MEAN)
        add_tisr("mh", np.asarray(mh_group["inputs"][si, 1], dtype=np.float64)
                 * TISR_STD_JM2 + TISR_MEAN_JM2)

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
        "climates": np.asarray(CLIMATES),
        "ell": np.arange(ell_max + 1, dtype=np.int64),
        "spherical_absolute_c_ell_sum_norm2": sph_abs,
        "spherical_error_c_ell_sum_norm2": sph_err,
        "spherical_absolute_fields": np.asarray(SPECTRA_ABS_FIELDS),
        "spherical_error_fields": np.asarray(SPECTRA_ERR_FIELDS),
        "k": np.arange(K_MAX + 1, dtype=np.int64),
        "zonal_absolute_power_sum_norm2": zon_abs,
        "zonal_error_power_sum_norm2": zon_err,
        "zonal_row_weight_sum": np.float64(zon_row_weight_sum),
        "f11_t2m_grid": np.asarray([F11_T2M_LO, F11_T2M_HI,
                                    float(F11_T2M_NBINS_FINE)]),
        "f11_tisr_grid": np.asarray([F11_TISR_LO, F11_TISR_HI,
                                     float(F11_TISR_NBINS)]),
        "target_std_K": np.float64(std),
    })
    tmp = out_path.with_suffix(".writing.npz")
    np.savez(tmp, **payload)
    os.replace(tmp, out_path)
    log(f"shard {s:02d}: partial written -> {out_path}")
    print(json.dumps({"status": "PASS", "mode": "prepare-shard", "shard": s,
                      "n_days": n, "partial": str(out_path)}))
    return 0


def cmd_f11_targets(args) -> int:

    import zarr

    global OUT
    OUT = guard(args.output_root)
    OUT.mkdir(parents=True, exist_ok=True)
    retention = guard(args.retention_root)
    meta = read_retention(retention)
    set_labels_from_meta(meta)
    height, width = meta["grid"]["height"], meta["grid"]["width"]
    n_days = len(meta["dates"])


    for key, frozen in (("t2m_tgt_mean_K", TARGET_MEAN_K),
                        ("t2m_tgt_std_K", TARGET_STD_K)):
        got = float(meta["normalization"][key])
        if got != frozen:
            raise SystemExit(f"frozen target normalization mismatch: {key} "
                             f"{got!r} in the accepted retention metadata vs "
                             f"{frozen!r} compiled into this program")

    stores = {"pd": read_only_reference(CANONICAL_PD_TEST_STORE),
              "mh": read_only_reference(CANONICAL_MH_STORE),
              "ssp585": read_only_reference(Path(meta["input_store"]))}
    if sorted(stores) != sorted(CLIMATES):
        raise SystemExit("climate set changed")

    w_c, _ = area_weights_from_lat(grid_lat_centers(height))
    whist = np.ascontiguousarray(
        np.broadcast_to(w_c[:, None], (height, width))).ravel()

    fine = {c: np.zeros(F11_T2M_NBINS_FINE, dtype=np.float64) for c in CLIMATES}
    outside = {}
    store_info = {}
    for climate in CLIMATES:
        group = zarr.open_group(str(stores[climate]), mode="r")
        tgt = group["targets"]
        if tuple(tgt.shape) != (n_days, 1, height, width):
            raise SystemExit(f"{climate} targets shape {tuple(tgt.shape)} does "
                             f"not match ({n_days}, 1, {height}, {width})")
        dates = [d.decode() if isinstance(d, (bytes, np.bytes_)) else str(d)
                 for d in np.asarray(group["dates"]).tolist()]
        if len(dates) != n_days:
            raise SystemExit(f"{climate} store has {len(dates)} dates")
        under = over = 0.0
        t_start = time.time()
        for i in range(n_days):
            values_K = (np.asarray(tgt[i, 0], dtype=np.float64) * TARGET_STD_K
                        + TARGET_MEAN_K)
            counts, u, o = uniform_weighted_histogram(
                values_K, F11_T2M_LO, F11_T2M_HI, F11_T2M_NBINS_FINE, whist)
            fine[climate] += counts
            under += u
            over += o
            if (i + 1) % 100 == 0 or i == n_days - 1:
                log(f"{climate}: {i + 1}/{n_days} days "
                    f"({time.time() - t_start:.1f}s)")
        outside[climate] = {"in_range_weight": float(fine[climate].sum()),
                            "underflow_weight": under, "overflow_weight": over}
        store_info[climate] = {
            "store": str(stores[climate]), "array": "targets[:, 0]",
            "dtype_on_disk": str(tgt.dtype), "shape": list(tgt.shape),
            "n_days_streamed": n_days,
            "first_date": dates[0], "last_date": dates[-1],
            "seconds": float(time.time() - t_start)}
        log(f"{climate}: HR target T2M histogram complete "
            f"(in-range weight {outside[climate]['in_range_weight']:.6f})")


    width_bin = (F11_T2M_HI - F11_T2M_LO) / F11_T2M_NBINS_DISPLAY
    display, density = {}, {}
    for c in CLIMATES:
        display[c] = fine[c].reshape(F11_T2M_NBINS_DISPLAY,
                                     F11_T2M_AGGREGATE).sum(axis=1)
        total_in = float(display[c].sum())
        if total_in <= 0.0:
            raise SystemExit(f"empty HR target T2M histogram for {c}")
        density[c] = display[c] / (total_in * width_bin)
        integral = float((density[c] * width_bin).sum())
        outside[c]["integral_check"] = integral
        outside[c]["underflow_fraction_of_total"] = (
            outside[c]["underflow_weight"]
            / (total_in + outside[c]["underflow_weight"]
               + outside[c]["overflow_weight"]))
        outside[c]["overflow_fraction_of_total"] = (
            outside[c]["overflow_weight"]
            / (total_in + outside[c]["underflow_weight"]
               + outside[c]["overflow_weight"]))
        if abs(integral - 1.0) > 1e-12:
            raise SystemExit(f"{c} HR target density integral {integral!r}")

    centers = 0.5 * (F11_T2M_DISPLAY_EDGES[:-1] + F11_T2M_DISPLAY_EDGES[1:])
    csv_path = write_csv(
        OUT / F11_TARGET_HIST_CSV,
        ["variable", "bin_center", "pd_density", "mh_density", "ssp585_density"],
        [["t2m_target_K", repr(float(centre)),
          repr(float(density["pd"][i])), repr(float(density["mh"][i])),
          repr(float(density["ssp585"][i]))]
         for i, centre in enumerate(centers)])
    npz_path = OUT / F11_TARGET_HIST_NPZ
    np.savez_compressed(npz_path, edges_K=F11_T2M_FINE_EDGES,
                        **{f"{c}_weighted_counts": fine[c] for c in CLIMATES})
    report = jdump(
        {"kind": "F11_HR_TARGET_T2M_HISTOGRAMS", "created_utc": utc(),
         "purpose": "panel A of the revised F11: physical HR TARGET T2M "
                    "distributions of the three climates",
         "field": "targets[:, 0] of each canonical store (the HR target T2M "
                  "the models are scored against), NOT the model input",
         "denormalization": F11_TARGET_DENORM,
         "frozen_target_statistics": {"mean_K": TARGET_MEAN_K,
                                      "std_K": TARGET_STD_K},
         "input_statistics_not_used": {"mean_K": NORM_T2M_INP_MEAN,
                                       "std_K": NORM_T2M_INP_STD},
         "grid": {"range_K": [F11_T2M_LO, F11_T2M_HI],
                  "accumulation_bins": F11_T2M_NBINS_FINE,
                  "accumulation_resolution_K": 0.01,
                  "display_bins": F11_T2M_NBINS_DISPLAY,
                  "display_resolution_K": 0.1,
                  "aggregation": f"{F11_T2M_AGGREGATE} adjacent 0.01 K bins",
                  "identical_to_r4_input_histogram_grid": True},
         "weights": "frozen float64 mean-one cosine of the regular-grid "
                    "latitude centres (CCA convention), identical to r4",
         "normalization": "each climate density integrates to one over the "
                          "in-range weight; out-of-range weight is reported "
                          "and never redistributed",
         "in_range_and_out_of_range_weight": outside,
         "stores": store_info,
         "retention_root": str(retention),
         "grid_shape": [height, width], "n_days_per_climate": n_days,
         "prediction_arrays_opened": False,
         "checkpoint_or_eof_basis_opened": False,
         "performance_quantity_computed": False,
         "inference_rerun": False, "eof_projection_rerun": False,
         "cca_reconstruction_rerun": False, "reduction_rerun": False,
         "kde_or_smoothing": "none",
         "outputs": {"display_density_csv": str(csv_path),
                     "fine_counts_npz": str(npz_path)}},
        OUT / F11_TARGET_HIST_REPORT)
    print(json.dumps({"status": "PASS", "mode": "f11-targets",
                      "output_root": str(OUT),
                      "csv": str(csv_path), "npz": str(npz_path),
                      "report": str(report),
                      "integral_checks": {c: outside[c]["integral_check"]
                                          for c in CLIMATES}}, indent=2))
    return 0


def pd_regional_mse_K2() -> dict:

    payload = json.loads(read_only_reference(PD_REGIONAL_JSON).read_text())
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


def three_climate_statistics() -> dict:


    daily = {}
    dates = {}
    stds = {}
    daily["PD"], stds["PD"], dates["PD"] = daily_from_partials(PD_ROOT)
    daily["MH"], stds["MH"], dates["MH"] = daily_from_partials(MH_ROOT)
    daily["SSP5-8.5"], stds["SSP5-8.5"], dates["SSP5-8.5"] = \
        daily_from_partials(OUT)
    log(f"daily sufficient statistics: PD {len(dates['PD'])}, "
        f"MH {len(dates['MH'])}, SSP5-8.5 {len(dates['SSP5-8.5'])} days "
        f"(prediction arrays not opened)")

    rng = np.random.default_rng(BOOTSTRAP["seed"])
    idx, point, draws = {}, {}, {}
    for climate in CLIMATE_DRAW_ORDER:
        idx[climate] = circular_block_bootstrap_indices(
            len(dates[climate]), BOOTSTRAP["block_length_days"],
            BOOTSTRAP["n_resamples"], rng)
        point[climate] = point_estimates(daily[climate], stds[climate])
        draws[climate] = bootstrap_quantities(daily[climate], idx[climate],
                                              stds[climate])
    log("three-climate bootstrap complete (fixed order PD, MH, SSP5-8.5)")

    methods_rmse = ["bilinear_rmse_K", "cca_rmse_K", "unet_rmse_K"]
    skill_keys = {"cca": "cca_skill_vs_bilinear",
                  "unet": "unet_skill_vs_bilinear"}
    stats: dict = {}
    for climate in CLIMATE_DRAW_ORDER:
        tag = climate.replace("5-8.5", "585").replace("SSP", "SSP").replace("-", "_")
        for q in methods_rmse:
            stats[f"{tag}_{q}"] = {"estimate": float(point[climate][q]),
                                   "ci": ci_of(draws[climate][q]), "units": "K"}
        for m, key in skill_keys.items():
            stats[f"{tag}_{m}_skill_vs_bilinear"] = {
                "estimate": float(point[climate][key]),
                "ci": ci_of(draws[climate][key]), "units": "-"}
        stats[f"{tag}_skill_vs_cca"] = {
            "estimate": float(point[climate]["skill_vs_cca"]),
            "ci": ci_of(draws[climate]["skill_vs_cca"]), "units": "-"}
        stats[f"{tag}_delta_rmse_K"] = {
            "estimate": float(point[climate]["delta_rmse_K"]),
            "ci": ci_of(draws[climate]["delta_rmse_K"]), "units": "K"}

    for climate, tag in (("MH", "MH"), ("SSP5-8.5", "SSP585")):
        for m, key in skill_keys.items():
            chg = 100.0 * (draws[climate][key] - draws["PD"][key])
            stats[f"{tag}_{m}_skill_change_vs_PD_pp"] = {
                "estimate": 100.0 * (point[climate][key] - point["PD"][key]),
                "ci": ci_of(chg), "units": "pp"}
        dit = 100.0 * ((draws[climate][skill_keys["unet"]]
                        - draws["PD"][skill_keys["unet"]])
                       - (draws[climate][skill_keys["cca"]]
                          - draws["PD"][skill_keys["cca"]]))
        dit_point = 100.0 * ((point[climate][skill_keys["unet"]]
                              - point["PD"][skill_keys["unet"]])
                             - (point[climate][skill_keys["cca"]]
                                - point["PD"][skill_keys["cca"]]))
        stats[f"{tag}_difference_in_transfer_unet_minus_cca_pp"] = {
            "estimate": dit_point, "ci": ci_of(dit), "units": "pp",
            "sign": "positive favors U-Net"}


    for q in ["bilinear_rmse_K", "cca_rmse_K", "unet_rmse_K", "delta_rmse_K"]:
        chg = draws["SSP5-8.5"][q] - draws["PD"][q]
        stats[f"SSP585_change_vs_PD__{q}"] = {
            "estimate": float(point["SSP5-8.5"][q] - point["PD"][q]),
            "ci": ci_of(chg), "units": "K"}
    for q in ["cca_skill_vs_bilinear", "unet_skill_vs_bilinear", "skill_vs_cca"]:
        chg = draws["SSP5-8.5"][q] - draws["PD"][q]
        stats[f"SSP585_change_vs_PD__{q}"] = {
            "estimate": float(point["SSP5-8.5"][q] - point["PD"][q]),
            "ci": ci_of(chg), "units": "-"}

    return {"statistics": stats,
            "populations": {c: {"n_days": len(dates[c]), "first": dates[c][0],
                                "last": dates[c][-1], "target_std_K": stds[c]}
                            for c in CLIMATE_DRAW_ORDER},
            "draw_order": CLIMATE_DRAW_ORDER,
            "roots": {"PD": str(PD_ROOT), "MH": str(MH_ROOT),
                      "SSP5-8.5": str(OUT)}}


RMSE_SKILL_KEYS = {"cca": "cca_rmse_skill_vs_bilinear",
                   "unet": "unet_rmse_skill_vs_bilinear"}

F15_GATE_TOL = 1.0e-12

F15_GATE_QUANTITIES = ["bilinear_rmse_K", "cca_rmse_K", "unet_rmse_K",
                       "delta_rmse_K"]


def three_climate_rmse_skill_statistics(ssp_root: Path) -> dict:
    daily, dates, stds = {}, {}, {}
    daily["PD"], stds["PD"], dates["PD"] = daily_from_partials(PD_ROOT)
    daily["MH"], stds["MH"], dates["MH"] = daily_from_partials(MH_ROOT)
    daily["SSP5-8.5"], stds["SSP5-8.5"], dates["SSP5-8.5"] = \
        daily_from_partials(guard(ssp_root))
    log(f"daily sufficient statistics: PD {len(dates['PD'])}, "
        f"MH {len(dates['MH'])}, SSP5-8.5 {len(dates['SSP5-8.5'])} days "
        f"(prediction arrays not opened)")

    rng = np.random.default_rng(BOOTSTRAP["seed"])
    idx, point, draws = {}, {}, {}
    for climate in CLIMATE_DRAW_ORDER:
        idx[climate] = circular_block_bootstrap_indices(
            len(dates[climate]), BOOTSTRAP["block_length_days"],
            BOOTSTRAP["n_resamples"], rng)
        point[climate] = point_estimates(daily[climate], stds[climate])
        draws[climate] = bootstrap_quantities(daily[climate], idx[climate],
                                              stds[climate])
    log("three-climate bootstrap complete under the RMSE-skill definition "
        "(fixed order PD, MH, SSP5-8.5, seed 20260715, 10,000 replicates)")

    stats: dict = {}
    for climate in CLIMATE_DRAW_ORDER:
        tag = climate.replace("5-8.5", "585").replace("-", "_")
        for q in ("bilinear_rmse_K", "bilinear_rmse_unet_pipeline_K",
                  "cca_rmse_K", "unet_rmse_K", "delta_rmse_K"):
            stats[f"{tag}_{q}"] = {"estimate": float(point[climate][q]),
                                   "ci": ci_of(draws[climate][q]), "units": "K"}
        for m, key in RMSE_SKILL_KEYS.items():
            stats[f"{tag}_{m}_rmse_skill_vs_bilinear"] = {
                "estimate": float(point[climate][key]),
                "ci": ci_of(draws[climate][key]), "units": "-",
                "definition": "1 - RMSE_model / RMSE_bilinear"}
        stats[f"{tag}_rmse_skill_vs_cca"] = {
            "estimate": float(point[climate]["rmse_skill_vs_cca"]),
            "ci": ci_of(draws[climate]["rmse_skill_vs_cca"]), "units": "-",
            "definition": "1 - RMSE_UNet / RMSE_CCA"}

    for climate, tag in (("MH", "MH"), ("SSP5-8.5", "SSP585")):
        for m, key in RMSE_SKILL_KEYS.items():
            chg = 100.0 * (draws[climate][key] - draws["PD"][key])
            stats[f"{tag}_{m}_rmse_skill_change_vs_PD_pp"] = {
                "estimate": 100.0 * (point[climate][key] - point["PD"][key]),
                "ci": ci_of(chg), "units": "pp",
                "definition": "100*(S^RMSE_climate - S^RMSE_PD) vs bilinear"}
        dit = 100.0 * ((draws[climate][RMSE_SKILL_KEYS["unet"]]
                        - draws["PD"][RMSE_SKILL_KEYS["unet"]])
                       - (draws[climate][RMSE_SKILL_KEYS["cca"]]
                          - draws["PD"][RMSE_SKILL_KEYS["cca"]]))
        dit_point = 100.0 * ((point[climate][RMSE_SKILL_KEYS["unet"]]
                              - point["PD"][RMSE_SKILL_KEYS["unet"]])
                             - (point[climate][RMSE_SKILL_KEYS["cca"]]
                                - point["PD"][RMSE_SKILL_KEYS["cca"]]))
        stats[f"{tag}_difference_in_transfer_unet_minus_cca_pp"] = {
            "estimate": dit_point, "ci": ci_of(dit), "units": "pp",
            "sign": "positive favors U-Net",
            "definition": "100*[(dS^RMSE_UNet) - (dS^RMSE_CCA)] vs bilinear"}

        chg = (draws[climate]["rmse_skill_vs_cca"]
               - draws["PD"]["rmse_skill_vs_cca"])
        stats[f"{tag}_rmse_skill_vs_cca_change_vs_PD"] = {
            "estimate": float(point[climate]["rmse_skill_vs_cca"]
                              - point["PD"]["rmse_skill_vs_cca"]),
            "ci": ci_of(chg), "units": "-",
            "definition": "S^RMSE_{UNet|CCA}(climate) - S^RMSE_{UNet|CCA}(PD)"}

    for q in ("bilinear_rmse_K", "cca_rmse_K", "unet_rmse_K", "delta_rmse_K"):
        chg = draws["SSP5-8.5"][q] - draws["PD"][q]
        stats[f"SSP585_change_vs_PD__{q}"] = {
            "estimate": float(point["SSP5-8.5"][q] - point["PD"][q]),
            "ci": ci_of(chg), "units": "K"}
    for q in ("cca_rmse_skill_vs_bilinear", "unet_rmse_skill_vs_bilinear",
              "rmse_skill_vs_cca"):
        chg = draws["SSP5-8.5"][q] - draws["PD"][q]
        stats[f"SSP585_change_vs_PD__{q}"] = {
            "estimate": float(point["SSP5-8.5"][q] - point["PD"][q]),
            "ci": ci_of(chg), "units": "-"}


    accepted = {r["quantity"]: r for r in
                load_csv(read_only_reference(
                    guard(ssp_root) / "plot_sources"
                    / "F15__PD_MH_SSP585_generalization_source.csv"))}
    residuals = {}
    for tag in ("PD", "MH", "SSP585"):
        for q in F15_GATE_QUANTITIES:
            key = f"{tag}_{q}"
            ref, mine = accepted[key], stats[key]
            for what, a, b in (("estimate", float(ref["estimate"]),
                                mine["estimate"]),
                               ("ci_lo", float(ref["ci_lo_95"]), mine["ci"][0]),
                               ("ci_hi", float(ref["ci_hi_95"]), mine["ci"][1])):
                d = abs(a - b)
                residuals[f"{key}.{what}"] = d
                if d > F15_GATE_TOL:
                    raise SystemExit(
                        f"F15 gate: {key}.{what} {b!r} does not reproduce the "
                        f"accepted {a!r} (|d|={d:.3e} > {F15_GATE_TOL:.1e}); "
                        "the resampling indices or the population differ")
    log("F15 gate PASS: every RMSE point estimate and interval reproduces the "
        f"accepted run (max |residual| = {max(residuals.values()):.3e})")

    return {"statistics": stats,
            "skill_definition": SKILL_DEFINITION,
            "skill_definition_long": SKILL_DEFINITION_LONG,
            "mse_skill_used_anywhere": False,
            "populations": {c: {"n_days": len(dates[c]), "first": dates[c][0],
                                "last": dates[c][-1], "target_std_K": stds[c]}
                            for c in CLIMATE_DRAW_ORDER},
            "draw_order": CLIMATE_DRAW_ORDER,
            "bootstrap": {**BOOTSTRAP, "climate_draw_order": CLIMATE_DRAW_ORDER,
                          "recomputed_inside_every_replicate": True,
                          "old_interval_endpoints_transformed": False},
            "reproduction_gate": {
                "reference": str(guard(ssp_root) / "plot_sources"
                                 / "F15__PD_MH_SSP585_generalization_source.csv"),
                "quantities": F15_GATE_QUANTITIES,
                "tolerance_abs": F15_GATE_TOL,
                "max_abs_residual": max(residuals.values()),
                "passed": True},
            "roots": {"PD": str(PD_ROOT), "MH": str(MH_ROOT),
                      "SSP5-8.5": str(guard(ssp_root))}}


def cmd_reduce(args) -> int:
    set_roots(Path(args.output_root), None)
    assert_reference_roots_read_only()
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
    for reference in (PD_F4_NPZ, PD_F7_ERR_SPH, PD_F7_ERR_ZON, PD_F7_ABS_SPH,
                      PD_F7_ABS_ZON, PD_REGIONAL_JSON, MH_F8_NPZ, MH_F9_CSV,
                      MH_ROOT / "retention_metadata.json"):
        if not Path(reference).exists():
            missing.append(str(reference))
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
    hist = {}
    for climate in CLIMATES:
        hist[f"{climate}_t2m_fine"] = np.zeros(F11_T2M_NBINS_FINE,
                                               dtype=np.float64)
        hist[f"{climate}_tisr"] = np.zeros(F11_TISR_NBINS, dtype=np.float64)
    hist_out = {f"{climate}_{var}_{side}": 0.0
                for climate in CLIMATES for var in F11_VARIABLES
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
            if not (np.array_equal(d["f11_t2m_grid"],
                                   np.asarray([F11_T2M_LO, F11_T2M_HI,
                                               float(F11_T2M_NBINS_FINE)]))
                    and np.array_equal(d["f11_tisr_grid"],
                                       np.asarray([F11_TISR_LO, F11_TISR_HI,
                                                   float(F11_TISR_NBINS)]))):
                raise SystemExit(f"partial {s}: revised-F11 bin grids differ "
                                 f"from the predeclared policy")
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
    if n_days != 1096:
        raise SystemExit(f"SSP5-8.5 population is {n_days} days, expected 1096")
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
    log("SSP5-8.5 headline bootstrap complete")


    three = three_climate_statistics()
    tri = three["statistics"]


    def ratio(pipe, key, p, r):
        return acc[f"{pipe}_{key}"][p, r] / acc[f"{pipe}_w"][p, r]

    regional = {}
    for ri, reg in enumerate(REGIONS):
        for pi, per in enumerate(PERIODS):
            mse_cca = ratio("cca", "e2", pi, ri)
            mse_unet = ratio("unet", "e2", pi, ri)
            regional[(reg, per)] = {
                "bilinear": math.sqrt(ratio("cca", "bil_e2", pi, ri)) * std,
                "cca": math.sqrt(mse_cca) * std,
                "unet": math.sqrt(mse_unet) * std,
                "mse_cca_K2": mse_cca * std ** 2,
                "mse_unet_K2": mse_unet * std ** 2,
                "bias_bilinear_K": ratio("cca", "bil_e1", pi, ri) * std,
                "bias_cca_K": ratio("cca", "e1", pi, ri) * std,
                "bias_unet_K": ratio("unet", "e1", pi, ri) * std,
                "mae_bilinear_K": ratio("cca", "bil_ea", pi, ri) * std,
                "mae_cca_K": ratio("cca", "ea", pi, ri) * std,
                "mae_unet_K": ratio("unet", "ea", pi, ri) * std,
                "delta_rmse_K": (math.sqrt(mse_cca) - math.sqrt(mse_unet)) * std,
                "skill_vs_cca": 1.0 - mse_unet / mse_cca,
            }


    write_csv(SRC / "T3__SSP585_regional_rmse_source.csv",
              ["region", "method", "rmse_K"],
              [[r, m, f"{regional[(r, 'annual')][m]:.6f}"]
               for r in T1_REGIONS for m in ("bilinear", "cca", "unet")])
    write_csv(SRC / "T3__SSP585_regional_bias_mae_source.csv",
              ["region", "period", "method", "bias_K", "mae_K"],
              [[r, p, m, repr(float(regional[(r, p)][f"bias_{m}_K"])),
                repr(float(regional[(r, p)][f"mae_{m}_K"]))]
               for r in REGIONS for p in PERIODS
               for m in ("bilinear", "cca", "unet")])

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
    t3_rows = [[q, meth, repr(point[q]), repr(ci[q][0]), repr(ci[q][1]), unit, ref,
                sign, BOOTSTRAP["scheme"], BOOTSTRAP["block_length_days"],
                BOOTSTRAP["n_resamples"]] for q, meth, unit, ref, sign in order]
    change_scheme = (BOOTSTRAP["scheme"] + "; three-climate fixed draw order "
                     "PD, MH, SSP5-8.5; climates independent, methods paired")
    for q, meth, unit, ref, _sign in order:
        key = f"SSP585_change_vs_PD__{q}"
        entry = tri[key]
        t3_rows.append([key, meth, repr(float(entry["estimate"])),
                        repr(float(entry["ci"][0])), repr(float(entry["ci"][1])),
                        entry["units"], "PD test",
                        "SSP5-8.5 minus PD; sign as for the base quantity",
                        change_scheme, BOOTSTRAP["block_length_days"],
                        BOOTSTRAP["n_resamples"]])
    t3_csv = write_csv(
        SRC / "T3__SSP585_global_metrics.csv",
        ["quantity", "method_or_pair", "estimate", "ci_lo_95", "ci_hi_95",
         "units", "reference_method", "sign_interpretation", "bootstrap_scheme",
         "block_length_days", "n_resamples"], t3_rows)


    lat_g = np.load(retention_root / meta["grid"]["coords"]["hr_lat"])
    lon_g = np.load(retention_root / meta["grid"]["coords"]["hr_lon"])
    np.savez_compressed(
        SRC / "F12__SSP585_global_rmse_maps_source.npz",
        latitude=lat_g, longitude=lon_g,
        bilinear_rmse_K=np.sqrt(maps["bilinear_e2"] / n_days) * std,
        cca_rmse_K=np.sqrt(maps["cca_e2"] / n_days) * std,
        unet_rmse_K=np.sqrt(maps["unet_e2"] / n_days) * std,
        delta_rmse_K=(np.sqrt(maps["cca_e2"] / n_days)
                      - np.sqrt(maps["unet_e2"] / n_days)) * std)


    pd_reg = pd_regional_mse_K2()
    write_csv(SRC / "F13__SSP585_PD_shared_sufficient_statistics.csv",
              ["region", "period", "pd_mse_cca_K2", "pd_mse_unet_K2",
               "ssp585_mse_cca_K2", "ssp585_mse_unet_K2"],
              [[r, p, repr(pd_reg[(r, p)]["mse_cca_K2"]),
                repr(pd_reg[(r, p)]["mse_unet_K2"]),
                repr(float(regional[(r, p)]["mse_cca_K2"])),
                repr(float(regional[(r, p)]["mse_unet_K2"]))]
               for r in REGIONS for p in PERIODS])


    revised_f9 = SRC / "revised_F9__MH_PD_shared_sufficient_statistics.csv"
    shutil.copyfile(read_only_reference(MH_F9_CSV), revised_f9)
    if sha256(revised_f9) != sha256(MH_F9_CSV):
        raise SystemExit("copied MH F9 sufficient statistics differ from the "
                         "accepted source")


    ell = np.asarray(ell)
    keep = (ell >= 1) & (ell <= ELL_MAX)
    ssp_abs_sph = {name: sph_abs[fi] / n_days * std ** 2
                   for fi, name in enumerate(SPECTRA_ABS_FIELDS)}
    ssp_err_sph = {name: sph_err[fi] / n_days * std ** 2
                   for fi, name in enumerate(SPECTRA_ERR_FIELDS)}
    kk = np.asarray(kk)
    kkeep = (kk >= 1) & (kk <= K_MAX)
    zon_denom = zon_row_weight_sum * float(width)
    ssp_abs_zon = {name: zon_abs[fi] / zon_denom * std ** 2
                   for fi, name in enumerate(SPECTRA_ABS_FIELDS)}
    ssp_err_zon = {name: zon_err[fi] / zon_denom * std ** 2
                   for fi, name in enumerate(SPECTRA_ERR_FIELDS)}

    pd_err_sph = load_csv(read_only_reference(PD_F7_ERR_SPH))
    pd_err_zon = load_csv(read_only_reference(PD_F7_ERR_ZON))
    pd_abs_sph = load_csv(read_only_reference(PD_F7_ABS_SPH))
    pd_abs_zon = load_csv(read_only_reference(PD_F7_ABS_ZON))

    err_rows = []
    for m in SPECTRA_ERR_FIELDS:
        pd_by_ell = {int(r["ell"]): float(r[f"{m}_error_K2"]) for r in pd_err_sph}
        for e in ell[keep]:
            err_rows.append(["spherical", m, int(e),
                             f"{pd_by_ell[int(e)]:.10e}",
                             f"{ssp_err_sph[m][int(e)]:.10e}"])
        pd_by_k = {int(r["k"]): float(r[f"{m}_error_K2"]) for r in pd_err_zon}
        for k_ in kk[kkeep]:
            err_rows.append(["zonal", m, int(k_),
                             f"{pd_by_k[int(k_)]:.10e}",
                             f"{ssp_err_zon[m][int(k_)]:.10e}"])
    write_csv(SRC / "F14__SSP585_prediction_error_source.csv",
              ["spectrum", "method", "index", "pd_power_K2", "ssp585_power_K2"],
              err_rows)

    abs_rows = []
    for name in SPECTRA_ABS_FIELDS:
        pd_by_ell = {int(r["ell"]): float(r["C_ell_K2"])
                     for r in pd_abs_sph if r["series"] == name}
        for e in ell[keep]:
            abs_rows.append(["spherical", name, int(e),
                             f"{pd_by_ell[int(e)]:.10e}",
                             f"{ssp_abs_sph[name][int(e)]:.10e}"])
        pd_by_k = {int(r["k"]): float(r["power_K2"])
                   for r in pd_abs_zon if r["series"] == name}
        for k_ in kk[kkeep]:
            abs_rows.append(["zonal", name, int(k_),
                             f"{pd_by_k[int(k_)]:.10e}",
                             f"{ssp_abs_zon[name][int(k_)]:.10e}"])
    write_csv(SRC / "F14__SSP585_absolute_field_source.csv",
              ["spectrum", "series", "index", "pd_power_K2", "ssp585_power_K2"],
              abs_rows)


    t2m_display = {c: hist[f"{c}_t2m_fine"].reshape(
        F11_T2M_NBINS_DISPLAY, F11_T2M_AGGREGATE).sum(axis=1) for c in CLIMATES}
    t2m_width = (F11_T2M_HI - F11_T2M_LO) / F11_T2M_NBINS_DISPLAY
    tisr_width = (F11_TISR_HI - F11_TISR_LO) / F11_TISR_NBINS
    density: dict = {}
    outside: dict = {}
    for c in CLIMATES:
        for var, counts, wbin in (("t2m_K", t2m_display[c], t2m_width),
                                  ("tisr_Jm2", hist[f"{c}_tisr"], tisr_width)):
            total_in = float(counts.sum())
            if total_in <= 0.0:
                raise SystemExit(f"empty revised-F11 histogram for {c}/{var}")
            density[(c, var)] = counts / (total_in * wbin)
            under = hist_out[f"{c}_{var}_under_w"]
            over = hist_out[f"{c}_{var}_over_w"]
            outside[f"{c}_{var}"] = {
                "in_range_weight": total_in,
                "underflow_weight": under, "overflow_weight": over,
                "underflow_fraction_of_total": under / (total_in + under + over),
                "overflow_fraction_of_total": over / (total_in + under + over),
                "integral_check": float((density[(c, var)] * wbin).sum())}

    f11_rows = []
    t2m_centers = 0.5 * (F11_T2M_DISPLAY_EDGES[:-1] + F11_T2M_DISPLAY_EDGES[1:])
    tisr_centers = 0.5 * (F11_TISR_EDGES[:-1] + F11_TISR_EDGES[1:])
    for var, centers in (("t2m_K", t2m_centers), ("tisr_Jm2", tisr_centers)):
        for i, centre in enumerate(centers):
            f11_rows.append([var, repr(float(centre)),
                             repr(float(density[("pd", var)][i])),
                             repr(float(density[("mh", var)][i])),
                             repr(float(density[("ssp585", var)][i]))])
    write_csv(SRC / "revised_F11__PD_MH_SSP585_input_distributions_source.csv",
              ["variable", "bin_center", "pd_density", "mh_density",
               "ssp585_density"], f11_rows)
    np.savez_compressed(
        SRC / "revised_F11__t2m_fine_counts_0p01K.npz",
        edges_K=F11_T2M_FINE_EDGES,
        **{f"{c}_weighted_counts": hist[f"{c}_t2m_fine"] for c in CLIMATES})
    jdump({"kind": "REVISED_F11_BIN_POLICY", "created_utc": utc(),
           "policy": "predeclared fixed physical grids, identical for all three "
                     "climates, fixed before any SSP5-8.5 value was read; no "
                     "KDE, no smoothing spline, no climate-dependent tuning",
           "t2m": {"range_K": [F11_T2M_LO, F11_T2M_HI],
                   "accumulation_bins": F11_T2M_NBINS_FINE,
                   "accumulation_resolution_K": 0.01,
                   "display_bins": F11_T2M_NBINS_DISPLAY,
                   "display_resolution_K": 0.1,
                   "aggregation": f"{F11_T2M_AGGREGATE} adjacent 0.01 K bins",
                   "retained_counts":
                       "plot_sources/revised_F11__t2m_fine_counts_0p01K.npz",
                   "y_axis": "probability density [K^-1]"},
           "tisr": {"range_Jm2": [F11_TISR_LO, F11_TISR_HI],
                    "range_rule": "frozen F11 physical plotting range: "
                                  "0 .. tisr_mean + 4*tisr_std from the frozen "
                                  "PD normalization statistics",
                    "display_bins": F11_TISR_NBINS,
                    "bin_width_Jm2": tisr_width,
                    "y_axis": "probability density [(J m^-2)^-1]"},
           "normalization": "each climate density integrates to one over the "
                            "in-range weight; out-of-range weight is reported, "
                            "never redistributed",
           "weights": "frozen float64 mean-one cosine (CCA convention)",
           "fields": {"t2m_K": "store input channel 0 (bilinear input T2M), "
                               "denormalized with frozen t2m_inp stats",
                      "tisr_Jm2": "store input channel 1, denormalized with "
                                  "frozen tisr stats; J m-2"},
           "sources": {"pd": str(CANONICAL_PD_TEST_STORE),
                       "mh": str(CANONICAL_MH_STORE),
                       "ssp585": str(meta["input_store"])},
           "pairing": "positional day index; all three stores are 1,096 days",
           "underflow_overflow": outside},
          REP / "revised_F11__bin_policy.json")


    f15_csv = write_csv(
        SRC / "F15__PD_MH_SSP585_generalization_source.csv",
        ["quantity", "estimate", "ci_lo_95", "ci_hi_95", "units"],
        [[k, repr(float(v["estimate"])), repr(float(v["ci"][0])),
          repr(float(v["ci"][1])), v["units"]] for k, v in tri.items()])
    jdump({"kind": "PD_MH_SSP585_GENERALIZATION_STATISTICS",
           "created_utc": utc(), **three,
           "definitions": {
               "skill": "1 - MSE_model/MSE_bilinear (established per-pipeline "
                        "weight conventions)",
               "transfer_change_pp": "100*(skill_climate - skill_PD)",
               "difference_in_transfer_pp":
                   "100*[(skill_c,unet - skill_PD,unet) - "
                   "(skill_c,cca - skill_PD,cca)]; positive favors U-Net"},
           "bootstrap": {**BOOTSTRAP,
                         "climate_draw_order": CLIMATE_DRAW_ORDER,
                         "independence": "one deterministic generator seeded "
                         "20260715 draws PD, then MH, then SSP5-8.5; climates "
                         "resampled independently, methods paired within each "
                         "climate"},
           "prediction_arrays_opened": False,
           "pd_and_mh_reduction_rerun": False},
          REP / "F15__generalization_statistics.json")

    reference_hashes = {
        str(p): sha256(p) for p in (PD_F4_NPZ, PD_F7_ERR_SPH, PD_F7_ERR_ZON,
                                    PD_F7_ABS_SPH, PD_F7_ABS_ZON,
                                    PD_REGIONAL_JSON, MH_F8_NPZ, MH_F9_CSV)}

    reduced = {
        "kind": "REDUCE_REPORT", "created_utc": utc(),
        "population": SSP585_LABEL, "scenario": FIG_SCENARIO_LABEL,
        "scenario_internal_id": "ssp585",
        "n_shards": n_shards, "n_days": n_days,
        "date_coverage_exact_once": True,
        "first_date": covered[0], "last_date": covered[-1],
        "period_days": {p: int(period_days[i]) for i, p in enumerate(PERIODS)},
        "target_std_K": std,
        "headline_point_estimates": point,
        "headline_ci_95": {k: list(v) for k, v in ci.items()},
        "bootstrap": BOOTSTRAP,
        "bootstrap_scheme_note":
            "T3 headline CIs use the established per-run scheme (identical to "
            "T1/T2: one generator seeded 20260715 on this climate alone). The "
            "change-vs-PD rows and every F15 quantity use the three-climate "
            "fixed-order scheme (PD, MH, SSP5-8.5).",
        "three_climate_statistics": tri,
        "three_climate_populations": three["populations"],
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
        "revised_f11_underflow_overflow": outside,
        "sources_recomputed_from_this_run": [
            "T3__SSP585_regional_rmse_source.csv",
            "T3__SSP585_regional_bias_mae_source.csv",
            "T3__SSP585_global_metrics.csv",
            "F12__SSP585_global_rmse_maps_source.npz",
            "F13__SSP585_PD_shared_sufficient_statistics.csv",
            "F14__SSP585_prediction_error_source.csv",
            "F14__SSP585_absolute_field_source.csv",
            "revised_F11__PD_MH_SSP585_input_distributions_source.csv",
            "revised_F11__t2m_fine_counts_0p01K.npz",
            "F15__PD_MH_SSP585_generalization_source.csv"],
        "sources_copied_read_only": [
            "revised_F9__MH_PD_shared_sufficient_statistics.csv"],
        "pd_mh_reference_read_only": reference_hashes,
        "pd_or_mh_inference_rerun": False,
        "pd_or_mh_reduction_rerun": False,
        "no_ssp585_derived_statistics": True,
        "plot_sources_dir": str(SRC),
        "t3_csv": str(t3_csv),
        "f15_csv": str(f15_csv),
    }
    jdump(reduced, REP / "reduce_report.json")
    jdump({"regions": REGIONS, "periods": PERIODS,
           "regional_period_rmse_K": {f"{r}|{p}": regional[(r, p)]
                                      for r in REGIONS for p in PERIODS}},
          REP / "regional_period_rmse.json")
    write_csv(TAB / "T3__SSP585_regional_period_metrics.csv",
              ["region", "period", "bilinear_rmse_K", "cca_rmse_K",
               "unet_rmse_K", "bias_bilinear_K", "bias_cca_K", "bias_unet_K",
               "mae_bilinear_K", "mae_cca_K", "mae_unet_K", "delta_rmse_K",
               "skill_vs_cca"],
              [[r, p] + [repr(float(regional[(r, p)][key])) for key in
                         ("bilinear", "cca", "unet", "bias_bilinear_K",
                          "bias_cca_K", "bias_unet_K", "mae_bilinear_K",
                          "mae_cca_K", "mae_unet_K", "delta_rmse_K",
                          "skill_vs_cca")]
               for r in REGIONS for p in PERIODS])
    print(json.dumps({"status": "PASS", "mode": "reduce", "n_days": n_days,
                      "headline": point}, indent=2))
    return 0


def _maps_from_npz(path):
    with np.load(guard(path), allow_pickle=False) as d:
        lat = d["latitude"].astype(np.float64)
        lon = d["longitude"].astype(np.float64)
        fields = {k: np.asarray(d[k], np.float64) for k in
                  ("bilinear_rmse_K", "cca_rmse_K", "unet_rmse_K",
                   "delta_rmse_K")}
    return lat, lon, fields


def _cell_ratio(unet, cca):


    unet = np.asarray(unet, np.float64)
    cca = np.asarray(cca, np.float64)
    bad = (~np.isfinite(unet)) | (~np.isfinite(cca)) | (cca == 0.0)
    ratio = np.full(unet.shape, np.nan, dtype=np.float64)
    ok = ~bad
    ratio[ok] = unet[ok] / cca[ok]
    return ratio, bad


def _transfer_stat(climate_maps, pd_maps) -> tuple[np.ndarray, dict]:


    ratio_c, bad_c = _cell_ratio(climate_maps["unet_rmse_K"],
                                 climate_maps["cca_rmse_K"])
    ratio_p, bad_p = _cell_ratio(pd_maps["unet_rmse_K"], pd_maps["cca_rmse_K"])
    bad = bad_c | bad_p
    stat = ratio_p - ratio_c
    stat[bad] = np.nan
    finite_in = ~bad
    unet_c = np.asarray(climate_maps["unet_rmse_K"], np.float64)
    cca_c = np.asarray(climate_maps["cca_rmse_K"], np.float64)
    cca_beats_unet = int((finite_in & (unet_c > cca_c)).sum())
    counts = {
        "n_cells_total": int(stat.size),
        "n_cells_masked": int(bad.sum()),
        "n_cells_used": int(finite_in.sum()),
        "masking_rule": "non-finite value in any input, or an exactly zero "
                        "CCA denominator (transfer climate or PD)",
        "n_masked_nonfinite_or_zero_cca_transfer_climate": int(bad_c.sum()),
        "n_masked_nonfinite_or_zero_cca_PD": int(bad_p.sum()),
        "n_transfer_climate_cells_with_RMSE_UNet_gt_RMSE_CCA": cca_beats_unet,
        "pct_transfer_climate_cells_with_RMSE_UNet_gt_RMSE_CCA":
            100.0 * cca_beats_unet / max(1, int(finite_in.sum())),
        "units": "dimensionless",
        "min": float(np.nanmin(stat)), "max": float(np.nanmax(stat)),
    }
    return stat, counts


TRANSFER_CACHE: dict = {}


def transfer_panels() -> dict:


    if TRANSFER_CACHE:
        return TRANSFER_CACHE
    lat, lon, mh = _maps_from_npz(read_only_reference(MH_F8_NPZ))
    lat2, lon2, ssp = _maps_from_npz(
        source_path("F12__SSP585_global_rmse_maps_source.npz"))
    if not (np.array_equal(lat, lat2) and np.array_equal(lon, lon2)):
        raise SystemExit("MH and SSP5-8.5 cellwise maps are on different grids")
    with np.load(read_only_reference(PD_F4_NPZ), allow_pickle=False) as d:
        if not (np.array_equal(d["latitude"].astype(np.float64), lat)
                and np.array_equal(d["longitude"].astype(np.float64), lon)):
            raise SystemExit("PD reference maps are on a different grid")
        pd_maps = {k: np.asarray(d[k], np.float64)
                   for k in ("cca_rmse_K", "unet_rmse_K")}

    stat_mh, counts_mh = _transfer_stat(mh, pd_maps)
    stat_ssp, counts_ssp = _transfer_stat(ssp, pd_maps)

    w_row, _ = area_weights_from_lat(grid_lat_centers(lat.size))
    w2d = np.ascontiguousarray(np.broadcast_to(w_row[:, None], stat_mh.shape))
    p99 = {}
    for tag, stat in (("F8", stat_mh), ("F12", stat_ssp)):
        ok = np.isfinite(stat)
        p99[tag] = _weighted_quantile(np.abs(stat[ok]), w2d[ok], TRANSFER_P)
    limit = float(TRANSFER_LIMIT)
    ticks = list(TRANSFER_TICKS)
    scale = {
        "rule": TRANSFER_SCALE_RULE,
        "units": "dimensionless",
        "rule_fixed_before_reading_values": True,
        "shared_limits": [-limit, limit],
        "colorbar_ticks": ticks,
        "centred_exactly_at_zero": True,
        "percentage_point_scaling_removed": True,
        "colormap": TRANSFER_CMAP,
        "area_weighted_p99_abs_dimensionless_diagnostic_only":
            {k: float(v) for k, v in p99.items()},
        "area_weight_convention": "float64 mean-one cosine of the regular grid "
                                  "latitude centres (frozen CCA convention)",
    }
    log(f"transfer scale: exact +/-{limit:g} dimensionless "
        f"(diagnostic p99|F8| = {p99['F8']:.6f}, p99|F12| = {p99['F12']:.6f})")
    TRANSFER_CACHE.update({
        "lat": lat, "lon": lon, "mh": mh, "ssp": ssp, "pd": pd_maps,
        "stat": {"F8": stat_mh, "F12": stat_ssp},
        "counts": {"F8": counts_mh, "F12": counts_ssp},
        "limit": limit, "ticks": ticks, "scale": scale})
    return TRANSFER_CACHE


def _rmse_skill_change(g) -> np.ndarray:


    for name, arr in (("PD CCA", g["a_cca"]), ("climate CCA", g["b_cca"])):
        if not np.all(arr > 0.0):
            raise SystemExit(f"non-positive {name} MSE in the shared "
                             f"sufficient-statistics table")
    skill_pd = 1.0 - np.sqrt(g["a_un"]) / np.sqrt(g["a_cca"])
    skill_cl = 1.0 - np.sqrt(g["b_un"]) / np.sqrt(g["b_cca"])
    return skill_cl - skill_pd


VAL_ROOT = Path(f"{RESULTS_ROOT}/"
                "paper_pipeline_full_validation_20260728T104924Z")

VAL_F4_NPZ = VAL_ROOT / "plot_sources" / "F4__PD_global_rmse_maps_source.npz"

VAL_F4_SHA = "e1f7089fe00c08fc058f8e4fc57125909b6cbb1096c620c0f3ed69ec0895768a"

HARD_THRESHOLD_K = 2.9775419407231962

HARD_N_CELLS = 35849

HARD_AREA_FRACTION = 0.010000335860979903

HARD_GATE_TOL = 1.0e-9

HARD_METHODS = ["bilinear", "cca", "unet"]


def _weighted_quantile(values: np.ndarray, weights: np.ndarray,
                       q: float) -> float:


    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    cw = cw / cw[-1]
    return float(np.interp(q, cw, v))


def _cell_rmse_maps(path) -> dict:

    with np.load(read_only_reference(path), allow_pickle=False) as d:
        return {"bilinear": np.asarray(d["bilinear_rmse_K"], np.float64),
                "cca": np.asarray(d["cca_rmse_K"], np.float64),
                "unet": np.asarray(d["unet_rmse_K"], np.float64)}


def frozen_hard_mask() -> tuple[np.ndarray, np.ndarray, dict]:

    got = sha256(read_only_reference(VAL_F4_NPZ))
    if got != VAL_F4_SHA:
        raise SystemExit(f"validation mask authority hash {got} != frozen "
                         f"{VAL_F4_SHA}")
    val = _cell_rmse_maps(VAL_F4_NPZ)["bilinear"]
    height = val.shape[0]
    w_row, _ = area_weights_from_lat(grid_lat_centers(height))
    w2d = np.ascontiguousarray(np.broadcast_to(w_row[:, None], val.shape))
    threshold = _weighted_quantile(val.ravel(), w2d.ravel(), 0.99)
    mask = val >= threshold
    area_fraction = float(w2d[mask].sum() / w2d.sum())
    n_cells = int(mask.sum())
    if abs(threshold - HARD_THRESHOLD_K) > HARD_GATE_TOL:
        raise SystemExit(f"hard-mask threshold {threshold!r} != established "
                         f"{HARD_THRESHOLD_K!r}")
    if n_cells != HARD_N_CELLS:
        raise SystemExit(f"hard-mask cell count {n_cells} != established "
                         f"{HARD_N_CELLS}")
    if abs(area_fraction - HARD_AREA_FRACTION) > HARD_GATE_TOL:
        raise SystemExit(f"hard-mask area fraction {area_fraction!r} != "
                         f"established {HARD_AREA_FRACTION!r}")
    meta = {
        "definition": "frozen VALIDATION cellwise bilinear RMSE >= its "
                      "area-weighted 99th percentile; defined before reading "
                      "any PD-test, MH or SSP5-8.5 performance and applied "
                      "unchanged to all three climates",
        "no_per_climate_reselection": True,
        "source_map": str(VAL_F4_NPZ), "source_map_sha256": got,
        "threshold_K": threshold, "n_cells": n_cells,
        "n_cells_total": int(mask.size), "area_fraction": area_fraction,
        "expected_threshold_K": HARD_THRESHOLD_K,
        "expected_n_cells": HARD_N_CELLS,
        "expected_area_fraction": HARD_AREA_FRACTION,
        "gate_tolerance_abs": HARD_GATE_TOL,
        "area_weight_convention": "float64 mean-one cosine of the regular "
                                  "grid latitude centres (frozen CCA "
                                  "convention)"}
    log(f"frozen hard mask: threshold {threshold:.10f} K, {n_cells} cells, "
        f"area fraction {area_fraction:.12f} (gated)")
    return mask, w2d, meta


def hard_cell_rmse(maps: dict, mask: np.ndarray, w2d: np.ndarray) -> dict:


    out = {}
    for m in HARD_METHODS:
        field = maps[m]
        if field.shape != mask.shape:
            raise SystemExit(f"cellwise map grid {field.shape} differs from "
                             f"the validation mask grid {mask.shape}")
        if not np.isfinite(field).all():
            raise SystemExit(f"non-finite value in the {m} cellwise RMSE map")
        mse = float((w2d[mask] * field[mask] ** 2).sum() / w2d[mask].sum())
        out[m] = math.sqrt(mse)
    return out


def _f6_regional_rmse() -> dict:

    payload = json.loads(read_only_reference(PD_REGIONAL_JSON).read_text())
    if payload["regions"] != REGIONS or payload["periods"] != PERIODS:
        raise SystemExit("PD regional report region/period order changed")
    out = {}
    for key, entry in payload["regional_period_rmse_K"].items():
        region, period = key.split("|")
        out[(region, period)] = {"cca": float(entry["cca"]),
                                 "unet": float(entry["unet"]),
                                 "delta_rmse_K": float(entry["delta_rmse_K"])}
    return out


def cmd_rmse_skill(args) -> int:
    global PD_ROOT, MH_ROOT, READONLY_ROOTS
    if args.pd_root:
        PD_ROOT = guard(args.pd_root)
    if args.mh_root:
        MH_ROOT = guard(args.mh_root)
    ssp_root = guard(args.retention_root)
    output_root = guard(args.output_root)
    READONLY_ROOTS = [PD_ROOT, MH_ROOT, ssp_root]
    for source in READONLY_ROOTS:
        if output_root.resolve() == source.resolve() or source.resolve() in output_root.resolve().parents:
            raise SystemExit("Use an output directory outside the three retained input roots")
    output_root.mkdir(parents=True, exist_ok=True)
    payload = three_climate_rmse_skill_statistics(ssp_root)
    jdump(payload, output_root / "rmse_skill_statistics.json")
    stats = payload["statistics"]
    write_csv(output_root / "rmse_skill_statistics.csv",
              ["quantity", "estimate", "ci_lo_95", "ci_hi_95", "units", "definition"],
              [[q, repr(float(stats[q]["estimate"])), repr(float(stats[q]["ci"][0])),
                repr(float(stats[q]["ci"][1])), stats[q]["units"],
                stats[q].get("definition", "")] for q in sorted(stats)])
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

    ft = sub.add_parser("f11-targets", help="accumulate physical HR-target temperature histograms")
    ft.add_argument("--retention-root", required=True)
    ft.add_argument("--output-root", required=True)
    ft.set_defaults(func=cmd_f11_targets)

    rs = sub.add_parser("rmse-skill", help="export the final three-climate RMSE-ratio skill and intervals")
    rs.add_argument("--retention-root", required=True, help="retained SSP5-8.5 root")
    rs.add_argument("--output-root", required=True)
    rs.add_argument("--pd-root", default=None, help="override the retained present-day root")
    rs.add_argument("--mh-root", default=None, help="override the retained mid-Holocene root")
    rs.set_defaults(func=cmd_rmse_skill)

    return p.parse_args(argv)


def main(argv=None) -> int:
    assert_artifact_only()
    args = parse_args(argv)
    rc = args.func(args)
    assert_artifact_only()
    return rc


if __name__ == "__main__":
    sys.exit(main())
