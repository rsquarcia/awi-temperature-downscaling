"""Calculate present-day metrics from retained predictions and targets.
Keeps the original shard accumulation, regional/seasonal RMSE and spectra,
including the original case-study field extraction. Figure rendering and
review-package assembly are omitted. Historical intermediate skill columns
use MSE ratios; use the SSP5-8.5 rmse-skill command for the final paper's
RMSE-ratio skill and confidence intervals. External research inputs are required."""

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

    "midholocene", "mid_holocene", "awi_downscaling_mh", "/mh/", "_mh_",
    "holocene",

    "best_val_mse.pt", "selected_cca_model", "y_residual_eof_basis",
    "val_x_scores", "checkpoint",
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


ELL_MAX = 426

K_MAX = 437

PD_LABEL = "PD test (2012–2014)"                                          


def set_labels_from_meta(meta: dict) -> None:


    global PD_LABEL, F5_STATE_LABEL, F5_DATE
    first, last = meta["dates"][0][:4], meta["dates"][-1][:4]
    split = str(meta.get("split", "test"))
    name = "PD test" if split == "test" else f"PD {split}"
    PD_LABEL = f"{name} ({first}–{last})"
    F5_STATE_LABEL = (f"{name} ({first}-{last}), complete "
                      f"{meta['n_days']}-day split")
    F5_DATE = meta["dates"][0]


REGIONS = ["global", "land", "ocean", "elevation_gt_1000m", "elevation_le_1000m",
           "tropics", "midlat_north", "midlat_south", "highlat_north",
           "highlat_south", "arctic_gt_80N", "antarctic_lt_80S"]

PERIODS = ["annual", "DJF", "MAM", "JJA", "SON"]

T1_REGIONS = ["global", "land", "ocean", "elevation_gt_1000m",
              "arctic_gt_80N", "antarctic_lt_80S"]

HR_STATICS = Path(f"{DATA_ROOT}/grids/static_masks/"
                  "surface_fractions_hr.nc")

COARSENED_DIR = Path(f"{DATA_ROOT}/coarsened")

COARSENED_PATTERN = "atm_regular_1d_2t_1d_{ym}-{ym}.nc"

F5_LON_REQ = (5.5, 15.5)

F5_LAT_REQ = (44.0, 48.0)

F5_VMIN, F5_VMAX = 260.0, 285.0

F5_SATURATION_CSV = f"F5__Alps_saturation_{F5_VMIN:.0f}to{F5_VMAX:.0f}K.csv"

F5_STATE_LABEL = "PD test (2012-2014), complete 1096-day split"

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


def edges_from_centers(c):
    c = np.asarray(c, dtype=np.float64)
    mid = 0.5 * (c[1:] + c[:-1])
    return np.concatenate([[c[0] - (mid[0] - c[0])], mid,
                           [c[-1] + (c[-1] - mid[-1])]])


def lon180(lon):
    return ((np.asarray(lon, dtype=np.float64) + 180.0) % 360.0) - 180.0


BOOTSTRAP = {"block_length_days": 60, "n_resamples": 10000, "seed": 20260715,
             "ci_percentiles": [2.5, 97.5],
             "scheme": "paired circular moving-block bootstrap"}

ORO_NC = Path(f"{DATA_ROOT}/grids/oro_hr.nc")


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


NORM_T2M_INP_MEAN = 277.8568272051984

NORM_T2M_INP_STD = 21.581598298286263


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


def retained_day(retention_root: Path, meta: dict, date: str) -> dict:

    if date not in meta["dates"]:
        raise SystemExit(f"date {date} is not in the retained output")
    i = meta["dates"].index(date)
    names = ("b2_prediction_norm", "b2_baseline_norm", "cca_prediction_norm")
    for s, part in enumerate(meta["shards"]):
        if part["lo"] <= i < part["hi"]:
            sh = open_shard(retention_root, meta, s)
            k = i - part["lo"]
            out = {name: sh[name][k] for name in names}
            out["_shapes"] = {name: list(sh[name].shape) for name in names}
            out["_dtypes"] = {name: str(sh[name].dtype) for name in names}
            return out
    raise SystemExit(f"date {date} is not covered by any shard")


def realized_window(lat_sorted, lon_sorted):


    lat_e = edges_from_centers(lat_sorted)
    lon_e = edges_from_centers(lon_sorted)

    def nearest(edges, value):
        i = int(np.argmin(np.abs(edges - value)))
        return float(edges[i]), i

    lat_min_g, i0 = nearest(lat_e, F5_LAT_REQ[0])
    lat_max_g, i1 = nearest(lat_e, F5_LAT_REQ[1])
    lon_min_g, j0 = nearest(lon_e, F5_LON_REQ[0])
    lon_max_g, j1 = nearest(lon_e, F5_LON_REQ[1])
    if not (i0 < i1 and j0 < j1):
        raise SystemExit("realized F5 window is degenerate")
    rows = np.arange(i0, i1)
    cols = np.arange(j0, j1)
    crop_lat_e = edges_from_centers(lat_sorted[rows])
    crop_lon_e = edges_from_centers(lon_sorted[cols])
    return {"lon_min": float(crop_lon_e[0]), "lon_max": float(crop_lon_e[-1]),
            "lat_min": float(crop_lat_e[0]), "lat_max": float(crop_lat_e[-1]),
            "rows": rows, "cols": cols,
            "lat_edges": crop_lat_e, "lon_edges": crop_lon_e,
            "global_grid_snapped_lonlat": [lon_min_g, lon_max_g,
                                           lat_min_g, lat_max_g],
            "global_grid_edge_indices": [int(j0), int(j1), int(i0), int(i1)]}


def saturation_stats(name: str, values: np.ndarray) -> dict:


    a = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(a)
    n = int(finite.sum())
    if n == 0:
        raise SystemExit(f"{name}: no finite displayed cells")
    lo = int((a[finite] < F5_VMIN).sum())
    hi = int((a[finite] > F5_VMAX).sum())
    return {"panel": name, "n_displayed_cells": int(a.size),
            "n_finite_displayed_cells": n,
            "n_below": lo, "pct_below": 100.0 * lo / n,
            "n_above": hi, "pct_above": 100.0 * hi / n,
            "min_K": float(a[finite].min()), "max_K": float(a[finite].max())}


def prepare_f5(retention_root: Path, date: str) -> dict:


    import netCDF4
    import zarr

    meta = read_retention(retention_root)
    if date not in meta["dates"]:
        raise SystemExit(f"{date} not retained (retained: {meta['dates']})")
    fields = retained_day(retention_root, meta, date)
    tgt_mean = float(meta["normalization"]["t2m_tgt_mean_K"])
    tgt_std = float(meta["normalization"]["t2m_tgt_std_K"])

    lat_g = np.load(retention_root / meta["grid"]["coords"]["hr_lat"])
    lon_g = np.load(retention_root / meta["grid"]["coords"]["hr_lon"])
    lon_g180 = lon180(lon_g)
    lat_order = np.argsort(lat_g)
    lon_order = np.argsort(lon_g180)
    win = realized_window(lat_g[lat_order], lon_g180[lon_order])
    rows = lat_order[win["rows"]]
    cols = lon_order[win["cols"]]
    sel = np.ix_(rows, cols)
    hr_lat = lat_g[rows]
    hr_lon = lon_g180[cols]
    if not (np.all(np.diff(hr_lat) > 0) and np.all(np.diff(hr_lon) > 0)):
        raise SystemExit("HR crop coordinates are not ascending")

    to_K = lambda a: tgt_mean + np.asarray(a, dtype=np.float64) * tgt_std
    group = zarr.open_group(str(guard(meta["input_store"])), mode="r")
    day = [d.decode() if isinstance(d, bytes) else str(d)
           for d in np.asarray(group["dates"][:]).tolist()].index(date)
    target_K = to_K(np.asarray(group["targets"][day, 0])[sel])
    bilinear_K = to_K(np.asarray(fields["b2_baseline_norm"])[sel])
    unet_K = to_K(np.asarray(fields["b2_prediction_norm"])[sel])
    cca_K = to_K(np.asarray(fields["cca_prediction_norm"])[sel])


    coarse = guard(COARSENED_DIR / COARSENED_PATTERN.format(ym=date[:4] + date[5:7]))
    with netCDF4.Dataset(coarse, "r") as ds:
        tvar = ds.variables["time_counter"]
        labels = [f"{d.year:04d}-{d.month:02d}-{d.day:02d}" for d in
                  netCDF4.num2date(tvar[:], tvar.units,
                                   getattr(tvar, "calendar", "standard"))]
        if date not in labels:
            raise SystemExit(f"{date} not in {coarse}")
        it = labels.index(date)
        lr = np.asarray(ds.variables["2t"][it], dtype=np.float64)
        lr_lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        lr_lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
    lr_lat_order = np.argsort(lr_lat)
    lr_lon_order = np.argsort(lon180(lr_lon))
    lr_lat_s = lr_lat[lr_lat_order]
    lr_lon_s = lon180(lr_lon)[lr_lon_order]
    lr_lat_e = edges_from_centers(lr_lat_s)
    lr_lon_e = edges_from_centers(lr_lon_s)
    li = np.where((lr_lat_e[1:] > win["lat_min"])
                  & (lr_lat_e[:-1] < win["lat_max"]))[0]
    lo = np.where((lr_lon_e[1:] > win["lon_min"])
                  & (lr_lon_e[:-1] < win["lon_max"]))[0]
    lr_crop = lr[np.ix_(lr_lat_order, lr_lon_order)][np.ix_(li, lo)]
    lr_lat_edges = lr_lat_e[li[0]:li[-1] + 2]
    lr_lon_edges = lr_lon_e[lo[0]:lo[-1] + 2]


    panels = [("coarsened HR input", lr_crop),
              ("bilinearly upsampled input", bilinear_K),
              ("HR target", target_K),
              ("CCA prediction", cca_K),
              ("B2 U-Net prediction", unet_K)]
    sat = [saturation_stats(n, a) for n, a in panels]
    csv_path = write_csv(
        OUT / F5_SATURATION_CSV,
        ["panel", "n_finite_displayed_cells",
         f"n_cells_below_{F5_VMIN:.0f}K", f"pct_cells_below_{F5_VMIN:.0f}K",
         f"n_cells_above_{F5_VMAX:.0f}K", f"pct_cells_above_{F5_VMAX:.0f}K",
         "min_K", "max_K", "scale_min_K", "scale_max_K", "date", "state"],
        [[s["panel"], s["n_finite_displayed_cells"], s["n_below"],
          f"{s['pct_below']:.6f}", s["n_above"],
          f"{s['pct_above']:.6f}", f"{s['min_K']:.6f}",
          f"{s['max_K']:.6f}", F5_VMIN, F5_VMAX, date,
          F5_STATE_LABEL]
         for s in sat])

    mean_lat = 0.5 * (win["lat_min"] + win["lat_max"])
    cosphi = math.cos(math.radians(mean_lat))
    npz_path = SRC / f"F5__fields_{date}.npz"
    np.savez_compressed(
        npz_path,
        date=np.asarray(date),
        native_lr_K=lr_crop,
        lr_lat_centers=lr_lat_s[li], lr_lon_centers=lr_lon_s[lo],
        lr_lat_edges=lr_lat_edges, lr_lon_edges=lr_lon_edges,
        bilinear_K=bilinear_K, cca_K=cca_K, unet_K=unet_K, target_K=target_K,
        hr_lat=hr_lat, hr_lon=hr_lon,
        hr_lat_edges=win["lat_edges"], hr_lon_edges=win["lon_edges"],
        window_lon_min=np.float64(win["lon_min"]),
        window_lon_max=np.float64(win["lon_max"]),
        window_lat_min=np.float64(win["lat_min"]),
        window_lat_max=np.float64(win["lat_max"]),
        scale_min_K=np.float64(F5_VMIN), scale_max_K=np.float64(F5_VMAX),
    )
    ver = {
        "kind": "F5_PLOT_SOURCE_EXTRACTION",
        "created_utc": utc(), "date": date,
        "state": F5_STATE_LABEL,
        "requested_window_lonlat": [F5_LON_REQ[0], F5_LON_REQ[1],
                                    F5_LAT_REQ[0], F5_LAT_REQ[1]],
        "realized_window_lonlat": [win["lon_min"], win["lon_max"],
                                   win["lat_min"], win["lat_max"]],
        "realized_bounds_are_true_grid_cell_edges": True,
        "global_grid_snapped_window_lonlat": win["global_grid_snapped_lonlat"],
        "global_grid_edge_indices_lon0_lon1_lat0_lat1":
            win["global_grid_edge_indices"],
        "realized_vs_global_snapped_max_abs_diff_deg": max(
            abs(win["lon_min"] - win["global_grid_snapped_lonlat"][0]),
            abs(win["lon_max"] - win["global_grid_snapped_lonlat"][1]),
            abs(win["lat_min"] - win["global_grid_snapped_lonlat"][2]),
            abs(win["lat_max"] - win["global_grid_snapped_lonlat"][3])),
        "delta_lon_deg": win["lon_max"] - win["lon_min"],
        "delta_lat_deg": win["lat_max"] - win["lat_min"],
        "mean_latitude_deg": mean_lat, "cos_mean_latitude": cosphi,
        "data_aspect_1_over_cos": 1.0 / cosphi,
        "box_aspect_h_over_w": (win["lat_max"] - win["lat_min"])
                               / ((win["lon_max"] - win["lon_min"]) * cosphi),
        "hr_crop_shape": list(target_K.shape),
        "hr_lat_range": [float(hr_lat.min()), float(hr_lat.max())],
        "hr_lon_range": [float(hr_lon.min()), float(hr_lon.max())],
        "lr_cells_intersecting": list(lr_crop.shape),
        "lr_lat_edge_range": [float(lr_lat_edges[0]), float(lr_lat_edges[-1])],
        "lr_lon_edge_range": [float(lr_lon_edges[0]), float(lr_lon_edges[-1])],
        "shared_scale_K": {"min": F5_VMIN, "max": F5_VMAX,
                           "note": "exact requested revision; not data-derived"},
        f"saturation_{F5_VMIN:.0f}_{F5_VMAX:.0f}K": sat,
        "saturation_csv": str(csv_path),
        "units": f"physical Kelvin: field_K = {tgt_mean} + field_norm * {tgt_std}",
        "retained_source": {
            "retention_root": str(retention_root),
            "retention_metadata_sha256":
                sha256(retention_root / "retention_metadata.json"),
            "arrays": fields["_shapes"], "dtypes": fields["_dtypes"],
            "b2_identity": {k: v for k, v in meta["identity"].items()
                            if k.startswith("b2_") and k.endswith("sha256")},
            "cca_identity": {"model_sha256": meta["identity"]["cca_model_sha256"],
                             "hyperparameters":
                                 meta["identity"]["cca_hyperparameters"]},
        },
        "canonical_read_only_sources": {
            "target": str(meta["input_store"]), "native_lr": str(coarse),
            "hr_grid": str(HR_STATICS)},
        "no_model_access": True, "no_inference": True,
        "npz": str(npz_path), "npz_sha256": sha256(npz_path),
    }
    jdump(ver, SRC / f"F5__extraction_verification_{date}.json")
    log(f"F5 plot source prepared -> {npz_path}")
    return ver


F2_FROZEN_TABLE = Path(f"{RESULTS_ROOT}/cca_final_run/"
                       "outputs/user_facing/cca_eval_suite/cca_specific/tables/"
                       "cca_grid_profiled_kx_ky_heatmap.csv")

SPECTRA_ABS_FIELDS = ["HR target", "Bilinear", "CCA", "U-Net"]

SPECTRA_ERR_FIELDS = ["bilinear", "cca", "unet"]

SUM_KEYS = ["w", "e2", "ea", "e1", "bil_e2", "bil_ea", "bil_e1"]

F5_DATE = "2009-01-01"


def cmd_prepare_shard(args) -> int:


    global F5_DATE
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
    log(f"shard {s:02d}: {n} days {dates[0]}..{dates[-1]}, ell_max={ell_max}")

    for k in range(n):
        date = dates[k]
        si = store_idx[k]
        period_ids = (0, PERIODS.index(season_name(date)))

        b2_pred = np.asarray(sh["b2_prediction_norm"][k], dtype=np.float64)
        b2_base = np.asarray(sh["b2_baseline_norm"][k], dtype=np.float64)
        cca_pred = np.asarray(sh["cca_prediction_norm"][k], dtype=np.float64)
        tgt_u = np.asarray(group["targets"][si, 0], dtype=np.float64)


        base_c = baseline_from_inputs(
            np.asarray(group["inputs"][si, 0], dtype=np.float64),
            tgt_mean, float(std))

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

        if date == F5_DATE:
            F5_DATE = date
            prepare_f5(retention_root, date)
            log(f"shard {s:02d}: F5 case-study fields extracted for {date}")
        if (k + 1) % 10 == 0 or k == n - 1:
            log(f"shard {s:02d}: {k + 1}/{n} days")

    payload = {f"acc_{k}": v for k, v in acc.items()}
    payload.update({f"daily_{k}": v for k, v in daily.items()})
    payload.update({f"map_{k}": v for k, v in map_acc.items()})
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
        "target_std_K": np.float64(std),
    })
    tmp = out_path.with_suffix(".writing.npz")
    np.savez(tmp, **payload)
    os.replace(tmp, out_path)
    log(f"shard {s:02d}: partial written -> {out_path}")
    print(json.dumps({"status": "PASS", "mode": "prepare-shard", "shard": s,
                      "n_days": n, "partial": str(out_path)}))
    return 0


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
                "delta_rmse_K": (math.sqrt(mse_cca) - math.sqrt(mse_unet)) * std,
                "skill_vs_cca": 1.0 - mse_unet / mse_cca,
            }


    write_csv(SRC / "T1__PD_regional_rmse_source.csv",
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
    t1_csv = write_csv(
        SRC / "T1__PD_global_metrics.csv",
        ["quantity", "method_or_pair", "estimate", "ci_lo_95", "ci_hi_95",
         "units", "reference_method", "sign_interpretation", "bootstrap_scheme",
         "block_length_days", "n_resamples"],
        [[q, meth, repr(point[q]), repr(ci[q][0]), repr(ci[q][1]), unit, ref,
          sign, BOOTSTRAP["scheme"], BOOTSTRAP["block_length_days"],
          BOOTSTRAP["n_resamples"]] for q, meth, unit, ref, sign in order])

    write_csv(SRC / "F6__region_period_source.csv",
              ["region", "period", "delta_rmse_K", "skill_vs_cca"],
              [[r, p, repr(float(regional[(r, p)]["delta_rmse_K"])),
                repr(float(regional[(r, p)]["skill_vs_cca"]))]
               for r in REGIONS for p in PERIODS])

    lat_g = np.load(retention_root / meta["grid"]["coords"]["hr_lat"])
    lon_g = np.load(retention_root / meta["grid"]["coords"]["hr_lon"])
    np.savez_compressed(
        SRC / "F4__PD_global_rmse_maps_source.npz",
        latitude=lat_g, longitude=lon_g,
        bilinear_rmse_K=np.sqrt(maps["bilinear_e2"] / n_days) * std,
        cca_rmse_K=np.sqrt(maps["cca_e2"] / n_days) * std,
        unet_rmse_K=np.sqrt(maps["unet_e2"] / n_days) * std,
        delta_rmse_K=(np.sqrt(maps["cca_e2"] / n_days)
                      - np.sqrt(maps["unet_e2"] / n_days)) * std)


    ell = np.asarray(ell)
    keep = (ell >= 1) & (ell <= ELL_MAX)
    abs_rows = []
    for fi, name in enumerate(SPECTRA_ABS_FIELDS):
        series = sph_abs[fi] / n_days * std ** 2
        for e, v in zip(ell[keep], series[keep]):
            abs_rows.append([int(e), name, f"{v:.10e}"])
    write_csv(SRC / "F7__absolute_field_spherical.csv",
              ["ell", "series", "C_ell_K2"], abs_rows)
    err_series = {name: sph_err[fi] / n_days * std ** 2
                  for fi, name in enumerate(SPECTRA_ERR_FIELDS)}
    write_csv(SRC / "F7__prediction_error_spherical.csv",
              ["ell", "bilinear_error_K2", "cca_error_K2", "unet_error_K2"],
              [[int(e), f"{err_series['bilinear'][e]:.10e}",
                f"{err_series['cca'][e]:.10e}", f"{err_series['unet'][e]:.10e}"]
               for e in ell[keep]])


    kk = np.asarray(kk)
    kkeep = (kk >= 1) & (kk <= K_MAX)
    zon_denom = zon_row_weight_sum * float(width)
    zon_abs_K2 = zon_abs / zon_denom * std ** 2
    zon_err_K2 = zon_err / zon_denom * std ** 2
    zrows = []
    for fi, name in enumerate(SPECTRA_ABS_FIELDS):
        for k_, v in zip(kk[kkeep], zon_abs_K2[fi][kkeep]):
            zrows.append([int(k_), name, f"{v:.10e}"])
    write_csv(SRC / "F7__absolute_field_zonal.csv", ["k", "series", "power_K2"],
              zrows)
    zerr = {name: zon_err_K2[fi] for fi, name in enumerate(SPECTRA_ERR_FIELDS)}
    write_csv(SRC / "F7__prediction_error_zonal.csv",
              ["k", "bilinear_error_K2", "cca_error_K2", "unet_error_K2"],
              [[int(k_), f"{zerr['bilinear'][k_]:.10e}",
                f"{zerr['cca'][k_]:.10e}", f"{zerr['unet'][k_]:.10e}"]
               for k_ in kk[kkeep]])


    carried = []
    rows = load_csv(F2_FROZEN_TABLE)
    write_csv(SRC / "F2__CCA_dimension_selection_source.csv",
              ["Kx", "Ky", "val_rmse_K", "is_selected"],
              [[r["Kx"], r["Ky"], r["val_rmse_K"], r["is_selected"]]
               for r in rows])
    carried.append({"file": "F2__CCA_dimension_selection_source.csv",
                    "from": str(F2_FROZEN_TABLE),
                    "sha256": sha256(F2_FROZEN_TABLE),
                    "note": "frozen CCA dimension-selection landscape from the "
                            "VALIDATION model-selection grid; it is a frozen "
                            "model-selection artifact and is deliberately not "
                            "recomputed on the held-out test split"})

    reduced = {
        "kind": "REDUCE_REPORT", "created_utc": utc(),
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
        "f7_sources_recomputed_from_this_run": [
            "F7__absolute_field_spherical.csv", "F7__absolute_field_zonal.csv",
            "F7__prediction_error_spherical.csv",
            "F7__prediction_error_zonal.csv"],
        "f7_provenance": "recomputed from the retained complete validation "
                         "predictions",
        "carried_forward_frozen_sources": carried,
        "plot_sources_dir": str(SRC),
        "t1_csv": str(t1_csv),
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
