"""Shared definitions and numerical helpers for the error-decomposition analyses.
Contains climate/input locations, frozen normalization and dimensions, area
weights, region masks, spectral transforms, block resampling and array loaders.
The three-climate diagnostics import these common conventions to keep their
weighted projections and metrics consistent. External files are loaded only
when the corresponding loader functions are called."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from downscaling_numerics import (
    GaussLegendreSHTPlan,
    area_weights_from_lat,
    circular_block_bootstrap_indices,
    cubic_latitude_interpolation,
    derive_spherical_ell_max,
    exact_cell_area_row_weights,
    one_sided_zonal_power,
    season_name,
    zonal_row_weighted_power,
)

import os


DATA_ROOT = os.environ.get("DATA_ROOT", "data_root")                                                     
RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "results_root")                                                  


PD_TRAIN_ZARR = Path(f"{DATA_ROOT}/zarr/awi_downscaling_train.zarr")
PD_VAL_ZARR = Path(f"{DATA_ROOT}/zarr/awi_downscaling_val.zarr")
PD_ZARR = Path(f"{DATA_ROOT}/zarr/awi_downscaling_test.zarr")
MH_ZARR = Path(f"{RESULTS_ROOT}/mh_preprocess_20260728T173229Z/results/"
               "mh_2076_2078.zarr")
SSP_ZARR = Path(f"{RESULTS_ROOT}/n43_ssp585_final_20260729T130103Z/results/"
                "n43_ssp585_2096_2098.zarr")
NORM_STATS = Path(f"{DATA_ROOT}/grids/norm_stats.json")
LR_GRID_TXT = Path(f"{DATA_ROOT}/grids/lr_grid.txt")
HR_GRID_TXT = Path(f"{DATA_ROOT}/grids/hr_grid.txt")
W_LR2HR = Path(f"{DATA_ROOT}/grids/weights_lr2hr.nc")
W_HR2LR = Path(f"{DATA_ROOT}/grids/weights_hr2lr.nc")
LR_HEIGHT, LR_WIDTH = 192, 400
HR_STATICS = Path(f"{DATA_ROOT}/grids/static_masks/"
                  "surface_fractions_hr.nc")
ORO_NC = Path(f"{DATA_ROOT}/grids/oro_hr.nc")
CCA_MODEL = Path(f"{RESULTS_ROOT}/cca_final_run/"
                 "outputs/cca_grid/selected_cca_model.npz")
CCA_HEAVY = Path(f"{RESULTS_ROOT}/cca_final_run/"
                 "outputs_heavy")
Y_BASIS = CCA_HEAVY / "eof_bases" / "y_residual_eof_basis_maxK1024.npz"
X_BASIS = CCA_HEAVY / "eof_bases" / "x_input_eof_basis_maxK10592.npz"
PD_ROOT = Path(f"{RESULTS_ROOT}/"
               "paper_pipeline_pd_test_r2_20260728T130323Z")
MH_ROOT = Path(f"{RESULTS_ROOT}/mh_full_run_20260728T180741Z")
SSP_ROOT = Path(f"{RESULTS_ROOT}/ssp585_full_run_20260729T130103Z")
VAL_ROOT = Path(f"{RESULTS_ROOT}/"
                "paper_pipeline_full_validation_20260728T104924Z")


MH_SUPPORT_RUN = Path(f"{RESULTS_ROOT}/"
                      "mh_pd_training_support_conditional_shift_20260802T075915Z")
MH_SPATIAL_RUN = Path(f"{RESULTS_ROOT}/"
                      "mh_pd_spatial_pattern_shift_20260801T204818Z")

CLIMATES = ("pd", "mh", "ssp")
POPULATIONS = ("train", "val", "test", "mh", "ssp")
MODEL_POPULATIONS = ("val", "test", "mh", "ssp")                               
POP_ZARR = {"train": PD_TRAIN_ZARR, "val": PD_VAL_ZARR,
            "test": PD_ZARR, "mh": MH_ZARR, "ssp": SSP_ZARR}
POP_NDAYS = {"train": 10593, "val": 1095, "test": 1096, "mh": 1096, "ssp": 1096}
POP_YEARS = {"train": list(range(1980, 2009)), "val": [2009, 2010, 2011],
             "test": [2012, 2013, 2014], "mh": [2076, 2077, 2078],
             "ssp": [2096, 2097, 2098]}
POP_LABEL = {"train": "PD training (1980-2008)", "val": "PD validation (2009-2011)",
             "test": "PD test (2012-2014)", "mh": "MH (2076-2078)",
             "ssp": "SSP5-8.5 (2096-2098)"}
POP_PRED_ROOT = {"val": VAL_ROOT, "test": PD_ROOT, "mh": MH_ROOT, "ssp": SSP_ROOT}
CLIMATE_ZARR = {"pd": PD_ZARR, "mh": MH_ZARR, "ssp": SSP_ZARR}
CLIMATE_ROOT = {"pd": PD_ROOT, "mh": MH_ROOT, "ssp": SSP_ROOT}
CLIMATE_SCORES = {"pd": PD_ROOT / "scores" / "test_x_scores_maxK10592.npy",
                  "mh": MH_ROOT / "scores" / "mh_x_scores_maxK10592.npy",
                  "ssp": SSP_ROOT / "scores" / "ssp585_x_scores_maxK10592.npy"}
CLIMATE_LABEL = {"pd": "PD test (2012-2014)", "mh": "MH (2076-2078)",
                 "ssp": "SSP5-8.5 (2096-2098)"}


ACCEPTED_RMSE = {
    "val":  {"bilinear": None,          "cca": 0.293560267423705,
             "unet": None},
    "test": {"bilinear": 0.7006977149,  "cca": 0.2953558449,
             "unet": 0.1985303218},
    "mh":   {"bilinear": 0.7193696818,  "cca": 0.3219676866,
             "unet": 0.2148129915},
    "ssp":  {"bilinear": 0.6763885226,  "cca": 0.3154044439,
             "unet": 0.1943038825},
}
ACCEPTED_RMSE_TOL_REL = 1.0e-6


NORM_TISR_MEAN = 3223042.5108223786
NORM_TISR_STD = 1786193.5906606768
NORM_T2M_INP_MEAN = 277.8568272051984
NORM_T2M_INP_STD = 21.581598298286263
TARGET_MEAN_K = 277.8518781731243
TARGET_STD_K = 21.627892139211994

HEIGHT, WIDTH = 1280, 2624
N_DAYS = 1096
KY = 512
KX = 10592

SEASONS = ("DJF", "MAM", "JJA", "SON")
PERIODS = ("annual",) + SEASONS

ELL_MAX = 426
K_MAX = 437
K_FULL = WIDTH // 2

BOOTSTRAP = {"block_length_days": 60, "n_resamples": 10000, "seed": 20260715,
             "ci_percentiles": [2.5, 97.5],
             "scheme": "independent circular moving-block bootstrap per climate"}
PERM = {"block_length_days": 60, "n_permutations": 4000, "seed": 20260715,
        "scheme": "calendar-position-preserving 60-day block label swap"}


REGIONS = ["global", "land", "ocean", "elevation_gt_1000m",
           "elevation_le_1000m", "tropics", "midlat_north", "midlat_south",
           "highlat_north", "highlat_south", "arctic_gt_80N",
           "antarctic_lt_80S"]
DERIVED_REGIONS = ["coastal_mixed", "elevation_gt_2000m"]
ALL_REGIONS = REGIONS + DERIVED_REGIONS

REGION_DISP = {
    "global": "Global", "land": "Land", "ocean": "Ocean",
    "elevation_gt_1000m": "Elevation > 1000 m",
    "elevation_le_1000m": "Elevation <= 1000 m",
    "tropics": "Tropics (|lat| < 23.5 deg)",
    "midlat_north": "N midlatitudes (23.5-60 deg)",
    "midlat_south": "S midlatitudes (23.5-60 deg)",
    "highlat_north": "N high latitudes (60-80 deg)",
    "highlat_south": "S high latitudes (60-80 deg)",
    "arctic_gt_80N": "Arctic (>= 80N)",
    "antarctic_lt_80S": "Antarctic (<= 80S)",
    "coastal_mixed": "Coastal mixed cells (0.05 < lsm < 0.95)",
    "elevation_gt_2000m": "Elevation > 2000 m",
}


DAILY_QUANTITIES = ["w", "B", "T", "r", "r2", "absr", "gradmag", "lapmag",
                    "lvar1", "lvar3", "lvar6", "o2", "rhat2", "anom2"]


WINDOWS = {"lvar1": 7, "lvar3": 21, "lvar6": 43}


HIST_SPEC = {
    "absr":    (0.0, 25.0, 2500),
    "gradmag": (0.0, 10.0, 2000),
    "lapmag":  (0.0, 25.0, 2500),
    "lstd1":   (0.0, 10.0, 2000),
    "lstd3":   (0.0, 10.0, 2000),
    "lstd6":   (0.0, 10.0, 2000),
}


def grid_lat_centers(height: int = HEIGHT) -> np.ndarray:
    return -90.0 + (np.arange(height, dtype=np.float64) + 0.5) * (180.0 / height)


def grid_lon_centers(width: int = WIDTH) -> np.ndarray:
    return (np.arange(width, dtype=np.float64) + 0.5) * (360.0 / width)


def lon180(lon: np.ndarray) -> np.ndarray:
    return ((np.asarray(lon, dtype=np.float64) + 180.0) % 360.0) - 180.0


def region_masks(lat_deg, lsm, oro_m) -> dict:

    height, width = lsm.shape
    lat = np.broadcast_to(np.asarray(lat_deg, dtype=np.float64)[:, None],
                          (height, width))
    ones = np.ones((height, width), dtype=bool)
    m = {
        "global": ones,
        "land": lsm >= 0.5,
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
    m["coastal_mixed"] = (lsm > 0.05) & (lsm < 0.95)
    m["elevation_gt_2000m"] = oro_m > 2000.0
    return m


def baseline_from_inputs(input_t2m_norm, tgt_mean=TARGET_MEAN_K,
                         tgt_std=TARGET_STD_K) -> np.ndarray:


    scale = NORM_T2M_INP_STD / tgt_std
    offset = (NORM_T2M_INP_MEAN - tgt_mean) / tgt_std
    return np.asarray(input_t2m_norm, dtype=np.float64) * scale + offset


def npz_member_memmap(path: Path, member: str) -> np.memmap:


    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise SystemExit(f"{path}:{member} is compressed; cannot memmap")
        header_offset = info.header_offset
        with zf.open(member) as handle:
            version = np.lib.format.read_magic(handle)
            shape, fortran, dtype = np.lib.format._read_array_header(handle, version)
            npy_header_bytes = handle.tell()
    with open(path, "rb") as fh:
        fh.seek(header_offset)
        local = fh.read(30)
        if local[:4] != b"PK\x03\x04":
            raise SystemExit(f"{path}:{member} bad local zip header")
        name_len = int.from_bytes(local[26:28], "little")
        extra_len = int.from_bytes(local[28:30], "little")
    data_start = header_offset + 30 + name_len + extra_len + npy_header_bytes
    order = "F" if fortran else "C"
    return np.memmap(path, dtype=dtype, shape=shape, order=order,
                     mode="r", offset=data_start)


def npz_small_member(path: Path, member: str) -> np.ndarray:
    with np.load(path, allow_pickle=False, mmap_mode=None) as z:
        return np.asarray(z[member])


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_statics():
    import netCDF4
    with netCDF4.Dataset(HR_STATICS, "r") as ds:
        lsm = np.asarray(ds.variables["lsm"][:], dtype=np.float64)
        lat_file = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        lon_file = np.asarray(ds.variables["lon"][:], dtype=np.float64)
    with netCDF4.Dataset(ORO_NC, "r") as ds:
        oro = np.asarray(ds.variables["var129"][:], dtype=np.float64)
    lsm = lsm.reshape(lsm.shape[-2], lsm.shape[-1])
    oro_m = oro.reshape(oro.shape[-2], oro.shape[-1])
    if lsm.shape != (HEIGHT, WIDTH) or oro_m.shape != (HEIGHT, WIDTH):
        raise SystemExit(f"statics shape mismatch {lsm.shape}/{oro_m.shape}")
    return lsm, oro_m, lat_file, lon_file


def read_norm_stats() -> dict:
    return json.loads(NORM_STATS.read_text())


def shard_slices(n_items: int, n_shards: int):
    base, rem = divmod(n_items, n_shards)
    out, start = [], 0
    for i in range(n_shards):
        size = base + (1 if i < rem else 0)
        out.append((start, start + size))
        start += size
    return out


LAT_BANDS = ["tropics", "midlat_north", "midlat_south", "highlat_north",
             "highlat_south", "arctic_gt_80N", "antarctic_lt_80S"]
SURF_CLASSES = ["ocean_open", "ocean_coastal", "land_coastal", "land_interior"]
ELEV_CLASSES = ["le1000", "1000_2000", "gt2000"]
N_CELL_CLASSES = len(LAT_BANDS) * len(SURF_CLASSES) * len(ELEV_CLASSES)


REGION_FROM_CLASSES = {
    "global": (None, None, None),
    "land": (["land_coastal", "land_interior"], None, None),
    "ocean": (["ocean_open", "ocean_coastal"], None, None),
    "coastal_mixed": (["ocean_coastal", "land_coastal"], None, None),
    "elevation_le_1000m": (None, ["le1000"], None),
    "elevation_gt_1000m": (None, ["1000_2000", "gt2000"], None),
    "elevation_gt_2000m": (None, ["gt2000"], None),
    "tropics": (None, None, ["tropics"]),
    "midlat_north": (None, None, ["midlat_north"]),
    "midlat_south": (None, None, ["midlat_south"]),
    "highlat_north": (None, None, ["highlat_north"]),
    "highlat_south": (None, None, ["highlat_south"]),
    "arctic_gt_80N": (None, None, ["arctic_gt_80N"]),
    "antarctic_lt_80S": (None, None, ["antarctic_lt_80S"]),
}


def cell_class_index(lat_deg, lsm, oro_m):

    import numpy as _np
    h, w = lsm.shape
    lat = _np.broadcast_to(_np.asarray(lat_deg, dtype=_np.float64)[:, None], (h, w))
    ilat = _np.full((h, w), -1, dtype=_np.int32)
    ilat[_np.abs(lat) < 23.5] = 0
    ilat[(lat >= 23.5) & (lat < 60.0)] = 1
    ilat[(lat <= -23.5) & (lat > -60.0)] = 2
    ilat[(lat >= 60.0) & (lat < 80.0)] = 3
    ilat[(lat <= -60.0) & (lat > -80.0)] = 4
    ilat[lat >= 80.0] = 5
    ilat[lat <= -80.0] = 6
    if (ilat < 0).any():
        raise SystemExit("latitude band partition is not exhaustive")
    isurf = _np.full((h, w), -1, dtype=_np.int32)
    isurf[lsm <= 0.05] = 0
    isurf[(lsm > 0.05) & (lsm < 0.5)] = 1
    isurf[(lsm >= 0.5) & (lsm < 0.95)] = 2
    isurf[lsm >= 0.95] = 3
    if (isurf < 0).any():
        raise SystemExit("surface class partition is not exhaustive")
    ielev = _np.full((h, w), 0, dtype=_np.int32)
    ielev[(oro_m > 1000.0) & (oro_m <= 2000.0)] = 1
    ielev[oro_m > 2000.0] = 2
    return ((ilat * len(SURF_CLASSES) + isurf) * len(ELEV_CLASSES)
            + ielev).astype(_np.int32)


def class_members(region: str):

    surf, elev, latb = REGION_FROM_CLASSES[region]
    out = []
    for il, lb in enumerate(LAT_BANDS):
        if latb is not None and lb not in latb:
            continue
        for isf, sc in enumerate(SURF_CLASSES):
            if surf is not None and sc not in surf:
                continue
            for ie, ec in enumerate(ELEV_CLASSES):
                if elev is not None and ec not in elev:
                    continue
                out.append((il * len(SURF_CLASSES) + isf) * len(ELEV_CLASSES) + ie)
    return out


TAIL_SPEC = {
    "absr":     (0.0, 30.0, 3000),
    "anom":     (0.0, 30.0, 3000),
    "gradmag":  (0.0, 12.0, 2400),
    "lapmag":   (0.0, 30.0, 3000),
    "lstd1":    (0.0, 12.0, 2400),
    "lstd3":    (0.0, 12.0, 2400),
    "lstd6":    (0.0, 12.0, 2400),
    "absout":   (0.0, 30.0, 3000),
}


SUPPORT_QUANTITIES = ["w", "B", "B2", "T", "T2", "r", "r2", "absr",
                      "gradmag", "lapmag", "lvar1", "lvar3", "lvar6",
                      "o2", "rhat2", "anom2", "tisr", "tisr2", "tisr_zero"]
