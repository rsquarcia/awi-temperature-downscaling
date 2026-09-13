"""Shared settings for the frozen-model uniform-temperature-shift experiment.
Defines the present-day test population, displacement grid, normalization,
checkpoint locations, zero-shift checks and temporal-bootstrap parameters.
Imports common loaders and statistics and provides day-sharding and RMSE
helpers used by the CCA sweep, U-Net sweep and final reducer.
External model artifacts and retained diagnostic arrays are required to run them."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import decomposition_common as C
import sup_core as S

import os


MODEL_ROOT = os.environ.get("MODEL_ROOT", "model_root")                                                   
RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "results_root")                                                  


POP = "test"                                                             
N_DAYS = C.POP_NDAYS[POP]
DELTAS = np.round(np.arange(-6.0, 6.0 + 1e-9, 0.5), 10)
I_ZERO = int(np.argmin(np.abs(DELTAS)))


ACCEPTED_CCA_RMSE_K = 0.2953558449
ACCEPTED_UNET_RMSE_K = 0.1985303193
GATE_TOL_K = 5e-6


BLOCK_LEN = 60
N_RESAMPLES = 10000
SEED = 20260715
CI_LO_PCT, CI_HI_PCT = 2.5, 97.5


MH_D0_K = -2.1432
SSP_D0_K = +4.5880


H, W = C.HEIGHT, C.WIDTH
NPIX = H * W
TS_K = C.TARGET_STD_K
TM_K = C.TARGET_MEAN_K
IS_K = C.NORM_T2M_INP_STD
IM_K = C.NORM_T2M_INP_MEAN
KX, KY = C.KX, C.KY


MECH_RUN = Path(f"{RESULTS_ROOT}/"
                "ssp585_pd_mh_full_mechanism_20260802T095350Z")
KEEPER = Path(f"{MODEL_ROOT}")
CKPT = KEEPER / "checkpoint" / "best_val_mse.pt"
CKPT_SHA = "903fab9a5c1e47299a87215785de658aa31de5594d60e3ae3c30116c8689aaa8"
PUBLIC_UNET_ROOT = Path(__file__).resolve().parents[1] / "unet"
TRAINING_CODE_TREE_SHA = "f8b646bbd6d0e0406580503e781926747db1241ee6fcb2bcb4ef1498888e5fdb"
NORM_STATS = Path(os.environ.get(
    "NORM_STATS_PATH",
    str(Path(__file__).resolve().parents[2] / "configs/unet/norm_stats.json")))
NORM_STATS_SHA = "d372733c0e513198b7a673ced2d5ef00a59f46406563177bb47b7d45dd0c112b"
CKPT_STEP = 24548


ACCEPTED_UNIFORM_1K_PRED_RMS_K = 0.03344068906


def shard_days(n_shards: int, shard: int) -> np.ndarray:

    edges = np.linspace(0, N_DAYS, n_shards + 1).round().astype(int)
    return np.arange(edges[shard], edges[shard + 1])


def circular_block_bootstrap_indices(n_days, block_len, n_resamples, rng):


    import math
    if block_len < 1 or block_len > n_days:
        raise SystemExit("block length must be in [1, n_days]")
    n_blocks = math.ceil(n_days / block_len)
    starts = rng.integers(0, n_days, size=(n_resamples, n_blocks), dtype=np.int64)
    idx = (starts[:, :, None]
           + np.arange(block_len, dtype=np.int64)[None, None, :]) % n_days
    return idx.reshape(n_resamples, n_blocks * block_len)[:, :n_days]


def rmse_from_day_mse(day_mse: np.ndarray, axis=-1) -> np.ndarray:


    return np.sqrt(np.mean(day_mse, axis=axis))
