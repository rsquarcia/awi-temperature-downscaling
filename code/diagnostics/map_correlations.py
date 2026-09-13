"""Compare present-day and transfer-climate cellwise RMSE patterns.
Reads retained squared-error arrays and computes spatial correlations, weighted
correlations, transfer-cell comparisons and bootstrap summaries.
Writes the resulting diagnostic record without fitting or running models.
This is a top-level analysis script: its external inputs are read when the
file is executed or imported, not through a separate main function."""

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

import os


RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "results_root")                                                  


W = Path(f"{RESULTS_ROOT}/AWI_numeric_audit/work/recompute")
OUT = Path(f"{RESULTS_ROOT}/AWI_numeric_audit/work")
CL = ["pd", "mh", "ssp585"]
D = {c: np.load(W / f"recompute_{c}.npz", allow_pickle=True) for c in CL}
REG = list(D["pd"]["regions"])
PER = list(D["pd"]["periods"])
MET = list(D["pd"]["methods"])
res = {}


print("=" * 100)
print("CELLWISE RMSE MAPS - Pearson / Spearman correlation with the present-day map")
print("=" * 100)
lat = D["pd"]["lat"]
w = np.cos(np.deg2rad(lat))
w = w / w.mean()
maps = {}
for c in CL:
    d = D[c]
    n = int(d["n_days"])
    maps[c] = np.sqrt(d["cell_sse"] / n)                                                 
W2D = np.broadcast_to(w[:, None], maps["pd"].shape[1:]).ravel()


def wpearson(x, y, wt):
    mx = np.average(x, weights=wt)
    my = np.average(y, weights=wt)
    cx, cy = x - mx, y - my
    return float(np.average(cx * cy, weights=wt) /
                 math.sqrt(np.average(cx * cx, weights=wt) * np.average(cy * cy, weights=wt)))


ui = MET.index("unet")
res["map_correlations"] = {}
for c in ("mh", "ssp585"):
    a = maps["pd"][ui].ravel()
    b = maps[c][ui].ravel()
    pr = float(np.corrcoef(a, b)[0, 1])
    sp = float(stats.spearmanr(a, b).statistic)
    wp = wpearson(a, b, W2D)
    ra, rb = stats.rankdata(a), stats.rankdata(b)
    wsp = wpearson(ra, rb, W2D)
    res["map_correlations"][c] = dict(pearson_unweighted=pr, spearman_unweighted=sp,
                                      pearson_area_weighted=wp, spearman_area_weighted=wsp)
    print(f"  U-Net RMSE map  PD vs {c:7s}:  Pearson  unweighted={pr:.6f}  area-weighted={wp:.6f}")
    print(f"                                Spearman unweighted={sp:.6f}  area-weighted={wsp:.6f}")


print()
print("=" * 100)
print("REGION-PERIOD TRANSFER CELLS WITH NEGATIVE dS")
print("=" * 100)
skpd = np.load(OUT / "pd_region_period_skill.npy")
res["negative_transfer_cells"] = {}
for c in ("mh", "ssp585"):
    sk = np.load(OUT / f"{c}_region_period_skill.npy")
    ds = sk - skpd
    neg = [(REG[i], PER[j], float(ds[i, j])) for i in range(len(REG)) for j in range(len(PER))
           if ds[i, j] <= 0]
    res["negative_transfer_cells"][c] = neg
    print(f"  {c}: {len(neg)} non-positive cells")
    for r, p, v in sorted(neg, key=lambda t: t[2]):
        print(f"     {r:20s} {p:6s} dS = {v:+.4f}")


print()
print("=" * 100)
print("PAIRED CIRCULAR MOVING-BLOCK BOOTSTRAP  (60-day blocks, 10000 resamples, seed 20260715)")
print("=" * 100)


def circular_block_indices(n, block, n_rep, seed):
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block))
    starts = rng.integers(0, n, size=(n_rep, n_blocks))
    off = np.arange(block)
    idx = (starts[:, :, None] + off[None, None, :]) % n
    return idx.reshape(n_rep, -1)[:, :n]


def ci_from(sse, wden, idx, q=(2.5, 97.5)):

    num = sse[:, idx]
    den = wden[idx]
    mse = num.sum(axis=2) / den.sum(axis=1)
    return mse


res["bootstrap"] = {}
for c in CL:
    d = D[c]
    n = int(d["n_days"])
    sse = d["daily_sse"]
    wden = d["daily_wden"]
    out = {}
    for block in (60, 30, 45, 90):
        idx = circular_block_indices(n, block, 10000, 20260715)
        mse = ci_from(sse, wden, idx)
        rm = np.sqrt(mse)
        bi, cci, uui = 0, 1, 2
        stat = {
            "bilinear_rmse_K": rm[bi], "cca_rmse_K": rm[cci], "unet_rmse_K": rm[uui],
            "delta_rmse_K": rm[cci] - rm[uui],
            "cca_skill_rmse_pct": 100 * (1 - rm[cci] / rm[bi]),
            "unet_skill_rmse_pct": 100 * (1 - rm[uui] / rm[bi]),
            "skill_unet_vs_cca_pct": 100 * (1 - rm[uui] / rm[cci]),
        }
        out[f"block{block}"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                               for k, v in stat.items()}

    dates = [str(x) for x in d["dates"]]
    doy = np.array([int(x[5:7]) for x in dates])
    rng = np.random.default_rng(20260715)

    idx_sa = np.empty((10000, n), dtype=np.int64)
    n_blocks = int(math.ceil(n / 60))
    for rr in range(10000):
        starts = rng.integers(0, n, size=n_blocks)

        tgt_months = doy[(np.arange(n_blocks) * 60) % n]
        for b in range(n_blocks):
            cand = np.flatnonzero(doy == tgt_months[b])
            starts[b] = cand[rng.integers(0, len(cand))]
        idx_sa[rr] = ((starts[:, None] + np.arange(60)[None, :]) % n).reshape(-1)[:n]
        if rr >= 999:
            break
    idx_sa = idx_sa[:1000]
    mse = ci_from(sse, wden, idx_sa)
    rm = np.sqrt(mse)
    out["season_aligned_block60_R1000"] = {
        "bilinear_rmse_K": [float(np.percentile(rm[0], 2.5)), float(np.percentile(rm[0], 97.5))],
        "cca_rmse_K": [float(np.percentile(rm[1], 2.5)), float(np.percentile(rm[1], 97.5))],
        "unet_rmse_K": [float(np.percentile(rm[2], 2.5)), float(np.percentile(rm[2], 97.5))],
        "delta_rmse_K": [float(np.percentile(rm[1] - rm[2], 2.5)),
                         float(np.percentile(rm[1] - rm[2], 97.5))],
        "cca_skill_rmse_pct": [float(np.percentile(100 * (1 - rm[1] / rm[0]), 2.5)),
                               float(np.percentile(100 * (1 - rm[1] / rm[0]), 97.5))],
        "unet_skill_rmse_pct": [float(np.percentile(100 * (1 - rm[2] / rm[0]), 2.5)),
                                float(np.percentile(100 * (1 - rm[2] / rm[0]), 97.5))],
        "skill_unet_vs_cca_pct": [float(np.percentile(100 * (1 - rm[2] / rm[1]), 2.5)),
                                  float(np.percentile(100 * (1 - rm[2] / rm[1]), 97.5))],
    }
    res["bootstrap"][c] = out
    print(f"\n--- {c.upper()} ---")
    for scheme in out:
        o = out[scheme]
        print(f"  {scheme}")
        for k in ("bilinear_rmse_K", "cca_rmse_K", "unet_rmse_K", "delta_rmse_K",
                  "cca_skill_rmse_pct", "unet_skill_rmse_pct", "skill_unet_vs_cca_pct"):
            lo, hi = o[k]
            flag = "  EXCLUDES 0" if (lo > 0 or hi < 0) else "  INCLUDES 0"
            print(f"     {k:24s} [{lo:.4f}, {hi:.4f}]{flag if 'delta' in k or 'skill' in k else ''}")

(OUT / "maps_and_ci.json").write_text(json.dumps(res, indent=2, default=float))
print("\nwrote", OUT / "maps_and_ci.json")
