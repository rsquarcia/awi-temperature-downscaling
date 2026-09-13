"""Shared loaders and statistics for support and climate-shift diagnostics.
Loads saved population partials and predictor/target EOF scores, and provides
calendar masks, weighted map statistics, covariance whitening, neighbour
distances and distribution comparisons, together with output helpers.
Its constants come from decomposition_common.py. Importing this module does
not read the external score matrices or run a diagnostic analysis."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import decomposition_common as C

H, W = C.HEIGHT, C.WIDTH
NPIX = H * W
LAT = C.grid_lat_centers(H)
W_ROW, SQRT_W_ROW = C.area_weights_from_lat(LAT)
W2D = np.ascontiguousarray(np.broadcast_to(W_ROW[:, None], (H, W)))
W_SUM = float(W_ROW.sum() * W)

POP_SHARDS = {"train": 24, "val": 3, "test": 3, "mh": 3, "ssp": 3}
MODEL_SHARDS = {"val": 8, "test": 8, "mh": 8, "ssp": 8}


PARTIAL_SEARCH = (None, C.MH_SUPPORT_RUN)


def partial_path(root: Path, pop: str, shard: int) -> Path:
    name = f"map_{pop}_{shard:02d}.npz"
    for base in PARTIAL_SEARCH:
        p = (root if base is None else base) / "partials" / name
        if p.exists():
            return p
    raise SystemExit(f"missing partial {name} in {root} or {C.MH_SUPPORT_RUN}")


PROJ = C.CCA_HEAVY / "projections"
PROJ_DURABLE = Path(C.RESULTS_ROOT) / "cca_final_run" / "preserved_reanalysis_cache" / "projections"
FROZEN_X = {"train": "train_x_scores_maxK10592.npy",
            "val": "val_x_scores_maxK10592.npy"}
FROZEN_Y = {"train": "train_y_scores_maxK1024.npy",
            "val": "val_y_scores_maxK1024.npy"}

YR_KEYS = ("T_sum", "T_sq", "B_sum", "r_sum", "r_sq", "o_sq")


CLIM_OF_POP = {"test": "pd", "mh": "mh", "ssp": "ssp"}


def frozen_x_scores(pop: str, kmax: int) -> np.ndarray:

    if pop in CLIM_OF_POP:
        p = C.CLIMATE_SCORES[CLIM_OF_POP[pop]]
    else:
        p = PROJ / FROZEN_X[pop]
        if not p.exists():
            p = PROJ_DURABLE / FROZEN_X[pop]
    a = np.load(p, mmap_mode="r")
    return np.asarray(a[:, :kmax], dtype=np.float64), str(p)


def frozen_y_scores(pop: str, kmax: int = C.KY):

    if pop not in FROZEN_Y:
        return None, None
    p = PROJ / FROZEN_Y[pop]
    a = np.load(p, mmap_mode="r")
    return np.asarray(a[:, :kmax], dtype=np.float64), str(p)


def load_pop(root: Path, pop: str) -> dict:
    n_sh = POP_SHARDS[pop]
    parts = []
    for s in range(n_sh):
        parts.append(np.load(partial_path(root, pop, s), allow_pickle=False))
    parts = [parts[i] for i in np.argsort([int(z["lo"]) for z in parts])]
    out = {"population": pop}
    for k in ("dates", "day_index", "season_index", "year_index_global",
              "is_feb29"):
        out[k] = np.concatenate([z[k] for z in parts])
    out["dates"] = out["dates"].astype("U10")
    if out["dates"].size != C.POP_NDAYS[pop]:
        raise SystemExit(f"{pop}: {out['dates'].size} days combined")
    if not np.array_equal(out["day_index"], np.arange(C.POP_NDAYS[pop])):
        raise SystemExit(f"{pop}: incomplete day cover")
    for k in ("daily", "x_scores", "y_scores", "c_ell_norm2",
              "zonal_power_norm2", "tisr_zonal_mean", "bilinear_t2m_zonal_mean"):
        out[k] = np.concatenate([z[k] for z in parts], axis=0)
    for k in ("p_tot_weighted", "p_in_weighted"):
        out[k] = np.concatenate([z[k] for z in parts])
    for k in C.TAIL_SPEC:
        out[f"tail_{k}"] = sum(z[f"tailhist_{k}"] for z in parts)
    out["regions"] = list(parts[0]["regions"].astype(str))
    out["quantities"] = list(parts[0]["quantities"].astype(str))
    out["zonal_row_weight_sum"] = float(parts[0]["zonal_row_weight_sum"])
    years = C.POP_YEARS[pop]
    out["years"] = years
    out["year_days"] = np.zeros(len(years), dtype=np.int64)
    for k in YR_KEYS:
        out[f"year_{k}"] = np.zeros((len(years), H, W))
    for z in parts:
        loc = list(z["years_local"])
        gidx = [years.index(int(y)) for y in loc]
        out["year_days"][gidx] += z["year_days"]
        for k in YR_KEYS:
            out[f"year_{k}"][gidx] += z[f"year_{k}"]
    out["gates"] = {k[5:]: float(z[k]) for z in parts for k in z.files
                    if k.startswith("gate_")}
    for z in parts:
        z.close()
    return out


def qidx(d):
    return {q: i for i, q in enumerate(d["quantities"])}


def ridx(d):
    return {r: i for i, r in enumerate(d["regions"])}


def no_leap_mask(d) -> np.ndarray:

    return ~d["is_feb29"]


def rolling_blocks(years, length=3):
    return [tuple(years[i:i + length])
            for i in range(len(years) - length + 1)]


def nonoverlapping_blocks(years, length=3, offset=0):
    ys = years[offset:]
    return [tuple(ys[i:i + length]) for i in range(0, len(ys) - length + 1, length)]


def wmean(f):
    return float((np.asarray(f, dtype=np.float64) * W2D).sum() / W_SUM)


def wmeansq(f):
    a = np.asarray(f, dtype=np.float64)
    return float((a * a * W2D).sum() / W_SUM)


def wcorr_unc(a, b):
    na, nb = wmeansq(a), wmeansq(b)
    return float(((np.asarray(a) * np.asarray(b) * W2D).sum() / W_SUM)
                 / math.sqrt(na * nb)) if na > 0 and nb > 0 else float("nan")


class ShrinkageWhitener:


    def __init__(self, Xtr: np.ndarray, shrinkage: float | None = None):
        self.mean = Xtr.mean(axis=0)
        Z = Xtr - self.mean
        n, p = Z.shape
        S = (Z.T @ Z) / (n - 1)
        mu = float(np.trace(S) / p)
        if shrinkage is None:


            d2 = float(((S - mu * np.eye(p)) ** 2).sum() / p)
            sf2 = float((S * S).sum())
            q4 = float((np.einsum("ij,ij->i", Z, Z) ** 2).sum())
            b_bar = max((q4 - (n - 2) * sf2) / (n * n * p), 0.0)
            shrinkage = float(min(1.0, max(0.0, b_bar / d2))) if d2 > 0 else 1.0
        self.shrinkage = float(shrinkage)
        Sh = (1.0 - self.shrinkage) * S + self.shrinkage * mu * np.eye(p)
        ev, V = np.linalg.eigh(Sh)
        ev = np.maximum(ev, 1e-12 * ev.max())
        self.Wmat = V @ np.diag(ev ** -0.5) @ V.T
        self.sd = np.sqrt(np.maximum(np.diag(S), 1e-30))
        self.eig_min, self.eig_max = float(ev.min()), float(ev.max())

    def maha(self, X):
        Z = (np.asarray(X, dtype=np.float64) - self.mean) @ self.Wmat
        return np.sqrt(np.einsum("ij,ij->i", Z, Z))

    def diag(self, X):
        Z = (np.asarray(X, dtype=np.float64) - self.mean) / self.sd
        return np.sqrt(np.einsum("ij,ij->i", Z, Z))


def knn_distance(Q: np.ndarray, R: np.ndarray, ks=(1, 5, 20),
                 block=512, exclude_self=False):

    out = {k: np.empty(Q.shape[0]) for k in ks}
    r2 = np.einsum("ij,ij->i", R, R)
    kmax = max(ks) + (1 if exclude_self else 0)
    for i in range(0, Q.shape[0], block):
        q = Q[i:i + block]
        d2 = (np.einsum("ij,ij->i", q, q)[:, None] + r2[None, :]
              - 2.0 * (q @ R.T))
        np.maximum(d2, 0.0, out=d2)
        part = np.partition(d2, kmax - 1, axis=1)[:, :kmax]
        part.sort(axis=1)
        if exclude_self:
            part = part[:, 1:]
        for k in ks:
            out[k][i:i + block] = np.sqrt(part[:, k - 1])
    return out


def energy_and_mmd(x, y, perm=None):
    n, m = x.shape[0], y.shape[0]
    Z = np.vstack([x, y])
    sq = np.einsum("ij,ij->i", Z, Z)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T), 0.0)
    D = np.sqrt(d2)
    iu = np.triu_indices(n + m, k=1)
    sigma = float(np.median(D[iu]))
    K = np.exp(-d2 / (2.0 * sigma * sigma))
    u = np.concatenate([np.ones(n), np.zeros(m)])
    v = 1.0 - u

    def stat(M):
        return (float(u @ M @ v) / (n * m),
                float(u @ M @ u) / (n * n), float(v @ M @ v) / (m * m))
    exy, exx, eyy = stat(D)
    kxy, kxx, kyy = stat(K)
    res = {"energy_distance": 2 * exy - exx - eyy,
           "mmd2_rbf": kxx + kyy - 2 * kxy, "bandwidth": sigma,
           "n_x": n, "n_y": m}
    if perm is not None and n == m:
        U = np.vstack([np.hstack([1.0 - perm, perm]).astype(np.float64)])
        V = 1.0 - U
        eD = ((np.einsum("ij,ij->i", U @ D, V) / (n * m))* 2
              - np.einsum("ij,ij->i", U @ D, U) / (n * n)
              - np.einsum("ij,ij->i", V @ D, V) / (m * m))
        eK = (np.einsum("ij,ij->i", U @ K, U) / (n * n)
              + np.einsum("ij,ij->i", V @ K, V) / (m * m)
              - 2 * np.einsum("ij,ij->i", U @ K, V) / (n * m))
        res["energy_block_perm_p"] = float((1 + (eD >= res["energy_distance"]).sum())
                                           / (1 + eD.size))
        res["mmd2_block_perm_p"] = float((1 + (eK >= res["mmd2_rbf"]).sum())
                                         / (1 + eK.size))
    return res


def percentile_of(value, sample):
    s = np.asarray(sample, dtype=np.float64)
    s = s[np.isfinite(s)]
    return float(100.0 * (s < value).mean() + 50.0 / max(s.size, 1)
                 * (s == value).mean() * 2)


def robust_z(value, sample):
    s = np.asarray(sample, dtype=np.float64)
    s = s[np.isfinite(s)]
    med = float(np.median(s))
    mad = float(np.median(np.abs(s - med)))
    scale = 1.4826 * mad
    return float((value - med) / scale) if scale > 0 else float("nan")


def block_percentile_entry(name, value, sample, nonoverlap=None):
    s = np.asarray(sample, dtype=np.float64)
    e = {"metric": name, "value": float(value),
         "block_median": float(np.median(s)),
         "block_p5": float(np.percentile(s, 5)),
         "block_p95": float(np.percentile(s, 95)),
         "block_min": float(s.min()), "block_max": float(s.max()),
         "n_blocks": int(s.size),
         "rank": int((s < value).sum()) + 1,
         "percentile": percentile_of(value, s),
         "robust_z": robust_z(value, s),
         "beyond_block_range": bool(value < s.min() or value > s.max())}
    if nonoverlap is not None and len(nonoverlap):
        t = np.asarray(nonoverlap, dtype=np.float64)
        e.update({"nonoverlap_median": float(np.median(t)),
                  "nonoverlap_min": float(t.min()),
                  "nonoverlap_max": float(t.max()),
                  "nonoverlap_n": int(t.size),
                  "nonoverlap_percentile": percentile_of(value, t),
                  "nonoverlap_robust_z": robust_z(value, t),
                  "nonoverlap_beyond_range": bool(value < t.min() or value > t.max())})
    return e


def write_csv(path: Path, rows, header=None):
    if not rows:
        path.write_text((",".join(header) if header else "") + "\n")
        return
    keys = header or list(rows[0].keys())
    with open(path, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(_fmt(r.get(k)) for k in keys) + "\n")


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.10g}"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return str(v).replace(",", ";")


def jdump(path: Path, obj):
    path.write_text(json.dumps(obj, indent=1, default=float))
