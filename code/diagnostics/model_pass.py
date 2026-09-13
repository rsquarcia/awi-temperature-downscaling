"""Compute per-day orthogonal error components from retained model predictions.
Projects target, CCA and U-Net residuals into the fixed 512-mode target EOF
subspace, separates in-span and out-of-span errors and accumulates regional,
seasonal, pixelwise and spectral partials. Defines the shared MODEL_QUANTITIES
ordering used by decomposition_exact.py. Requires external predictions,
targets and the frozen EOF basis; no model is fitted or inferred here."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import decomposition_common as C

N_PRED_SHARDS = 20


MODEL_QUANTITIES = [
    "w",                             
    "r2",                                                              
    "e_cca2", "e_unet2",
    "Pe_cca2", "Qe_cca2", "PQe_cca",                                    
    "Pe_unet2", "Qe_unet2", "PQe_unet",                                   
    "o_true2",                                            
    "o_unet2",                                                           
    "o_cross",                                                       
]


ANNUAL_FIELDS = ["T_sq", "B_sq", "r_sum", "r_sq", "e_cca_sq", "e_unet_sq",
                 "o_true_sq", "o_unet_sq", "Pe_unet_sq", "Pe_cca_sq"]
SEASON_FIELDS = ["T_sum", "B_sum", "r_sum", "r_sq"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def pred_shards(pop: str):
    root = C.POP_PRED_ROOT[pop]
    out = []
    for s in range(N_PRED_SHARDS):
        j = json.loads((root / "predictions" / "b2" / f"shard_{s:02d}"
                        / "shard_done.json").read_text())
        out.append({"shard": s, "lo": int(j["lo"]), "hi": int(j["hi"]),
                    "dates": [str(x) for x in j["dates"]]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", required=True,
                    choices=list(C.MODEL_POPULATIONS))
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--n-tasks", type=int, required=True)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()

    import zarr

    pop = args.population
    root = Path(args.out_root)
    part_dir = root / "partials"
    part_dir.mkdir(parents=True, exist_ok=True)
    out_path = part_dir / f"model_{pop}_{args.task:02d}.npz"

    H, W = C.HEIGHT, C.WIDTH
    npix = H * W
    TS, TM = C.TARGET_STD_K, C.TARGET_MEAN_K
    IS, IM = C.NORM_T2M_INP_STD, C.NORM_T2M_INP_MEAN
    scale_b = IS / TS
    offset_b = (IM - TM) / TS

    lat = C.grid_lat_centers(H)
    lsm, oro_m, lat_file, lon_file = C.load_statics()
    w_row, sqrt_w_row = C.area_weights_from_lat(lat)
    w_flat64 = np.ascontiguousarray(
        np.broadcast_to(w_row[:, None], (H, W))).ravel()
    sqrt_w_flat = np.repeat(sqrt_w_row, W).astype(np.float32)
    masks = C.region_masks(lat_file, lsm, oro_m)
    wm = np.stack([np.asarray(masks[r], dtype=np.float64).ravel() * w_flat64
                   for r in C.ALL_REGIONS])
    log(f"masks built, global weight {wm[0].sum():.6f}")


    t0 = time.time()
    y_mean = np.asarray(C.npz_small_member(C.Y_BASIS, "mean_field.npy"),
                        dtype=np.float64).reshape(-1)
    mm = C.npz_member_memmap(C.Y_BASIS, "eofs_weighted.npy")
    if mm.shape[0] < C.KY or int(np.prod(mm.shape[1:])) != npix:
        raise SystemExit(f"unexpected y basis shape {mm.shape}")
    E = np.empty((C.KY, npix), dtype=np.float64)
    for k0 in range(0, C.KY, 32):
        k1 = min(k0 + 32, C.KY)
        E[k0:k1] = np.asarray(mm[k0:k1], dtype=np.float64).reshape(k1 - k0, npix)
    del mm
    sqrt_w_flat64 = sqrt_w_flat.astype(np.float64)
    log(f"y basis {E.shape} {E.dtype} loaded in {time.time() - t0:.0f} s")


    shards = pred_shards(pop)
    lo_s, hi_s = C.shard_slices(N_PRED_SHARDS, args.n_tasks)[args.task]
    mine = shards[lo_s:hi_s]
    n = sum(s["hi"] - s["lo"] for s in mine)
    day0 = mine[0]["lo"]
    log(f"{pop} task {args.task}: prediction shards {lo_s}..{hi_s-1}, "
        f"days {day0}..{mine[-1]['hi']-1} ({n} days)")

    group = zarr.open_group(str(C.POP_ZARR[pop]), mode="r")
    zdates = np.asarray(group["dates"][:]).astype("U10")
    pred_root = C.POP_PRED_ROOT[pop] / "predictions"

    daily = np.zeros((n, len(C.ALL_REGIONS), len(MODEL_QUANTITIES)))
    s_true = np.zeros((n, C.KY)); s_unet = np.zeros((n, C.KY))
    s_cca = np.zeros((n, C.KY))
    zon_out_unet = np.zeros((n, C.K_FULL + 1))
    zon_out_true = np.zeros((n, C.K_FULL + 1))
    gate_baseline = np.zeros(n)
    gate_cca_outside = np.zeros(n)
    gate_pythag_unet = np.zeros(n)
    dates = np.empty(n, dtype="U10")
    seasons = np.zeros(n, dtype=np.int64)
    yidx = np.zeros(n, dtype=np.int64)
    is_feb29 = np.zeros(n, dtype=bool)
    day_index = np.zeros(n, dtype=np.int64)

    years_all = C.POP_YEARS[pop]
    ann = {k: np.zeros((H, W)) for k in ANNUAL_FIELDS}
    sea = {k: np.zeros((4, H, W)) for k in SEASON_FIELDS}
    sea_days = np.zeros(4, dtype=np.int64)
    X = np.empty((len(MODEL_QUANTITIES), npix)); X[0] = 1.0

    g = 0
    t0 = time.time()
    for sh in mine:
        sd = sh["shard"]
        b2p = np.load(pred_root / "b2" / f"shard_{sd:02d}" /
                      "b2_prediction_norm.npy", mmap_mode="r")
        b2b = np.load(pred_root / "b2" / f"shard_{sd:02d}" /
                      "b2_baseline_norm.npy", mmap_mode="r")
        ccp = np.load(pred_root / "cca" / f"shard_{sd:02d}" /
                      "cca_prediction_norm.npy", mmap_mode="r")
        nd = sh["hi"] - sh["lo"]
        if b2p.shape[0] != nd or ccp.shape[0] != nd:
            raise SystemExit(f"{pop} shard {sd}: retained day count mismatch")
        for k in range(nd):
            d = sh["lo"] + k
            if str(zdates[d]) != sh["dates"][k]:
                raise SystemExit(f"{pop} shard {sd} day {k}: "
                                 f"{zdates[d]} != {sh['dates'][k]}")
            inp = np.asarray(group["inputs"][d, 0], dtype=np.float32).reshape(-1)
            tgt = np.asarray(group["targets"][d, 0],
                             dtype=np.float32).reshape(-1)
            base32 = inp * np.float32(scale_b) + np.float32(offset_b)
            ub = np.asarray(b2b[k], dtype=np.float32).reshape(-1)
            gate_baseline[g] = float(np.max(np.abs(ub - base32)))


            base = ub.astype(np.float64)
            tgt64 = tgt.astype(np.float64)
            r_n = tgt64 - base
            up = np.asarray(b2p[k], dtype=np.float64).reshape(-1)
            cp = np.asarray(ccp[k], dtype=np.float64).reshape(-1)
            ru_n = up - base
            rc_n = cp - base

            a_t = (r_n - y_mean) * sqrt_w_flat64
            a_u = (ru_n - y_mean) * sqrt_w_flat64
            a_c = (rc_n - y_mean) * sqrt_w_flat64
            st = a_t @ E.T; su = a_u @ E.T; sc = a_c @ E.T
            o_t = a_t - st @ E
            o_u = a_u - su @ E
            o_c = a_c - sc @ E
            s_true[g] = st; s_unet[g] = su; s_cca[g] = sc
            gate_cca_outside[g] = float(
                np.einsum("i,i->", o_c, o_c)
                / max(float(np.einsum("i,i->", a_c, a_c)), 1e-30))


            Pe_u = (((su - st) @ E) / sqrt_w_flat64) * TS
            Qe_u = ((o_u - o_t) / sqrt_w_flat64) * TS
            Pe_c = (((sc - st) @ E) / sqrt_w_flat64) * TS
            Qe_c = ((o_c - o_t) / sqrt_w_flat64) * TS
            o_tK = (o_t / sqrt_w_flat64) * TS
            o_uK = (o_u / sqrt_w_flat64) * TS
            e_u = Pe_u + Qe_u
            e_c = Pe_c + Qe_c
            rK = r_n * TS
            TK = tgt64 * TS + TM
            BK = base * TS + TM

            X[1] = rK * rK
            X[2] = e_c * e_c; X[3] = e_u * e_u
            X[4] = Pe_c * Pe_c; X[5] = Qe_c * Qe_c; X[6] = Pe_c * Qe_c
            X[7] = Pe_u * Pe_u; X[8] = Qe_u * Qe_u; X[9] = Pe_u * Qe_u
            X[10] = o_tK * o_tK; X[11] = o_uK * o_uK; X[12] = o_uK * o_tK
            daily[g] = wm @ X.T

            tot = float((X[3] * w_flat64).sum())
            par = float(((X[7] + X[8]) * w_flat64).sum())
            gate_pythag_unet[g] = abs(tot - par) / max(tot, 1e-30)

            s = C.SEASONS.index(C.season_name(str(zdates[d])))
            dates[g] = str(zdates[d]); seasons[g] = s
            yidx[g] = years_all.index(int(str(zdates[d])[:4]))
            is_feb29[g] = str(zdates[d])[5:] == "02-29"
            day_index[g] = d
            T2 = TK.reshape(H, W); B2 = BK.reshape(H, W); R2 = rK.reshape(H, W)
            ann["T_sq"] += T2 * T2; ann["B_sq"] += B2 * B2
            ann["r_sum"] += R2; ann["r_sq"] += R2 * R2
            ann["e_cca_sq"] += (e_c * e_c).reshape(H, W)
            ann["e_unet_sq"] += (e_u * e_u).reshape(H, W)
            ann["o_true_sq"] += (o_tK * o_tK).reshape(H, W)
            ann["o_unet_sq"] += (o_uK * o_uK).reshape(H, W)
            ann["Pe_unet_sq"] += (Pe_u * Pe_u).reshape(H, W)
            ann["Pe_cca_sq"] += (Pe_c * Pe_c).reshape(H, W)
            sea["T_sum"][s] += T2; sea["B_sum"][s] += B2
            sea["r_sum"][s] += R2; sea["r_sq"][s] += R2 * R2
            sea_days[s] += 1

            zon_out_unet[g] = C.zonal_row_weighted_power(
                (o_uK / TS).reshape(1, H, W), w_row, C.K_FULL)[0]
            zon_out_true[g] = C.zonal_row_weighted_power(
                (o_tK / TS).reshape(1, H, W), w_row, C.K_FULL)[0]
            g += 1
        del b2p, b2b, ccp
        el = time.time() - t0
        log(f"  {pop} task {args.task}: shard {sd} done, {g}/{n} days "
            f"({el:.0f} s, {el/max(g,1):.2f} s/day)")

    if g != n:
        raise SystemExit(f"day count mismatch {g} != {n}")

    payload = {
        "population": np.str_(pop), "task": np.int64(args.task),
        "n_tasks": np.int64(args.n_tasks), "n_days": np.int64(n),
        "lo": np.int64(day0), "hi": np.int64(mine[-1]["hi"]),
        "dates": dates, "day_index": day_index, "season_index": seasons,
        "year_index_global": yidx, "is_feb29": is_feb29,
        "regions": np.asarray(C.ALL_REGIONS),
        "quantities": np.asarray(MODEL_QUANTITIES),
        "daily": daily, "s_true": s_true, "s_unet": s_unet, "s_cca": s_cca,
        "zonal_out_unet": zon_out_unet, "zonal_out_true": zon_out_true,
        "zonal_row_weight_sum": np.float64(float(w_row.sum())),
        "gate_baseline_max_abs_dev": gate_baseline,
        "gate_cca_outside_energy_fraction": gate_cca_outside,
        "gate_unet_pythagorean_rel_dev": gate_pythag_unet,
        "season_days": sea_days,
        "years_local": np.asarray(sorted(set(int(d[:4]) for d in dates))),
        "target_std_K": np.float64(TS),
    }
    for k in ANNUAL_FIELDS:
        payload[f"ann_{k}"] = ann[k].astype(np.float32)
    for k in SEASON_FIELDS:
        payload[f"sea_{k}"] = sea[k].astype(np.float32)
    tmp = out_path.with_suffix(".writing.npz")
    np.savez(tmp, **payload)
    tmp.replace(out_path)
    log(f"wrote {out_path} ({out_path.stat().st_size/1e6:.0f} MB); "
        f"gates: baseline max dev {gate_baseline.max():.3e}, "
        f"cca outside frac max {gate_cca_outside.max():.3e}, "
        f"unet Pythagorean max rel dev {gate_pythag_unet.max():.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
