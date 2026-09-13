"""Estimate sensitivity of the reported errors to a prescribed XIOS grid artifact.
Fits a fixed spatial template to retained errors with one domain-wide amplitude
or separate latitude-row amplitudes, then reports the effect on headline metrics.
The row-wise adjustment is a same-population fitted upper-bound diagnostic,
not an independently validated correction or a retrained downscaler.
Requires the external template and error arrays; runs as a top-level analysis."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import os


DATA_ROOT = os.environ.get("DATA_ROOT", "data_root")                                                     
RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "results_root")                                                  


PAPER_ROOT = Path(f"{RESULTS_ROOT}/xios-audit-paper-impact")
STAGE_A = PAPER_ROOT / "analysis" / "stage_a"
OUT = PAPER_ROOT / "analysis" / "results"

CLIMATES = {
    "pd": (f"{RESULTS_ROOT}/paper_pipeline_pd_test_r2_20260728T130323Z",
           "full_pd_test_summary.json"),
    "mh": (f"{RESULTS_ROOT}/mh_full_run_20260728T180741Z",
           "full_mh_summary.json"),
    "ssp585": (f"{RESULTS_ROOT}/ssp585_full_run_20260729T130103Z",
               "full_ssp585_summary.json"),
}
COMB_NPZ = Path(f"{RESULTS_ROOT}/source_lineage_checks/"
                "source_only_comb_map_20260715T102013Z/"
                "source_only_comb_diagnostic.npz")
HR_STATICS = Path(f"{DATA_ROOT}/grids/static_masks/"
                  "surface_fractions_hr.nc")
ORO_NC = Path(f"{DATA_ROOT}/grids/oro_hr.nc")

REGIONS = ["global", "land", "ocean", "elevation_gt_1000m", "elevation_le_1000m",
           "tropics", "midlat_north", "midlat_south", "highlat_north",
           "highlat_south", "arctic_gt_80N", "antarctic_lt_80S"]
PERIODS = ["annual", "DJF", "MAM", "JJA", "SON"]
METHODS = ["bilinear", "cca", "unet"]
BELT_LAT = 48.796875
K_MAX = 437
BOOTSTRAP = {"block_length_days": 60, "n_resamples": 10000, "seed": 20260715,
             "ci_percentiles": [2.5, 97.5]}
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def grid_lat_centers(h):
    return -90.0 + (np.arange(h, dtype=np.float64) + 0.5) * (180.0 / h)


def area_weights_from_lat(lat):
    w = np.cos(np.deg2rad(np.asarray(lat, dtype=np.float64)))
    return w / w.mean()


def unet_pipeline_row_weights(lat_file):
    w = np.cos(np.deg2rad(np.asarray(lat_file, dtype=np.float64))).astype(np.float32)
    w /= w.mean()
    return np.asarray(w, dtype=np.float64)


def load_statics(h, w):
    import netCDF4
    with netCDF4.Dataset(str(HR_STATICS), "r") as ds:
        lsm = np.asarray(ds.variables["lsm"][:], dtype=np.float64)
        lat_file = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        lon_file = np.asarray(ds.variables["lon"][:], dtype=np.float64)
    with netCDF4.Dataset(str(ORO_NC), "r") as ds:
        oro = np.asarray(ds.variables["var129"][:], dtype=np.float64)
    return (lsm.reshape(h, w), oro.reshape(h, w), lat_file, lon_file)


def region_masks(lat_deg, lsm, oro_m):
    h, w = lsm.shape
    lat = np.broadcast_to(np.asarray(lat_deg, dtype=np.float64)[:, None], (h, w))
    return {
        "global": np.ones((h, w), dtype=bool),
        "land": lsm >= 0.5, "ocean": lsm < 0.5,
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


def one_sided_zonal_power(field, detrend_constant=True):
    v = np.asarray(field, dtype=np.float64)
    if detrend_constant:
        v = v - v.mean(axis=-1, keepdims=True)
    width = v.shape[-1]
    c = np.fft.rfft(v, axis=-1, norm="ortho")
    p = c.real * c.real + c.imag * c.imag
    if width % 2 == 0:
        p[..., 1:-1] *= 2.0
    else:
        p[..., 1:] *= 2.0
    return p


def zonal_row_weighted_power(fields, row_weights, k_max):
    p = one_sided_zonal_power(fields, detrend_constant=True)
    return np.einsum("frk,r->fk", p[..., : k_max + 1], row_weights, optimize=True)


def circular_block_bootstrap_indices(n_days, block_len, n_resamples, rng):
    n_blocks = math.ceil(n_days / block_len)
    starts = rng.integers(0, n_days, size=(n_resamples, n_blocks), dtype=np.int64)
    idx = (starts[:, :, None]
           + np.arange(block_len, dtype=np.int64)[None, None, :]) % n_days
    return idx.reshape(n_resamples, n_blocks * block_len)[:, :n_days]


def bootstrap_quantities(daily, idx, std):
    def rs(x):
        return x[idx].sum(axis=1)
    sw_c, sw_u = rs(daily["w_c"]), rs(daily["w_u"])
    mse_bil_c = rs(daily["bil_e2_c"]) / sw_c
    mse_cca = rs(daily["cca_e2"]) / sw_c
    mse_bil_u = rs(daily["bil_e2_u"]) / sw_u
    mse_unet = rs(daily["unet_e2"]) / sw_u
    rmse_cca = np.sqrt(mse_cca) * std
    rmse_unet = np.sqrt(mse_unet) * std
    return {"bilinear_rmse_K": np.sqrt(mse_bil_c) * std,
            "cca_rmse_K": rmse_cca, "unet_rmse_K": rmse_unet,
            "cca_skill_vs_bilinear": 1.0 - mse_cca / mse_bil_c,
            "unet_skill_vs_bilinear": 1.0 - mse_unet / mse_bil_u,
            "delta_rmse_K": rmse_cca - rmse_unet,
            "skill_vs_cca": 1.0 - mse_unet / mse_cca}


def fit_rigid(E, T, w2d, mask):

    m = mask
    num = float((w2d * E * T)[m].sum())
    den = float((w2d * T * T)[m].sum())
    W = float(w2d[m].sum())
    if den <= 0.0:
        return 0.0, 0.0, np.zeros_like(E)
    a = num / den
    return a, a * a * den / W, a * T


def fit_per_latitude(E, T, w2d, mask):


    Em = np.where(mask, E, 0.0)
    Tm = np.where(mask, T, 0.0)
    num = (Em * Tm).sum(axis=1)
    den = (Tm * Tm).sum(axis=1)
    a = np.zeros_like(num)
    good = den > 0.0
    a[good] = num[good] / den[good]
    W = float(w2d[mask].sum())
    wrow = w2d[:, 0]
    aligned = float((wrow * a * a * den).sum() / W)
    return a, aligned, a[:, None] * Tm


def weighted_mse(E, w2d, mask):
    return float((w2d * E * E)[mask].sum() / w2d[mask].sum())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    gates = []

    def gate(name, ok, observed, expected=None, extra=None):
        gates.append({"name": name, "pass": bool(ok), "observed": observed,
                      "expected": expected, "extra": extra})
        if not ok:
            log(f"GATE FAIL {name}: {observed} vs {expected}")

    with np.load(COMB_NPZ) as z:
        T_K = 0.5 * (np.asarray(z["1980_comb_map_K"], dtype=np.float64)
                     + np.asarray(z["2009_comb_map_K"], dtype=np.float64))
        C_K = 0.5 * (np.asarray(z["1980_control_map_K"], dtype=np.float64)
                     + np.asarray(z["2009_control_map_K"], dtype=np.float64))
        comb_k_1980 = np.asarray(z["1980_chosen_k"], dtype=np.int64)
        comb_k_2009 = np.asarray(z["2009_chosen_k"], dtype=np.int64)
        comb_valid = (np.asarray(z["1980_valid"], dtype=bool)
                      & np.asarray(z["2009_valid"], dtype=bool))
    H, W = T_K.shape
    gate("template_rows_all_valid", bool(comb_valid.all()), int(comb_valid.sum()), H)
    gate("template_zero_row_means", float(np.abs(T_K.mean(axis=1)).max()) < 1e-8,
         float(np.abs(T_K.mean(axis=1)).max()), "<1e-8")

    lsm, oro_m, lat_file, lon_file = load_statics(H, W)
    lat_c = grid_lat_centers(H)
    masks = region_masks(lat_file, lsm, oro_m)
    belt_rows = np.abs(lat_c) <= BELT_LAT
    masks["belt_le_48.796875"] = np.broadcast_to(belt_rows[:, None], (H, W)).copy()
    masks["poleward_exterior"] = ~masks["belt_le_48.796875"]
    all_regions = REGIONS + ["belt_le_48.796875", "poleward_exterior"]

    w_c1 = area_weights_from_lat(lat_c)
    w_u1 = unet_pipeline_row_weights(lat_file)
    w_c = np.broadcast_to(w_c1[:, None], (H, W))
    w_u = np.broadcast_to(w_u1[:, None], (H, W))
    WROW = {"bilinear": (w_c, w_c1), "cca": (w_c, w_c1), "unet": (w_u, w_u1)}

    ACC = {"bilinear": ("cca", "bil_e2"), "cca": ("cca", "e2"),
           "unet": ("unet", "e2")}

    rows = []
    headline = {}
    spectra = {}
    boot = {}
    per_lat_profiles = {}
    stationary_maps = {}

    for clim, (root_s, summary_name) in CLIMATES.items():
        root = Path(root_s)
        a = np.load(STAGE_A / f"stage_a_{clim}.npz", allow_pickle=True)
        std = float(a["target_std_K"])
        n_days = int(a["n_days"])
        period_days = np.asarray(a["period_days"], dtype=np.int64)
        meta = json.loads((root / "retention_metadata.json").read_text())
        summ = json.loads((root / summary_name).read_text())


        acc = {}
        daily = {}
        map_e2 = {k: np.zeros((H, W)) for k in ("bilinear_e2", "cca_e2", "unet_e2")}
        zon_err = None
        sph_err = None
        for s in range(meta["n_shards"]):
            with np.load(root / "partials_complete" / f"partial_shard_{s:02d}.npz",
                         allow_pickle=True) as p:
                for k in p.files:
                    if k.startswith("acc_"):
                        acc[k[4:]] = acc.get(k[4:], 0) + np.asarray(p[k])
                    elif k.startswith("daily_") and not k.startswith("daily_dotT"):
                        daily.setdefault(k[6:], []).append(np.asarray(p[k]))
                    elif k.startswith("map_"):
                        map_e2[k[4:]] += np.asarray(p[k])
                zon_err = (np.asarray(p["zonal_error_power_sum_norm2"])
                           if zon_err is None else
                           zon_err + np.asarray(p["zonal_error_power_sum_norm2"]))
                sph_err = (np.asarray(p["spherical_error_c_ell_sum_norm2"])
                           if sph_err is None else
                           sph_err + np.asarray(p["spherical_error_c_ell_sum_norm2"]))
        daily = {k: np.concatenate(v) for k, v in daily.items()}
        log(f"{clim}: frozen accumulators reduced")


        pe = bootstrap_quantities(
            daily, np.arange(n_days, dtype=np.int64)[None, :], std)
        pe = {k: float(v[0]) for k, v in pe.items()}
        ref = (summ.get("held_out_headline_values")
               or summ.get("mh_headline_values")
               or summ.get("ssp585_headline_values"))
        for k, v in pe.items():
            if ref and k in ref:
                rel = abs(v - ref[k]) / max(abs(ref[k]), 1e-12)
                gate(f"{clim}_reproduce_{k}", rel < 1e-12, v, ref[k], {"rel": rel})


        for m in METHODS:
            rec = np.asarray(a[f"sum_e2_{m}"])
            frz = map_e2[f"{m}_e2"]
            rel = float(np.abs(rec - frz).max() / max(frz.max(), 1e-30))
            tol = 1e-6 if m == "bilinear" else 1e-10
            gate(f"{clim}_map_e2_{m}", rel < tol, rel, f"<{tol}")


        E = {m: {} for m in METHODS}
        for m in METHODS:
            for pi, per in enumerate(PERIODS):
                E[m][per] = std * np.asarray(a[f"sum_e1_{m}"])[pi] / period_days[pi]
        stationary_maps[clim] = {m: E[m]["annual"] for m in METHODS}


        for m in METHODS:
            w2d, wrow = WROW[m]
            pipe, key = ACC[m]
            for pi, per in enumerate(PERIODS):
                Em = E[m][per]
                for reg in all_regions:
                    mask = masks[reg]
                    if reg in REGIONS:
                        ri = REGIONS.index(reg)
                        tot = float(acc[f"{pipe}_{key}"][pi, ri]
                                    / acc[f"{pipe}_w"][pi, ri]) * std * std
                    else:
                        tot = float("nan")                                   
                    stat = weighted_mse(Em, w2d, mask)
                    a_r, mse_r, _ = fit_rigid(Em, T_K, w2d, mask)
                    al_p, mse_p, _ = fit_per_latitude(Em, T_K, w2d, mask)
                    _, mse_rc, _ = fit_rigid(Em, C_K, w2d, mask)
                    _, mse_pc, _ = fit_per_latitude(Em, C_K, w2d, mask)
                    rec = {"climate": clim, "method": m, "period": per,
                           "region": reg, "n_days": int(period_days[pi]),
                           "total_mse_K2": tot,
                           "stationary_mse_K2": stat,
                           "stationary_rms_K": math.sqrt(stat),
                           "rigid_alpha": a_r,
                           "rigid_aligned_mse_K2": mse_r,
                           "rigid_aligned_rms_K": math.sqrt(mse_r),
                           "perlat_aligned_mse_K2": mse_p,
                           "perlat_aligned_rms_K": math.sqrt(mse_p),
                           "rigid_frac_of_stationary": mse_r / stat if stat else 0.0,
                           "perlat_frac_of_stationary": mse_p / stat if stat else 0.0,
                           "control_rigid_aligned_mse_K2": mse_rc,
                           "control_perlat_aligned_mse_K2": mse_pc}
                    if np.isfinite(tot):
                        rec.update({
                            "rmse_K": math.sqrt(tot),
                            "rigid_frac_of_total": mse_r / tot,
                            "perlat_frac_of_total": mse_p / tot,
                            "rmse_adj_rigid_K": math.sqrt(max(tot - mse_r, 0.0)),
                            "rmse_adj_perlat_K": math.sqrt(max(tot - mse_p, 0.0)),
                            "d_rmse_rigid_K": math.sqrt(tot)
                                              - math.sqrt(max(tot - mse_r, 0.0)),
                            "d_rmse_perlat_K": math.sqrt(tot)
                                               - math.sqrt(max(tot - mse_p, 0.0))})
                        rec["rel_d_rmse_rigid"] = rec["d_rmse_rigid_K"] / rec["rmse_K"]
                        rec["rel_d_rmse_perlat"] = (rec["d_rmse_perlat_K"]
                                                    / rec["rmse_K"])
                    rows.append(rec)

            al, _, _ = fit_per_latitude(E[m]["annual"], T_K, WROW[m][0],
                                        masks["global"])
            per_lat_profiles.setdefault(clim, {})[m] = al


        glob = {r["method"]: r for r in rows
                if r["climate"] == clim and r["period"] == "annual"
                and r["region"] == "global"}
        hd = {"n_days": n_days, "std_K": std, "point": pe}
        for est in ("rigid", "perlat"):
            mse = {m: glob[m]["total_mse_K2"] for m in METHODS}
            adj = {m: mse[m] - glob[m][f"{est}_aligned_mse_K2"] for m in METHODS}
            hd[est] = {
                "aligned_mse_K2": {m: glob[m][f"{est}_aligned_mse_K2"]
                                   for m in METHODS},
                "rmse_K": {m: math.sqrt(mse[m]) for m in METHODS},
                "rmse_adj_K": {m: math.sqrt(adj[m]) for m in METHODS},
                "delta_rmse_K": math.sqrt(mse["cca"]) - math.sqrt(mse["unet"]),
                "delta_rmse_adj_K": math.sqrt(adj["cca"]) - math.sqrt(adj["unet"]),
                "skill_vs_cca": 1.0 - mse["unet"] / mse["cca"],
                "skill_vs_cca_adj": 1.0 - adj["unet"] / adj["cca"],
                "unet_skill_vs_bilinear": 1.0 - mse["unet"] / mse["bilinear"],
                "unet_skill_vs_bilinear_adj": 1.0 - adj["unet"] / adj["bilinear"],
                "cca_skill_vs_bilinear": 1.0 - mse["cca"] / mse["bilinear"],
                "cca_skill_vs_bilinear_adj": 1.0 - adj["cca"] / adj["bilinear"],
            }
        headline[clim] = hd


        TT = (T_K * T_K).sum(axis=1)
        dseries = {est: dict(daily) for est in ("rigid", "perlat")}
        for m in METHODS:
            w2d, wrow = WROW[m]
            dot = np.asarray(a[f"daily_dotT_{m}"])
            key = {"bilinear": ("bil_e2_c" if m == "bilinear" else None),
                   "cca": "cca_e2", "unet": "unet_e2"}[m]
            if m == "bilinear":
                key = "bil_e2_c"
            for est in ("rigid", "perlat"):
                if est == "rigid":
                    alpha = np.full(H, glob[m]["rigid_alpha"])
                else:
                    alpha = per_lat_profiles[clim][m]
                cross = 2.0 * (dot @ (wrow * alpha / std))
                const = float((wrow * alpha * alpha * TT).sum() / (std * std))
                dseries[est][key] = daily[key] - cross + const
        rng = np.random.default_rng(BOOTSTRAP["seed"])
        idx = circular_block_bootstrap_indices(n_days,
                                               BOOTSTRAP["block_length_days"],
                                               BOOTSTRAP["n_resamples"], rng)
        lo_p, hi_p = BOOTSTRAP["ci_percentiles"]
        bres = {"original": {"point": pe, "ci": {}}}
        d0 = bootstrap_quantities(daily, idx, std)
        bres["original"]["ci"] = {k: [float(np.percentile(v, lo_p)),
                                      float(np.percentile(v, hi_p))]
                                  for k, v in d0.items()}
        for est in ("rigid", "perlat"):
            pe_a = {k: float(v[0]) for k, v in bootstrap_quantities(
                dseries[est], np.arange(n_days, dtype=np.int64)[None, :],
                std).items()}
            da = bootstrap_quantities(dseries[est], idx, std)
            bres[est] = {"point": pe_a,
                         "ci": {k: [float(np.percentile(v, lo_p)),
                                    float(np.percentile(v, hi_p))]
                                for k, v in da.items()}}
        boot[clim] = bres
        log(f"{clim}: bootstrap done")


        sp = {}
        for m in METHODS:
            w2d, wrow = WROW[m]
            Em = E[m]["annual"]
            entry = {}
            for est in ("rigid", "perlat"):
                if est == "rigid":
                    _, _, P = fit_rigid(Em, T_K, w2d, masks["global"])
                else:
                    _, _, P = fit_per_latitude(Em, T_K, w2d, masks["global"])
                fields = np.stack([Em / std, (Em - P) / std])
                zp = zonal_row_weighted_power(fields, w_c1, W // 2)
                dz = n_days * (zp[1] - zp[0])
                entry[est] = {"zonal_delta_full": dz}
            sp[m] = entry
        spectra[clim] = {"zon_err_frozen": zon_err, "sph_err_frozen": sph_err,
                         "per_method": sp}
        log(f"{clim}: spectra done")


    import csv
    fields = sorted({k for r in rows for k in r})
    order = ["climate", "method", "period", "region"]
    fields = order + [f for f in fields if f not in order]
    with open(OUT / "xios_sensitivity_full_table.csv", "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=fields)
        wtr.writeheader()
        for r in rows:
            wtr.writerow(r)

    zon_out = {}
    for clim, s in spectra.items():
        zf = s["zon_err_frozen"]                                              
        entry = {}
        for mi, m in enumerate(METHODS):
            e = {}
            for est in ("rigid", "perlat"):
                dz = s["per_method"][m][est]["zonal_delta_full"]
                e[est] = {
                    "delta_k_le_437_sum": float(dz[: K_MAX + 1].sum()),
                    "frozen_k_le_437_sum": float(zf[mi].sum()),
                    "rel_change_k_le_437": float(dz[: K_MAX + 1].sum()
                                                 / zf[mi].sum()),
                    "delta_full_sum": float(dz.sum()),
                    "max_abs_bin_delta_k_le_437": float(
                        np.abs(dz[: K_MAX + 1]).max()),
                    "argmax_bin_k_le_437": int(np.argmax(np.abs(dz[: K_MAX + 1]))),
                    "worst_bin_rel_change": float(
                        (dz[: K_MAX + 1] / np.maximum(zf[mi], 1e-300))[
                            int(np.argmax(np.abs(dz[: K_MAX + 1])))]),
                    "per_bin_rel_change_k_le_437": (
                        dz[: K_MAX + 1] / np.maximum(zf[mi], 1e-300)).tolist(),
                }
            entry[m] = e
        zon_out[clim] = entry

    summary = {
        "kind": "PAPER_XIOS_IMPACT_STAGE_B",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "headline": headline,
        "bootstrap": boot,
        "zonal_spectra": zon_out,
        "gates": gates,
        "all_gates_pass": all(g["pass"] for g in gates),
        "comb_line_k": {"1980": comb_k_1980.tolist(), "2009": comb_k_2009.tolist()},
    }

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(str(type(o)))

    (OUT / "xios_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=1, default=default))
    np.savez(OUT / "stationary_maps_and_profiles.npz",
             **{f"E_{c}_{m}": stationary_maps[c][m]
                for c in stationary_maps for m in METHODS},
             **{f"alpha_{c}_{m}": per_lat_profiles[c][m]
                for c in per_lat_profiles for m in METHODS},
             template_comb_mean_K=T_K, latitude=lat_c)
    log(f"gates: {sum(g['pass'] for g in gates)}/{len(gates)} pass")
    print(json.dumps({"status": "PASS" if summary["all_gates_pass"] else "GATEFAIL",
                      "rows": len(rows), "out": str(OUT)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
