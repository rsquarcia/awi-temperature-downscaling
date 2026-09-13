"""Reduce model-error partials into exact inside/outside EOF-space components.
Uses the frozen 512-mode CCA target subspace to separate CCA mapping error and
the unresolved target floor, and U-Net error inside and outside the same span.
Attributes changes in the CCA-minus-U-Net gap in squared-error units, then
writes summaries and per-day quantities for downstream uncertainty analyses.
Reads existing model-pass outputs; it does not rerun inference."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import decomposition_common as C
import sup_core as S
from model_pass import MODEL_QUANTITIES

POPS = C.MODEL_POPULATIONS
NPIX = C.HEIGHT * C.WIDTH
TS2 = C.TARGET_STD_K ** 2
N_MODEL_TASKS = 5


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_model_pop(root: Path, pop: str) -> dict:
    parts = []
    for t in range(N_MODEL_TASKS):
        p = root / "partials" / f"model_{pop}_{t:02d}.npz"
        if not p.exists():
            raise SystemExit(f"missing model partial {p}")
        parts.append(np.load(p, allow_pickle=False))
    parts = [parts[i] for i in np.argsort([int(z["lo"]) for z in parts])]
    out = {"population": pop}
    for k in ("dates", "day_index", "season_index", "year_index_global",
              "is_feb29"):
        out[k] = np.concatenate([z[k] for z in parts])
    out["dates"] = out["dates"].astype("U10")
    for k in ("daily", "s_true", "s_unet", "s_cca", "zonal_out_unet",
              "zonal_out_true"):
        out[k] = np.concatenate([z[k] for z in parts], axis=0)
    for k in ("gate_baseline_max_abs_dev", "gate_cca_outside_energy_fraction",
              "gate_unet_pythagorean_rel_dev"):
        out[k] = np.concatenate([z[k] for z in parts])
    out["regions"] = list(parts[0]["regions"].astype(str))
    out["quantities"] = list(parts[0]["quantities"].astype(str))
    out["zonal_row_weight_sum"] = float(parts[0]["zonal_row_weight_sum"])
    out["season_days"] = sum(z["season_days"] for z in parts)
    for k in [f"ann_{x}" for x in
              ("T_sq", "B_sq", "r_sum", "r_sq", "e_cca_sq", "e_unet_sq",
               "o_true_sq", "o_unet_sq", "Pe_unet_sq", "Pe_cca_sq")]:
        out[k] = sum(np.asarray(z[k], dtype=np.float64) for z in parts)
    for k in [f"sea_{x}" for x in ("T_sum", "B_sum", "r_sum", "r_sq")]:
        out[k] = sum(np.asarray(z[k], dtype=np.float64) for z in parts)
    if out["dates"].size != C.POP_NDAYS[pop]:
        raise SystemExit(f"{pop}: {out['dates'].size} model days")
    if not np.array_equal(out["day_index"], np.arange(C.POP_NDAYS[pop])):
        raise SystemExit(f"{pop}: incomplete model-pass day cover")
    for z in parts:
        z.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    out = root / "outputs"; tab = root / "tables"
    out.mkdir(parents=True, exist_ok=True); tab.mkdir(parents=True, exist_ok=True)

    R = {"definition": {
        "projector": "P = orthogonal projector onto the frozen 512-mode CCA "
                     "output EOF span in the cos-latitude-weighted space",
        "anomalies": "all residuals are taken as anomalies about the frozen "
                     "PD-training residual mean mu_r (the basis mean field); "
                     "mu_r cancels exactly in every error term",
        "cca": "MSE_CCA = ||yhat - P r||^2 + ||P r - r||^2  (mapping + floor); "
               "exact because the CCA prediction lies in the span of P",
        "unet": "MSE_UNet = ||P(rhat - r)||^2 + ||(I-P)(rhat - r)||^2",
        "gap": "MSE_CCA - MSE_UNet = [mapping - inside] + [floor - outside]",
        "units": "weighted mean squared error in K^2; the cos-latitude weight "
                 "is mean-one so the weighted mean equals the area mean"}}

    dat = {p: load_model_pop(root, p) for p in POPS}
    q = {k: i for i, k in enumerate(MODEL_QUANTITIES)}
    reg = {k: i for i, k in enumerate(C.ALL_REGIONS)}
    gi = reg["global"]


    gates = {}
    for p in POPS:
        d = dat[p]
        D = d["daily"]
        wsum = D[:, gi, q["w"]].sum()
        rm = {"bilinear": float(np.sqrt(D[:, gi, q["r2"]].sum() / wsum)),
              "cca": float(np.sqrt(D[:, gi, q["e_cca2"]].sum() / wsum)),
              "unet": float(np.sqrt(D[:, gi, q["e_unet2"]].sum() / wsum))}
        acc = C.ACCEPTED_RMSE[p]
        g = {"reproduced_rmse_K": rm, "accepted_rmse_K": acc,
             "rel_dev": {k: (abs(rm[k] - acc[k]) / acc[k]
                             if acc[k] is not None else None) for k in rm},
             "baseline_max_abs_dev_norm": float(
                 d["gate_baseline_max_abs_dev"].max()),
             "cca_outside_energy_fraction_max": float(
                 d["gate_cca_outside_energy_fraction"].max()),
             "unet_pythagorean_max_rel_dev": float(
                 d["gate_unet_pythagorean_rel_dev"].max())}
        bad = [k for k in rm if acc[k] is not None
               and g["rel_dev"][k] > C.ACCEPTED_RMSE_TOL_REL]
        g["all_accepted_reproduced"] = not bad
        gates[p] = g
        log(f"GATE {p}: bilinear {rm['bilinear']:.10f} cca {rm['cca']:.10f} "
            f"unet {rm['unet']:.10f}  rel dev "
            + ", ".join(f"{k}={g['rel_dev'][k]:.2e}" for k in rm
                        if g["rel_dev"][k] is not None))
        if bad:
            raise SystemExit(f"ACCEPTED-RMSE GATE FAILED for {p}: {bad}")
    R["gates"] = gates


    def terms(p, mask=None, region="global"):
        D = dat[p]["daily"]
        m = np.ones(D.shape[0], bool) if mask is None else mask
        ir = reg[region]
        w = D[m, ir, q["w"]].sum()
        f = lambda k: float(D[m, ir, q[k]].sum() / w)
        return {"n_days": int(m.sum()), "region_weight": float(w),
                "bilinear_mse": f("r2"),
                "cca_mse": f("e_cca2"), "unet_mse": f("e_unet2"),
                "cca_mapping_mse": f("Pe_cca2"), "cca_floor_mse": f("Qe_cca2"),
                "cca_cross_mse": f("PQe_cca"),
                "unet_inside_mse": f("Pe_unet2"),
                "unet_outside_mse": f("Qe_unet2"),
                "unet_cross_mse": f("PQe_unet"),
                "true_out_of_subspace_mse": f("o_true2"),
                "unet_out_correction_mse": f("o_unet2"),
                "unet_out_cross": f("o_cross")}

    keep = {p: ~dat[p]["is_feb29"] for p in POPS}
    rows, per_day = [], {}
    for p in POPS:
        for scope, mask in ([("annual", None)]
                            + [(sn, dat[p]["season_index"] == si)
                               for si, sn in enumerate(C.SEASONS)]
                            + [("annual_noleap", keep[p])]):
            t = terms(p, mask)
            closure_cca = abs(t["cca_mse"] - (t["cca_mapping_mse"]
                                              + t["cca_floor_mse"]))
            closure_unet = abs(t["unet_mse"] - (t["unet_inside_mse"]
                                                + t["unet_outside_mse"]))
            row = {"population": p, "scope": scope, **t,
                   "bilinear_rmse_K": np.sqrt(t["bilinear_mse"]),
                   "cca_rmse_K": np.sqrt(t["cca_mse"]),
                   "unet_rmse_K": np.sqrt(t["unet_mse"]),
                   "cca_skill_vs_bilinear": 1 - np.sqrt(t["cca_mse"] / t["bilinear_mse"]),
                   "unet_skill_vs_bilinear": 1 - np.sqrt(t["unet_mse"] / t["bilinear_mse"]),
                   "unet_skill_vs_cca": 1 - np.sqrt(t["unet_mse"] / t["cca_mse"]),
                   "cca_mapping_share": t["cca_mapping_mse"] / t["cca_mse"],
                   "cca_floor_share": t["cca_floor_mse"] / t["cca_mse"],
                   "unet_inside_share": t["unet_inside_mse"] / t["unet_mse"],
                   "unet_outside_share": t["unet_outside_mse"] / t["unet_mse"],
                   "cca_pythagorean_abs_closure_K2": closure_cca,
                   "cca_pythagorean_rel_closure": closure_cca / t["cca_mse"],
                   "unet_pythagorean_abs_closure_K2": closure_unet,
                   "unet_pythagorean_rel_closure": closure_unet / t["unet_mse"],
                   "gap_mse_cca_minus_unet_K2": t["cca_mse"] - t["unet_mse"],
                   "within_subspace_comparative_advantage_K2":
                       t["cca_mapping_mse"] - t["unet_inside_mse"],
                   "outside_subspace_comparative_advantage_K2":
                       t["cca_floor_mse"] - t["unet_outside_mse"],
                   "unet_out_variance_recovered_fraction":
                       (2 * t["unet_out_cross"] - t["unet_out_correction_mse"])
                       / t["true_out_of_subspace_mse"],
                   "unet_out_correlation":
                       t["unet_out_cross"] / np.sqrt(
                           max(t["unet_out_correction_mse"], 1e-30)
                           * t["true_out_of_subspace_mse"])}
            rows.append(row)
        D = dat[p]["daily"]
        w = D[:, gi, q["w"]]
        per_day[p] = {k: D[:, gi, q[k]] / w for k in
                      ("r2", "e_cca2", "e_unet2", "Pe_cca2", "Qe_cca2",
                       "Pe_unet2", "Qe_unet2", "o_true2", "o_unet2",
                       "o_cross")}
        per_day[p]["dates"] = dat[p]["dates"]
        per_day[p]["season"] = dat[p]["season_index"]
        per_day[p]["is_feb29"] = dat[p]["is_feb29"]
    S.write_csv(tab / "exact_cca_mse_decomposition.csv",
                [r for r in rows], list(rows[0].keys()))
    S.write_csv(tab / "exact_unet_subspace_decomposition.csv",
                [r for r in rows], list(rows[0].keys()))
    R["decomposition"] = rows
    log(f"exact decomposition: {len(rows)} rows")


    regional = []
    for p in POPS:
        for rg in C.ALL_REGIONS:
            t = terms(p, None, rg)
            tot_c = t["cca_mapping_mse"] + t["cca_floor_mse"] + 2 * t["cca_cross_mse"]
            tot_u = (t["unet_inside_mse"] + t["unet_outside_mse"]
                     + 2 * t["unet_cross_mse"])
            regional.append({
                "population": p, "region": rg, **t,
                "cca_three_term_sum_K2": tot_c,
                "cca_three_term_abs_closure_K2": abs(tot_c - t["cca_mse"]),
                "unet_three_term_sum_K2": tot_u,
                "unet_three_term_abs_closure_K2": abs(tot_u - t["unet_mse"]),
                "cca_cross_share_of_total": 2 * t["cca_cross_mse"] / t["cca_mse"],
                "unet_cross_share_of_total": 2 * t["unet_cross_mse"] / t["unet_mse"],
                "bilinear_rmse_K": np.sqrt(t["bilinear_mse"]),
                "cca_rmse_K": np.sqrt(t["cca_mse"]),
                "unet_rmse_K": np.sqrt(t["unet_mse"]),
                "unet_skill_vs_cca": 1 - np.sqrt(t["unet_mse"] / t["cca_mse"])})
    S.write_csv(tab / "regional_exact_decomposition.csv", regional)
    R["regional"] = regional


    att = {}
    for tgt in ("ssp", "mh"):
        for base in ("test", "val"):
            a0 = terms(base, keep[base]); a1 = terms(tgt, keep[tgt])
            gap0 = a0["cca_mse"] - a0["unet_mse"]
            gap1 = a1["cca_mse"] - a1["unet_mse"]
            d_within = ((a1["cca_mapping_mse"] - a1["unet_inside_mse"])
                        - (a0["cca_mapping_mse"] - a0["unet_inside_mse"]))
            d_outside = ((a1["cca_floor_mse"] - a1["unet_outside_mse"])
                         - (a0["cca_floor_mse"] - a0["unet_outside_mse"]))
            dg = gap1 - gap0
            e = {"target": tgt, "baseline": base,
                 "gap_mse_baseline_K2": gap0, "gap_mse_target_K2": gap1,
                 "delta_gap_mse_K2": dg,
                 "delta_within_subspace_K2": d_within,
                 "delta_outside_subspace_K2": d_outside,
                 "share_within": d_within / dg, "share_outside": d_outside / dg,
                 "closure_abs_K2": abs(dg - (d_within + d_outside)),

                 "cca_delta_total_mse_K2": a1["cca_mse"] - a0["cca_mse"],
                 "cca_delta_mapping_mse_K2": (a1["cca_mapping_mse"]
                                              - a0["cca_mapping_mse"]),
                 "cca_delta_floor_mse_K2": a1["cca_floor_mse"] - a0["cca_floor_mse"],
                 "cca_floor_share_of_increase": (
                     (a1["cca_floor_mse"] - a0["cca_floor_mse"])
                     / (a1["cca_mse"] - a0["cca_mse"])),
                 "cca_mapping_share_of_increase": (
                     (a1["cca_mapping_mse"] - a0["cca_mapping_mse"])
                     / (a1["cca_mse"] - a0["cca_mse"])),

                 "unet_delta_total_mse_K2": a1["unet_mse"] - a0["unet_mse"],
                 "unet_delta_inside_mse_K2": (a1["unet_inside_mse"]
                                              - a0["unet_inside_mse"]),
                 "unet_delta_outside_mse_K2": (a1["unet_outside_mse"]
                                               - a0["unet_outside_mse"]),
                 "unet_inside_share_of_change": (
                     (a1["unet_inside_mse"] - a0["unet_inside_mse"])
                     / (a1["unet_mse"] - a0["unet_mse"])),
                 "unet_outside_share_of_change": (
                     (a1["unet_outside_mse"] - a0["unet_outside_mse"])
                     / (a1["unet_mse"] - a0["unet_mse"])),

                 "cca_rmse_before_K": np.sqrt(a0["cca_mse"]),
                 "cca_rmse_after_K": np.sqrt(a1["cca_mse"]),
                 "unet_rmse_before_K": np.sqrt(a0["unet_mse"]),
                 "unet_rmse_after_K": np.sqrt(a1["unet_mse"]),
                 "skill_vs_cca_before": 1 - np.sqrt(a0["unet_mse"] / a0["cca_mse"]),
                 "skill_vs_cca_after": 1 - np.sqrt(a1["unet_mse"] / a1["cca_mse"]),
                 "rmse_gap_before_K": np.sqrt(a0["cca_mse"]) - np.sqrt(a0["unet_mse"]),
                 "rmse_gap_after_K": np.sqrt(a1["cca_mse"]) - np.sqrt(a1["unet_mse"])}
            att[f"{tgt}_minus_{base}"] = e
            log(f"{tgt} - {base}: dGap {dg:.6e} K2  within {e['share_within']:.4f} "
                f"outside {e['share_outside']:.4f}; CCA floor share of its own "
                f"increase {e['cca_floor_share_of_increase']:.4f}")


    nb = C.BOOTSTRAP["n_resamples"]
    rng = np.random.default_rng(C.BOOTSTRAP["seed"])
    idx = {p: C.circular_block_bootstrap_indices(
        int(keep[p].sum()), C.BOOTSTRAP["block_length_days"], nb, rng)
        for p in POPS}
    bs = {}
    for p in POPS:
        m = keep[p]
        bs[p] = {k: per_day[p][k][m][idx[p]].mean(axis=1)
                 for k in ("r2", "e_cca2", "e_unet2", "Pe_cca2", "Qe_cca2",
                           "Pe_unet2", "Qe_unet2", "o_true2", "o_unet2",
                           "o_cross")}
    for key, e in att.items():
        tgt, base = e["target"], e["baseline"]
        g0 = bs[base]["e_cca2"] - bs[base]["e_unet2"]
        g1 = bs[tgt]["e_cca2"] - bs[tgt]["e_unet2"]
        w0 = ((bs[tgt]["Pe_cca2"] - bs[tgt]["Pe_unet2"])
              - (bs[base]["Pe_cca2"] - bs[base]["Pe_unet2"]))
        o0 = ((bs[tgt]["Qe_cca2"] - bs[tgt]["Qe_unet2"])
              - (bs[base]["Qe_cca2"] - bs[base]["Qe_unet2"]))
        dg = g1 - g0
        good = np.isfinite(dg) & (dg != 0)
        pc = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
        e["delta_gap_ci95"] = pc(dg)
        e["delta_within_ci95"] = pc(w0)
        e["delta_outside_ci95"] = pc(o0)
        e["share_within_ci95"] = pc((w0 / dg)[good])
        e["share_outside_ci95"] = pc((o0 / dg)[good])
        dc = bs[tgt]["e_cca2"] - bs[base]["e_cca2"]
        gc = np.isfinite(dc) & (dc > 0)
        e["cca_floor_share_of_increase_ci95"] = pc(
            ((bs[tgt]["Qe_cca2"] - bs[base]["Qe_cca2"]) / dc)[gc])
        e["cca_mapping_share_of_increase_ci95"] = pc(
            ((bs[tgt]["Pe_cca2"] - bs[base]["Pe_cca2"]) / dc)[gc])
        du = bs[tgt]["e_unet2"] - bs[base]["e_unet2"]
        gu = np.isfinite(du) & (du != 0)
        e["unet_inside_share_of_change_ci95"] = pc(
            ((bs[tgt]["Pe_unet2"] - bs[base]["Pe_unet2"]) / du)[gu])
        e["unet_outside_share_of_change_ci95"] = pc(
            ((bs[tgt]["Qe_unet2"] - bs[base]["Qe_unet2"]) / du)[gu])
        e["n_bootstrap"] = int(nb)
    R["gap_widening"] = att
    hk = []
    for e in att.values():
        for k in e:
            if k not in hk:
                hk.append(k)
    S.write_csv(tab / "exact_gap_widening_decomposition.csv",
                [{k: (";".join(f"{x:.6g}" for x in v) if isinstance(v, list) else v)
                  for k, v in e.items()} for e in att.values()], hk)


    cal = []
    for p in POPS:
        yh, yo = dat[p]["s_cca"], dat[p]["s_true"]
        for K in (1, 5, 20, 50, 100, 512):
            a, b = yh[:, :K], yo[:, :K]
            err = a - b
            cal.append({
                "population": p, "modes": K,
                "bias_rms": float(np.sqrt((err.mean(axis=0) ** 2).mean())),
                "rmse": float(np.sqrt((err ** 2).mean())),
                "pred_over_oracle_variance_ratio":
                    float(a.var(axis=0).sum() / b.var(axis=0).sum()),
                "mean_correlation": float(np.mean(
                    [np.corrcoef(a[:, k], b[:, k])[0, 1] for k in range(K)])),
                "mean_calibration_slope": float(np.mean(
                    [np.polyfit(a[:, k], b[:, k], 1)[0] for k in range(K)])),
                "mean_calibration_intercept": float(np.mean(
                    [np.polyfit(a[:, k], b[:, k], 1)[1] for k in range(K)]))})
    S.write_csv(tab / "cca_score_calibration.csv", cal)
    R["cca_calibration"] = cal

    mode_rows = []
    for tgt in ("ssp", "mh"):
        a1 = dat[tgt]["s_cca"] - dat[tgt]["s_true"]
        a0 = dat["test"]["s_cca"] - dat["test"]["s_true"]
        d = (a1 ** 2).mean(axis=0) - (a0 ** 2).mean(axis=0)
        order = np.argsort(-d)[:20]
        R[f"top20_modes_mapping_error_increase_{tgt}"] = [
            {"mode": int(k), "delta_mean_sq_error": float(d[k]),
             "test_mean_sq": float((a0[:, k] ** 2).mean()),
             "target_mean_sq": float((a1[:, k] ** 2).mean()),
             "oracle_sd_test": float(dat["test"]["s_true"][:, k].std(ddof=1)),
             "oracle_sd_target": float(dat[tgt]["s_true"][:, k].std(ddof=1))}
            for k in order]
        R[f"mapping_error_increase_top20_share_{tgt}"] = float(
            d[order].sum() / d.sum())
        for k in order:
            mode_rows.append({"target": tgt, "mode": int(k),
                              "delta_mean_sq_error": float(d[k])})


    rec = []
    for p in POPS:
        st, su = dat[p]["s_true"], dat[p]["s_unet"]
        D = dat[p]["daily"]; w = D[:, gi, q["w"]].sum()
        f = lambda k: float(D[:, gi, q[k]].sum() / w)
        ot2, ou2, oc = f("o_true2"), f("o_unet2"), f("o_cross")
        row = {"population": p,
               "inside_true_variance": float((st ** 2).mean(axis=0).sum()),
               "inside_pred_variance": float((su ** 2).mean(axis=0).sum()),
               "inside_pred_over_true_variance_ratio":
                   float((su ** 2).mean(axis=0).sum()
                         / (st ** 2).mean(axis=0).sum()),
               "inside_mean_correlation": float(np.mean(
                   [np.corrcoef(su[:, k], st[:, k])[0, 1] for k in range(C.KY)])),
               "inside_mean_calibration_slope": float(np.mean(
                   [np.polyfit(su[:, k], st[:, k], 1)[0] for k in range(C.KY)])),
               "outside_true_mse_K2": ot2,
               "outside_pred_mse_K2": ou2,
               "outside_cross_K2": oc,
               "outside_correlation": oc / np.sqrt(max(ou2, 1e-30) * ot2),
               "outside_error_mse_K2": ot2 - 2 * oc + ou2,
               "outside_variance_recovered_fraction":
                   (2 * oc - ou2) / ot2,
               "outside_amplitude_ratio": np.sqrt(ou2 / ot2)}

        zu = dat[p]["zonal_out_unet"].mean(axis=0) * TS2
        zt = dat[p]["zonal_out_true"].mean(axis=0) * TS2
        for nm, sl in (("k1_10", slice(1, 11)), ("k11_100", slice(11, 101)),
                       ("k101_437", slice(101, 438)),
                       ("k438_1000", slice(438, 1001)),
                       ("k1001_1312", slice(1001, None))):
            row[f"out_unet_power_{nm}"] = float(zu[sl].sum())
            row[f"out_true_power_{nm}"] = float(zt[sl].sum())
            row[f"out_power_ratio_{nm}"] = float(zu[sl].sum() / zt[sl].sum())
        rec.append(row)
        log(f"  {p}: outside recovered fraction "
            f"{row['outside_variance_recovered_fraction']:.4f}, corr "
            f"{row['outside_correlation']:.4f}")
    S.write_csv(tab / "unet_inside_outside_recovery.csv", rec)
    R["unet_recovery"] = rec

    np.savez_compressed(
        out / "exact_decomposition_per_day.npz",
        **{f"{p}__{k}": np.asarray(v) for p in POPS
           for k, v in per_day[p].items()},
        **{f"{p}__zonal_out_unet": dat[p]["zonal_out_unet"].mean(axis=0)
           for p in POPS},
        **{f"{p}__zonal_out_true": dat[p]["zonal_out_true"].mean(axis=0)
           for p in POPS})
    np.savez_compressed(
        out / "exact_decomposition_maps.npz",
        lat=S.LAT, lon=C.grid_lon_centers(C.WIDTH),
        **{f"{p}__{k}": (dat[p][k] / C.POP_NDAYS[p]).astype(np.float32)
           for p in POPS for k in
           ("ann_e_cca_sq", "ann_e_unet_sq", "ann_o_true_sq", "ann_o_unet_sq",
            "ann_Pe_unet_sq", "ann_Pe_cca_sq", "ann_r_sq", "ann_r_sum")})
    S.jdump(out / "section_i.json", R)
    log("wrote section_i.json and the exact-decomposition tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
