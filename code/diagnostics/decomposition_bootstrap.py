"""Estimate uncertainty for the saved error-decomposition components.
Resamples per-day errors and coefficient vectors to obtain intervals for CCA
floor, mapping, bias and variance, and U-Net in-span/out-of-span error, including
between-climate changes. Uses the configured circular temporal-block scheme.
Requires retained diagnostic arrays and the shipped full-precision reference
values; it does not recompute predictions or refit the EOF bases or CCA mapping."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from downscaling_numerics import (
    circular_block_bootstrap_indices,
)

import os


RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "results_root")                                                  


MECH = Path(f"{RESULTS_ROOT}/ssp585_pd_mh_full_mechanism_20260802T095350Z")
POINT_REFERENCES = (Path(__file__).resolve().parents[2]
                    / "results/decomposition_reference_values.json")

CLIMATES = ("pd", "mh", "ssp")
POP = {"pd": "test", "mh": "mh", "ssp": "ssp"}
LABEL = {"pd": "PD test", "mh": "MH", "ssp": "SSP5-8.5"}

H, W = 1280, 2624
TARGET_STD_K = 21.627892139211994
N_DAYS = 1096


BLOCK_LEN = 60
N_RESAMPLES = 10000
SEED = 20260715
CI_LO_PCT, CI_HI_PCT = 2.5, 97.5


APPENDIX_AB_INDEX_SHA256 = ("2a348a483d3d0ff4b78f1f8b08d384e1f9a375f813d131196"
                            "a1158f24cea9e11")
BLOCK_LENS_ROBUST = (30, 60, 90)

START = time.time()
ISSUES: list[str] = []
GATES: list[dict] = []


def log(msg: str) -> None:
    print(f"[{time.time() - START:7.1f}s] {msg}", flush=True)


def flag(msg: str) -> None:
    ISSUES.append(msg)
    print(f"[{time.time() - START:7.1f}s] *** FLAG *** {msg}", flush=True)


def gate(name, value, ref, tol, kind="abs"):
    dev = abs(float(value) - float(ref))
    rel = dev / max(abs(float(ref)), 1e-300)
    ok = (dev if kind == "abs" else rel) <= tol
    GATES.append({"gate": name, "value": float(value), "reference": float(ref),
                  "abs_deviation": dev, "rel_deviation": rel,
                  "tolerance": tol, "tolerance_kind": kind, "pass": bool(ok)})
    print(f"[{time.time() - START:7.1f}s] GATE {'PASS' if ok else 'FAIL'} {name}: "
          f"value={value!r} ref={ref!r} abs_dev={dev:.6e} rel_dev={rel:.6e} "
          f"tol={tol:g} ({kind})", flush=True)
    if not ok:
        flag(f"GATE FAILED {name}: value={value!r} ref={ref!r} abs_dev={dev:.6e}")
    return ok


_LAT = -90.0 + (np.arange(H, dtype=np.float64) + 0.5) * (180.0 / H)
_W_ROW = np.cos(np.deg2rad(_LAT))
_W_ROW /= _W_ROW.mean()
W_SUM = float(np.repeat(_W_ROW, W).sum())
COEF_SCALE = TARGET_STD_K / math.sqrt(W_SUM)


def sha256_array(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def pct(a):
    a = np.asarray(a, dtype=np.float64)
    good = np.isfinite(a)
    if not good.all():
        a = a[good]
    return (float(np.percentile(a, CI_LO_PCT)),
            float(np.percentile(a, CI_HI_PCT)), int(good.size - good.sum()))


def load_per_day():

    z = np.load(MECH / "outputs/exact_decomposition_per_day.npz", allow_pickle=False)
    out = {}
    for c in CLIMATES:
        p = POP[c]
        parts = [np.load(MECH / "partials" / f"model_{p}_{i:02d}.npz",
                         allow_pickle=False) for i in range(5)]
        parts = [parts[i] for i in np.argsort([int(x["lo"]) for x in parts])]
        day_index = np.concatenate([x["day_index"] for x in parts])
        if not np.array_equal(day_index, np.arange(N_DAYS)):
            raise SystemExit(f"{p}: partial day cover failed")
        s_true = np.concatenate([x["s_true"] for x in parts]).astype(np.float64)
        s_cca = np.concatenate([x["s_cca"] for x in parts]).astype(np.float64)
        dates = np.concatenate([x["dates"] for x in parts]).astype(str)
        is_feb29 = np.concatenate([x["is_feb29"] for x in parts]).astype(bool)


        q = {n: i for i, n in enumerate(parts[0]["quantities"].astype(str))}
        gi = list(parts[0]["regions"].astype(str)).index("global")
        daily = np.concatenate([x["daily"] for x in parts], axis=0)
        wsum = daily[:, gi, q["w"]]
        if float(np.ptp(wsum)) != 0.0:
            raise SystemExit(f"{p}: per-day weight sum is not constant")

        eps = (s_cca - s_true) * COEF_SCALE
        m_t = np.einsum("ij,ij->i", eps, eps)
        out[c] = {
            "eps": eps,
            "m_t": m_t,
            "f_t": np.asarray(z[f"{p}__Qe_cca2"], dtype=np.float64),
            "ein_t": np.asarray(z[f"{p}__Pe_unet2"], dtype=np.float64),
            "eout_t": np.asarray(z[f"{p}__Qe_unet2"], dtype=np.float64),
            "dates": dates, "is_feb29": is_feb29,
            "weight_sum_per_day": float(wsum[0]),
        }
        for k in ("f_t", "ein_t", "eout_t"):
            if out[c][k].shape != (N_DAYS,):
                raise SystemExit(f"{p}: {k} shape {out[c][k].shape}")
    z.close()
    return out


def point_estimates(d, mask=None):

    sl = slice(None) if mask is None else mask
    eps = d["eps"][sl]
    m_t = d["m_t"][sl]
    ebar = eps.mean(axis=0)
    B = float(ebar @ ebar)
    M = float(m_t.mean())
    V_direct = float(np.mean(np.einsum("ij,ij->i", eps - ebar, eps - ebar)))
    return {"F": float(d["f_t"][sl].mean()), "M": M, "B": B,
            "V": V_direct, "V_identity": M - B,
            "E_in": float(d["ein_t"][sl].mean()),
            "E_out": float(d["eout_t"][sl].mean()),
            "n_days": int(len(m_t))}


def replicate_means(x, idx, chunk=500):


    R = idx.shape[0]
    out = np.empty(R, dtype=np.float64)
    for a in range(0, R, chunk):
        b = min(R, a + chunk)
        out[a:b] = x[idx[a:b]].mean(axis=1)
    return out


def replicate_ebar(eps, idx):


    R, n = idx.shape
    flat = (idx + (np.arange(R, dtype=np.int64) * n)[:, None]).ravel()
    counts = np.bincount(flat, minlength=R * n).reshape(R, n).astype(np.float64)
    return (counts @ eps) / float(n), counts


def bootstrap_climate(d, idx, mask=None):

    sl = slice(None) if mask is None else mask
    eps = np.ascontiguousarray(d["eps"][sl])
    rep = {}
    for name, key in (("F", "f_t"), ("M", "m_t"),
                      ("E_in", "ein_t"), ("E_out", "eout_t")):
        rep[name] = replicate_means(np.ascontiguousarray(d[key][sl]), idx)
    ebar, counts = replicate_ebar(eps, idx)
    rep["B"] = np.einsum("ij,ij->i", ebar, ebar)
    rep["V"] = rep["M"] - rep["B"]

    m_counts = (counts @ np.ascontiguousarray(d["m_t"][sl])) / float(idx.shape[1])
    rep["_M_path_max_abs_dev"] = float(np.max(np.abs(m_counts - rep["M"])))
    with np.errstate(divide="ignore", invalid="ignore"):
        rep["B_over_M"] = rep["B"] / rep["M"]
    return rep


def draw_index_set(block_len, n_days, seed=SEED, climates=CLIMATES):

    rng = np.random.default_rng(seed)
    return {c: circular_block_bootstrap_indices(n_days, block_len,
                                                N_RESAMPLES, rng)
            for c in climates}


def integrated_autocorr_time(x, c_sokal=5.0, max_lag=None):

    x = np.asarray(x, dtype=np.float64)
    n = x.size
    y = x - x.mean()
    max_lag = n - 1 if max_lag is None else min(max_lag, n - 1)
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(y, nfft)
    acf = np.fft.irfft(f * np.conjugate(f), nfft)[: max_lag + 1]
    acf /= acf[0]
    tau, window = 1.0, max_lag
    for k in range(1, max_lag + 1):
        tau += 2.0 * acf[k]
        if k >= c_sokal * tau:
            window = k
            break
    return {"tau_int": float(tau), "window": int(window),
            "acf_lag1": float(acf[1]), "acf_lag5": float(acf[5]),
            "acf_lag10": float(acf[10]), "acf_lag30": float(acf[30]),
            "acf_lag60": float(acf[60]), "acf_lag90": float(acf[90]),
            "acf_lag180": float(acf[180]), "acf_lag365": float(acf[365]),
            "tau_int_fixed_lag60": float(1.0 + 2.0 * acf[1:61].sum()),
            "n_over_tau": float(n / tau)}


def main() -> int:
    root = Path(sys.argv[1])
    out = root / "outputs"
    arr = root / "arrays"
    out.mkdir(parents=True, exist_ok=True)
    arr.mkdir(parents=True, exist_ok=True)

    log(f"numpy {np.__version__}  python {sys.version.split()[0]}")
    log(f"W_SUM={W_SUM!r}  COEF_SCALE={COEF_SCALE!r}")


    with open(POINT_REFERENCES) as fh:
        references = json.load(fh)
    coefficient = references["coefficient_space"]
    pixel = references["pixel_space"]
    REF = {c: {"F": pixel[c]["F"], "M": coefficient[c]["M"],
               "B": coefficient[c]["B"], "V": coefficient[c]["V"],
               "E_in": pixel[c]["E_in"],
               "E_out": pixel[c]["E_out"]} for c in CLIMATES}
    REF_AUDIT_M = {c: pixel[c]["M"] for c in CLIMATES}


    log("STEP 1: assembling per-day quantities")
    D = load_per_day()
    pt = {c: point_estimates(D[c]) for c in CLIMATES}
    for c in CLIMATES:
        log(f"  {LABEL[c]}: n={pt[c]['n_days']} weight_sum/day="
            f"{D[c]['weight_sum_per_day']!r}")
        for k in ("F", "M", "B", "V", "E_in", "E_out"):


            tol = 1e-15 if k in ("M", "B", "V") else 1e-11
            gate(f"point_{c}_{k}", pt[c][k], REF[c][k], tol, "abs")


        gate(f"point_{c}_M_vs_audit_pixel_space_INFORMATIONAL",
             pt[c]["M"], REF_AUDIT_M[c], 1e-10, "abs")
        closure = pt[c]["V"] - pt[c]["V_identity"]
        log(f"  {LABEL[c]}: B+V-M closure = {closure:.3e} K^2 "
            f"(V_direct={pt[c]['V']!r}, V=M-B={pt[c]['V_identity']!r})")
        gate(f"closure_{c}_BplusV_minus_M", closure, 0.0, 1e-15, "abs")

        log(f"  {LABEL[c]}: mean_t(m_t) - M_accepted = "
            f"{pt[c]['M'] - REF[c]['M']:+.6e} K^2   "
            f"||mean_t(eps_t)||^2 - B_accepted = {pt[c]['B'] - REF[c]['B']:+.6e} K^2")


    log(f"STEP 2: block bootstrap, block={BLOCK_LEN} d, {N_RESAMPLES} resamples, "
        f"seed {SEED}")
    IDX = draw_index_set(BLOCK_LEN, N_DAYS)
    shas = {c: sha256_array(IDX[c]) for c in CLIMATES}
    for c in CLIMATES:
        log(f"  idx[{c}] shape={IDX[c].shape} sha256={shas[c]}")
    if shas["pd"] != APPENDIX_AB_INDEX_SHA256:
        flag("first drawn index array does NOT match the recorded Appendix A/B "
             f"array (got {shas['pd']}, expected {APPENDIX_AB_INDEX_SHA256})")
    else:
        log("  GATE PASS appendix_AB_index_reproduction: idx[pd] is bitwise the "
            "accepted uniform-shift-run index array")
    GATES.append({"gate": "appendix_AB_index_reproduction",
                  "value": shas["pd"], "reference": APPENDIX_AB_INDEX_SHA256,
                  "pass": shas["pd"] == APPENDIX_AB_INDEX_SHA256})

    REP = {}
    for c in CLIMATES:
        t0 = time.time()
        REP[c] = bootstrap_climate(D[c], IDX[c])
        log(f"  {LABEL[c]} replicates done ({time.time() - t0:.1f} s); "
            f"M path max_abs_dev={REP[c]['_M_path_max_abs_dev']:.3e}")

    rows = []

    def add_row(climate, quantity, point, reps, kind, note=""):
        lo, hi, nbad = pct(reps)
        r = {"climate": climate, "quantity": quantity, "kind": kind,
             "point_estimate": repr(float(point)),
             "ci_lo_2.5": repr(lo), "ci_hi_97.5": repr(hi),
             "ci_width": repr(hi - lo),
             "bootstrap_median": repr(float(np.median(reps[np.isfinite(reps)]))),
             "bootstrap_sd": repr(float(np.std(reps[np.isfinite(reps)], ddof=1))),
             "excludes_zero": "" if kind == "level" else str(bool(lo > 0 or hi < 0)),
             "point_outside_ci": str(bool(point < lo or point > hi)),
             "bootstrap_mean_minus_point": repr(
                 float(np.mean(reps[np.isfinite(reps)])) - float(point)),
             "n_nonfinite_replicates": nbad,
             "block_length_days": BLOCK_LEN, "n_resamples": N_RESAMPLES,
             "n_days": N_DAYS, "calendar": "native (1096 d, 29 Feb retained)",
             "scheme": ("within-climate circular moving-block; one index array "
                        "shared by every quantity of that climate"),
             "note": note}
        rows.append(r)
        return r

    for c in CLIMATES:
        for k in ("F", "M", "B", "V", "E_in", "E_out"):
            add_row(LABEL[c], k, pt[c][k], REP[c][k], "level")
        add_row(LABEL[c], "B_over_M", pt[c]["B"] / pt[c]["M"],
                REP[c]["B_over_M"], "level",
                "B recomputed inside each resample (mean of the resampled days)")


    log("STEP 3: cross-climate changes (UNPAIRED, independent draws per climate)")
    DELTAS = (("mh", "pd"), ("ssp", "pd"))
    for tgt, base in DELTAS:
        lab = f"{LABEL[tgt]} - {LABEL[base]}"
        for k in ("F", "M", "B", "V", "E_in", "E_out"):
            d_pt = pt[tgt][k] - pt[base][k]
            d_rep = REP[tgt][k] - REP[base][k]
            r = add_row(lab, f"delta_{k}", d_pt, d_rep, "delta",
                        "UNPAIRED: independent block resamples of two different "
                        "simulations, differenced. This DIFFERS from the paired "
                        "scheme used for delta-RMSE in Appendix B.")
            log(f"  {lab} d{k}: {d_pt:+.10g} K^2  CI [{r['ci_lo_2.5']}, "
                f"{r['ci_hi_97.5']}]  excl0={r['excludes_zero']}")


    log("STEP 4: derived comparisons")
    comp_rows = []

    def add_comp(name, question, point, reps, note, block_len=BLOCK_LEN,
                 calendar="native (1096 d)"):
        lo, hi, nbad = pct(reps)
        good = reps[np.isfinite(reps)]
        r = {"comparison": name, "question": question,
             "point_estimate": repr(float(point)),
             "ci_lo_2.5": repr(lo), "ci_hi_97.5": repr(hi),
             "ci_width": repr(hi - lo),
             "bootstrap_median": repr(float(np.median(good))),
             "bootstrap_sd": repr(float(np.std(good, ddof=1))),
             "excludes_zero": str(bool(lo > 0 or hi < 0)),
             "P_gt_0": repr(float(np.mean(good > 0.0))),
             "n_nonfinite_replicates": nbad,
             "block_length_days": block_len, "n_resamples": N_RESAMPLES,
             "calendar": calendar, "note": note}
        comp_rows.append(r)
        log(f"  {name} [{block_len}d,{calendar}]: {point:+.10g}  "
            f"CI [{lo:.6g}, {hi:.6g}]  excl0={r['excludes_zero']}  "
            f"P(>0)={float(np.mean(good > 0.0)):.4f}")
        return r

    def step4(rep, pts, block_len=BLOCK_LEN, calendar="native (1096 d)"):
        dF_mh = rep["mh"]["F"] - rep["pd"]["F"]
        dM_mh = rep["mh"]["M"] - rep["pd"]["M"]
        dF_ssp = rep["ssp"]["F"] - rep["pd"]["F"]
        dM_ssp = rep["ssp"]["M"] - rep["pd"]["M"]
        dB_ssp = rep["ssp"]["B"] - rep["pd"]["B"]
        pF_mh = pts["mh"]["F"] - pts["pd"]["F"]
        pM_mh = pts["mh"]["M"] - pts["pd"]["M"]
        pF_ssp = pts["ssp"]["F"] - pts["pd"]["F"]
        pM_ssp = pts["ssp"]["M"] - pts["pd"]["M"]
        pB_ssp = pts["ssp"]["B"] - pts["pd"]["B"]
        add_comp("MH: dF - dM",
                 "Is the MH degradation target-space dominated?",
                 pF_mh - pM_mh, dF_mh - dM_mh,
                 "UNPAIRED cross-climate. >0 supports 'MH is target-dominated'.",
                 block_len, calendar)
        add_comp("SSP5-8.5: dM - dF",
                 "Is the SSP degradation mapping dominated?",
                 pM_ssp - pF_ssp, dM_ssp - dF_ssp,
                 "UNPAIRED cross-climate. >0 supports 'SSP is mapping-dominated'.",
                 block_len, calendar)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = dB_ssp / dM_ssp
        add_comp("SSP5-8.5: dB / dM",
                 "Share of the SSP mapping-error increase that is persistent bias "
                 "(paper quotes 97.7%)",
                 pB_ssp / pM_ssp, ratio,
                 "Ratio of two UNPAIRED cross-climate differences; heavy-tailed "
                 "whenever a replicate's dM approaches zero.",
                 block_len, calendar)
        add_comp("dF(MH) - dF(SSP)",
                 "Is the target-space degradation genuinely different between the "
                 "two climates?",
                 pF_mh - pF_ssp, dF_mh - dF_ssp,
                 "Both terms share the SAME PD replicate, so the common PD draw "
                 "cancels; the MH and SSP draws remain independent.",
                 block_len, calendar)

    step4(REP, pt)


    log("ADDITION 1: PD Table-3 ratios, PAIRED (one shared index array)")
    for name, num, den, quoted in (("PD: M / E_in", "M", "E_in", 9.7),
                                   ("PD: F / E_out", "F", "E_out", 2.0)):
        with np.errstate(divide="ignore", invalid="ignore"):
            reps = REP["pd"][num] / REP["pd"][den]
        add_comp(name,
                 f"CCA-vs-U-Net ratio on the same PD days (paper quotes {quoted}x)",
                 pt["pd"][num] / pt["pd"][den], reps,
                 "PAIRED: numerator and denominator are evaluated on the SAME "
                 "resampled days (idx[pd]), as in the Appendix A/B usage.")


    log("ADDITION 2: U-Net net MSE change under SSP5-8.5")
    d_unet_ssp = ((REP["ssp"]["E_in"] + REP["ssp"]["E_out"])
                  - (REP["pd"]["E_in"] + REP["pd"]["E_out"]))
    p_unet_ssp = ((pt["ssp"]["E_in"] + pt["ssp"]["E_out"])
                  - (pt["pd"]["E_in"] + pt["pd"]["E_out"]))
    add_comp("SSP5-8.5: dE_in + dE_out",
             "Is the U-Net's 'slight net improvement' under SSP5-8.5 distinguishable "
             "from zero?",
             p_unet_ssp, d_unet_ssp,
             "UNPAIRED cross-climate; equals the change in total U-Net MSE. "
             "NEGATIVE = improvement.")
    d_unet_mh = ((REP["mh"]["E_in"] + REP["mh"]["E_out"])
                 - (REP["pd"]["E_in"] + REP["pd"]["E_out"]))
    p_unet_mh = ((pt["mh"]["E_in"] + pt["mh"]["E_out"])
                 - (pt["pd"]["E_in"] + pt["pd"]["E_out"]))
    add_comp("MH: dE_in + dE_out",
             "Companion to the SSP net-change question (reported for symmetry)",
             p_unet_mh, d_unet_mh,
             "UNPAIRED cross-climate; equals the change in total U-Net MSE.")


    log("ADDITION 3: block-length robustness (protocol unchanged; 60 d stays)")
    for bl in BLOCK_LENS_ROBUST:
        if bl == BLOCK_LEN:
            continue
        idx_b = draw_index_set(bl, N_DAYS)
        rep_b = {c: bootstrap_climate(D[c], idx_b[c]) for c in CLIMATES}
        step4(rep_b, pt, block_len=bl)
        del rep_b, idx_b


    log("REQUESTED: 1095-day noleap variant of the Step-4 comparisons")
    keep = {c: ~D[c]["is_feb29"] for c in CLIMATES}
    n_noleap = int(keep["pd"].sum())
    idx_nl = draw_index_set(BLOCK_LEN, n_noleap)
    pt_nl = {c: point_estimates(D[c], keep[c]) for c in CLIMATES}
    rep_nl = {c: bootstrap_climate(D[c], idx_nl[c], keep[c]) for c in CLIMATES}
    for c in CLIMATES:
        log(f"  noleap {LABEL[c]}: F={pt_nl[c]['F']!r} M={pt_nl[c]['M']!r} "
            f"B={pt_nl[c]['B']!r} V={pt_nl[c]['V']!r}")
    step4(rep_nl, pt_nl, calendar=f"noleap ({n_noleap} d)")


    log("STEP 5: noise floor on B and integrated autocorrelation time")
    n_eff = N_DAYS / BLOCK_LEN
    floor_rows = []
    for c in CLIMATES:
        tau = {k: integrated_autocorr_time(D[c][k])
               for k in ("m_t", "f_t", "ein_t", "eout_t")}
        fl = pt[c]["V"] / n_eff
        ratio = pt[c]["B"] / fl
        lo, hi, _ = pct(REP[c]["B"])


        shift = float(np.mean(REP[c]["B"])) - pt[c]["B"]
        p_below = float(np.mean(REP[c]["B"] < pt[c]["B"]))


        _sm = np.convolve(D[c]["m_t"], np.ones(31) / 31.0, mode="valid")
        seas = float(_sm.max() / _sm.min())
        floor_rows.append({
            "climate": LABEL[c], "B": repr(pt[c]["B"]), "V": repr(pt[c]["V"]),
            "n_days": N_DAYS, "block_length_days": BLOCK_LEN,
            "n_eff_blocks": repr(float(n_eff)),
            "V_over_n_eff": repr(float(fl)),
            "B_over_floor": repr(float(ratio)),
            "bootstrap_noise_floor_meanBstar_minus_B": repr(shift),
            "B_over_bootstrap_noise_floor": repr(float(pt[c]["B"] / shift)),
            "bootstrap_shift_over_V_over_n_eff": repr(float(shift / fl)),
            "P_Bstar_below_B": repr(p_below),
            "B_debiased_2B_minus_meanBstar": repr(float(2 * pt[c]["B"]
                                                        - np.mean(REP[c]["B"]))),
            "B_ci_lo_2.5": repr(lo), "B_ci_hi_97.5": repr(hi),
            "B_ci_lo_above_floor": str(bool(lo > fl)),
            "tau_int_m_t": repr(tau["m_t"]["tau_int"]),
            "tau_int_m_t_window": tau["m_t"]["window"],
            "tau_int_m_t_fixed_lag60": repr(tau["m_t"]["tau_int_fixed_lag60"]),
            "n_over_tau_m_t": repr(tau["m_t"]["n_over_tau"]),
            "acf_m_t_lag1": repr(tau["m_t"]["acf_lag1"]),
            "acf_m_t_lag5": repr(tau["m_t"]["acf_lag5"]),
            "acf_m_t_lag10": repr(tau["m_t"]["acf_lag10"]),
            "acf_m_t_lag30": repr(tau["m_t"]["acf_lag30"]),
            "acf_m_t_lag60": repr(tau["m_t"]["acf_lag60"]),
            "acf_m_t_lag90": repr(tau["m_t"]["acf_lag90"]),
            "acf_m_t_lag180": repr(tau["m_t"]["acf_lag180"]),
            "acf_m_t_lag365": repr(tau["m_t"]["acf_lag365"]),
            "m_t_seasonal_amplitude_ratio_31d": repr(seas),
            "tau_int_f_t": repr(tau["f_t"]["tau_int"]),
            "tau_int_ein_t": repr(tau["ein_t"]["tau_int"]),
            "tau_int_eout_t": repr(tau["eout_t"]["tau_int"]),
            "block_len_over_tau_m_t": repr(float(BLOCK_LEN / tau["m_t"]["tau_int"])),
        })
        log(f"  {LABEL[c]}: B={pt[c]['B']:.6e}  V/n_eff={fl:.6e}  B/floor={ratio:.3f}"
            f"  | empirical floor={shift:.6e}  B/emp={pt[c]['B']/shift:.3f}"
            f"  P(B*<B)={p_below:.4f}"
            f"  | tau_int(m_t)={tau['m_t']['tau_int']:.3f} d "
            f"(window {tau['m_t']['window']})  60/tau={60/tau['m_t']['tau_int']:.2f}")


    with open(out / "decomposition_bootstrap.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out / "decomposition_comparisons.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(comp_rows[0].keys()))
        w.writeheader()
        w.writerows(comp_rows)
    with open(out / "b_noise_floor.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(floor_rows[0].keys()))
        w.writeheader()
        w.writerows(floor_rows)

    np.savez(arr / "bootstrap_replicates_block60.npz",
             **{f"{c}__{k}": REP[c][k] for c in CLIMATES
                for k in ("F", "M", "B", "V", "E_in", "E_out", "B_over_M")},
             **{f"idx_sha256__{c}": np.array(shas[c]) for c in CLIMATES})

    prov = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "numpy": np.__version__, "python": sys.version.split()[0],
        "declarations": {"models_rerun": False, "cca_refit": False,
                         "eofs_recomputed": False, "unet_inference_run": False,
                         "accepted_files_modified": False,
                         "only_resampling_of_existing_per_day_series": True},
        "sources": {
            "per_day": str(MECH / "outputs/exact_decomposition_per_day.npz"),
            "partials": str(MECH / "partials/model_{test,mh,ssp}_0[0-4].npz"),
            "bootstrap_generator": str(Path(__file__).resolve().parents[1]
                                       / "downscaling_numerics.py"
                                       "::circular_block_bootstrap_indices"),
            "point_reference_pixel_space": str(POINT_REFERENCES),
            "point_reference_coefficient_space": str(POINT_REFERENCES),
        },
        "conventions": {
            "COEF_SCALE": COEF_SCALE, "W_SUM": W_SUM,
            "TARGET_STD_K": TARGET_STD_K,
            "F_source": "Qe_cca2 (NOT o_true2)",
            "calendar": "native, 1096 days, 29 February retained",
        },
        "bootstrap": {
            "block_length_days": BLOCK_LEN, "n_resamples": N_RESAMPLES,
            "seed": SEED, "ci_percentiles": [CI_LO_PCT, CI_HI_PCT],
            "bias_correction": "none (plain percentile)",
            "rng": f"numpy.random.default_rng({SEED}), one stream, "
                   "climates drawn in order (pd, mh, ssp)",
            "within_climate": "one index array shared by every quantity",
            "cross_climate": "UNPAIRED - independent draws per climate",
            "index_sha256": shas,
            "appendix_AB_index_reproduced": shas["pd"] == APPENDIX_AB_INDEX_SHA256,
            "block_lengths_robustness": list(BLOCK_LENS_ROBUST),
        },
        "point_estimates": {c: pt[c] for c in CLIMATES},
        "point_estimates_noleap": {c: pt_nl[c] for c in CLIMATES},
        "point_reference_values": REF,
        "point_reference_M_audit_pixel_space": REF_AUDIT_M,
        "M_reference_note": (
            "M is gated against B_in+V_in of the accepted no-refit four-way "
            "triage (score space, bitwise). The paper audit's M is the "
            "separately accumulated pixel-space Pe_cca2 and differs by "
            "1.7e-11 (PD) / 2.6e-11 (MH) / 5.6e-11 (SSP) K^2 -- a property of "
            "the accepted files, reported as an informational gate."),
        "gates": GATES,
        "issues": ISSUES,
    }
    with open(out / "provenance.json", "w") as fh:
        json.dump(prov, fh, indent=1, default=float)

    npass = sum(1 for g in GATES if g["pass"])
    log(f"GATES {npass}/{len(GATES)} PASS; {len(ISSUES)} issues")
    for i in ISSUES:
        print("  ISSUE:", i)
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
