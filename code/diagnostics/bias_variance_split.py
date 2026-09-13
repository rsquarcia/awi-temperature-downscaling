"""Split retained CCA mapping error into persistent bias and temporal variance.
Combines saved per-day errors and target/predicted EOF coefficients, checks the
coefficient-space decomposition and estimates climate changes and angles using
temporal resampling. Writes numerical summaries and bootstrap samples.
Requires retained mechanism-run artifacts; no model or EOF basis is fitted.
This is a top-level analysis script and reads its inputs when executed/imported."""

from __future__ import annotations

import hashlib
import json
import math
import platform
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


MECH = Path(f"{RESULTS_ROOT}/"
            "ssp585_pd_mh_full_mechanism_20260802T095350Z")
PER_DAY = MECH / "outputs" / "per_day_novelty_and_error.npz"
PER_DAY_X = MECH / "outputs" / "exact_decomposition_per_day.npz"
MASTER = MECH / "outputs" / "master_results.json"
MANIFEST_IN = MECH / "manifest.json"
T_THREE = MECH / "tables" / "three_climate_mechanism_table.csv"
MODEL_PASS = Path(__file__).resolve().with_name("model_pass.py")
OUT_PARENT = Path(f"{RESULTS_ROOT}")

POPS = ("test", "mh", "ssp")
POP_LABEL = {"test": "PD test (2012-2014)", "mh": "MH (2076-2078)",
             "ssp": "SSP5-8.5 (2096-2098)"}
SEASONS = ("DJF", "MAM", "JJA", "SON")
KY = 512
N_SHARDS = 5

BLOCK_LEN = 60
N_RESAMPLES = 10000
SEED = 20260715
DRAW_ORDER = ("test", "mh", "ssp")
EPS_DEGENERATE = 1e-12


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def utc_stamp():
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def utc_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


print("=" * 78)
print("1. RETAINED SCALARS  (native 1096-day convention)")
print("=" * 78)
z = np.load(PER_DAY, allow_pickle=True)
F, M_ret, TOT, SEA, DATES = {}, {}, {}, {}, {}
for p in POPS:
    F[p] = np.asarray(z[f"{p}__cca_floor_mse"], np.float64)
    M_ret[p] = np.asarray(z[f"{p}__cca_map_mse"], np.float64)
    TOT[p] = np.asarray(z[f"{p}__cca_total_mse"], np.float64)
    SEA[p] = np.asarray(z[f"{p}__season"], np.int64)
    DATES[p] = np.asarray(z[f"{p}__dates"])
    cnt = np.bincount(SEA[p], minlength=4)
    print(f"  {p:5s} n={F[p].size}  season days "
          + " ".join(f"{SEASONS[s]}={cnt[s]}" for s in range(4)))

print()
print("=" * 78)
print("2. RETAINED CCA / TARGET EOF COEFFICIENTS")
print("=" * 78)
TS = json.loads(MANIFEST_IN.read_text())["normalization_constants"]["TARGET_STD_K"]
print(f"  TARGET_STD_K = {TS!r}")

E_COEF, W_GLOBAL, GATE_OUT = {}, None, {}
for p in POPS:
    st, sc, di, dt, gt = [], [], [], [], []
    wg = []
    for s in range(N_SHARDS):
        f = np.load(MECH / "partials" / f"model_{p}_{s:02d}.npz", allow_pickle=True)
        st.append(np.asarray(f["s_true"], np.float64))
        sc.append(np.asarray(f["s_cca"], np.float64))
        di.append(np.asarray(f["day_index"], np.int64))
        dt.append(np.asarray(f["dates"]))
        gt.append(np.asarray(f["gate_cca_outside_energy_fraction"], np.float64))
        q = list(f["quantities"]); wg.append(float(f["daily"][0, 0, q.index("w")]))
    st = np.concatenate(st); sc = np.concatenate(sc)
    di = np.concatenate(di); dt = np.concatenate(dt); gt = np.concatenate(gt)
    o = np.argsort(di); st, sc, dt, gt, di = st[o], sc[o], dt[o], gt[o], di[o]
    if not (di == np.arange(di.size)).all():
        raise SystemExit(f"{p}: day_index is not a contiguous 0..n-1 cover")
    if not (dt == DATES[p]).all():
        raise SystemExit(f"{p}: shard dates disagree with the per-day file")
    if W_GLOBAL is None:
        W_GLOBAL = wg[0]
    if max(abs(w - W_GLOBAL) for w in wg) > 1e-6:
        raise SystemExit("global region weight is not constant across shards")


    E_COEF[p] = (sc - st) * (TS / math.sqrt(W_GLOBAL))
    GATE_OUT[p] = float(gt.max())
    print(f"  {p:5s} coefficients {E_COEF[p].shape}  "
          f"CCA out-of-span energy fraction max {GATE_OUT[p]:.3e}")
print(f"  global region weight W = {W_GLOBAL!r}  (H*W = {1280 * 2624})")

print()
print("=" * 78)
print("3. GATE: coefficient-space M vs retained pixel-space M")
print("=" * 78)
M_coef, coef_gate = {}, {}
for p in POPS:
    m = np.einsum("tk,tk->t", E_COEF[p], E_COEF[p])
    M_coef[p] = m
    da = float(np.abs(m - M_ret[p]).max())
    dr = float(np.abs(m / M_ret[p] - 1).max())
    coef_gate[p] = {"max_abs_daily_K2": da, "max_rel_daily": dr,
                    "mean_coef_K2": float(m.mean()),
                    "mean_retained_K2": float(M_ret[p].mean()),
                    "mean_abs_diff_K2": float(abs(m.mean() - M_ret[p].mean()))}
    print(f"  {p:5s} max|abs| {da:.3e} K2   max|rel| {dr:.3e}   "
          f"mean coef {m.mean():.15f} vs retained {M_ret[p].mean():.15f}")
print("  (this residual is the float floor of E-orthonormality + accumulation")
print("   over 3.36e6 pixels; the run's own U-Net Pythagorean gate sits at the")
print("   same order, 2.14e-10 relative)")
if max(v["max_rel_daily"] for v in coef_gate.values()) > 1e-7:
    raise SystemExit("coefficient reconstruction does not match retained M")

print()
print("=" * 78)
print("4. EXACT THREE-COMPONENT DECOMPOSITION   M = B + V")
print("=" * 78)


def bv(e):
    mu = e.mean(axis=0)
    B = float(mu @ mu)
    Mm = float(np.einsum("tk,tk->t", e, e).mean())
    V = float(np.einsum("tk,tk->t", e - mu, e - mu).mean())
    return B, V, Mm


comp = {}
for p in POPS:
    B, V, Mm = bv(E_COEF[p])
    Fm = float(F[p].mean())
    comp[p] = {"n_days": int(F[p].size), "F_K2": Fm, "B_K2": B, "V_K2": V,
               "M_coef_K2": Mm, "M_retained_K2": float(M_ret[p].mean()),
               "B_plus_V_K2": B + V,
               "closure_M_minus_B_minus_V_K2": float(Mm - B - V),
               "MSE_CCA_retained_K2": float(TOT[p].mean()),
               "F_plus_B_plus_V_K2": Fm + B + V,
               "closure_MSE_minus_F_B_V_K2": float(TOT[p].mean() - Fm - B - V),
               "B_share_of_M": B / Mm, "V_share_of_M": V / Mm}
    c = comp[p]
    print(f"  {POP_LABEL[p]:24s} F={Fm:.10f}  B={B:.10f}  V={V:.10f}")
    print(f"  {'':24s} M=B+V closes to {c['closure_M_minus_B_minus_V_K2']:+.3e} K2"
          f"   MSE=F+B+V closes to {c['closure_MSE_minus_F_B_V_K2']:+.3e} K2"
          f"   B/M={B / Mm:.5f}")

deltas = {}
for t in ("mh", "ssp"):
    deltas[t] = {k: comp[t][k] - comp["test"][k]
                 for k in ("F_K2", "B_K2", "V_K2", "M_coef_K2",
                           "MSE_CCA_retained_K2")}
    d = deltas[t]
    d["B_share_of_dM"] = d["B_K2"] / d["M_coef_K2"]
    d["V_share_of_dM"] = d["V_K2"] / d["M_coef_K2"]
    print(f"\n  CHANGE PD test -> {POP_LABEL[t]}")
    print(f"    dF={d['F_K2']:+.10f}  dB={d['B_K2']:+.10f}  dV={d['V_K2']:+.10f}"
          f"  dM={d['M_coef_K2']:+.10f}")
    print(f"    dB share of dM {d['B_share_of_dM']:+.5f}   "
          f"dV share of dM {d['V_share_of_dM']:+.5f}")

print()
print("=" * 78)
print("5. BOOTSTRAP -- two schemes")
print("=" * 78)


def counts_from_idx(idx, n):
    out = np.empty((idx.shape[0], n), np.float64)
    for b in range(idx.shape[0]):
        out[b] = np.bincount(idx[b], minlength=n)
    return out


def plain_idx(p, rng):
    return circular_block_bootstrap_indices(F[p].size, BLOCK_LEN,
                                            N_RESAMPLES, rng)


def seasonal_idx(p, rng):


    n = F[p].size
    out = np.empty((N_RESAMPLES, n), np.int64)
    col = 0
    for s in range(4):
        days = np.flatnonzero(SEA[p] == s)
        ns = days.size
        loc = circular_block_bootstrap_indices(ns, min(BLOCK_LEN, ns),
                                               N_RESAMPLES, rng)
        out[:, col:col + ns] = days[loc]
        col += ns
    if col != n:
        raise SystemExit(f"{p}: seasonal cover {col} != {n}")
    return out


def run_scheme(idx_fn, tag):
    rng = np.random.default_rng(SEED)
    st = {}
    for p in DRAW_ORDER:
        n = F[p].size
        idx = idx_fn(p, rng)
        cnt = counts_from_idx(idx, n)
        Fb = cnt @ F[p] / n
        Mb = cnt @ M_coef[p] / n
        Bb = np.empty(N_RESAMPLES)
        for s0 in range(0, N_RESAMPLES, 2000):
            s1 = min(s0 + 2000, N_RESAMPLES)
            mu = (cnt[s0:s1] @ E_COEF[p]) / n
            Bb[s0:s1] = np.einsum("bk,bk->b", mu, mu)
        st[p] = {"F": Fb, "M": Mb, "B": Bb, "V": Mb - Bb}
        del cnt, idx
    v_mh = np.stack([st["mh"]["F"] - st["test"]["F"],
                     st["mh"]["M"] - st["test"]["M"]], axis=1)
    v_ssp = np.stack([st["ssp"]["F"] - st["test"]["F"],
                      st["ssp"]["M"] - st["test"]["M"]], axis=1)
    n1 = np.linalg.norm(v_mh, axis=1); n2 = np.linalg.norm(v_ssp, axis=1)
    deg = (n1 <= EPS_DEGENERATE) | (n2 <= EPS_DEGENERATE)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.clip(np.einsum("ij,ij->i", v_mh, v_ssp) / (n1 * n2), -1, 1)
    ang = np.degrees(np.arccos(cos))
    d = {t: {k: st[t][k] - st["test"][k] for k in ("F", "B", "V", "M")}
         for t in ("mh", "ssp")}
    print(f"  [{tag}] angle median {np.median(ang[~deg]):.4f}  "
          f"95% [{np.percentile(ang[~deg], 2.5):.4f}, "
          f"{np.percentile(ang[~deg], 97.5):.4f}]  degenerate {int(deg.sum())}")
    return dict(tag=tag, ang=ang, cos=cos, deg=deg, v_mh=v_mh, v_ssp=v_ssp,
                d=d, n1=n1, n2=n2)


SCHEMES = {"plain_circular_mbb": plain_idx,
           "seasonal_within_block": seasonal_idx}
boot = {k: run_scheme(fn, k) for k, fn in SCHEMES.items()}

THRESH = [("P_angle_lt_15", lambda a: a < 15), ("P_angle_lt_30", lambda a: a < 30),
          ("P_angle_lt_45", lambda a: a < 45), ("P_angle_lt_60", lambda a: a < 60),
          ("P_60_le_angle_le_120", lambda a: (a >= 60) & (a <= 120)),
          ("P_angle_gt_90", lambda a: a > 90)]


def q(a, good=None):
    a = a if good is None else a[good]
    return {"point": None, "median": float(np.median(a)),
            "p2.5": float(np.percentile(a, 2.5)),
            "p97.5": float(np.percentile(a, 97.5)),
            "mean": float(a.mean()), "sd": float(a.std(ddof=1))}


print()
print("=" * 78)
print("6. ANGLE EXCEEDANCE PROBABILITIES")
print("=" * 78)
summary = {}
for k, b in boot.items():
    good = ~b["deg"]
    a = b["ang"][good]
    s = {"n_replicates": N_RESAMPLES, "n_degenerate": int(b["deg"].sum()),
         "n_used": int(good.sum()), "angle_deg": q(b["ang"], good),
         "cosine": q(b["cos"], good),
         "angle_min_deg": float(a.min()), "angle_max_deg": float(a.max())}
    for name, fn in THRESH:
        s[name] = float(fn(a).mean())
        s[name + "__n"] = int(fn(a).sum())
    for t in ("mh", "ssp"):
        for c in ("F", "B", "V", "M"):
            s[f"d{c}_{t}_K2"] = q(b["d"][t][c])
    summary[k] = s
    print(f"  [{k}]")
    for name, _ in THRESH:
        print(f"    {name:24s} {s[name]:.4f}")

print()
print("=" * 78)
print("7. dB / dV UNDER BOTH SCHEMES")
print("=" * 78)
for t in ("mh", "ssp"):
    print(f"  {POP_LABEL[t]}")
    for c in ("F", "B", "V"):
        pt = deltas[t][f"{c}_K2"]
        row = f"    d{c}  point {pt:+.10f}"
        for k in SCHEMES:
            s = summary[k][f"d{c}_{t}_K2"]
            row += f"   [{k[:6]}] 95% [{s['p2.5']:+.8f}, {s['p97.5']:+.8f}]"
        print(row)


STAMP = utc_stamp()
OUT = OUT_PARENT / f"transfer_mechanism_audit_{STAMP}"
OUT.mkdir(parents=True, exist_ok=False)
print()
print("=" * 78)
print(f"8. WRITING {OUT}")
print("=" * 78)

np.savez_compressed(
    OUT / "bootstrap_samples.npz",
    **{f"angle_deg__{k}": boot[k]["ang"] for k in boot},
    **{f"cosine__{k}": boot[k]["cos"] for k in boot},
    **{f"degenerate__{k}": boot[k]["deg"] for k in boot},
    **{f"v_MH__{k}": boot[k]["v_mh"] for k in boot},
    **{f"v_SSP__{k}": boot[k]["v_ssp"] for k in boot},
    **{f"d{c}_{t}__{k}": boot[k]["d"][t][c]
       for k in boot for t in ("mh", "ssp") for c in ("F", "B", "V", "M")})

src = {str(p): sha256(p) for p in (PER_DAY, PER_DAY_X, MASTER, MANIFEST_IN,
                                   T_THREE, MODEL_PASS)}
for p in POPS:
    for s in range(N_SHARDS):
        f = MECH / "partials" / f"model_{p}_{s:02d}.npz"
        src[str(f)] = sha256(f)

res = {
    "kind": "TRANSFER_MECHANISM_ANGLE_AUDIT_AND_THREE_COMPONENT_DECOMPOSITION",
    "created_utc": utc_iso(),
    "output_root": str(OUT),
    "read_only_declaration":
        "Artifact-only. No preprocessing, model inference, EOF fitting, CCA "
        "fitting, U-Net inference, normalization or model selection was rerun. "
        "No accepted artifact was modified. Inputs are retained per-day scalars "
        "and retained per-day EOF coefficients; all results are arithmetic on "
        "those plus resampling of day indices.",
    "mechanism_run_root": str(MECH),
    "calendar_convention": "native 1096-day, 29 February retained "
                           "(manuscript authority)",
    "source_files_sha256": src,
    "definitions": {
        "s_true": "a_t @ E.T, coefficients of the TRUE target residual in the "
                  "frozen orthonormal 512-mode weighted output EOF basis E "
                  "(model_pass.py); a_t = (r_true - mu_r) * sqrt_w",
        "s_cca": "a_c @ E.T, same basis, frozen CCA PREDICTED residual; the "
                 "CCA lies in the span by construction (retained out-of-span "
                 "energy fraction <= 1.7e-11)",
        "e": "(s_cca - s_true) * TARGET_STD_K / sqrt(W_global), scaled so that "
             "mean_t ||e_t||^2 IS the retained cos-latitude-weighted "
             "coefficient-mapping MSE in K^2",
        "W_global": W_GLOBAL,
        "TARGET_STD_K": TS,
        "B_c": "|| mean_t e[c,t,:] ||^2   (systematic / mean coefficient bias)",
        "V_c": "mean_t || e[c,t,:] - mean_t e[c,t,:] ||^2  (time-varying)",
        "M_c": "mean_t || e[c,t,:] ||^2 = B_c + V_c (algebraic identity)",
        "F_c": "retained selected-output-space floor MSE",
    },
    "coefficient_gate": coef_gate,
    "cca_out_of_span_energy_fraction_max": GATE_OUT,
    "components": comp,
    "changes_vs_pd_test": deltas,
    "bootstrap_design": {
        "block_length_days": BLOCK_LEN, "n_resamples": N_RESAMPLES,
        "seed": SEED, "ci": "percentile 2.5 / 97.5",
        "draw_order": list(DRAW_ORDER),
        "pd_sharing": "the SAME PD-test resample enters both transfer vectors "
                      "within a replicate; MH and SSP resamples are independent",
        "schemes": {
            "plain_circular_mbb":
                "circular overlapping moving blocks of 60 days over the whole "
                "1096-day population; generator transcribed unchanged from the "
                "run's common_defs.circular_block_bootstrap_indices",
            "seasonal_within_block":
                "blocks drawn independently WITHIN each of DJF/MAM/JJA/SON, "
                "each season's days taken in calendar order as one circular "
                "sequence, retaining the observed day count per season "
                "(so every replicate has the observed seasonal composition)"},
        "degenerate_threshold_K2": EPS_DEGENERATE,
    },
    "season_day_counts": {p: {SEASONS[s]: int((SEA[p] == s).sum())
                              for s in range(4)} for p in POPS},
    "angle_audit": summary,
    "software_environment": {
        "python": sys.version.split()[0], "numpy": np.__version__,
        "platform": platform.platform(), "node": platform.node()},
}
(OUT / "angle_audit_and_decomposition.json").write_text(
    json.dumps(res, indent=2) + "\n")


def g(x):
    return f"{x:.12g}"


L = ["population,n_days,F_K2,B_K2,V_K2,M_coef_K2,M_retained_K2,"
     "MSE_CCA_retained_K2,B_plus_V_K2,F_plus_B_plus_V_K2,"
     "closure_M_minus_B_minus_V_K2,closure_MSE_minus_F_B_V_K2,"
     "B_share_of_M,V_share_of_M"]
for p in POPS:
    c = comp[p]
    L.append(",".join([POP_LABEL[p], str(c["n_days"])] + [g(c[k]) for k in
             ("F_K2", "B_K2", "V_K2", "M_coef_K2", "M_retained_K2",
              "MSE_CCA_retained_K2", "B_plus_V_K2", "F_plus_B_plus_V_K2",
              "closure_M_minus_B_minus_V_K2", "closure_MSE_minus_F_B_V_K2",
              "B_share_of_M", "V_share_of_M")]))
L += ["", "# changes relative to PD test, with 95 % intervals under both schemes",
      "change,component,point_K2,plain_p2.5,plain_p97.5,seasonal_p2.5,"
      "seasonal_p97.5,share_of_dM"]
for t in ("mh", "ssp"):
    for c in ("F", "B", "V", "M"):
        key = "M_coef_K2" if c == "M" else f"{c}_K2"
        sh = ("" if c in ("F", "M")
              else g(deltas[t][key] / deltas[t]["M_coef_K2"]))
        L.append(",".join([
            f"PD test -> {POP_LABEL[t]}", f"d{c}", g(deltas[t][key]),
            g(summary["plain_circular_mbb"][f"d{c}_{t}_K2"]["p2.5"]),
            g(summary["plain_circular_mbb"][f"d{c}_{t}_K2"]["p97.5"]),
            g(summary["seasonal_within_block"][f"d{c}_{t}_K2"]["p2.5"]),
            g(summary["seasonal_within_block"][f"d{c}_{t}_K2"]["p97.5"]), sh]))
(OUT / "three_component_decomposition.csv").write_text("\n".join(L) + "\n")

A = ["scheme,quantity,value"]
for k in boot:
    s = summary[k]
    for name, _ in THRESH:
        A.append(f"{k},{name},{g(s[name])}")
        A.append(f"{k},{name}__n_of_{s['n_used']},{s[name + '__n']}")
    for name in ("median", "p2.5", "p97.5", "mean", "sd"):
        A.append(f"{k},angle_{name}_deg,{g(s['angle_deg'][name])}")
    A.append(f"{k},angle_min_deg,{g(s['angle_min_deg'])}")
    A.append(f"{k},angle_max_deg,{g(s['angle_max_deg'])}")
    A.append(f"{k},n_degenerate,{s['n_degenerate']}")
(OUT / "angle_audit.csv").write_text("\n".join(A) + "\n")

(OUT / ".fig.json").write_text(json.dumps(
    {"comp": comp, "deltas": deltas, "summary": summary, "out": str(OUT)},
    indent=2))
print(f"  OUT={OUT}")
