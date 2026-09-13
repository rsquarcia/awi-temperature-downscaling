"""Combine the CCA and U-Net uniform-displacement sweeps and report their responses.
Merges saved daily errors, verifies the unshifted reference results and estimates
paired 60-day block-bootstrap intervals using shared draws across models and
displacements. Writes the response figure, metrics and supporting summaries.
Requires the completed sweep outputs and referenced model/diagnostic artifacts;
it does not train or run either model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uniform_shift_defs import (ACCEPTED_CCA_RMSE_K, ACCEPTED_UNET_RMSE_K, BLOCK_LEN,
                        CI_HI_PCT, CI_LO_PCT, C, CKPT, CKPT_SHA, DELTAS,
                        GATE_TOL_K, I_ZERO, MECH_RUN, MH_D0_K, N_DAYS,
                        N_RESAMPLES, NPIX, S, SEED, SSP_D0_K,
                        circular_block_bootstrap_indices, rmse_from_day_mse)


RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "results_root")                                                  


MODELS = ("cca", "unet")
LABEL = {"cca": "CCA", "unet": "U-Net"}
COL = {"cca": "#E69F00", "unet": "#0072B2"}                                    
ACCEPTED = {"cca": ACCEPTED_CCA_RMSE_K, "unet": ACCEPTED_UNET_RMSE_K}
MM = 1 / 25.4
WIDTH_MM = 180.0
INK = "#222222"
STYLE_SRC = Path(f"{RESULTS_ROOT}/"
                 "paper_figures_final_review_20260803T142934Z/"
                 "make_transfer_figures.py")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def set_paper_style() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "figure.dpi": 120, "savefig.bbox": "tight", "pdf.fonttype": 42,
        "ps.fonttype": 42, "axes.linewidth": 0.6, "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
    })


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_day_mse(root: Path) -> tuple[dict, dict, np.ndarray]:
    out, extra = {}, {}
    za = np.load(root / "partials" / "cca_day_mse.npz")
    if not np.allclose(za["deltas"], DELTAS, atol=0, rtol=0):
        raise SystemExit("stage A displacement grid does not match")
    out["cca"] = np.asarray(za["day_mse"], dtype=np.float64)
    dates = np.asarray(za["dates"]).astype("U10")
    extra["cca_pred_change_rms_K"] = (
        np.abs(DELTAS) * float(np.sqrt(za["uu"]) * C.TARGET_STD_K
                               / np.sqrt(NPIX)))
    extra["n_feb29"] = int(np.asarray(za["is_feb29"]).sum())
    za.close()

    shards = sorted((root / "partials").glob("unet_day_mse_*.npz"))
    if not shards:
        raise SystemExit("no stage B shards found")
    mse = np.full((N_DAYS, DELTAS.size), np.nan)
    pcs = np.full((N_DAYS, DELTAS.size), np.nan)
    udates = np.empty(N_DAYS, dtype="U10")
    for p in shards:
        z = np.load(p)
        if not np.allclose(z["deltas"], DELTAS, atol=0, rtol=0):
            raise SystemExit(f"{p.name}: displacement grid does not match")
        d = np.asarray(z["days"], dtype=int)
        if np.isfinite(mse[d]).any():
            raise SystemExit(f"{p.name}: overlapping day range")
        mse[d] = z["day_mse"]
        pcs[d] = z["pred_change_sq"]
        udates[d] = np.asarray(z["dates"]).astype("U10")
        z.close()
    if not np.isfinite(mse).all():
        missing = np.where(~np.isfinite(mse[:, 0]))[0]
        raise SystemExit(f"stage B incomplete: {missing.size} days missing, "
                         f"first {missing[:5].tolist()}")
    if not np.array_equal(udates, dates):
        raise SystemExit("stage A and stage B day calendars differ")
    out["unet"] = mse
    extra["unet_pred_change_rms_K"] = np.sqrt(pcs.mean(axis=0))
    extra["n_shards"] = len(shards)
    return out, extra, dates


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    out = root
    aux = root / "outputs"                                           
    aux.mkdir(parents=True, exist_ok=True)
    gates = {}


    gates["W_SUM"] = float(S.W_SUM)
    gates["NPIX"] = int(NPIX)
    gates["weight_convention_consistent"] = bool(
        abs(S.W_SUM / NPIX - 1.0) < 1e-12)
    log(f"GATE weight convention W_SUM/NPIX - 1 = "
        f"{S.W_SUM / NPIX - 1.0:.3e} -> "
        f"{gates['weight_convention_consistent']}")
    if not gates["weight_convention_consistent"]:
        raise SystemExit("weight-normalization gate FAILED")

    day_mse, extra, dates = load_day_mse(root)
    gates["n_days"] = int(day_mse["cca"].shape[0])
    gates["n_unet_shards"] = int(extra["n_shards"])
    gates["first_date"], gates["last_date"] = str(dates[0]), str(dates[-1])
    log(f"{gates['n_days']} days {dates[0]}..{dates[-1]}, "
        f"{extra['n_shards']} U-Net shards merged")

    rmse = {m: rmse_from_day_mse(day_mse[m], axis=0) for m in MODELS}
    drmse = {m: rmse[m] - rmse[m][I_ZERO] for m in MODELS}


    j1 = int(np.argmin(np.abs(DELTAS - 1.0)))
    for m, acc in (("cca", 0.03344068906), ("unet", 0.01370119734)):
        got = float(extra[f"{m}_pred_change_rms_K"][j1])
        gates[f"{m}_pred_change_rms_at_plus1K_K"] = got
        gates[f"{m}_pred_change_rms_at_plus1K_accepted_K"] = acc
        gates[f"{m}_pred_change_rms_at_plus1K_rel_dev"] = abs(got / acc - 1.0)
        log(f"info {LABEL[m]} +1 K predicted-correction change RMS "
            f"{got:.9f} K vs accepted Section-L {acc:.9f} K "
            f"(rel dev {abs(got/acc - 1.0):.3e}"
            f"{'; accepted value is a 122-day subset' if m == 'unet' else ''})")
    if gates["cca_pred_change_rms_at_plus1K_rel_dev"] > 1e-6:
        raise SystemExit("CCA +1 K response does not reproduce Section L")

    for m in MODELS:
        r0 = float(rmse[m][I_ZERO])
        dev = abs(r0 - ACCEPTED[m])
        gates[f"{m}_rmse_at_zero_K"] = r0
        gates[f"{m}_rmse_at_zero_accepted_K"] = ACCEPTED[m]
        gates[f"{m}_rmse_at_zero_abs_dev_K"] = dev
        gates[f"{m}_rmse_at_zero_matches"] = bool(dev < GATE_TOL_K)
        log(f"GATE {LABEL[m]} RMSE(0) = {r0:.10f} K vs accepted "
            f"{ACCEPTED[m]:.10f} K (dev {dev:.3e}) -> "
            f"{gates[f'{m}_rmse_at_zero_matches']}")
    if not all(gates[f"{m}_rmse_at_zero_matches"] for m in MODELS):
        raise SystemExit("delta = 0 reproduction gate FAILED")


    log(f"bootstrap: {N_RESAMPLES} replicates, block {BLOCK_LEN} d, seed {SEED}")
    rng = np.random.default_rng(SEED)
    idx = circular_block_bootstrap_indices(N_DAYS, BLOCK_LEN, N_RESAMPLES, rng)
    gates["bootstrap_index_shape"] = list(idx.shape)
    gates["bootstrap_index_sha256"] = hashlib.sha256(
        np.ascontiguousarray(idx).tobytes()).hexdigest()
    log(f"shared day-index array {idx.shape}, sha256 "
        f"{gates['bootstrap_index_sha256'][:16]}...")

    rep = {m: np.empty((N_RESAMPLES, DELTAS.size)) for m in MODELS}
    chunk = 500
    t0 = time.time()
    for m in MODELS:
        for a in range(0, N_RESAMPLES, chunk):
            b = min(N_RESAMPLES, a + chunk)
            rep[m][a:b] = np.sqrt(day_mse[m][idx[a:b]].mean(axis=1))
        log(f"  {LABEL[m]} replicates done ({time.time()-t0:.0f} s)")
    rep_d = {m: rep[m] - rep[m][:, [I_ZERO]] for m in MODELS}

    ci = {m: np.percentile(rep[m], [CI_LO_PCT, CI_HI_PCT], axis=0)
          for m in MODELS}
    ci_d = {m: np.percentile(rep_d[m], [CI_LO_PCT, CI_HI_PCT], axis=0)
            for m in MODELS}
    med = {m: np.median(rep[m], axis=0) for m in MODELS}
    sd = {m: rep[m].std(axis=0, ddof=1) for m in MODELS}


    csv_path = out / "uniform_shift_RMSE_metrics.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "delta_K", "rmse_K", "rmse_ci_lo_K", "rmse_ci_hi_K",
                    "rmse_bootstrap_median_K", "rmse_bootstrap_sd_K",
                    "delta_rmse_K", "delta_rmse_ci_lo_K", "delta_rmse_ci_hi_K",
                    "pred_correction_change_rms_K", "n_days", "n_resamples"])
        for m in MODELS:
            pc = extra[f"{m}_pred_change_rms_K"]
            for j, c in enumerate(DELTAS):
                w.writerow([LABEL[m], f"{c:+.1f}", repr(float(rmse[m][j])),
                            repr(float(ci[m][0, j])), repr(float(ci[m][1, j])),
                            repr(float(med[m][j])), repr(float(sd[m][j])),
                            repr(float(drmse[m][j])),
                            repr(float(ci_d[m][0, j])),
                            repr(float(ci_d[m][1, j])),
                            repr(float(pc[j])), N_DAYS, N_RESAMPLES])
    log(f"wrote {csv_path.name}")


    boot = {
        "generated_utc": utc(),
        "scheme": "paired circular overlapping moving-block bootstrap over days",
        "generator": ("verbatim transcription of the accepted mechanism run's "
                      "code/common_defs.py :: circular_block_bootstrap_indices"),
        "block_length_days": BLOCK_LEN,
        "n_resamples": N_RESAMPLES,
        "seed": SEED,
        "rng": "numpy.random.default_rng(20260715), one stream",
        "n_days": N_DAYS,
        "n_blocks_per_replicate": int(np.ceil(N_DAYS / BLOCK_LEN)),
        "pairing": ("ONE day-index array of shape (10000, 1096) is drawn and "
                    "reused for every displacement value and both models, so "
                    "differences between curves are not inflated by "
                    "independent resampling noise"),
        "index_sha256": gates["bootstrap_index_sha256"],
        "interval": f"percentile [{CI_LO_PCT}, {CI_HI_PCT}]",
        "interpretation": ("temporal sampling uncertainty of the 2012-2014 "
                           "PD-test population for these FIXED frozen models "
                           "only; it is not parameter, training or structural "
                           "uncertainty"),
        "deltas_K": [float(c) for c in DELTAS],
        "results": {
            LABEL[m]: {
                "rmse_K": [float(x) for x in rmse[m]],
                "rmse_ci_lo_K": [float(x) for x in ci[m][0]],
                "rmse_ci_hi_K": [float(x) for x in ci[m][1]],
                "rmse_bootstrap_median_K": [float(x) for x in med[m]],
                "rmse_bootstrap_sd_K": [float(x) for x in sd[m]],
                "delta_rmse_K": [float(x) for x in drmse[m]],
                "delta_rmse_ci_lo_K": [float(x) for x in ci_d[m][0]],
                "delta_rmse_ci_hi_K": [float(x) for x in ci_d[m][1]],
            } for m in MODELS},
        "gates": gates,
    }
    (out / "uniform_shift_RMSE_bootstrap.json").write_text(
        json.dumps(boot, indent=2, default=lambda o: float(o)) + "\n")
    np.savez_compressed(aux / "uniform_shift_RMSE_replicates.npz",
                        deltas=DELTAS,
                        **{f"{m}_rmse_replicates": rep[m] for m in MODELS},
                        **{f"{m}_delta_rmse_replicates": rep_d[m]
                           for m in MODELS},
                        **{f"{m}_day_mse": day_mse[m] for m in MODELS},
                        dates=dates)
    log("wrote bootstrap record and replicate archive")


    set_paper_style()
    os.environ["SOURCE_DATE_EPOCH"] = "1770000000"
    fig, (axA, axB) = plt.subplots(
        2, 1, figsize=(WIDTH_MM * MM, 132 * MM), sharex=True,
        gridspec_kw={"height_ratios": [2.15, 1.0]})

    def refs(ax, ytext):
        for x, txt in ((MH_D0_K, "MH annual mean displacement"),
                       (SSP_D0_K, "SSP5-8.5 annual mean displacement")):
            ax.axvline(x, color="#777777", linestyle=(0, (5, 3)), linewidth=0.7,
                       zorder=1)
            if ytext is not None:


                ax.text(x, ytext, f"{txt}   {x:+.4f} K", rotation=90,
                        ha="center", va="center", fontsize=6.1,
                        color="#555555", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                                  edgecolor="none", alpha=0.85))
        ax.axvline(0.0, color="#BBBBBB", linewidth=1.4, zorder=0)

    for m in MODELS:
        axA.fill_between(DELTAS, ci[m][0], ci[m][1], color=COL[m], alpha=0.16,
                         linewidth=0, zorder=2)
        axA.plot(DELTAS, rmse[m], "-", color=COL[m], linewidth=1.3, zorder=4,
                 label=f"{LABEL[m]}   (RMSE at $\\delta=0$: "
                       f"{rmse[m][I_ZERO]:.6f} K)")
        axA.plot(DELTAS, rmse[m], "o", color=COL[m], markersize=2.6,
                 markeredgecolor="white", markeredgewidth=0.35, zorder=5)
        axA.plot([0.0], [rmse[m][I_ZERO]], "o", color="white",
                 markeredgecolor=COL[m], markeredgewidth=1.1, markersize=5.0,
                 zorder=6)

    lo = min(ci[m][0].min() for m in MODELS)
    hi = max(ci[m][1].max() for m in MODELS)
    pad = 0.055 * (hi - lo)
    axA.set_ylim(lo - pad, hi + 1.5 * pad)


    y_gap = 0.5 * (ci["unet"][1].max() + ci["cca"][0].min())
    refs(axA, y_gap)
    axA.set_ylabel("RMSE [K]")
    axA.set_title("Frozen-model RMSE under a uniform displacement of the "
                  "complete T2M state\nPD test 2012–2014, all 1096 days, "
                  "cosine-latitude weighted", fontsize=8.4, pad=7)
    axA.grid(axis="y", linewidth=0.4, color="#E4E4E4", zorder=0)
    axA.set_axisbelow(True)
    for s in ("top", "right"):
        axA.spines[s].set_visible(False)
    axA.legend(frameon=False, loc="upper left", handlelength=1.6,
               borderaxespad=0.3)

    for m in MODELS:
        axB.fill_between(DELTAS, ci_d[m][0], ci_d[m][1], color=COL[m],
                         alpha=0.16, linewidth=0, zorder=2)
        axB.plot(DELTAS, drmse[m], "-o", color=COL[m], linewidth=1.1,
                 markersize=2.3, markeredgecolor="white",
                 markeredgewidth=0.3, zorder=4, label=LABEL[m])
    axB.axhline(0.0, color="#444444", linewidth=0.7, zorder=3)
    refs(axB, None)
    axB.set_xlabel("Uniform T2M displacement, $\\delta$  [K]")
    axB.set_ylabel("$\\Delta$RMSE [K]")
    axB.set_title("supplementary: change relative to $\\delta = 0$",
                  fontsize=8.0, pad=5)
    axB.grid(axis="y", linewidth=0.4, color="#E4E4E4", zorder=0)
    axB.set_axisbelow(True)
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)
    axB.legend(frameon=False, loc="upper left", handlelength=1.6,
               borderaxespad=0.3)
    axB.set_xticks(np.arange(-6, 6.5, 1.0))
    axB.set_xlim(DELTAS[0] - 0.35, DELTAS[-1] + 0.35)

    fig.subplots_adjust(left=0.085, right=0.985, top=0.915, bottom=0.075,
                        hspace=0.20)
    for ax, letter in ((axA, "a"), (axB, "b")):
        pos = ax.get_position()
        fig.text(pos.x0 - 0.062, pos.y1 + 0.012, f"({letter})", ha="left",
                 va="bottom", fontsize=9, weight="bold", color=INK)

    pdf = out / "uniform_shift_RMSE.pdf"
    png = out / "uniform_shift_RMSE.png"
    fig.savefig(pdf, format="pdf")
    fig.savefig(png, format="png", dpi=600)
    plt.close(fig)
    log(f"wrote {pdf.name} and {png.name}")

    (aux / "stage_c_gates.json").write_text(
        json.dumps(gates, indent=2, default=lambda o: float(o)) + "\n")


    def rng_txt(m):
        j_lo, j_hi = int(np.argmin(rmse[m])), int(np.argmax(rmse[m]))
        return (f"minimum {rmse[m][j_lo]:.6f} K at delta = {DELTAS[j_lo]:+.1f} K, "
                f"maximum {rmse[m][j_hi]:.6f} K at delta = {DELTAS[j_hi]:+.1f} K")

    j_mh = int(np.argmin(np.abs(DELTAS - round(MH_D0_K * 2) / 2)))
    j_ssp = int(np.argmin(np.abs(DELTAS - round(SSP_D0_K * 2) / 2)))
    cap = f"""Figure. Frozen-model RMSE under a uniform displacement of the complete
temperature state.

Both the bilinear baseline field and the high-resolution target are displaced by
the same uniform amount delta, so the HR-minus-bilinear correction that either
model must predict is EXACTLY unchanged at every displacement. The diagnostic
therefore asks a single question: does shifting the absolute temperature level
change model accuracy even though the correct correction does not move?

(a) Global cosine-latitude-weighted RMSE of the frozen CCA (orange) and the
frozen adopted U-Net (blue) against the displaced target, over all
{N_DAYS} days of the native-calendar PD test population ({dates[0]} to
{dates[-1]}). Evaluated displacements are delta = {DELTAS[0]:+.1f} to
{DELTAS[-1]:+.1f} K in {DELTAS[1]-DELTAS[0]:.1f} K steps, shown as markers and
joined by straight lines; no smoothing of any kind is applied. At delta = 0
(open markers on the pale vertical line) both curves reproduce the accepted
PD-test values: CCA {rmse['cca'][I_ZERO]:.7f} K against the accepted
{ACCEPTED_CCA_RMSE_K:.7f} K, U-Net {rmse['unet'][I_ZERO]:.7f} K against the
accepted {ACCEPTED_UNET_RMSE_K:.7f} K. Shaded bands are 95 % percentile
intervals from the accepted paired 60-day circular moving-block bootstrap
({N_RESAMPLES} replicates, seed {SEED}); a single day-index array is shared by
every displacement value and by both models, so the bands describe temporal
sampling uncertainty of the {dates[0][:4]}-{dates[-1][:4]} population for these
FIXED frozen models only -- they are not parameter, training or structural
uncertainty. Dashed vertical lines mark the annual-mean T2M displacement of the
two transfer climates relative to PD: {MH_D0_K:+.4f} K (MH) and
{SSP_D0_K:+.4f} K (SSP5-8.5).

(b) Supplementary panel: the same curves as the change relative to delta = 0,
with a horizontal zero line. It repeats panel (a) on a difference axis and does
not replace it.

Over the evaluated range the CCA spans {rng_txt('cca')}, and the U-Net spans
{rng_txt('unet')}. At the nearest grid points to the two transfer-climate
displacements the RMSE changes are: CCA {drmse['cca'][j_mh]:+.6f} K and U-Net
{drmse['unet'][j_mh]:+.6f} K at delta = {DELTAS[j_mh]:+.1f} K; CCA
{drmse['cca'][j_ssp]:+.6f} K and U-Net {drmse['unet'][j_ssp]:+.6f} K at
delta = {DELTAS[j_ssp]:+.1f} K. Neither model is invariant: a displacement the
correct correction does not see still moves both curves.

This is a synthetic frozen-model sensitivity diagnostic, NOT a complete
counterfactual climate. Only the absolute temperature level of the bilinear
input and of the target is displaced; TISR, orography, land fraction and lake
fraction are passed through unchanged, and the correction field itself is held
exactly fixed. A real climate change would also alter the correction, the
circulation, the seasonal cycle and the residual field. The dashed reference
lines only locate where the two transfer climates' mean displacement falls on
this axis. They do NOT imply that uniform warming uniquely causes the models'
SSP5-8.5 behaviour, which also involves zonal and geographic redistribution and
a different residual target; no such causal claim is made here.

Both models are frozen throughout: no retraining, refitting, recalibration,
recomputed normalization, recomputed EOFs or checkpoint reselection was
performed, and no accepted artifact was modified.
"""
    (out / "CAPTION.txt").write_text(cap)
    log("wrote CAPTION.txt")


    gate_rows = "\n".join(
        f"| {LABEL[m]} RMSE at delta = 0 | {rmse[m][I_ZERO]:.10f} K | "
        f"{ACCEPTED[m]:.10f} K | {gates[f'{m}_rmse_at_zero_abs_dev_K']:.2e} K | "
        f"{'PASS' if gates[f'{m}_rmse_at_zero_matches'] else 'FAIL'} |"
        for m in MODELS)
    readme = f"""# uniform_shift_RMSE -- frozen-model uniform-displacement diagnostic

Generated {utc()}. **Frozen-model sensitivity diagnostic.** Nothing was
retrained, refitted, recalibrated or reselected; no normalization statistic and
no EOF basis was recomputed; no accepted artifact was modified. Both models were
loaded read-only and run in inference/analytic evaluation only.

## The experiment

For each displacement `delta` in {DELTAS[0]:+.1f} .. {DELTAS[-1]:+.1f} K
(step {DELTAS[1]-DELTAS[0]:.1f} K, {DELTAS.size} values):

    shifted_bilinear  = original_bilinear  + delta
    shifted_HR_target = original_HR_target + delta

so the HR-minus-bilinear correction is exactly unchanged. Population: all
{N_DAYS} days of the native-calendar PD test set ({dates[0]} .. {dates[-1]}),
{extra['n_feb29']} of which are 29 February.

**CCA.** The frozen CCA is affine in the predictor field, so its response is
exact and analytic. The uniform +1 K field (`np.ones((1280, 2624))`, the
mechanism run's own `uniform_1K` intervention) is projected onto the frozen
Kx = {C.KX} predictor EOFs and mapped through the frozen CCA matrix B to a fixed
predicted-residual score increment `u`; the prediction at displacement `delta`
is `yhat(0) + delta*u`. Frozen PD-training predictor mean (`x_score_mean`),
frozen predictor EOFs, frozen CCA mapping (`B`, `y_score_mean`), frozen target
EOFs and the frozen residual mean are all used exactly as stored. The final CCA
temperature prediction is `shifted_bilinear + predicted_CCA_correction`.

**U-Net.** `delta / NORM_T2M_INP_STD` ({C.NORM_T2M_INP_STD:.12f} K) is added to
the normalized bilinear-T2M input channel (channel 0) and nothing else: TISR,
orography, land fraction and lake fraction pass through unchanged. The adopted
frozen checkpoint is used with all frozen PD-training normalization statistics;
four identity gates (sha256 `{CKPT_SHA[:16]}...`, step, architecture,
normalization store) pass before any forward pass. The final U-Net prediction is
`shifted_bilinear + predicted_U-Net_correction`. Run as {extra['n_shards']}
contiguous day shards on one A100 each; shards are merged with a day-coverage
and calendar check.

**Metric.** The paper's global cosine-latitude-weighted RMSE,
`sqrt(weighted_mean_over_days_and_space((prediction - shifted_target)^2))`,
accumulated as a per-day weighted MSE so the bootstrap can resample days.

## Gates

| gate | value | accepted | abs. deviation | verdict |
|---|---|---|---|---|
{gate_rows}

Also checked: the cos-latitude weights are normalized to mean 1, so stage A's
`/NPIX` and stage B's `/W_SUM` conventions coincide (deviation
{abs(S.W_SUM / NPIX - 1.0):.1e}); the analytic quadratic form used for the CCA
reproduces the explicit per-day sum; and the uniform +1 K CCA predicted-change
RMS reproduces the accepted value 0.03344068906 K.

## Uncertainty

Accepted paired 60-day circular overlapping moving-block bootstrap,
{N_RESAMPLES} replicates, seed {SEED}, generator transcribed verbatim from the
mechanism run's `common_defs.circular_block_bootstrap_indices`. **One**
day-index array of shape ({N_RESAMPLES}, {N_DAYS}) is drawn and reused for every
displacement and both models, so the curves are paired and their differences are
not inflated by independent resampling noise. Index array sha256
`{gates['bootstrap_index_sha256'][:16]}...`. The intervals are temporal sampling
uncertainty of the PD-test population for these fixed models only.

## Files

```
uniform_shift_RMSE.pdf / .png        the figure (vector PDF, 600 dpi PNG)
uniform_shift_RMSE_metrics.csv       one row per model x displacement
uniform_shift_RMSE_bootstrap.json    bootstrap scheme, CIs, all gates
CAPTION.txt                          figure caption
README.md                            this file
MANIFEST.json / MANIFEST.sha256      sha256 of every source and output
outputs/uniform_shift_RMSE_replicates.npz   all replicate RMSEs + per-day MSE
outputs/stage_*_gates.json           per-stage gate records
partials/                            per-stage per-day MSE arrays
code/                                the three stages + the frozen mechanism
                                     code archive they import (code/mech)
logs/                                Slurm stdout/stderr
```

## Reading limits

Synthetic frozen-model sensitivity diagnostic, not a complete counterfactual
climate; see CAPTION.txt. The reference lines locate the two transfer climates'
mean displacement on this axis and imply no causal claim that uniform warming
uniquely produces the models' SSP5-8.5 behaviour.
"""
    (out / "README.md").write_text(readme)
    log("wrote README.md")


    sources = [
        C.CCA_MODEL, CKPT, C.PD_ZARR / ".zgroup",
        MECH_RUN / "outputs" / "code_archive.tar.gz",
        MECH_RUN / "tables" / "lr_representable_uniform_scaling.csv",
        STYLE_SRC,
    ]


    big = [C.X_BASIS, C.CLIMATE_SCORES["pd"]]
    deliverables = ["uniform_shift_RMSE.pdf", "uniform_shift_RMSE.png",
                    "uniform_shift_RMSE_metrics.csv",
                    "uniform_shift_RMSE_bootstrap.json",
                    "CAPTION.txt", "README.md",
                    "outputs/uniform_shift_RMSE_replicates.npz",
                    "outputs/stage_a_cca_gates.json",
                    "outputs/stage_c_gates.json",
                    "code/shift_defs.py", "code/stage_a_cca.py",
                    "code/stage_b_unet.py", "code/stage_c_finish.py",
                    "code/run_cpu.sbatch", "code/run_gpu.sbatch"]
    deliverables += sorted(
        str(p.relative_to(root))
        for p in (root / "outputs").glob("stage_b_gates_*.json"))
    man = {
        "kind": "uniform_shift_rmse_frozen_model_diagnostic",
        "generated_utc": utc(),
        "frozen_model_declaration":
            "No retraining, refitting, recalibration, recomputed normalization, "
            "recomputed EOFs or checkpoint reselection. No accepted artifact "
            "was created, modified or removed; all were opened read-only.",
        "population": f"PD test {dates[0]}..{dates[-1]}, {N_DAYS} native-"
                      f"calendar days",
        "displacements_K": [float(c) for c in DELTAS],
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "sources_sha256": {str(p): sha256(p) for p in sources if p.is_file()},
        "large_sources_identified_by_size_mtime": {
            str(p): {"bytes": p.stat().st_size,
                     "mtime_utc": time.strftime(
                         "%Y-%m-%dT%H:%M:%SZ", time.gmtime(p.stat().st_mtime))}
            for p in big if p.is_file()},
        "outputs_sha256": {rel: sha256(root / rel) for rel in deliverables
                           if (root / rel).is_file()},
        "gates": gates,
    }
    (out / "MANIFEST.json").write_text(
        json.dumps(man, indent=2, default=lambda o: float(o)) + "\n")
    (out / "MANIFEST.sha256").write_text("\n".join(
        f"{h}  {n}" for n, h in sorted(man["outputs_sha256"].items())) + "\n")
    log(f"wrote manifest: {len(man['outputs_sha256'])} outputs, "
        f"{len(man['sources_sha256'])} sources")
    log("stage C complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
