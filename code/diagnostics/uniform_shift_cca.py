"""Evaluate the frozen CCA response to uniform shifts of input and target temperature.
Uses the affine score-space mapping to compute prediction changes and daily
errors over the displacement grid without refitting CCA. Shifting input and
target by the same amount leaves the true target-minus-bilinear residual fixed.
Checks the unshifted error and saves results for uniform_shift_reduce.py.
Requires the saved CCA mapping, predictor basis and retained target scores."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from uniform_shift_defs import (C, S, DELTAS, I_ZERO, KX, KY, NPIX, N_DAYS, POP, TS_K,
                        ACCEPTED_CCA_RMSE_K, ACCEPTED_UNIFORM_1K_PRED_RMS_K,
                        GATE_TOL_K, MECH_RUN, rmse_from_day_mse)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    gates = {}


    mdl = np.load(C.CCA_MODEL, allow_pickle=False)
    Kx, Ky = int(mdl["Kx"]), int(mdl["Ky"])
    gates["cca_Kx"] = Kx
    gates["cca_Ky"] = Ky
    gates["cca_Kx_matches_frozen"] = (Kx == KX)
    gates["cca_Ky_matches_frozen"] = (Ky == KY)
    log(f"frozen CCA model Kx={Kx} Ky={Ky} from {C.CCA_MODEL}")


    _w, sqrt_w = C.area_weights_from_lat(S.LAT)
    sqrt_w_flat = np.repeat(sqrt_w, C.WIDTH)


    Dw = (np.ones(NPIX, dtype=np.float64) / TS_K) * sqrt_w_flat

    mm = C.npz_member_memmap(C.X_BASIS, "eofs_weighted.npy")
    dx = np.zeros(Kx, dtype=np.float64)
    t0 = time.time()
    step = 1 << 17
    for p0 in range(0, NPIX, step):
        p1 = min(NPIX, p0 + step)
        blk = np.asarray(mm[:Kx].reshape(Kx, -1)[:, p0:p1], dtype=np.float32)
        dx += (Dw[p0:p1].astype(np.float32) @ blk.T).astype(np.float64)
        del blk
    del mm
    log(f"uniform +1 K predictor projection complete in {time.time()-t0:.0f} s")

    u = dx @ np.asarray(mdl["B"])                                              
    pred_rms = float(np.linalg.norm(u) * TS_K / np.sqrt(NPIX))
    gates["uniform_1K_pred_change_rms_K"] = pred_rms
    gates["uniform_1K_pred_change_rms_accepted_K"] = ACCEPTED_UNIFORM_1K_PRED_RMS_K
    gates["uniform_1K_pred_change_rms_abs_dev"] = abs(
        pred_rms - ACCEPTED_UNIFORM_1K_PRED_RMS_K)
    gates["uniform_1K_pred_change_rms_rel_dev"] = abs(
        pred_rms / ACCEPTED_UNIFORM_1K_PRED_RMS_K - 1.0)


    gates["uniform_1K_pred_change_rms_tol_rel"] = 1e-6
    gates["uniform_1K_pred_change_rms_matches"] = bool(
        gates["uniform_1K_pred_change_rms_rel_dev"] < 1e-6)
    log(f"GATE uniform +1 K predicted-change RMS {pred_rms:.11f} K "
        f"(accepted {ACCEPTED_UNIFORM_1K_PRED_RMS_K:.11f}, rel dev "
        f"{gates['uniform_1K_pred_change_rms_rel_dev']:.3e}) -> "
        f"{gates['uniform_1K_pred_change_rms_matches']}")
    if not gates["uniform_1K_pred_change_rms_matches"]:
        raise SystemExit("uniform +1 K response gate FAILED")


    try:
        z = np.load(MECH_RUN / "outputs" / "cca_invariance_sensitivity_maps.npz")
        dm = np.asarray(z["cca_sensitivity_ssp_B_uniform_1K"], dtype=np.float64)
        z.close()
        map_rms = float(np.sqrt(S.wmeansq(dm)))
        gates["retained_sensitivity_map_rms_K"] = map_rms
        gates["retained_sensitivity_map_rel_dev"] = abs(map_rms / pred_rms - 1.0)
        log(f"cross-check vs retained cca_sensitivity map: {map_rms:.11f} K "
            f"(rel dev {gates['retained_sensitivity_map_rel_dev']:.3e})")
        del dm
    except Exception as exc:
        gates["retained_sensitivity_map_rms_K"] = None
        log(f"cross-check skipped: {exc}")


    X, xpath = S.frozen_x_scores(POP, Kx)
    log(f"frozen predictor scores {X.shape} from {xpath}")
    yhat = (X - mdl["x_score_mean"]) @ mdl["B"] + mdl["y_score_mean"]
    dat = S.load_pop(root, POP)
    yo = dat["y_scores"][:, :Ky]
    d = yhat - yo
    floor = np.asarray(dat["p_tot_weighted"] - dat["p_in_weighted"],
                       dtype=np.float64)
    dates = np.asarray(dat["dates"]).astype("U10")
    n = d.shape[0]
    gates["n_days"] = int(n)
    gates["n_days_matches"] = (n == N_DAYS)
    gates["first_date"], gates["last_date"] = str(dates[0]), str(dates[-1])
    gates["n_feb29"] = int(np.asarray(dat["is_feb29"]).sum())
    log(f"population {POP}: {n} days {dates[0]}..{dates[-1]}, "
        f"{gates['n_feb29']} Feb-29 days retained (native calendar)")


    dd = np.einsum("ij,ij->i", d, d)
    du = d @ u
    uu = float(u @ u)
    day_mse = np.empty((n, DELTAS.size), dtype=np.float64)
    for j, c in enumerate(DELTAS):
        day_mse[:, j] = ((dd + 2.0 * c * du + (c * c) * uu) + floor) \
            / NPIX * TS_K ** 2
    rmse = rmse_from_day_mse(day_mse, axis=0)


    j_probe = int(np.argmax(np.abs(DELTAS)))
    explicit = ((np.einsum("ij,ij->i",
                           d + DELTAS[j_probe] * u, d + DELTAS[j_probe] * u)
                 + floor) / NPIX * TS_K ** 2)
    gates["quadratic_form_max_abs_dev_K2"] = float(
        np.abs(explicit - day_mse[:, j_probe]).max())
    log(f"quadratic-form identity max |dev| "
        f"{gates['quadratic_form_max_abs_dev_K2']:.3e} K^2 at delta="
        f"{DELTAS[j_probe]:+.1f} K")

    r0 = float(rmse[I_ZERO])
    gates["rmse_at_zero_K"] = r0
    gates["rmse_at_zero_accepted_K"] = ACCEPTED_CCA_RMSE_K
    gates["rmse_at_zero_abs_dev_K"] = abs(r0 - ACCEPTED_CCA_RMSE_K)
    gates["rmse_at_zero_matches"] = bool(
        gates["rmse_at_zero_abs_dev_K"] < GATE_TOL_K)
    log(f"GATE CCA RMSE(delta=0) = {r0:.10f} K vs accepted "
        f"{ACCEPTED_CCA_RMSE_K:.10f} K (dev "
        f"{gates['rmse_at_zero_abs_dev_K']:.3e}) -> "
        f"{gates['rmse_at_zero_matches']}")
    if not gates["rmse_at_zero_matches"]:
        raise SystemExit("CCA delta=0 gate FAILED")


    import csv as _csv
    acc_grid = {}
    with open(MECH_RUN / "tables" / "lr_representable_uniform_scaling.csv") as fh:
        for r in _csv.DictReader(fh):
            acc_grid[float(r["shift_K"])] = float(r["rmse_test_change_K"])
    grid = []
    for c, a in sorted(acc_grid.items()):
        j = int(np.argmin(np.abs(DELTAS - c)))
        if abs(DELTAS[j] - c) > 1e-12:
            continue
        mine = float(rmse[j] - r0)
        grid.append({"shift_K": c, "drmse_K": mine, "accepted_drmse_K": a,
                     "abs_dev_K": abs(mine - a)})
        log(f"  retained grid {c:+.1f} K: dRMSE {mine:.12f} vs accepted "
            f"{a:.12f} (dev {abs(mine - a):.3e} K)")
    gates["retained_shift_grid"] = grid
    gates["retained_shift_grid_max_abs_dev_K"] = max(
        g["abs_dev_K"] for g in grid) if grid else None
    gates["retained_shift_grid_tol_K"] = 1e-6
    gates["retained_shift_grid_matches"] = bool(
        grid and gates["retained_shift_grid_max_abs_dev_K"] < 1e-6)
    log(f"GATE retained accepted shift grid ({len(grid)} points) max |dev| "
        f"{gates['retained_shift_grid_max_abs_dev_K']:.3e} K -> "
        f"{gates['retained_shift_grid_matches']}")
    if not gates["retained_shift_grid_matches"]:
        raise SystemExit("retained accepted shift-grid gate FAILED")

    for j, c in enumerate(DELTAS):
        log(f"  delta {c:+5.1f} K -> RMSE {rmse[j]:.8f} K "
            f"(dRMSE {rmse[j]-r0:+.8f})")

    np.savez(root / "partials" / "cca_day_mse.npz",
             deltas=DELTAS, day_mse=day_mse, dates=dates,
             u=u, uu=np.float64(uu), rmse=rmse,
             is_feb29=np.asarray(dat["is_feb29"]))
    import json
    (root / "outputs" / "stage_a_cca_gates.json").write_text(
        json.dumps({k: (bool(v) if isinstance(v, np.bool_) else v)
                    for k, v in gates.items()}, indent=2,
                   default=lambda o: float(o)) + "\n")
    log("stage A complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
