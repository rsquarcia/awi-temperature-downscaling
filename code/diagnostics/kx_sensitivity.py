"""Measure CCA sensitivity to the number of retained predictor EOFs.
Refits the score-space CCA mapping at reduced predictor dimensions using the
same training scores and fixed target EOF basis, then evaluates validation
error, uniform-shift response and climate-dependent mapping-error components.
Keeps the final headline model and U-Net unchanged. Requires the saved training
scores, EOF bases and climate diagnostic arrays."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np


DATA_ROOT = os.environ.get("DATA_ROOT", "data_root")                                                     
RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "results_root")                                                  


CCA_HOME = Path(f"{RESULTS_ROOT}/cca_final_run")
CCA_HEAVY = Path(f"{RESULTS_ROOT}/cca_final_run/outputs_heavy")
CCA_MODEL = CCA_HOME / "outputs/cca_grid/selected_cca_model.npz"
CCA_GRID_CSV = CCA_HOME / "outputs/cca_grid/cca_grid_results.csv"
X_BASIS = CCA_HEAVY / "eof_bases/x_input_eof_basis_maxK10592.npz"
X_BLOCKS = CCA_HEAVY / "eof_bases/stage01_work/x_input/eof_blocks"

Y_BASIS = CCA_HOME / "preserved_reanalysis_cache/eof_bases/y_residual_eof_basis_maxK1024.npz"
Y_BASIS_SHA = "d1d408be8cfa300943b678643bb710e71953f0a16bb7f03fecaf84e0f8f39b97"
TRAIN_X = CCA_HOME / "preserved_reanalysis_cache/projections/train_x_scores_maxK10592.npy"
TRAIN_Y = CCA_HOME / "preserved_reanalysis_cache/projections/train_y_scores_maxK1024.npy"

MECH = Path(f"{RESULTS_ROOT}/ssp585_pd_mh_full_mechanism_20260802T095350Z")
UNIFORM_RUN = Path(f"{RESULTS_ROOT}/uniform_shift_rmse_20260805T131319Z")
NORM_STATS = Path(f"{DATA_ROOT}/grids/norm_stats.json")

PRED_ROOTS = {
    "pd": Path(f"{RESULTS_ROOT}/paper_pipeline_pd_test_r2_20260728T130323Z"),
    "mh": Path(f"{RESULTS_ROOT}/mh_full_run_20260728T180741Z"),
    "ssp": Path(f"{RESULTS_ROOT}/ssp585_full_run_20260729T130103Z"),
}
X_SCORE_PATHS = {
    "pd": PRED_ROOTS["pd"] / "scores/test_x_scores_maxK10592.npy",
    "mh": PRED_ROOTS["mh"] / "scores/mh_x_scores_maxK10592.npy",
    "ssp": PRED_ROOTS["ssp"] / "scores/ssp585_x_scores_maxK10592.npy",
}
PD_TEST_ZARR = Path(f"{DATA_ROOT}/zarr/awi_downscaling_test.zarr")

EXPECTED_SHA = {
    "cca_model": "e1e3dc68e6e9736032c34aa2a3b93b53869c3c4752d4e850cb8a70498410a934",
    "train_x": "30f359060ee4a2bd8ee89d41bcbde9ce5fe84900ef98ce22570e3bd715ff89b7",
    "train_y": "322cb1bfad424dac24e882d0ee08391ca1222b077d17753c2f8153b5103e682a",
}

H, W = 1280, 2624
NPIX = H * W
KX_FULL, KY = 10592, 512
N_DAYS = 1096
N_TRAIN = 10593
TS = 21.627892139211994
TM = 277.8518781731243
IS = 21.581598298286263
IM = 277.8568272051984
EIG_FLOOR_REL = 1.0e-12
RIDGE = 0.0

CLIMATES = ("pd", "mh", "ssp")
POP_FOR_CLIMATE = {"pd": "test", "mh": "mh", "ssp": "ssp"}
CLIMATE_LABEL = {"pd": "PD test 2012-2014", "mh": "Mid-Holocene", "ssp": "SSP5-8.5"}


KX_GRID = [256, 512, 1024, 2048, 4096, 8192, 10592]


ACCEPTED = {
    "uniform_response_rms_K": 0.0334406926347,                                  
    "uniform_retention": 0.9978502691235478,
    "uniform_outside_rms_K": 0.04636519035280926,
    "components": {                                                
        "pd":  {"F": 0.07829307616, "M": 0.008941998961,
                "B": 0.0003099062021, "V": 0.008632092776, "RMSE": 0.2953558449},
        "mh":  {"F": 0.09086951761, "M": 0.01279367362,
                "B": 0.002333539218, "V": 0.01046013442, "RMSE": 0.3219676866},
        "ssp": {"F": 0.07364986162, "M": 0.0258301016,
                "B": 0.0168167243,  "V": 0.009013377358, "RMSE": 0.3154044439},
    },
    "ssp_mean_hr_displacement_K": 4.588,
    "mh_mean_hr_displacement_K": -2.143,
}

START = time.time()
ISSUES: list[str] = []
GATES: list[dict] = []


def log(msg: str) -> None:
    print(f"[{time.time() - START:9.1f}s] {msg}", flush=True)


def flag(msg: str) -> None:
    ISSUES.append(msg)
    print(f"[{time.time() - START:9.1f}s] *** FLAG *** {msg}", flush=True)


def gate(name: str, value, reference, tol, kind: str = "abs", fatal: bool = False):

    dev = abs(float(value) - float(reference))
    rel = dev / max(abs(float(reference)), 1e-300)
    metric = dev if kind == "abs" else rel
    ok = metric <= tol
    rec = {"gate": name, "value": float(value), "reference": float(reference),
           "abs_deviation": dev, "rel_deviation": rel, "tolerance": tol,
           "tolerance_kind": kind, "pass": bool(ok)}
    GATES.append(rec)
    status = "PASS" if ok else "FAIL"
    print(f"[{time.time() - START:9.1f}s] GATE {status} {name}: "
          f"value={value!r} ref={reference!r} abs_dev={dev:.6e} rel_dev={rel:.6e} "
          f"tol={tol:g} ({kind})", flush=True)
    if not ok:
        flag(f"GATE FAILED {name}: value={value!r} reference={reference!r} "
             f"abs_dev={dev:.6e} rel_dev={rel:.6e} tol={tol:g} ({kind})")
        if fatal:
            raise SystemExit(f"HARD GATE FAILED: {name}")
    return rec


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


LAT = -90.0 + (np.arange(H, dtype=np.float64) + 0.5) * (180.0 / H)
LON = (np.arange(W, dtype=np.float64) + 0.5) * (360.0 / W)
_W_ROW = np.cos(np.deg2rad(LAT))
_W_ROW /= _W_ROW.mean()
_SQRT_W_ROW = np.sqrt(_W_ROW)
W_FLAT = np.repeat(_W_ROW, W)
SQRT_W_FLAT = np.repeat(_SQRT_W_ROW, W)
W_SUM = float(W_FLAT.sum())
COEF_SCALE = TS / math.sqrt(W_SUM)


def sym_inv_sqrt_and_sqrt(S, ridge_alpha: float, eig_floor_rel: float):
    S = np.asarray(S, dtype=np.float64)
    k = S.shape[0]
    if ridge_alpha > 0.0:
        S = S + ridge_alpha * np.eye(k)
    evals, evecs = np.linalg.eigh((S + S.T) * 0.5)
    emax = float(evals[-1])
    if not math.isfinite(emax) or emax <= 0.0:
        raise RuntimeError(f"covariance block has non-positive max eigenvalue {emax}")
    kept = evals > emax * eig_floor_rel
    n_dropped = int(k - int(kept.sum()))
    ve = evecs[:, kept]
    lam = evals[kept]
    inv_sqrt = (ve / np.sqrt(lam)) @ ve.T
    sqrt_m = (ve * np.sqrt(lam)) @ ve.T
    diag = {"n_dropped": n_dropped, "min_eig": float(evals[0]),
            "max_eig": emax, "min_kept_eig": float(lam.min())}
    return inv_sqrt, sqrt_m, diag


def compute_score_stats(train_x64, train_y64):
    n = train_x64.shape[0]
    x_mean = train_x64.mean(axis=0)
    y_mean = train_y64.mean(axis=0)
    xc = train_x64 - x_mean[None, :]
    yc = train_y64 - y_mean[None, :]
    denom = float(n - 1)
    return {"n": n, "x_mean": x_mean, "y_mean": y_mean,
            "sxx": (xc.T @ xc) / denom,
            "sxy": (xc.T @ yc) / denom,
            "syy": (yc.T @ yc) / denom}


class CCAFitCache:
    def __init__(self, stats, eig_floor_rel):
        self.stats = stats
        self.eig_floor_rel = eig_floor_rel
        self._x_key = None
        self._x_entry = None
        self._y_entries = {}
        self._svd_key = None
        self._svd_entry = None

    def x_whitener(self, kx, alpha):
        key = (kx, alpha)
        if self._x_key != key:
            isx, _, diag = sym_inv_sqrt_and_sqrt(
                self.stats["sxx"][:kx, :kx], alpha, self.eig_floor_rel)
            self._x_key, self._x_entry = key, (isx, diag)
            log(f"WHITENER x Kx={kx} min_eig={diag['min_eig']:.6e} "
                f"max_eig={diag['max_eig']:.6e} min_kept_eig={diag['min_kept_eig']:.6e} "
                f"dropped={diag['n_dropped']}")
        return self._x_entry

    def y_whitener(self, ky, alpha):
        key = (ky, alpha)
        entry = self._y_entries.get(key)
        if entry is None:
            isy, sy_sqrt, diag = sym_inv_sqrt_and_sqrt(
                self.stats["syy"][:ky, :ky], alpha, self.eig_floor_rel)
            entry = (isy, sy_sqrt, diag)
            self._y_entries[key] = entry
            log(f"WHITENER y Ky={ky} min_eig={diag['min_eig']:.6e} "
                f"max_eig={diag['max_eig']:.6e} min_kept_eig={diag['min_kept_eig']:.6e} "
                f"dropped={diag['n_dropped']}")
        return entry

    def whitened_svd(self, kx, ky, alpha):
        key = (kx, ky, alpha)
        if self._svd_key != key:
            isx, _ = self.x_whitener(kx, alpha)
            isy, _, _ = self.y_whitener(ky, alpha)
            m = isx @ self.stats["sxy"][:kx, :ky] @ isy
            self._svd_key, self._svd_entry = key, np.linalg.svd(m, full_matrices=False)
        return self._svd_entry


def fit_cca_row(cache, kx, ky, r, ridge_alpha):
    isx, dx = cache.x_whitener(kx, ridge_alpha)
    isy, sy_sqrt, dy = cache.y_whitener(ky, ridge_alpha)
    u, s, vt = cache.whitened_svd(kx, ky, ridge_alpha)
    if not np.all(np.isfinite(s)):
        raise RuntimeError(f"non-finite canonical correlations Kx={kx} Ky={ky} r={r}")
    if np.any(np.diff(s) > 1e-12):
        raise RuntimeError(f"canonical correlations not sorted Kx={kx} Ky={ky} r={r}")
    rho_r = np.clip(s[:r], 0.0, None)
    b = (isx @ u[:, :r]) @ (rho_r[:, None] * (vt[:r, :] @ sy_sqrt))
    status = "ok"
    if dx["n_dropped"] or dy["n_dropped"]:
        status = f"ok_dropped_eigs(sxx:{dx['n_dropped']},syy:{dy['n_dropped']})"
    return {"B_map": b, "rho_full": s, "rho_retained": rho_r,
            "sxx_diag": dx, "syy_diag": dy, "fit_status": status}


def baseline_target_norm(input_norm):

    scale = IS / TS
    offset = (IM - TM) / TS
    return (np.asarray(input_norm, dtype=np.float64) * scale + offset).astype(np.float32)


def basis_block_paths():
    paths = sorted(Path(p) for p in glob.glob(str(X_BLOCKS / "eof_block_*.npy")))
    if len(paths) != 52:
        raise RuntimeError(f"expected 52 predictor EOF blocks, found {len(paths)}")
    npx = 0
    for p in paths:
        a = np.load(p, mmap_mode="r")
        if a.shape[0] != KX_FULL or a.dtype != np.float32:
            raise RuntimeError(f"bad EOF block {p}: {a.shape} {a.dtype}")
        npx += a.shape[1]
        del a
    if npx != NPIX:
        raise RuntimeError(f"EOF block pixel cover {npx} != {NPIX}")
    return paths


def project_vectors(paths, vectors_w):

    coef = np.zeros((vectors_w.shape[0], KX_FULL), dtype=np.float64)
    p0 = 0
    for ib, path in enumerate(paths):
        e = np.load(path, mmap_mode="r")
        nb = e.shape[1]
        for c0 in range(0, nb, 8192):
            c1 = min(c0 + 8192, nb)
            eb = np.asarray(e[:, c0:c1], dtype=np.float64)
            vb = np.asarray(vectors_w[:, p0 + c0:p0 + c1], dtype=np.float64)
            coef += vb @ eb.T
            del eb, vb
        p0 += nb
        log(f"predictor EOF projection block {ib + 1}/{len(paths)}")
        del e
    return coef


def _parse_npy_header(handle) -> dict:

    if handle.read(6) != b"\x93NUMPY":
        raise RuntimeError("bad npy magic")
    major = handle.read(1)
    handle.read(1)
    if major == b"\x01":
        length = int.from_bytes(handle.read(2), "little")
        preamble = 10
    else:
        length = int.from_bytes(handle.read(4), "little")
        preamble = 12
    header = eval(handle.read(length).decode("latin1"), {"__builtins__": {}})
    return {"descr": header["descr"], "fortran_order": header["fortran_order"],
            "shape": header["shape"], "header_bytes": preamble + length}


def npz_member_memmap(path: Path, member: str):

    import zipfile
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise RuntimeError(f"{path}:{member} is compressed")
        offset = info.header_offset
    with open(path, "rb") as handle:
        handle.seek(offset)
        local = handle.read(30)
        if local[:4] != b"PK\x03\x04":
            raise RuntimeError(f"bad local ZIP header in {path}")
        name_length = int.from_bytes(local[26:28], "little")
        extra_length = int.from_bytes(local[28:30], "little")
        data_offset = offset + 30 + name_length + extra_length
        handle.seek(data_offset)
        header = _parse_npy_header(handle)
    if header["fortran_order"]:
        raise RuntimeError(f"Fortran-order member unsupported: {member}")
    return np.memmap(path, dtype=np.dtype(header["descr"]), mode="r",
                     offset=data_offset + header["header_bytes"], shape=header["shape"])


def reconstruct_y_maps(score_rows):

    e = npz_member_memmap(Y_BASIS, "eofs_weighted.npy")
    out = np.empty((score_rows.shape[0], NPIX), dtype=np.float64)
    ef = e[:KY].reshape(KY, -1)
    for p0 in range(0, NPIX, 131072):
        p1 = min(p0 + 131072, NPIX)
        eb = np.asarray(ef[:, p0:p1], dtype=np.float64)
        out[:, p0:p1] = (score_rows @ eb) / SQRT_W_FLAT[p0:p1] * TS
        del eb
    del e
    return out.reshape(score_rows.shape[0], H, W)


def wrms(field) -> float:
    a = np.asarray(field, dtype=np.float64).reshape(-1)
    return float(math.sqrt(float(np.dot(a * a, W_FLAT)) / W_SUM))


def load_climate_target_scores():

    out = {}
    per_day = np.load(MECH / "outputs/exact_decomposition_per_day.npz", allow_pickle=False)
    for climate in CLIMATES:
        pop = POP_FOR_CLIMATE[climate]
        parts = [np.load(MECH / "partials" / f"model_{pop}_{i:02d}.npz", allow_pickle=False)
                 for i in range(5)]
        order = np.argsort([int(z["lo"]) for z in parts])
        parts = [parts[i] for i in order]
        day_index = np.concatenate([z["day_index"] for z in parts])
        if not np.array_equal(day_index, np.arange(N_DAYS)):
            raise RuntimeError(f"{pop}: retained model partial day cover failed")
        st = np.concatenate([z["s_true"] for z in parts]).astype(np.float64)
        sc = np.concatenate([z["s_cca"] for z in parts]).astype(np.float64)
        daily = np.concatenate([z["daily"] for z in parts], axis=0)
        quantities = list(parts[0]["quantities"].astype(str))
        regions = list(parts[0]["regions"].astype(str))
        q = {n: i for i, n in enumerate(quantities)}
        gi = regions.index("global")
        weight = float(daily[:, gi, q["w"]].sum())
        f = lambda name: float(daily[:, gi, q[name]].sum() / weight)


        out[climate] = {
            "s_true": st, "s_cca_accepted": sc,
            "F_K2": f("Qe_cca2"),
            "F_from_o_true2_K2": f("o_true2"),
            "M_accepted_pixel_space_K2": f("Pe_cca2"),
            "cca_total_mse_K2": f("e_cca2"),
            "cca_split_closure_K2": f("e_cca2") - f("Pe_cca2") - f("Qe_cca2"),
            "bilinear_mse_K2": f("r2"),
            "dates": np.concatenate([z["dates"] for z in parts]).astype(str),
        }
    per_day.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    outdir = root / "outputs"
    arrdir = root / "arrays"
    figdir = root / "figures"
    for d in (outdir, arrdir, figdir):
        d.mkdir(parents=True, exist_ok=True)

    provenance = {"utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                  "numpy": np.__version__, "python": sys.version.split()[0],
                  "root": str(root)}


    log("verifying frozen input hashes")
    for key, path in (("cca_model", CCA_MODEL), ("train_x", TRAIN_X), ("train_y", TRAIN_Y)):
        actual = sha256_file(path)
        provenance[f"sha256_{key}"] = actual
        if actual != EXPECTED_SHA[key]:
            raise SystemExit(f"HARD GATE FAILED: {key} sha256 {actual} != {EXPECTED_SHA[key]}")
        log(f"  {key} sha256 OK  {actual}")
    ns = json.loads(NORM_STATS.read_text())
    for k, (m, s) in {"t2m_inp": (IM, IS), "t2m_tgt": (TM, TS)}.items():
        if float(ns[k]["mean"]) != m or float(ns[k]["std"]) != s:
            raise SystemExit(f"HARD GATE FAILED: frozen normalization {k} drifted")
    log("  norm_stats.json OK")


    log("GATE A: loading training scores")
    train_x = np.asarray(np.load(TRAIN_X), dtype=np.float64)
    train_y = np.asarray(np.load(TRAIN_Y), dtype=np.float64)
    log(f"  train_x {train_x.shape} train_y {train_y.shape}")
    if train_x.shape != (N_TRAIN, KX_FULL):
        raise SystemExit(f"unexpected train_x shape {train_x.shape}")
    log("GATE A: computing score stats (sxx/sxy/syy, float64)")
    stats = compute_score_stats(train_x, train_y)
    cache = CCAFitCache(stats, EIG_FLOOR_REL)
    log("GATE A: refitting Kx=10592 Ky=512 r=512 ridge=0")
    fit_full = fit_cca_row(cache, KX_FULL, KY, KY, RIDGE)
    B_map_full = fit_full["B_map"]

    with np.load(CCA_MODEL, allow_pickle=False) as model:
        B_map_stored = np.asarray(model["B"], dtype=np.float64)
        x_score_mean_stored = np.asarray(model["x_score_mean"], dtype=np.float64)
        y_score_mean_stored = np.asarray(model["y_score_mean"], dtype=np.float64)
        rho_stored = np.asarray(model["canonical_correlations_full"], dtype=np.float64)

    bitwise = bool(np.array_equal(B_map_full, B_map_stored))
    max_abs = float(np.max(np.abs(B_map_full - B_map_stored)))
    ref_scale = float(np.max(np.abs(B_map_stored)))
    rel = max_abs / ref_scale
    log(f"GATE A: B_map bitwise={bitwise} max_abs_dev={max_abs:.6e} "
        f"rel_to_max={rel:.6e} (max|B_map_stored|={ref_scale:.6e})")
    GATES.append({"gate": "A_B_map_reproduction", "bitwise_identical": bitwise,
                  "max_abs_deviation": max_abs, "relative_deviation": rel,
                  "tolerance": 1e-12, "tolerance_kind": "rel",
                  "pass": bool(bitwise or rel <= 1e-12)})
    if not (bitwise or rel <= 1e-12):
        raise SystemExit(f"HARD GATE FAILED: B_map reproduction rel dev {rel:.6e} > 1e-12")
    gate("A_x_score_mean_rel",
         float(np.max(np.abs(stats["x_mean"][:KX_FULL] - x_score_mean_stored))
               / np.max(np.abs(x_score_mean_stored))), 0.0, 1e-12, "abs", fatal=True)
    gate("A_y_score_mean_rel",
         float(np.max(np.abs(stats["y_mean"][:KY] - y_score_mean_stored))
               / np.max(np.abs(y_score_mean_stored))), 0.0, 1e-12, "abs", fatal=True)
    gate("A_canonical_corr", float(np.max(np.abs(fit_full["rho_full"] - rho_stored))),
         0.0, 1e-12, "abs")
    log("GATE A PASSED")
    x_score_mean = stats["x_mean"]
    y_score_mean = stats["y_mean"][:KY]
    del train_x, train_y


    log("STAGE B: building weighted vectors for the single predictor-basis pass")
    x_mean_field = None
    import zipfile as _zf
    with _zf.ZipFile(X_BASIS) as z:
        import io as _io
        x_mean_field = np.load(_io.BytesIO(z.read("mean_field.npy"))).astype(np.float32)
        x_eigenvalues = np.load(_io.BytesIO(z.read("eigenvalues.npy"))).astype(np.float64)
        x_total_ss = float(np.load(_io.BytesIO(z.read("total_weighted_sum_squares.npy"))))
    mean_flat32 = x_mean_field.reshape(-1)
    sw32 = SQRT_W_FLAT.astype(np.float32)

    import zarr
    grp = zarr.open_group(str(PD_TEST_ZARR), mode="r")
    if grp["inputs"].shape[0] != N_DAYS:
        raise SystemExit("PD test zarr day count mismatch")
    dates_pd = np.asarray(grp["dates"][:]).astype(str)
    day_ids = [0, 548, 1095]
    log(f"  differencing days {day_ids} -> {[dates_pd[i] for i in day_ids]}")

    vectors = [(np.ones(NPIX, dtype=np.float64) / TS) * SQRT_W_FLAT]
    vector_names = ["uniform_1K_exact"]
    for d in day_ids:
        raw = np.asarray(grp["inputs"][d, 0]).reshape(-1)
        base0 = baseline_target_norm(raw)
        base1 = baseline_target_norm(raw.astype(np.float64) + (1.0 / IS))
        vectors.append(((base0 - mean_flat32) * sw32).astype(np.float64))
        vector_names.append(f"day{d}_x")
        vectors.append(((base1 - mean_flat32) * sw32).astype(np.float64))
        vector_names.append(f"day{d}_x_plus_1K")
        del raw, base0, base1
    vectors_w = np.stack(vectors)
    del vectors
    log(f"  vectors_w {vectors_w.shape} ({vectors_w.nbytes / 2**30:.2f} GiB)")

    paths = basis_block_paths()
    log("STAGE B: single streaming pass over the 52 predictor EOF blocks")
    coef = project_vectors(paths, vectors_w)
    del vectors_w

    s_uniform = coef[0].copy()
    np.save(arrdir / "s_uniform_K10592.npy", s_uniform)
    np.save(arrdir / "basis_pass_coefficients.npy", coef)
    (arrdir / "basis_pass_vector_names.json").write_text(
        json.dumps({"names": vector_names, "day_ids": day_ids,
                    "dates": [dates_pd[i] for i in day_ids]}, indent=2) + "\n")
    log(f"STAGE B: s_uniform saved -> {arrdir / 's_uniform_K10592.npy'} "
        "(the perishable basis is no longer needed except for the final netCDF)")


    accepted_day0 = np.load(X_SCORE_PATHS["pd"], mmap_mode="r")[0].astype(np.float64)
    dev_day0 = float(np.max(np.abs(coef[1] - accepted_day0)))
    rel_day0 = dev_day0 / float(np.max(np.abs(accepted_day0)))
    gate("B_projection_path_vs_retained_pd_day0", rel_day0, 0.0, 1e-6, "abs")
    log(f"  (retained PD day-0 scores are float32: max_abs_dev={dev_day0:.6e})")


    log("STAGE 1: refit sweep")
    fits = {}
    for kx in sorted(KX_GRID, reverse=True):
        r = min(kx, KY)
        t0 = time.time()
        f = fit_cca_row(cache, kx, KY, r, RIDGE) if kx != KX_FULL else fit_full
        dt = time.time() - t0
        fits[kx] = {"B_map": f["B_map"], "rho_full": f["rho_full"],
                    "rho_retained": f["rho_retained"], "r": r,
                    "rank_limited": bool(r < KY), "fit_status": f["fit_status"],
                    "sxx_diag": f["sxx_diag"], "fit_seconds": dt}
        log(f"  Kx={kx:6d} r={r:4d} rank_limited={r < KY} "
            f"rho1={f['rho_full'][0]:.10f} rho_min_ret={f['rho_retained'][-1]:.10f} "
            f"min_kept_eig={f['sxx_diag']['min_kept_eig']:.6e} "
            f"dropped={f['sxx_diag']['n_dropped']} ({dt:.1f}s)")
    del stats, cache


    import csv
    grid_rows = list(csv.DictReader(open(CCA_GRID_CSV)))
    val_rmse = {}
    for kx in KX_GRID:
        r = min(kx, KY)
        hit = [q for q in grid_rows if int(q["Kx"]) == kx and int(q["Ky"]) == KY
               and int(q["r"]) == r and float(q["ridge_alpha"]) == 0.0]
        if len(hit) != 1:
            flag(f"stored grid search has {len(hit)} rows for Kx={kx} Ky={KY} r={r}")
            val_rmse[kx] = float("nan")
        else:
            val_rmse[kx] = float(hit[0]["val_rmse_K_physical"])
    log("  stored validation RMSE (K): " +
        ", ".join(f"{k}:{val_rmse[k]:.10f}" for k in KX_GRID))


    log("STAGE 2: uniform-shift response")
    u_energy_total = W_SUM / (TS * TS)
    cum_s2 = np.cumsum(s_uniform ** 2)
    retention_curve = cum_s2 / u_energy_total
    np.save(arrdir / "uniform_retention_curve.npy", retention_curve)
    gate("uniform_retention_K10592", float(retention_curve[-1]),
         ACCEPTED["uniform_retention"], 1e-9, "rel")
    outside_rms = math.sqrt(max(u_energy_total - cum_s2[-1], 0.0)) * TS / math.sqrt(W_SUM)
    gate("uniform_outside_rms_K", outside_rms, ACCEPTED["uniform_outside_rms_K"], 1e-6, "rel")

    accepted_u = np.load(UNIFORM_RUN / "partials/cca_day_mse.npz",
                         allow_pickle=False)["u"].astype(np.float64)

    response = {}
    for kx in KX_GRID:
        c_exact = s_uniform[:kx] @ fits[kx]["B_map"]
        rms_exact = float(np.linalg.norm(c_exact)) * TS / math.sqrt(W_SUM)
        c_days = []
        for j, d in enumerate(day_ids):
            c0 = coef[1 + 2 * j][:kx]
            c1 = coef[2 + 2 * j][:kx]
            c_days.append((c1 - c0) @ fits[kx]["B_map"])
        c_days = np.stack(c_days)
        rms_days = [float(np.linalg.norm(c)) * TS / math.sqrt(W_SUM) for c in c_days]
        spread = float(np.max(np.abs(c_days - c_days.mean(axis=0)[None, :])))
        spread_rms_K = float(np.max([
            np.linalg.norm(c_days[a] - c_days[b]) for a in range(3) for b in range(a + 1, 3)
        ])) * TS / math.sqrt(W_SUM)
        dev_vs_exact = float(np.max([
            np.linalg.norm(c - c_exact) for c in c_days])) * TS / math.sqrt(W_SUM)
        response[kx] = {
            "c_exact": c_exact, "rms_exact_K": rms_exact,
            "c_days": c_days, "rms_days_K": rms_days,
            "day_spread_max_abs_score": spread,
            "day_pair_max_rms_K": spread_rms_K,
            "day_vs_exact_max_rms_K": dev_vs_exact,
            "retention": float(retention_curve[kx - 1]),
        }
        log(f"  Kx={kx:6d} response_rms={rms_exact:.12f} K  retention={retention_curve[kx-1]:.10f}"
            f"  day-diff rms={['%.12f' % v for v in rms_days]}"
            f"  max pairwise day spread={spread_rms_K:.3e} K"
            f"  max |day-exact|={dev_vs_exact:.3e} K")

    gate("uniform_response_rms_K10592_vs_accepted_score_vector",
         response[KX_FULL]["rms_exact_K"], ACCEPTED["uniform_response_rms_K"], 1e-6, "rel")
    gate("uniform_response_scores_K10592_vs_accepted_l2",
         float(np.linalg.norm(response[KX_FULL]["c_exact"] - accepted_u)
               / np.linalg.norm(accepted_u)), 0.0, 1e-6, "abs")


    log("STAGE 3: per-mode decomposition at Kx=10592")
    Bf = fits[KX_FULL]["B_map"]                                               
    row_norm2 = np.einsum("ij,ij->i", Bf, Bf)                        
    per_mode = (s_uniform ** 2) * row_norm2
    total_norm2 = float(np.dot(response[KX_FULL]["c_exact"],
                               response[KX_FULL]["c_exact"]))

    partial = np.cumsum(s_uniform[:, None] * Bf, axis=0)
    partial_norm2 = np.einsum("ij,ij->i", partial, partial)
    del partial
    cum_coherent = partial_norm2 / total_norm2
    cum_incoherent = np.cumsum(per_mode) / float(per_mode.sum())

    lam1_over_lamK = float(x_eigenvalues[0] / x_eigenvalues[KX_FULL - 1])
    k_half, k_tail10 = 5296, 9533
    stage3 = {
        "lambda_1": float(x_eigenvalues[0]),
        "lambda_10592": float(x_eigenvalues[KX_FULL - 1]),
        "lambda_1_over_lambda_10592": lam1_over_lamK,
        "smallest_retained_eigenvalue": float(x_eigenvalues[KX_FULL - 1]),
        "sxx_min_kept_eig_K10592": float(fits[KX_FULL]["sxx_diag"]["min_kept_eig"]),
        "sxx_max_eig_K10592": float(fits[KX_FULL]["sxx_diag"]["max_eig"]),
        "sxx_condition_number_K10592": float(fits[KX_FULL]["sxx_diag"]["max_eig"]
                                             / fits[KX_FULL]["sxx_diag"]["min_kept_eig"]),
        "total_response_norm2": total_norm2,
        "total_response_rms_K": response[KX_FULL]["rms_exact_K"],
        "sum_of_per_mode_contributions": float(per_mode.sum()),
        "coherence_ratio_total_over_sum_per_mode": total_norm2 / float(per_mode.sum()),
        "trailing_half_k_gt_5296": {
            "incoherent_fraction": float(per_mode[k_half:].sum() / per_mode.sum()),
            "coherent_fraction_of_response_added": float(1.0 - cum_coherent[k_half - 1]),
        },
        "trailing_10pct_k_gt_9533": {
            "incoherent_fraction": float(per_mode[k_tail10:].sum() / per_mode.sum()),
            "coherent_fraction_of_response_added": float(1.0 - cum_coherent[k_tail10 - 1]),
        },
        "leading_512_incoherent_fraction": float(per_mode[:512].sum() / per_mode.sum()),
        "leading_512_coherent_cumulative": float(cum_coherent[511]),
    }
    for k, v in stage3.items():
        log(f"  {k}: {v}")


    log("STAGE 4: transfer diagnostics")
    clim = load_climate_target_scores()
    for c in CLIMATES:
        gate(f"F_{c}_vs_accepted", clim[c]["F_K2"], ACCEPTED["components"][c]["F"], 1e-9, "abs")
        gate(f"F_{c}_Qe_vs_o_true2", clim[c]["F_K2"], clim[c]["F_from_o_true2_K2"], 1e-9, "abs")
        gate(f"cca_split_closure_{c}", clim[c]["cca_split_closure_K2"], 0.0, 1e-9, "abs")

    x_scores = {c: np.load(X_SCORE_PATHS[c], mmap_mode="r") for c in CLIMATES}
    for c in CLIMATES:
        if x_scores[c].shape != (N_DAYS, KX_FULL):
            raise SystemExit(f"{c}: x score shape {x_scores[c].shape}")

    stage4 = {}
    ebar_store = {}
    for kx in KX_GRID:
        Bk = fits[kx]["B_map"]
        for c in CLIMATES:
            xs = np.asarray(x_scores[c][:, :kx], dtype=np.float64)
            yhat = (xs - x_score_mean[None, :kx]) @ Bk + y_score_mean[None, :]
            del xs
            e = (yhat - clim[c]["s_true"]) * COEF_SCALE
            ebar = e.mean(axis=0)
            B_val = float(ebar @ ebar)
            V_val = float(np.mean(np.einsum("ij,ij->i", e - ebar, e - ebar)))
            M_val = B_val + V_val
            F_val = clim[c]["F_K2"]
            mse = F_val + M_val
            stage4[(kx, c)] = {"F_K2": F_val, "M_K2": M_val, "B_K2": B_val,
                               "V_K2": V_val, "MSE_K2": mse, "RMSE_K": math.sqrt(mse),
                               "ebar_rms_K": math.sqrt(B_val)}
            ebar_store[(kx, c)] = ebar
            if kx == KX_FULL:
                dev = float(np.max(np.abs(yhat - clim[c]["s_cca_accepted"])))
                stage4[(kx, c)]["yhat_vs_accepted_s_cca_max_abs"] = dev
                log(f"  Kx=10592 {c}: recomputed yhat vs retained s_cca max_abs_dev={dev:.6e} "
                    "(finite-basis Gram, expected ~1e-8)")
            del yhat, e
        log(f"  Kx={kx:6d} done: " + "  ".join(
            f"{c}:RMSE={stage4[(kx, c)]['RMSE_K']:.10f} B={stage4[(kx, c)]['B_K2']:.10g}"
            for c in CLIMATES))

    for c in CLIMATES:
        ref = ACCEPTED["components"][c]
        gate(f"K10592_{c}_M", stage4[(KX_FULL, c)]["M_K2"], ref["M"], 1e-9, "abs")
        gate(f"K10592_{c}_M_vs_accepted_pixel_space", stage4[(KX_FULL, c)]["M_K2"],
             clim[c]["M_accepted_pixel_space_K2"], 1e-9, "abs")
        gate(f"K10592_{c}_B", stage4[(KX_FULL, c)]["B_K2"], ref["B"], 1e-9, "abs")
        gate(f"K10592_{c}_V", stage4[(KX_FULL, c)]["V_K2"], ref["V"], 1e-9, "abs")
        gate(f"K10592_{c}_RMSE", stage4[(KX_FULL, c)]["RMSE_K"], ref["RMSE"], 1e-8, "abs")


    for c in CLIMATES:
        vals = [stage4[(kx, c)]["F_K2"] for kx in KX_GRID]
        spread = float(max(vals) - min(vals))
        gate(f"F_invariance_across_Kx_{c}_structural", spread, 0.0, 0.0, "abs")


    log("STAGE 5: persistent-bias alignment with the uniform-shift direction")
    stage5 = {}
    for kx in KX_GRID:
        d_c = response[kx]["c_exact"] * COEF_SCALE                                  
        d_norm2 = float(d_c @ d_c)
        for c in ("ssp", "mh", "pd"):
            eb = ebar_store[(kx, c)]
            eb_norm = float(math.sqrt(eb @ eb))
            ip = float(eb @ d_c)
            cos = ip / (eb_norm * math.sqrt(d_norm2)) if eb_norm > 0 else float("nan")
            stage5[(kx, c)] = {
                "ebar_rms_K": eb_norm,
                "d_rms_K": math.sqrt(d_norm2),
                "cosine": cos,
                "delta_star_K": ip / d_norm2,
                "cos2_fraction_of_B": cos * cos,
                "B_K2": float(eb @ eb),
                "B_explained_K2": (ip * ip) / d_norm2,
            }
        s = stage5[(kx, "ssp")]
        m = stage5[(kx, "mh")]
        log(f"  Kx={kx:6d} SSP cos={s['cosine']:+.8f} delta*={s['delta_star_K']:+.8f} K "
            f"cos2={s['cos2_fraction_of_B']:.8f} |ebar|={s['ebar_rms_K']:.8f} K "
            f"| MH cos={m['cosine']:+.8f} delta*={m['delta_star_K']:+.8f} K "
            f"cos2={m['cos2_fraction_of_B']:.8f} |ebar|={m['ebar_rms_K']:.8f} K")


    log("STAGE 6: writing CSVs")
    summary_cols = ["K_x", "K_y", "r", "rank_limited", "validation_rmse_K",
                    "uniform_response_rms_K", "uniform_projection_retention",
                    "first_canonical_corr", "min_canonical_corr_retained",
                    "sxx_min_kept_eig", "sxx_condition_number"]
    for c in CLIMATES:
        summary_cols += [f"{c}_rmse_K", f"{c}_F_K2", f"{c}_M_K2", f"{c}_B_K2", f"{c}_V_K2"]
    summary_cols += ["delta_B_ssp_minus_pd_K2", "delta_M_ssp_minus_pd_K2",
                     "delta_F_ssp_minus_pd_K2", "delta_B_mh_minus_pd_K2",
                     "delta_M_mh_minus_pd_K2", "delta_F_mh_minus_pd_K2"]
    with open(outdir / "kx_sensitivity_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(summary_cols)
        for kx in KX_GRID:
            row = {
                "K_x": kx, "K_y": KY, "r": fits[kx]["r"],
                "rank_limited": bool(fits[kx]["rank_limited"]),
                "validation_rmse_K": f"{val_rmse[kx]:.16g}",
                "uniform_response_rms_K": f"{response[kx]['rms_exact_K']:.16g}",
                "uniform_projection_retention": f"{response[kx]['retention']:.16g}",
                "first_canonical_corr": f"{fits[kx]['rho_full'][0]:.16g}",
                "min_canonical_corr_retained": f"{fits[kx]['rho_retained'][-1]:.16g}",
                "sxx_min_kept_eig": f"{fits[kx]['sxx_diag']['min_kept_eig']:.16g}",
                "sxx_condition_number": f"{fits[kx]['sxx_diag']['max_eig'] / fits[kx]['sxx_diag']['min_kept_eig']:.16g}",
            }
            for c in CLIMATES:
                s4 = stage4[(kx, c)]
                row[f"{c}_rmse_K"] = f"{s4['RMSE_K']:.16g}"
                row[f"{c}_F_K2"] = f"{s4['F_K2']:.16g}"
                row[f"{c}_M_K2"] = f"{s4['M_K2']:.16g}"
                row[f"{c}_B_K2"] = f"{s4['B_K2']:.16g}"
                row[f"{c}_V_K2"] = f"{s4['V_K2']:.16g}"
            for tgt in ("ssp", "mh"):
                for comp in ("B", "M", "F"):
                    row[f"delta_{comp}_{tgt}_minus_pd_K2"] = (
                        f"{stage4[(kx, tgt)][comp + '_K2'] - stage4[(kx, 'pd')][comp + '_K2']:.16g}")
            w.writerow([row[c0] for c0 in summary_cols])

    with open(outdir / "kx_permode_response.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["k", "lambda_k", "s_k", "s_k_squared",
                    "map_row_norm_squared", "per_mode_contribution",
                    "cumulative_fraction_incoherent", "cumulative_fraction_coherent",
                    "cumulative_uniform_retention"])
        for k in range(KX_FULL):
            w.writerow([k + 1, f"{x_eigenvalues[k]:.16g}", f"{s_uniform[k]:.16g}",
                        f"{s_uniform[k] ** 2:.16g}", f"{row_norm2[k]:.16g}",
                        f"{per_mode[k]:.16g}", f"{cum_incoherent[k]:.16g}",
                        f"{cum_coherent[k]:.16g}", f"{retention_curve[k]:.16g}"])

    with open(outdir / "ssp_bias_alignment.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["K_x", "r", "rank_limited", "climate", "ebar_rms_K",
                    "uniform_response_rms_K", "cosine", "delta_star_K",
                    "cos_squared_fraction_of_B", "B_K2", "B_explained_by_uniform_K2",
                    "hr_target_mean_displacement_K"])
        for kx in KX_GRID:
            for c in ("ssp", "mh", "pd"):
                s5 = stage5[(kx, c)]
                disp = {"ssp": ACCEPTED["ssp_mean_hr_displacement_K"],
                        "mh": ACCEPTED["mh_mean_hr_displacement_K"], "pd": 0.0}[c]
                w.writerow([kx, fits[kx]["r"], bool(fits[kx]["rank_limited"]),
                            CLIMATE_LABEL[c], f"{s5['ebar_rms_K']:.16g}",
                            f"{s5['d_rms_K']:.16g}", f"{s5['cosine']:.16g}",
                            f"{s5['delta_star_K']:.16g}",
                            f"{s5['cos2_fraction_of_B']:.16g}", f"{s5['B_K2']:.16g}",
                            f"{s5['B_explained_K2']:.16g}", disp])

    with open(outdir / "uniform_response_linearity_check.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["K_x", "source", "date", "response_rms_K",
                    "max_pairwise_day_difference_rms_K", "max_day_minus_exact_rms_K"])
        for kx in KX_GRID:
            rr = response[kx]
            w.writerow([kx, "analytic_uniform_field_f64", "", f"{rr['rms_exact_K']:.16g}",
                        f"{rr['day_pair_max_rms_K']:.16g}", f"{rr['day_vs_exact_max_rms_K']:.16g}"])
            for j, d in enumerate(day_ids):
                w.writerow([kx, f"difference_day{d}", dates_pd[d],
                            f"{rr['rms_days_K'][j]:.16g}",
                            f"{rr['day_pair_max_rms_K']:.16g}",
                            f"{rr['day_vs_exact_max_rms_K']:.16g}"])

    np.savez_compressed(
        arrdir / "kx_sensitivity_score_space.npz",
        s_uniform=s_uniform, eigenvalues=x_eigenvalues,
        retention_curve=retention_curve, per_mode_contribution=per_mode,
        cum_coherent=cum_coherent, cum_incoherent=cum_incoherent,
        row_norm2=row_norm2,
        **{f"response_c_Kx{kx}": response[kx]["c_exact"] for kx in KX_GRID},
        **{f"ebar_{c}_Kx{kx}": ebar_store[(kx, c)] for kx in KX_GRID for c in CLIMATES},
    )


    log("STAGE 6: figures")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6,
                         "xtick.direction": "in", "ytick.direction": "in",
                         "savefig.bbox": "tight", "pdf.fonttype": 42})

    kk = np.arange(1, KX_FULL + 1)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    ax.loglog(kk, x_eigenvalues, lw=0.8, color="#0072B2", label=r"$\lambda_k$")
    ax.set_xlabel(r"predictor mode index $k$")
    ax.set_ylabel(r"$\lambda_k$", color="#0072B2")
    ax.tick_params(axis="y", colors="#0072B2")
    ax2 = ax.twinx()
    ax2.loglog(kk, np.maximum(s_uniform ** 2, 1e-300), lw=0.6, color="#E69F00",
               alpha=0.85, label=r"$s_k^2$")
    ax2.set_ylabel(r"$s_k^2$  (uniform $+1$ K)", color="#E69F00")
    ax2.tick_params(axis="y", colors="#E69F00")
    ax.set_title(r"(a) predictor spectrum and uniform-field scores", fontsize=8)
    ax.set_xlim(1, KX_FULL)

    ax = axes[1]
    ax.semilogx(kk, cum_coherent, lw=1.0, color="#000000",
                label="coherent (nested partial sum)")
    ax.semilogx(kk, cum_incoherent, lw=1.0, color="#D55E00", ls="--",
                label="incoherent (per-mode energy)")
    ax.axvline(5296, color="#888888", lw=0.5, ls=":")
    ax.axvline(9533, color="#888888", lw=0.5, ls=":")
    ax.set_xlabel(r"predictor mode index $k$")
    ax.set_ylabel(r"cumulative fraction of $\|As\|^2$")
    ax.set_title("(b) cumulative uniform-shift response", fontsize=8)
    ax.set_xlim(1, KX_FULL)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figdir / f"kx_permode_response.{ext}", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    kxs = np.array(KX_GRID, dtype=float)
    resp = np.array([response[k]["rms_exact_K"] for k in KX_GRID])
    vr = np.array([val_rmse[k] for k in KX_GRID])
    lim = np.array([fits[k]["rank_limited"] for k in KX_GRID])
    ax.semilogx(kxs, resp, "o-", color="#E69F00", lw=1.2, ms=4,
                label="uniform $+1$ K response RMS")
    if lim.any():
        ax.semilogx(kxs[lim], resp[lim], "o", mfc="none", mec="#000000", ms=9,
                    label="rank limited ($r<K_y$)")
    ax.set_xlabel(r"$K_x$")
    ax.set_ylabel("uniform-shift response RMS [K]", color="#E69F00")
    ax.tick_params(axis="y", colors="#E69F00")
    ax.set_xticks(KX_GRID)
    ax.set_xticklabels([str(k) for k in KX_GRID], rotation=45, fontsize=6.5)
    ax.minorticks_off()
    ax2 = ax.twinx()
    ax2.semilogx(kxs, vr, "s--", color="#0072B2", lw=1.2, ms=4,
                 label="PD validation RMSE")
    ax2.set_ylabel("present-day validation RMSE [K]", color="#0072B2")
    ax2.tick_params(axis="y", colors="#0072B2")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=6.5, loc="center right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figdir / f"kx_response_vs_validation.{ext}", dpi=300)
    plt.close(fig)


    log("STAGE 7: netCDF reconstruction (final basis-dependent step)")
    import netCDF4
    y_sha = sha256_file(Y_BASIS)
    provenance["sha256_y_basis"] = y_sha
    if y_sha != Y_BASIS_SHA:
        raise SystemExit(f"HARD GATE FAILED: Y basis sha256 {y_sha} != {Y_BASIS_SHA}")
    log(f"  Y basis sha256 OK  {y_sha}")
    rows, names, longnames = [], [], []
    for kx in KX_GRID:
        rows.append(response[kx]["c_exact"])
        names.append(f"d_uniform1K_Kx{kx}")
        longnames.append(f"CCA correction response to a uniform +1 K input shift, K_x={kx}, "
                         f"r={fits[kx]['r']}" + (" (rank limited)" if fits[kx]["rank_limited"] else ""))
    for c in ("mh", "ssp"):
        rows.append(ebar_store[(KX_FULL, c)] / COEF_SCALE)
        names.append(f"ebar_{c}_Kx10592")
        longnames.append(f"time-mean within-space mapping error e^map for {CLIMATE_LABEL[c]}, K_x=10592")
    maps = reconstruct_y_maps(np.stack(rows))

    nc_path = outdir / "kx_uniform_response_and_bias_fields.nc"
    with netCDF4.Dataset(nc_path, "w", format="NETCDF4") as ds:
        ds.title = "K_x sensitivity of the CCA uniform-shift response"
        ds.summary = ("Supplementary sensitivity analysis. Frozen EOF bases; only the "
                      "canonical map is refitted at reduced K_x. K_y = 512 throughout.")
        ds.created_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ds.source_run = str(root)
        ds.area_weight_convention = ("w = cos(phi) on analytic cell centres, normalised to "
                                     f"mean one; sum(w) = {W_SUM!r}")
        ds.createDimension("latitude", H)
        ds.createDimension("longitude", W)
        v = ds.createVariable("latitude", "f8", ("latitude",))
        v.units = "degrees_north"; v[:] = LAT
        v = ds.createVariable("longitude", "f8", ("longitude",))
        v.units = "degrees_east"; v[:] = LON
        for nm, ln, arr in zip(names, longnames, maps):
            var = ds.createVariable(nm, "f8", ("latitude", "longitude"),
                                    zlib=True, complevel=4)
            var.units = "K"
            var.long_name = ln
            var.area_weighted_rms_K = wrms(arr)
            var[:, :] = arr
    log(f"  wrote {nc_path}")


    acc_map = np.load(MECH / "outputs/cca_invariance_sensitivity_maps.npz",
                      allow_pickle=False)["cca_sensitivity_ssp_B_uniform_1K"].astype(np.float64)
    d_full = maps[KX_GRID.index(KX_FULL)]
    gate("uniform_response_field_K10592_vs_retained_max_abs_K",
         float(np.max(np.abs(d_full - acc_map))), 0.0, 1e-6, "abs")
    gate("uniform_response_field_K10592_vs_retained_wrms_K",
         wrms(d_full - acc_map), 0.0, 1e-7, "abs")
    gate("uniform_response_field_K10592_wrms_vs_paper",
         wrms(d_full), ACCEPTED["uniform_response_rms_K"], 1e-6, "rel")


    results = {
        "provenance": provenance,
        "grid": {"K_x": KX_GRID, "K_y": KY,
                 "r": {str(k): fits[k]["r"] for k in KX_GRID},
                 "rank_limited": {str(k): bool(fits[k]["rank_limited"]) for k in KX_GRID}},
        "gate_A": {"bitwise_identical": bitwise, "max_abs_deviation": max_abs,
                   "relative_deviation": rel},
        "validation_rmse_K": {str(k): val_rmse[k] for k in KX_GRID},
        "uniform_response": {
            str(k): {"rms_K": response[k]["rms_exact_K"],
                     "retention": response[k]["retention"],
                     "rms_by_differencing_K": response[k]["rms_days_K"],
                     "max_pairwise_day_difference_rms_K": response[k]["day_pair_max_rms_K"],
                     "max_day_minus_exact_rms_K": response[k]["day_vs_exact_max_rms_K"]}
            for k in KX_GRID},
        "uniform_outside_rms_K": outside_rms,
        "per_mode": stage3,
        "transfer": {f"{k}|{c}": stage4[(k, c)] for k in KX_GRID for c in CLIMATES},
        "alignment": {f"{k}|{c}": stage5[(k, c)] for k in KX_GRID for c in ("ssp", "mh", "pd")},
        "gates": GATES,
        "flags": ISSUES,
        "utc_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": time.time() - START,
    }
    (outdir / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True,
                                                    default=float) + "\n")

    n_fail = sum(1 for g in GATES if not g.get("pass", True))
    log(f"DONE. gates: {len(GATES)} total, {n_fail} FAILED. flags: {len(ISSUES)}")
    for m in ISSUES:
        print("FLAG: " + m, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
