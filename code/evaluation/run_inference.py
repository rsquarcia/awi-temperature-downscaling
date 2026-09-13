"""Run the frozen U-Net and CCA for present-day, mid-Holocene, or SSP5-8.5.
Use --split test, mh, or ssp585 with each command. The three original runners
share one implementation here; dates, input paths, metadata and the additional
SSP5-8.5 window checks remain climate-specific. Model loading, normalization,
score projection and prediction precision retain the final settings. Executes
the shipped U-Net and records its module paths separately from the checkpoint's
historical training-source identity. Requires external Zarr stores, checkpoint
and fitted EOF/CCA artifacts. For direct calls, first configure_split(split)."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np


DATA_ROOT = os.environ.get("DATA_ROOT", "data_root")                                                     
MODEL_ROOT = os.environ.get("MODEL_ROOT", "model_root")                                                   
RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "results_root")                                                  


KEEPER = Path(f"{MODEL_ROOT}")
CKPT = KEEPER / "checkpoint" / "best_val_mse.pt"
CKPT_SHA = "903fab9a5c1e47299a87215785de658aa31de5594d60e3ae3c30116c8689aaa8"
CKPT_STEP = 24548
PUBLIC_UNET_ROOT = Path(__file__).resolve().parents[1] / "unet"
TRAINING_CODE_TREE_SHA = "f8b646bbd6d0e0406580503e781926747db1241ee6fcb2bcb4ef1498888e5fdb"
NORM_STATS = Path(os.environ.get(
    "NORM_STATS_PATH",
    str(Path(__file__).resolve().parents[2] / "configs/unet/norm_stats.json")))
NORM_STATS_SHA = "d372733c0e513198b7a673ced2d5ef00a59f46406563177bb47b7d45dd0c112b"
HR_STATICS = Path(f"{DATA_ROOT}/grids/static_masks/"
                  "surface_fractions_hr.nc")


CANONICAL_TEST_STORE = Path(f"{DATA_ROOT}/zarr/"
                            "awi_downscaling_test.zarr")
CANONICAL_MH_STORE = Path(f"{RESULTS_ROOT}/mh_preprocess_20260728T173229Z/"
                          "results/mh_2076_2078.zarr")
CANONICAL_SSP585_STORE = Path(
    f"{RESULTS_ROOT}/n43_ssp585_final_20260729T130103Z/results/"
    "n43_ssp585_2096_2098.zarr")

CCA_MODEL = Path(f"{RESULTS_ROOT}/cca_final_run/"
                 "outputs/cca_grid/selected_cca_model.npz")
CCA_MODEL_SHA = "e1e3dc68e6e9736032c34aa2a3b93b53869c3c4752d4e850cb8a70498410a934"
CCA_HYPER = {"Kx": 10592, "Ky": 512, "r": 512, "ridge_alpha": 0.0}
CCA_HEAVY = Path(f"{RESULTS_ROOT}/cca_final_run/"
                 "outputs_heavy")
Y_BASIS = CCA_HEAVY / "eof_bases" / "y_residual_eof_basis_maxK1024.npz"


X_BASIS = CCA_HEAVY / "eof_bases" / "x_input_eof_basis_maxK10592.npz"


SPLIT_INPUTS = {
    'test': {
        'store': CANONICAL_TEST_STORE,
        'n_days': 1096,
        'start_date': '2012-01-01',
        'end_date': '2014-12-31',
        'x_scores_rel': 'scores/test_x_scores_maxK10592.npy',
        'x_scores_shape': (1096, 10592),
    },
    'mh': {
        'store': CANONICAL_MH_STORE,
        'n_days': 1096,
        'start_date': '2076-01-01',
        'end_date': '2078-12-31',
        'x_scores_rel': 'scores/mh_x_scores_maxK10592.npy',
        'x_scores_shape': (1096, 10592),
    },
    'ssp585': {
        'store': CANONICAL_SSP585_STORE,
        'n_days': 1096,
        'start_date': '2096-01-01',
        'end_date': '2098-12-31',
        'x_scores_rel': 'scores/ssp585_x_scores_maxK10592.npy',
        'x_scores_shape': (1096, 10592),
    },
}


def resolve_x_scores_path(split_cfg: dict, output_root) -> Path:


    if "x_scores" in split_cfg:
        return Path(split_cfg["x_scores"])
    return Path(output_root) / split_cfg["x_scores_rel"]


NORM_T2M_INP_MEAN = 277.8568272051984
NORM_T2M_INP_STD = 21.581598298286263

ARCH_REQUIRED = {"latitude_padding": "pole_aware",
                 "pole_aware_boundary_scheme": "conv3x3_and_bilinear_upsample_v1",
                 "base": 96, "depth": 6}

TARGET_MEAN_K = 277.8518781731243
TARGET_STD_K = 21.627892139211994


SPLIT_SETTINGS = {'test': {'forbidden': ('midholocene',
                        'mid_holocene',
                        'awi_downscaling_mh',
                        '/mh/',
                        '_mh_',
                        'holocene'),
          'guard_message': 'BLOCKED: forbidden (sealed test / MH) path: ',
          'store_message': 'BLOCKED: only the canonical PD-test store is authorized here, '
                           'got ',
          'baseline_description': 'derived from the canonical test input with frozen TRAINING '
                                  'normalization (no legacy shards)',
          'scenario_metadata': {},
          'projection_flags': {'no_test_mean_no_refit': True, 'test_targets_read': False}},
 'mh': {'forbidden': ('awi_downscaling_test',
                      'awi_downscaling_train',
                      'awi_downscaling_val',
                      'awi_downscaling.zarr'),
        'guard_message': 'BLOCKED: forbidden (canonical PD split) path: ',
        'store_message': 'BLOCKED: only the verified canonical MH 2076-2078 store is '
                         'authorized here, got ',
        'baseline_description': 'derived from the canonical MH input with frozen TRAINING '
                                'normalization (no legacy shards)',
        'scenario_metadata': {},
        'projection_flags': {'no_mh_mean_no_refit': True,
                             'mh_targets_read': False,
                             'no_mh_derived_statistics': True}},
 'ssp585': {'forbidden': ('awi_downscaling_test',
                          'awi_downscaling_train',
                          'awi_downscaling_val',
                          'awi_downscaling.zarr',
                          'mh_2076_2078',
                          'mh_preprocess',
                          'mh_full_run',
                          'paper_pipeline_pd_test',
                          '2095_development',
                          'n43_ssp585_preprocess_'),
            'guard_message': 'BLOCKED: forbidden (PD/MH or development) path: ',
            'store_message': 'BLOCKED: only the final SSP5-8.5 2096-2098 store is authorized '
                             'here, got ',
            'baseline_description': 'derived from the canonical SSP5-8.5 input with frozen '
                                    'TRAINING normalization (no legacy shards)',
            'scenario_metadata': {'scenario': 'SSP5-8.5',
                                  'scenario_internal_id': 'ssp585',
                                  'scenario_figure_label': 'SSP5-8.5'},
            'projection_flags': {'no_ssp585_mean_no_refit': True,
                                 'ssp585_targets_read': False,
                                 'no_ssp585_derived_statistics': True,
                                 'no_recentering_on_future_climate': True}}}

ACTIVE_SPLIT = None

T0 = time.time()
GATE_PATH = Path("identity_gates.json")


def log(message: str) -> None:
    print(f"[{time.time() - T0:8.1f}s] {message}", flush=True)


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def configure_split(split: str) -> None:
    global ACTIVE_SPLIT
    if split not in SPLIT_INPUTS:
        raise ValueError(f"Unknown split: {split}")
    ACTIVE_SPLIT = split


def split_settings() -> dict:
    if ACTIVE_SPLIT is None:
        raise RuntimeError("Select a climate with configure_split(split) before direct calls.")
    return SPLIT_SETTINGS[ACTIVE_SPLIT]


def guard(path) -> Path:
    p = Path(path)
    low = str(p).lower()
    settings = split_settings()
    if any(token in low for token in settings["forbidden"]):
        raise SystemExit(f"{settings['guard_message']}{p}")
    return p


def guard_store(path) -> Path:
    p = guard(path).resolve()
    if p != SPLIT_INPUTS[ACTIVE_SPLIT]["store"].resolve():
        raise SystemExit(f"{split_settings()['store_message']}{p}")
    return p


def sha256_file(path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def jdump(obj, path: Path) -> Path:
    def default(o):
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"not JSON serializable: {type(o).__name__}")
    path = Path(path)
    tmp = path.with_name(path.name + ".writing")
    tmp.write_text(json.dumps(obj, indent=2, default=default) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)
    return path


def shard_slices(n_items: int, n_shards: int) -> list[tuple[int, int]]:


    if n_shards < 1 or n_items < n_shards:
        raise SystemExit(f"cannot split {n_items} items into {n_shards} shards")
    base, rem = divmod(n_items, n_shards)
    out, start = [], 0
    for i in range(n_shards):
        size = base + (1 if i < rem else 0)
        out.append((start, start + size))
        start += size
    if start != n_items:
        raise SystemExit("shard partition does not cover every item")
    return out


def parse_npy_header(handle) -> dict:
    if handle.read(6) != b"\x93NUMPY":
        raise SystemExit("bad npy magic")
    major = handle.read(1)
    handle.read(1)
    if major == b"\x01":
        hlen = int.from_bytes(handle.read(2), "little")
        preamble = 10
    else:
        hlen = int.from_bytes(handle.read(4), "little")
        preamble = 12
    header = eval(handle.read(hlen).decode("latin1"))                          
    return {"descr": header["descr"], "fortran_order": header["fortran_order"],
            "shape": header["shape"], "header_bytes": preamble + hlen}


def npy_memmap(path: Path):
    with open(path, "rb") as fh:
        info = parse_npy_header(fh)
    if info["fortran_order"]:
        raise SystemExit(f"fortran order not supported: {path}")
    return np.memmap(path, dtype=np.dtype(info["descr"]), mode="r",
                     offset=info["header_bytes"], shape=info["shape"])


def npz_member_info(path: Path, member: str) -> dict:
    with zipfile.ZipFile(path) as z:
        info = z.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise SystemExit(f"{member} is compressed; cannot memmap")
        offset = info.header_offset
    with open(path, "rb") as fh:
        fh.seek(offset)
        local = fh.read(30)
        if local[:4] != b"PK\x03\x04":
            raise SystemExit("bad local zip header")
        name_len = int.from_bytes(local[26:28], "little")
        extra_len = int.from_bytes(local[28:30], "little")
        data_offset = offset + 30 + name_len + extra_len
        fh.seek(data_offset)
        head = parse_npy_header(fh)
    head["data_offset"] = data_offset + head["header_bytes"]
    return head


def npz_member_memmap(path: Path, member: str):
    head = npz_member_info(path, member)
    if head["fortran_order"]:
        raise SystemExit(f"fortran order not supported: {member}")
    return np.memmap(path, dtype=np.dtype(head["descr"]), mode="r",
                     offset=head["data_offset"], shape=head["shape"])


def npz_small_member(path: Path, member: str, cap: int = 1 << 26):
    with zipfile.ZipFile(path) as z:
        if z.getinfo(member).file_size > cap:
            raise SystemExit(f"{member} too large for a direct read")
        payload = z.read(member)
    return np.load(io.BytesIO(payload), allow_pickle=False)


def new_gates(mode: str) -> dict:
    return {"created_utc": utc(), "mode": mode, "checks": []}


def gate(gates: dict, name: str, ok: bool, observed, expected, extra=None) -> None:
    entry = {"check": name, "pass": bool(ok), "observed": observed,
             "expected": expected}
    if extra:
        entry.update(extra)
    gates["checks"].append(entry)
    jdump(gates, GATE_PATH)
    if not ok:
        raise SystemExit(f"IDENTITY GATE FAILED: {name}: observed {observed!r} "
                         f"!= expected {expected!r}")


def gate_b2_identity(gates: dict) -> None:
    observed = sha256_file(CKPT)
    gate(gates, "b2_checkpoint_sha256", observed == CKPT_SHA, observed,
         CKPT_SHA, {"path": str(CKPT)})
    log("B2 checkpoint SHA-256 OK")
    observed = sha256_file(NORM_STATS)
    gate(gates, "normalization_sha256", observed == NORM_STATS_SHA, observed,
         NORM_STATS_SHA, {"path": str(NORM_STATS)})


def gate_cca_identity(gates: dict) -> dict:
    observed = sha256_file(CCA_MODEL)
    gate(gates, "cca_model_sha256", observed == CCA_MODEL_SHA, observed,
         CCA_MODEL_SHA, {"path": str(CCA_MODEL)})
    log("frozen CCA model SHA-256 OK")
    with np.load(CCA_MODEL, allow_pickle=False) as m:
        kx, ky, r = int(m["Kx"]), int(m["Ky"]), int(m["r"])
        ridge = float(m["ridge_alpha"])
        B = np.asarray(m["B"], dtype=np.float64)
        x_mean = np.asarray(m["x_score_mean"], dtype=np.float64)
        y_mean = np.asarray(m["y_score_mean"], dtype=np.float64)
    observed = {"Kx": kx, "Ky": ky, "r": r, "ridge_alpha": ridge}
    gate(gates, "cca_hyperparameters", observed == CCA_HYPER, observed, CCA_HYPER)
    gate(gates, "cca_B_shape", B.shape == (kx, ky), list(B.shape), [kx, ky])
    gate(gates, "cca_model_finite",
         bool(np.isfinite(B).all() and np.isfinite(x_mean).all()
              and np.isfinite(y_mean).all()), True, True)
    return {"Kx": kx, "Ky": ky, "r": r, "ridge_alpha": ridge, "B": B,
            "x_score_mean": x_mean, "y_score_mean": y_mean}


def store_dates(store: Path) -> list[str]:
    import zarr
    group = zarr.open_group(str(store), mode="r")
    return [d.decode() if isinstance(d, bytes) else str(d)
            for d in np.asarray(group["dates"][:]).tolist()]


def select_dates(all_dates, start, end, gates):
    gate(gates, "store_dates_chronologically_sorted",
         all_dates == sorted(all_dates), "sorted", "sorted")
    picked = [(i, d) for i, d in enumerate(all_dates) if start <= d <= end]
    if not picked:
        raise SystemExit(f"no dates in [{start}, {end}]")
    dates = [d for _, d in picked]
    gate(gates, "requested_dates_unique", len(dates) == len(set(dates)),
         len(dates), len(set(dates)))
    return [i for i, _ in picked], dates


def hr_coordinates():
    import netCDF4
    with netCDF4.Dataset(HR_STATICS, "r") as ds:
        lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
    return lat, lon, ((lon + 180.0) % 360.0) - 180.0


def load_b2(device: str, gates: dict):
    import torch

    sys.path.insert(0, str(PUBLIC_UNET_ROOT))
    import importlib
    stage = importlib.import_module("training.global_unet_second_stage")
    unet = importlib.import_module("training.global_unet")
    dataset = importlib.import_module("dataloader.zarr_dataset")
    for module in (stage, unet, dataset):
        observed = Path(module.__file__).resolve()
        expected = PUBLIC_UNET_ROOT / (module.__name__.replace(".", "/") + ".py")
        gate(gates, f"public_module_{module.__name__}",
             observed == expected, str(observed), str(expected))
    stage.configure_fp32()

    checkpoint = torch.load(CKPT, map_location="cpu", weights_only=False)
    gate(gates, "b2_checkpoint_step", int(checkpoint.get("step", -1)) == CKPT_STEP,
         int(checkpoint.get("step", -1)), CKPT_STEP)
    config = checkpoint["config"]
    training_sha = config["provenance"]["code_snapshot"]["tree_sha256"]
    gate(gates, "b2_historical_training_code_tree_sha256",
         training_sha == TRAINING_CODE_TREE_SHA, training_sha,
         TRAINING_CODE_TREE_SHA)
    model_arguments = config.get("model_arguments") or checkpoint["model_arguments"]
    model = unet.GlobalUNet(**model_arguments)
    architecture = model.architecture_dict()
    gate(gates, "b2_architecture_matches_checkpoint",
         checkpoint.get("architecture") == architecture, "architecture_dict()",
         "checkpoint['architecture']")
    for key, want in ARCH_REQUIRED.items():
        gate(gates, f"b2_architecture_{key}", architecture.get(key) == want,
             architecture.get(key), want)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    gate(gates, "b2_normalization_matches_checkpoint",
         stage.load_normalization_statistics(NORM_STATS) == config["normalization"],
         str(NORM_STATS), "checkpoint config['normalization']")
    tgt_std = float(config["normalization"]["t2m_tgt"]["std"])
    tgt_mean = float(config["normalization"]["t2m_tgt"]["mean"])
    gate(gates, "b2_target_normalization_constants",
         abs(tgt_std - TARGET_STD_K) < 1e-9 and abs(tgt_mean - TARGET_MEAN_K) < 1e-9,
         [tgt_mean, tgt_std], [TARGET_MEAN_K, TARGET_STD_K])
    model = model.to(device)
    log(f"B2 model loaded on {device}")
    return stage, model, tgt_mean, tgt_std, architecture


def run_unet(args) -> int:
    global GATE_PATH
    import torch

    store = guard_store(args.input_store)
    out_root = guard(args.output_root)
    (out_root / "predictions" / "b2").mkdir(parents=True, exist_ok=True)
    (out_root / "coords").mkdir(parents=True, exist_ok=True)
    GATE_PATH = out_root / "identity_gates_unet.json"
    gates = new_gates("unet")
    for path in (CKPT, PUBLIC_UNET_ROOT, NORM_STATS, HR_STATICS, store):
        guard(path)
        if not Path(path).exists():
            raise SystemExit(f"missing required input: {path}")

    gate_b2_identity(gates)
    indices, dates = select_dates(store_dates(store), args.start_date,
                                  args.end_date, gates)
    if args.split == "ssp585":
        gate(gates, "ssp585_population_is_1096_days",
             len(dates) == SPLIT_INPUTS[args.split]["n_days"],
             len(dates), SPLIT_INPUTS[args.split]["n_days"])
        gate(gates, "ssp585_window_endpoints",
             dates[0] == SPLIT_INPUTS[args.split]["start_date"]
             and dates[-1] == SPLIT_INPUTS[args.split]["end_date"],
             [dates[0], dates[-1]],
             [SPLIT_INPUTS[args.split]["start_date"],
              SPLIT_INPUTS[args.split]["end_date"]])
    log(f"{len(dates)} date(s) selected: {dates[0]} .. {dates[-1]}")

    lat, lon, lon180 = hr_coordinates()
    np.save(out_root / "coords" / "hr_lat.npy", lat)
    np.save(out_root / "coords" / "hr_lon.npy", lon)
    np.save(out_root / "coords" / "hr_lon180.npy", lon180)
    height, width = int(lat.size), int(lon.size)

    stage, model, tgt_mean, tgt_std, architecture = load_b2(args.device, gates)
    dataset = stage.AWIDownscalingZarrDataset(store, include_static_inputs=True)

    parts = shard_slices(len(dates), args.n_shards)
    meta = {
        "kind": "PAPER_PIPELINE_RETENTION",
        "created_utc": utc(), "split": args.split, "input_store": str(store),
        **split_settings()["scenario_metadata"],
        "start_date": args.start_date, "end_date": args.end_date,
        "dates": dates, "n_days": len(dates),
        "store_indices": indices,
        "n_shards": args.n_shards,
        "shards": [{"shard": i, "lo": lo, "hi": hi, "n_days": hi - lo,
                    "first_date": dates[lo], "last_date": dates[hi - 1]}
                   for i, (lo, hi) in enumerate(parts)],
        "grid": {"height": height, "width": width,
                 "coords": {"hr_lat": "coords/hr_lat.npy",
                            "hr_lon": "coords/hr_lon.npy",
                            "hr_lon180": "coords/hr_lon180.npy"}},
        "normalization": {"t2m_tgt_mean_K": tgt_mean, "t2m_tgt_std_K": tgt_std,
                          "physical_units": "field_K = mean + field_norm * std"},
        "identity": {
            "b2_checkpoint": str(CKPT), "b2_checkpoint_sha256": CKPT_SHA,
            "b2_step": CKPT_STEP,
            "b2_public_implementation_root": str(PUBLIC_UNET_ROOT),
            "b2_training_code_tree_sha256": TRAINING_CODE_TREE_SHA,
            "b2_normalization_sha256": NORM_STATS_SHA,
            "b2_architecture": {k: architecture.get(k) for k in
                                ("base", "depth", "norm", "latitude_padding",
                                 "pole_aware_boundary_scheme")},
            "cca_model": str(CCA_MODEL), "cca_model_sha256": CCA_MODEL_SHA,
            "cca_hyperparameters": CCA_HYPER,
            "cca_x_scores": str(resolve_x_scores_path(SPLIT_INPUTS[args.split],
                                                      out_root)),
            "cca_y_basis": str(Y_BASIS),
            "cca_baseline": split_settings()["baseline_description"],
            "cca_predictor_scores": "projected onto the frozen training "
                                    "x_input EOF basis by project-scores",
            "x_basis": str(X_BASIS),
            "cca_reconstruction":
                "yhat = (x[:Kx] - x_score_mean) @ B + y_score_mean; "
                "resid_hat = mean_field + (yhat @ eofs_weighted[:Ky]) / sqrt_w; "
                "cca_prediction = baseline + resid_hat  (normalized units)"},
        "native_precision": {"b2": "float32 (model output)",
                             "cca": "float64 (frozen pipeline)"},
        "device": args.device,
        "b2_shards_complete": [],
    }
    jdump(meta, out_root / "retention_metadata.json")
    (out_root / "dates.txt").write_text("\n".join(dates) + "\n", encoding="utf-8")

    for s, (lo, hi) in enumerate(parts):
        sdir = out_root / "predictions" / "b2" / f"shard_{s:02d}"
        sdir.mkdir(parents=True, exist_ok=True)
        n = hi - lo
        pred_path = sdir / "b2_prediction_norm.npy"
        base_path = sdir / "b2_baseline_norm.npy"
        pred_tmp = pred_path.with_suffix(".writing.npy")
        base_tmp = base_path.with_suffix(".writing.npy")
        pred_mm = np.lib.format.open_memmap(
            pred_tmp, mode="w+", dtype=np.float32, shape=(n, height, width))
        base_mm = np.lib.format.open_memmap(
            base_tmp, mode="w+", dtype=np.float32, shape=(n, height, width))
        for k in range(n):
            day = indices[lo + k]
            cpu_inputs, _ = dataset[day]
            with torch.inference_mode():
                inputs = cpu_inputs[None].to(args.device)
                prediction = model(inputs)
                baseline = model.baseline_in_target_units(inputs)
                p = prediction[0, 0].to("cpu").numpy()
                b = baseline[0, 0].to("cpu").numpy()
            if p.dtype != np.float32 or b.dtype != np.float32:
                raise SystemExit(f"unexpected B2 output dtype {p.dtype}/{b.dtype}")
            if not (np.isfinite(p).all() and np.isfinite(b).all()):
                raise SystemExit(f"non-finite B2 output on {dates[lo + k]}")
            pred_mm[k] = p
            base_mm[k] = b
        pred_mm.flush()
        base_mm.flush()
        del pred_mm, base_mm
        os.replace(pred_tmp, pred_path)
        os.replace(base_tmp, base_path)
        (sdir / "dates.txt").write_text("\n".join(dates[lo:hi]) + "\n",
                                        encoding="utf-8")
        entry = {"shard": s, "lo": lo, "hi": hi, "n_days": n,
                 "dates": dates[lo:hi],
                 "arrays": {
                     "b2_prediction_norm": {
                         "file": f"b2/shard_{s:02d}/b2_prediction_norm.npy",
                         "shape": [n, height, width], "dtype": "float32"},
                     "b2_baseline_norm": {
                         "file": f"b2/shard_{s:02d}/b2_baseline_norm.npy",
                         "shape": [n, height, width], "dtype": "float32"}}}
        jdump(entry, sdir / "shard_done.json")
        meta["b2_shards_complete"].append(s)
        jdump(meta, out_root / "retention_metadata.json")
        log(f"B2 shard {s:02d} retained ({n} days, {dates[lo]} .. {dates[hi-1]})")

    print(json.dumps({"status": "PASS", "mode": "unet",
                      "output_root": str(out_root), "n_days": len(dates),
                      "n_shards": args.n_shards,
                      "elapsed_s": round(time.time() - T0, 1)}))
    return 0


def baseline_from_inputs(input_t2m_norm: np.ndarray, tgt_mean: float,
                         tgt_std: float) -> np.ndarray:


    scale = NORM_T2M_INP_STD / tgt_std
    offset = (NORM_T2M_INP_MEAN - tgt_mean) / tgt_std
    return np.asarray(input_t2m_norm, dtype=np.float64) * scale + offset


def run_project_scores(args) -> int:
    global GATE_PATH
    out_root = guard(args.output_root)
    meta_path = out_root / "retention_metadata.json"
    if not meta_path.exists():
        raise SystemExit(f"missing {meta_path}; run the unet mode first")
    meta = json.loads(meta_path.read_text())
    split_cfg = SPLIT_INPUTS[args.split]
    store = guard_store(meta["input_store"])
    (out_root / "scores").mkdir(parents=True, exist_ok=True)
    GATE_PATH = out_root / "identity_gates_projection.json"
    gates = new_gates("project-scores")

    if not X_BASIS.exists():
        raise SystemExit(f"missing frozen predictor EOF basis: {X_BASIS}")
    mean_field = np.asarray(npz_small_member(X_BASIS, "mean_field.npy"),
                            dtype=np.float32)
    eofs = npz_member_memmap(X_BASIS, "eofs_weighted.npy")
    height, width = mean_field.shape
    kx = CCA_HYPER["Kx"]
    gate(gates, "x_basis_shape_usable",
         eofs.shape[0] >= kx and eofs.shape[1:] == (height, width),
         list(eofs.shape), [f">={kx}", height, width],
         {"path": str(X_BASIS), "dtype": str(eofs.dtype)})
    gate(gates, "x_basis_grid_matches_retention",
         (height, width) == (meta["grid"]["height"], meta["grid"]["width"]),
         [height, width], [meta["grid"]["height"], meta["grid"]["width"]])


    _w, sqrt_w = area_weights(height)
    basis_lat = np.asarray(npz_small_member(X_BASIS, "latitude.npy"),
                           dtype=np.float64)
    formula_lat = -90.0 + (np.arange(height, dtype=np.float64) + 0.5) * (180.0 / height)
    gate(gates, "basis_latitude_matches_grid_formula",
         float(np.max(np.abs(basis_lat - formula_lat))) <= 1e-4,
         float(np.max(np.abs(basis_lat - formula_lat))), "<= 1e-4 deg")
    sqrt_w_flat = np.repeat(sqrt_w, width)
    mean_flat = mean_field.reshape(-1)

    import zarr
    group = zarr.open_group(str(store), mode="r")
    n_days = meta["n_days"]
    indices = meta["store_indices"]
    tgt_mean = float(meta["normalization"]["t2m_tgt_mean_K"])
    tgt_std = float(meta["normalization"]["t2m_tgt_std_K"])
    pixels = height * width


    log(f"building baseline field for {n_days} days")
    fields = np.empty((n_days, pixels), dtype=np.float32)
    for k, si in enumerate(indices):
        inp = np.asarray(group["inputs"][si, 0], dtype=np.float32)
        fields[k] = baseline_from_inputs(inp, tgt_mean, tgt_std).astype(
            np.float32).reshape(-1)
        if (k + 1) % 100 == 0 or k == n_days - 1:
            log(f"baseline {k + 1}/{n_days}")

    scores = np.zeros((n_days, kx), dtype=np.float64)
    block = int(args.block_pixels)
    mm_chunk = int(args.matmul_chunk_pixels)
    for p0 in range(0, pixels, block):
        p1 = min(pixels, p0 + block)
        a_block = (fields[:, p0:p1] - mean_flat[None, p0:p1]) * sqrt_w_flat[None, p0:p1].astype(np.float32)
        e_block = np.asarray(eofs.reshape(eofs.shape[0], -1)[:kx, p0:p1],
                             dtype=np.float32)
        for c0 in range(0, p1 - p0, mm_chunk):
            c1 = min(p1 - p0, c0 + mm_chunk)
            scores += (a_block[:, c0:c1].astype(np.float64)
                       @ e_block[:, c0:c1].astype(np.float64).T)
        del a_block, e_block
        log(f"projected pixels [{p0},{p1}) of {pixels}")
    del fields

    if not np.isfinite(scores).all():
        raise SystemExit("non-finite predictor scores")
    if "x_scores" in split_cfg:
        raise SystemExit("project-scores must never write to a split whose "
                         "scores are a frozen absolute x_scores artifact")
    out = resolve_x_scores_path(split_cfg, out_root)
    tmp = out.with_suffix(".writing.npy")
    np.save(tmp, scores.astype(np.float32))
    os.replace(tmp, out)
    gate(gates, "test_x_scores_shape",
         tuple(np.load(out, mmap_mode="r").shape) == tuple(split_cfg["x_scores_shape"]),
         list(np.load(out, mmap_mode="r").shape),
         list(split_cfg["x_scores_shape"]), {"path": str(out)})
    jdump({"kind": "PREDICTOR_PROJECTION", "created_utc": utc(),
           "formula": "score = ((baseline - train_mean) * sqrt_weight) . "
                      "eofs_weighted^T",
           "frozen_basis": str(X_BASIS), "Kx": kx,
           "train_mean_member": "mean_field.npy (frozen Stage-01 x_input basis)",
           "sqrt_weight": "sqrt(cos(lat_center)/mean(cos(lat_center))), grid formula",
           "baseline_definition": "t2m_inp_norm * (std_inp/std_tgt) + "
                                  "(mean_inp - mean_tgt)/std_tgt, frozen "
                                  "TRAINING normalization only",
           **split_settings()["projection_flags"],
           "output": str(out), "output_sha256": sha256_file(out)},
          out_root / "scores" / "projection_provenance.json")
    log(f"predictor scores retained -> {out}")
    print(json.dumps({"status": "PASS", "mode": "project-scores",
                      "shape": list(split_cfg["x_scores_shape"]),
                      "output": str(out),
                      "elapsed_s": round(time.time() - T0, 1)}))
    return 0


def area_weights(height: int) -> tuple[np.ndarray, np.ndarray]:


    lat = -90.0 + (np.arange(height, dtype=np.float64) + 0.5) * (180.0 / height)
    w = np.cos(np.deg2rad(lat))
    w = w / w.mean()
    return w, np.sqrt(w)


def cca_predict_scores(x_scores_block: np.ndarray, model: dict) -> np.ndarray:

    kx = model["Kx"]
    centered = (np.asarray(x_scores_block[:, :kx], dtype=np.float64)
                - model["x_score_mean"][None, :])
    return centered @ model["B"] + model["y_score_mean"][None, :]


def run_cca_shard(args) -> int:
    global GATE_PATH
    out_root = guard(args.output_root)
    meta_path = out_root / "retention_metadata.json"
    if not meta_path.exists():
        raise SystemExit(f"missing {meta_path}; run the unet mode first")
    meta = json.loads(meta_path.read_text())
    if meta["n_shards"] != args.n_shards:
        raise SystemExit(f"shard count mismatch: retention has "
                         f"{meta['n_shards']}, asked for {args.n_shards}")
    s = args.shard_index
    part = meta["shards"][s]
    lo, hi = part["lo"], part["hi"]
    dates = meta["dates"][lo:hi]
    store_idx = meta["store_indices"][lo:hi]
    n = hi - lo

    sdir = out_root / "predictions" / "cca" / f"shard_{s:02d}"
    sdir.mkdir(parents=True, exist_ok=True)
    GATE_PATH = sdir / "identity_gates_cca.json"
    gates = new_gates(f"cca-shard[{s}]")


    existing = sdir / "cca_prediction_norm.npy"
    done = sdir / "shard_done.json"
    if args.reuse_if_present and existing.exists() and done.exists():
        rec = json.loads(done.read_text())
        arr = np.load(existing, mmap_mode="r")
        ok = (list(arr.shape) == [n, meta["grid"]["height"], meta["grid"]["width"]]
              and str(arr.dtype) == "float64"
              and rec.get("dates") == dates)
        if ok:
            log(f"CCA shard {s:02d} already retained and valid — reusing "
                f"({n} days, {dates[0]} .. {dates[-1]}); no reconstruction")
            print(json.dumps({"status": "PASS", "mode": "cca-shard", "shard": s,
                              "n_days": n, "reused": True,
                              "output": str(existing)}))
            return 0
        log(f"CCA shard {s:02d} present but not valid for reuse — reconstructing")

    split_cfg = SPLIT_INPUTS[args.split]
    scores_path = resolve_x_scores_path(split_cfg, out_root)
    for path in (CCA_MODEL, Y_BASIS, scores_path):
        guard(path)
    for path in (CCA_MODEL, Y_BASIS):
        if not Path(path).exists():
            raise SystemExit(f"missing required input: {path}")
    model = gate_cca_identity(gates)

    if not scores_path.exists():
        raise SystemExit(f"missing retained predictor scores: {scores_path}; "
                         f"run the project-scores mode first")
    x_scores = npy_memmap(scores_path)
    gate(gates, "cca_x_scores_shape",
         tuple(x_scores.shape) == tuple(split_cfg["x_scores_shape"]),
         list(x_scores.shape), list(split_cfg["x_scores_shape"]),
         {"dtype": str(x_scores.dtype), "path": str(scores_path)})
    mean_field = np.asarray(npz_small_member(Y_BASIS, "mean_field.npy"),
                            dtype=np.float64)
    eofs = npz_member_memmap(Y_BASIS, "eofs_weighted.npy")
    height, width = mean_field.shape
    ky = model["Ky"]
    gate(gates, "cca_eof_basis_usable",
         eofs.shape[0] >= ky and eofs.shape[1:] == (height, width),
         list(eofs.shape), [f">={ky}", height, width], {"dtype": str(eofs.dtype)})
    gate(gates, "cca_grid_matches_retention",
         (height, width) == (meta["grid"]["height"], meta["grid"]["width"]),
         [height, width], [meta["grid"]["height"], meta["grid"]["width"]])
    _w, sqrt_w = area_weights(height)


    yhat = cca_predict_scores(np.asarray(x_scores[store_idx]), model)
    if yhat.shape != (n, ky):
        raise SystemExit(f"yhat shape {yhat.shape} != {(n, ky)}")

    out_path = sdir / "cca_prediction_norm.npy"
    tmp_path = out_path.with_suffix(".writing.npy")
    out = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.float64,
                                    shape=(n, height, width))
    import zarr
    group = zarr.open_group(str(guard_store(meta["input_store"])), mode="r")
    tgt_mean = float(meta["normalization"]["t2m_tgt_mean_K"])
    tgt_std = float(meta["normalization"]["t2m_tgt_std_K"])
    for r0 in range(0, height, args.block_rows):
        r1 = min(height, r0 + args.block_rows)
        e_blk = np.asarray(eofs[:ky, r0:r1, :], dtype=np.float64).reshape(ky, -1)
        recon = (yhat @ e_blk).reshape(n, r1 - r0, width)
        del e_blk
        resid_hat = mean_field[r0:r1][None, :, :] + recon / sqrt_w[r0:r1][None, :, None]
        del recon
        for k, si in enumerate(store_idx):
            inp = np.asarray(group["inputs"][si, 0, r0:r1, :], dtype=np.float64)
            resid_hat[k] += baseline_from_inputs(inp, tgt_mean, tgt_std)
        if not np.isfinite(resid_hat).all():
            raise SystemExit(f"non-finite CCA reconstruction in rows [{r0},{r1})")
        out[:, r0:r1, :] = resid_hat
        del resid_hat
        log(f"shard {s:02d}: rows [{r0},{r1}) done")
    out.flush()
    del out
    os.replace(tmp_path, out_path)
    (sdir / "dates.txt").write_text("\n".join(dates) + "\n", encoding="utf-8")
    jdump({"shard": s, "lo": lo, "hi": hi, "n_days": n, "dates": dates,
           "created_utc": utc(),
           "arrays": {"cca_prediction_norm": {
               "file": f"cca/shard_{s:02d}/cca_prediction_norm.npy",
               "shape": [n, height, width], "dtype": "float64"}},
           "identity": {"cca_model_sha256": CCA_MODEL_SHA,
                        "cca_hyperparameters": CCA_HYPER}},
          sdir / "shard_done.json")
    log(f"CCA shard {s:02d} retained ({n} days, {dates[0]} .. {dates[-1]})")
    print(json.dumps({"status": "PASS", "mode": "cca-shard", "shard": s,
                      "n_days": n, "output": str(out_path),
                      "elapsed_s": round(time.time() - T0, 1)}))
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Coordinated adopted-B2 + frozen-CCA inference and "
                    "retention for present-day, mid-Holocene, or SSP5-8.5 "
                    "(predictions only; no metrics, no figures).")
    sub = p.add_subparsers(dest="mode", required=True)

    u = sub.add_parser("unet", help="adopted B2 on GPU over all requested dates")
    u.add_argument("--input-store", required=True)
    u.add_argument("--split", required=True, choices=sorted(SPLIT_INPUTS))
    u.add_argument("--start-date", required=True)
    u.add_argument("--end-date", required=True)
    u.add_argument("--output-root", required=True)
    u.add_argument("--device", default="cuda")
    u.add_argument("--n-shards", type=int, default=20)
    u.set_defaults(func=run_unet)

    j = sub.add_parser("project-scores",
                       help="frozen Stage-02 predictor projection for the split")
    j.add_argument("--split", required=True, choices=sorted(SPLIT_INPUTS))
    j.add_argument("--output-root", required=True)
    j.add_argument("--block-pixels", type=int, default=65536)
    j.add_argument("--matmul-chunk-pixels", type=int, default=8192)
    j.set_defaults(func=run_project_scores)

    c = sub.add_parser("cca-shard",
                       help="frozen CCA reconstruction for one date shard (CPU)")
    c.add_argument("--input-store", required=True)
    c.add_argument("--split", required=True, choices=sorted(SPLIT_INPUTS))
    c.add_argument("--output-root", required=True)
    c.add_argument("--shard-index", type=int, required=True)
    c.add_argument("--n-shards", type=int, default=20)
    c.add_argument("--block-rows", type=int, default=64)
    c.add_argument("--reuse-if-present", action="store_true",
                   help="reuse an already retained, complete and valid shard "
                        "instead of reconstructing it")
    c.set_defaults(func=run_cca_shard)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    configure_split(args.split)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
