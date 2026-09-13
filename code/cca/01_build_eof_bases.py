"""Build train-only, area-weighted EOF bases for the input and residual fields.
Uses spatially blocked anomaly matrices and the temporal Gram matrix to obtain
EOFs without materializing the entire training matrix in memory. Provides
planning and the phased full-data basis workflow.
Writes bases and metadata used by the later projection and CCA stages.
The full workflow requires the external training fields and configuration.
Standalone development smoke modes are omitted; production checks remain."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import struct
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cca_utils import (
    IssueLog,
    _np,
    area_weights_from_lat,
    done_matches,
    done_path,
    dtype_is_float32,
    grid_lat_centers,
    is_relative_to,
    load_json,
    resolve,
    sha256_file,
    spatial_blocks,
    utc_now,
)


RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "results_root")                                                  


HOME_RUN_ROOT = Path(f"{RESULTS_ROOT}/cca_final_run")
ALLOWED_HOME_OUTPUT_ROOT = HOME_RUN_ROOT / "outputs"
DEFAULT_HEAVY_OUTPUT_ROOT = Path(f"{RESULTS_ROOT}/cca_final_run/outputs_heavy")
ALLOWED_SCRATCH_PREFIX = Path(f"{RESULTS_ROOT}")
FORBIDDEN_HOME_PREFIX = Path("/home")
FORBIDDEN_WORK_PREFIX = Path("/work")
DEFAULT_CONFIG = HOME_RUN_ROOT / "config" / "eof_basis_config.json"
STAGE_NAME = "01_build_eof_bases"
VALID_KINDS = {"x_input", "y_residual"}
SEAL_TOKENS = ("awi_downscaling_test", "sealed", "holdout")
REAL_RUN_ENV = "CCA_STAGE01_ENABLE_REAL_RUN"
HEAVY_PHASES = ("gram-block", "reduce-gram", "temporal-eig", "eof-block", "assemble")


def guard_home_output_root(raw: str | Path, log: IssueLog) -> Path:
    out = resolve(Path(raw))
    allowed = resolve(ALLOWED_HOME_OUTPUT_ROOT)
    if out != allowed and not is_relative_to(out, allowed):
        log.fail(f"home output root {out} is outside allowed tree {allowed}")
    else:
        log.pass_(f"home manifest output root confined to {allowed}")
    return out


def guard_heavy_output_root(raw: str | Path, log: IssueLog) -> Path:
    out = resolve(Path(raw))
    scratch = resolve(ALLOWED_SCRATCH_PREFIX)
    home = resolve(FORBIDDEN_HOME_PREFIX)
    work = resolve(FORBIDDEN_WORK_PREFIX)
    if is_relative_to(out, home):
        log.fail(f"heavy output root must not be under /home: {out}")
    elif is_relative_to(out, work):
        log.fail(f"heavy output root must not be under /work: {out}")
    elif out != scratch and not is_relative_to(out, scratch):
        log.fail(f"heavy output root {out} is outside allowed scratch tree {scratch}")
    else:
        log.pass_(f"heavy output root confined to scratch tree {scratch}")
    return out


def guard_tmp_root(raw: str | Path, log: IssueLog) -> Path:
    tmp = resolve(Path(raw))
    scratch = resolve(ALLOWED_SCRATCH_PREFIX)
    home = resolve(FORBIDDEN_HOME_PREFIX)
    work = resolve(FORBIDDEN_WORK_PREFIX)
    if is_relative_to(tmp, home):
        log.fail(f"tmp_root must not be under /home for heavy mode: {tmp}")
    elif is_relative_to(tmp, work):
        log.fail(f"tmp_root must not be under /work for heavy mode: {tmp}")
    elif tmp != scratch and not is_relative_to(tmp, scratch):
        log.fail(f"tmp_root {tmp} is outside allowed scratch tree {scratch}")
    else:
        log.pass_(f"tmp_root configured under scratch tree {scratch}")
    return tmp


def suspicious_path(value: str) -> list[str]:
    lower = value.lower()
    hits = [token for token in SEAL_TOKENS if token in lower]
    parts = [part for part in re.split(r"[/_.\\-]+", lower) if part]
    if "test" in parts:
        hits.append("test")
    return sorted(set(hits))


def reject_suspicious_path(label: str, raw: str | Path, log: IssueLog) -> None:
    hits = suspicious_path(str(raw))
    if hits:
        log.fail(f"{label} contains sealed/test token(s) {hits}: {raw}")


def read_npy_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        magic = handle.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"not a .npy file: {path}")
        major, minor = handle.read(2)
        if major == 1:
            header_len = struct.unpack("<H", handle.read(2))[0]
        elif major in {2, 3}:
            header_len = struct.unpack("<I", handle.read(4))[0]
        else:
            raise ValueError(f"unsupported npy version {major}.{minor}: {path}")
        header = handle.read(header_len).decode("latin1")
    parsed = ast.literal_eval(header)
    return {
        "descr": parsed.get("descr"),
        "fortran_order": parsed.get("fortran_order"),
        "shape": tuple(parsed.get("shape") or ()),
        "version": (major, minor),
    }


def load_manifest(path: Path, log: IssueLog) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not path.is_file():
        log.fail(f"missing source train manifest: {path}")
        return [], []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    train_rows = sorted(
        (row for row in rows if row.get("split") == "train"),
        key=lambda row: int(row["start"]),
    )
    val_rows = [row for row in rows if row.get("split") == "val"]
    other = [row for row in rows if row.get("split") not in {"train", "val"}]
    if other:
        log.fail(f"manifest has unsupported split rows: {sorted({row.get('split') for row in other})}")
    return train_rows, val_rows


def check_storage_policy(config: dict[str, Any], home_output_root: Path, heavy_output_root: Path, log: IssueLog) -> Path:
    if config.get("home_heavy_outputs_allowed") is not False:
        log.fail("home_heavy_outputs_allowed must be false")
    else:
        log.pass_("home heavy outputs disabled")
    if config.get("work_is_read_only_for_new_outputs") is not True:
        log.fail("work_is_read_only_for_new_outputs must be true")
    else:
        log.pass_("/work marked read-only for new outputs")
    if config.get("scratch_is_archival") is not False:
        log.fail("scratch_is_archival must be false")
    else:
        log.pass_("scratch marked non-archival")
    if int(config.get("scratch_lifetime_days_since_last_access", -1)) != 14:
        log.fail("scratch_lifetime_days_since_last_access must be 14")
    else:
        log.pass_("scratch lifetime recorded as 14 days since last access")
    if int(config.get("home_quota_gib", -1)) != 60:
        log.fail("home_quota_gib must be 60")
    else:
        log.pass_("home quota recorded as 60 GiB")

    tmp_root = guard_tmp_root(config.get("tmp_root", ""), log)

    intended_home = config.get("intended_outputs", {})
    for label, raw in intended_home.items():
        path = resolve(Path(str(raw)))
        if not is_relative_to(path, home_output_root):
            log.fail(f"home intended output {label} is outside home output root: {path}")
        if path.suffix in {".npy", ".npz", ".zarr"}:
            log.fail(f"home intended output {label} looks like a heavy array artifact: {path}")
    if intended_home:
        log.pass_("home intended outputs are lightweight manifest/status artifacts")

    intended_heavy = config.get("intended_heavy_outputs", {})
    for label, raw in intended_heavy.items():
        path = resolve(Path(str(raw)))
        if not is_relative_to(path, heavy_output_root):
            log.fail(f"heavy intended output {label} is outside heavy output root: {path}")
        if is_relative_to(path, resolve(FORBIDDEN_HOME_PREFIX)) or is_relative_to(path, resolve(FORBIDDEN_WORK_PREFIX)):
            log.fail(f"heavy intended output {label} is under forbidden filesystem: {path}")
    if intended_heavy:
        log.pass_("heavy intended outputs are under the scratch heavy root")
    return tmp_root


def check_config(
    config: dict[str, Any],
    kind: str,
    max_modes: int,
    home_output_root: Path,
    heavy_output_root: Path,
    overwrite: bool,
    log: IssueLog,
) -> dict[str, Any] | None:
    if config.get("creation_mode") != "recompute_from_train_split":
        log.fail(f"creation_mode must be recompute_from_train_split, got {config.get('creation_mode')!r}")
    if config.get("non_final_dev_only") is not False:
        log.fail("non_final_dev_only must be false for canonical Stage 01")
    if config.get("no_test_access") is not True:
        log.fail("no_test_access must be true")
    if config.get("validation_used_for_eof_fit") not in {False, None}:
        log.fail("validation_used_for_eof_fit must be false")
    if config.get("expected_grid_shape") != [1280, 2624]:
        log.fail(f"expected_grid_shape mismatch: {config.get('expected_grid_shape')}")
    if int(config.get("expected_train_days", -1)) != 10593:
        log.fail(f"expected_train_days mismatch: {config.get('expected_train_days')}")

    grid_source = Path(config.get("grid_recommendation_source_path", ""))
    if grid_source.is_file():
        grid = load_json(grid_source)
        if grid.get("valid_row_count") != 465:
            log.fail(f"grid recommendation valid_row_count is {grid.get('valid_row_count')}, expected 465")
        else:
            log.pass_("grid recommendation valid_row_count=465")
        if grid.get("maxKx") != 10592 or grid.get("maxKy") != 1024 or grid.get("max_r") != 1024:
            log.fail("grid recommendation max dimensions are not 10592/1024/1024")
    else:
        log.fail(f"missing grid recommendation source: {grid_source}")

    check_storage_policy(config, home_output_root, heavy_output_root, log)

    kind_cfg = (config.get("kind_configs") or {}).get(kind)
    if not kind_cfg:
        log.fail(f"missing kind_configs.{kind}")
        return None
    configured_max = int(kind_cfg.get("max_modes", -1))
    if max_modes != configured_max:
        log.fail(f"--max-modes {max_modes} does not match configured {kind} max_modes {configured_max}")
    output_npz = resolve(Path(kind_cfg.get("output_npz", "")))
    work_dir = resolve(Path(kind_cfg.get("work_dir", "")))
    if output_npz.exists() and not overwrite:
        log.fail(f"intended heavy output exists and --overwrite was not set: {output_npz}")
    if not is_relative_to(output_npz, heavy_output_root):
        log.fail(f"intended basis output {output_npz} is outside heavy output root {heavy_output_root}")
    else:
        log.pass_(f"{kind} basis output planned under scratch heavy root")
    if not is_relative_to(work_dir, heavy_output_root):
        log.fail(f"stage01 work_dir {work_dir} is outside heavy output root {heavy_output_root}")
    else:
        log.pass_(f"{kind} work_dir planned under scratch heavy root")
    return kind_cfg


def check_metadata(config: dict[str, Any], log: IssueLog) -> None:
    paths = config.get("source_metadata_paths", {})
    for label, raw in paths.items():
        reject_suspicious_path(label, raw, log)
        path = Path(str(raw))
        if not path.is_file():
            log.fail(f"missing source metadata {label}: {path}")
            continue
        try:
            obj = load_json(path)
        except Exception as exc:
            log.fail(f"could not read JSON metadata {label}: {exc}")
            continue
        if "test_split_accessed" in obj:
            if obj.get("test_split_accessed") is not False:
                log.fail(f"{label} does not record test_split_accessed=false")
        elif label in {"stage_a_metadata", "stage_b_metadata", "stage00_run_manifest", "stage00_preflight_report"}:
            log.warn(f"{label} does not expose test_split_accessed; relying on Stage 00 manifest checks")
    stage_b = load_json(Path(paths["stage_b_metadata"]))
    if stage_b.get("train_only_means_for_eof") is True:
        log.pass_("Stage B records train_only_means_for_eof=true")
    else:
        log.fail("Stage B does not record train_only_means_for_eof=true")


def check_mean(kind_cfg: dict[str, Any], expected_shape: tuple[int, int], log: IssueLog) -> None:
    path = Path(kind_cfg["mean_path"])
    reject_suspicious_path("mean_path", path, log)
    if not path.is_file():
        log.fail(f"missing mean file: {path}")
        return
    try:
        header = read_npy_header(path)
    except Exception as exc:
        log.fail(f"could not read mean header {path}: {exc}")
        return
    if header["shape"] == expected_shape:
        log.pass_(f"mean header shape {expected_shape}: {path}")
    else:
        log.fail(f"mean shape mismatch {path}: {header['shape']} expected {expected_shape}")
    if dtype_is_float32(header["descr"]):
        log.pass_(f"mean dtype float32: {path}")
    else:
        log.fail(f"mean dtype is {header['descr']}, expected float32: {path}")


def check_train_shards(config: dict[str, Any], kind_cfg: dict[str, Any], log: IssueLog, *, inspect_headers: bool) -> dict[str, Any]:
    manifest = Path(config["source_train_manifest"])
    reject_suspicious_path("source_train_manifest", manifest, log)
    train_rows, val_rows = load_manifest(manifest, log)
    expected_h, expected_w = config["expected_grid_shape"]
    expected_train = int(config["expected_train_days"])
    column = kind_cfg["source_manifest_column"]
    total_train = 0
    bytes_total = 0
    for index, row in enumerate(train_rows):
        start = int(row["start"])
        end = int(row["end"])
        n = end - start
        total_train += n
        if int(row.get("height", -1)) != expected_h or int(row.get("width", -1)) != expected_w:
            log.fail(f"manifest grid mismatch row {index}: {row.get('height')}x{row.get('width')}")
        path = Path(row[column])
        reject_suspicious_path(f"train row {index} {column}", path, log)
        if not path.is_file():
            log.fail(f"missing train shard row {index}: {path}")
            continue
        bytes_total += path.stat().st_size
        if inspect_headers:
            try:
                header = read_npy_header(path)
            except Exception as exc:
                log.fail(f"could not read shard header row {index} {path}: {exc}")
                continue
            expected_shape = (n, expected_h, expected_w)
            if header["shape"] != expected_shape:
                log.fail(f"shard shape mismatch row {index}: {header['shape']} expected {expected_shape}")
            if not dtype_is_float32(header["descr"]):
                log.fail(f"shard dtype mismatch row {index}: {header['descr']} expected float32")
    if len(train_rows) == int(config.get("expected_train_shards", -1)):
        log.pass_(f"train shard count {len(train_rows)}")
    else:
        log.fail(f"train shard count {len(train_rows)} expected {config.get('expected_train_shards')}")
    if total_train == expected_train:
        log.pass_(f"train day count {total_train}")
    else:
        log.fail(f"train day count {total_train} expected {expected_train}")
    expected_val_shards = int(config.get("expected_validation_shards_excluded_from_eof_fit", -1))
    if len(val_rows) == expected_val_shards:
        log.pass_(f"validation rows present but excluded from EOF fit: {len(val_rows)}")
    else:
        log.warn(f"validation row count {len(val_rows)} differs from expected excluded count {expected_val_shards}")
    return {
        "manifest": str(manifest),
        "train_rows": len(train_rows),
        "validation_rows_excluded": len(val_rows),
        "train_days": total_train,
        "train_source_bytes": bytes_total,
        "headers_inspected": inspect_headers,
    }


def gib(num_bytes: int | float) -> float:
    return float(num_bytes) / (1024.0 ** 3)


def scratch_probe_dir(root: Path) -> None:

    root = resolve(root)
    if is_relative_to(root, resolve(FORBIDDEN_HOME_PREFIX)):
        raise RuntimeError(f"scratch probe refused: {root} is under /home")
    if is_relative_to(root, resolve(FORBIDDEN_WORK_PREFIX)):
        raise RuntimeError(f"scratch probe refused: {root} is under /work")
    if not is_relative_to(root, resolve(ALLOWED_SCRATCH_PREFIX)):
        raise RuntimeError(f"scratch probe refused: {root} is outside {ALLOWED_SCRATCH_PREFIX}")
    root.mkdir(parents=True, exist_ok=True)
    probe = root / f".{STAGE_NAME}_write_probe_{os.getpid()}_{int(time.time())}.tmp"
    probe.write_text("stage01 scratch write probe\n", encoding="utf-8")
    probe.unlink()


def scratch_write_probe(heavy_output_root: Path, tmp_root: Path, log: IssueLog) -> None:
    if os.environ.get("CCA_STAGE01_ALLOW_SCRATCH_WRITE_PROBE") != "1":
        log.fail("scratch write probe requested but CCA_STAGE01_ALLOW_SCRATCH_WRITE_PROBE=1 is not set")
        return
    for root, label in [(heavy_output_root, "heavy_output_root"), (tmp_root, "tmp_root")]:
        try:
            scratch_probe_dir(root)
        except RuntimeError as exc:
            log.fail(str(exc))
            continue
        log.pass_(f"scratch write probe succeeded for {label}: {root}")


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_save_npy(path: Path, arr: Any) -> None:
    np = _np()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, arr)
    os.replace(tmp, path)


def write_done(path: Path, ident: dict[str, Any], extra: dict[str, Any]) -> None:
    write_json_atomic(path, {"ident": ident, "created_utc": utc_now(), **extra})


@dataclass
class TrainSource:


    segments: list[tuple[int, int, Any]]
    n_total: int
    pixels: int
    height: int
    width: int


def open_heavy_train_source(train_rows: list[dict[str, str]], column: str, height: int, width: int) -> TrainSource:
    np = _np()
    segments: list[tuple[int, int, Any]] = []
    pos = 0
    pixels = height * width
    for row in train_rows:
        n = int(row["end"]) - int(row["start"])
        path = Path(row[column])
        hits = suspicious_path(str(path))
        if hits:
            raise RuntimeError(f"refusing sealed/test-token path in train source: {path} {hits}")
        arr = np.load(path, mmap_mode="r")
        if arr.shape != (n, height, width):
            raise RuntimeError(f"shard shape mismatch {path}: {arr.shape} expected {(n, height, width)}")
        segments.append((pos, n, arr.reshape(n, pixels)))
        pos += n
    return TrainSource(segments=segments, n_total=pos, pixels=pixels, height=height, width=width)


def build_weighted_anomaly_block(source: TrainSource, mean_flat, sqrt_w_flat, p0: int, p1: int, dtype) -> Any:

    np = _np()
    width = p1 - p0
    out = np.empty((source.n_total, width), dtype=dtype)
    mean_b = np.asarray(mean_flat[p0:p1], dtype=dtype)
    sw_b = np.asarray(sqrt_w_flat[p0:p1], dtype=dtype)
    for t0, n, flat in source.segments:
        seg = np.asarray(flat[:, p0:p1], dtype=dtype)
        seg = seg - mean_b[None, :]
        seg *= sw_b[None, :]
        out[t0:t0 + n] = seg
    return out


def gram_from_anomaly_block(a_block, chunk_pixels: int):

    np = _np()
    t = a_block.shape[0]
    gram = np.zeros((t, t), dtype=np.float64)
    for c0 in range(0, a_block.shape[1], chunk_pixels):
        chunk = np.asarray(a_block[:, c0:c0 + chunk_pixels], dtype=np.float64)
        gram += chunk @ chunk.T
    return gram


def eig_from_gram(gram, max_modes: int) -> dict[str, Any]:

    np = _np()
    gram = 0.5 * (gram + gram.T)
    total_ss = float(np.trace(gram))
    evals, vecs = np.linalg.eigh(gram)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 0.0)
    vecs = vecs[:, order]
    k = min(int(max_modes), evals.shape[0])
    singular = np.sqrt(evals[:k])
    return {
        "eigenvalues_full": evals,
        "eigenvalues": evals[:k],
        "singular_values": singular,
        "temporal_u": vecs[:, :k],
        "total_weighted_sum_squares": total_ss,
        "k": k,
    }


def u_over_s(temporal_u, singular_values):

    np = _np()
    out = np.array(temporal_u, dtype=np.float64, copy=True)
    s = np.asarray(singular_values, dtype=np.float64)
    good = s > 0
    out[:, good] /= s[good][None, :]
    out[:, ~good] = 0.0
    return out


def eof_rows_for_block(u_div_s, a_block, chunk_pixels: int):

    np = _np()
    k = u_div_s.shape[1]
    width = a_block.shape[1]
    out = np.empty((k, width), dtype=np.float64)
    for c0 in range(0, width, chunk_pixels):
        chunk = np.asarray(a_block[:, c0:c0 + chunk_pixels], dtype=np.float64)
        out[:, c0:c0 + chunk.shape[1]] = u_div_s.T @ chunk
    return out


def project_scores(fields_flat, mean_flat, sqrt_w_flat, eofs_flat):

    np = _np()
    anom = (np.asarray(fields_flat, dtype=np.float64) - np.asarray(mean_flat, dtype=np.float64)[None, :])
    anom *= np.asarray(sqrt_w_flat, dtype=np.float64)[None, :]
    return anom @ np.asarray(eofs_flat, dtype=np.float64).T


def reconstruct_fields_flat(scores, eofs_flat, sqrt_w_flat, mean_flat):

    np = _np()
    weighted = np.asarray(scores, dtype=np.float64) @ np.asarray(eofs_flat, dtype=np.float64)
    return np.asarray(mean_flat, dtype=np.float64)[None, :] + weighted / np.asarray(sqrt_w_flat, dtype=np.float64)[None, :]


def gram_block_path(work_dir: Path, index: int) -> Path:
    return work_dir / "gram_blocks" / f"gram_block_{index:04d}.npy"


def eof_block_path(work_dir: Path, index: int) -> Path:
    return work_dir / "eof_blocks" / f"eof_block_{index:04d}.npy"


def block_ident(ident: dict[str, Any], phase: str, block: tuple[int, int, int]) -> dict[str, Any]:
    index, p0, p1 = block
    return {**ident, "phase": phase, "block_index": index, "p0": p0, "p1": p1}


def phase_gram_block(
    source: TrainSource,
    mean_flat,
    sqrt_w_flat,
    block: tuple[int, int, int],
    work_dir: Path,
    ident: dict[str, Any],
    *,
    resume: bool,
    anomaly_dtype,
    chunk_pixels: int,
) -> dict[str, Any]:
    np = _np()
    index, p0, p1 = block
    out = gram_block_path(work_dir, index)
    done = done_path(out)
    bid = block_ident(ident, "gram-block", block)
    if resume and out.is_file() and done_matches(done, bid):
        return {"path": out, "skipped": True}
    a_block = build_weighted_anomaly_block(source, mean_flat, sqrt_w_flat, p0, p1, anomaly_dtype)
    gram = gram_from_anomaly_block(a_block, chunk_pixels)
    atomic_save_npy(out, gram)
    write_done(done, bid, {"bytes": out.stat().st_size, "block_trace": float(np.trace(gram))})
    return {"path": out, "skipped": False, "block_trace": float(np.trace(gram))}


def phase_reduce_gram(
    work_dir: Path,
    blocks: list[tuple[int, int, int]],
    ident: dict[str, Any],
    n_total: int,
) -> dict[str, Any]:
    np = _np()
    gram = np.zeros((n_total, n_total), dtype=np.float64)
    for block in blocks:
        index = block[0]
        artifact = gram_block_path(work_dir, index)
        bid = block_ident(ident, "gram-block", block)
        if not done_matches(done_path(artifact), bid):
            raise RuntimeError(f"reduce-gram refuses to run: missing/mismatched done JSON for gram block {index}")
        part = np.load(artifact)
        if part.shape != (n_total, n_total):
            raise RuntimeError(f"gram block {index} shape {part.shape} != {(n_total, n_total)}")
        gram += part
    gram = 0.5 * (gram + gram.T)
    out = work_dir / "gram_reduced" / "gram_total.npy"
    atomic_save_npy(out, gram)
    rid = {**ident, "phase": "reduce-gram", "n_blocks": len(blocks)}
    write_done(done_path(out), rid, {"bytes": out.stat().st_size, "trace": float(np.trace(gram))})
    return {"path": out, "gram": gram}


GRAM_IDENT_INSENSITIVE_KEYS = frozenset({"config_sha256", "max_modes"})


def gram_ident_compatible(payload_ident: dict[str, Any], expected_ident: dict[str, Any]) -> bool:

    got = {k: v for k, v in payload_ident.items() if k not in GRAM_IDENT_INSENSITIVE_KEYS}
    want = {k: v for k, v in expected_ident.items() if k not in GRAM_IDENT_INSENSITIVE_KEYS}
    return got == want


def phase_temporal_eig(
    work_dir: Path,
    ident: dict[str, Any],
    max_modes: int,
    n_total: int,
    n_blocks: int,
    gram=None,
) -> dict[str, Any]:
    np = _np()
    gram_path = work_dir / "gram_reduced" / "gram_total.npy"
    rid = {**ident, "phase": "reduce-gram", "n_blocks": n_blocks}
    gram_source_ident: dict[str, Any] | None = None
    if gram is None:
        if not done_matches(done_path(gram_path), rid):
            payload = load_json(done_path(gram_path)) if done_path(gram_path).is_file() else None
            source_ident = (payload or {}).get("ident") or {}
            if payload is None or not gram_ident_compatible(source_ident, rid):
                raise RuntimeError("temporal-eig refuses to run: reduce-gram done JSON missing or identity-incompatible")
            gram_source_ident = source_ident
        gram = np.load(gram_path)
        if gram.shape != (n_total, n_total):
            raise RuntimeError(f"reused/loaded gram shape {gram.shape} != {(n_total, n_total)}")
    eig = eig_from_gram(gram, max_modes)
    u_orth = float(np.max(np.abs(eig["temporal_u"].T @ eig["temporal_u"] - np.eye(eig["k"]))))
    out = work_dir / "temporal" / "temporal_eig.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez(
            handle,
            temporal_u=eig["temporal_u"],
            eigenvalues=eig["eigenvalues"],
            eigenvalues_full=eig["eigenvalues_full"],
            singular_values=eig["singular_values"],
            total_weighted_sum_squares=np.float64(eig["total_weighted_sum_squares"]),
        )
    os.replace(tmp, out)
    tid = {**ident, "phase": "temporal-eig", "k": eig["k"]}
    extra: dict[str, Any] = {"bytes": out.stat().st_size, "temporal_u_orth_error": u_orth}
    if gram_source_ident is not None:
        extra["gram_reused_source_ident"] = gram_source_ident
    write_done(done_path(out), tid, extra)
    return {"path": out, "eig": eig, "temporal_u_orth_error": u_orth}


def phase_eof_block(
    source: TrainSource,
    mean_flat,
    sqrt_w_flat,
    block: tuple[int, int, int],
    work_dir: Path,
    ident: dict[str, Any],
    u_div_s,
    *,
    resume: bool,
    anomaly_dtype,
    chunk_pixels: int,
    save_dtype,
) -> dict[str, Any]:
    np = _np()
    index, p0, p1 = block
    out = eof_block_path(work_dir, index)
    done = done_path(out)
    bid = block_ident(ident, "eof-block", block)
    if resume and out.is_file() and done_matches(done, bid):
        return {"path": out, "skipped": True}
    a_block = build_weighted_anomaly_block(source, mean_flat, sqrt_w_flat, p0, p1, anomaly_dtype)
    rows = eof_rows_for_block(u_div_s, a_block, chunk_pixels)
    atomic_save_npy(out, rows.astype(save_dtype))
    write_done(done, bid, {"bytes": out.stat().st_size})
    return {"path": out, "skipped": False}


def load_temporal_eig(work_dir: Path, ident: dict[str, Any], max_modes: int) -> dict[str, Any]:
    np = _np()
    out = work_dir / "temporal" / "temporal_eig.npz"
    payload = load_json(done_path(out)) if done_path(out).is_file() else None
    if payload is None:
        raise RuntimeError("temporal_eig.npz done JSON missing")
    tid = payload.get("ident", {})
    base = {key: value for key, value in tid.items() if key not in {"phase", "k"}}
    expect = {key: value for key, value in ident.items() if key not in {"phase", "k"}}
    if base != expect:
        raise RuntimeError("temporal_eig.npz done JSON identity mismatch")
    with np.load(out) as data:
        eig = {
            "temporal_u": data["temporal_u"],
            "eigenvalues": data["eigenvalues"],
            "eigenvalues_full": data["eigenvalues_full"],
            "singular_values": data["singular_values"],
            "total_weighted_sum_squares": float(data["total_weighted_sum_squares"]),
            "k": int(data["temporal_u"].shape[1]),
        }
    if eig["k"] > max_modes:
        raise RuntimeError(f"temporal eig k={eig['k']} exceeds max_modes={max_modes}")
    return eig


def phase_assemble(
    work_dir: Path,
    blocks: list[tuple[int, int, int]],
    ident: dict[str, Any],
    eig: dict[str, Any],
    mean_field,
    latitude,
    longitude,
    out_npz: Path,
    *,
    save_dtype,
    height: int,
    width: int,
) -> dict[str, Any]:


    np = _np()
    k = int(eig["k"])
    pixels = height * width
    for block in blocks:
        bid = block_ident(ident, "eof-block", block)
        if not done_matches(done_path(eof_block_path(work_dir, block[0])), bid):
            raise RuntimeError(f"assemble refuses to run: missing/mismatched done JSON for eof block {block[0]}")

    members_dir = work_dir / "assemble_members"
    if members_dir.exists():
        shutil.rmtree(members_dir)
    members_dir.mkdir(parents=True, exist_ok=True)

    from numpy.lib.format import open_memmap

    eofs_member = members_dir / "eofs_weighted.npy"
    eofs = open_memmap(eofs_member, mode="w+", dtype=save_dtype, shape=(k, height, width))
    eofs_flat = eofs.reshape(k, pixels)
    gram_check = np.zeros((k, k), dtype=np.float64)
    for block in blocks:
        index, p0, p1 = block
        rows = np.load(eof_block_path(work_dir, index))
        if rows.shape != (k, p1 - p0):
            raise RuntimeError(f"eof block {index} shape {rows.shape} != {(k, p1 - p0)}")
        eofs_flat[:, p0:p1] = rows.astype(save_dtype, copy=False)
        rows64 = rows.astype(np.float64, copy=False)
        gram_check += rows64 @ rows64.T
    eofs.flush()
    del eofs, eofs_flat

    nonzero = np.asarray(eig["singular_values"], dtype=np.float64) > 0
    eye = np.eye(k)
    orth_dev = gram_check - eye
    orth_error_nonzero = float(np.max(np.abs(orth_dev[np.ix_(nonzero, nonzero)]))) if nonzero.any() else float("nan")
    zero_modes = int((~nonzero).sum())

    total_ss = float(eig["total_weighted_sum_squares"])
    evf = np.asarray(eig["eigenvalues"], dtype=np.float64) / total_ss if total_ss > 0 else np.zeros(k)
    evals_full = np.asarray(eig["eigenvalues_full"], dtype=np.float64)
    cumulative = np.cumsum(evals_full)[:k] / total_ss if total_ss > 0 else np.zeros(k)

    def member(name: str, arr) -> Path:
        path = members_dir / f"{name}.npy"
        atomic_save_npy(path, arr)
        return path

    members = [eofs_member]
    members.append(member("mean_field", np.asarray(mean_field, dtype=np.float32)))
    members.append(member("eigenvalues", np.asarray(eig["eigenvalues"], dtype=np.float64)))
    members.append(member("singular_values", np.asarray(eig["singular_values"], dtype=np.float64)))
    members.append(member("explained_variance_fraction", evf))
    members.append(member("cumulative_variance_fraction", cumulative))
    members.append(member("total_weighted_sum_squares", np.float64(total_ss)))
    members.append(member("latitude", np.asarray(latitude, dtype=np.float64)))
    members.append(member("longitude", np.asarray(longitude, dtype=np.float64)))

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_npz.with_name(out_npz.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in members:
            archive.write(path, arcname=path.name)
    os.replace(tmp, out_npz)
    shutil.rmtree(members_dir)

    aid = {**ident, "phase": "assemble", "k": k}
    write_done(
        done_path(out_npz),
        aid,
        {
            "bytes": out_npz.stat().st_size,
            "stored_orthonormality_max_abs_dev_nonzero_modes": orth_error_nonzero,
            "zero_singular_value_modes": zero_modes,
        },
    )
    return {
        "path": out_npz,
        "stored_orth_error": orth_error_nonzero,
        "zero_modes": zero_modes,
        "explained_variance_fraction": evf,
        "cumulative_variance_fraction": cumulative,
    }


def heavy_ident(config_path: Path, config: dict[str, Any], kind: str, max_modes: int, block_pixels: int) -> dict[str, Any]:
    manifest = Path(config["source_train_manifest"])
    return {
        "stage": STAGE_NAME,
        "mode": "production",
        "kind": kind,
        "config_sha256": sha256_file(config_path),
        "train_manifest_sha256": sha256_file(manifest),
        "grid_shape": list(config["expected_grid_shape"]),
        "n_train": int(config["expected_train_days"]),
        "max_modes": int(max_modes),
        "block_pixels": int(block_pixels),
    }


def load_static_lat_lon(config: dict[str, Any]):

    np = _np()
    height, width = config["expected_grid_shape"]
    lat_formula = grid_lat_centers(height)
    lon_formula = (np.arange(width, dtype=np.float64) + 0.5) * (360.0 / width)
    static = config.get("static_grid_fields") or {}
    path = static.get("latitude")
    if path:
        try:
            import netCDF4

            with netCDF4.Dataset(path) as ds:
                lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
                lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
            if lat.shape != (height,) or lon.shape != (width,):
                raise RuntimeError(f"static lat/lon shape mismatch: {lat.shape} {lon.shape}")
            if float(np.max(np.abs(lat - lat_formula))) > 1e-4:
                raise RuntimeError("static latitude deviates from grid formula by more than 1e-4 deg")
            return lat, lon, "static_netcdf_checked_against_formula"
        except ImportError:
            pass
    return lat_formula, lon_formula, "grid_formula_fallback"


def run_heavy_phase(
    args: argparse.Namespace,
    config: dict[str, Any],
    config_path: Path,
    kind_cfg: dict[str, Any],
    home_output_root: Path,
    heavy_output_root: Path,
) -> int:
    np = _np()
    phase = args.phase
    if config.get("production_execution_approved") is not True:
        print(f"FAIL heavy phase {phase!r} refused: config production_execution_approved is not true")
        return 2
    if os.environ.get(REAL_RUN_ENV) != "1":
        print(f"FAIL heavy phase {phase!r} refused: environment {REAL_RUN_ENV}=1 is not set")
        return 2

    height, width = config["expected_grid_shape"]
    pixels = height * width
    block_pixels = int(args.spatial_block_pixels or config.get("algorithm", {}).get("spatial_block_pixels", 65536))
    chunk_pixels = int(args.gram_chunk_pixels)
    max_modes = int(args.max_modes)
    work_dir = resolve(Path(kind_cfg["work_dir"]))
    out_npz = resolve(Path(kind_cfg["output_npz"]))
    ident = heavy_ident(config_path, config, args.kind, max_modes, block_pixels)
    blocks = spatial_blocks(pixels, block_pixels)
    if args.block_index is not None and not 0 <= args.block_index < len(blocks):
        print(f"FAIL --block-index {args.block_index} out of range 0..{len(blocks) - 1} for {len(blocks)} spatial blocks")
        return 2

    scratch_probe_dir(work_dir)
    log = IssueLog()
    train_rows, _ = load_manifest(Path(config["source_train_manifest"]), log)
    if not log.ok():
        print("FAIL heavy phase could not load train manifest")
        return 2

    lat_full = grid_lat_centers(height)
    _, sqrt_w = area_weights_from_lat(lat_full)
    sqrt_w_flat = np.repeat(sqrt_w[:, None], width, axis=1).reshape(-1)

    def open_source() -> TrainSource:
        source = open_heavy_train_source(train_rows, kind_cfg["source_manifest_column"], height, width)
        if source.n_total != int(config["expected_train_days"]):
            raise RuntimeError(f"train source day count {source.n_total} != expected {config['expected_train_days']}")
        return source

    def mean_flat64():
        mean = np.load(Path(kind_cfg["mean_path"]))
        return np.asarray(mean, dtype=np.float64).reshape(-1), mean

    t0 = time.time()
    if phase == "gram-block":
        source = open_source()
        mflat, _ = mean_flat64()
        todo = blocks if args.block_index is None else [blocks[args.block_index]]
        for block in todo:
            result = phase_gram_block(
                source, mflat, sqrt_w_flat, block, work_dir, ident,
                resume=args.resume, anomaly_dtype=np.float32, chunk_pixels=chunk_pixels,
            )
            print(f"GRAM_BLOCK_DONE index={block[0]} skipped={result['skipped']} seconds={time.time() - t0:.1f}", flush=True)
    elif phase == "reduce-gram":
        phase_reduce_gram(work_dir, blocks, ident, int(config["expected_train_days"]))
        print(f"REDUCE_GRAM_DONE n_blocks={len(blocks)} seconds={time.time() - t0:.1f}")
    elif phase == "temporal-eig":
        result = phase_temporal_eig(work_dir, ident, max_modes, int(config["expected_train_days"]), len(blocks))
        print(f"TEMPORAL_EIG_DONE k={result['eig']['k']} u_orth_error={result['temporal_u_orth_error']:.3e} seconds={time.time() - t0:.1f}")
    elif phase == "eof-block":
        eig = load_temporal_eig(work_dir, ident, max_modes)
        u_div_s = u_over_s(eig["temporal_u"], eig["singular_values"])
        source = open_source()
        mflat, _ = mean_flat64()
        todo = blocks if args.block_index is None else [blocks[args.block_index]]
        for block in todo:
            result = phase_eof_block(
                source, mflat, sqrt_w_flat, block, work_dir, ident, u_div_s,
                resume=args.resume, anomaly_dtype=np.float32, chunk_pixels=chunk_pixels, save_dtype=np.float32,
            )
            print(f"EOF_BLOCK_DONE index={block[0]} skipped={result['skipped']} seconds={time.time() - t0:.1f}", flush=True)
    elif phase == "assemble":
        eig = load_temporal_eig(work_dir, ident, max_modes)
        _, mean_field = mean_flat64()
        latitude, longitude, latlon_source = load_static_lat_lon(config)
        assembled = phase_assemble(
            work_dir, blocks, ident, eig, mean_field, latitude, longitude, out_npz,
            save_dtype=np.float32, height=height, width=width,
        )
        write_heavy_home_manifests(config, args.kind, kind_cfg, ident, eig, assembled, home_output_root, latlon_source)
        print(f"ASSEMBLE_DONE npz={out_npz} stored_orth_error={assembled['stored_orth_error']:.3e} seconds={time.time() - t0:.1f}")
    else:
        print(f"FAIL unknown heavy phase {phase!r}")
        return 2
    return 0


def write_heavy_home_manifests(
    config: dict[str, Any],
    kind: str,
    kind_cfg: dict[str, Any],
    ident: dict[str, Any],
    eig: dict[str, Any],
    assembled: dict[str, Any],
    home_output_root: Path,
    latlon_source: str,
) -> None:

    np = _np()
    out_dir = home_output_root / "eof_bases"
    out_npz = Path(kind_cfg["output_npz"])
    npz_sha = sha256_file(out_npz)

    intended = config.get("intended_outputs", {})

    def merge(path_key: str, update) -> None:
        path = Path(intended[path_key])
        payload = {}
        if path.is_file():
            try:
                payload = load_json(path)
            except Exception:
                payload = {}
        update(payload)
        write_json_atomic(path, payload)

    entry = {
        "kind": kind,
        "ident": ident,
        "output_npz": str(out_npz),
        "output_npz_sha256": npz_sha,
        "output_npz_bytes": out_npz.stat().st_size,
        "k": int(eig["k"]),
        "total_weighted_sum_squares": float(eig["total_weighted_sum_squares"]),
        "stored_orthonormality_max_abs_dev": assembled["stored_orth_error"],
        "zero_singular_value_modes": assembled["zero_modes"],
        "latlon_source": latlon_source,
        "test_split_accessed": False,
        "validation_used_for_eof_fit": False,
        "created_utc": utc_now(),
    }

    def upd_meta(payload: dict[str, Any]) -> None:
        payload.setdefault("stage", STAGE_NAME)
        payload.setdefault("kinds", {})[kind] = entry

    def upd_status(payload: dict[str, Any]) -> None:
        payload.setdefault("stage", STAGE_NAME)
        payload.setdefault("kinds", {})[kind] = {"phase": "assemble", "status": "complete", "updated_utc": utc_now()}

    def upd_manifest(payload: dict[str, Any]) -> None:
        payload.setdefault("stage", STAGE_NAME)
        payload.setdefault("heavy_artifacts", {})[kind] = {
            "path": str(out_npz),
            "sha256": npz_sha,
            "bytes": out_npz.stat().st_size,
            "location": "scratch_heavy_data_plane",
        }

    def upd_warning(payload: dict[str, Any]) -> None:
        payload.update({
            "stage": STAGE_NAME,
            "warning": "Scratch files have a 14-day lifetime since last access and are not archival. "
                       "Copy final heavy artifacts to an approved archival target.",
            "scratch_lifetime_days_since_last_access": 14,
            "updated_utc": utc_now(),
        })

    merge("metadata_json", upd_meta)
    merge("stage_status_json", upd_status)
    merge("heavy_artifact_manifest_json", upd_manifest)
    merge("scratch_lifetime_warning_json", upd_warning)


    csv_path = Path(intended["variance_spectra_csv"])
    rows: list[dict[str, str]] = []
    if csv_path.is_file():
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row.get("kind") != kind]
    evf = np.asarray(assembled["explained_variance_fraction"], dtype=np.float64)
    cumulative = np.asarray(assembled["cumulative_variance_fraction"], dtype=np.float64)
    singular = np.asarray(eig["singular_values"], dtype=np.float64)
    evals = np.asarray(eig["eigenvalues"], dtype=np.float64)
    for i in range(int(eig["k"])):
        rows.append({
            "kind": kind,
            "component": str(i + 1),
            "eigenvalue": f"{evals[i]:.12e}",
            "singular_value": f"{singular[i]:.12e}",
            "explained_variance_fraction": f"{evf[i]:.12e}",
            "cumulative_variance_fraction": f"{cumulative[i]:.12e}",
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = csv_path.with_name(csv_path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "kind", "component", "eigenvalue", "singular_value",
            "explained_variance_fraction", "cumulative_variance_fraction",
        ])
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, csv_path)


def print_plan(
    config: dict[str, Any],
    kind: str,
    kind_cfg: dict[str, Any],
    max_modes: int,
    home_output_root: Path,
    heavy_output_root: Path,
    tmp_root: Path,
    shard_summary: dict[str, Any],
) -> None:
    storage = config.get("storage_estimate_bytes", {})
    key = "x_eofs_weighted_float32" if kind == "x_input" else "y_eofs_weighted_float32"
    ablock = "x_A_block_65536_float64" if kind == "x_input" else "y_A_block_65536_float64"
    full_a = "x_full_A_if_materialized_float64" if kind == "x_input" else "y_full_A_if_materialized_float64"
    block_pixels = int(config.get("algorithm", {}).get("spatial_block_pixels", 65536))
    pixels = int(config["expected_grid_shape"][0]) * int(config["expected_grid_shape"][1])
    n_blocks = math.ceil(pixels / block_pixels)
    t_days = int(config["expected_train_days"])
    print("STAGE01_DRY_RUN_PLAN")
    print(f"created_utc={utc_now()}")
    print(f"kind={kind}")
    print(f"max_modes={max_modes}")
    print(f"home_manifest_output_root={home_output_root}")
    print(f"heavy_output_root={heavy_output_root}")
    print(f"tmp_root={tmp_root}")
    print(f"scratch_lifetime_days_since_last_access={config.get('scratch_lifetime_days_since_last_access')}")
    print(f"scratch_is_archival={config.get('scratch_is_archival')}")
    print(f"home_heavy_outputs_allowed={config.get('home_heavy_outputs_allowed')}")
    print(f"source_manifest_column={kind_cfg['source_manifest_column']}")
    print(f"mean_path={kind_cfg['mean_path']}")
    print(f"heavy_output_npz={kind_cfg['output_npz']}")
    print(f"heavy_work_dir={kind_cfg['work_dir']}")
    print(f"home_eof_metadata={config.get('intended_outputs', {}).get('metadata_json')}")
    print(f"home_heavy_artifact_manifest={config.get('intended_outputs', {}).get('heavy_artifact_manifest_json')}")
    print(f"home_scratch_lifetime_warning={config.get('intended_outputs', {}).get('scratch_lifetime_warning_json')}")
    print(f"train_rows={shard_summary['train_rows']}")
    print(f"validation_rows_excluded={shard_summary['validation_rows_excluded']}")
    print(f"train_source_bytes_header_checked={shard_summary['train_source_bytes']}")
    print(f"basis_float32_size_gib={gib(storage[key]):.3f}")
    print(f"full_A_if_materialized_float64_gib={gib(storage[full_a]):.3f}")
    print(f"A_block_65536_float64_gib={gib(storage[ablock]):.3f}")
    print(f"temporal_gram_float64_gib={gib(t_days * t_days * 8):.3f}")
    print(f"spatial_block_pixels={block_pixels}")
    print(f"spatial_block_count={n_blocks}")
    print(f"gram_block_intermediates_total_gib={gib(n_blocks * t_days * t_days * 8):.3f}")
    print("phases=plan,gram-block,reduce-gram,temporal-eig,eof-block,assemble")
    print("algorithm=method of snapshots: temporal Gram from weighted centered train anomalies; eofs_weighted orthonormal in weighted pixel space")
    print("projection_formula=score = ((field - train_mean) * sqrt_weight) dot eofs_weighted.T")
    print("reconstruction_formula=field = train_mean + (scores dot eofs_weighted) / sqrt_weight")
    print("parallel_plan=gram-block and eof-block Slurm arrays with %20 throttle; reduce-gram/temporal-eig/assemble single reducer jobs")
    print("preservation_warning=scratch is temporary; final heavy artifacts must be copied to stable archival storage when approved")
    print("heavy_compute_implemented=true")
    print(f"production_execution_approved={config.get('production_execution_approved')}")
    print(f"heavy_phase_env_guard={REAL_RUN_ENV}=1 required")


def print_summary(log: IssueLog) -> None:
    for msg in log.failures:
        print(f"FAIL {msg}")
    for msg in log.warnings:
        print(f"WARN {msg}")
    for msg in log.passes:
        print(f"PASS {msg}")
    print(f"SUMMARY pass={len(log.passes)} warn={len(log.warnings)} fail={len(log.failures)}")


def run(args: argparse.Namespace) -> int:
    log = IssueLog()
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"FAIL missing config: {config_path}")
        return 2
    config = load_json(config_path)
    if args.kind not in VALID_KINDS:
        print(f"FAIL invalid kind: {args.kind}")
        return 2

    home_output_root = guard_home_output_root(args.output_root or config.get("home_output_root") or config.get("output_root") or ALLOWED_HOME_OUTPUT_ROOT, log)
    heavy_output_root = guard_heavy_output_root(args.heavy_output_root or config.get("heavy_output_root") or DEFAULT_HEAVY_OUTPUT_ROOT, log)

    check_metadata(config, log)
    kind_cfg = check_config(config, args.kind, args.max_modes, home_output_root, heavy_output_root, args.overwrite, log)
    tmp_root = resolve(Path(config.get("tmp_root", "")))
    if args.write_probe:
        scratch_write_probe(heavy_output_root, tmp_root, log)
    if kind_cfg is None:
        print_summary(log)
        return 2
    check_mean(kind_cfg, tuple(config["expected_grid_shape"]), log)
    inspect_headers = bool(args.metadata_only or args.inspect_npy_headers)
    shard_summary = check_train_shards(config, kind_cfg, log, inspect_headers=inspect_headers)

    print_summary(log)
    if not log.ok():
        return 2

    if args.dry_run or args.plan_only or args.phase == "plan":
        print_plan(config, args.kind, kind_cfg, args.max_modes, home_output_root, heavy_output_root, tmp_root, shard_summary)
        if args.phase != "plan":
            print(f"DRY_RUN phase {args.phase!r} NOT executed because --dry-run/--plan-only was set")
        print("DRY_RUN no outputs written; no dynamic arrays loaded")
        return 0

    if args.phase in HEAVY_PHASES:
        return run_heavy_phase(args, config, config_path, kind_cfg, home_output_root, heavy_output_root)

    print(f"FAIL unhandled phase {args.phase!r}")
    return 2


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 01 clean EOF basis builder (phased; heavy phases double-gated).")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Stage 01 EOF basis config JSON.")
    parser.add_argument("--kind", choices=sorted(VALID_KINDS), required=True, help="EOF side to build.")
    parser.add_argument("--max-modes", type=int, required=True, help="Maximum EOF modes for this kind; must match config.")
    parser.add_argument("--output-root", help="Home manifest output root; must be inside the clean home outputs tree.")
    parser.add_argument("--heavy-output-root", help=f"Scratch heavy output root; must be under {RESULTS_ROOT} and not under /home or /work.")
    parser.add_argument("--dry-run", action="store_true", help="Validate metadata and print plan without writing outputs.")
    parser.add_argument("--metadata-only", action="store_true", default=True, help="Inspect metadata and .npy headers only; default true.")
    parser.add_argument("--plan-only", action="store_true", help="Alias-like mode for dry-run planning output.")
    parser.add_argument("--resume", action="store_true", help="Reuse completed block artifacts whose done JSON identity matches.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing Stage 01 outputs. Default false.")
    parser.add_argument("--write-probe", action="store_true", help="Tiny scratch write probe; requires CCA_STAGE01_ALLOW_SCRATCH_WRITE_PROBE=1.")
    parser.add_argument(
        "--phase",
        choices=["plan", "gram-block", "reduce-gram", "temporal-eig", "eof-block", "assemble"],
        default="plan",
        help="Phase selector. Heavy phases additionally require CCA_STAGE01_ENABLE_REAL_RUN=1 and config production_execution_approved=true.",
    )
    parser.add_argument("--block-index", type=int, help="Spatial block index for gram-block/eof-block (Slurm array task id). Omit to run all blocks sequentially.")
    parser.add_argument("--spatial-block-pixels", type=int, help="Override spatial block size in pixels; default from config algorithm.spatial_block_pixels.")
    parser.add_argument("--gram-chunk-pixels", type=int, default=8192, help="float64 pixel chunk size inside Gram/EOF accumulation.")
    parser.add_argument("--inspect-npy-headers", action="store_true", help="Explicitly inspect .npy headers without loading arrays.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
