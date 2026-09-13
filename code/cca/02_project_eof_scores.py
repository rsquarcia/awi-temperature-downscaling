"""Project training and validation fields onto the fixed training EOF bases.
Applies the saved training means and latitude weights, computes score chunks
and assembles the predictor and residual score matrices for CCA fitting.
No new EOF basis or validation-dependent transform is fitted here.
Requires the stage-01 bases and the external training/validation fields.
Standalone development smoke modes are omitted; production checks remain."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
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
    npz_member_memmap,
    npz_member_npy_info,
    npz_small_member,
    parse_npy_header,
    read_npy_header,
    resolve,
    sha256_file,
    spatial_blocks,
    unique_tmp,
    utc_now,
    write_done,
    write_json_atomic,
    zip_member_data_offset,
)


RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "results_root")                                                  


HOME_RUN_ROOT = Path(f"{RESULTS_ROOT}/cca_final_run")
ALLOWED_HOME_OUTPUT_ROOT = HOME_RUN_ROOT / "outputs"
DEFAULT_HEAVY_OUTPUT_ROOT = Path(f"{RESULTS_ROOT}/cca_final_run/outputs_heavy")
ALLOWED_SCRATCH_PREFIX = Path(f"{RESULTS_ROOT}")
FORBIDDEN_HOME_PREFIX = Path("/home")
FORBIDDEN_WORK_PREFIX = Path("/work")
DEFAULT_CONFIG = HOME_RUN_ROOT / "config" / "projection_config.json"
STAGE_NAME = "02_project_eof_scores"
VALID_KINDS = {"x_input", "y_residual"}
VALID_SPLITS = ("train", "val")
SEAL_TOKENS = ("awi_downscaling_test", "sealed", "holdout")
REAL_RUN_ENV = "CCA_STAGE02_ENABLE_REAL_RUN"
WRITE_PROBE_ENV = "CCA_STAGE02_ALLOW_SCRATCH_WRITE_PROBE"
HEAVY_PHASES = ("project-chunk", "assemble-scores")
SMALL_MEMBER_MAX_BYTES = 64 << 20


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
    if is_relative_to(out, resolve(FORBIDDEN_HOME_PREFIX)):
        log.fail(f"heavy output root must not be under /home: {out}")
    elif is_relative_to(out, resolve(FORBIDDEN_WORK_PREFIX)):
        log.fail(f"heavy output root must not be under /work: {out}")
    elif out != scratch and not is_relative_to(out, scratch):
        log.fail(f"heavy output root {out} is outside allowed scratch tree {scratch}")
    else:
        log.pass_(f"heavy output root confined to scratch tree {scratch}")
    return out


def guard_tmp_root(raw: str | Path, log: IssueLog) -> Path:
    tmp = resolve(Path(raw))
    scratch = resolve(ALLOWED_SCRATCH_PREFIX)
    if is_relative_to(tmp, resolve(FORBIDDEN_HOME_PREFIX)):
        log.fail(f"tmp_root must not be under /home for heavy mode: {tmp}")
    elif is_relative_to(tmp, resolve(FORBIDDEN_WORK_PREFIX)):
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


def atomic_save_npy(path: Path, arr: Any) -> None:
    np = _np()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = unique_tmp(path)
    with tmp.open("wb") as handle:
        np.save(handle, arr)
    os.replace(tmp, path)


def load_manifest_split(path: Path, split: str, log: IssueLog) -> tuple[list[dict[str, str]], int]:

    if split not in VALID_SPLITS:
        log.fail(f"invalid split {split!r}; allowed: {VALID_SPLITS}")
        return [], 0
    if not path.is_file():
        log.fail(f"missing source manifest: {path}")
        return [], 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    other = sorted({row.get("split") for row in rows} - set(VALID_SPLITS))
    if other:
        log.fail(f"manifest has unsupported split rows: {other}")
    picked = sorted((row for row in rows if row.get("split") == split), key=lambda row: int(row["start"]))
    pos = 0
    for index, row in enumerate(picked):
        start, end = int(row["start"]), int(row["end"])
        if start != pos or end <= start:
            log.fail(f"{split} manifest rows not contiguous at row {index}: start={start} expected {pos}")
            return picked, 0
        pos = end
    return picked, pos


@dataclass
class SplitSource:


    segments: list[tuple[int, int, Any]]
    n_total: int
    pixels: int
    height: int
    width: int


def open_split_source(rows: list[dict[str, str]], column: str, height: int, width: int) -> SplitSource:
    np = _np()
    segments: list[tuple[int, int, Any]] = []
    pos = 0
    pixels = height * width
    for row in rows:
        n = int(row["end"]) - int(row["start"])
        path = Path(row[column])
        hits = suspicious_path(str(path))
        if hits:
            raise RuntimeError(f"refusing sealed/test-token path in projection source: {path} {hits}")
        arr = np.load(path, mmap_mode="r")
        if arr.shape != (n, height, width):
            raise RuntimeError(f"shard shape mismatch {path}: {arr.shape} expected {(n, height, width)}")
        segments.append((pos, n, arr.reshape(n, pixels)))
        pos += n
    return SplitSource(segments=segments, n_total=pos, pixels=pixels, height=height, width=width)


def slice_source_days(source: SplitSource, d0: int, d1: int) -> SplitSource:

    segments: list[tuple[int, int, Any]] = []
    for t0, n, flat in source.segments:
        lo, hi = max(d0, t0), min(d1, t0 + n)
        if lo < hi:
            segments.append((lo - d0, hi - lo, flat[lo - t0:hi - t0]))
    covered = sum(n for _, n, _ in segments)
    if covered != d1 - d0:
        raise RuntimeError(f"day range [{d0},{d1}) not fully covered by source segments (got {covered})")
    return SplitSource(segments=segments, n_total=d1 - d0, pixels=source.pixels, height=source.height, width=source.width)


def build_weighted_anomaly_block(source: SplitSource, mean_flat, sqrt_w_flat, p0: int, p1: int, dtype) -> Any:

    np = _np()
    out = np.empty((source.n_total, p1 - p0), dtype=dtype)
    mean_b = np.asarray(mean_flat[p0:p1], dtype=dtype)
    sw_b = np.asarray(sqrt_w_flat[p0:p1], dtype=dtype)
    for t0, n, flat in source.segments:
        seg = np.asarray(flat[:, p0:p1], dtype=dtype)
        seg = seg - mean_b[None, :]
        seg *= sw_b[None, :]
        out[t0:t0 + n] = seg
    return out


def day_chunks(n_days: int, chunk_days: int) -> list[tuple[int, int, int]]:
    return [
        (index, d0, min(n_days, d0 + chunk_days))
        for index, d0 in enumerate(range(0, n_days, chunk_days))
    ]


def chunk_scores_path(work_dir: Path, split: str, index: int) -> Path:
    return work_dir / split / "chunks" / f"scores_chunk_{index:04d}.npy"


def project_days_onto_basis(source: SplitSource, mean_flat, sqrt_w_flat, eofs_flat, k: int, *, block_pixels: int, matmul_chunk_pixels: int):


    np = _np()
    scores = np.zeros((source.n_total, k), dtype=np.float64)
    for _, p0, p1 in spatial_blocks(source.pixels, block_pixels):
        a_block = build_weighted_anomaly_block(source, mean_flat, sqrt_w_flat, p0, p1, np.float32)
        e_block = np.asarray(eofs_flat[:, p0:p1], dtype=np.float32)
        for c0 in range(0, p1 - p0, matmul_chunk_pixels):
            c1 = min(p1 - p0, c0 + matmul_chunk_pixels)
            scores += a_block[:, c0:c1].astype(np.float64) @ e_block[:, c0:c1].astype(np.float64).T
    return scores


def phase_project_chunk(
    source: SplitSource,
    mean_flat,
    sqrt_w_flat,
    eofs_flat,
    k: int,
    chunk: tuple[int, int, int],
    work_dir: Path,
    split: str,
    ident: dict[str, Any],
    *,
    resume: bool,
    block_pixels: int,
    matmul_chunk_pixels: int,
) -> dict[str, Any]:
    np = _np()
    index, d0, d1 = chunk
    out = chunk_scores_path(work_dir, split, index)
    done = done_path(out)
    cid = {**ident, "phase": "project-chunk", "chunk_index": index, "d0": d0, "d1": d1}
    if resume and out.is_file() and done_matches(done, cid):
        return {"path": out, "skipped": True}
    sub = slice_source_days(source, d0, d1)
    scores = project_days_onto_basis(
        sub, mean_flat, sqrt_w_flat, eofs_flat, k,
        block_pixels=block_pixels, matmul_chunk_pixels=matmul_chunk_pixels,
    )
    if not np.all(np.isfinite(scores)):
        raise RuntimeError(f"non-finite scores in chunk {index} [{d0},{d1}) of split {split}")
    atomic_save_npy(out, scores.astype(np.float32))
    write_done(done, cid, {
        "bytes": out.stat().st_size,
        "scores_min": float(scores.min()),
        "scores_max": float(scores.max()),
        "scores_rms": float(np.sqrt(np.mean(scores ** 2))),
        "block_pixels": block_pixels,
        "matmul_chunk_pixels": matmul_chunk_pixels,
    })
    return {"path": out, "skipped": False}


def phase_assemble_scores(
    work_dir: Path,
    split: str,
    chunks: list[tuple[int, int, int]],
    ident: dict[str, Any],
    k: int,
    n_samples: int,
    out_npy: Path,
) -> dict[str, Any]:


    np = _np()
    from numpy.lib.format import open_memmap

    for chunk in chunks:
        index, d0, d1 = chunk
        cid = {**ident, "phase": "project-chunk", "chunk_index": index, "d0": d0, "d1": d1}
        if not done_matches(done_path(chunk_scores_path(work_dir, split, index)), cid):
            raise RuntimeError(f"assemble-scores refuses to run: missing/mismatched done JSON for chunk {index} of split {split}")

    out_npy.parent.mkdir(parents=True, exist_ok=True)
    tmp = unique_tmp(out_npy)
    scores = open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n_samples, k))
    for index, d0, d1 in chunks:
        rows = np.load(chunk_scores_path(work_dir, split, index))
        if rows.shape != (d1 - d0, k):
            raise RuntimeError(f"chunk {index} shape {rows.shape} != {(d1 - d0, k)}")
        scores[d0:d1] = rows
    finite = bool(np.all(np.isfinite(scores)))
    stats = {
        "scores_min": float(scores.min()),
        "scores_max": float(scores.max()),
        "scores_rms": float(np.sqrt(np.mean(np.asarray(scores, dtype=np.float64) ** 2))),
        "all_finite": finite,
    }
    column_norms = np.sqrt(np.sum(np.asarray(scores, dtype=np.float64) ** 2, axis=0))
    scores.flush()
    del scores
    if not finite:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"non-finite values in assembled scores for split {split}")
    os.replace(tmp, out_npy)

    aid = {**ident, "phase": "assemble-scores", "n_chunks": len(chunks), "output_shape": [n_samples, k]}
    extra = {"bytes": out_npy.stat().st_size, "sha256": sha256_file(out_npy), **stats}
    write_done(done_path(out_npy), aid, extra)
    return {"path": out_npy, "ident": aid, "stats": stats, "sha256": extra["sha256"], "column_norms": column_norms}


BASIS_MEMBERS = ("eofs_weighted.npy", "mean_field.npy", "latitude.npy", "longitude.npy", "singular_values.npy")


def check_basis(config: dict[str, Any], kind_cfg: dict[str, Any], log: IssueLog, *, manifest_sha256: str | None) -> dict[str, Any] | None:

    height, width = config["expected_grid_shape"]
    max_modes = int(kind_cfg["max_modes"])
    npz_path = resolve(Path(kind_cfg["basis_npz"]))
    done_json = resolve(Path(kind_cfg["basis_done_json"]))
    if not npz_path.is_file():
        log.fail(f"missing Stage 01 basis NPZ: {npz_path}")
        return None
    if not done_json.is_file():
        log.fail(f"missing Stage 01 basis done JSON: {done_json}")
        return None
    try:
        payload = load_json(done_json)
    except Exception as exc:
        log.fail(f"unreadable basis done JSON {done_json}: {exc}")
        return None
    ident = payload.get("ident") or {}
    expected_ident = kind_cfg.get("expected_basis_ident")
    if expected_ident is not None:
        if ident == expected_ident:
            log.pass_(f"basis done identity matches pinned expected_basis_ident ({kind_cfg['basis_label']})")
        else:
            log.fail(f"basis done identity does not match pinned expected_basis_ident: {done_json}")
            return None
    for key, want in (("stage", "01_build_eof_bases"), ("phase", "assemble"), ("mode", "production"),
                      ("kind", kind_cfg["basis_label"]), ("max_modes", max_modes), ("k", max_modes)):
        if ident.get(key) != want:
            log.fail(f"basis done ident {key}={ident.get(key)!r} expected {want!r}: {done_json}")
    if ident.get("grid_shape") != [height, width]:
        log.fail(f"basis done ident grid_shape {ident.get('grid_shape')} expected {[height, width]}")
    if int(ident.get("n_train", -1)) != int(config["expected_train_days"]):
        log.fail(f"basis done ident n_train {ident.get('n_train')} expected {config['expected_train_days']}")
    if manifest_sha256 is not None:
        if ident.get("train_manifest_sha256") == manifest_sha256:
            log.pass_("basis was fit on the SAME train manifest used for projection (sha256 match)")
        else:
            log.fail("basis train_manifest_sha256 does not match the projection source manifest hash")

    expected_bytes = kind_cfg.get("expected_basis_bytes")
    if expected_bytes is not None and npz_path.stat().st_size != int(expected_bytes):
        log.fail(f"basis NPZ size {npz_path.stat().st_size} != pinned expected_basis_bytes {expected_bytes}")

    try:
        eofs_info = npz_member_npy_info(npz_path, "eofs_weighted.npy")
        mean_info = npz_member_npy_info(npz_path, "mean_field.npy")
        lat_info = npz_member_npy_info(npz_path, "latitude.npy")
        for member in BASIS_MEMBERS:
            npz_member_npy_info(npz_path, member)
    except Exception as exc:
        log.fail(f"basis NPZ member inspection failed for {npz_path}: {exc}")
        return None
    if eofs_info["shape"] != (max_modes, height, width):
        log.fail(f"eofs_weighted shape {eofs_info['shape']} expected {(max_modes, height, width)}")
    elif not dtype_is_float32(eofs_info["descr"]) or eofs_info["fortran_order"]:
        log.fail(f"eofs_weighted dtype/order unexpected: {eofs_info['descr']} fortran={eofs_info['fortran_order']}")
    else:
        log.pass_(f"eofs_weighted header {eofs_info['shape']} float32 C-order (ZIP_STORED, memmap-able)")
    if mean_info["shape"] != (height, width) or not dtype_is_float32(mean_info["descr"]):
        log.fail(f"mean_field header {mean_info['shape']} {mean_info['descr']} expected {(height, width)} float32")
    if lat_info["shape"] != (height,):
        log.fail(f"latitude header shape {lat_info['shape']} expected {(height,)}")
    return {
        "npz_path": npz_path,
        "done_json": done_json,
        "done_sha256": sha256_file(done_json),
        "ident": ident,
        "npz_bytes": npz_path.stat().st_size,
        "eofs_shape": list(eofs_info["shape"]),
    }


def open_basis_arrays(config: dict[str, Any], basis: dict[str, Any], k: int):


    np = _np()
    height, width = config["expected_grid_shape"]
    npz_path = basis["npz_path"]
    eofs = npz_member_memmap(npz_path, "eofs_weighted.npy")
    if eofs.shape != (k, height, width):
        raise RuntimeError(f"eofs_weighted memmap shape {eofs.shape} expected {(k, height, width)}")
    mean_field = npz_small_member(npz_path, "mean_field.npy")
    if mean_field.shape != (height, width):
        raise RuntimeError(f"mean_field shape {mean_field.shape} expected {(height, width)}")
    lat_member = np.asarray(npz_small_member(npz_path, "latitude.npy"), dtype=np.float64)
    lat_formula = grid_lat_centers(height)
    lat_dev = float(np.max(np.abs(lat_member - lat_formula)))
    if lat_dev > 1e-4:
        raise RuntimeError(f"basis latitude deviates from grid formula by {lat_dev} deg (> 1e-4)")
    _, sqrt_w = area_weights_from_lat(lat_formula)
    sqrt_w_flat = np.repeat(sqrt_w[:, None], width, axis=1).reshape(-1)
    mean_flat = np.asarray(mean_field, dtype=np.float64).reshape(-1)
    return eofs.reshape(k, height * width), mean_flat, sqrt_w_flat, lat_dev


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
    probe.write_text("stage02 scratch write probe\n", encoding="utf-8")
    probe.unlink()


def scratch_write_probe(heavy_output_root: Path, tmp_root: Path, log: IssueLog) -> None:
    if os.environ.get(WRITE_PROBE_ENV) != "1":
        log.fail(f"scratch write probe requested but {WRITE_PROBE_ENV}=1 is not set")
        return
    for root, label in [(heavy_output_root, "heavy_output_root"), (tmp_root, "tmp_root")]:
        try:
            scratch_probe_dir(root)
        except RuntimeError as exc:
            log.fail(str(exc))
            continue
        log.pass_(f"scratch write probe succeeded for {label}: {root}")


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

    tmp_root = guard_tmp_root(config.get("tmp_root", ""), log)

    for label, raw in (config.get("intended_outputs") or {}).items():
        path = resolve(Path(str(raw)))
        if not is_relative_to(path, home_output_root):
            log.fail(f"home intended output {label} is outside home output root: {path}")
        if path.suffix in {".npy", ".npz", ".zarr"}:
            log.fail(f"home intended output {label} looks like a heavy array artifact: {path}")
    if config.get("intended_outputs"):
        log.pass_("home intended outputs are lightweight manifest/status artifacts")

    for label, raw in (config.get("intended_heavy_outputs") or {}).items():
        path = resolve(Path(str(raw)))
        if not is_relative_to(path, heavy_output_root):
            log.fail(f"heavy intended output {label} is outside heavy output root: {path}")
    if config.get("intended_heavy_outputs"):
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
    if config.get("stage") != STAGE_NAME:
        log.fail(f"config stage must be {STAGE_NAME}, got {config.get('stage')!r}")
    if config.get("no_test_access") is not True:
        log.fail("no_test_access must be true")
    if config.get("no_new_eof_fit") is not True:
        log.fail("no_new_eof_fit must be true")
    if config.get("no_mean_refit") is not True:
        log.fail("no_mean_refit must be true")
    if config.get("expected_grid_shape") != [1280, 2624]:
        log.fail(f"expected_grid_shape mismatch: {config.get('expected_grid_shape')}")
    if int(config.get("expected_train_days", -1)) != 10593:
        log.fail(f"expected_train_days mismatch: {config.get('expected_train_days')}")
    if int(config.get("expected_val_days", -1)) != 1095:
        log.fail(f"expected_val_days mismatch: {config.get('expected_val_days')}")

    check_storage_policy(config, home_output_root, heavy_output_root, log)

    kind_cfg = (config.get("kind_configs") or {}).get(kind)
    if not kind_cfg:
        log.fail(f"missing kind_configs.{kind}")
        return None
    configured_max = int(kind_cfg.get("max_modes", -1))
    if max_modes != configured_max:
        log.fail(f"--max-modes {max_modes} does not match configured {kind} max_modes {configured_max}")
    work_dir = resolve(Path(kind_cfg.get("work_dir", "")))
    if not is_relative_to(work_dir, heavy_output_root):
        log.fail(f"stage02 work_dir {work_dir} is outside heavy output root {heavy_output_root}")
    else:
        log.pass_(f"{kind} work_dir planned under scratch heavy root")
    outputs = kind_cfg.get("outputs") or {}
    for split in VALID_SPLITS:
        raw = outputs.get(split)
        if not raw:
            log.fail(f"missing kind_configs.{kind}.outputs.{split}")
            continue
        out = resolve(Path(raw))
        if not is_relative_to(out, heavy_output_root):
            log.fail(f"{kind} {split} scores output {out} is outside heavy output root")


    if outputs:
        log.pass_(f"{kind} split score outputs planned under scratch heavy root")
    return kind_cfg


def check_manifest_counts(config: dict[str, Any], kind_cfg: dict[str, Any], log: IssueLog) -> dict[str, Any]:
    manifest = Path(config["source_train_manifest"])
    reject_suspicious_path("source_train_manifest", manifest, log)
    column = kind_cfg["source_manifest_column"]
    expected_h, expected_w = config["expected_grid_shape"]
    summary: dict[str, Any] = {"manifest": str(manifest)}
    expectations = {
        "train": (int(config["expected_train_days"]), int(config["expected_train_shards"])),
        "val": (int(config["expected_val_days"]), int(config["expected_val_shards"])),
    }
    for split in VALID_SPLITS:
        rows, n_days = load_manifest_split(manifest, split, log)
        want_days, want_shards = expectations[split]
        if len(rows) == want_shards:
            log.pass_(f"{split} shard count {len(rows)}")
        else:
            log.fail(f"{split} shard count {len(rows)} expected {want_shards}")
        if n_days == want_days:
            log.pass_(f"{split} day count {n_days}")
        else:
            log.fail(f"{split} day count {n_days} expected {want_days}")
        missing = 0
        for index, row in enumerate(rows):
            if int(row.get("height", -1)) != expected_h or int(row.get("width", -1)) != expected_w:
                log.fail(f"{split} manifest grid mismatch row {index}: {row.get('height')}x{row.get('width')}")
            path = Path(row[column])
            reject_suspicious_path(f"{split} row {index} {column}", path, log)
            if not path.is_file():
                log.fail(f"missing {split} shard row {index}: {path}")
                missing += 1
        summary[split] = {"rows": len(rows), "days": n_days, "missing_shards": missing}
    if manifest.is_file():
        summary["manifest_sha256"] = sha256_file(manifest)
    return summary


def print_plan(
    config: dict[str, Any],
    kind: str,
    kind_cfg: dict[str, Any],
    max_modes: int,
    home_output_root: Path,
    heavy_output_root: Path,
    tmp_root: Path,
    manifest_summary: dict[str, Any],
    basis: dict[str, Any] | None,
) -> None:
    algo = config.get("algorithm", {})
    chunk_days = int(algo.get("chunk_days", 1024))
    print("STAGE02_DRY_RUN_PLAN")
    print(f"created_utc={utc_now()}")
    print(f"kind={kind}")
    print(f"max_modes={max_modes}")
    print(f"home_manifest_output_root={home_output_root}")
    print(f"heavy_output_root={heavy_output_root}")
    print(f"tmp_root={tmp_root}")
    print(f"source_manifest_column={kind_cfg['source_manifest_column']}")
    print(f"basis_npz={kind_cfg['basis_npz']}")
    print(f"basis_done_json={kind_cfg['basis_done_json']}")
    if basis:
        print(f"basis_done_sha256={basis['done_sha256']}")
        print(f"basis_npz_bytes={basis['npz_bytes']}")
        print(f"basis_eofs_shape={basis['eofs_shape']}")
    print(f"work_dir={kind_cfg['work_dir']}")
    for split in VALID_SPLITS:
        n = int(manifest_summary[split]["days"])
        chunks = day_chunks(n, chunk_days)
        out = kind_cfg["outputs"][split]
        print(f"{split}_days={n}")
        print(f"{split}_chunks={len(chunks)} (array 0-{len(chunks) - 1}) chunk_days={chunk_days}")
        print(f"{split}_scores_output={out}")
        print(f"{split}_scores_shape=[{n}, {max_modes}] float32 ({n * max_modes * 4} bytes)")
    print(f"home_projection_metadata={config.get('intended_outputs', {}).get('projection_metadata_json')}")
    print(f"home_heavy_artifact_manifest={config.get('intended_outputs', {}).get('heavy_artifact_manifest_json')}")
    print(f"home_stage_status={config.get('intended_outputs', {}).get('stage_status_json')}")
    print("phases=plan,project-chunk,assemble-scores")
    print("projection_formula=score = ((field - train_mean) * sqrt_weight) dot eofs_weighted.T")
    print("mean_source=mean_field member of the Stage 01 basis NPZ (no mean refit)")
    print("weights=cos(lat_center)/mean(cos(lat_center)) from grid-formula lat, identical to Stage 01; basis latitude cross-checked <=1e-4 deg")
    print("basis_access=ZIP_STORED member memmap; eofs_weighted is never fully loaded")
    print("no_new_eof_fit=true no_mean_refit=true validation_projection_allowed=true sealed_test_untouched=true")
    print(f"production_execution_approved={config.get('production_execution_approved')}")
    print(f"heavy_phase_env_guard={REAL_RUN_ENV}=1 required")


def heavy_ident(
    config_path: Path,
    config: dict[str, Any],
    kind: str,
    split: str,
    max_modes: int,
    chunk_days: int,
    n_samples: int,
    manifest_sha256: str,
    basis: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": STAGE_NAME,
        "mode": "production",
        "kind": kind,
        "split": split,
        "config_sha256": sha256_file(config_path),
        "source_manifest_sha256": manifest_sha256,
        "basis_npz": str(basis["npz_path"]),
        "basis_done_sha256": basis["done_sha256"],
        "grid_shape": list(config["expected_grid_shape"]),
        "n_samples": int(n_samples),
        "max_modes": int(max_modes),
        "chunk_days": int(chunk_days),
    }


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
    split = args.split
    if split not in VALID_SPLITS:
        print(f"FAIL heavy phase requires --split train|val, got {split!r}")
        return 2

    height, width = config["expected_grid_shape"]
    max_modes = int(args.max_modes)
    algo = config.get("algorithm", {})
    chunk_days = int(algo.get("chunk_days", 1024))
    block_pixels = int(args.pixel_block_pixels or algo.get("pixel_block_pixels", 65536))
    matmul_chunk_pixels = int(args.matmul_chunk_pixels)
    work_dir = resolve(Path(kind_cfg["work_dir"]))
    out_npy = resolve(Path(kind_cfg["outputs"][split]))

    log = IssueLog()
    manifest = Path(config["source_train_manifest"])
    manifest_sha = sha256_file(manifest)
    basis = check_basis(config, kind_cfg, log, manifest_sha256=manifest_sha)
    rows, n_samples = load_manifest_split(manifest, split, log)
    expected_days = int(config["expected_train_days"] if split == "train" else config["expected_val_days"])
    if n_samples != expected_days:
        log.fail(f"{split} day count {n_samples} != expected {expected_days}")
    if basis is None or not log.ok():
        for msg in log.failures:
            print(f"FAIL {msg}")
        return 2

    ident = heavy_ident(config_path, config, args.kind, split, max_modes, chunk_days, n_samples, manifest_sha, basis)
    chunks = day_chunks(n_samples, chunk_days)
    if args.chunk_index is not None and not 0 <= args.chunk_index < len(chunks):
        print(f"FAIL --chunk-index {args.chunk_index} out of range 0..{len(chunks) - 1} for {len(chunks)} day chunks of split {split}")
        return 2

    scratch_probe_dir(work_dir)

    t0 = time.time()
    if phase == "project-chunk":
        eofs_flat, mean_flat, sqrt_w_flat, lat_dev = open_basis_arrays(config, basis, max_modes)
        print(f"BASIS_OPENED memmap_shape=({max_modes},{height * width}) latitude_vs_formula_max_abs_deg={lat_dev:.3e}", flush=True)
        source = open_split_source(rows, kind_cfg["source_manifest_column"], height, width)
        if source.n_total != n_samples:
            print(f"FAIL source day count {source.n_total} != manifest count {n_samples}")
            return 2
        todo = chunks if args.chunk_index is None else [chunks[args.chunk_index]]
        for chunk in todo:
            result = phase_project_chunk(
                source, mean_flat, sqrt_w_flat, eofs_flat, max_modes, chunk, work_dir, split, ident,
                resume=args.resume, block_pixels=block_pixels, matmul_chunk_pixels=matmul_chunk_pixels,
            )
            print(f"PROJECT_CHUNK_DONE split={split} index={chunk[0]} days=[{chunk[1]},{chunk[2]}) "
                  f"skipped={result['skipped']} seconds={time.time() - t0:.1f}", flush=True)
    elif phase == "assemble-scores":
        aid = {**ident, "phase": "assemble-scores", "n_chunks": len(chunks), "output_shape": [n_samples, max_modes]}
        if args.resume and out_npy.is_file() and done_matches(done_path(out_npy), aid):
            print(f"ASSEMBLE_SCORES_SKIPPED split={split} output already complete with matching identity: {out_npy}")
            return 0
        if out_npy.exists() and not args.overwrite:
            print(f"FAIL final scores output exists and --overwrite was not set: {out_npy}")
            return 2
        assembled = phase_assemble_scores(work_dir, split, chunks, ident, max_modes, n_samples, out_npy)
        norm_check = train_column_norm_check(basis, assembled, max_modes) if split == "train" else None
        if norm_check is not None and not norm_check["pass"]:
            print(f"FAIL train scores column norms deviate from Stage 01 singular values: "
                  f"max_rel_dev={norm_check['max_rel_dev']:.3e} (tolerance {norm_check['tolerance']})")
            return 2
        write_projection_home_manifests(config, args.kind, split, assembled, basis, norm_check, home_output_root)
        print(f"ASSEMBLE_SCORES_DONE split={split} npy={out_npy} sha256={assembled['sha256'][:16]}... "
              f"seconds={time.time() - t0:.1f}")
        if norm_check is not None:
            print(f"TRAIN_COLUMN_NORM_CHECK max_rel_dev={norm_check['max_rel_dev']:.3e} pass={norm_check['pass']}")
    else:
        print(f"FAIL unknown heavy phase {phase!r}")
        return 2
    return 0


def train_column_norm_check(basis: dict[str, Any], assembled: dict[str, Any], k: int) -> dict[str, Any]:


    np = _np()
    singular = np.asarray(npz_small_member(basis["npz_path"], "singular_values.npy"), dtype=np.float64)[:k]
    norms = np.asarray(assembled["column_norms"], dtype=np.float64)
    good = singular > 0
    rel = np.abs(norms[good] - singular[good]) / singular[0]
    max_rel = float(rel.max()) if good.any() else float("nan")
    tolerance = 1e-3
    return {
        "max_rel_dev": max_rel,
        "tolerance": tolerance,
        "pass": bool(max_rel < tolerance),
        "zero_singular_modes": int((~good).sum()),
    }


def write_projection_home_manifests(
    config: dict[str, Any],
    kind: str,
    split: str,
    assembled: dict[str, Any],
    basis: dict[str, Any],
    norm_check: dict[str, Any] | None,
    home_output_root: Path,
) -> None:

    intended = config["intended_outputs"]
    out_npy = Path(assembled["path"])

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
        "split": split,
        "ident": assembled["ident"],
        "output_npy": str(out_npy),
        "output_npy_sha256": assembled["sha256"],
        "output_npy_bytes": out_npy.stat().st_size,
        "output_shape": assembled["ident"]["output_shape"],
        "dtype": "float32",
        "basis_npz": str(basis["npz_path"]),
        "basis_done_sha256": basis["done_sha256"],
        "stats": assembled["stats"],
        "train_column_norm_check": norm_check,
        "projection_formula": "score = ((field - train_mean) * sqrt_weight) dot eofs_weighted.T",
        "no_new_eof_fit": True,
        "no_mean_refit": True,
        "test_split_accessed": False,
        "created_utc": utc_now(),
    }

    def upd_meta(payload: dict[str, Any]) -> None:
        payload.setdefault("stage", STAGE_NAME)
        payload.setdefault("kinds", {}).setdefault(kind, {})[split] = entry

    def upd_status(payload: dict[str, Any]) -> None:
        payload.setdefault("stage", STAGE_NAME)
        payload.setdefault("kinds", {}).setdefault(kind, {})[split] = {
            "phase": "assemble-scores", "status": "complete", "updated_utc": utc_now(),
        }

    def upd_manifest(payload: dict[str, Any]) -> None:
        payload.setdefault("stage", STAGE_NAME)
        payload.setdefault("scratch_lifetime_warning",
                           "Scratch files have a 14-day lifetime since last access and are not archival. "
                           "Copy final heavy artifacts to an approved archival target.")
        payload.setdefault("heavy_artifacts", {}).setdefault(kind, {})[split] = {
            "path": str(out_npy),
            "sha256": assembled["sha256"],
            "bytes": out_npy.stat().st_size,
            "location": "scratch_heavy_data_plane",
        }

    merge("projection_metadata_json", upd_meta)
    merge("stage_status_json", upd_status)
    merge("heavy_artifact_manifest_json", upd_manifest)


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

    home_output_root = guard_home_output_root(args.output_root or config.get("home_output_root") or ALLOWED_HOME_OUTPUT_ROOT, log)
    heavy_output_root = guard_heavy_output_root(args.heavy_output_root or config.get("heavy_output_root") or DEFAULT_HEAVY_OUTPUT_ROOT, log)

    kind_cfg = check_config(config, args.kind, args.max_modes, home_output_root, heavy_output_root, args.overwrite, log)
    tmp_root = resolve(Path(config.get("tmp_root", "")))
    if args.write_probe:
        scratch_write_probe(heavy_output_root, tmp_root, log)
    if kind_cfg is None:
        print_summary(log)
        return 2

    manifest_summary = check_manifest_counts(config, kind_cfg, log)
    basis = check_basis(config, kind_cfg, log, manifest_sha256=manifest_summary.get("manifest_sha256"))

    print_summary(log)
    if not log.ok():
        return 2

    if args.dry_run or args.phase == "plan":
        print_plan(config, args.kind, kind_cfg, args.max_modes, home_output_root, heavy_output_root, tmp_root, manifest_summary, basis)
        if args.phase != "plan":
            print(f"DRY_RUN phase {args.phase!r} NOT executed because --dry-run was set")
        print("DRY_RUN no outputs written; no dynamic arrays loaded; eofs_weighted not loaded (header reads only)")
        return 0

    if args.phase in HEAVY_PHASES:
        return run_heavy_phase(args, config, config_path, kind_cfg, home_output_root, heavy_output_root)

    print(f"FAIL unhandled phase {args.phase!r}")
    return 2


def print_summary(log: IssueLog) -> None:
    for msg in log.failures:
        print(f"FAIL {msg}")
    for msg in log.warnings:
        print(f"WARN {msg}")
    for msg in log.passes:
        print(f"PASS {msg}")
    print(f"SUMMARY pass={len(log.passes)} warn={len(log.warnings)} fail={len(log.failures)}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 02 EOF score projection (phased; heavy phases double-gated).")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Stage 02 projection config JSON.")
    parser.add_argument("--kind", choices=sorted(VALID_KINDS), required=True, help="Projection side.")
    parser.add_argument("--max-modes", type=int, required=True, help="EOF mode count; must match config and the Stage 01 basis.")
    parser.add_argument("--split", choices=VALID_SPLITS, help="Split to project (required for heavy phases).")
    parser.add_argument("--output-root", help="Home manifest output root; must be inside the clean home outputs tree.")
    parser.add_argument("--heavy-output-root", help=f"Scratch heavy output root; must be under {RESULTS_ROOT} and not under /home or /work.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print plan without writing outputs or loading arrays.")
    parser.add_argument("--resume", action="store_true", help="Reuse completed chunk artifacts whose done JSON identity matches.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing final score outputs. Default false.")
    parser.add_argument("--write-probe", action="store_true", help=f"Tiny scratch write probe; requires {WRITE_PROBE_ENV}=1.")
    parser.add_argument(
        "--phase",
        choices=["plan", "project-chunk", "assemble-scores"],
        default="plan",
        help=f"Phase selector. Heavy phases additionally require {REAL_RUN_ENV}=1 and config production_execution_approved=true.",
    )
    parser.add_argument("--chunk-index", type=int, help="Day-chunk index for project-chunk (Slurm array task id). Omit to run all chunks sequentially.")
    parser.add_argument("--pixel-block-pixels", type=int, help="Pixel block size for basis/shard streaming; default from config algorithm.pixel_block_pixels.")
    parser.add_argument("--matmul-chunk-pixels", type=int, default=8192, help="float64 pixel chunk size inside score accumulation.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
