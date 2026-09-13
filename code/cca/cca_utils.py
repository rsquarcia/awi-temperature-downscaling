"""Shared CCA file handling, validation records, and latitude-grid helpers.
These routines are moved unchanged from the numbered research scripts.
NumPy is imported lazily, preserving the original numerical import timing.
Stage-specific input guards and production approval checks remain in each runner."""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import os
import struct
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SMALL_MEMBER_MAX_BYTES = 64 << 20


@dataclass
class IssueLog:
    passes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def pass_(self, msg: str) -> None:
        self.passes.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def fail(self, msg: str) -> None:
        self.failures.append(msg)

    def ok(self) -> bool:
        return not self.failures


def _np():
    import numpy as np

    return np


def area_weights_from_lat(lat_centers):


    np = _np()
    w = np.cos(np.deg2rad(np.asarray(lat_centers, dtype=np.float64)))
    w = w / w.mean()
    return w, np.sqrt(w)


def atomic_savez(path: Path, **arrays: Any) -> None:
    np = _np()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = unique_tmp(path)
    with tmp.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(tmp, path)


def done_matches(path: Path, ident: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        payload = load_json(path)
    except Exception:
        return False
    return payload.get("ident") == ident


def done_path(artifact: Path) -> Path:
    return artifact.with_name(artifact.name + ".done.json")


def dtype_is_float32(descr: Any) -> bool:
    return str(descr) in {"<f4", "|f4", "float32"}


def fmt_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def grid_lat_centers(height: int):
    np = _np()
    return -90.0 + (np.arange(height, dtype=np.float64) + 0.5) * (180.0 / height)


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def npz_member_memmap(path: Path, member: str):

    np = _np()
    info = npz_member_npy_info(path, member)
    if info["fortran_order"]:
        raise RuntimeError(f"npz member {member} is fortran-ordered; memmap path assumes C order")
    return np.memmap(path, dtype=np.dtype(info["descr"]), mode="r", offset=info["data_offset"], shape=info["shape"])


def npz_member_npy_info(path: Path, member: str) -> dict[str, Any]:

    data_offset, member_bytes = zip_member_data_offset(path, member)
    with path.open("rb") as handle:
        handle.seek(data_offset)
        header = parse_npy_header(handle)
    header["data_offset"] = data_offset + header["header_bytes"]
    header["member_bytes"] = member_bytes
    return header


def npz_small_member(path: Path, member: str, max_bytes: int = SMALL_MEMBER_MAX_BYTES):


    np = _np()
    info = npz_member_npy_info(path, member)
    if info["member_bytes"] > max_bytes:
        raise RuntimeError(
            f"refusing to load npz member {member} ({info['member_bytes']} bytes > cap {max_bytes}); "
            "large members must be accessed via npz_member_memmap"
        )
    with zipfile.ZipFile(path) as archive:
        payload = archive.read(member)
    return np.load(io.BytesIO(payload), allow_pickle=False)


def output_state(out_path: Path, ident: dict[str, Any], *, resume: bool, overwrite: bool) -> str:

    if out_path.exists():
        if resume and done_matches(done_path(out_path), ident):
            return "skip"
        if overwrite:
            return "proceed"
        return "refuse"
    return "proceed"


def parse_npy_header(handle) -> dict[str, Any]:

    magic = handle.read(6)
    if magic != b"\x93NUMPY":
        raise ValueError("not npy data at expected offset")
    major, minor = handle.read(2)
    if major == 1:
        header_len = struct.unpack("<H", handle.read(2))[0]
        preamble = 10
    elif major in {2, 3}:
        header_len = struct.unpack("<I", handle.read(4))[0]
        preamble = 12
    else:
        raise ValueError(f"unsupported npy version {major}.{minor}")
    parsed = ast.literal_eval(handle.read(header_len).decode("latin1"))
    return {
        "descr": parsed.get("descr"),
        "fortran_order": bool(parsed.get("fortran_order")),
        "shape": tuple(parsed.get("shape") or ()),
        "header_bytes": preamble + header_len,
    }


def read_npy_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return parse_npy_header(handle)


def resolve(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def spatial_blocks(p_total: int, block_pixels: int) -> list[tuple[int, int, int]]:
    return [
        (index, p0, min(p_total, p0 + block_pixels))
        for index, p0 in enumerate(range(0, p_total, block_pixels))
    ]


def unique_tmp(path: Path) -> Path:

    tag = f"{os.environ.get('SLURM_JOB_ID', 'nojob')}-{os.getpid()}-{time.time_ns()}"
    return path.with_name(f"{path.name}.tmp.{tag}")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_csv_atomic(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([fmt_cell(row.get(col, "")) for col in columns])
    write_text_atomic(path, buf.getvalue())


def write_done(path: Path, ident: dict[str, Any], extra: dict[str, Any]) -> None:
    write_json_atomic(path, {"ident": ident, "created_utc": utc_now(), **extra})


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = unique_tmp(path)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = unique_tmp(path)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def zip_member_data_offset(path: Path, member: str) -> tuple[int, int]:

    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise RuntimeError(f"npz member {member} in {path} is not ZIP_STORED; cannot memmap")
        header_offset = info.header_offset
        member_bytes = info.file_size
    with path.open("rb") as handle:
        handle.seek(header_offset)
        local = handle.read(30)
        if local[:4] != b"PK\x03\x04":
            raise RuntimeError(f"bad local zip header for {member} in {path}")
        name_len = int.from_bytes(local[26:28], "little")
        extra_len = int.from_bytes(local[28:30], "little")
    return header_offset + 30 + name_len + extra_len, member_bytes
