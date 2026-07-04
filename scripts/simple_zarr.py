from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _dtype_from_meta(data_type: Any, endian: str = "little") -> np.dtype:
    text = str(data_type).lower()
    prefix = "<" if endian == "little" else ">"
    mapping = {
        "bool": "?",
        "uint8": "u1",
        "uint16": f"{prefix}u2",
        "uint32": f"{prefix}u4",
        "uint64": f"{prefix}u8",
        "int8": "i1",
        "int16": f"{prefix}i2",
        "int32": f"{prefix}i4",
        "int64": f"{prefix}i8",
        "float32": f"{prefix}f4",
        "float64": f"{prefix}f8",
    }
    if text not in mapping:
        raise ValueError(f"Unsupported Zarr dtype: {data_type}")
    return np.dtype(mapping[text])


def _codec_names(codecs: list[dict[str, Any]]) -> list[str]:
    return [str(codec.get("name", "")).lower() for codec in codecs]


def _decode_blosc(data: bytes) -> bytes:
    try:
        from numcodecs import Blosc  # type: ignore

        return Blosc().decode(data)
    except Exception:
        try:
            import blosc2  # type: ignore

            return blosc2.decompress(data)
        except Exception as exc:
            raise ModuleNotFoundError(
                "Need numcodecs or blosc2 to decode Biohub image Zarr chunks."
            ) from exc


def _decode_zstd(data: bytes) -> bytes:
    try:
        from numcodecs import Zstd  # type: ignore

        return Zstd().decode(data)
    except Exception:
        try:
            import zstandard  # type: ignore

            return zstandard.ZstdDecompressor().decompress(data)
        except Exception as exc:
            raise ModuleNotFoundError(
                "Need numcodecs or zstandard to decode GEFF Zstd chunks."
            ) from exc


def _decode_chunk(data: bytes, codecs: list[dict[str, Any]]) -> bytes:
    decoded = data
    names = _codec_names(codecs)
    if "blosc" in names:
        decoded = _decode_blosc(decoded)
    if "zstd" in names:
        decoded = _decode_zstd(decoded)
    return decoded


class SimpleZarr3Array:
    def __init__(self, array_path: Path):
        self.array_path = Path(array_path)
        with (self.array_path / "zarr.json").open(encoding="utf-8") as f:
            self.meta = json.load(f)

        self.shape = tuple(int(v) for v in self.meta["shape"])
        chunk_grid = self.meta["chunk_grid"]["configuration"]
        self.chunk_shape = tuple(int(v) for v in chunk_grid["chunk_shape"])
        endian = "little"
        for codec in self.meta.get("codecs", []):
            if codec.get("name") == "bytes":
                endian = codec.get("configuration", {}).get("endian", "little")
        self.dtype = _dtype_from_meta(self.meta.get("data_type") or self.meta.get("dtype"), endian=endian)
        self.ndim = len(self.shape)
        self.attrs = self.meta.get("attributes", {})

    def _chunk_path(self, chunk_indices: tuple[int, ...]) -> Path:
        return self.array_path / "c" / Path(*[str(v) for v in chunk_indices])

    def _read_chunk(self, chunk_indices: tuple[int, ...]) -> np.ndarray:
        chunk_path = self._chunk_path(chunk_indices)
        if not chunk_path.exists():
            return np.full(self.chunk_shape, self.meta.get("fill_value", 0), dtype=self.dtype)
        decoded = _decode_chunk(chunk_path.read_bytes(), self.meta.get("codecs", []))
        return np.frombuffer(decoded, dtype=self.dtype).reshape(self.chunk_shape)

    def read_all(self) -> np.ndarray:
        out = np.full(self.shape, self.meta.get("fill_value", 0), dtype=self.dtype)
        chunk_counts = tuple(int(math.ceil(s / c)) for s, c in zip(self.shape, self.chunk_shape))
        for chunk_indices in np.ndindex(chunk_counts):
            chunk = self._read_chunk(tuple(int(v) for v in chunk_indices))
            dst_slices = []
            src_slices = []
            for axis, chunk_index in enumerate(chunk_indices):
                start = int(chunk_index) * self.chunk_shape[axis]
                stop = min(start + self.chunk_shape[axis], self.shape[axis])
                dst_slices.append(slice(start, stop))
                src_slices.append(slice(0, stop - start))
            out[tuple(dst_slices)] = chunk[tuple(src_slices)]
        return out

    def __array__(self, dtype=None) -> np.ndarray:
        array = self.read_all()
        if dtype is not None:
            return array.astype(dtype, copy=False)
        return array

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            return self.read_all()[item]
        if (
            self.ndim == 4
            and len(item) == 4
            and isinstance(item[0], (int, np.integer))
            and item[1] == slice(None)
            and item[2] == slice(None)
            and item[3] == slice(None)
        ):
            t = int(item[0])
            chunk = self._read_chunk((t, 0, 0, 0))
            return chunk[0, : self.shape[1], : self.shape[2], : self.shape[3]]
        return self.read_all()[item]


class SimpleZarr3Group:
    def __init__(self, group_path: Path):
        self.group_path = Path(group_path)
        with (self.group_path / "zarr.json").open(encoding="utf-8") as f:
            self.meta = json.load(f)
        self.attrs = self.meta.get("attributes", {})

    def __getitem__(self, key: str) -> SimpleZarr3Array:
        return SimpleZarr3Array(self.group_path / key)


def open_array(path: Path, mode: str = "r") -> SimpleZarr3Array:
    if mode != "r":
        raise ValueError("simple_zarr only supports read mode")
    return SimpleZarr3Array(Path(path))


def open_group(path: Path, mode: str = "r") -> SimpleZarr3Group:
    if mode != "r":
        raise ValueError("simple_zarr only supports read mode")
    return SimpleZarr3Group(Path(path))
