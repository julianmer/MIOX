####################################################################################################
#                                            jnifti.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 13/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for JNIfTI images (NeuroJSON): the NIfTI header as JSON fields and the voxels    #
#          as a JData array, in JSON text (.jnii) or binary BJData (.bnii), with the array raw,    #
#          zlib, gzip or lzma compressed.                                                          #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import base64
import gzip
import json
import lzma
import struct
import zlib

import numpy as np

# own
from miox.data import MRImage
from miox.meta.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                              BJData                                              #
#**************************************************************************************************#
#                                                                                                  #
# Binary JData (BJData draft 2, UBJSON with little-endian numbers and N-D typed arrays): values    #
# start with a one-byte marker; strings and object keys carry their length as an integer value;    #
# "[" and "{" containers may declare a type ("$") and a count ("#", an integer or "[dims]"), in    #
# which case typed numbers follow without markers and are returned as one numpy array.             #
#                                                                                                  #
#**************************************************************************************************#
class BJData:
    NUMBERS = {b"i": "<i1", b"U": "<u1", b"I": "<i2", b"u": "<u2", b"l": "<i4", b"m": "<u4",
               b"L": "<i8", b"M": "<u8", b"h": "<f2", b"d": "<f4", b"D": "<f8", b"B": "<u1"}
    CONSTANTS = {b"Z": None, b"T": True, b"F": False}

    def __init__(self, blob: bytes):
        self.blob, self.at = blob, 0

    @classmethod
    def loads(cls, blob: bytes):
        return cls(blob).value()

    def take(self, n: int) -> bytes:
        if self.at + n > len(self.blob):
            raise ValueError("BJData ends early")
        out = self.blob[self.at:self.at + n]
        self.at += n
        return out

    def marker(self) -> bytes:
        while (byte := self.take(1)) == b"N":  # no-op
            pass
        return byte

    def number(self, marker: bytes):
        dtype = np.dtype(self.NUMBERS[marker])
        return np.frombuffer(self.take(dtype.itemsize), dtype)[0].item()

    def integer(self) -> int:
        return int(self.number(self.marker()))

    def text(self) -> str:
        return self.take(self.integer()).decode("utf-8")

    def value(self, marker: bytes | None = None):
        marker = marker or self.marker()
        if marker in self.CONSTANTS:
            return self.CONSTANTS[marker]
        if marker in self.NUMBERS:
            return self.number(marker)
        if marker == b"C":
            return self.take(1).decode("latin-1")
        if marker in (b"S", b"H"):
            text = self.text()
            return text if marker == b"S" else float(text)
        if marker == b"[":
            return self.array()
        if marker == b"{":
            return self.object()
        raise ValueError(f"unknown BJData marker {marker!r} at byte {self.at - 1}")

    def header(self) -> tuple[bytes | None, list[int] | None]:
        """The optional type ("$") and count ("#", an integer or [dims]) of a container."""
        kind = shape = None
        if self.blob[self.at:self.at + 1] == b"$":
            self.at += 1
            kind = self.take(1)
        if self.blob[self.at:self.at + 1] == b"#":
            self.at += 1
            if self.blob[self.at:self.at + 1] == b"[":
                self.at += 1
                shape = [int(v) for v in self.array()]
            else:
                shape = [self.integer()]
        return kind, shape

    def array(self):
        kind, shape = self.header()
        if shape is None:
            items = []
            while (marker := self.marker()) != b"]":
                items.append(self.value(marker))
            return items
        count = int(np.prod(shape))
        if kind in self.NUMBERS:
            dtype = np.dtype(self.NUMBERS[kind])
            values = np.frombuffer(self.take(count * dtype.itemsize), dtype)
            return values.reshape(shape) if len(shape) > 1 else values
        return [self.value(kind) for _ in range(count)]

    def object(self) -> dict:
        kind, shape = self.header()
        out = {}
        if shape is None:
            while self.blob[self.at:self.at + 1] != b"}":
                key = self.text()
                out[key] = self.value()
            self.at += 1
            return out
        for _ in range(shape[0]):
            key = self.text()
            out[key] = self.value(kind)
        return out


#**************************************************************************************************#
#                                           JNiftiReader                                           #
#**************************************************************************************************#
#                                                                                                  #
# NIFTIHeader holds the NIfTI fields by name; NIFTIData is a JData array: _ArrayType_ and          #
# _ArraySize_, the values in _ArrayData_ (a flat or nested list) or compressed in _ArrayZipData_   #
# (base64 in JSON, bytes in BJData; _ArrayZipType_ zlib, gzip or lzma), row-major unless           #
# _ArrayOrder_ says "c"; _ArrayIsComplex_ stores real and imaginary rows. The geometry is the      #
# sform (Affine, three rows) when SForm is set, else the qform (Quatern, QuaternOffset, VoxelSize, #
# the sign of the third axis from Orientation), else the voxel sizes. ScaleSlope and ScaleOffset   #
# apply as in NIfTI (a slope of 0 means none).                                                     #
#                                                                                                  #
#**************************************************************************************************#
class JNiftiReader(Reader):
    FORMATS = (Format.JNIFTI, Format.JNIFTI_BINARY)
    TYPES = {"int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2", "int32": "i4",
             "uint32": "u4", "int64": "i8", "uint64": "u8", "single": "f4", "double": "f8",
             "float32": "f4", "float64": "f8", "logical": "u1", "bool": "u1"}

    def read(self, sniff: Sniff) -> list[MRImage]:
        try:
            blob = sniff.path.read_bytes()
            tree = (BJData.loads(blob) if sniff.format == Format.JNIFTI_BINARY
                    else json.loads(blob.decode("utf-8"), strict=False))  # raw newlines in text
            header = tree["NIFTIHeader"]
            data = self.array(tree["NIFTIData"])
        except (ValueError, KeyError, TypeError, OSError, lzma.LZMAError, zlib.error,
                struct.error) as e:
            raise ReaderError(f"{sniff.path.name}: {type(e).__name__}: {e}") from e
        slope = float(header.get("ScaleSlope", 0) or 0)
        if slope:
            data = data * slope + float(header.get("ScaleOffset", 0) or 0)
        data = data.reshape(data.shape + (1,) * (3 - data.ndim))
        notes: list[str] = []
        affine = self.affine(header, notes)
        meta = Meta({"header": self.plain(header), "info": self.plain(tree.get("_DataInfo_", {}))})
        dims = ("volume",) * max(data.ndim - 3, 0)
        name = str(header.get("Name") or sniff.path.name.rsplit(".", 1)[0])
        return [MRImage(data, affine, meta, self.provenance(sniff, tuple(notes)), name, dims)]

    @classmethod
    def array(cls, node) -> np.ndarray:
        if not isinstance(node, dict):
            return np.asarray(node)
        dtype = np.dtype(cls.TYPES[node["_ArrayType_"]]).newbyteorder("<")
        shape = [int(v) for v in np.ravel(node["_ArraySize_"])]
        complex_ = bool(node.get("_ArrayIsComplex_", False))
        if "_ArrayZipData_" in node:
            raw = node["_ArrayZipData_"]
            raw = base64.b64decode(raw) if isinstance(raw, str) else np.asarray(raw).tobytes()
            kind = str(node.get("_ArrayZipType_", "zlib")).lower()
            if kind == "zlib":
                raw = zlib.decompress(raw)
            elif kind == "gzip":
                raw = gzip.decompress(raw)
            elif kind == "lzma":
                raw = lzma.decompress(raw, format=lzma.FORMAT_ALONE)
            else:
                raise ValueError(f"_ArrayZipType_ {kind} not supported")
            values = np.frombuffer(raw, dtype)
        else:
            values = np.asarray(node["_ArrayData_"], dtype=dtype).ravel()
        count = int(np.prod(shape))
        if complex_:
            values = values[:count] + 1j * values[count:2 * count]
        order = "F" if str(node.get("_ArrayOrder_", "r")).lower().startswith("c") else "C"
        return values[:count].reshape(shape, order=order)

    @staticmethod
    def affine(header: dict, notes: list[str]) -> np.ndarray:
        sizes = [float(v) for v in np.ravel(header.get("VoxelSize", [1, 1, 1]))][:3]
        sizes += [1.0] * (3 - len(sizes))
        affine = np.diag(sizes + [1.0])
        rows = np.asarray(header.get("Affine", []), dtype=float)
        if header.get("SForm") and rows.shape == (3, 4) and np.any(rows[:, :3]):
            affine[:3] = rows
            return affine
        quatern = header.get("Quatern")
        if header.get("QForm") and isinstance(quatern, dict):
            b, c, d = (float(quatern.get(k, 0)) for k in "bcd")
            a = np.sqrt(max(0.0, 1.0 - b * b - c * c - d * d))
            rotation = np.array([
                [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - c * c - b * b]])
            qfac = -1.0 if str(header.get("Orientation", {}).get("z", "s")).lower() == "i" else 1.0
            affine[:3, :3] = rotation * np.array([sizes[0], sizes[1], sizes[2] * qfac])
            offset = header.get("QuaternOffset", {})
            affine[:3, 3] = [float(offset.get(k, 0)) for k in "xyz"]
            return affine
        notes.append("neither sform nor qform set: voxel sizes only")
        return affine

    @classmethod
    def plain(cls, value):
        """Header values JSON-ready (arrays from BJData as lists)."""
        if isinstance(value, dict):
            return {k: cls.plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls.plain(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value
