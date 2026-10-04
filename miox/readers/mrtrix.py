####################################################################################################
#                                            mrtrix.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 04/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for MRtrix images (.mif, .mif.gz with attached data, .mih with a detached data   #
#          file).                                                                                  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import gzip
from pathlib import Path
from typing import BinaryIO

import numpy as np

# own
from miox.data import MRImage
from miox.meta.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                           MrtrixReader                                           #
#**************************************************************************************************#
#                                                                                                  #
# "key: value" lines after "mrtrix image" up to "END" (or the end of a .mih header); a key may     #
# repeat (transform rows, dw_scheme rows, comments). "file: . offset" points at the data in the    #
# same file, "file: name offset" at a detached file. "layout" gives, for each image axis, its rank #
# among the strides and its direction ("-0,-1,+2,+3": axis 0 fastest and reversed, ...); the       #
# stored array is undone into image order. "transform" (3 rows of 4) maps voxel indices to scanner #
# RAS coordinates in mm, the centre of the first voxel at its translation; "vox" the voxel sizes.  #
# "dw_scheme" rows are gradient directions in scanner coordinates with the b-value,                #
# "PhaseEncodingDirection", "TotalReadoutTime", "EchoTime" and the like plain BIDS-named values.   #
# Bit images unpack to 0/1. Checked against the PNG exports and the mean over volumes that MRtrix  #
# itself wrote for its test data.                                                                  #
#                                                                                                  #
#**************************************************************************************************#
class MrtrixReader(Reader):
    FORMATS = (Format.MRTRIX,)
    SOURCE = "MRtrix header"
    TYPES = {
        "bit": "u1", "int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2",
        "int32": "i4", "uint32": "u4", "int64": "i8", "uint64": "u8",
        "float32": "f4", "float64": "f8", "cfloat32": "c8", "cfloat64": "c16",
    }
    PLAIN_KEYS = {  # header key: (standard key, factor)
        "EchoTime": ("EchoTime", 1.0),
        "RepetitionTime": ("RepetitionTime", 1.0),
        "FlipAngle": ("ExcitationFlipAngle", 1.0),
        "PhaseEncodingDirection": ("PhaseEncodingDirection", None),
        "TotalReadoutTime": ("TotalReadoutTime", 1.0),
        "EffectiveEchoSpacing": ("EffectiveEchoSpacing", 1.0),
        "SliceTiming": ("SliceTiming", None),
        "MultibandAccelerationFactor": ("MultibandAccelerationFactor", 1.0),
        "Manufacturer": ("Manufacturer", None),
        "ManufacturersModelName": ("ManufacturersModelName", None),
        "MagneticFieldStrength": ("FieldStrength", 1.0),
        "SeriesDescription": ("SeriesDescription", None),
        "ProtocolName": ("ProtocolName", None),
        "SequenceName": ("SequenceName", None),
    }

    def read(self, sniff: Sniff) -> list[MRImage]:
        header = self._header(sniff.path)
        try:
            data, notes = self._data(sniff.path, header)
        except (ValueError, OSError, KeyError) as e:
            raise ReaderError(f"{sniff.path.name}: {e}") from e
        affine = self._affine(header)
        meta = Meta({"header": header})
        self._keys(meta, header)
        while data.ndim < 3:
            data = data[..., np.newaxis]
        dims = ("volume",) * (data.ndim - 3)
        name = sniff.path.name
        for suffix in (".gz", ".mif", ".mih"):
            name = name.removesuffix(suffix)
        return [MRImage(data, affine, meta, self.provenance(sniff, tuple(notes)), name, dims)]

    # --- header ----------------------------------------------------------------------------------

    def _open(self, path: Path) -> BinaryIO:
        with open(path, "rb") as fh:
            gzipped = fh.read(2) == b"\x1f\x8b"
        return gzip.open(path, "rb") if gzipped else open(path, "rb")

    def _header(self, path: Path) -> dict[str, list[str]]:
        header: dict[str, list[str]] = {}
        with self._open(path) as fh:
            if not fh.readline().startswith(b"mrtrix image"):
                raise ReaderError(f"{path.name}: no 'mrtrix image' magic")
            while True:
                line = fh.readline()
                text = line.decode("utf-8", "replace").rstrip("\r\n")
                if not line or text.strip() == "END":  # a .mih may end without END
                    break
                if ":" not in text:
                    continue
                key, value = text.split(":", 1)
                header.setdefault(key.strip(), []).append(value.strip())
        for key in ("dim", "vox", "datatype", "layout", "file"):
            if key not in header:
                raise ReaderError(f"{path.name}: header without {key}")
        return header

    @staticmethod
    def _first(header: dict, key: str, default: str = "") -> str:
        return header.get(key, [default])[0]

    # --- data ------------------------------------------------------------------------------------

    def _data(self, path: Path, header: dict[str, list[str]]) -> tuple[np.ndarray, list[str]]:
        dim = [int(v) for v in self._first(header, "dim").split(",")]
        layout = self._first(header, "layout").split(",")
        ranks = [int(v.lstrip("+-")) for v in layout]
        reversed_axes = [v.strip().startswith("-") for v in layout]
        datatype = self._first(header, "datatype")
        base = datatype.lower().removesuffix("le").removesuffix("be")
        dtype = np.dtype(self.TYPES[base])
        if dtype.itemsize > 1:
            dtype = dtype.newbyteorder(">" if datatype.lower().endswith("be") else "<")
        name, _, offset = self._first(header, "file").partition(" ")
        offset = int(offset or 0)
        source = path if name == "." else path.parent / name
        notes = [] if name == "." else [f"data in {name}"]
        count = int(np.prod(dim))
        with self._open(source) as fh:
            fh.seek(offset)
            blob = fh.read(count * dtype.itemsize if base != "bit" else (count + 7) // 8)
        if base == "bit":
            raw = np.unpackbits(np.frombuffer(blob, dtype=np.uint8), bitorder="little")[:count]
        else:
            raw = np.frombuffer(blob, dtype=dtype)
        if raw.size < count:
            raise ValueError(f"{raw.size} values, expected {count}")
        # storage order: the axis of rank 0 varies fastest
        axes_by_rank = [ranks.index(r) for r in range(len(dim))]
        stored_shape = [dim[a] for a in axes_by_rank][::-1]
        data = raw[:count].reshape(stored_shape)
        data = np.transpose(data, [len(dim) - 1 - ranks[a] for a in range(len(dim))])
        for axis, flipped in enumerate(reversed_axes):
            if flipped:
                data = np.flip(data, axis=axis)
        return np.ascontiguousarray(data), notes

    # --- geometry and keys -----------------------------------------------------------------------

    def _affine(self, header: dict[str, list[str]]) -> np.ndarray:
        affine = np.eye(4)
        rows = header.get("transform", [])
        for i, row in enumerate(rows[:3]):
            values = [float(v) for v in row.split(",")]
            affine[i, : len(values)] = values[:4]
        sizes = self._first(header, "vox").split(",")
        vox = [float(v) if v.strip().lower() != "nan" else 1.0 for v in sizes]
        for column, size in enumerate(vox[:3]):
            affine[:3, column] *= size
        return affine

    def _keys(self, meta: Meta, header: dict[str, list[str]]) -> None:
        for field, (key, factor) in self.PLAIN_KEYS.items():
            if field not in header:
                continue
            value: object = header[field][0]
            if factor is not None:
                try:
                    value = float(value) * factor
                except ValueError:
                    pass
            if field == "SliceTiming":
                value = [float(v) for v in str(value).split(",")]
            if key == "FieldStrength":
                meta.set(key, value, f"{self.SOURCE}: {field}", f"header/{field}")
            else:
                meta.set(key, value, f"{self.SOURCE}: {field}", f"header/{field}")
        for field in ("comments", "command_history", "mrtrix_version"):
            if field in header:
                value = header[field] if len(header[field]) > 1 else header[field][0]
                meta.set(f"mrtrix_{field}", value, f"{self.SOURCE}: {field}", f"header/{field}")
        scheme = header.get("dw_scheme")
        if scheme:
            rows = [[float(v) for v in row.split(",")] for row in scheme]
            values, directions = [], []
            for row in rows:
                g, b = np.asarray(row[:3], float), float(row[3]) if len(row) > 3 else 0.0
                norm = float(np.linalg.norm(g))
                values.append(b)
                if b > 0 and norm > 0:
                    directions.append([float(f"{v:.6f}") for v in g / norm])
                else:
                    directions.append([0.0, 0.0, 0.0])
            meta.set("DiffusionBValue", values, f"{self.SOURCE}: dw_scheme", "header/dw_scheme")
            meta.set("DiffusionGradientOrientation", directions,
                     f"{self.SOURCE}: dw_scheme (scanner coordinates, RAS)", "header/dw_scheme")
