####################################################################################################
#                                             nrrd.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 04/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for NRRD images (.nrrd with attached data, .nhdr with detached data files).      #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import bz2
import gzip
import re
import zlib
from pathlib import Path

import numpy as np

# own
from miox.data import MRImage
from miox.meta.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                            NrrdReader                                            #
#**************************************************************************************************#
#                                                                                                  #
# The header is text ("field: value" lines and "key:=value" pairs) up to a blank line; the data    #
# follow in the same file or in the files the "data file" field names (one, a list, or a printf    #
# pattern with a range), raw, gzip, bzip2, ascii or hex encoded, after "line skip" lines and       #
# "byte skip" bytes (-1: the data end the file). The first axis varies fastest. Axes of kind       #
# "domain"/"space" are spatial; "space directions" give their world steps in the named space       #
# (left-posterior-superior, right-anterior-superior, ..., turned into RAS here), "space origin"    #
# the centre of the first voxel; "spacings" alone give an axis-aligned grid. Other axes (vectors,  #
# colours, lists) stay as trailing dimensions. NA-MIC diffusion key/value pairs (DWMRI_b-value,    #
# DWMRI_gradient_NNNN) give the b-values and gradient directions; the directions are in the        #
# measurement frame, mapped to RAS world coordinates. Checked against the NIfTI next to pynrrd's   #
# test image.                                                                                      #
#                                                                                                  #
#**************************************************************************************************#
class NrrdReader(Reader):
    FORMATS = (Format.NRRD,)
    SOURCE = "NRRD header"
    TYPES = {
        "signed char": "i1", "int8": "i1", "int8_t": "i1",
        "uchar": "u1", "unsigned char": "u1", "uint8": "u1", "uint8_t": "u1",
        "short": "i2", "short int": "i2", "signed short": "i2", "signed short int": "i2",
        "int16": "i2", "int16_t": "i2",
        "ushort": "u2", "unsigned short": "u2", "unsigned short int": "u2",
        "uint16": "u2", "uint16_t": "u2",
        "int": "i4", "signed int": "i4", "int32": "i4", "int32_t": "i4",
        "uint": "u4", "unsigned int": "u4", "uint32": "u4", "uint32_t": "u4",
        "longlong": "i8", "long long": "i8", "long long int": "i8", "signed long long": "i8",
        "signed long long int": "i8", "int64": "i8", "int64_t": "i8",
        "ulonglong": "u8", "unsigned long long": "u8", "unsigned long long int": "u8",
        "uint64": "u8", "uint64_t": "u8",
        "float": "f4", "double": "f8", "block": "V",
    }
    SPACES = {  # NRRD space -> signs that turn its axes into RAS
        "right-anterior-superior": (1, 1, 1), "ras": (1, 1, 1),
        "left-anterior-superior": (-1, 1, 1), "las": (-1, 1, 1),
        "left-posterior-superior": (-1, -1, 1), "lps": (-1, -1, 1),
        "scanner-xyz": (-1, -1, 1),  # DICOM patient coordinates
    }
    SPATIAL_KINDS = ("domain", "space", "")
    DIM_NAMES = {"vector": "vector", "rgb-color": "rgb", "rgba-color": "rgba", "list": "list",
                 "3-vector": "vector", "3-gradient": "gradient", "time": "time"}
    GRADIENT = re.compile(r"^DWMRI_gradient_(\d+)$")

    def read(self, sniff: Sniff) -> list[MRImage]:
        fields, keys, data_offset = self._header(sniff.path)
        try:
            data, notes = self._data(sniff.path, fields, data_offset)
        except (ValueError, OSError) as e:
            raise ReaderError(f"{sniff.path.name}: {e}") from e
        sizes = self._ints(fields["sizes"])
        kinds = self._words(fields.get("kinds", "")) or ["domain"] * len(sizes)
        spatial = [i for i, k in enumerate(kinds) if k.lower() in self.SPATIAL_KINDS]
        others = [i for i in range(len(sizes)) if i not in spatial]
        if len(spatial) > 3:
            spatial, others = spatial[:3], spatial[3:] + others
        data = np.transpose(data, spatial + others)
        while data.ndim < 3:
            data = data[..., np.newaxis]
            spatial.append(None)
        affine, geometry_notes = self._affine(fields, spatial)
        notes += geometry_notes
        meta = Meta({"fields": fields, "keys": keys})
        self._keys(meta, fields, keys, len(spatial), notes)
        dims = tuple(self.DIM_NAMES.get(kinds[i].lower(), kinds[i].lower() or "volume")
                     for i in others)
        name = sniff.path.name.rsplit(".", 1)[0]
        provenance = self.provenance(sniff, tuple(notes))
        return [MRImage(data, affine, meta, provenance, name, dims)]

    # --- header ----------------------------------------------------------------------------------

    def _header(self, path: Path) -> tuple[dict[str, str], dict[str, str], int]:
        """Fields, key/value pairs and the byte offset of the data (attached files)."""
        fields: dict[str, str] = {}
        keys: dict[str, str] = {}
        with open(path, "rb") as fh:
            magic = fh.readline()
            if not magic.startswith(b"NRRD"):
                raise ReaderError(f"{path.name}: no NRRD magic")
            fields["magic"] = magic.decode("ascii", "replace").strip()
            while True:
                line = fh.readline()
                if not line or line in (b"\n", b"\r\n"):
                    break
                text = line.decode("utf-8", "replace").rstrip("\r\n")
                if text.startswith("#"):
                    continue
                if ":=" in text:
                    key, value = text.split(":=", 1)
                    keys[key.strip()] = value.strip()
                elif ": " in text or text.endswith(":"):
                    field, value = text.split(":", 1)
                    fields[field.strip().lower()] = value.strip()
            offset = fh.tell()
        if "sizes" not in fields or "type" not in fields:
            raise ReaderError(f"{path.name}: header without sizes or type")
        return fields, keys, offset

    @staticmethod
    def _ints(text: str) -> list[int]:
        return [int(v) for v in text.replace(",", " ").split()]

    @staticmethod
    def _words(text: str) -> list[str]:
        return text.split()

    @staticmethod
    def _vectors(text: str) -> list[list[float] | None]:
        """Parenthesised vectors, "none" for axes without one."""
        out: list[list[float] | None] = []
        for item in re.findall(r"\([^)]*\)|none", text):
            if item == "none":
                out.append(None)
            else:
                out.append([float(v) for v in item.strip("()").split(",")])
        return out

    # --- data ------------------------------------------------------------------------------------

    def _data(
        self, path: Path, fields: dict[str, str], offset: int
    ) -> tuple[np.ndarray, list[str]]:
        sizes = self._ints(fields["sizes"])
        dtype = np.dtype(self.TYPES[fields["type"].strip().lower()])
        if dtype.itemsize > 1:
            endian = fields.get("endian", "little").lower()
            dtype = dtype.newbyteorder("<" if endian == "little" else ">")
        encoding = fields.get("encoding", "raw").lower()
        count = int(np.prod(sizes))
        notes = []
        if "data file" in fields or "datafile" in fields:
            files = self._data_files(path, fields.get("data file") or fields["datafile"], sizes)
            notes.append(f"data in {len(files)} detached file(s)")
            chunks = [self._decode(f, 0, fields, encoding, dtype) for f in files]
            raw = np.concatenate(chunks)
        else:
            raw = self._decode(path, offset, fields, encoding, dtype)
        if raw.size != count:
            raise ValueError(f"{raw.size} values, expected {count}")
        data = raw.reshape(sizes[::-1]).T  # first axis fastest
        return data, notes

    def _data_files(self, path: Path, spec: str, sizes: list[int]) -> list[Path]:
        parts = spec.split()
        if parts[0].upper() == "LIST":
            with open(path, "rb") as fh:
                text = fh.read().decode("utf-8", "replace")
            listed = text.split("\n\n", 1)[1].splitlines() if "\n\n" in text else []
            return [path.parent / name.strip() for name in listed if name.strip()]
        if len(parts) >= 4 and "%" in parts[0]:
            pattern, start, stop, step = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
            numbers = range(start, stop + (1 if step > 0 else -1), step)
            return [path.parent / (pattern % n) for n in numbers]
        return [path.parent / spec.strip()]

    def _decode(self, path: Path, offset: int, fields: dict, encoding: str, dtype) -> np.ndarray:
        with open(path, "rb") as fh:
            fh.seek(offset)
            blob = fh.read()
        for _ in range(int(fields.get("line skip", fields.get("lineskip", 0)) or 0)):
            blob = blob.split(b"\n", 1)[1] if b"\n" in blob else b""
        if encoding in ("gz", "gzip"):
            gzipped = blob[:2] == b"\x1f\x8b"
            blob = zlib.decompressobj(31).decompress(blob) if gzipped else gzip.decompress(blob)
        elif encoding in ("bz2", "bzip2"):
            blob = bz2.decompress(blob)
        elif encoding in ("ascii", "txt", "text"):
            return np.array(blob.decode("ascii", "replace").split(), dtype=dtype.newbyteorder("="))
        elif encoding == "hex":
            blob = bytes.fromhex("".join(blob.decode("ascii", "replace").split()))
        skip = int(fields.get("byte skip", fields.get("byteskip", 0)) or 0)
        count = int(np.prod(self._ints(fields["sizes"]))) * dtype.itemsize
        if skip == -1:
            blob = blob[len(blob) - count:]
        elif skip > 0:
            blob = blob[skip:]
        return np.frombuffer(blob[: (len(blob) // dtype.itemsize) * dtype.itemsize], dtype=dtype)

    # --- geometry and keys -----------------------------------------------------------------------

    def _affine(self, fields: dict[str, str], spatial: list) -> tuple[np.ndarray, list[str]]:
        notes = []
        affine = np.eye(4)
        directions = self._vectors(fields.get("space directions", ""))
        space = fields.get("space", "").lower()
        signs = np.asarray(self.SPACES.get(space, (1, 1, 1)), float)
        if space and space not in self.SPACES:
            notes.append(f"space {space!r} not a patient space; axes taken as RAS")
        if directions:
            steps = [d for d in directions if d is not None]
            for column, step in zip(range(3), steps, strict=False):
                vector = np.asarray(step, float)
                affine[: len(vector), column] = vector[:3] * signs[: len(vector)]
            origin = self._vectors(fields.get("space origin", ""))
            if origin and origin[0] is not None:
                vector = np.asarray(origin[0], float)
                affine[: len(vector), 3] = vector[:3] * signs[: len(vector)]
        else:
            words = fields.get("spacings", "").split()
            spacings = [float(v) if v.lower() != "nan" else 1.0 for v in words]
            axes = [s for s in spatial if s is not None]
            for column, axis in enumerate(axes[:3]):
                affine[column, column] = spacings[axis] if axis < len(spacings) else 1.0
            notes.append("no space directions: axis-aligned grid from spacings, origin at zero")
        return affine, notes

    def _keys(self, meta: Meta, fields: dict, keys: dict, nspatial: int, notes: list[str]) -> None:
        for field in ("content", "space", "kinds", "space units", "measurement frame", "labels",
                      "thicknesses", "centerings"):
            if field in fields:
                key = f"nrrd_{field.replace(' ', '_')}"
                meta.set(key, fields[field], f"{self.SOURCE}: {field}", f"fields/{field}")
        b_value = keys.get("DWMRI_b-value")
        gradients = sorted(
            (int(m.group(1)), v) for k, v in keys.items() if (m := self.GRADIENT.match(k))
        )
        if b_value is None or not gradients:
            return
        nominal = float(b_value)
        frame = np.eye(3)
        if "measurement frame" in fields:
            vectors = [v for v in self._vectors(fields["measurement frame"]) if v is not None]
            if len(vectors) == 3:
                frame = np.asarray(vectors, float).T
        signs = np.asarray(self.SPACES.get(fields.get("space", "").lower(), (1, 1, 1)), float)
        values, directions = [], []
        for _, text in gradients:
            g = np.asarray([float(v) for v in text.split()], float)
            norm = float(np.linalg.norm(g))
            values.append(nominal * norm * norm if norm > 0 else 0.0)  # NA-MIC scaling by length
            if norm > 0:
                world = (frame @ g) * signs / norm
                directions.append([float(f"{v:.6f}") for v in world])
            else:
                directions.append([0.0, 0.0, 0.0])
        meta.set("DiffusionBValue", values, "DWMRI_b-value scaled by the gradient lengths squared")
        meta.set("DiffusionGradientOrientation", directions,
                 "DWMRI_gradient_* in the measurement frame, as RAS world unit vectors")
        notes.append("NA-MIC DWI key/value pairs read")
