####################################################################################################
#                                           varian_fdf.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 04/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for Varian/Agilent VnmrJ fdf images: one file, or a folder of files (one per     #
#          slice, echo and array element) assembled into a volume.                                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

# own
from miox.data import MRImage
from miox.meta.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                          VarianFdfReader                                         #
#**************************************************************************************************#
#                                                                                                  #
# An fdf file is a C-like text header ("float matrix[] = {128, 128};" lines, ending at the first   #
# NUL byte) followed by the binary data, the first matrix axis (read-out) varying fastest. The     #
# header's "orientation" holds the direction cosines of the read, phase and slice axes in the      #
# magnet frame (X right to left looking along the bore, Y up, Z along the bore towards the sample  #
# end), "location" the centre of the acquired volume in the user frame, "span" and "roi" its       #
# signed extent and size in cm. The magnet frame is mapped to RAS as Agilent2Dicom does, which     #
# assumes a head-first supine subject (R = -X, A = Y, S = Z); the position fields are kept for     #
# the user. Pixel values were checked against ImageJ text-image exports of the same files.         #
#                                                                                                  #
#**************************************************************************************************#
class VarianFdfReader(Reader):
    FORMATS = (Format.VARIAN_FDF,)
    SOURCE = "fdf header"
    FIELD = re.compile(r"^\s*(?:\w+)\s+\*?(\w+)(\[\])?\s*=\s*(.*?)\s*;\s*$")
    MAGNET_TO_RAS = np.array([-1.0, 1.0, 1.0])
    POSITIONS = {"head first": "HF", "feet first": "FF"}
    POSTURES = {"supine": "S", "prone": "P", "left": "DL", "right": "DR"}
    KEYS = {  # standard key: (header field, scale to the standard unit)
        "EchoTime": ("te", 1.0),
        "RepetitionTime": ("tr", 1.0),
        "InversionTime": ("ti", 1.0),
        "SequenceName": ("sequence", None),
        "EchoNumber": ("echo_no", None),
        "AcquisitionNumber": ("array_index", None),
    }

    def read(self, sniff: Sniff) -> list[MRImage]:
        path = sniff.path
        files = sorted(path.glob("*.fdf")) if path.is_dir() else [path]
        if not files:
            raise ReaderError(f"{path.name}: no fdf files")
        parsed = [self._parse(f) for f in files]
        headers = [h for h, _ in parsed]
        first = headers[0]
        notes: list[str] = []
        for h in headers[1:]:
            if h.get("matrix") != first.get("matrix") or h.get("orientation") != first.get(
                "orientation"
            ):
                raise ReaderError(f"{path.name}: fdf files differ in matrix or orientation")
        data, dims, slice_headers = self._assemble(parsed, notes)
        affine = self._affine(slice_headers, notes)
        meta = Meta({"header": first})
        self._keys(meta, first)
        name = path.name.rsplit(".", 1)[0] if path.is_dir() else path.stem
        return [MRImage(data, affine, meta, self.provenance(sniff, tuple(notes)), name, dims)]

    # --- header and data -------------------------------------------------------------------------

    def _parse(self, path: Path) -> tuple[dict[str, Any], np.ndarray]:
        blob = path.read_bytes()
        end = blob.find(b"\0")
        if end < 0:
            raise ReaderError(f"{path.name}: no end of header")
        header: dict[str, Any] = {}
        for line in blob[:end].decode("ascii", "replace").splitlines():
            if m := self.FIELD.match(line):
                name, is_list, value = m.groups()
                header[name] = self._value(value, bool(is_list))
        matrix = header.get("matrix")
        if not isinstance(matrix, list) or not matrix:
            raise ReaderError(f"{path.name}: header without matrix")
        dtype = self._dtype(header)
        count = int(np.prod(matrix))
        payload = blob[end + 1 :]
        if len(payload) < count * dtype.itemsize:
            needed = count * dtype.itemsize
            raise ReaderError(f"{path.name}: {len(payload)} data bytes, expected {needed}")
        values = np.frombuffer(payload[len(payload) - count * dtype.itemsize :], dtype)
        if header.get("type") == "complex":
            values = values[0::2] + 1j * values[1::2]
            matrix = matrix[:-1] if matrix[-1] == 2 else matrix
        data = values.reshape([int(n) for n in matrix[::-1]]).T  # first axis fastest
        return header, data

    @staticmethod
    def _value(text: str, is_list: bool) -> Any:
        def scalar(item: str) -> Any:
            item = item.strip()
            if item.startswith('"') and item.endswith('"'):
                return item[1:-1]
            try:
                return int(item)
            except ValueError:
                try:
                    return float(item)
                except ValueError:
                    return item

        if is_list or (text.startswith("{") and text.endswith("}")):
            inner = text.strip()[1:-1] if text.strip().startswith("{") else text
            return [scalar(v) for v in inner.split(",") if v.strip()]
        return scalar(text)

    @staticmethod
    def _dtype(header: dict[str, Any]) -> np.dtype:
        bits = int(header.get("bits", 32))
        storage = str(header.get("storage", "float")).lower()
        kind = {"float": "f", "integer": "i", "int": "i", "short": "i", "char": "u"}.get(storage)
        if kind is None:
            raise ReaderError(f"fdf storage {storage!r} is not read")
        order = ">" if int(header.get("bigendian", 0)) else "<"
        return np.dtype(f"{order}{kind}{bits // 8}")

    def _assemble(
        self, parsed: list[tuple[dict, np.ndarray]], notes: list[str]
    ) -> tuple[np.ndarray, tuple[str, ...], list[dict]]:
        """Stack the files into (read, phase, slice[, echo][, array]) and return the headers of
        the slices of the first echo and array element, in slice order."""
        index: dict[tuple[int, int, int], tuple[dict, np.ndarray]] = {}
        for header, data in parsed:
            key = (int(header.get("slice_no", 1)), int(header.get("echo_no", 1)),
                   int(header.get("array_index", 1)))
            if key in index:
                raise ReaderError(f"two fdf files for slice/echo/array {key}")
            index[key] = (header, data)
        slices, echoes, arrays = (sorted({k[i] for k in index}) for i in range(3))
        if len(index) != len(slices) * len(echoes) * len(arrays):
            raise ReaderError("fdf files do not fill the slice x echo x array grid")
        first = index[(slices[0], echoes[0], arrays[0])][1]
        if first.ndim == 3 and len(slices) > 1:
            raise ReaderError("3D fdf files with several slice numbers are not read")
        spatial = first.shape if first.ndim == 3 else first.shape + (len(slices),)
        data = np.empty(spatial + (len(echoes), len(arrays)), first.dtype)
        for (s, e, a), (_, block) in index.items():
            if first.ndim == 3:
                data[..., echoes.index(e), arrays.index(a)] = block
            else:
                data[:, :, slices.index(s), echoes.index(e), arrays.index(a)] = block
        keep = [i for i, n in enumerate((len(echoes), len(arrays))) if n > 1]
        data = data.reshape(spatial + tuple(data.shape[3 + i] for i in keep))
        dims = tuple(("echo", "array")[i] for i in keep)
        if len(slices) > 1:
            notes.append(f"{len(slices)} slices assembled from one fdf file each")
        return data, dims, [index[(s, echoes[0], arrays[0])][0] for s in slices]

    # --- geometry and keys -----------------------------------------------------------------------

    def _affine(self, headers: list[dict], notes: list[str]) -> np.ndarray:
        first = headers[0]
        matrix = [int(n) for n in first["matrix"]]
        rank = len(matrix)
        rotation = np.asarray(first.get("orientation", [1, 0, 0, 0, 1, 0, 0, 0, 1]), float)
        rotation = rotation.reshape(3, 3)  # rows: read, phase, slice axes in the magnet frame
        rotation /= np.linalg.norm(rotation, axis=1, keepdims=True)  # written to four decimals
        span = np.asarray(first.get("span", first.get("roi", [1, 1, 1]))[:rank], float) * 10
        roi = np.asarray(first.get("roi", [1, 1, 1]), float) * 10
        location = np.asarray(first.get("location", [0, 0, 0]), float) * 10
        steps = span / matrix
        thickness = roi[2] / matrix[2] if rank == 3 else roi[2]
        if rank == 2:
            steps = np.append(steps, thickness)
        user_origin = location.copy()  # the centre, in the user frame
        user_origin[:rank] -= span / 2 - steps[:rank] / 2  # centre of the first voxel
        if rank == 2 and len(headers) > 1:
            centres = [np.asarray(h.get("location", [0, 0, 0]), float) * 10 for h in headers]
            along = [float(c[2]) for c in centres]
            gaps = np.diff(along)
            if gaps.size and np.all(np.abs(gaps - gaps[0]) < 1e-3):
                steps[2] = gaps[0]
            else:
                notes.append("slice positions not evenly spaced; thickness used as slice step")
            user_origin[2] = along[0]
        axes = rotation.T * steps  # columns: voxel steps in the magnet frame
        origin = rotation.T @ user_origin
        affine = np.eye(4)
        affine[:3, :3] = axes * self.MAGNET_TO_RAS[:, np.newaxis]
        affine[:3, 3] = origin * self.MAGNET_TO_RAS
        notes.append("magnet frame mapped to RAS for a head-first supine subject")
        return affine

    @staticmethod
    def _nucleus(name: str) -> str:
        """VnmrJ writes the symbol first ("H1", "C13"); the standard puts the mass number first."""
        if m := re.fullmatch(r"([A-Za-z]+)(\d+)", name.strip()):
            return m.group(2) + m.group(1)
        return name

    def _keys(self, meta: Meta, header: dict) -> None:
        for key, (field, scale) in self.KEYS.items():
            value = header.get(field)
            if value in (None, "", 0) and key != "EchoNumber":
                continue
            if scale is not None:
                value = float(value) * scale
            meta.set(key, value, f"{self.SOURCE}: {field}", f"header/{field}")
        if freqs := header.get("nucfreq"):
            meta.set("SpectrometerFrequency", [float(freqs[0])], f"{self.SOURCE}: nucfreq[0]",
                     "header/nucfreq")
        if nuclei := header.get("nucleus"):
            meta.set("ResonantNucleus", [self._nucleus(str(nuclei[0]))],
                     f"{self.SOURCE}: nucleus[0]", "header/nucleus")
        roi, matrix = header.get("roi"), header.get("matrix", [])
        if roi and len(roi) >= 3:
            thickness = float(roi[2]) * 10 / (int(matrix[2]) if len(matrix) == 3 else 1)
            meta.set("SliceThickness", thickness, f"{self.SOURCE}: roi[2] in cm", "header/roi")
            if len(matrix) == 2:
                gap = float(header.get("gap", 0.0)) * 10
                meta.set("SpacingBetweenSlices", thickness + gap, f"{self.SOURCE}: roi[2] + gap",
                         "header/gap")
        entry = str(header.get("position1", "")).lower()
        posture = str(header.get("position2", "")).lower()
        code = self.POSITIONS.get(entry, "") + next(
            (v for k, v in self.POSTURES.items() if k in posture), ""
        )
        if code and code[:2] in self.POSITIONS.values() and len(code) > 2:
            meta.set("PatientPosition", code, f"{self.SOURCE}: position1, position2",
                     "header/position1")
