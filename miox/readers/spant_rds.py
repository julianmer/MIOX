####################################################################################################
#                                           spant_rds.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 30/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for spant's objects saved as R data (.rds): spectra (mrs_data), basis sets       #
#          (basis_set) and fit results (fit_result).                                               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import bz2
import gzip
import lzma
import re
import struct
from typing import Any

import numpy as np

# own
from miox.data import BasisSet, FitResult, MRSData
from miox.meta import KeyTable
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                             RObject                                              #
#**************************************************************************************************#
#                                                                                                  #
# One R value: its kind ("double", "integer", "logical", "complex", "character", "list", "raw",    #
# or what is kept only by name: "environment", "closure", ...), its values (numpy arrays for       #
# numbers, lists for strings and lists) and its attributes (names, dim, class, levels, ...).       #
#                                                                                                  #
#**************************************************************************************************#
class RObject:
    def __init__(self, kind: str, value: Any = None, attributes: dict | None = None):
        self.kind = kind
        self.value = value
        self.attributes = attributes or {}

    def attribute(self, name: str) -> Any:
        """An attribute's values, None when it is not set."""
        item = self.attributes.get(name)
        return item.value if isinstance(item, RObject) else None

    @property
    def classes(self) -> list[str]:
        return list(self.attribute("class") or [])

    @property
    def names(self) -> list[str]:
        return list(self.attribute("names") or [])

    def __getitem__(self, name: str) -> Any:
        return self.value[self.names.index(name)]

    def __contains__(self, name: str) -> bool:
        return name in self.names

    def array(self) -> np.ndarray:
        """The values shaped by the dim attribute (R's order: first index fastest)."""
        dim = self.attribute("dim")
        values = np.asarray(self.value)
        return values.reshape([int(n) for n in dim], order="F") if dim is not None else values

    def plain(self) -> Any:
        """As Python values: a named list as a dict, a vector of one as a scalar, NA as None,
        factors as their labels; what is code or an environment by its kind."""
        if self.kind == "list":
            values = [plain(v) for v in self.value]
            return dict(zip(self.names, values, strict=True)) if self.names else values
        if self.kind in ("double", "integer", "logical", "complex"):
            values = self.value.tolist()
            if self.kind == "logical":
                values = [None if v == RSerial.NA_INTEGER else bool(v) for v in values]
            elif self.kind == "integer":
                values = [None if v == RSerial.NA_INTEGER else v for v in values]
                if levels := self.attribute("levels"):
                    values = [None if v is None else levels[v - 1] for v in values]
            elif self.kind == "double":
                values = [None if np.isnan(v) else v for v in values]
            else:
                values = [[v.real, v.imag] for v in values]
        elif self.kind == "character":
            values = list(self.value)
        else:
            return f"<R {self.kind}>"
        if self.names and len(self.names) == len(values):
            return dict(zip(self.names, values, strict=True))
        return values[0] if len(values) == 1 else values


def plain(value: Any) -> Any:
    return value.plain() if isinstance(value, RObject) else value


#**************************************************************************************************#
#                                             RSerial                                              #
#**************************************************************************************************#
#                                                                                                  #
# R's serialization format (as saveRDS writes it), XDR (big-endian binary), version 2 or 3, gzip,  #
# bzip2, xz or uncompressed. Every item starts with flags: the type in the low byte, then bits for #
# object, attributes and tag. Vectors are a length and their values (NA_integer_ is the smallest   #
# int); pairlists (attribute lists, calls, closures) are attributes, tag, value, then the rest of  #
# the list; symbols and environments enter a reference table that later items point back into.     #
# Byte code and ALTREP objects other than compact sequences and wrappers are refused.              #
#                                                                                                  #
#**************************************************************************************************#
class RSerial:
    NA_INTEGER = -(2**31)
    VECTORS = {10: ("logical", ">i4"), 13: ("integer", ">i4"), 14: ("double", ">f8"),
               15: ("complex", ">f8"), 24: ("raw", "u1")}
    PAIRLISTS = {2: "pairlist", 3: "closure", 5: "promise", 6: "call", 17: "dots",
                 239: "pairlist", 240: "call"}
    SPECIAL = {242: "environment", 241: "environment", 253: "environment", 252: "unbound",
               251: "missing", 247: "environment"}  # empty, base, global env, base namespace

    def __init__(self, blob: bytes):
        for decompress in (gzip.decompress, bz2.decompress, lzma.decompress):
            try:
                blob = decompress(blob)
                break
            except (OSError, ValueError, lzma.LZMAError):
                continue
        if blob[:2] != b"X\n":
            raise ValueError("not R data in XDR serialization")
        self.blob, self.position, self.references = blob, 2, []
        version = self.integer()
        if version not in (2, 3):
            raise ValueError(f"serialization version {version}")
        self.position += 8  # the writing and the oldest reading R versions
        if version == 3:
            length = self.integer()  # of the native encoding's name
            self.position += length

    def integer(self) -> int:
        value = struct.unpack_from(">i", self.blob, self.position)[0]
        self.position += 4
        return value

    def length(self) -> int:
        n = self.integer()
        return (self.integer() << 32) + self.integer() if n == -1 else n

    def read(self) -> Any:
        flags = self.integer()
        kind, has_attributes, has_tag = flags & 0xFF, flags & 0x200, flags & 0x400
        if kind == 254:  # NULL
            return None
        if kind in self.SPECIAL:
            return RObject(self.SPECIAL[kind])
        if kind == 255:  # reference to an earlier symbol or environment
            return self.references[(flags >> 8 or self.integer()) - 1]
        if kind in (248, 249, 250):  # persistent name, namespace, package: a string vector
            self.integer()
            item = RObject("namespace", [self.read() for _ in range(self.integer())])
            self.references.append(item)
            return item
        if kind == 1:  # symbol
            item = RObject("symbol")
            self.references.append(item)
            item.value = self.read()
            return item
        if kind == 4:  # environment
            item = RObject("environment")
            self.references.append(item)
            self.integer()  # locked
            item.value = {"enclosure": self.read(), "frame": self.read(), "hashtab": self.read()}
            item.attributes = self._attributes(self.read())
            return item
        if kind in self.PAIRLISTS:
            return self._pairlist(kind, has_attributes, has_tag)
        if kind == 9:  # one string
            n = self.integer()
            if n == -1:
                return None
            text = self.blob[self.position:self.position + n].decode("utf-8", "replace")
            self.position += n
            return text
        if kind in self.VECTORS:
            name, dtype = self.VECTORS[kind]
            n = self.length() * (2 if kind == 15 else 1)
            values = np.frombuffer(self.blob, dtype, n, self.position)
            self.position += n * np.dtype(dtype).itemsize
            if kind == 15:
                values = values[0::2] + 1j * values[1::2]
            elif kind == 24:
                values = bytes(values)
            else:
                values = values.astype(np.int64 if kind != 14 else float)
            item = RObject(name, values)
        elif kind in (16, 19, 20):  # strings, list, expressions
            n = self.length()
            item = RObject("character" if kind == 16 else "list", [self.read() for _ in range(n)])
        elif kind == 238:
            return self._altrep()
        elif kind == 25:  # S4 object: only its attributes
            item = RObject("s4")
        else:
            raise ValueError(f"R type {kind} not read")
        if has_attributes:
            item.attributes = self._attributes(self.read())
        return item

    def _pairlist(self, kind: int, has_attributes: int, has_tag: int) -> RObject:
        entries = []
        while True:
            if has_attributes:
                self.read()
            tag = self.read() if has_tag else None
            entries.append((tag.value if isinstance(tag, RObject) else None, self.read()))
            flags = self.integer()
            if flags & 0xFF not in (2, 239):  # the rest is not a further pairlist node
                self.position -= 4
                rest = self.read()
                if rest is not None:
                    entries.append((None, rest))
                break
            has_attributes, has_tag = flags & 0x200, flags & 0x400
        return RObject(self.PAIRLISTS[kind], entries)

    @staticmethod
    def _attributes(pairlist: Any) -> dict:
        return dict(pairlist.value) if isinstance(pairlist, RObject) else {}

    def _altrep(self) -> RObject:
        info, state, attributes = self.read(), self.read(), self.read()
        name = info.value[0][1].value if isinstance(info, RObject) else ""
        if name in ("compact_intseq", "compact_realseq"):
            n, start, step = state.value[:3]
            values = start + step * np.arange(int(n))
            kind = "integer" if name == "compact_intseq" else "double"
            item = RObject(kind, values.astype(np.int64 if kind == "integer" else float))
        elif name.startswith("wrap_"):  # the wrapped vector, then its metadata
            item = state.value[0][1]
        else:
            raise ValueError(f"ALTREP class {name} not read")
        item.attributes = self._attributes(attributes)
        return item


#**************************************************************************************************#
#                                         SpantRdsReader                                           #
#**************************************************************************************************#
#                                                                                                  #
# mrs_data: a 7D array (dummy, x, y, z, dynamic, coil, points), spectra where freq_domain marks    #
# the last dimension (back to FIDs by spant's ift_shift, ifft(ifftshift)), with the transmitter    #
# frequency ft (Hz), the dwell time resolution[7], the nucleus and ref, the shift of 0 Hz; the     #
# FIDs are in the NIfTI-MRS convention (checked against spant's fitted spectra). basis_set: one    #
# spectrum per column (spant keeps them in the frequency domain), names, fs, ft and ref; no        #
# nucleus (spant assumes 1H). fit_result: res_tab (X, Y, Z, Dynamic, Coil, the amp_cols            #
# amplitudes, their CRLBs as ".sd", then diagnostics) gives one FitResult per row, "sd" in         #
# amplitude units and sd_percent from it; fits gives each row's curves (PPMScale, Data, Fit,       #
# Baseline, one per signal) and the fitted data's frequency, nucleus and ref; the fitted data and  #
# basis are read as above. A plain list holding these objects gives each of them, its other values #
# in their vendor headers.                                                                         #
#                                                                                                  #
#**************************************************************************************************#
class SpantRdsReader(Reader):
    FORMATS = (Format.SPANT_RDS,)
    PACKAGES = ("miox",)
    SOURCE = "spant R data"
    CLASSES = ("mrs_data", "basis_set", "fit_result")
    CURVES = {"PPMScale": "ppm", "Data": "data", "Fit": "fit", "Baseline": "baseline"}
    LABELS = ("X", "Y", "Z", "Dynamic", "Coil")
    NUCLEUS = "1H"

    def read(self, sniff: Sniff) -> list[MRSData | BasisSet | FitResult]:
        try:
            top = RSerial(sniff.path.read_bytes()).read()
        except (ValueError, IndexError, struct.error) as e:
            raise ReaderError(f"{sniff.path.name}: {e}") from e
        stem = sniff.path.name.removesuffix(sniff.path.suffix)
        if not isinstance(top, RObject):
            raise ReaderError(f"{sniff.path.name}: no spant object")
        if any(c in self.CLASSES for c in top.classes):
            return self._object(sniff, top, stem, {})
        if top.kind != "list" or not top.names:
            raise ReaderError(f"{sniff.path.name}: no spant object")
        members = {n: v for n, v in zip(top.names, top.value, strict=True)
                   if isinstance(v, RObject) and any(c in self.CLASSES for c in v.classes)}
        others = {n: plain(v) for n, v in zip(top.names, top.value, strict=True)
                  if n not in members}
        if not members:
            raise ReaderError(f"{sniff.path.name}: no spant object in the list")
        return [d for name, member in members.items()
                for d in self._object(sniff, member, f"{stem} {name}", {"list": others})]

    def _object(self, sniff: Sniff, item: RObject, name: str, vendor: dict) -> list:
        if "mrs_data" in item.classes:
            return [self._mrs_data(sniff, item, name, vendor)]
        if "basis_set" in item.classes:
            return [self._basis(sniff, item, name, vendor)]
        return self._fit(sniff, item, name, vendor)

    def _mrs_data(self, sniff: Sniff, item: RObject, name: str, vendor: dict) -> MRSData:
        data = item["data"].array()
        frequency_domain = [bool(v) for v in item["freq_domain"].value]
        if data.ndim != 7 or data.shape[0] != 1:
            raise ReaderError(f"{sniff.path.name}: data of shape {data.shape}, not 7D")
        if any(frequency_domain[:6]):
            raise ReaderError(f"{sniff.path.name}: spatial dimensions in k-space not read")
        notes = []
        if frequency_domain[6]:
            data = np.fft.ifft(np.fft.ifftshift(data, axes=6), axis=6)
            notes.append("stored as spectra (freq_domain); transformed to FIDs as spant does")
        data = data[0].transpose(0, 1, 2, 5, 4, 3)  # x, y, z, points, coil, dynamic
        tags, keep = [], [0, 1, 2, 3]
        for axis, tag in ((4, "DIM_COIL"), (5, "DIM_DYN")):
            if data.shape[axis] > 1:
                tags.append(tag)
                keep.append(axis)
        data = data.reshape([data.shape[a] for a in keep])
        resolution = item["resolution"].value
        values = {"SpectrometerFrequency": ([float(item["ft"].value[0]) / 1e6], "ft"),
                  "ResonantNucleus": ([str(item["nuc"].value[0])], "nuc"),
                  "ReferencePPM": (float(item["ref"].value[0]), "ref")}
        header = {k: plain(v) for k, v in zip(item.names, item.value, strict=True)
                  if k != "data"}
        keys = KeyTable()
        for key, value in (header.get("meta") or {}).items():
            if key in keys and key not in values and value is not None:
                values[key] = (value, f"meta/{key}")
        if header.get("affine") is not None:
            notes.append("affine stored but not used: no sample to check its convention against")
        d = self.dataset(sniff, data, float(resolution[6]), values, vendor | header, name, notes,
                         self.SOURCE, dim_tags=tuple(tags))
        d.meta.set("DwellTime", float(resolution[6]), f"{self.SOURCE}: resolution[7]",
                   "resolution")
        return d

    def _basis(self, sniff: Sniff, item: RObject, name: str, vendor: dict) -> BasisSet:
        spectra = item["data"].array()
        fids = np.fft.ifft(np.fft.ifftshift(spectra, axes=0), axis=0)
        header = {k: plain(v) for k, v in zip(item.names, item.value, strict=True)
                  if k != "data"}
        values = {"SpectrometerFrequency": ([float(item["ft"].value[0]) / 1e6], "ft"),
                  "ResonantNucleus": ([self.NUCLEUS], None),
                  "ReferencePPM": (float(item["ref"].value[0]), "ref")}
        notes = [f"nucleus not stored in spant basis sets; {self.NUCLEUS} assumed",
                 "stored as spectra; transformed to FIDs as spant does"]
        dwelltime = 1 / float(item["fs"].value[0])
        d = self.basis_set(sniff, fids, [str(n) for n in item["names"].value], dwelltime,
                           values, vendor | header, name, notes, self.SOURCE)
        d.meta.set("ResonantNucleus", [self.NUCLEUS], f"assumed: {self.NUCLEUS}")
        return d

    def _fit(self, sniff: Sniff, item: RObject, name: str, vendor: dict) -> list:
        table = item["res_tab"]
        columns = table.names
        count = int(item["amp_cols"].value[0])
        amplitudes = columns[5:5 + count]
        signals = self._signals(amplitudes, item["basis"] if "basis" in item else None)
        diagnostics = columns[5 + 2 * count:]
        fits = item["fits"].value if "fits" in item and item["fits"] is not None else []
        header = {k: plain(v) for k, v in zip(item.names, item.value, strict=True)
                  if k not in ("res_tab", "fits", "data", "basis")}
        acquisition = {}
        if "data" in item and isinstance(item["data"], RObject):  # of the data fitted
            data = item["data"]
            acquisition = {
                "SpectrometerFrequency": ([float(data["ft"].value[0]) / 1e6], "data/ft"),
                "ResonantNucleus": ([str(data["nuc"].value[0])], "data/nuc"),
                "ReferencePPM": (float(data["ref"].value[0]), "data/ref"),
            }
        results = []
        for row in range(len(table.value[0].value)):
            labels = tuple(int(table[label].value[row]) for label in self.LABELS)
            rows = []
            for column, signal in zip(amplitudes, signals, strict=True):
                amplitude = self._number(table[column], row)
                sd = self._number(table[f"{column}.sd"], row) if f"{column}.sd" in table else None
                percent = 100 * sd / amplitude if sd is not None and amplitude else None
                rows.append({"metabolite": signal, "concentration": amplitude, "sd": sd,
                             "sd_percent": percent})
            values = {column: plain(RObject(table[column].kind, table[column].value[row:row + 1],
                                            {"levels": table[column].attributes.get("levels")}))
                      for column in diagnostics}
            curves = {}
            if row < len(fits) and isinstance(fits[row], RObject):
                curves = {self.CURVES.get(n, n): np.asarray(v.value, dtype=float)
                          for n, v in zip(fits[row].names, fits[row].value, strict=True)}
            label = name if len(table.value[0].value) == 1 else f"{name} {labels}"
            results.append(self.fit_result(sniff, rows, values, curves, acquisition,
                                           vendor | header, label, [], self.SOURCE, labels))
        if "data" in item and isinstance(item["data"], RObject):
            results.append(self._mrs_data(sniff, item["data"], f"{name} data", vendor))
        if "basis" in item and isinstance(item["basis"], RObject):
            results.append(self._basis(sniff, item["basis"], f"{name} basis", vendor))
        return results

    @staticmethod
    def _signals(columns: list[str], basis: RObject | None) -> list[str]:
        """The signal names: the basis set's own where R made the column names from them
        (an X before a name that cannot start one, then invalid characters as dots)."""
        names = [str(n) for n in basis["names"].value] if basis is not None else []
        signals = list(columns)
        for index, original in enumerate(names[:len(columns)]):
            made = original if re.match(r"[A-Za-z]|\.(?!\d)", original) else "X" + original
            made = re.sub(r"[^A-Za-z0-9._]", ".", made)
            if made == columns[index]:
                signals[index] = original
        return signals

    @staticmethod
    def _number(column: RObject, row: int) -> float | None:
        value = float(column.value[row])
        return None if np.isnan(value) else value
