####################################################################################################
#                                             jcamp.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 09/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for NMR JCAMP-DX files (.jdx, .dx): FIDs and spectra, 1D and nD, as XYDATA,      #
#          NTUPLES pages or XY tables, one dataset per block of a compound (LINK) file. Blocks and #
#          pages are split and decoded by nmrglue's JCAMP-DX parser; a plain-number (AFFN) data    #
#          table of an nD page is read through numpy, which is what makes 2D spectra fast.         #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re
from typing import Any

import numpy as np
from nmrglue.fileio import jcampdx

# own
from miox.data import MRImage, MRSData
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.readers.topspin import TopSpinReader
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                           JcampReader                                            #
#**************************************************************************************************#
#                                                                                                  #
# Labels are compared as nmrglue keeps them: upper case without spaces, dashes or slashes (DATA    #
# TYPE is DATATYPE, Bruker's ##$SW_h is $SWH). An FID gives an MRSData: R + iI conjugated into the #
# NIfTI-MRS convention (Bruker, Nanalysis and Q Magnetics FIDs then show their known signals at    #
# their shifts), except when $REVERSE of the direct axis is yes (Oxford Instruments FIDs are in    #
# the convention as stored); the pages of an nD FID as rows on DIM_INDIRECT_0 with their page      #
# values as IndirectTime. A spectrum gives an MRImage whose chemical shift falls with the index,   #
# as TopSpin stores spectra; the shift of each point comes from X in ppm, or in Hz divided by the  #
# observe frequency, where .SHIFT REFERENCE (type, compound, point counted from 1, shift) places   #
# the point when X is not counted from 0 ppm (Bruker's NTUPLES run X from the spectral width down  #
# to 0). The carrier of an FID lies at (SFO1 - SF) / SF when the export carries those parameters   #
# (their $OFFSET and $SW_p may belong to another processing), else at the centre of its spectrum   #
# (point N / 2 + 1) on the .SHIFT REFERENCE axis.                                                  #
#                                                                                                  #
#**************************************************************************************************#
class JcampReader(Reader):
    FORMATS = (Format.JCAMP_NMR,)
    PACKAGES = Reader.PACKAGES + ("nmrglue",)
    FIDS = ("NMRFID", "NDNMRFID")
    SPECTRA = ("NMRSPECTRUM", "NDNMRSPECTRUM")
    NUCLEUS = re.compile(r"\^?(\d+)([A-Za-z]+)|([A-Za-z]+)(\d+)")
    COMMENT = re.compile(r"\$\$.*")
    PAGE = re.compile(r"^\s*(\w+)\s*=\s*(\S+)")
    PLAIN = re.compile(r"[\s,0-9.eE+-]*")  # the characters of an AFFN table
    PACKED = re.compile(r"[0-9.][+-]")  # a sign right after a digit: PAC form
    KEYS = (  # (standard key, label, kind)
        ("SequenceName", ".PULSESEQUENCE", str),
        ("Solvent", ".SOLVENTNAME", str),
        ("ManufacturersModelName", "SPECTROMETERDATASYSTEM", str),
        ("NumberOfAverages", ".AVERAGES", int),
        ("SampleTemperature", "$TE", float),
        ("TxCoil", "$PROBHD", str),
        ("RxCoil", "$PROBHD", str),
    )

    def read(self, sniff: Sniff) -> list[MRSData | MRImage]:
        notes = []
        with self.captured(notes):
            raw = jcampdx._readrawdic(str(sniff.path))
        notes = list(dict.fromkeys(notes))  # nmrglue warns once per occurrence of a label
        datasets = []
        for label, blocks in raw.items():
            kind = label.removeprefix("_datatype_")
            if kind not in self.FIDS + self.SPECTRA:
                continue
            for block in blocks:
                name = f"{sniff.path.stem}_{len(datasets)}" if datasets else sniff.path.stem
                if kind in self.FIDS:
                    datasets.append(self._fid(sniff, block, name, list(notes)))
                else:
                    datasets.append(self._spectrum(sniff, block, name, list(notes)))
        if not datasets:
            raise ReaderError(f"{sniff.path}: no NMR FID or spectrum block")
        return datasets

    # --- FIDs ------------------------------------------------------------------------------------

    def _fid(self, sniff: Sniff, block: dict, name: str, notes: list[str]) -> MRSData:
        rows, pages = self._rows(block)
        data = np.asarray([r + 1j * i for r, i in rows]).T
        if (self._first(block, "$REVERSE") or "").lower() != "yes":
            data = np.conj(data)
        symbols = self._columns(block, "SYMBOL")
        direct = symbols[1] if pages else symbols[0]  # T2 of (T1, T2, R, I), X of (X, R, I)
        n = data.shape[0]
        first, last = self._number(block, "FIRST", direct), self._number(block, "LAST", direct)
        dwelltime = (last - first) / (n - 1) * self._seconds(block, direct)
        values = self._keys(block)
        if (reference := self._reference(block, n, 1 / dwelltime)) is not None:
            values["ReferencePPM"] = reference
        tags = ()
        if pages:
            unit = self._seconds(block, symbols[0])
            times = [p * unit for p in pages]
            header = {"IndirectTime": {"Value": times, "Description": "evolution time in s"}}
            nuclei = [self._normal(n) for n in reversed(self._columns(block, ".NUCLEUS"))]
            frequencies = block.get("$SFO1", [])
            if len(nuclei) == 2 and None not in nuclei and len(frequencies) >= 2:
                values["SpectrometerFrequency"] = ([float(f) for f in frequencies[:2]], "$SFO1")
                values["ResonantNucleus"] = (nuclei, ".NUCLEUS")
            nucleus = f" ({nuclei[1]})" if len(nuclei) == 2 and nuclei[1] else ""
            tags = (("DIM_INDIRECT_0", f"pages of {symbols[0]}{nucleus}", header),)
        else:
            data = data[:, 0]
        data = data.reshape((1, 1, 1) + data.shape)
        source = "JCAMP-DX labels"
        d = self.dataset(sniff, data, dwelltime, values, self._vendor(block), name, notes,
                         source, dim_tags=tags)
        d.meta.set("DwellTime", dwelltime, f"{source}: FIRST and LAST of {direct}", "FIRST")
        return d

    def _reference(self, block: dict, n: int, width: float) -> tuple[float, str] | None:
        """Shift of the carrier. Exports that carry the acquisition and processing frequencies
        $SFO1 and $SF (Bruker, Nanalysis) place it as TopSpin does: (SFO1 - SF) / SF, SF being
        the frequency of 0 ppm; otherwise .SHIFT REFERENCE names the shift of one point of the
        spectrum, counted from 1."""
        carrier, zero = self._first(block, "$SFO1"), self._first(block, "$SF")
        if carrier and zero:
            return (float(carrier) - float(zero)) / float(zero) * 1e6, "($SFO1 - $SF) / $SF"
        stated = self._shift_reference(block)
        frequency = self._observe(block)
        if stated is None or frequency is None:
            return None
        point, shift = stated
        centre = n / 2 + 1
        return shift - (centre - point) * width / n / frequency, ".SHIFTREFERENCE"

    # --- spectra ---------------------------------------------------------------------------------

    def _spectrum(self, sniff: Sniff, block: dict, name: str, notes: list[str]) -> MRImage:
        frequency = self._observe(block)
        if frequency is None:
            raise ReaderError(f"{sniff.path}: spectrum without .OBSERVE FREQUENCY")
        x, values = self._spectrum_values(block, notes)
        ppm = self._ppm(block, x, frequency)
        if ppm[0] < ppm[-1]:  # chemical shift falls with the index
            ppm, values = ppm[::-1], values[..., ::-1]
        data = values.T
        dims = ("frequency",) + (("frequency_indirect_0",) if data.ndim > 1 else ())
        step = float(abs(ppm[1] - ppm[0]))
        n = data.shape[0]
        meta = Meta(self._vendor(block))
        source = "JCAMP-DX labels"
        meta.set("SpectrometerFrequency", [frequency], f"{source}: .OBSERVEFREQUENCY",
                 ".OBSERVEFREQUENCY")
        if nucleus := self._nucleus(block):
            meta.set("ResonantNucleus", [nucleus], f"{source}: .OBSERVENUCLEUS", ".OBSERVENUCLEUS")
        meta.set("NumberOfPoints", int(n), "data shape")
        meta.set("SpectralWidth", step * frequency * n, f"{source}: X spacing * points")
        meta.set("DwellTime", 1 / (step * frequency * n), "derived: 1 / SpectralWidth")
        meta.set("ReferencePPM", float(ppm[n // 2]), f"{source}: shift of point N / 2")
        for key, (value, field) in self._keys(block).items():
            if key not in ("SpectrometerFrequency", "ResonantNucleus"):
                meta.set(key, value, f"{source}: {field}", field)
        if data.ndim > 1:
            notes.append("indirect axis: page values in the vendor header (PAGE)")
        data = data.reshape((1, 1, 1) + data.shape)
        return MRImage(data, self.NO_GEOMETRY, meta, self.provenance(sniff, tuple(notes)), name,
                       dims)

    def _spectrum_values(self, block: dict, notes: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """The X value of each point and the spectrum (rows of an nD spectrum first); real, or
        complex when real and imaginary parts are given."""
        if "DATATABLE" in block:
            rows, pages = self._rows(block)
            symbols = self._columns(block, "SYMBOL")
            direct = symbols[1] if pages else symbols[0]
            values = np.asarray([r if i is None else r + 1j * i for r, i in rows])
            values = values if pages else values[0]
            n = values.shape[-1]
            first, last = self._number(block, "FIRST", direct), self._number(block, "LAST", direct)
            return np.linspace(first, last, n), values
        if "XYDATA" in block:
            values, _ = jcampdx._parse_data(block["XYDATA"][0])
            stated = self._first(block, "NPOINTS")
            if stated and int(float(stated)) != values.size:
                notes.append(f"NPOINTS {stated} but {values.size} values, which FIRSTX, "
                                   "LASTX and DELTAX span")
            values = values * float(self._first(block, "YFACTOR") or 1)
            first, last = float(self._first(block, "FIRSTX")), float(self._first(block, "LASTX"))
            return np.linspace(first, last, values.size), values
        if "PEAKTABLE" in block:  # X and Y pairs; a spectrum when X is evenly spaced
            text = block["PEAKTABLE"][0].split("\n", 1)[1]
            pairs = np.array(re.split(r"[,;\s]+", text.strip()), dtype=float).reshape(-1, 2)
            x, y = pairs[:, 0], pairs[:, 1]
            if not np.allclose(np.diff(x), x[1] - x[0], rtol=1e-4):
                raise ReaderError("XY table with unevenly spaced X: a peak list, not a spectrum")
            return x, y
        raise ReaderError("no XYDATA, DATA TABLE or PEAK TABLE")

    def _ppm(self, block: dict, x: np.ndarray, frequency: float) -> np.ndarray:
        """Shift of each point from X in ppm or Hz and the shift reference."""
        units = self._columns(block, "UNITS")
        unit = (self._first(block, "XUNITS") or (units[0] if units else "HZ"))
        unit = unit.strip("() ").upper()
        if unit == "PPM":
            return x
        ppm = x / frequency
        stated = self._shift_reference(block)
        if stated is not None:
            point, shift = stated
            index = min(max(int(round(point)) - 1, 0), x.size - 1)
            if abs(ppm[index] - shift) > abs(ppm[1] - ppm[0]):  # X not counted from 0 ppm
                ppm = shift + (x - x[index]) / frequency
        return ppm

    # --- labels ----------------------------------------------------------------------------------

    def _rows(self, block: dict) -> tuple[list[tuple[np.ndarray, np.ndarray | None]], list]:
        """NTUPLES pages as (real, imaginary) rows in page order, with the page values of an nD
        block (empty for the numbered pages of a 1D block)."""
        symbols = self._columns(block, "SYMBOL")
        factors = self._columns(block, "FACTOR")
        parts: dict[str, dict[str, np.ndarray]] = {}
        for page, table in zip(block.get("PAGE", []), block["DATATABLE"], strict=True):
            values, part = self._table(table)
            if part in symbols and symbols.index(part) < len(factors):
                values = values * float(factors[symbols.index(part)] or 1)
            label = self.PAGE.match(page)
            key = label.group(2) if label and label.group(1) != "N" else ""
            parts.setdefault(key, {})["I" if part == "I" else "R"] = values
        rows = [(p["R"], p.get("I")) for p in parts.values()]
        pages = [float(k) for k in parts if k]
        return rows, pages

    @classmethod
    def _table(cls, text: str) -> tuple[np.ndarray, str]:
        """A data table's values and part (R or I) as nmrglue decodes it; a table of plain
        numbers (AFFN, one X then the Y values per line) goes through numpy instead."""
        header, _, body = text.partition("\n")
        if cls.PLAIN.fullmatch(body) and not cls.PACKED.search(body):
            rows = [np.fromstring(line.replace(",", " "), dtype=np.float64, sep=" ")
                    for line in body.split("\n")]
            values = np.concatenate([row[1:] for row in rows if row.size > 1])
            return values, "I" if "I..I" in header else "R"
        return jcampdx._parse_data(text)

    def _keys(self, block: dict) -> dict[str, tuple[Any, str]]:
        values = {}
        if (frequency := self._observe(block)) is not None:
            values["SpectrometerFrequency"] = ([frequency], ".OBSERVEFREQUENCY")
        if nucleus := self._nucleus(block):
            values["ResonantNucleus"] = ([nucleus], ".OBSERVENUCLEUS")
        for key, label, kind in self.KEYS:
            if text := self._first(block, label):
                try:
                    values[key] = (kind(float(text)) if kind is not str else text, label)
                except ValueError:
                    continue
        delay = TopSpinReader._group_delay({
            name: self._first(block, f"${name}") for name in ("GRPDLY", "DSPFVS", "DECIM")
            if self._first(block, f"${name}") is not None
        })
        if delay:
            values["GroupDelay"] = (delay[0], "$GRPDLY" if "GRPDLY" in delay[1] else "$DSPFVS")
        return values

    def _observe(self, block: dict) -> float | None:
        text = self._first(block, ".OBSERVEFREQUENCY")
        return float(text) if text else None

    def _nucleus(self, block: dict) -> str | None:
        return self._normal(self._first(block, ".OBSERVENUCLEUS") or "")

    def _normal(self, text: str) -> str | None:
        """A nucleus as NIfTI-MRS spells it (1H, 13C) from ^1H, 1H or H1."""
        match = self.NUCLEUS.fullmatch(text.replace(" ", ""))
        if not match:
            return None
        mass, symbol = (match.group(1), match.group(2)) if match.group(1) else match.group(4, 3)
        return mass + symbol

    def _shift_reference(self, block: dict) -> tuple[float, float] | None:
        """(point, shift) of .SHIFT REFERENCE = (type, compound, point, shift)."""
        text = (self._first(block, ".SHIFTREFERENCE") or "").strip("() ")
        parts = [p.strip() for p in text.split(",")]
        try:
            return float(parts[2]), float(parts[3])
        except (IndexError, ValueError):
            return None

    def _seconds(self, block: dict, symbol: str) -> float:
        units = self._columns(block, "UNITS")
        symbols = self._columns(block, "SYMBOL")
        unit = units[symbols.index(symbol)].upper() if symbol in symbols else "SECONDS"
        return {"MILLISECONDS": 1e-3, "MICROSECONDS": 1e-6}.get(unit, 1.0)

    def _number(self, block: dict, label: str, symbol: str) -> float:
        return float(self._columns(block, label)[self._columns(block, "SYMBOL").index(symbol)])

    def _columns(self, block: dict, label: str) -> list[str]:
        text = self._first(block, label)
        return [c.strip() for c in text.split(",")] if text else []

    def _first(self, block: dict, label: str) -> str | None:
        """The first value of a label without its comment."""
        values = block.get(label)
        if not values:
            return None
        return self.COMMENT.sub("", str(values[0])).strip() or None

    def _vendor(self, block: dict) -> dict[str, Any]:
        """Every label of the block except the data, values as written."""
        skip = ("DATATABLE", "XYDATA", "PEAKTABLE")
        return {k: v[0] if len(v) == 1 else v for k, v in block.items()
                if k not in skip and isinstance(v, list)}
