####################################################################################################
#                                          spinsolve.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 10/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for Magritek Spinsolve data (Prospa .1d files: data.1d, fid.1d, spectrum.1d,     #
#          spectrum_processed.1d) with the acquisition parameters of acqu.par as standard keys; a  #
#          Spinsolve experiment folder gives all of its data files.                                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Any

import numpy as np

# own
from miox.data import MRImage, MRSData
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                         SpinsolveReader                                          #
#**************************************************************************************************#
#                                                                                                  #
# A Prospa .1d file holds a 32-byte header (owner, format, version, data type, four dimensions),   #
# for data type 504 the x axis (s for an FID, ppm for a spectrum), and the complex values as       #
# float32 pairs (type 501: no axis, the dwell time from acqu.par dwellTime in us). FIDs are        #
# in the NIfTI-MRS convention as stored (their transform lies on Spinsolve's own spectrum; the     #
# JCAMP-DX export nmr_fid.dx holds the same samples). Spectra run with the shift rising and are    #
# reversed, their complex values kept. The carrier lies at (lowestFrequency + bandwidth / 2) /     #
# b1Freq ppm (acqu.par; centerFreqPPM rounds it), the dwell time is the step of the time axis.     #
#                                                                                                  #
#**************************************************************************************************#
class SpinsolveReader(Reader):
    FORMATS = (Format.SPINSOLVE,)
    MAGIC = b"SORPATAD"
    COMPLEX_FLOAT = 501  # data type of complex float32 values
    COMPLEX_FLOAT_AXIS = 504  # data type of complex float32 values after an x axis
    DATA_FILES = ("data.1d", "fid.1d", "spectrum.1d", "spectrum_processed.1d")
    PARAMETER = re.compile(r"^\s*([^=]+?)\s*=\s*(.*?)\s*$")
    NUCLEUS = re.compile(r"(\d+)([A-Za-z]+)")

    def read(self, sniff: Sniff) -> list[MRSData | MRImage]:
        path = Path(sniff.path)
        files = [path / name for name in self.DATA_FILES if (path / name).is_file()] \
            if path.is_dir() else [path]
        if not files:
            raise ReaderError(f"{path}: no Spinsolve data file")
        return [self._file(sniff, file) for file in files]

    def _file(self, sniff: Sniff, path: Path) -> MRSData | MRImage:
        raw = path.read_bytes()
        if len(raw) < 32 or raw[:8] != self.MAGIC:
            raise ReaderError(f"{path}: not a Prospa data file")
        kind, *dims = struct.unpack("<5i", raw[12:32])
        known = kind in (self.COMPLEX_FLOAT, self.COMPLEX_FLOAT_AXIS)
        if not known or any(d != 1 for d in dims[1:]):
            raise ReaderError(f"{path}: Prospa data type {kind} with dimensions {dims} is not "
                              "supported (no sample)")
        n = dims[0]
        width = 3 if kind == self.COMPLEX_FLOAT_AXIS else 2
        values = np.frombuffer(raw[32:32 + 4 * width * n], "<f4")
        if values.size != width * n:
            raise ReaderError(f"{path}: {values.size} values, {width * n} expected")
        axis = values[:n].astype(np.float64) if kind == self.COMPLEX_FLOAT_AXIS else None
        start = n if axis is not None else 0
        data = (values[start::2] + 1j * values[start + 1::2]).astype(np.complex128)
        parameters = self._parameters(path.parent)
        vendor = {"header": dict(zip(("data_type", "x", "y", "z", "q"), [kind, *dims],
                                     strict=True)), **parameters}
        if path.name.startswith("spectrum"):
            if axis is None:
                raise ReaderError(f"{path}: spectrum without its ppm axis (no sample)")
            return self._spectrum(sniff, path, axis, data, parameters, vendor)
        return self._fid(sniff, path, axis, data, parameters, vendor)

    def _fid(self, sniff: Sniff, path: Path, axis: np.ndarray | None, data: np.ndarray,
             parameters: dict, vendor: dict) -> MRSData:
        if axis is not None:
            dwelltime, field = float(axis[1] - axis[0]), f"{path.name}: step of the time axis"
        else:
            dwell = parameters.get("acqu", {}).get("dwellTime")
            if not isinstance(dwell, (int, float)):
                raise ReaderError(f"{path}: no time axis and no dwellTime in acqu.par")
            dwelltime, field = float(dwell) * 1e-6, "acqu.par: dwellTime (us)"
        values = self._keys(parameters)
        if (reference := self._reference(parameters)) is not None:
            values["ReferencePPM"] = reference
        name = f"{path.parent.name}_{path.stem}"
        d = self.dataset(sniff, data.reshape(1, 1, 1, -1), dwelltime, values, vendor, name, [],
                         "acqu.par")
        d.meta.set("DwellTime", dwelltime, field, "x axis" if axis is not None else "dwellTime")
        return d

    def _spectrum(self, sniff: Sniff, path: Path, axis: np.ndarray, data: np.ndarray,
                  parameters: dict, vendor: dict) -> MRImage:
        if axis[0] < axis[-1]:  # the shift falls with the index
            axis, data = axis[::-1], data[::-1]
        n = data.size
        meta = Meta(vendor)
        for key, (value, field) in self._keys(parameters).items():
            meta.set(key, value, f"acqu.par: {field}", field)
        frequency = meta.get("SpectrometerFrequency", [None])[0]
        step = float(axis[0] - axis[-1]) / (n - 1)  # the span: float32 neighbours are coarse
        meta.set("NumberOfPoints", int(n), "data shape")
        meta.set("ReferencePPM", float(axis[0]) - step * (n // 2),
                 f"{path.name}: ppm axis at point N / 2", "x axis")
        if frequency:
            meta.set("SpectralWidth", step * frequency * n, f"{path.name}: ppm axis * b1Freq",
                     "x axis")
            meta.set("DwellTime", 1 / (step * frequency * n), "derived: 1 / SpectralWidth")
        name = f"{path.parent.name}_{path.stem}"
        return MRImage(data.reshape(1, 1, 1, -1), self.NO_GEOMETRY, meta, self.provenance(sniff),
                       name, ("frequency",))

    # --- parameters ------------------------------------------------------------------------------

    def _parameters(self, folder: Path) -> dict[str, dict[str, Any]]:
        """acqu.par and proc.par of the folder as dicts of values (numbers where they are)."""
        out = {}
        for name in ("acqu.par", "proc.par"):
            file = folder / name
            if not file.is_file():
                continue
            values = {}
            for line in file.read_text(errors="replace").splitlines():
                if match := self.PARAMETER.match(line):
                    values[match.group(1)] = self._value(match.group(2))
            out[name.removesuffix(".par")] = values
        return out

    @staticmethod
    def _value(text: str) -> Any:
        if text.startswith('"') and text.endswith('"'):
            return text[1:-1]
        try:
            return int(text)
        except ValueError:
            try:
                return float(text)
            except ValueError:
                return text

    def _keys(self, parameters: dict) -> dict[str, tuple]:
        acqu = parameters.get("acqu", {})
        values: dict[str, tuple] = {"Manufacturer": ("Magritek", "file format")}
        if isinstance(acqu.get("b1Freq"), (int, float)):
            values["SpectrometerFrequency"] = ([float(acqu["b1Freq"])], "b1Freq")
        if (match := self.NUCLEUS.fullmatch(str(acqu.get("rxChannel", "")))):
            values["ResonantNucleus"] = ([match.group(1) + match.group(2)], "rxChannel")
        if isinstance(acqu.get("nrScans"), int):
            values["NumberOfAverages"] = (acqu["nrScans"], "nrScans")
        if str(acqu.get("Solvent", "None")) not in ("None", ""):
            values["Solvent"] = (str(acqu["Solvent"]), "Solvent")
        if acqu.get("startTime"):
            values["AcquisitionDateTime"] = (str(acqu["startTime"]), "startTime")
        for field in ("Protocol", "experiment"):
            if acqu.get(field):
                values["SequenceName"] = (str(acqu[field]), field)
                break
        return values

    @staticmethod
    def _reference(parameters: dict) -> tuple[float, str] | None:
        acqu = parameters.get("acqu", {})
        try:
            low, width = float(acqu["lowestFrequency"]), float(acqu["bandwidth"]) * 1e3
            return (low + width / 2) / float(acqu["b1Freq"]), \
                "(lowestFrequency + bandwidth / 2) / b1Freq"
        except (KeyError, TypeError, ValueError):
            if isinstance(acqu.get("centerFreqPPM"), (int, float)):
                return float(acqu["centerFreqPPM"]), "centerFreqPPM"
        return None
