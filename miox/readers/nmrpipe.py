####################################################################################################
#                                           nmrpipe.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 10/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for NMRPipe files (one file of 1 to 4 dimensions): time-domain data as MRSData   #
#          and processed spectra as MRImage, with the header's axes as standard keys. The binary   #
#          layout is read by nmrglue.                                                              #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import nmrglue as ng
import numpy as np

# own
from miox.data import MRImage, MRSData
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                          NmrPipeReader                                           #
#**************************************************************************************************#
#                                                                                                  #
# NMRPipe numbers its dimensions F2 (acquired, X), F1, F3, F4; FDDIMORDER says which one each axis #
# of the stored array is. Time-domain samples are stored conjugated with respect to the NIfTI-MRS  #
# convention (var2pipe writes the conjugate of the Varian samples) and are conjugated back; the    #
# acquired dimension comes first, each further one on DIM_INDIRECT_0 to 2 with its rows as stored  #
# (complex points as pairs of rows; for States data the pair combines as even - i odd, which       #
# reproduces NMRPipe's own processing). A spectrum (every dimension transformed) becomes an        #
# MRImage whose axes are placed by the header (FDF*ORIG, SW, OBS, as nmrglue's unit conversion     #
# reads it), with the chemical shift falling with the index. Transposed time-domain data and data  #
# mixing time and frequency dimensions are not read.                                               #
#                                                                                                  #
#**************************************************************************************************#
class NmrPipeReader(Reader):
    FORMATS = (Format.NMRPIPE,)
    PACKAGES = Reader.PACKAGES + ("nmrglue",)
    INDIRECT = ("DIM_INDIRECT_0", "DIM_INDIRECT_1", "DIM_INDIRECT_2")
    NUCLEUS = re.compile(r"([A-Za-z]+)(\d+)")  # H1, C13, N15
    MODES = {0: "magnitude", 1: "TPPI", 2: "States", 3: "image", 4: "array"}  # FD2DPHASE

    def read(self, sniff: Sniff) -> list[MRSData | MRImage]:
        path = Path(sniff.path)
        try:
            dic, data = ng.pipe.read(str(path))
        except (ValueError, OSError, KeyError) as e:
            raise ReaderError(f"{path}: not a readable NMRPipe file ({e})") from e
        data = np.asarray(data)
        dims = [int(dic["FDDIMORDER"][data.ndim - 1 - axis]) for axis in range(data.ndim)]
        order = sorted(range(data.ndim), key=lambda axis: (2, 1, 3, 4).index(dims[axis]))
        transformed = [bool(dic[f"FDF{dims[axis]}FTFLAG"]) for axis in order]
        notes = []
        if int(dic.get("FDFILECOUNT", 1)) > 1:
            notes.append(f"one file of a series of {int(dic['FDFILECOUNT'])}: only its planes")
        if all(transformed):
            return [self._spectrum(sniff, dic, data, order, dims, notes)]
        if any(transformed):
            raise ReaderError(f"{path}: time and frequency dimensions mixed (not supported)")
        if order[0] != data.ndim - 1:
            raise ReaderError(f"{path}: transposed time-domain data (not supported)")
        return [self._fid(sniff, dic, data, order, dims, notes)]

    # --- time domain -----------------------------------------------------------------------------

    def _fid(self, sniff: Sniff, dic: dict, data: np.ndarray, order: list[int], dims: list[int],
             notes: list[str]) -> MRSData:
        samples = np.conj(np.transpose(data, order)).astype(np.complex128)
        if samples.ndim > 4:
            raise ReaderError(f"{sniff.path}: {samples.ndim} dimensions, NIfTI-MRS holds at most 4")
        header = [dims[axis] for axis in order]
        values = self._keys(dic, header)
        width = float(dic["FDF2SW"])
        values["ReferencePPM"] = (float(dic["FDF2CAR"]), "FDF2CAR")
        tags = tuple((self.INDIRECT[i], self._info(dic, f), None)
                     for i, f in enumerate(header[1:]))
        if dic.get("FDF2AQSIGN"):
            notes.append(f"FDF2AQSIGN {int(dic['FDF2AQSIGN'])}: sign alternation as stored")
        d = self.dataset(sniff, samples.reshape((1, 1, 1) + samples.shape), 1 / width, values,
                         self._vendor(dic), Path(sniff.path).stem, notes, "NMRPipe header",
                         dim_tags=tags)
        d.meta.set("DwellTime", 1 / width, "NMRPipe header: 1 / FDF2SW", "FDF2SW")
        return d

    def _info(self, dic: dict, f: int) -> str:
        """An indirect time dimension in words."""
        quad = "complex points as pairs of rows" if not dic[f"FDF{f}QUADFLAG"] else "real"
        mode = self.MODES.get(int(dic.get("FD2DPHASE", -1)), "unknown mode")
        if mode == "States" and not dic[f"FDF{f}QUADFLAG"]:
            mode = "States (after the conjugation the indirect signal is even - i odd)"
        return (f"F{f} {dic[f'FDF{f}LABEL']}: SW {float(dic[f'FDF{f}SW']):.6g} Hz, OBS "
                f"{float(dic[f'FDF{f}OBS']):.9g} MHz, CAR {float(dic[f'FDF{f}CAR']):.6g} ppm, "
                f"{quad}, {mode}, rows as stored")

    # --- spectra ---------------------------------------------------------------------------------

    def _spectrum(self, sniff: Sniff, dic: dict, data: np.ndarray, order: list[int],
                  dims: list[int], notes: list[str]) -> MRImage:
        axes = [ng.pipe.make_uc(dic, data, axis).ppm_scale() for axis in range(data.ndim)]
        values = np.transpose(data, order)
        axes = [axes[axis] for axis in order]
        for i, ppm in enumerate(axes):
            if ppm[0] < ppm[-1]:  # the shift falls with the index
                values, axes[i] = np.flip(values, axis=i), ppm[::-1]
        values = values.astype(np.complex128 if np.iscomplexobj(values) else np.float64)
        header = [dims[axis] for axis in order]
        if any(not dic[f"FDF{f}QUADFLAG"] for f in header[1:]):
            notes.append("complex indirect points as pairs of rows, as stored")
        meta = Meta(self._vendor(dic))
        for key, (value, field) in self._keys(dic, header).items():
            meta.set(key, value, f"NMRPipe header: {field}", field)
        n, ppm = values.shape[0], axes[0]
        step = float(abs(ppm[1] - ppm[0])) if n > 1 else 0.0
        frequency = float(dic["FDF2OBS"])
        meta.set("NumberOfPoints", int(n), "data shape")
        if step:
            meta.set("SpectralWidth", step * frequency * n, "NMRPipe header: axis spacing * points",
                     "FDF2SW")
            meta.set("DwellTime", 1 / (step * frequency * n), "derived: 1 / SpectralWidth")
        meta.set("ReferencePPM", float(ppm[n // 2]), "NMRPipe header: shift of point N / 2",
                 "FDF2ORIG")
        if values.ndim > 1:
            meta.set("IndirectReferencePPM", [float(a[a.size // 2]) for a in axes[1:]],
                     "NMRPipe header: shift of point N / 2 of each indirect axis", "FDF1ORIG")
            meta.set("IndirectSpectralWidth",
                     [float(abs(a[1] - a[0])) * float(dic[f"FDF{f}OBS"]) * a.size
                      for a, f in zip(axes[1:], header[1:], strict=True)],
                     "NMRPipe header: axis spacing * points of each indirect axis", "FDF1SW")
        names = ("frequency",) + tuple(f"frequency_indirect_{i}" for i in range(values.ndim - 1))
        return MRImage(values.reshape((1, 1, 1) + values.shape), self.NO_GEOMETRY, meta,
                       self.provenance(sniff, tuple(notes)), Path(sniff.path).stem, names)

    # --- header ----------------------------------------------------------------------------------

    def _keys(self, dic: dict, header: list[int]) -> dict[str, tuple]:
        """Frequencies and nuclei of the dimensions in ``header`` order (acquired first)."""
        values = {"SpectrometerFrequency": ([float(dic[f"FDF{f}OBS"]) for f in header],
                                            "FDF2OBS")}
        nuclei = [self._nucleus(str(dic[f"FDF{f}LABEL"])) for f in header]
        if None not in nuclei:
            values["ResonantNucleus"] = (nuclei, "FDF2LABEL")
        return values

    def _nucleus(self, label: str) -> str | None:
        """A nucleus as NIfTI-MRS spells it (1H) from an NMRPipe label (H1, HN is not one)."""
        match = self.NUCLEUS.fullmatch(label.strip())
        return match.group(2) + match.group(1) if match else None

    def _vendor(self, dic: dict) -> dict[str, Any]:
        return {k: self.plain(v) for k, v in dic.items() if k != "FDATA"}
