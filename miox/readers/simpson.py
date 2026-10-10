####################################################################################################
#                                           simpson.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 10/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for SIMPSON simulation output (fsave files, text or binary): FIDs as MRSData and #
#          spectra as MRImage. The files are read by nmrglue.                                      #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

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
#                                          SimpsonReader                                           #
#**************************************************************************************************#
#                                                                                                  #
# SIMPSON writes the FID with lines of positive chemical shift at positive frequencies of a        #
# Fourier transform, the conjugate of the NIfTI-MRS convention: FIDs are conjugated. Its spectra   #
# are that transform (fftshift order, the first FID point halved), the shift rising with the       #
# index; they are turned into the transform of the conjugated FID, conj(spe[(N - j) mod N]), whose #
# shift falls with the index and whose centre point stays the carrier. The files state the         #
# spectral width but neither the spectrometer frequency, the nucleus nor the shift of the carrier  #
# (the simulation's rotating frame): those are left missing. Two-dimensional files (NI > 1) keep   #
# their rows on DIM_INDIRECT_0.                                                                    #
#                                                                                                  #
#**************************************************************************************************#
class SimpsonReader(Reader):
    FORMATS = (Format.SIMPSON,)
    PACKAGES = Reader.PACKAGES + ("nmrglue",)

    def read(self, sniff: Sniff) -> list[MRSData | MRImage]:
        path = Path(sniff.path)
        try:
            dic, data = ng.simpson.read(str(path))
        except (ValueError, OSError, KeyError, IndexError) as e:
            raise ReaderError(f"{path}: not a readable SIMPSON file ({e})") from e
        data = np.asarray(data, dtype=np.complex128)
        data = data.reshape(-1, data.shape[-1])  # (rows, points)
        if data.shape[0] == 1:
            data = data[0]
        kind = str(dic.get("TYPE", "FID")).upper()
        width = float(dic["SW"])
        notes = ["SIMPSON states no spectrometer frequency, nucleus or carrier shift"]
        if kind == "FID":
            return [self._fid(sniff, dic, data, width, notes)]
        if kind == "SPE":
            return [self._spectrum(sniff, dic, data, width, notes)]
        raise ReaderError(f"{path}: SIMPSON data of TYPE {kind!r} are not supported")

    def _fid(self, sniff: Sniff, dic: dict, data: np.ndarray, width: float,
             notes: list[str]) -> MRSData:
        samples = np.conj(data).T  # direct dimension first
        tags = ()
        if samples.ndim > 1:
            info = f"NI {samples.shape[1]} rows as stored"
            if dic.get("SW1"):
                info += f", SW1 {float(dic['SW1']):.6g} Hz"
            tags = (("DIM_INDIRECT_0", info, None),)
        d = self.dataset(sniff, samples.reshape((1, 1, 1) + samples.shape), 1 / width, {},
                         self._vendor(dic), Path(sniff.path).stem, notes, "SIMPSON header",
                         dim_tags=tags)
        d.meta.set("DwellTime", 1 / width, "SIMPSON header: 1 / SW", "SW")
        return d

    def _spectrum(self, sniff: Sniff, dic: dict, data: np.ndarray, width: float,
                  notes: list[str]) -> MRImage:
        n = data.shape[-1]
        values = np.conj(data[..., (n - np.arange(n)) % n]).T  # direct axis first
        meta = Meta(self._vendor(dic))
        meta.set("NumberOfPoints", int(n), "data shape")
        meta.set("SpectralWidth", width, "SIMPSON header: SW", "SW")
        meta.set("DwellTime", 1 / width, "derived: 1 / SpectralWidth")
        names = ("frequency",) + (("rows",) if values.ndim > 1 else ())
        return MRImage(values.reshape((1, 1, 1) + values.shape), self.NO_GEOMETRY, meta,
                       self.provenance(sniff, tuple(notes)), Path(sniff.path).stem, names)

    def _vendor(self, dic: dict) -> dict[str, Any]:
        return {k: self.plain(v) for k, v in dic.items()}
