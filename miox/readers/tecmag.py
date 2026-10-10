####################################################################################################
#                                            tecmag.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 10/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for Tecmag TNMR data files (.tnt): the time-domain data with the TMAG and TMG2   #
#          parameters as standard keys. The binary layout is read by nmrglue.                      #
#                                                                                                  #
####################################################################################################


#*************                                                                                     #
#   imports                                                                                        #
#*************                                                                                     #
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from nmrglue.fileio import tecmag as ng_tecmag

# own
from miox.data import MRSData
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                           TecmagReader                                           #
#**************************************************************************************************#
#                                                                                                  #
# The samples are kept as stored (TNMR's own text export lists the same values): no sample with a  #
# known shift settles their sign convention. The file states the carrier (ob_freq) but not the     #
# frequency of 0 ppm, so ReferencePPM is left out. Further dimensions (actual_npts 2 to 4, in      #
# Fortran order) go on DIM_INDIRECT_0 to 2 as stored. Frequency-domain data (fft_flag set) are     #
# not read.                                                                                        #
#                                                                                                  #
#**************************************************************************************************#
class TecmagReader(Reader):
    FORMATS = (Format.TECMAG_TNT,)
    PACKAGES = Reader.PACKAGES + ("nmrglue",)
    INDIRECT = ("DIM_INDIRECT_0", "DIM_INDIRECT_1", "DIM_INDIRECT_2")
    NUCLEUS = re.compile(r"([A-Za-z]+)(\d+)|(\d+)([A-Za-z]+)")  # Li7, 7Li

    def read(self, sniff: Sniff) -> list[MRSData]:
        path = Path(sniff.path)
        try:
            dic, data = ng_tecmag.read(str(path))
        except (ValueError, KeyError, AssertionError) as e:
            raise ReaderError(f"{path}: not a readable TNMR file ({e})") from e
        if np.any(dic["fft_flag"]):
            raise ReaderError(f"{path}: frequency-domain data (fft_flag set) are not supported")
        data = np.asarray(data, dtype=np.complex128)
        while data.ndim > 1 and data.shape[-1] == 1:
            data = data[..., 0]
        notes = ["sign convention not verified (no sample with a known shift): samples kept as "
                 "stored", "the frequency of 0 ppm is not stated: no ReferencePPM"]
        tags = tuple(
            (self.INDIRECT[i], f"dimension {i + 2} as stored: sw {float(dic['sw'][i + 1]):.6g} "
             f"Hz, dwell {float(dic['dwell'][i + 1]):.6g} s", None)
            for i in range(data.ndim - 1)
        )
        dwelltime = float(dic["dwell"][0])
        d = self.dataset(sniff, data.reshape((1, 1, 1) + data.shape), dwelltime,
                         self._keys(dic), self._vendor(dic), path.stem, notes, "TNMR parameters",
                         dim_tags=tags)
        d.meta.set("DwellTime", dwelltime, "TNMR parameters: dwell", "dwell")
        return [d]

    def _keys(self, dic: dict) -> dict[str, tuple]:
        values: dict[str, tuple] = {
            "Manufacturer": ("Tecmag", "file format"),
            "SpectrometerFrequency": ([float(dic["ob_freq"][0])], "ob_freq"),
            "NumberOfAverages": (int(dic["actual_scans"]), "actual_scans"),
        }
        if nucleus := self._nucleus(self._text(dic["nuclei"][0])):
            values["ResonantNucleus"] = ([nucleus], "nuclei")
        if float(dic["actual_temperature"]) > 0:
            values["SampleTemperature"] = (float(dic["actual_temperature"]), "actual_temperature")
        for key, field in (("Solvent", "lock_solvent"), ("SequenceName", "sequence")):
            if text := self._text(dic[field]):
                values[key] = (text, field)
        return values

    def _nucleus(self, text: str) -> str | None:
        """A nucleus as NIfTI-MRS spells it (7Li) from TNMR's Li7 or 7Li."""
        if not (match := self.NUCLEUS.fullmatch(text)):
            return None
        return match.group(2) + match.group(1) if match.group(1) else match.group(3) + \
            match.group(4)

    @staticmethod
    def _text(value: bytes) -> str:
        """A fixed-length text field up to its first NUL (the rest is unused buffer)."""
        return bytes(value).split(b"\0")[0].decode("latin-1").strip()

    def _vendor(self, dic: dict) -> dict[str, Any]:
        vendor = {}
        for key, value in dic.items():
            if isinstance(value, bytes):
                vendor[key] = self._text(value)
            elif isinstance(value, np.ndarray) and value.dtype.kind == "S":
                vendor[key] = [self._text(v) for v in value]
            else:
                vendor[key] = self.plain(value)
        return vendor
