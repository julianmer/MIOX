####################################################################################################
#                                            rnmrtk.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 10/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for Rowland NMR Toolkit files (.sec data with its .par parameters) in the time   #
#          domain, read by nmrglue.                                                                #
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
from miox.data import MRSData
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                           RnmrtkReader                                           #
#**************************************************************************************************#
#                                                                                                  #
# The .par file names each dimension (Dom T1, T2, ...; the last array axis is the acquired one)    #
# with its spectral width (Sw, Hz), frequency (Sf, MHz) and the shift of its carrier (Ppm). The    #
# samples are kept as stored: no RNMRTK file written by RNMRTK itself is public, and the files of  #
# nmrglue's converter keep the sign of whatever they were converted from. The nucleus is not       #
# stated. Further time dimensions go on DIM_INDIRECT_0 to 2 with their rows as stored; files with  #
# a frequency dimension are not read (no sample).                                                  #
#                                                                                                  #
#**************************************************************************************************#
class RnmrtkReader(Reader):
    FORMATS = (Format.RNMRTK,)
    PACKAGES = Reader.PACKAGES + ("nmrglue",)
    INDIRECT = ("DIM_INDIRECT_0", "DIM_INDIRECT_1", "DIM_INDIRECT_2")

    def read(self, sniff: Sniff) -> list[MRSData]:
        path = Path(sniff.path).with_suffix(".sec")
        try:
            dic, data = ng.rnmrtk.read(str(path))
        except (ValueError, OSError, KeyError, IndexError) as e:
            raise ReaderError(f"{path}: not a readable RNMRTK file ({e})") from e
        if any(d != "T" for d in dic["dom"]):
            raise ReaderError(f"{path}: frequency-domain RNMRTK data (not supported: no sample)")
        samples = np.asarray(data, dtype=np.complex128)
        samples = np.transpose(samples, list(range(samples.ndim))[::-1])  # acquired first
        if samples.ndim > 4:
            raise ReaderError(f"{path}: {samples.ndim} dimensions, NIfTI-MRS holds at most 4")
        order = list(range(len(dic["dom"])))[::-1]  # dictionary dimensions, acquired first
        direct = order[0]
        tags = tuple(
            (self.INDIRECT[i], f"{dic['dom'][k]}{k + 1}: Sw {dic['sw'][k]:.6g} Hz, Sf "
             f"{dic['sf'][k]:.9g} MHz, Ppm {dic['ppm'][k]:.6g}, {dic['nptype'][k]}, quad "
             f"{dic['quad'][k]}, rows as stored", None)
            for i, k in enumerate(order[1:])
        )
        values = {
            "SpectrometerFrequency": ([float(dic["sf"][k]) for k in order], "Sf"),
            "ReferencePPM": (float(dic["ppm"][direct]), "Ppm"),
        }
        dwelltime = float(dic["xstep"][direct])
        notes = ["sign convention not verified (no file written by RNMRTK itself): samples kept "
                 "as stored", "the nucleus is not stated"]
        d = self.dataset(sniff, samples.reshape((1, 1, 1) + samples.shape), dwelltime, values,
                         self._vendor(dic), path.stem, notes, "RNMRTK .par", dim_tags=tags)
        d.meta.set("DwellTime", dwelltime, "RNMRTK .par: Xstep", "Xstep")
        return [d]

    def _vendor(self, dic: dict) -> dict[str, Any]:
        return {k: self.plain(v) for k, v in dic.items()}
