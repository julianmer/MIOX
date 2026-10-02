####################################################################################################
#                                          tarquin_dpt.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for TARQUIN "dangerplot" text files (.dpt): tab-separated header lines, then the #
#          FID as real and imaginary columns. The vendor header keeps every header line.           #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import numpy as np

# own
from miox.data import MRSData
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                         TarquinDptReader                                         #
#**************************************************************************************************#
#                                                                                                  #
# Reads one .dpt file. The format stores no nucleus, so 1H is assumed and noted. The FID is in the #
# NIfTI-MRS sign convention as written (checked against the SPAR/SDAT files of the same scans).    #
#                                                                                                  #
#**************************************************************************************************#
class TarquinDptReader(Reader):
    FORMATS = (Format.TARQUIN_DPT,)
    SOURCE = "TARQUIN dangerplot header"
    NUCLEUS = "1H"
    DATA_START = "Real_FID"

    def read(self, sniff: Sniff) -> list[MRSData]:
        header, rows = {}, []
        with open(sniff.path, errors="replace") as fh:
            for line in fh:
                if line.startswith(self.DATA_START):
                    break
                key, _, value = line.strip().partition("\t")
                header[key] = self._number(value.strip())
            rows = np.loadtxt(fh, ndmin=2)
        if "Sampling_frequency" not in header or "Transmitter_frequency" not in header:
            raise ReaderError(f"{sniff.path.name} has no sampling or transmitter frequency")
        fid = (rows[:, 0] + 1j * rows[:, 1]).reshape(1, 1, 1, -1)
        values = {
            "SpectrometerFrequency": (
                [header["Transmitter_frequency"] / 1e6],
                "Transmitter_frequency",
            ),
            "ResonantNucleus": ([self.NUCLEUS], None),
        }
        if "Echo_time" in header:
            values["EchoTime"] = (header["Echo_time"], "Echo_time")
        notes = [f"nucleus not stored in .dpt files; {self.NUCLEUS} assumed"]
        dwelltime = 1 / header["Sampling_frequency"]
        d = self.dataset(
            sniff, fid, dwelltime, values, header, sniff.path.stem, notes, self.SOURCE
        )
        d.meta.set("ResonantNucleus", [self.NUCLEUS], f"assumed: {self.NUCLEUS}")
        source = f"{self.SOURCE}: 1 / Sampling_frequency"
        d.meta.set("DwellTime", dwelltime, source, "Sampling_frequency")
        return [d]

    @staticmethod
    def _number(text: str) -> float | str:
        try:
            return float(text)
        except ValueError:
            return text
