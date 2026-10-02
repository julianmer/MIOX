####################################################################################################
#                                         profit_basis.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for ProFit basis sets (.mat with a basis variable: ProFit's basis_set object or  #
#          a struct): fid and met cells, f0 (Hz), dt (s), bw (Hz), pts; the object holds dt, bw    #
#          and pts as (indirect, direct). The vendor header keeps every variable, FIDs summarised. #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import numpy as np

# own
from miox.data import MRSData
from miox.matfile import MatFile
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                        ProfitBasisReader                                         #
#**************************************************************************************************#
#                                                                                                  #
# The FIDs are stored as the conjugate of NIfTI-MRS: checked against the LCModel basis holding the #
# same spectra, and by peak positions in ProFit's own sets (0 Hz at 4.70 ppm there, not stored).   #
# Sets with more than one indirect point (J-resolved) are refused until one can be checked. The    #
# format stores no nucleus, so 1H is assumed and noted.                                            #
#                                                                                                  #
#**************************************************************************************************#
class ProfitBasisReader(Reader):
    FORMATS = (Format.PROFIT_BASIS,)
    PACKAGES = Reader.PACKAGES + ("mat-io",)
    SOURCE = "ProFit basis"
    NUCLEUS = "1H"

    def read(self, sniff: Sniff) -> list[MRSData]:
        values = MatFile.load(sniff.path)
        basis = values.get("basis")
        if not isinstance(basis, dict) or "fid" not in basis or "met" not in basis:
            raise ReaderError(f"{sniff.path.name}: basis holds no fid and met cells")
        dt, points = np.ravel(basis["dt"]), np.ravel(basis.get("pts", 0))
        if points.size == 2 and points[0] > 1:
            raise ReaderError(f"{sniff.path.name}: J-resolved basis ({points[0]} indirect points)")
        names = [basis["met"]] if isinstance(basis["met"], str) else list(basis["met"])
        fids = [basis["fid"]] if len(names) == 1 else basis["fid"]
        fids = np.stack([np.conj(np.ravel(fid)) for fid in fids], axis=-1)
        keys = {
            "SpectrometerFrequency": ([float(basis["f0"]) / 1e6], "basis/f0"),
            "ResonantNucleus": ([self.NUCLEUS], None),
        }
        notes = [
            f"nucleus not stored in ProFit basis sets; {self.NUCLEUS} assumed",
            "stored as the conjugate; conjugated",
        ]
        dwelltime = float(dt[-1])
        vendor = MatFile.header(values)
        d = self.basis_set(
            sniff, fids, names, dwelltime, keys, vendor, sniff.path.stem, notes, self.SOURCE
        )
        d.meta.set("ResonantNucleus", [self.NUCLEUS], f"assumed: {self.NUCLEUS}")
        d.meta.set("DwellTime", dwelltime, f"{self.SOURCE}: basis/dt (direct)", "basis/dt")
        return [d]
