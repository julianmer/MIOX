####################################################################################################
#                                         profit_basis.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for ProFit basis sets stored as a struct (.mat with a basis struct): fid and met #
#          cells, bw (Hz), dt (s), f0 (Hz), pts and Bo. The vendor header keeps every variable,    #
#          FIDs summarised.                                                                        #
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
# same spectra. ProFit's own basis_set class objects are refused: their properties are not         #
# decoded. The format stores no nucleus, so 1H is assumed and noted.                               #
#                                                                                                  #
#**************************************************************************************************#
class ProfitBasisReader(Reader):
    FORMATS = (Format.PROFIT_BASIS,)
    PACKAGES = Reader.PACKAGES + ("scipy", "h5py")
    SOURCE = "ProFit basis struct"
    NUCLEUS = "1H"

    def read(self, sniff: Sniff) -> list[MRSData]:
        values = MatFile.load(sniff.path)
        basis = values.get("basis")
        if not isinstance(basis, dict):
            raise ReaderError(
                f"{sniff.path.name}: basis is a MATLAB class object, which is not decoded"
            )
        if "fid" not in basis or "met" not in basis:
            raise ReaderError(f"{sniff.path.name}: basis holds no fid and met cells")
        names = [basis["met"]] if isinstance(basis["met"], str) else list(basis["met"])
        fids = [basis["fid"]] if len(names) == 1 else basis["fid"]
        fids = np.stack([np.conj(np.ravel(fid)) for fid in fids], axis=-1)
        keys = {
            "SpectrometerFrequency": ([float(basis["f0"]) / 1e6], "basis/f0"),
            "ResonantNucleus": ([self.NUCLEUS], None),
        }
        notes = [
            f"nucleus not stored in ProFit basis structs; {self.NUCLEUS} assumed",
            "stored as the conjugate; conjugated",
        ]
        dwelltime = float(basis["dt"])
        vendor = MatFile.header(values)
        d = self.basis_set(
            sniff, fids, names, dwelltime, keys, vendor, sniff.path.stem, notes, self.SOURCE
        )
        d.meta.set("ResonantNucleus", [self.NUCLEUS], f"assumed: {self.NUCLEUS}")
        d.meta.set("DwellTime", dwelltime, f"{self.SOURCE}: basis/dt", "basis/dt")
        return [d]
