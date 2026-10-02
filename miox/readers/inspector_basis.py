####################################################################################################
#                                        inspector_basis.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for INSPECTOR basis sets (.mat with an lcmBasis struct): data holds one cell per #
#          metabolite (name, two numbers, FID, a text), with sw_h (Hz), sf (MHz) and ppmCalib. The #
#          vendor header keeps the whole struct, FIDs summarised.                                  #
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
#                                       InspectorBasisReader                                       #
#**************************************************************************************************#
#                                                                                                  #
# INSPECTOR stores FIDs as the conjugate of NIfTI-MRS: checked against the LCModel basis holding   #
# the same spectra. The format stores no nucleus, so 1H is assumed and noted.                      #
#                                                                                                  #
#**************************************************************************************************#
class InspectorBasisReader(Reader):
    FORMATS = (Format.INSPECTOR_BASIS,)
    PACKAGES = Reader.PACKAGES + ("scipy", "h5py")
    SOURCE = "INSPECTOR lcmBasis struct"
    NUCLEUS = "1H"
    NAME, FID = 0, 3  # positions in a metabolite's cell

    def read(self, sniff: Sniff) -> list[MRSData]:
        values = MatFile.load(sniff.path)
        basis = values.get("lcmBasis")
        if not isinstance(basis, dict) or not basis.get("data"):
            raise ReaderError(f"{sniff.path.name}: lcmBasis holds no data cells")
        cells = basis["data"]
        cells = [cells] if isinstance(cells[0], str) else cells  # a single metabolite
        names = [str(cell[self.NAME]) for cell in cells]
        fids = np.stack([np.conj(np.ravel(cell[self.FID])) for cell in cells], axis=-1)
        keys = {
            "SpectrometerFrequency": ([float(basis["sf"])], "lcmBasis/sf"),
            "ResonantNucleus": ([self.NUCLEUS], None),
        }
        notes = [
            f"nucleus not stored in INSPECTOR basis sets; {self.NUCLEUS} assumed",
            "stored as the conjugate; conjugated",
        ]
        dwelltime = 1 / float(basis["sw_h"])
        vendor = MatFile.header(values)
        d = self.basis_set(
            sniff, fids, names, dwelltime, keys, vendor, sniff.path.stem, notes, self.SOURCE
        )
        d.meta.set("ResonantNucleus", [self.NUCLEUS], f"assumed: {self.NUCLEUS}")
        d.meta.set("DwellTime", dwelltime, f"{self.SOURCE}: 1 / lcmBasis/sw_h", "lcmBasis/sw_h")
        return [d]
