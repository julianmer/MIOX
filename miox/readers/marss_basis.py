####################################################################################################
#                                          marss_basis.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for MARSS basis spectra (.mat with an exptDat struct): one metabolite per file,  #
#          with sw_h (Hz), sf (MHz), nspecC, ppmCalib and fid. The vendor header keeps the struct, #
#          the FID summarised.                                                                     #
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
#                                         MarssBasisReader                                         #
#**************************************************************************************************#
#                                                                                                  #
# MARSS stores the FID as the conjugate of NIfTI-MRS: checked against the LCModel basis holding    #
# the same spectra. One file is a BasisSet of one metabolite, named after the file; a folder of    #
# them is read by the basis-folder reader. The format stores no nucleus, so 1H is assumed.         #
#                                                                                                  #
#**************************************************************************************************#
class MarssBasisReader(Reader):
    FORMATS = (Format.MARSS_BASIS,)
    PACKAGES = Reader.PACKAGES + ("scipy", "h5py")
    SOURCE = "MARSS exptDat struct"
    NUCLEUS = "1H"

    def read(self, sniff: Sniff) -> list[MRSData]:
        values = MatFile.load(sniff.path)
        expt = values.get("exptDat")
        if not isinstance(expt, dict) or "fid" not in expt:
            raise ReaderError(f"{sniff.path.name}: exptDat holds no fid")
        fid = np.conj(np.ravel(expt["fid"]))
        keys = {
            "SpectrometerFrequency": ([float(expt["sf"])], "exptDat/sf"),
            "ResonantNucleus": ([self.NUCLEUS], None),
        }
        notes = [
            f"nucleus not stored in MARSS basis spectra; {self.NUCLEUS} assumed",
            "stored as the conjugate; conjugated",
        ]
        dwelltime = 1 / float(expt["sw_h"])
        name = sniff.path.stem
        vendor = MatFile.header(values)
        d = self.basis_set(
            sniff, fid[:, None], [name], dwelltime, keys, vendor, name, notes, self.SOURCE
        )
        d.meta.set("ResonantNucleus", [self.NUCLEUS], f"assumed: {self.NUCLEUS}")
        d.meta.set("DwellTime", dwelltime, f"{self.SOURCE}: 1 / exptDat/sw_h", "exptDat/sw_h")
        return [d]
