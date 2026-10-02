####################################################################################################
#                                         fsl_mrs_basis.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for one FSL-MRS basis spectrum (.json): the FID as basis_re and basis_im, with   #
#          basis_dwell (s), basis_centre (MHz), basis_width and basis_name. The vendor header      #
#          keeps every key except the two arrays.                                                  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import json

import numpy as np

# own
from miox.data import MRSData
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                        FslMrsBasisReader                                         #
#**************************************************************************************************#
#                                                                                                  #
# FSL-MRS stores the FID in its own convention, the conjugate of NIfTI-MRS: checked against the    #
# LCModel basis the same spectra were converted from. One file is a BasisSet of one metabolite; a  #
# folder of them is read by the basis-folder reader. The format stores no nucleus, so 1H is        #
# assumed and noted.                                                                               #
#                                                                                                  #
#**************************************************************************************************#
class FslMrsBasisReader(Reader):
    FORMATS = (Format.FSL_MRS_BASIS,)
    SOURCE = "FSL-MRS basis JSON"
    NUCLEUS = "1H"
    ARRAYS = ("basis_re", "basis_im")

    def read(self, sniff: Sniff) -> list[MRSData]:
        content = json.loads(sniff.path.read_text())
        basis = content.get("basis")
        if not isinstance(basis, dict) or any(key not in basis for key in self.ARRAYS):
            raise ReaderError(f"{sniff.path.name}: no basis_re and basis_im arrays")
        fid = np.conj(np.asarray(basis["basis_re"]) + 1j * np.asarray(basis["basis_im"]))
        vendor = {key: value for key, value in content.items() if key != "basis"}
        vendor["basis"] = {key: value for key, value in basis.items() if key not in self.ARRAYS}
        name = str(basis.get("basis_name") or sniff.path.stem)
        keys = {
            "SpectrometerFrequency": ([float(basis["basis_centre"])], "basis/basis_centre"),
            "ResonantNucleus": ([self.NUCLEUS], None),
        }
        notes = [
            f"nucleus not stored in FSL-MRS basis files; {self.NUCLEUS} assumed",
            "stored as the conjugate (FSL-MRS convention); conjugated",
        ]
        dwelltime = float(basis["basis_dwell"])
        d = self.basis_set(
            sniff, fid[:, None], [name], dwelltime, keys, vendor, name, notes, self.SOURCE
        )
        d.meta.set("ResonantNucleus", [self.NUCLEUS], f"assumed: {self.NUCLEUS}")
        d.meta.set("DwellTime", dwelltime, f"{self.SOURCE}: basis/basis_dwell", "basis/basis_dwell")
        return [d]
