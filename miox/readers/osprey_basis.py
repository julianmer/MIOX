####################################################################################################
#                                         osprey_basis.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for Osprey basis sets (.mat with a BASIS struct, also written by MRSCloud): fids #
#          as (points, metabolites) or (points, metabolites, sub-spectra), names, Bo, te. The      #
#          vendor header keeps the whole struct, large arrays summarised.                          #
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
#                                        OspreyBasisReader                                         #
#**************************************************************************************************#
#                                                                                                  #
# Osprey stores FIDs as the conjugate of NIfTI-MRS and no frequency: it is Bo times Osprey's       #
# gyromagnetic ratio (42.577 MHz/T), which equals the txfrq of MRSCloud's single-metabolite files. #
# 0 Hz sits at centerFreq ppm (kept in the vendor header): with both, NAA is at 2.008 ppm in every #
# Osprey set tested. The files do not name edited sub-spectra; they stay a dimension as stored.    #
#                                                                                                  #
#**************************************************************************************************#
class OspreyBasisReader(Reader):
    FORMATS = (Format.OSPREY_BASIS,)
    PACKAGES = Reader.PACKAGES + ("scipy", "h5py")
    SOURCE = "Osprey BASIS struct"
    GAMMA = 42.577  # MHz/T
    NUCLEUS = "1H"
    SUB_SPECTRA = ("DIM_USER_1", "sub-spectra as stored; the file does not name them", None)

    def read(self, sniff: Sniff) -> list[MRSData]:
        values = MatFile.load(sniff.path)
        basis = values.get("BASIS")
        if not isinstance(basis, dict) or "fids" not in basis or "name" not in basis:
            raise ReaderError(f"{sniff.path.name}: BASIS is not an Osprey basis struct")
        names = [basis["name"]] if isinstance(basis["name"], str) else list(basis["name"])
        fids = np.conj(np.asarray(basis["fids"]))
        fids = fids.reshape(fids.shape[0], len(names), -1)
        tags = (self.SUB_SPECTRA,) if fids.shape[2] > 1 else ()
        fids = fids if tags else fids[:, :, 0]
        keys = {
            "SpectrometerFrequency": ([float(basis["Bo"]) * self.GAMMA], "BASIS/Bo"),
            "ResonantNucleus": ([self.NUCLEUS], None),
        }
        if float(basis.get("te", 0) or 0) > 0:
            keys["EchoTime"] = (float(basis["te"]) * 1e-3, "BASIS/te")
        notes = [
            f"nucleus not stored in Osprey basis sets; {self.NUCLEUS} assumed",
            "stored as the conjugate; conjugated",
            f"0 Hz is at {float(basis.get('centerFreq', np.nan))} ppm (BASIS/centerFreq)",
        ]
        dwelltime = float(basis["dwelltime"])
        vendor = MatFile.header(values)
        d = self.basis_set(
            sniff, fids, names, dwelltime, keys, vendor, sniff.path.stem, notes, self.SOURCE, tags
        )
        source = f"derived: BASIS/Bo x {self.GAMMA} MHz/T"
        d.meta.set("SpectrometerFrequency", keys["SpectrometerFrequency"][0], source, "BASIS/Bo")
        d.meta.set("ResonantNucleus", [self.NUCLEUS], f"assumed: {self.NUCLEUS}")
        d.meta.set("DwellTime", dwelltime, f"{self.SOURCE}: BASIS/dwelltime", "BASIS/dwelltime")
        return [d]
