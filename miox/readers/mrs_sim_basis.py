####################################################################################################
#                                         mrs_sim_basis.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 30/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for MRS-Sim basis sets (.mat with header, metabolites and artifacts structs).    #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import numpy as np
from mrs_tools.constants import GYRO_MAG_RATIO

# own
from miox.data import BasisSet
from miox.matfile import MatFile
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                       MrsSimBasisReader                                          #
#**************************************************************************************************#
#                                                                                                  #
# One FID per metabolite, its real and imaginary parts as two rows, stored as the conjugate of     #
# NIfTI-MRS (checked: conjugated, NAA, Cr and water lie at their shifts). The header gives the     #
# carrier frequency (MHz), dwell time, TE (ms) and centerFreq, the ppm at 0 Hz; the nucleus is not #
# stored and is the one whose gyromagnetic ratio is carrier_frequency / B0. Each metabolite's      #
# concentration range (min, max) and the artifacts' ranges stay in the vendor header, as do the    #
# header's notes (the FIDs are scaled to [0, 1] in the frequency domain).                          #
#                                                                                                  #
#**************************************************************************************************#
class MrsSimBasisReader(Reader):
    FORMATS = (Format.MRS_SIM_BASIS,)
    PACKAGES = Reader.PACKAGES + ("mat-io",)
    SOURCE = "MRS-Sim basis header"

    def read(self, sniff: Sniff) -> list[BasisSet]:
        values = MatFile.load(sniff.path)
        header, metabolites = values.get("header"), values.get("metabolites")
        if not isinstance(header, dict) or not isinstance(metabolites, dict):
            raise ReaderError(f"{sniff.path.name}: no header and metabolites structs")
        names = [n for n, m in metabolites.items() if isinstance(m, dict) and "fid" in m]
        if not names:
            raise ReaderError(f"{sniff.path.name}: no metabolite FIDs")
        parts = [np.asarray(metabolites[n]["fid"], dtype=float) for n in names]
        fids = np.conj(np.stack([p[0] + 1j * p[1] for p in parts], axis=1))
        frequency = float(header["carrier_frequency"])
        notes = ["stored as the conjugate; conjugated"]
        nucleus = self._nucleus(frequency, header.get("B0"))
        if nucleus is None:
            raise ReaderError(f"{sniff.path.name}: nucleus not stored and not derivable")
        notes.append(f"nucleus not stored; {nucleus} from carrier_frequency / B0")
        keys = {
            "SpectrometerFrequency": ([frequency], "header/carrier_frequency"),
            "ResonantNucleus": ([nucleus], None),
            "ReferencePPM": (float(header["centerFreq"]), "header/centerFreq"),
            "EchoTime": (float(header["TE"]) / 1e3, "header/TE"),
        }
        vendor = {
            "header": {k: self.plain(v) for k, v in header.items() if k not in ("ppm", "t")},
            "ranges": {n: {k: self.plain(v) for k, v in metabolites[n].items() if k != "fid"}
                       for n in names},
            "artifacts": self.plain(values.get("artifacts", {})),
        }
        dwelltime = float(header["dwelltime"])
        d = self.basis_set(sniff, fids, names, dwelltime, keys, vendor, sniff.path.stem, notes,
                           self.SOURCE)
        d.meta.set("ResonantNucleus", [nucleus], "derived: carrier_frequency / B0")
        d.meta.set("DwellTime", dwelltime, f"{self.SOURCE}: header/dwelltime", "header/dwelltime")
        return [d]

    @staticmethod
    def _nucleus(frequency: float, field: object) -> str | None:
        """The nucleus whose gyromagnetic ratio (MHz/T) is frequency / field, within 1 %."""
        if not isinstance(field, (int, float, np.number)) or not field:
            return None
        ratio = frequency / float(field)
        return next((n for n, g in GYRO_MAG_RATIO.items() if abs(ratio - g) < 1e-2 * g), None)
