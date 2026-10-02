####################################################################################################
#                                       spinwizard_basis.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for SpinWizard (and JET) basis-set folders: one text file per metabolite with    #
#          real and imaginary columns, and BasisSetParameters.txt (frequency in MHz, spectral      #
#          width in kHz). The vendor header keeps every .txt file in the folder by line.           #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

from pathlib import Path

import numpy as np

# own
from miox.data import MRSData
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                      SpinwizardBasisReader                                       #
#**************************************************************************************************#
#                                                                                                  #
# Metabolite files are the files without an extension, named after their metabolite. Their FIDs    #
# are stored as the conjugate of NIfTI-MRS: checked against the LCModel basis the same spectra     #
# were converted from, and by peak positions. BasisRFOffset.txt (the ppm at the centre) and        #
# LCMBasisSet.txt (a selection for LCModel) stay in the vendor header. The format stores no        #
# nucleus, so 1H is assumed and noted.                                                             #
#                                                                                                  #
#**************************************************************************************************#
class SpinwizardBasisReader(Reader):
    FORMATS = (Format.SPINWIZARD_BASIS,)
    SOURCE = "SpinWizard BasisSetParameters.txt"
    PARAMETERS = "BasisSetParameters.txt"
    NUCLEUS = "1H"

    def read(self, sniff: Sniff) -> list[MRSData]:
        files = {p.name: p for p in sniff.companions}
        if self.PARAMETERS not in files:
            raise ReaderError(f"{sniff.path.name}: no {self.PARAMETERS}")
        vendor = {name: self._lines(p) for name, p in files.items() if p.suffix.lower() == ".txt"}
        members = [p for p in files.values() if not p.suffix]
        if not members:
            raise ReaderError(f"{sniff.path.name}: no metabolite files")
        columns = [np.loadtxt(p, ndmin=2) for p in members]
        lengths = {c.shape[0] for c in columns}
        if len(lengths) > 1:
            raise ReaderError(f"{sniff.path.name}: metabolite files differ in length {lengths}")
        fids = np.stack([np.conj(c[:, 0] + 1j * c[:, 1]) for c in columns], axis=-1)
        parameters = vendor[self.PARAMETERS]
        keys = {
            "SpectrometerFrequency": ([float(parameters["1"])], f"{self.PARAMETERS}/1"),
            "ResonantNucleus": ([self.NUCLEUS], None),
        }
        notes = [
            f"nucleus not stored in SpinWizard basis sets; {self.NUCLEUS} assumed",
            "stored as the conjugate; conjugated",
        ]
        dwelltime = 1 / (float(parameters["2"]) * 1e3)
        names = [p.name for p in members]
        d = self.basis_set(
            sniff, fids, names, dwelltime, keys, vendor, sniff.path.name, notes, self.SOURCE
        )
        field = f"{self.PARAMETERS}/2"
        d.meta.set("ResonantNucleus", [self.NUCLEUS], f"assumed: {self.NUCLEUS}")
        d.meta.set("DwellTime", dwelltime, f"{self.SOURCE}: 1 / spectral width (kHz)", field)
        return [d]

    @staticmethod
    def _lines(path: Path) -> dict[str, str]:
        """Non-empty lines by line number, from 1."""
        lines = path.read_text(errors="replace").splitlines()
        return {str(i): line.strip() for i, line in enumerate(lines, 1) if line.strip()}
