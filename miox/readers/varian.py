####################################################################################################
#                                            varian.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for Varian/Agilent spectroscopy (.fid directory with fid and procpar). spec2nii  #
#          converts; the vendor header keeps every procpar parameter, the text file and the log.   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from nifti_mrs.nifti_mrs import NIFTI_MRS
from spec2nii.varian import read_procpar
from spec2nii.varian_importer import read_varian

# own
from miox.data import MRSData
from miox.readers.base import Spec2niiReader
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                           VarianReader                                           #
#**************************************************************************************************#
#                                                                                                  #
# Reads one .fid directory. procpar values stay the strings written in the file; times in procpar  #
# are in s (delays, te, tr) or µs (pulses, alfa).                                                  #
#                                                                                                  #
#**************************************************************************************************#
class VarianReader(Spec2niiReader):
    FORMATS = (Format.VARIAN,)
    CONVERTER = "varian"
    SOURCES = {
        "SpectrometerFrequency": "procpar/sfrq",
        "ResonantNucleus": "procpar/tn",
        "EchoTime": ("procpar/te", "procpar/p1", "procpar/pw"),
        "RepetitionTime": "procpar/tr",
        "InversionTime": "procpar/ti",
        "ExcitationFlipAngle": "procpar/flip1",
        "ProtocolName": "procpar/seqfil",
        "PatientName": "procpar/comment",
        "SoftwareVersions": "procpar/parver",
        "TxCoil": "procpar/rfcoil",
        "RxCoil": "procpar/rfcoil",
        "DwellTime": "procpar/sw",
    }

    def convert(self, sniff: Sniff) -> tuple[list[NIFTI_MRS], list[str], dict]:
        folder = sniff.path if sniff.path.is_dir() else sniff.path.parent
        args = Namespace(file=str(folder), tag6="DIM_DYN", dump_headers=False, fileout=None)
        images, names = read_varian(args)
        return list(images), list(names), self._header(folder)

    def correct(self, d: MRSData) -> None:
        """spec2nii overwrites te with a pulse time plus alfa, both in µs, stored as s."""
        super().correct(d)
        procpar = d.meta.vendor["procpar"]
        if "te" in procpar:
            te = float(procpar["te"][0])
            d.meta.set("EchoTime", te, "procpar/te (s)", "procpar/te")
        elif "alfa" in procpar and "EchoTime" in d.meta:
            pulse = "p1" if "p1" in procpar else "pw"
            delay = (float(procpar[pulse][0]) + float(procpar["alfa"][0])) * 1e-6
            d.meta.set(
                "EchoTime",
                delay,
                f"procpar/{pulse} + procpar/alfa in µs, converted to s "
                "(acquisition delay of a pulse-acquire sequence)",
                f"procpar/{pulse}",
            )

    def _header(self, folder: Path) -> dict:
        procpar = read_procpar(str(folder / "procpar"))
        header = {"procpar": {name: p["values"] for name, p in procpar.items()}}
        for name in ("text", "log"):
            if (folder / name).is_file():
                header[name] = (folder / name).read_text(errors="replace").splitlines()
        return header
