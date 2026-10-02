####################################################################################################
#                                          lcmodel_raw.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for LCModel .RAW files. spec2nii converts; the vendor header keeps every         #
#          namelist ($SEQPAR, $NMID, ...) with its values as written, which spec2nii reduces to    #
#          frequency and dwell time.                                                               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re
import warnings
from argparse import Namespace
from pathlib import Path

from nifti_mrs.nifti_mrs import NIFTI_MRS
from spec2nii.other_formats import lcm_raw

# own
from miox.readers.base import Spec2niiReader
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                         LcmodelRawReader                                         #
#**************************************************************************************************#
#                                                                                                  #
# Reads one .RAW file. The format does not store the nucleus, so 1H is assumed and noted.          #
# Namelist group and variable names are upper-cased (Fortran namelists ignore case).               #
#                                                                                                  #
#**************************************************************************************************#
class LcmodelRawReader(Spec2niiReader):
    FORMATS = (Format.LCMODEL_RAW,)
    CONVERTER = "raw"
    NUCLEUS = "1H"
    NAMELIST = re.compile(r"[$&](\w+)(.*?)[$&]END", re.S | re.I)
    PAIR = re.compile(r"(\w+)\s*=\s*('[^']*'|\"[^\"]*\"|[^,\s]+)")
    SOURCES = {
        "SpectrometerFrequency": "SEQPAR/HZPPPM",
        "DwellTime": ("SEQPAR/DWELLTIME", "NMID/DELTAT", "NMID/BADELT"),
    }

    def convert(self, sniff: Sniff) -> tuple[list[NIFTI_MRS], list[str], dict]:
        args = Namespace(
            file=str(sniff.path),
            nucleus=self.NUCLEUS,
            imagingfreq=None,
            bandwidth=None,
            affine=None,
            fileout=None,
        )
        images, names = lcm_raw(args)
        message = f"nucleus not stored in LCModel .RAW files; {self.NUCLEUS} assumed"
        warnings.warn(message, stacklevel=2)
        return list(images), list(names), self._namelists(sniff.path)

    def _namelists(self, path: Path) -> dict[str, dict[str, str]]:
        text = path.read_text(errors="replace")
        return {
            group.upper(): {key.upper(): value for key, value in self.PAIR.findall(body)}
            for group, body in self.NAMELIST.findall(text)
        }
