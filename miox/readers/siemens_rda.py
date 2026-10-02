####################################################################################################
#                                          siemens_rda.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for Siemens RDA spectroscopy exports. spec2nii converts; the vendor header keeps #
#          every line of the text header, which spec2nii parses but does not hand out.             #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

from pathlib import Path

from nifti_mrs.nifti_mrs import NIFTI_MRS
from spec2nii.Siemens.rda import convert_rda

# own
from miox.readers.base import Spec2niiReader
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                         SiemensRdaReader                                         #
#**************************************************************************************************#
#                                                                                                  #
# Reads one RDA file. Header values stay the strings written in the file.                          #
#                                                                                                  #
#**************************************************************************************************#
class SiemensRdaReader(Spec2niiReader):
    FORMATS = (Format.SIEMENS_RDA,)
    CONVERTER = "rda"
    HEADER_END = ">>> End of header <<<"
    SOURCES = {
        "SpectrometerFrequency": "MRFrequency",
        "ResonantNucleus": "Nucleus",
        "EchoTime": "TE",
        "RepetitionTime": "TR",
        "InversionTime": "TI",
        "MixingTime": "TM",
        "ExcitationFlipAngle": "FlipAngle",
        "ManufacturersModelName": "ModelName",
        "DeviceSerialNumber": "DeviceSerialNumber",
        "SoftwareVersions": "SoftwareVersion[0]",
        "InstitutionName": "InstitutionName",
        "TxCoil": "TransmitCoil",
        "SequenceName": "SequenceName",
        "ProtocolName": "ProtocolName",
        "PatientPosition": "PatientPosition",
        "PatientName": "PatientName",
        "PatientID": "PatientID",
        "PatientWeight": "PatientWeight",
        "PatientDoB": "PatientBirthDate",
        "PatientSex": "PatientSex",
        "DwellTime": "DwellTime",
    }

    def convert(self, sniff: Sniff) -> tuple[list[NIFTI_MRS], list[str], dict]:
        images, names = convert_rda(sniff.path, None, False)
        return list(images), list(names), self._header(sniff.path)

    def _header(self, path: Path) -> dict[str, str]:
        """Every ``key: value`` line between the header markers."""
        header = {}
        with open(path, "rb") as fh:
            next(fh)  # >>> Begin of header <<<
            for raw in fh:
                try:
                    line = raw.decode("utf-8")
                except UnicodeDecodeError:
                    line = raw.decode("latin-1")
                line = line.rstrip("\r\n")
                if line.startswith(self.HEADER_END):
                    break
                key, _, value = line.partition(": ")
                header[key] = value
        return header
