####################################################################################################
#                                           ge_pfile.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for GE P-files (.7). spec2nii converts; the vendor header holds every named      #
#          field of spec2nii's P-file header structure (a subset of the full rdb header, padding   #
#          skipped).                                                                               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import ctypes
from pathlib import Path

from nifti_mrs.nifti_mrs import NIFTI_MRS
from spec2nii.GE.ge_pfile import read_pfile
from spec2nii.GE.ge_read_pfile import Pfile

# own
from miox.readers.base import Spec2niiReader
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                           PfileHeader                                            #
#**************************************************************************************************#
#                                                                                                  #
# spec2nii's Pfile without the data: its constructor would also map the whole file.                #
#                                                                                                  #
#**************************************************************************************************#
class PfileHeader(Pfile):
    def __init__(self, fname: str):
        self.file_name = fname
        self.version = 0
        self.hdr = None
        self.map = None
        self.endian = "little"
        self.encoding = None
        self.read_header()


#**************************************************************************************************#
#                                          GePfileReader                                           #
#**************************************************************************************************#
#                                                                                                  #
# Reads one P-file; a file with water references or several sub-experiments gives one MRSData      #
# per output spec2nii makes (``_ref``, ``_edited``, ...).                                          #
#                                                                                                  #
#**************************************************************************************************#
class GePfileReader(Spec2niiReader):
    FORMATS = (Format.GE_PFILE,)
    CONVERTER = "ge"
    SOURCES = {
        "SpectrometerFrequency": "rhr_rh_ps_mps_freq",
        "EchoTime": "rhi_te",
        "RepetitionTime": "rhi_tr",
        "InversionTime": "rhi_ti",
        "ExcitationFlipAngle": "rhi_mr_flip",
        "ManufacturersModelName": "rhe_ex_sysid",
        "DeviceSerialNumber": "rhe_uniq_sys_id",
        "SoftwareVersions": "rhe_ex_verscre",
        "InstitutionName": "rhe_hospname",
        "RxCoil": "rhi_cname",
        "SequenceName": "rhi_psdname",
        "ProtocolName": "rhs_se_desc",
        "PatientName": "rhe_patname",
        "PatientDoB": "rhe_dateofbirth",
        "PatientSex": "rhe_patsex",
        "DwellTime": "rhr_spectral_width",
    }

    def convert(self, sniff: Sniff) -> tuple[list[NIFTI_MRS], list[str], dict]:
        images, names = read_pfile(sniff.path, None)
        return list(images), list(names), self._header(sniff.path)

    def _header(self, path: Path) -> dict:
        pfile = PfileHeader(str(path))
        header = {"version": pfile.version}
        for info in pfile._dump_struct(pfile.hdr):
            if info.label.startswith("pad"):
                continue
            value = info.value
            if isinstance(value, ctypes.Array):
                value = list(value)
            if isinstance(value, bytes) and pfile.encoding:
                value = value.decode(pfile.encoding, "replace").rstrip("\0")
            header[info.label] = self.plain(value)
        return header
