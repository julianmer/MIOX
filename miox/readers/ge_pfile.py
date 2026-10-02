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

import contextlib
import ctypes
import warnings
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from nifti_mrs.nifti_mrs import NIFTI_MRS
from spec2nii.GE import ge_pfile
from spec2nii.GE.ge_pfile import _process_mrsi_pfile, _process_svs_pfile
from spec2nii.GE.ge_read_pfile import Pfile

# own
from miox.data import MRSData
from miox.readers.base import Spec2niiReader
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                             GePfile                                              #
#**************************************************************************************************#
#                                                                                                  #
# spec2nii's Pfile with the check for revisions below 12 fixed: the logo field is bytes, and       #
# spec2nii looks for the text "GE" in it, which raises. HBCD PRESS (psd "hbcd", protocol with      #
# "press") is mapped as spec2nii's "mrs-press" whatever the protocol is called; spec2nii maps it   #
# only when the protocol is named exactly that.                                                    #
#                                                                                                  #
#**************************************************************************************************#
class GePfile(Pfile):
    HBCD_PRESS = b"mrs-press"  # spec2nii's name for HBCD PRESS

    @property
    def is_ge_file(self) -> bool:
        if self.version < 12:
            return b"GE" in bytes(self.hdr.rhr_rh_logo)
        return Pfile.is_ge_file.fget(self)

    def map_data(self) -> None:
        """spec2nii's map_data; revisions before 11 have no data-offset field, so it is derived."""
        psd = bytes(self.hdr.rhi_psdname).split(b"\0")[0].lower()
        protocol = bytes(self.hdr.rhs_se_desc).split(b"\0")[0].lower()
        self.hbcd_press = psd == b"hbcd" and b"press" in protocol
        if self.hbcd_press:
            self.hdr.rhi_psdname = self.HBCD_PRESS
        self.map = self.get_mapper(self.file_name, self.hdr, self.version, self.endian)
        if not hasattr(self.hdr, "rhr_rdb_hdr_off_data"):
            self.hdr.rhr_rdb_hdr_off_data = self._data_offset()
        self.map.read_data()

    def _data_offset(self) -> int:
        """The data fill the end of the file; same byte count as spec2nii's read_data."""
        m, hdr = self.map, self.hdr
        spectrum = hdr.rhr_rh_frame_size * 2 * hdr.rhr_rh_point_size
        per_coil = m.get_num_kspace_points * spectrum * m.get_num_time_points
        per_coil += int(m.get_num_dummy_scans * spectrum / m.get_num_coils)
        return Path(self.file_name).stat().st_size - per_coil * m.get_num_coils


#**************************************************************************************************#
#                                           PfileHeader                                            #
#**************************************************************************************************#
#                                                                                                  #
# GePfile without the data: the constructor would also map the whole file.                         #
#                                                                                                  #
#**************************************************************************************************#
class PfileHeader(GePfile):
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
# per output spec2nii makes (``_ref``, ``_edited``, ...). HBCD PRESS files store user19 water      #
# frames, then user4 metabolite frames (checked by their amplitudes); spec2nii's PRESS split       #
# (16 / nex water frames) moves water frames into the metabolites, so they are split here. The     #
# sequence name comes from the header, not from the name used for mapping.                         #
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
        """spec2nii's read_pfile, with GePfile in place of its Pfile."""
        pfile = GePfile(str(sniff.path))
        process = _process_svs_pfile if pfile.is_svs else _process_mrsi_pfile
        with self._hbcd_press(pfile):
            images, suffixes = process(pfile)
        names = [sniff.path.stem + suffix for suffix in suffixes]
        return list(images), names, self._header(sniff.path)

    def correct(self, d: MRSData) -> None:
        super().correct(d)
        psdname = d.meta.vendor.get("rhi_psdname")
        if psdname and d.meta.get("SequenceName") != psdname:
            self.fix(d, "SequenceName", psdname, "spec2nii reported the name it mapped by")

    @contextlib.contextmanager
    def _hbcd_press(self, pfile: GePfile) -> Iterator[None]:
        """spec2nii's PRESS processing with the frames split by the HBCD header counts."""
        original = ge_pfile._process_probe_p
        hdr = pfile.hdr
        water, metabolite = int(hdr.rhr_rh_user19), int(hdr.rhr_rh_user4)
        frames = pfile.map.raw_data.shape[3] if pfile.hbcd_press else 0

        def split(pfile):
            raw = pfile.map.raw_data  # (1, 1, 1, frames, coils, points)
            order = [0, 1, 2, 5, 4, 3]  # to (1, 1, 1, points, coils, frames)
            images, metas, dwelltime, suffixes = original(pfile)
            ref = np.transpose(raw[:, :, :, :water], order)
            metab = np.transpose(raw[:, :, :, water:], order)
            return [metab, ref], metas, dwelltime, suffixes

        if pfile.hbcd_press and water + metabolite == frames:
            ge_pfile._process_probe_p = split
        elif pfile.hbcd_press:
            warnings.warn(f"HBCD PRESS: {water} water + {metabolite} metabolite frames do not "
                          f"make the {frames} frames; spec2nii's split kept", stacklevel=2)
        try:
            yield
        finally:
            ge_pfile._process_probe_p = original

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
