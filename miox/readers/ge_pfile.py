####################################################################################################
#                                           ge_pfile.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for GE P-files (.7). Spectroscopy sequences with a known layout are read as      #
#          MRS, every other P-file as raw k-space. The vendor header holds the named fields of the #
#          rdb header (GeRdbHeader).                                                               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

from pathlib import Path

from spec2nii.GE.ge_read_pfile import Pfile

# own
from miox.data import MRSData, RawData
from miox.readers.base import Reader
from miox.readers.ge_pfile_mrs import GePfileLayout, GePfileMrs
from miox.readers.ge_pfile_raw import GePfileRaw
from miox.readers.ge_rdb import GeRdbHeader
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                             GePfile                                              #
#**************************************************************************************************#
#                                                                                                  #
# spec2nii's Pfile as the rdb header parser only: the data are laid out by GePfileMrs. The check   #
# for revisions below 12 is fixed (the logo field is bytes; spec2nii looks for the text "GE" in    #
# it, which raises); revisions before 11 have no data-offset field, so it is derived (the data     #
# fill the end of the file).                                                                       #
#                                                                                                  #
#**************************************************************************************************#
class GePfile(Pfile):
    @property
    def is_ge_file(self) -> bool:
        if self.version < 12:
            return b"GE" in bytes(self.hdr.rhr_rh_logo)
        return Pfile.is_ge_file.fget(self)

    def map_data(self) -> None:
        if not hasattr(self.hdr, "rhr_rdb_hdr_off_data"):
            self.hdr.rhr_rdb_hdr_off_data = self._data_offset()

    def _data_offset(self) -> int:
        layout, hdr = GePfileLayout(self.hdr, self.version), self.hdr
        spectrum = hdr.rhr_rh_frame_size * 2 * hdr.rhr_rh_point_size
        per_coil = layout.kspace_points * spectrum * layout.time_points
        per_coil += int(layout.dummy_scans * spectrum / layout.coils)
        return Path(self.file_name).stat().st_size - per_coil * layout.coils


#**************************************************************************************************#
#                                          GePfileReader                                           #
#**************************************************************************************************#
#                                                                                                  #
# Reads one P-file. Spectroscopy sequences with a known layout (GePfileMrs) give one MRSData per   #
# dataset the sequence holds (metabolites, _ref, _edited, ...); any other P-file is raw k-space    #
# (GePfileRaw). Keys the rdb header's own reading adds (study, series, station, scan duration,     #
# patient, stated field strength) are set on every dataset.                                        #
#                                                                                                  #
#**************************************************************************************************#
class GePfileReader(Reader):
    FORMATS = (Format.GE_PFILE,)
    SOURCE = "P-file header"
    STATED = ("FieldStrength",)  # stated in the header: replaces the one derived from frequency
    RDB_KEYS = (  # keys of the rdb header's own reading
        "StudyID", "StudyDescription", "StationName", "SeriesDescription", "SeriesNumber",
        "PulseSequenceName", "ScanDuration", "AcquisitionDateTime", "PatientID", "PatientAge",
        "PatientWeight", "PatientPosition", "FieldStrength",
    )

    def read(self, sniff: Sniff) -> list[MRSData | RawData]:
        header = GeRdbHeader.from_file(sniff.path)
        if not header.spectroscopy_mapper():
            return GePfileRaw(sniff.path, header, self.provenance(sniff)).datasets()
        pfile = GePfile(str(sniff.path))
        mrs = GePfileMrs(sniff.path, pfile.hdr, pfile.version, pfile.endian)
        notes: list[str] = []
        with self.captured(notes):
            spectra, affine = mrs.spectra()
        rdb = header.meta()
        datasets = []
        for s in spectra:
            spectrum_notes = notes + s.notes
            data, tags = self.without_zero_indices(s.data, s.tags, spectrum_notes)
            d = self.dataset(sniff, data, mrs.dwelltime, s.values, header.values,
                             sniff.path.stem + s.suffix, spectrum_notes, self.SOURCE, affine, tags)
            for key in self.RDB_KEYS:
                if key in rdb and (key not in d.meta or key in self.STATED):
                    d.meta.set(key, rdb[key], rdb.source[key], rdb.fields.get(key))
            datasets.append(d)
        return datasets
