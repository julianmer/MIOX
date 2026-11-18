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

# own
from miox.data import MRSData, RawData
from miox.readers.base import Reader
from miox.readers.ge_pfile_mrs import GePfileMrs
from miox.readers.ge_pfile_raw import GePfileRaw
from miox.readers.ge_rdb import GeRdbHeader
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                          GePfileReader                                           #
#**************************************************************************************************#
#                                                                                                  #
# Reads one P-file. Spectroscopy sequences with a known layout (GePfileMrs) give one MRSData per   #
# dataset the sequence holds (metabolites, _ref, _edited, ...); any other P-file is raw k-space    #
# (GePfileRaw). Keys the rdb header's reading adds (study, series, station, scan duration,         #
# patient, stated field strength) are set on every dataset.                                        #
#                                                                                                  #
#**************************************************************************************************#
class GePfileReader(Reader):
    FORMATS = (Format.GE_PFILE,)
    SOURCE = "P-file header"
    STATED = ("FieldStrength",)  # stated in the header: replaces the one derived from frequency
    RDB_KEYS = (  # keys of the rdb header's reading
        "StudyID", "StudyDescription", "StationName", "SeriesDescription", "SeriesNumber",
        "PulseSequenceName", "ScanDuration", "AcquisitionDateTime", "PatientID", "PatientAge",
        "PatientWeight", "PatientPosition", "FieldStrength",
    )

    def read(self, sniff: Sniff) -> list[MRSData | RawData]:
        header = GeRdbHeader.from_file(sniff.path)
        if not GePfileMrs.recognises(header.hdr):
            return GePfileRaw(sniff.path, header, self.provenance(sniff)).datasets()
        mrs = GePfileMrs(sniff.path, header.hdr, header.version, header.endian)
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
