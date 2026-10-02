####################################################################################################
#                                            bruker.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for Bruker ParaVision spectroscopy scans (fid, fid_proc.64, 2dseq, rawdata).     #
#          spec2nii converts through brukerapi; the vendor header holds every parameter of every   #
#          parameter file of the scan and its processing folder.                                   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from brukerapi.jcampdx import JCAMPDX
from nifti_mrs.nifti_mrs import NIFTI_MRS
from spec2nii.bruker import read_bruker

# own
from miox.readers.base import ReaderError, Spec2niiReader
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                           BrukerReader                                           #
#**************************************************************************************************#
#                                                                                                  #
# A data file is read in its own mode; a scan folder or a parameter file reads the scan's fid,     #
# else the 2dseq or rawdata next to it. spec2nii only accepts spectroscopy scans. The frequency is #
# the working frequency (PVM_FrqWork, SFO1); its chemical shift, PVM_FrqWorkPpm, is ReferencePPM.  #
#                                                                                                  #
#**************************************************************************************************#
class BrukerReader(Spec2niiReader):
    FORMATS = (Format.BRUKER,)
    CONVERTER = "bruker"
    REFERENCE = "method/PVM_FrqWorkPpm"
    MODES = {"fid": "FID", "fid_proc.64": "FID", "2dseq": "2DSEQ", "rawdata.job0": "RAWDATA"}
    PROCESSED = ("2dseq", "fid_proc.64")  # stored in <study>/<scan>/pdata/<n>/
    PARAMETER_FILES = ("acqp", "method", "visu_pars", "reco", "subject")
    SOURCES_FID = {
        "SpectrometerFrequency": "acqp/BF1",
        "ResonantNucleus": "acqp/NUCLEUS",
        "EchoTime": "method/PVM_EchoTime",
        "RepetitionTime": "method/PVM_RepetitionTime",
        "TxOffset": "method/PVM_FrqWorkOffsetPpm",
        "SoftwareVersions": "acqp/ACQ_sw_version",
        "SequenceName": "method/Method",
        "PatientName": "subject/SUBJECT_id",
        "DwellTime": "acqp/SW_h",
    }
    SOURCES_2DSEQ = {
        "SpectrometerFrequency": "visu_pars/VisuAcqImagingFrequency",
        "ResonantNucleus": "visu_pars/VisuAcqImagedNucleus",
        "EchoTime": ("visu_pars/VisuAcqEchoTime", "method/PVM_EchoTime"),
        "RepetitionTime": ("visu_pars/VisuAcqRepetitionTime", "method/PVM_RepetitionTime"),
        "TxOffset": "method/PVM_FrqWorkOffsetPpm",
        "SoftwareVersions": "visu_pars/VisuAcqSoftwareVersion",
        "SequenceName": "visu_pars/VisuAcquisitionProtocol",
        "PatientName": "visu_pars/VisuSubjectName",
        "DwellTime": "visu_pars/VisuAcqPixelBandwidth",
    }

    def convert(self, sniff: Sniff) -> tuple[list[NIFTI_MRS], list[str], dict]:
        path = self._data_file(sniff.path)
        args = Namespace(
            file=str(path),
            mode=self.MODES[path.name],
            query=[],
            dump_headers=False,
            zeropad=False,
            nospectrum=False,
            fileout=None,
        )
        images, names = read_bruker(args)
        return list(images), list(names), self._header(path)

    def sources(self, vendor: dict) -> dict[str, str | tuple[str, ...]]:
        return self.SOURCES_2DSEQ if "visu_pars" in vendor else self.SOURCES_FID

    def _data_file(self, path: Path) -> Path:
        if path.name in self.MODES:
            return path
        folder = path if path.is_dir() else path.parent
        for name in self.MODES:
            if (folder / name).is_file():
                return folder / name
        raise ReaderError(f"no Bruker data file (fid, 2dseq, rawdata) in {folder}")

    def _header(self, data_file: Path) -> dict:
        """All parameters of the files that describe ``data_file``, by file name."""
        if data_file.name in self.PROCESSED:
            folders = (data_file.parent, data_file.parents[2], data_file.parents[3])
        else:  # <study>/<scan>/fid
            folders = (data_file.parent, data_file.parents[1])
        header = {}
        for folder in folders:
            for name in self.PARAMETER_FILES:
                if name not in header and (folder / name).is_file():
                    header[name] = self._parameters(folder / name)
        return header

    def _parameters(self, path: Path) -> dict:
        parameters = {}
        for key, parameter in JCAMPDX(path).params.items():
            try:
                parameters[key] = self.plain(parameter.value)
            except Exception:
                parameters[key] = parameter.val_str
        return parameters
