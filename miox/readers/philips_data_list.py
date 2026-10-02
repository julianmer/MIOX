####################################################################################################
#                                       philips_data_list.py                                       #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for Philips .data/.list raw spectroscopy. The pair has no frequency, dwell time  #
#          or geometry, so a SPAR or DICOM export of the same scan must sit in the same folder.    #
#          spec2nii converts; the vendor header holds the auxiliary file's header plus every line  #
#          of the .list file.                                                                      #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re
from pathlib import Path

from nifti_mrs.nifti_mrs import NIFTI_MRS
from spec2nii.Philips.philips import read_spar
from spec2nii.Philips.philips_data_list import read_data_list_pair

# own
from miox.data import MRSData
from miox.readers.base import Reader, ReaderError, Spec2niiReader
from miox.readers.dicom_mrs import DicomMrsReader
from miox.readers.philips_spar_sdat import PhilipsSparSdatReader
from miox.sniff import Format, Sniff, Sniffer


#**************************************************************************************************#
#                                      PhilipsDataListReader                                       #
#**************************************************************************************************#
#                                                                                                  #
# The auxiliary file is found in the folder: a SPAR ending in _act first, then any SPAR, then a    #
# Philips spectroscopy DICOM; SPARs that disagree on frequency or dwell time make the read fail    #
# rather than guess. Sources and corrections are those of the auxiliary file's own reader.         #
#                                                                                                  #
#**************************************************************************************************#
class PhilipsDataListReader(Spec2niiReader):
    FORMATS = (Format.PHILIPS_DATA_LIST,)
    CONVERTER = "philips_dl"
    INFO = re.compile(r"^\.\s+(\d+)\s+(\d+)\s+(\d+)\s+(\w+)\s*:\s*(.*)$")  # general-info line
    HEADER = re.compile(r"^#\s+([^:=]+?)\s*:\s*(.*)$")  # comment header line

    def __init__(self):
        self.spar = PhilipsSparSdatReader()
        self.dicom = DicomMrsReader()
        self.sniffer = Sniffer()

    def convert(self, sniff: Sniff) -> tuple[list[NIFTI_MRS], list[str], dict]:
        data, list_file = self._pair(sniff)
        aux = self._auxiliary(data.parent)
        images, names = read_data_list_pair(data, list_file, aux, None)
        if aux.suffix.lower() == ".spar":
            header = {k: self.plain(v) for k, v in read_spar(aux).items()}
        else:
            header = self.dicom.header(aux)
        header.update(self._list(list_file))
        header["auxiliary_file"] = aux.name
        return list(images), [f"{data.stem}_{name}" for name in names], header

    def sources(self, vendor: dict) -> dict[str, str | tuple[str, ...]]:
        return self._aux_reader(vendor).sources(vendor)

    def correct(self, d: MRSData) -> None:
        """The auxiliary reader's corrections, and the HYPER echo times spec2nii leaves at 80 ms.

        The water references of a HYPER scan alternate between the edited (80 ms) and short (35 ms)
        echo time, starting with 80 ms, so their echo time goes per index into the dimension header.
        """
        self._aux_reader(d.meta.vendor).correct(d)
        if d.name.endswith("_hyper_short_te"):
            self.fix(d, "EchoTime", self.spar.HYPER_SHORT_TE, "HYPER short-TE block is 35 ms")
        elif d.name.endswith("_hyper_water_ref") and d.data.ndim == 6:
            n = d.shape[5]
            times = [(self.spar.HYPER_EDITED_TE, self.spar.HYPER_SHORT_TE)[i % 2] for i in range(n)]
            d.dim_info.setdefault(6, {})["header"] = {"EchoTime": times}
            d.dim_info[6]["info"] = "HYPER water references, echo time alternating 80 / 35 ms"
            for record in (d.meta, d.meta.source, d.meta.fields):
                record.pop("EchoTime", None)

    def _aux_reader(self, vendor: dict) -> Reader:
        return self.dicom if "dicom" in vendor else self.spar

    def _pair(self, sniff: Sniff) -> tuple[Path, Path]:
        files = (sniff.path,) + sniff.companions
        data = next((f for f in files if f.suffix.lower() == ".data"), None)
        list_file = next((f for f in files if f.suffix.lower() == ".list"), None)
        if data is None or list_file is None:
            raise ReaderError(f"incomplete .data/.list pair: {sniff.path.name}")
        return data, list_file

    def _auxiliary(self, folder: Path) -> Path:
        spars = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".spar")
        spars.sort(key=lambda p: not p.stem.lower().endswith("_act"))
        dicoms = [
            p
            for p in sorted(folder.iterdir())
            if p.is_file()
            and p.suffix.lower() not in (".spar", ".sdat", ".data", ".list")
            and self.sniffer.sniff(p).format == Format.DICOM_MRS
        ]
        candidates = spars + dicoms
        if not candidates:
            raise ReaderError(f"no SPAR or DICOM of the scan next to the data/list: {folder}")
        settings = {self._settings(p) for p in spars}
        if len(settings) > 1:
            raise ReaderError(f"SPAR files in {folder} disagree on frequency or dwell time")
        return candidates[0]

    @staticmethod
    def _settings(spar: Path) -> tuple[float, float]:
        header = read_spar(spar)
        return float(header["synthesizer_frequency"]), float(header["sample_frequency"])

    def _list(self, path: Path) -> dict:
        """The .list header and general information, and its vector table as raw lines."""
        header, info, vectors = {}, {}, []
        for line in path.read_text(errors="replace").splitlines():
            if match := self.INFO.match(line):
                mix, echo, other, name, value = match.groups()
                info.setdefault(name, {})[f"{mix} {echo} {other}"] = value.strip()
            elif line.startswith("#"):
                if match := self.HEADER.match(line):
                    header[match[1]] = match[2]
            elif line.strip():
                vectors.append(line)
        return {"list_header": header, "list_info": info, "list_vectors": vectors}
