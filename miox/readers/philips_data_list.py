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

import dataclasses
import re
from pathlib import Path

import numpy as np
from nifti_mrs.nifti_mrs import NIFTI_MRS
from spec2nii.Philips.philips_data_list import read_data_list_pair

# own
from miox.data import MRSData, RawData
from miox.readers.base import Reader, ReaderError, Spec2niiReader
from miox.readers.dicom_mrs import DicomMrsReader
from miox.readers.philips_list_raw import PhilipsListRaw
from miox.readers.philips_spar_sdat import PhilipsSparSdatReader
from miox.sniff import Format, Sniff, Sniffer


#**************************************************************************************************#
#                                      PhilipsDataListReader                                       #
#**************************************************************************************************#
#                                                                                                  #
# Imaging scans (the readout spans the kx range of the .list) are read as raw k-space              #
# (PhilipsListRaw). For spectroscopy, the auxiliary file is found in the folder: a SPAR ending in  #
# _act first, then any SPAR, then a Philips spectroscopy DICOM; SPARs that disagree on frequency   #
# or dwell time make the read fail rather than guess. Without one there, the SPAR of the same scan #
# in a folder beside it is used: raw_<n>.data belongs to <name>_<n>_<m>_raw_act.SPAR, as the       #
# scanner numbers its exports. Sources and corrections are those of the auxiliary file's own       #
# reader. Spectroscopic imaging (k-space ranges in the .list) is refused: spec2nii would put every #
# k-space position on one spectrum.                                                                #
#                                                                                                  #
#**************************************************************************************************#
class PhilipsDataListReader(Spec2niiReader):
    FORMATS = (Format.PHILIPS_DATA_LIST,)
    CONVERTER = "philips_dl"
    INFO = re.compile(r"^\.\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S.*?)\s*:\s*(.*)$")  # general info
    SCAN = re.compile(r"^raw_(\d+)$", re.IGNORECASE)  # raw_<scan number>.data
    EXPORT = re.compile(r"_(\d+)_\d+_raw_act$", re.IGNORECASE)  # <name>_<scan>_<n>_raw_act.SPAR
    HEADER = re.compile(r"^#\s+([^:=]+?)\s*:\s*(.*)$")  # comment header line

    def __init__(self):
        self.spar = PhilipsSparSdatReader()
        self.dicom = DicomMrsReader()
        self.sniffer = Sniffer()

    def read(self, sniff: Sniff) -> list[MRSData | RawData]:
        """Imaging k-space as RawData; spectroscopy through spec2nii and the auxiliary file."""
        data, list_file = self._pair(sniff)
        parsed = self._list(list_file)
        rows = PhilipsListRaw.vectors(parsed["list_vectors"])
        if PhilipsListRaw.imaging(parsed["list_info"], rows):
            raw = PhilipsListRaw(data, list_file, parsed["list_info"], self.provenance(sniff))
            return raw.datasets(rows)
        return super().read(sniff)

    def convert(self, sniff: Sniff) -> tuple[list[NIFTI_MRS], list[str], dict]:
        data, list_file = self._pair(sniff)
        info = self._list(list_file)["list_info"]
        if any(len(set(r.split())) > 1 for k in ("kx_range", "ky_range", "kz_range")
               for r in info.get(k, {}).values()):
            raise ReaderError(f"{data.name}: spectroscopic imaging k-space (kx, ky, kz), which is "
                              "not reconstructed here: read the scan's SPAR/SDAT or DICOM")
        aux = self._auxiliary(data.parent, data.stem)
        if aux.suffix.lower() == ".spar":
            spar = self.spar.read_spar(aux)
            with self.spar.defaults(spar):
                images, names = read_data_list_pair(data, list_file, aux, None)
            header = {k: self.plain(v) for k, v in spar.items()}
        else:
            images, names = read_data_list_pair(data, list_file, aux, None)
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
        self._edit_pairs(d)

    @staticmethod
    def _edit_pairs(d: MRSData) -> None:
        """Edited scans (MEGA): spec2nii puts every transient on its own extr1 index and in the
        extr2 slot of its edit condition, the other slot empty. Here transient pairs (extr1 // 2)
        are DIM_DYN and extr2 DIM_EDIT, ON (extr2 1, as the scan's DICOM flags it) first; a pair
        with a transient (or a coil of one) missing from the export is left out, with a note."""
        infos = [d.dim_info.get(n, {}).get("info") for n in (6, 7)]
        if d.data.ndim != 7 or infos != ["data/list dim extr1", "data/list dim extr2"]:
            return
        data = np.asarray(d.data)  # (x, y, z, points, coils, extr1, extr2)
        if data.shape[6] != 2:
            return
        per_coil = np.abs(data).sum(axis=(0, 1, 2, 3)) > 0  # (coils, extr1, extr2)
        filled, complete = per_coil.any(axis=0), per_coil.all(axis=0)
        if (filled.sum(axis=1) > 1).any():
            return
        pairs, missing = [], []
        for j in range(data.shape[5] // 2):
            slots = [next((i for i in (2 * j, 2 * j + 1) if complete[i, e]), None) for e in (1, 0)]
            if None in slots:
                missing += [i for i in (2 * j, 2 * j + 1) if filled[i].any()]
                continue
            pairs.append([data[..., i, e] for i, e in zip(slots, (1, 0), strict=True)])
        if not pairs:
            return
        d.data = np.stack([np.stack(p, axis=-1) for p in pairs], axis=-2)  # (..., pairs, 2)
        d.dim_tags = ("DIM_COIL", "DIM_DYN", "DIM_EDIT")
        d.dim_info = {6: {"info": "transient pairs (extr1 // 2)"},
                      7: {"info": "edit condition (extr2)",
                          "header": {"EditCondition": ["ON", "OFF"]}}}
        if missing:
            d.provenance = dataclasses.replace(d.provenance, notes=d.provenance.notes + (
                f"transients {', '.join(map(str, missing))} (extr1) left out: their pair is "
                "incomplete in the export (a transient or a coil of one missing)",))

    def _aux_reader(self, vendor: dict) -> Reader:
        return self.dicom if "dicom" in vendor else self.spar

    def _pair(self, sniff: Sniff) -> tuple[Path, Path]:
        files = (sniff.path,) + sniff.companions
        data = next((f for f in files if f.suffix.lower() == ".data"), None)
        list_file = next((f for f in files if f.suffix.lower() == ".list"), None)
        if data is None or list_file is None:
            raise ReaderError(f"incomplete .data/.list pair: {sniff.path.name}")
        return data, list_file

    def _auxiliary(self, folder: Path, stem: str) -> Path:
        spars = sorted(p for p in folder.iterdir() if self._spar(p))
        if not spars and (scan := self.SCAN.match(stem)):
            beside = [p for sub in sorted(folder.parent.iterdir()) if sub.is_dir() and sub != folder
                      for p in sorted(sub.iterdir()) if self._spar(p)
                      and (m := self.EXPORT.search(p.stem)) and int(m[1]) == int(scan[1])]
            if beside:
                return beside[0]
        spars.sort(key=lambda p: not p.stem.lower().endswith("_act"))
        dicoms = [
            p
            for p in sorted(folder.iterdir())
            if p.is_file() and not p.name.startswith("._")
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
    def _spar(path: Path) -> bool:
        """A SPAR file, not the macOS resource fork (._name) copied beside it."""
        return path.suffix.lower() == ".spar" and not path.name.startswith("._")

    @staticmethod
    def _settings(spar: Path) -> tuple[float, float]:
        header = PhilipsSparSdatReader.read_spar(spar)
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
