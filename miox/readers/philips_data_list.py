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
#          The vendor header holds the auxiliary file's header plus every line of the .list file.  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np

# own
from miox import __version__
from miox.data import MRSData, RawData
from miox.readers.base import Reader, ReaderError
from miox.readers.dicom_mrs import DicomMrsReader
from miox.readers.philips_dicom_mrs import PhilipsDicomMrs
from miox.readers.philips_list_raw import PhilipsListRaw
from miox.readers.philips_spar_sdat import PhilipsSparSdatReader
from miox.sniff import Format, Sniff, Sniffer


#**************************************************************************************************#
#                                      PhilipsDataListReader                                       #
#**************************************************************************************************#
#                                                                                                  #
# Imaging scans (the readout spans the kx range of the .list) are read as raw k-space              #
# (PhilipsListRaw). For spectroscopy, the auxiliary file is found in the folder: a SPAR ending in  #
# _act first, then any SPAR, then a Philips spectroscopy DICOM; without one, or with SPARs that    #
# disagree on frequency or dwell time, the data/list is read by itself: the data and their         #
# dimensions as acquired, with the scan name and software release the head of the .list states,    #
# and without frequency, dwell time, voxel and the other scan keys, which it does not state.       #
# raw_<n>.data belongs to <name>_<n>_<m>_raw_act.SPAR, as the scanner numbers its exports: of      #
# several SPARs of one kind in the folder the one of that scan is taken; where the names tell none #
# apart, only what they all state alike is kept (keys, voxel, vendor header), with a note. Without #
# a SPAR there, the SPAR of the same scan in a folder beside it is used. Keys, dwell time and      #
# voxel come from the SPAR as the SPAR/SDAT reader reads them, or from the DICOM's header as       #
# PhilipsDicomMrs reads it. Every vector of the .list is a complex float32 readout at its offset   #
# in the .data. Vectors of one type and mix (STD_0, STD_1, ..., NOI) form one dataset of (points,  #
# chan, aver, dyn, card, echo, loca, extr1, extr2), each counter holding the values it takes in    #
# order; counters with one value are dropped (more than three left is refused), the data           #
# conjugated as the SDAT. Tags: chan DIM_COIL, aver and dyn DIM_DYN, echo DIM_INDIRECT_0, the      #
# others DIM_USER_n. HYPER (protocol name with hyper): STD_0 splits as the SPAR/SDAT reader splits #
# the act file, STD_1 holds the water references, echo time alternating 80 / 35 ms per index.      #
# Edited scans (MEGA: extr1 the transient, extr2 its edit condition) become transient pairs (extr1 #
# // 2, DIM_DYN) by condition (DIM_EDIT, ON = extr2 1 first, as the scan's DICOM flags its         #
# frames). Spectroscopic imaging (k-space ranges in the .list) is refused.                         #
#                                                                                                  #
#**************************************************************************************************#
class PhilipsDataListReader(Reader):
    FORMATS = (Format.PHILIPS_DATA_LIST,)
    INFO = re.compile(r"^\.\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S.*?)\s*:\s*(.*)$")  # general info
    SCAN = re.compile(r"^raw_(\d+)$", re.IGNORECASE)  # raw_<scan number>.data
    EXPORT = re.compile(r"_(\d+)_\d+_raw_act$", re.IGNORECASE)  # <name>_<scan>_<n>_raw_act.SPAR
    HEADER = re.compile(r"^#\s+([^:=]+?)\s*:\s*(.*)$")  # comment header line
    COUNTERS = ("chan", "aver", "dyn", "card", "echo", "loca", "extr1", "extr2")
    TAGS = {"chan": "DIM_COIL", "aver": "DIM_DYN", "dyn": "DIM_DYN", "echo": "DIM_INDIRECT_0"}
    DICOM_SOURCE = "DICOM"
    SOURCE = "data/list"
    WATER_REFERENCES = "HYPER water references, echo time alternating 80 / 35 ms"
    VOXEL = ("volume_selection_enable",) + tuple(
        f"{axis}_{field}" for axis in ("lr", "ap", "cc")
        for field in ("angulation", "size", "off_center"))

    def __init__(self):
        self.spar = PhilipsSparSdatReader()
        self.dicom = DicomMrsReader()
        self.sniffer = Sniffer()

    def read(self, sniff: Sniff) -> list[MRSData | RawData]:
        data, list_file = self._pair(sniff)
        parsed = self._list(list_file)
        rows = PhilipsListRaw.vectors(parsed["list_vectors"])
        if PhilipsListRaw.imaging(parsed["list_info"], rows):
            raw = PhilipsListRaw(data, list_file, parsed["list_info"], self.provenance(sniff))
            return raw.datasets(rows)
        info = parsed["list_info"]
        if any(len(set(r.split())) > 1 for k in ("kx_range", "ky_range", "kz_range")
               for r in info.get(k, {}).values()):
            raise ReaderError(f"{data.name}: spectroscopic imaging k-space (kx, ky, kz), which is "
                              "not reconstructed here: read the scan's SPAR/SDAT or DICOM")
        notes: list[str] = []
        aux, others = self._auxiliary(data.parent, data.stem, notes)
        if aux is None:
            values, vendor = self._alone(parsed["list_header"]), {}
            dwelltime, affine = None, self.spar.NO_VOXEL
            source = self.SOURCE
        elif aux.suffix.lower() == ".spar":
            stated = self.spar.read_spar(aux)
            header = self._alike(stated, others)
            values = self.spar._values(stated, aux, notes)
            unknown = [k for k, (_, field) in values.items() if field and field not in header]
            values = {k: v for k, v in values.items() if k not in unknown}
            dwelltime = 1.0 / float(header["sample_frequency"])
            values["DwellTime"] = (dwelltime, "sample_frequency")
            voxel = all(field in header for field in self.VOXEL)
            affine = self.spar.voxel(header) if voxel else self.spar.NO_VOXEL
            source = self.spar.SOURCE
            vendor = {key: self.plain(value) for key, value in header.items()}
            if others:
                names = ", ".join(p.name for p in [aux] + others)
                notes.append(f"several SPAR files may be this scan's ({names}), their names do "
                             "not tell which: only what they state alike is kept"
                             + (f"; unknown: {', '.join(unknown)}" if unknown else "")
                             + ("" if voxel else "; voxel unknown"))
        else:
            values, dwelltime, affine, vendor = self._dicom_values(aux, notes)
            source = self.DICOM_SOURCE
        vendor.update(parsed)
        auxiliaries = [p.name for p in [aux] + others if p is not None]
        if auxiliaries:
            vendor["auxiliary_file"] = aux.name if not others else auxiliaries
        values["OriginalFile"] = ([data.name, list_file.name] + auxiliaries, None)
        hyper = "hyper" in str(values.get("ProtocolName", ("",))[0]).lower()
        blob = np.fromfile(data, dtype="<c8")
        datasets = []
        for name, array, counters in self._arrays(blob, rows):
            for suffix, part, part_values, tags, part_notes in self._parts(
                    name, array, counters, values, hyper):
                part_notes = notes + part_notes
                part, tags = self.without_zero_indices(part, tags, part_notes)
                part, tags = self._edit_pairs(part, tags, part_notes)
                datasets.append(self.dataset(sniff, part, dwelltime, part_values, vendor,
                                             f"{data.stem}_{suffix}", part_notes, source, affine,
                                             tags))
        return datasets

    # --- data ------------------------------------------------------------------------------------

    def _arrays(self, blob: np.ndarray, rows: list[dict]) -> list[tuple[str, np.ndarray, list]]:
        """One array per vector type and mix, (1, 1, 1, points, counters...) conjugated, with the
        names of the counters kept."""
        groups: dict[str, dict[int, list[dict]]] = {}
        for row in rows:
            groups.setdefault(row["typ"], {}).setdefault(row["mix"], []).append(row)
        arrays = []
        for typ, mixes in groups.items():
            for mix in sorted(mixes):
                vectors = mixes[mix]
                taken = [sorted({row[c] for row in vectors}) for c in self.COUNTERS]
                position = [{value: n for n, value in enumerate(v)} for v in taken]
                array = np.zeros((max(row["size"] for row in vectors) // 8,)
                                 + tuple(len(v) for v in taken), np.complex64)
                for row in vectors:
                    start, size = row["offset"] // 8, row["size"] // 8
                    index = tuple(p[row[c]] for p, c in zip(position, self.COUNTERS, strict=True))
                    array[(slice(0, size),) + index] = blob[start:start + size]
                kept = [c for c, v in zip(self.COUNTERS, taken, strict=True) if len(v) > 1]
                if len(kept) > 3:
                    raise ReaderError(f"{typ}_{mix}: more than three counters vary ({kept}); "
                                      "NIfTI-MRS has three dimensions for them")
                shape = (1, 1, 1, array.shape[0]) + tuple(len(v) for v in taken if len(v) > 1)
                name = typ if typ == "NOI" else f"{typ}_{mix}"
                arrays.append((name, array.reshape(shape).conj(), kept))
        return arrays

    def _parts(self, name: str, array: np.ndarray, counters: list[str], values: dict,
               hyper: bool) -> list[tuple[str, np.ndarray, dict, tuple, list]]:
        """The datasets of one array: (name suffix, data, values, dimension tags, notes)."""
        if hyper and name == "STD_0":
            blocks, notes = self.spar.hyper_blocks(array, values, ("DIM_COIL",))
            return [(suffix.removeprefix("_"), data, block_values, tags, notes)
                    for suffix, data, block_values, tags in blocks]
        if hyper and name == "STD_1":
            n = array.shape[-1]
            times = [(self.spar.HYPER_EDITED_TE, self.spar.HYPER_SHORT_TE)[i % 2] for i in range(n)]
            tags = ("DIM_COIL", ("DIM_USER_0", self.WATER_REFERENCES, {"EchoTime": times}))
            references = {k: v for k, v in values.items() if k != "EchoTime"}
            return [("hyper_water_ref", array, references, tags, [])]
        tags, users = [], 0
        for counter in counters:
            tag = self.TAGS.get(counter)
            if tag is None:
                tag, users = f"DIM_USER_{users}", users + 1
            tags.append((tag, f"data/list dim {counter}", None))
        return [(name, array, values, tuple(tags), [])]

    @staticmethod
    def _edit_pairs(data: np.ndarray, tags: tuple, notes: list[str]) -> tuple[np.ndarray, tuple]:
        """Edited scans (MEGA) hold every transient on its own extr1 index, in the extr2 slot of
        its edit condition, the other slot empty. Here transient pairs (extr1 // 2) are DIM_DYN and
        extr2 DIM_EDIT, ON (extr2 1) first; a pair with a transient (or a coil of one) missing from
        the export is left out, with a note."""
        infos = [tag[1] if isinstance(tag, tuple) else None for tag in tags]
        if (data.ndim != 7 or data.shape[6] != 2
                or infos[1:] != ["data/list dim extr1", "data/list dim extr2"]):
            return data, tags
        per_coil = np.abs(data).sum(axis=(0, 1, 2, 3)) > 0  # (coils, extr1, extr2)
        filled, complete = per_coil.any(axis=0), per_coil.all(axis=0)
        if (filled.sum(axis=1) > 1).any():
            return data, tags
        pairs, missing = [], []
        for j in range(data.shape[5] // 2):
            slots = [next((i for i in (2 * j, 2 * j + 1) if complete[i, e]), None) for e in (1, 0)]
            if None in slots:
                missing += [i for i in (2 * j, 2 * j + 1) if filled[i].any()]
                continue
            pairs.append([data[..., i, e] for i, e in zip(slots, (1, 0), strict=True)])
        if not pairs:
            return data, tags
        if missing:
            notes.append(f"transients {', '.join(map(str, missing))} (extr1) left out: their pair "
                         "is incomplete in the export (a transient or a coil of one missing)")
        data = np.stack([np.stack(p, axis=-1) for p in pairs], axis=-2)  # (..., pairs, 2)
        return data, ("DIM_COIL", ("DIM_DYN", "transient pairs (extr1 // 2)", None),
                      ("DIM_EDIT", "edit condition (extr2)", {"EditCondition": ["ON", "OFF"]}))

    # --- auxiliary file --------------------------------------------------------------------------

    def _dicom_values(self, path: Path, notes: list[str]) -> tuple[dict, float, np.ndarray, dict]:
        """Keys (value, field), dwell time, voxel and vendor header of a Philips DICOM."""
        philips = PhilipsDicomMrs()
        ds = philips.dataset(path)
        affine, voxel_notes = philips.voxel(ds)
        notes += voxel_notes
        values = philips.values(ds, [path.name])
        return values, philips.dwelltime(ds), affine, self.dicom.header(path)

    def _pair(self, sniff: Sniff) -> tuple[Path, Path]:
        files = (sniff.path,) + sniff.companions
        data = next((f for f in files if f.suffix.lower() == ".data"), None)
        list_file = next((f for f in files if f.suffix.lower() == ".list"), None)
        if data is None or list_file is None:
            raise ReaderError(f"incomplete .data/.list pair: {sniff.path.name}")
        return data, list_file

    @staticmethod
    def _alone(header: dict) -> dict:
        """The keys of a data/list read by itself: what the head of the .list states."""
        values = {
            "Manufacturer": ("Philips", None),
            "ConversionMethod": (f"miox {__version__}", None),
            "ConversionTime": (datetime.now().isoformat(timespec="milliseconds"), None),
            "kSpace": ([False, False, False], None),
        }
        for key, name in (("ProtocolName", "Scan name"),
                          ("SoftwareVersions", "Gyroscan SW release")):
            if header.get(name):
                values[key] = (header[name], f"list_header/{name}")
        return values

    @staticmethod
    def _entries(folder: Path) -> list[Path]:
        """The entries of a folder, none where it cannot be listed."""
        try:
            return sorted(folder.iterdir())
        except OSError:
            return []

    def _auxiliary(self, folder: Path, stem: str,
                   notes: list[str]) -> tuple[Path | None, list[Path]]:
        """The scan's SPAR or DICOM, and the other SPARs that may as well be the scan's; none
        where the folder holds none, or SPARs that disagree on frequency or dwell time."""
        spars = sorted(p for p in folder.iterdir() if self._spar(p))
        scan = self.SCAN.match(stem)
        if not spars and scan:
            beside = [p for sub in sorted(folder.parent.iterdir()) if sub.is_dir() and sub != folder
                      for p in self._entries(sub) if self._spar(p)
                      and (m := self.EXPORT.search(p.stem)) and int(m[1]) == int(scan[1])]
            if beside:
                return beside[0], []
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
            notes.append("no SPAR or DICOM of the scan next to the data/list, which states "
                         "neither frequency, dwell time, voxel nor scan parameters: the data as "
                         "acquired (pass the missing keys in given)")
            return None, []
        settings = {self._settings(p) for p in spars}
        if len(settings) > 1:
            notes.append("the SPAR files next to the data/list disagree on frequency or dwell "
                         "time, none is taken as the scan's: the data as acquired (pass the "
                         "missing keys in given)")
            return None, []
        if not spars:
            return candidates[0], []
        kind = [p for p in spars if p.stem.lower().endswith("_act")] or spars
        if scan:
            kind = [p for p in kind if (m := self.EXPORT.search(p.stem))
                    and int(m[1]) == int(scan[1])] or kind
        return kind[0], kind[1:]

    def _alike(self, header: dict, others: list[Path]) -> dict:
        """The header's fields that every other SPAR states with the same value."""
        for other in others:
            stated = self.spar.read_spar(other)
            header = {k: v for k, v in header.items() if k in stated and stated[k] == v}
        return header

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
