####################################################################################################
#                                            physio.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/10/26                                                                                #
#                                                                                                  #
# Purpose: Readers for physiological logs recorded with MR scans: Siemens PMU files, CMRR physio   #
#          logs (as files or inside a DICOM file), Philips SCANPHYSLOG files, GE PPG/RESP/ECG      #
#          files and BIDS physio recordings.                                                       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import gzip
import json
import re
import struct
from pathlib import Path

import numpy as np
import pydicom

# own
from miox.data import PhysioData
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                         SiemensPmuReader                                         #
#**************************************************************************************************#
#                                                                                                  #
# A PMU file (.puls, .resp, .ecg, .ext) holds on its first lines four header values, then the      #
# samples as integers, with text sections between 5002 and 6002, trigger markers 5000 and 6000     #
# between the samples (not samples themselves: counting them would not match the logged duration)  #
# and 5003 after the last sample; a footer follows with the rates and statistics the unit computed #
# and the start and stop times of the log in ms since midnight on the scanner (MDH) and the        #
# physiological unit's (MPCU) clocks. The sampling frequency is the one a text section states,     #
# else the sample count over the MPCU duration rounded to whole hertz when within 1 %. The start   #
# time is LogStartMDHTime, on the clock of the scanner's DICOM times. ECG files may hold several   #
# leads; they are read as one channel (no sample was at hand).                                     #
#                                                                                                  #
#**************************************************************************************************#
class SiemensPmuReader(Reader):
    FORMATS = (Format.SIEMENS_PMU,)
    PACKAGES = ("miox",)
    SOURCE = "PMU file"
    CHANNELS = {".puls": "cardiac", ".resp": "respiratory", ".ecg": "ecg", ".ext": "external",
                ".ext2": "external2"}
    MARKERS = {5000: "trigger", 6000: "marker_6000"}
    FOOTER = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 /]*?)\s*:\s*(.*)$")

    def read(self, sniff: Sniff) -> list[PhysioData]:
        text = sniff.path.read_text(errors="replace")
        end = re.search(r"(?<!\d)5003(?!\d)", text)
        if end is None:
            raise ReaderError(f"{sniff.path.name}: no end marker 5003")
        body, footer = text[: end.start()], text[end.end():]
        sections = [s.strip() for s in re.findall(r"5002(.*?)6002", body, re.S)]
        body = re.sub(r"5002.*?6002", " ", body, flags=re.S)
        tokens = [int(t) for t in body.split() if re.fullmatch(r"-?\d+", t) and t != "6002"]
        notes: list[str] = []
        if any(not re.fullmatch(r"-?\d+", t) for t in body.split()):
            notes.append("words outside the text sections passed over")
        header, values = tokens[:4], tokens[4:]
        samples, events = [], {name: [] for name in self.MARKERS.values()}
        for value in values:
            if value in self.MARKERS:
                events[self.MARKERS[value]].append(len(samples))
            else:
                samples.append(value)
        vendor = {"header_values": header, "text_sections": sections,
                  "footer": self._footer(footer)}
        meta = Meta(vendor)
        rate = self._rate(len(samples), sections, vendor["footer"], meta, notes)
        start = vendor["footer"].get("LogStartMDHTime")
        start_time = start[0] / 1000 if start else None
        channel = self.CHANNELS.get(sniff.path.suffix.lower(), sniff.path.suffix.lstrip("."))
        if channel == "ecg":
            notes.append("ECG file read as one channel; several leads may be interleaved")
        data = np.asarray(samples, dtype=np.float64)[:, None]
        events_out = {k: np.asarray(v, dtype=np.int64) for k, v in events.items() if v}
        return [PhysioData(
            data, (channel,), rate, start_time, "s since midnight, scanner clock (LogStartMDHTime)",
            events_out, meta, self.provenance(sniff, tuple(notes)), sniff.path.stem,
        )]

    def _footer(self, text: str) -> dict:
        footer: dict = {}
        for line in text.splitlines():
            match = self.FOOTER.match(line)
            if match:
                values = [self._number(v) for v in match.group(2).split()]
                footer[match.group(1)] = values
        return footer

    @staticmethod
    def _number(text: str):
        try:
            return int(text)
        except ValueError:
            try:
                return float(text)
            except ValueError:
                return text

    def _rate(self, count: int, sections: list[str], footer: dict, meta: Meta,
              notes: list[str]) -> float | None:
        for section in sections:
            stated = re.search(r"_SAMPLES_PER_SECOND\s*=\s*(\d+)", section)
            if stated:
                return float(stated.group(1))
        start, stop = footer.get("LogStartMPCUTime"), footer.get("LogStopMPCUTime")
        if not start or not stop or stop[0] <= start[0] or count == 0:
            notes.append("sampling frequency not stated and not derivable")
            return None
        measured = count / ((stop[0] - start[0]) / 1000)
        rate = float(round(measured))
        if abs(measured - rate) > 0.01 * rate:
            notes.append(f"sampling frequency {measured:.3f} Hz from the logged duration")
            return measured
        notes.append(f"sampling frequency {rate:g} Hz: {count} samples over the logged MPCU "
                     f"duration give {measured:.3f} Hz")
        return rate


#**************************************************************************************************#
#                                        CmrrPhysioReader                                          #
#**************************************************************************************************#
#                                                                                                  #
# The CMRR multiband sequences log the signals as text files (Physio_*_PULS.log, _RESP, _ECG, _EXT #
# and _Info) or put those files into one DICOM file (Siemens raw data with image type PHYSIO, the  #
# files in element 7FE1,1010: chunks of AcquisitionNumber x 1024 bytes, each starting with the     #
# data length and the file name, the text from byte 1024). A log has "key = value" lines           #
# (LogDataType, SampleTime in tics) and rows of time in 2.5 ms tics, channel, value and signal     #
# (e.g. PULS_TRIGGER, kept as events). The samples are placed on the grid of SampleTime tics from  #
# the first row; missing rows stay NaN and are counted in a note. The Info log (volume, slice,     #
# start and finish tics, echo of every slice acquisition) becomes a table without sampling         #
# frequency, its volume starts the event "volume" of the signals of the same UUID.                 #
#                                                                                                  #
#**************************************************************************************************#
class CmrrPhysioReader(Reader):
    FORMATS = (Format.CMRR_PHYSIO,)
    PACKAGES = ("miox", "pydicom")
    TIC = 2.5e-3  # s
    CHANNELS = {"PULS": "cardiac", "RESP": "respiratory"}
    INFO_COLUMNS = ("volume", "slice", "acq_start_tics", "acq_finish_tics", "echo")

    def read(self, sniff: Sniff) -> list[PhysioData]:
        paths = sniff.companions if sniff.path.is_dir() else (sniff.path,)
        datasets = []
        for path in paths:
            if path.suffix.lower() == ".log":
                logs = [(path.name, path.read_text(errors="replace").splitlines())]
                lent = self._sibling_info(path)  # only for its volume starts
            else:
                logs, lent = self.dicom_logs(path), []
            parsed = [(name, *self.parse(lines)) for name, lines in logs]
            volumes = {}
            for _, header, rows in parsed + [(n, *self.parse(lines)) for n, lines in lent]:
                if header.get("LogDataType") == "ACQUISITION_INFO" and rows:
                    volumes[header.get("UUID")] = self._volume_starts(rows)
            datasets += [self._dataset(sniff, name, header, rows, volumes)
                         for name, header, rows in parsed]
        if not datasets:
            raise ReaderError(f"{sniff.path.name}: no CMRR physio logs")
        return datasets

    @staticmethod
    def dicom_logs(path: Path) -> list[tuple[str, list[str]]]:
        ds = pydicom.dcmread(path)
        if (0x7FE1, 0x1010) not in ds:
            raise ReaderError(f"{path.name}: no physio data element (7FE1,1010)")
        blob = bytes(ds[0x7FE1, 0x1010].value)
        chunk = int(ds.get("AcquisitionNumber", 0) or 0) * 1024
        if chunk <= 0 or len(blob) % chunk:
            raise ReaderError(f"{path.name}: {len(blob)} bytes are no whole number of "
                              f"AcquisitionNumber x 1024 byte chunks")
        logs = []
        for start in range(0, len(blob), chunk):
            size, name_size = struct.unpack_from("<II", blob, start)
            name = blob[start + 8 : start + 8 + name_size].decode("ascii", "replace")
            text = blob[start + 1024 : start + 1024 + size].decode("utf-8", "replace")
            lines = text.splitlines()
            if lines and not text.endswith("\n"):
                lines[-1] = "#cut " + lines[-1]  # a last line without its end is incomplete
            logs.append((name, lines))
        return logs

    @staticmethod
    def _sibling_info(path: Path) -> list[tuple[str, list[str]]]:
        """The Info log beside a signal log (same name up to the signal), if there is one."""
        if path.name.endswith("_Info.log"):
            return []
        info = path.with_name(re.sub(r"_[A-Za-z0-9]+\.log$", "_Info.log", path.name))
        if info.is_file() and info != path:
            return [(info.name, info.read_text(errors="replace").splitlines())]
        return []

    @staticmethod
    def parse(lines: list[str]) -> tuple[dict, list[list[str]]]:
        header, rows = {}, []
        for line in lines:
            if line.startswith("#cut "):
                header["_cut"] = line[5:]
            elif "=" in line:
                key, _, value = line.partition("=")
                header[key.strip()] = value.strip()
            elif line.strip() and line.split()[0].lstrip("-").isdigit():
                rows.append(line.split())
        return header, rows

    def _volume_starts(self, rows: list[list[str]]) -> dict[int, int]:
        starts: dict[int, int] = {}
        for row in rows:
            volume, start = int(row[0]), int(row[2])
            starts[volume] = min(start, starts.get(volume, start))
        return starts

    def _dataset(self, sniff: Sniff, name: str, header: dict, rows: list[list[str]],
                 volumes: dict) -> PhysioData:
        kind = header.get("LogDataType", "")
        notes: list[str] = []
        if "_cut" in header:
            notes.append(f"last line cut off in the file, not read: {header.pop('_cut')!r}")
        meta = Meta({"header": header, "file": name})
        stem = Path(name).stem.rsplit("_", 1)[-1] if "_" in name else Path(name).stem
        if kind == "ACQUISITION_INFO":
            table = np.array([[float(v) for v in row[:5]] for row in rows]).reshape(-1, 5)
            starts = self._volume_starts(rows)
            first_rows = [next(i for i, r in enumerate(rows) if int(r[0]) == v and int(r[2]) == s)
                          for v, s in sorted(starts.items())]
            start = table[0, 2] * self.TIC if len(table) else None
            return PhysioData(
                table, self.INFO_COLUMNS, None, start, "s since midnight, scanner clock (tics)",
                {"volume": np.asarray(first_rows, dtype=np.int64)}, meta,
                self.provenance(sniff, tuple(notes)), stem,
            )
        step = int(header.get("SampleTime", "1") or 1)
        tics = np.array([int(r[0]) for r in rows], dtype=np.int64)
        if len(tics) == 0:
            raise ReaderError(f"{name}: no samples")
        phase = int(np.bincount(tics % step).argmax())  # the grid most rows lie on
        first = int(tics.min()) - (int(tics.min()) - phase) % step
        offset = (tics - first) / step
        index = np.ceil(offset - 0.5).astype(np.int64)  # off-grid rows to the nearest point
        names = list(dict.fromkeys(r[1] for r in rows))
        channels = tuple(self.CHANNELS.get(n, n.lower()) for n in names)
        data = np.full((int(index.max()) + 1, len(names)), np.nan)
        events: dict[str, list[int]] = {}
        for row, i in zip(rows, index, strict=True):
            data[i, names.index(row[1])] = float(row[2])
            if len(row) > 3:
                events.setdefault(row[3].lower(), []).append(int(i))
        missing = int(np.isnan(data).any(axis=1).sum())
        if missing:
            notes.append(f"{missing} samples missing in the log, NaN")
        off = int(((tics - first) % step != 0).sum())
        if off:
            notes.append(f"{off} rows off the SampleTime grid, placed at the nearest grid point")
        pairs = {(int(i), r[1]) for i, r in zip(index, rows, strict=True)}
        if len(pairs) < len(rows):
            notes.append(f"{len(rows) - len(pairs)} rows on an occupied grid point; the last kept")
        uuid = header.get("UUID")
        if uuid in volumes:
            starts = np.array(sorted(volumes[uuid].values()), dtype=np.int64)
            at = np.ceil((starts - first) / step).astype(np.int64)
            inside = (at >= 0) & (at < data.shape[0])
            if not inside.all():
                notes.append(f"{int((~inside).sum())} volume starts outside the samples")
            events["volume"] = at[inside]
        notes.append(f"sampling frequency {1 / (step * self.TIC):g} Hz: SampleTime {step} tics "
                     f"of 2.5 ms")
        return PhysioData(
            data, channels, 1 / (step * self.TIC), first * self.TIC,
            "s since midnight, scanner clock (tics)",
            {k: np.unique(np.asarray(v, dtype=np.int64)) for k, v in events.items()}, meta,
            self.provenance(sniff, tuple(notes)), stem,
        )


#**************************************************************************************************#
#                                      PhilipsPhyslogReader                                        #
#**************************************************************************************************#
#                                                                                                  #
# A SCANPHYSLOG file has "##" lines (site, release, date, version), a "#" line naming the columns  #
# (v1raw v2raw v1 v2 ppu resp gx gy gz mark, from version 2 also mark2) and one row per sample;    #
# the mark columns are hexadecimal flags, kept as events per bit (0x0001 ECG, 0x0002 PPU and       #
# 0x0004 respiratory trigger, 0x0008 measurement, 0x0010 scan start, 0x0020 scan end, 0x0040       #
# external trigger, 0x0080 calibration, 0x0100 manual start, 0x0200 volume, 0x8000 reference ECG   #
# trigger). The file does not state its sampling frequency; it is 500 Hz (2 ms): the volume        #
# markers of the test data are 658 samples apart at a TR of 1.317 s. ppu and resp are named        #
# cardiac and respiratory.                                                                         #
#                                                                                                  #
#**************************************************************************************************#
class PhilipsPhyslogReader(Reader):
    FORMATS = (Format.PHILIPS_PHYSLOG,)
    PACKAGES = ("miox",)
    RATE = 500.0
    BITS = {0x0001: "ecg_trigger", 0x0002: "ppu_trigger", 0x0004: "resp_trigger",
            0x0008: "measurement", 0x0010: "scan_start", 0x0020: "scan_end",
            0x0040: "external_trigger", 0x0080: "calibration", 0x0100: "manual_start",
            0x0200: "volume", 0x8000: "reference_ecg_trigger"}
    NAMES = {"ppu": "cardiac", "resp": "respiratory"}

    def read(self, sniff: Sniff) -> list[PhysioData]:
        header, columns, rows = [], None, []
        with open(sniff.path, errors="replace") as fh:
            for line in fh:
                if line.startswith("##"):
                    header.append(line[2:].strip())
                elif line.startswith("#"):
                    names = line[1:].split()
                    if names:
                        columns = names
                elif line.strip():
                    rows.append(line.split())
        if columns is None or not rows:
            raise ReaderError(f"{sniff.path.name}: no column line or no samples")
        marks = [i for i, c in enumerate(columns) if c.startswith("mark")]
        values = [i for i in range(len(columns)) if i not in marks]
        width = len(columns)
        rows = [r for r in rows if len(r) == width]
        data = np.array([[float(r[i]) for i in values] for r in rows])
        events: dict[str, list[int]] = {}
        for i in marks:
            flags = np.array([int(r[i], 16) for r in rows])
            for bit in range(16):
                hits = np.flatnonzero(flags & (1 << bit))
                if len(hits) == 0:
                    continue
                if columns[i] == "mark":  # the meaning of further mark columns is not known
                    events[self.BITS.get(1 << bit, f"mark_0x{1 << bit:04x}")] = hits
                else:
                    events[f"{columns[i]}_0x{1 << bit:04x}"] = hits
        meta = Meta({"header": header, "columns": columns})
        notes = ["sampling frequency 500 Hz: not stated in the file"]
        channels = tuple(self.NAMES.get(columns[i], columns[i]) for i in values)
        return [PhysioData(
            data, channels, self.RATE, None, "", {k: np.asarray(v) for k, v in events.items()},
            meta, self.provenance(sniff, tuple(notes)), sniff.path.stem,
        )]


#**************************************************************************************************#
#                                         GePhysioReader                                           #
#**************************************************************************************************#
#                                                                                                  #
# GE writes PPGData_*, RESPData_* and ECGData_* files (with or without .gep) holding one integer   #
# per line and nothing else. The sampling frequencies are GE's fixed ones (PPG 100 Hz, respiration #
# 25 Hz, ECG 1000 Hz) and the log starts 30 s before the scan, as phys2bids takes them; the file   #
# states neither.                                                                                  #
#                                                                                                  #
#**************************************************************************************************#
class GePhysioReader(Reader):
    FORMATS = (Format.GE_PHYSIO,)
    PACKAGES = ("miox",)
    KINDS = {"PPG": ("cardiac", 100.0), "RESP": ("respiratory", 25.0), "ECG": ("ecg", 1000.0)}

    def read(self, sniff: Sniff) -> list[PhysioData]:
        match = re.match(r"(PPG|RESP|ECG)Data_", sniff.path.name)
        if match is None:
            raise ReaderError(f"{sniff.path.name}: not a GE PPGData, RESPData or ECGData file")
        channel, rate = self.KINDS[match.group(1)]
        data = np.loadtxt(sniff.path, ndmin=1).astype(np.float64)[:, None]
        notes = [f"sampling frequency {rate:g} Hz and start 30 s before the scan: GE's fixed "
                 f"values, not stated in the file"]
        return [PhysioData(
            data, (channel,), rate, -30.0, "s relative to the scan start", {}, Meta({}),
            self.provenance(sniff, tuple(notes)), sniff.path.name.removesuffix(".gep"),
        )]


#**************************************************************************************************#
#                                        BidsPhysioReader                                          #
#**************************************************************************************************#
#                                                                                                  #
# A BIDS physio recording: a tab-separated table (.tsv or .tsv.gz, no header line, n/a as NaN)     #
# with a JSON sidecar of the same name giving Columns, SamplingFrequency and StartTime (seconds    #
# relative to the first volume of the scan); further sidecar fields go to the vendor header.       #
#                                                                                                  #
#**************************************************************************************************#
class BidsPhysioReader(Reader):
    FORMATS = (Format.BIDS_PHYSIO,)
    PACKAGES = ("miox",)

    def read(self, sniff: Sniff) -> list[PhysioData]:
        sidecar = self.sidecar(sniff.path)
        if sidecar is None:
            raise ReaderError(f"{sniff.path.name}: no JSON sidecar")
        info = json.loads(sidecar.read_text())
        opener = gzip.open if sniff.path.suffix == ".gz" else open
        with opener(sniff.path, "rt") as fh:
            rows = [line.split() for line in fh if line.strip()]
        data = np.array([[np.nan if v == "n/a" else float(v) for v in r] for r in rows])
        columns = tuple(info.get("Columns", [f"column{i}" for i in range(data.shape[1])]))
        notes = []
        if data.ndim != 2 or data.shape[1] != len(columns):
            raise ReaderError(f"{sniff.path.name}: {data.shape} values for columns {columns}")
        rate = info.get("SamplingFrequency")
        start = info.get("StartTime")
        meta = Meta({"sidecar": info})
        return [PhysioData(
            data, columns, float(rate) if rate else None, float(start) if start is not None
            else None, "s relative to the first volume (BIDS StartTime)", {}, meta,
            self.provenance(sniff, tuple(notes)), self.stem(sniff.path),
        )]

    @staticmethod
    def stem(path: Path) -> str:
        return path.name.removesuffix(".gz").removesuffix(".tsv")

    @classmethod
    def sidecar(cls, path: Path) -> Path | None:
        candidate = path.with_name(cls.stem(path) + ".json")
        return candidate if candidate.is_file() else None
