####################################################################################################
#                                        ge_scanarchive.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 06/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for GE ScanArchive files (.h5): the rdb header the archive carries, every        #
#          acquired frame with the control packet that names its slice, echo and view, the saved   #
#          scanner files, and the archive's own metadata.                                          #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import base64
import re
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import h5py
import numpy as np

# own
from miox.data import RawData
from miox.readers.base import Reader, ReaderError
from miox.readers.ge_rdb import GeRdbHeader
from miox.readers.ismrmrd import IsmrmrdReader
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                       GeScanArchiveReader                                        #
#**************************************************************************************************#
#                                                                                                  #
# A ScanArchive is an HDF5 container. ``DownloadData.xml`` holds the rdb header (the P-file header #
# of the scan) base64-encoded as its pool header buffer; ``Header.xml``, ``ArchiveMetaData`` and   #
# ``Acquisition/StorageMetaData.xml`` describe the archive; further files the scanner saved (coil  #
# tables, EPI reference scans, row-flip tables) sit under their original paths. The acquisition    #
# blocks hold a stream of elements, each starting with its byte size and a time stamp in µs:       #
# control packets of 44 bytes (opcode at byte 8, echo at byte 10, slice number and view number as  #
# big-endian byte pairs at bytes 15-16 and 19-20, the packet index at byte 40), data frames (a     #
# 48-byte header with the channel count at byte 32, the sample count at byte 36 and the frame      #
# index at byte 40, then the samples of every channel as integer pairs, then six bytes per         #
# channel) and XML updates. A control packet with opcode 1 names one frame; a hyper-frame packet   #
# (opcode 6, EPI) names the block of frames whose count is at byte 20, with the slice number at    #
# bytes 13-14; the other opcodes (1024, 3072) carry no frames. The frames become one RawData in    #
# file order (readouts x coils x samples) with headers in the ISMRMRD layout as the P-file reader  #
# fills them (the packet's slice number is the acquisition index, kept in the second user counter; #
# the slice counter is the spatial index from the data acquisition table), the frame time stamp as #
# the acquisition time stamp and the packet words in the user integers; frames of view 0 form a    #
# baseline stream.                                                                                 #
#                                                                                                  #
#**************************************************************************************************#
class GeScanArchiveReader(Reader):
    FORMATS = (Format.GE_SCANARCHIVE,)
    PACKAGES = ("miox", "h5py")
    SOURCE = "ScanArchive"
    FLAGS = IsmrmrdReader.ACQUISITION_FLAGS
    POOL = re.compile(r"<PoolHeaderBuffer[^>]*>\s*<Size>(\d+)</Size>\s*<Data>\s*([^<]+)</Data>")
    BLOCK = re.compile(r"_Block(\d+)_Segment(\d+)$")
    CONTROL_SIZE = 44
    FRAME_HEADER = 48
    CHUNK = 1 << 24  # bytes read from a block at a time
    OPCODE_FRAME = 1
    OPCODE_HYPER = 6
    FIRST_IN_SLICE = 7  # flag bits as in ismrmrd.h
    LAST_IN_SLICE = 8
    DESCRIPTIONS = {  # the archive's own XML documents: vendor header key
        "Header.xml": "header", "ArchiveMetaData/ArchivedFiles.xml": "archived_files",
        "Acquisition/StorageMetaData.xml": "storage",
    }
    GEOMETRY = ("position", "read_dir", "phase_dir", "slice_dir")

    def read(self, sniff: Sniff) -> list[RawData]:
        notes: list[str] = []
        with h5py.File(sniff.path, "r") as fh:
            files = self._files(fh)
            packets, frames, updates = self._elements(fh["Acquisition"], notes)
        header = self._header(files, sniff.path)
        point_size = header.values.get("rhr_rh_point_size") or 2
        if not frames:
            raise ReaderError(f"{sniff.path.name}: no data frames in the acquisition blocks")
        assignment = self._assign(packets, len(frames), notes)
        geometry = header.geometry()
        if geometry is None:
            notes.append("no slice geometry: the header does not locate the data acquisition table")
        vendor = self._vendor(header, files, packets, frames, updates)
        meta = header.meta()
        meta.vendor = vendor
        meta.fields = {key: "rdb/" + field for key, field in meta.fields.items()}
        self._archive_keys(meta, vendor.get("header", {}))
        views = header.values.get("rhr_rh_nframes") or 0
        datasets = []
        for name, keep in ((sniff.path.stem, False), (sniff.path.stem + "_baseline", True)):
            chosen = [
                i for i, (p, _) in enumerate(assignment) if self._is_baseline(packets, p) == keep
            ]
            if not chosen:
                continue
            data, headers = self._readouts(
                frames, packets, assignment, chosen, point_size, header, geometry, views
            )
            encoding = {
                "readout_order": "file order", "frames": len(chosen), "coils": int(data.shape[1]),
                "frame_size": int(data.shape[2]), "point_size": point_size,
                "control_packets": len(packets),
            }
            provenance = self.provenance(sniff, tuple(notes))
            datasets.append(
                RawData(data, headers, meta, provenance, name, None, None, encoding, [], self.FLAGS)
            )
        return datasets

    # --- container -------------------------------------------------------------------------------

    def _files(self, fh: h5py.File) -> dict[str, Any]:
        """Every dataset outside the acquisition blocks: text as str, the rest as bytes."""
        files: dict[str, Any] = {}

        def visit(name: str, obj) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            if name.startswith("Acquisition/") and not name.endswith(".xml"):
                return
            value = obj[()]
            blob = value[0] if value.shape == (1,) and value.dtype.kind == "S" else value.tobytes()
            files[name] = self._text_or_bytes(blob)

        fh.visititems(visit)
        return files

    @staticmethod
    def _text_or_bytes(blob: bytes) -> str | bytes:
        try:
            text = blob.decode("utf-8")
        except UnicodeDecodeError:
            return blob
        return blob if "\0" in text else text

    def _header(self, files: dict[str, Any], path: Path) -> GeRdbHeader:
        match = self.POOL.search(files.get("DownloadData.xml", ""))
        if match is None:
            raise ReaderError(f"{path.name}: DownloadData.xml holds no pool header buffer")
        raw = base64.b64decode(match.group(2))
        if len(raw) != int(match.group(1)):
            raise ReaderError(
                f"{path.name}: pool header of {len(raw)} bytes, {match.group(1)} stated"
            )
        return GeRdbHeader(raw)

    def _elements(
        self, group: h5py.Group, notes: list[str]
    ) -> tuple[list[bytes], list[bytes], list[str]]:
        """The control packets, data frames and XML updates of the acquisition blocks in order."""
        blocks = []
        for name, dataset in group.items():
            match = self.BLOCK.search(name)
            if match and isinstance(dataset, h5py.Dataset):
                blocks.append((int(match.group(1)), int(match.group(2)), dataset))
        packets, frames, updates = [], [], []
        for _, _, dataset in sorted(blocks, key=lambda b: b[:2]):
            for element in self._walk(dataset):
                if len(element) == self.CONTROL_SIZE:
                    packets.append(element)
                elif element[8:14].lstrip().startswith(b"<?xml"):
                    updates.append(element[8:].rstrip(b"\0").decode("utf-8", "replace"))
                elif len(element) > self.FRAME_HEADER:
                    frames.append(element)
                else:
                    notes.append(f"{dataset.name}: element of {len(element)} bytes passed over")
        return packets, frames, updates

    def _walk(self, dataset: h5py.Dataset):
        buffer = bytearray()
        position, total = 0, dataset.shape[0]
        while True:
            while len(buffer) >= 4:
                size = struct.unpack_from("<I", buffer, 0)[0]
                if size == 0:
                    return
                if size > len(buffer):
                    break
                yield bytes(buffer[:size])
                del buffer[:size]
            if position >= total:
                return
            buffer += dataset[position : position + self.CHUNK].tobytes()
            position += self.CHUNK

    # --- packets and frames ----------------------------------------------------------------------

    @classmethod
    def packet(cls, raw: bytes) -> dict[str, int]:
        words = struct.unpack_from("<22H", raw, 0)
        opcode = words[4]
        hyper = opcode == cls.OPCODE_HYPER
        return {
            "opcode": opcode, "time": struct.unpack_from("<I", raw, 4)[0], "echo": raw[10],
            "slice": (raw[13] << 8 | raw[14]) if hyper else (raw[15] << 8 | raw[16]),
            "view": raw[19] << 8 | raw[20], "frames": words[10] if hyper else 0,
            "index": struct.unpack_from("<I", raw, 40)[0], "words": list(words),
        }

    def _assign(
        self, packets: list[bytes], n_frames: int, notes: list[str]
    ) -> list[tuple[int | None, int]]:
        """For every frame the index of its control packet and its position in that packet."""
        counts = []
        for p in (self.packet(raw) for raw in packets):
            per_opcode = {self.OPCODE_FRAME: 1, self.OPCODE_HYPER: p["frames"]}
            counts.append(per_opcode.get(p["opcode"], 0))
        if sum(counts) != n_frames:
            notes.append(f"control packets name {sum(counts)} frames, {n_frames} stored")
        assignment: list[tuple[int | None, int]] = []
        for i, count in enumerate(counts):
            for position in range(count):
                if len(assignment) < n_frames:
                    assignment.append((i, position))
        while len(assignment) < n_frames:
            assignment.append((None, 0))
        return assignment

    def _is_baseline(self, packets: list[bytes], index: int | None) -> bool:
        if index is None:
            return False
        p = self.packet(packets[index])
        return p["opcode"] == self.OPCODE_FRAME and p["view"] == 0

    def _readouts(
        self, frames: list[bytes], packets: list[bytes], assignment: list, chosen: list[int],
        point_size: int, header: GeRdbHeader, geometry: list[dict] | None, views: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        heads = [struct.unpack_from("<12I", frames[i], 0) for i in chosen]
        coils = max(h[8] for h in heads)
        samples = max(h[9] for h in heads)
        dtype = "<i2" if point_size == 2 else "<i4"
        real = np.float32 if point_size == 2 else np.float64  # 32-bit integers exactly
        complex_type = np.complex64 if point_size == 2 else np.complex128
        data = np.zeros((len(chosen), coils, samples), dtype=complex_type)
        headers = np.zeros(len(chosen), dtype=RawData.HEADER)
        headers["measurement_uid"] = header.values.get("rhi_rawrunnum") or 0
        geometric = header.geometric_slices() if geometry else {}
        repetitions: dict[int, int] = {}
        decoded = [self.packet(p) for p in packets]
        for row, i in enumerate(chosen):
            head, element = heads[row], frames[i]
            channels, length = head[8], head[9]
            count = channels * length * 2
            values = np.frombuffer(element, dtype, count, self.FRAME_HEADER).astype(real)
            values = values.reshape(channels, length, 2)
            data[row, :channels, :length] = values[..., 0] + 1j * values[..., 1]
            h = headers[row]
            h["scan_counter"], h["acquisition_time_stamp"] = head[10], head[1]
            h["number_of_samples"] = length
            h["available_channels"], h["active_channels"] = channels, channels
            h["center_sample"] = length // 2
            index, position = assignment[i]
            if index is None:
                continue
            p = decoded[index]
            k = geometric.get((0, p["slice"]))
            h["idx"]["slice"] = p["slice"] if k is None else k
            h["idx"]["user"][1] = p["slice"]
            h["user_int"] = [p["opcode"], p["index"], *p["words"][6:11], p["echo"]]
            if p["opcode"] == self.OPCODE_FRAME:
                h["idx"]["contrast"] = p["echo"]
                if p["view"]:
                    h["idx"]["kspace_encode_step_1"] = p["view"] - 1
                    flags = (1 << (self.FIRST_IN_SLICE - 1)) if p["view"] == 1 else 0
                    flags |= (1 << (self.LAST_IN_SLICE - 1)) if p["view"] == views else 0
                    h["flags"] = flags
            elif p["opcode"] == self.OPCODE_HYPER:
                h["idx"]["kspace_encode_step_1"] = position
                if position == 0:
                    repetitions[p["slice"]] = repetitions.get(p["slice"], -1) + 1
                h["idx"]["repetition"] = repetitions.get(p["slice"], 0)
            if k is not None:
                for field in self.GEOMETRY:
                    h[field] = geometry[k][field]
        return data, headers

    # --- metadata --------------------------------------------------------------------------------

    def _vendor(self, header, files, packets, frames, updates) -> dict[str, Any]:
        vendor: dict[str, Any] = {"rdb": dict(header.values), "slice_table": header.slice_table()}
        for name, key in self.DESCRIPTIONS.items():
            if name in files:
                vendor[key] = self.xml_dict(files[name])
        vendor["files"] = {
            k: v for k, v in files.items() if k not in self.DESCRIPTIONS and k != "DownloadData.xml"
        }
        vendor["control_packets"] = [self.packet(p)["words"] for p in packets]
        vendor["frame_headers"] = [list(struct.unpack_from("<12I", f, 0)) for f in frames]
        vendor["updates"] = updates
        return vendor

    def _archive_keys(self, meta, header: dict) -> None:
        px = header.get("Header", {}).get("px", {}) if isinstance(header, dict) else {}
        when = px.get("DateTime", {})
        date = str(when.get("ptime_date", {}).get("date", ""))
        time = when.get("ptime_time_duration", {})
        if len(date) == 8 and "time_duration_hours" in time:
            stamp = (
                f"{date[:4]}-{date[4:6]}-{date[6:]}T{int(time['time_duration_hours']):02d}:"
                f"{int(time['time_duration_minutes']):02d}:{int(time['time_duration_seconds']):02d}"
            )
            fraction = int(time.get("time_duration_fractional_seconds", 0))
            if fraction:
                stamp += f".{fraction // 1000:06d}"
            field = "header/Header/px/DateTime"
            meta.set("AcquisitionDateTime", stamp, f"{self.SOURCE}: {field}", field)
        label = px.get("SystemID", {}).get("SoftwareLabel")
        if label:
            field = "header/Header/px/SystemID/SoftwareLabel"
            meta.set("SoftwareVersions", str(label), f"{self.SOURCE}: {field}", field)

    @classmethod
    def xml_dict(cls, text: str) -> Any:
        """A boost-serialization XML document (its root tag is left unclosed) as nested dicts:
        the serialization root and the attributes dropped, repeated tags as lists, numbers as
        numbers; the text itself when it does not parse."""
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            match = re.search(r"<(boost_serialization)\b", text)
            if match is None:
                return text
            try:
                root = ET.fromstring(text + f"</{match.group(1)}>")
            except ET.ParseError:
                return text
        if root.tag == "boost_serialization":
            return cls._node(root)
        return {root.tag: cls._node(root)}

    @classmethod
    def _node(cls, node: ET.Element) -> Any:
        children = list(node)
        if not children:
            return cls._scalar(node.text)
        out: dict[str, Any] = {}
        for child in children:
            value = cls._node(child)
            if child.tag in out:
                if not isinstance(out[child.tag], list):
                    out[child.tag] = [out[child.tag]]
                out[child.tag].append(value)
            else:
                out[child.tag] = value
        return out

    @staticmethod
    def _scalar(text: str | None) -> Any:
        if text is None:
            return ""
        text = text.strip()
        try:
            return int(text)
        except ValueError:
            pass
        try:
            return float(text)
        except ValueError:
            return text
