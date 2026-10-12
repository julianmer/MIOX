####################################################################################################
#                                         tractograms.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 12/10/26                                                                                #
#                                                                                                  #
# Purpose: Readers for streamline files: TrackVis .trk and MRtrix .tck (read by nibabel), TRX     #
#          (.trx zip archives or folders), DSI Studio .tt and MRtrix track scalar files (.tsf).    #
#                                                                                                  #
####################################################################################################


#*************#
# imports                                                                                          #
#*************#
from __future__ import annotations

import gzip
import json
import re
import struct
import zipfile
from pathlib import PurePosixPath

import nibabel as nib
import numpy as np

# own
from miox.data import Tractogram
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff

RAS = "RAS world (mm)"


def concatenated(arrays) -> np.ndarray:
    """The per-streamline arrays of one property as one array with a row per point."""
    arrays = list(arrays)
    return np.concatenate(arrays) if arrays else np.zeros((0,))


#**************************************************************************************************#
#                                          TrackReader                                             #
#**************************************************************************************************#
#                                                                                                  #
# TrackVis .trk and MRtrix .tck through nibabel.streamlines, which returns points in RAS world mm  #
# (TrackVis stores voxel mm in its voxel order; nibabel applies the header's vox_to_ras). A .trk   #
# states its reference image (dimensions, voxel sizes, voxel_to_rasmm) and named per-point scalars #
# and per-streamline properties; a .tck states no image, its header fields go to the vendor header.#
#                                                                                                  #
#**************************************************************************************************#
class TrackReader(Reader):
    FORMATS = (Format.TRK, Format.TCK)
    PACKAGES = ("miox", "nibabel")

    def read(self, sniff: Sniff) -> list[Tractogram]:
        notes: list[str] = []
        try:
            with self.captured(notes):
                loaded = nib.streamlines.load(sniff.path, lazy_load=False)
        except Exception as e:  # nibabel raises several types for malformed headers
            raise ReaderError(f"{sniff.path}: not a readable tractogram ({e})") from e
        tractogram = loaded.tractogram
        lengths = np.array([len(s) for s in tractogram.streamlines], dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        points = concatenated(tractogram.streamlines).reshape(-1, 3).astype(np.float64)
        point_data = {k: concatenated(v) for k, v in tractogram.data_per_point.items()}
        streamline_data = {k: np.asarray(v) for k, v in tractogram.data_per_streamline.items()}
        header = {k: self.plain(v) for k, v in loaded.header.items() if not k.startswith("_")}
        reference = None
        if sniff.format == Format.TRK:
            reference = {"dimensions": header["dimensions"], "voxel_sizes": header["voxel_sizes"],
                         "voxel_to_rasmm": header["voxel_to_rasmm"],
                         "voxel_order": header["voxel_order"]}
        meta = Meta({"header": header})
        stated = header.get("nb_streamlines", header.get("count"))
        if stated is not None and int(stated) not in (0, len(lengths)):
            notes.append(f"header states {int(stated)} streamlines, {len(lengths)} found")
        return [Tractogram(points, offsets, RAS, point_data, streamline_data, reference, meta,
                           self.provenance(sniff, tuple(notes)), sniff.path.stem)]


#**************************************************************************************************#
#                                           TrxReader                                              #
#**************************************************************************************************#
#                                                                                                  #
# A TRX tractogram is a zip archive (stored or deflated) or a folder holding header.json           #
# (DIMENSIONS, VOXEL_TO_RASMM, NB_VERTICES, NB_STREAMLINES), positions.3.<dtype> (points in RAS    #
# world mm), offsets.<dtype> (the first point of each streamline; NB_STREAMLINES + 1 values with   #
# the closing NB_VERTICES in current files, NB_STREAMLINES in older ones) and optional folders:    #
# dpv/ (one row per point), dps/ (one row per streamline), groups/ (streamline indices per group,  #
# stored as membership masks "group/<name>"), dpg/<group>/ (one row per group, vendor header).     #
# Array files are named <name>[.<columns>].<dtype>, little endian.                                 #
#                                                                                                  #
#**************************************************************************************************#
class TrxReader(Reader):
    FORMATS = (Format.TRX,)
    PACKAGES = ("miox",)
    ARRAY = re.compile(r"^(?P<name>.+?)(\.(?P<columns>\d+))?\.(?P<dtype>(u?int|float)\d+)$")

    def read(self, sniff: Sniff) -> list[Tractogram]:
        path = sniff.path
        try:
            files = self._files(sniff)
            header = json.loads(files.pop("header.json"))
        except (OSError, KeyError, zipfile.BadZipFile, json.JSONDecodeError) as e:
            raise ReaderError(f"{path}: not a readable TRX tractogram ({e})") from e
        arrays = {}
        for name, raw in files.items():
            parts = PurePosixPath(name)
            match = self.ARRAY.match(parts.name)
            if match is None:
                continue
            columns = int(match["columns"] or 1)
            values = np.frombuffer(raw, np.dtype(match["dtype"]).newbyteorder("<"))
            if values.size % columns:
                raise ReaderError(f"{path}: {name} does not hold rows of {columns} values")
            values = values.reshape(-1, columns) if match["columns"] else values
            arrays[(str(parts.parent), match["name"])] = values
        points = arrays.pop((".", "positions"), None)
        offsets = arrays.pop((".", "offsets"), None)
        if points is None or offsets is None:
            raise ReaderError(f"{path}: positions or offsets missing")
        n_points, n_streamlines = int(header["NB_VERTICES"]), int(header["NB_STREAMLINES"])
        offsets = offsets.astype(np.int64)
        if len(offsets) == n_streamlines:
            offsets = np.append(offsets, n_points)
        if points.shape != (n_points, 3) or len(offsets) != n_streamlines + 1:
            raise ReaderError(f"{path}: {points.shape[0]} points and {len(offsets) - 1} "
                              f"streamlines, header states {n_points} and {n_streamlines}")
        point_data, streamline_data, group_data = {}, {}, {}
        for (folder, name), values in arrays.items():
            if folder == "dpv":
                point_data[name] = values
            elif folder == "dps":
                streamline_data[name] = values
            elif folder == "groups":
                mask = np.zeros(n_streamlines, dtype=bool)
                mask[values.astype(np.int64)] = True
                streamline_data[f"group/{name}"] = mask
            elif folder.startswith("dpg/"):
                group_data.setdefault(folder.removeprefix("dpg/"), {})[name] = values
        reference = {"dimensions": header.get("DIMENSIONS"),
                     "voxel_to_rasmm": header.get("VOXEL_TO_RASMM")}
        vendor = {"header": header, "dpg": self.plain(group_data)}
        return [Tractogram(points.astype(np.float64), offsets, RAS, point_data, streamline_data,
                           reference, Meta(vendor), self.provenance(sniff),
                           path.name.removesuffix(".trx"))]

    @staticmethod
    def _files(sniff: Sniff) -> dict[str, bytes]:
        """Every file of the tractogram by its path inside the archive or folder."""
        if sniff.path.is_dir():
            return {p.relative_to(sniff.path).as_posix(): p.read_bytes()
                    for p in sorted(sniff.path.rglob("*")) if p.is_file()}
        with zipfile.ZipFile(sniff.path) as archive:
            return {name: archive.read(name) for name in archive.namelist()
                    if not name.endswith("/")}


def matlab_v4(raw: bytes) -> dict[str, np.ndarray]:
    """The variables of a MATLAB version 4 file: per variable five int32 (type MOPT, rows,
    columns, imaginary flag, name length), the name and the values column by column."""
    precisions = {0: "f8", 1: "f4", 2: "i4", 3: "i2", 4: "u2", 5: "u1"}
    variables, position = {}, 0
    while position + 20 <= len(raw):
        kind, rows, columns, imaginary, length = struct.unpack_from("<5i", raw, position)
        if kind > 9999 or kind < 0:
            raise ReaderError(f"MATLAB v4 type {kind} at byte {position}: big-endian or not v4")
        order = "<" if kind // 1000 == 0 else ">"
        precision = precisions.get(kind // 10 % 10)
        if precision is None:
            raise ReaderError(f"MATLAB v4 precision {kind // 10 % 10} unknown")
        position += 20
        name = raw[position:position + length].rstrip(b"\0").decode("ascii", "replace")
        position += length
        count = rows * columns * (2 if imaginary else 1)
        dtype = np.dtype(order + precision)
        values = np.frombuffer(raw, dtype, count, position)
        position += count * dtype.itemsize
        if imaginary:
            values = values[:rows * columns] + 1j * values[rows * columns:]
        variables[name] = values.reshape(columns, rows).T
    return variables


#**************************************************************************************************#
#                                        DsiStudioTrackReader                                      #
#**************************************************************************************************#
#                                                                                                  #
# A DSI Studio .tt file (gzipped MATLAB v4) holds the grid (dimension, voxel_size), trans_to_mni   #
# (4 x 4, row by row) and the streamlines packed in the byte array "track": per streamline the     #
# byte count of its coordinates (uint32), the first point as three int32 and every further point   #
# as three int8 steps, all in 1/32 voxel; points are voxel centres. MIOX maps them through         #
# trans_to_mni. Other variables (report, colours, clusters) go to the vendor header.               #
#                                                                                                  #
#**************************************************************************************************#
class DsiStudioTrackReader(Reader):
    FORMATS = (Format.DSI_STUDIO_TT,)
    PACKAGES = ("miox",)

    def read(self, sniff: Sniff) -> list[Tractogram]:
        raw = sniff.path.read_bytes()
        try:
            variables = matlab_v4(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
            track = variables.pop("track").ravel().astype(np.uint8).tobytes()
            transform = variables.pop("trans_to_mni").ravel().astype(np.float64).reshape(4, 4)
        except (KeyError, OSError, EOFError, struct.error, ValueError) as e:
            raise ReaderError(f"{sniff.path}: not a readable DSI Studio track file ({e})") from e
        pieces, lengths, position = [], [], 0
        while position + 16 <= len(track):
            size = struct.unpack_from("<I", track, position)[0]
            first = np.array(struct.unpack_from("<3i", track, position + 4), dtype=np.int64)
            steps = np.frombuffer(track, np.int8, size - 3, position + 16).reshape(-1, 3)
            pieces.append(np.vstack([first, first + np.cumsum(steps, axis=0)]))
            lengths.append(size // 3)
            position += size + 13
        if position != len(track):
            raise ReaderError(f"{sniff.path}: track bytes end inside a streamline")
        voxels = np.concatenate(pieces) / 32.0 if pieces else np.zeros((0, 3))
        points = voxels @ transform[:3, :3].T + transform[:3, 3]
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        report = variables.pop("report", None)
        vendor = {k: self.plain(v.ravel() if v.size < 64 else v.shape)
                  for k, v in variables.items()}
        if report is not None:
            vendor["report"] = report.ravel().astype(np.uint8).tobytes().decode("utf-8", "replace")
        reference = {"dimensions": variables.get("dimension", np.zeros(0)).ravel().tolist(),
                     "voxel_sizes": variables.get("voxel_size", np.zeros(0)).ravel().tolist(),
                     "voxel_to_rasmm": transform.tolist()}
        name = sniff.path.name.removesuffix(".gz").removesuffix(".tt")
        return [Tractogram(points, offsets, "DSI Studio trans_to_mni (mm)", {}, {}, reference,
                           Meta(vendor), self.provenance(sniff), name)]


#**************************************************************************************************#
#                                            TsfReader                                             #
#**************************************************************************************************#
#                                                                                                  #
# An MRtrix track scalar file (.tsf) has a header like a .tck ("mrtrix track scalars", key: value  #
# lines to END, "file: . <offset>") and one value per point of the streamlines of a .tck: NaN      #
# ends a streamline, Inf the file. The result is a Tractogram without points (they live in the     #
# .tck the file was sampled on) whose point data holds the values.                                 #
#                                                                                                  #
#**************************************************************************************************#
class TsfReader(Reader):
    FORMATS = (Format.TSF,)
    PACKAGES = ("miox",)
    TYPES = {"float32le": "<f4", "float32be": ">f4", "float64le": "<f8", "float64be": ">f8"}

    def read(self, sniff: Sniff) -> list[Tractogram]:
        raw = sniff.path.read_bytes()
        end = raw.find(b"\nEND")
        if end < 0:
            raise ReaderError(f"{sniff.path}: header without END")
        header = {}
        for line in raw[:end].decode("ascii", "replace").splitlines()[1:]:
            key, sep, value = line.partition(":")
            if sep:
                header[key.strip()] = value.strip()
        dtype = self.TYPES.get(header.get("datatype", "").lower())
        try:
            offset = int(header["file"].split()[-1])
        except (KeyError, ValueError, IndexError) as e:
            raise ReaderError(f"{sniff.path}: no 'file: . <offset>' line") from e
        if dtype is None:
            raise ReaderError(f"{sniff.path}: datatype {header.get('datatype')} not read")
        values = np.frombuffer(raw, dtype, (len(raw) - offset) // np.dtype(dtype).itemsize,
                               offset).astype(np.float64)
        stop = np.flatnonzero(np.isinf(values))
        values = values[:stop[0]] if len(stop) else values
        breaks = np.flatnonzero(np.isnan(values))
        lengths = np.diff(np.concatenate([[-1], breaks])) - 1
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        notes = []
        if len(breaks) and breaks[-1] != len(values) - 1:
            notes.append("values after the last NaN dropped")
        if "count" in header and int(header["count"]) != len(lengths):
            notes.append(f"header states {header['count']} streamlines, {len(lengths)} found")
        return [Tractogram(None, offsets, RAS, {"values": values[~np.isnan(values)]
                                                [:offsets[-1]]}, {}, None, Meta({"header": header}),
                           self.provenance(sniff, tuple(notes)), sniff.path.stem)]
