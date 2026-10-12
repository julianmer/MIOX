####################################################################################################
#                                         tractograms.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 12/10/26                                                                                #
#                                                                                                  #
# Purpose: Readers for streamline files: TrackVis .trk and MRtrix .tck (read by nibabel) and TRX   #
#          (.trx zip archives or folders).                                                         #
#                                                                                                  #
####################################################################################################


#*************#
# imports                                                                                          #
#*************#
from __future__ import annotations

import json
import re
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
