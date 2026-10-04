####################################################################################################
#                                          bruker_2dseq.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 04/10/26                                                                                #
#                                                                                                  #
# Purpose: A processed Bruker ParaVision data set (pdata/<n>/2dseq) read through its visu_pars:    #
#          frames, their grouping, scaling, geometry and the spectroscopic axis.                   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


#**************************************************************************************************#
#                                           Bruker2dseq                                            #
#**************************************************************************************************#
#                                                                                                  #
# The 2dseq file holds VisuCoreFrameCount frames of VisuCoreSize values each, the first size       #
# fastest, scaled by VisuCoreDataSlope and VisuCoreDataOffs. Frames are ordered by the frame       #
# groups of VisuFGOrderDesc, the first group fastest (slices, echoes, cycles, real/imaginary       #
# parts, ...). Each frame's orientation (direction cosines of the read, phase and slice axes) and  #
# position (centre of the first voxel) are in patient coordinates, the same values the vendor      #
# writes into ImageOrientationPatient and ImagePositionPatient of its DICOM export; the extent     #
# over the matrix size is the voxel spacing. Spectroscopic dimensions carry their width in ppm.    #
#                                                                                                  #
#**************************************************************************************************#
class Bruker2dseq:
    WORD_TYPES = {
        "_8BIT_UNSGN_INT": "u1",
        "_16BIT_SGN_INT": "i2",
        "_32BIT_SGN_INT": "i4",
        "_32BIT_FLOAT": "f4",
        "_64BIT_FLOAT": "f8",
    }

    def __init__(self, path: Path, visu: dict):
        self.path = path
        self.visu = visu
        self.size = [int(v) for v in np.atleast_1d(visu["VisuCoreSize"])]
        self.dim_desc = [str(v) for v in np.atleast_1d(visu.get("VisuCoreDimDesc", "spatial"))]
        self.dim_desc += ["spatial"] * (len(self.size) - len(self.dim_desc))
        self.frame_count = int(visu.get("VisuCoreFrameCount", 1))
        self.frame_types = [str(v) for v in np.atleast_1d(visu.get("VisuCoreFrameType", ""))]
        self.extent = [float(v) for v in np.atleast_1d(visu.get("VisuCoreExtent", self.size))]
        self.units = [str(v) for v in np.atleast_1d(visu.get("VisuCoreUnits", ""))]

    # --- data ------------------------------------------------------------------------------------

    def frames(self) -> np.ndarray:
        """All frames, scaled, as (frame, size[-1], ..., size[0])."""
        word = str(self.visu.get("VisuCoreWordType", "_16BIT_SGN_INT"))
        little = str(self.visu.get("VisuCoreByteOrder", "littleEndian")) == "littleEndian"
        order = "<" if little else ">"
        dtype = np.dtype(order + self.WORD_TYPES[word])
        raw = np.fromfile(self.path, dtype=dtype)
        expected = self.frame_count * int(np.prod(self.size))
        if raw.size != expected:
            raise ValueError(f"{self.path}: {raw.size} values, expected {expected}")
        raw = raw.reshape((self.frame_count, *self.size[::-1])).astype(np.float64)
        slope = self._per_frame("VisuCoreDataSlope", 1.0)
        offset = self._per_frame("VisuCoreDataOffs", 0.0)
        shape = (self.frame_count,) + (1,) * len(self.size)
        return raw * slope.reshape(shape) + offset.reshape(shape)

    def _per_frame(self, name: str, default: float) -> np.ndarray:
        values = np.atleast_1d(np.asarray(self.visu.get(name, default), dtype=float)).ravel()
        if values.size == 1:
            values = np.repeat(values, self.frame_count)
        return values[: self.frame_count]

    # --- frame groups ----------------------------------------------------------------------------

    def groups(self) -> list[tuple[str, int]]:
        """(name, count) of the frame groups, the first varying fastest; e.g. [("FG_SLICE", 9)]."""
        desc = self.visu.get("VisuFGOrderDesc")
        if desc is None:
            return [] if self.frame_count == 1 else [("FG_FRAME", self.frame_count)]
        desc = list(desc)
        if desc and not isinstance(desc[0], (list, tuple)):
            desc = [desc]
        return [(str(d[1]).strip("<>"), int(d[0])) for d in desc]

    def group_index(self, name: str) -> int | None:
        names = [n for n, _ in self.groups()]
        return names.index(name) if name in names else None

    def frame_indices(self) -> np.ndarray:
        """Frame numbers arranged by group, shape (count of last group, ..., count of first)."""
        counts = [c for _, c in self.groups()] or [1]
        return np.arange(self.frame_count).reshape(counts[::-1])

    def slice_packs(self) -> list[tuple[int, int]]:
        """(first slice, slice count) of each slice pack; one pack with all slices by default."""
        packs = self.visu.get("VisuCoreSlicePacksSlices")
        slices = next((c for n, c in self.groups() if n == "FG_SLICE"), 1)
        if packs is None:
            return [(0, slices)]
        packs = np.atleast_2d(np.asarray(packs, dtype=int))
        return [(int(start), int(count)) for start, count in packs]

    # --- geometry --------------------------------------------------------------------------------

    def orientation(self, frame: int) -> np.ndarray:
        """Direction cosines (3, 3): rows are the read, phase and slice directions."""
        values = np.asarray(self.visu.get("VisuCoreOrientation", np.eye(3).ravel()), dtype=float)
        values = values.reshape(-1, 9)
        row = values[frame] if values.shape[0] > frame else values[0]
        return row.reshape(3, 3)

    def position(self, frame: int) -> np.ndarray:
        values = np.asarray(self.visu.get("VisuCorePosition", np.zeros(3)), dtype=float)
        values = values.reshape(-1, 3)
        return values[frame] if values.shape[0] > frame else values[0]

    def transposed(self, frame: int) -> bool:
        values = self.visu.get("VisuCoreTransposition")
        if values is None:
            return False
        values = np.atleast_1d(np.asarray(values, dtype=int))
        return bool(values[frame] if values.size > frame else values[0])

    def spacing(self) -> list[float]:
        """Voxel spacing along the stored dimensions (ppm per point for a spectroscopic one)."""
        return [extent / size for extent, size in zip(self.extent, self.size, strict=True)]

    def thickness(self) -> float:
        return float(np.atleast_1d(self.visu.get("VisuCoreFrameThickness", 1.0))[0])

    def spectral_axis(self) -> int | None:
        """Index into ``size`` of the spectroscopic dimension, None for an image."""
        return next((i for i, d in enumerate(self.dim_desc) if d == "spectroscopic"), None)

    @property
    def spatial_dims(self) -> int:
        return sum(1 for d in self.dim_desc if d == "spatial")

    def value(self, name: str, default: Any = None) -> Any:
        return self.visu.get(name, default)
