####################################################################################################
#                                          transforms.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 10/10/26                                                                                #
#                                                                                                  #
# Purpose: Readers for linear spatial transforms: FreeSurfer LTA, ITK text (.tfm) and HDF5 (.h5)   #
#          transform files, AFNI affine matrices (.aff12.1D and 3dvolreg rows), FSL FLIRT          #
#          matrices.                                                                               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np

# own
from miox.data import Transform
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff

LPS = np.diag([-1.0, -1.0, 1.0, 1.0])  # flips x and y between LPS (DICOM) and RAS


def affine(linear: np.ndarray, offset: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = linear, offset
    return out


#**************************************************************************************************#
#                                            LtaReader                                             #
#**************************************************************************************************#
#                                                                                                  #
# A FreeSurfer LTA holds the type (0 LINEAR_VOX_TO_VOX, 1 LINEAR_RAS_TO_RAS), one or more 4 x 4    #
# matrices and the geometry of the source and destination volumes (dimensions, voxel sizes,        #
# direction cosines xras/yras/zras, centre cras). A RAS-to-RAS matrix is the RAS map itself; a     #
# voxel-to-voxel one becomes dst_vox2ras @ M @ inv(src_vox2ras), the vox2ras of a volume placing   #
# its centre voxel (dimensions / 2) at cras, as FreeSurfer does.                                   #
#                                                                                                  #
#**************************************************************************************************#
class LtaReader(Reader):
    FORMATS = (Format.TRANSFORM_LTA,)
    PACKAGES = ("miox",)
    TYPES = {0: "LTA voxel to voxel", 1: "LTA RAS to RAS"}

    def read(self, sniff: Sniff) -> list[Transform]:
        lines = [line.split("#")[0].strip() for line in sniff.path.read_text().splitlines()]
        header = lines[:next((i for i, line in enumerate(lines)
                              if re.fullmatch(r"\d+\s+4\s+4", line)), len(lines))]
        fields = {}
        for line in header:
            key, sep, value = line.partition("=")
            if sep:
                fields[key.strip()] = value.strip()
        kind = int(fields.get("type", -1))
        if kind not in self.TYPES:
            raise ReaderError(f"{sniff.path}: LTA type {kind} is not supported")
        count = int(fields.get("nxforms", 1))
        matrices = self._matrices(lines, count, sniff.path)
        source, target = self._volume(lines, "src"), self._volume(lines, "dst")
        notes = []
        datasets = []
        for i, matrix in enumerate(matrices):
            if kind == 1:
                ras = matrix
            elif "vox2ras" in source and "vox2ras" in target:
                ras = target["vox2ras"] @ matrix @ np.linalg.inv(source["vox2ras"])
            else:
                ras = None
                notes.append("voxel-to-voxel matrix without valid volume geometry: no RAS map")
            meta = Meta({"fields": fields})
            name = sniff.path.stem + (f"_{i}" if count > 1 else "")
            datasets.append(Transform(
                "affine", self.TYPES[kind], matrix, None, None, ras,
                {k: self.plain(v) for k, v in source.items()},
                {k: self.plain(v) for k, v in target.items()}, meta,
                self.provenance(sniff, tuple(notes)), name,
            ))
        return datasets

    @staticmethod
    def _matrices(lines: list[str], count: int, path: Path) -> list[np.ndarray]:
        start = next((i for i, line in enumerate(lines) if re.fullmatch(r"\d+\s+4\s+4", line)),
                     None)
        if start is None:
            raise ReaderError(f"{path}: no '<n> 4 4' matrix block")
        rows = [line for line in lines[start + 1:] if line][:4 * count]
        values = np.array([[float(v) for v in row.split()] for row in rows])
        if values.shape != (4 * count, 4):
            raise ReaderError(f"{path}: matrix block of shape {values.shape}")
        return [values[4 * i:4 * i + 4] for i in range(count)]

    @staticmethod
    def _volume(lines: list[str], which: str) -> dict:
        """The volume info block after 'src volume info' or 'dst volume info'."""
        try:
            start = lines.index(f"{which} volume info") + 1
        except ValueError:
            return {}
        info: dict = {}
        for line in lines[start:start + 10]:
            key, sep, value = line.partition("=")
            if not sep:
                break
            info[key.strip()] = value.strip()
        try:
            valid = int(info.get("valid", 0)) == 1
            dims = np.array([float(v) for v in info["volume"].split()])
            sizes = np.array([float(v) for v in info["voxelsize"].split()])
            axes = np.array([[float(v) for v in info[k].split()] for k in ("xras", "yras",
                                                                          "zras")]).T
            centre = np.array([float(v) for v in info["cras"].split()])
        except (KeyError, ValueError):
            return info
        if valid:
            linear = axes * sizes
            info["vox2ras"] = affine(linear, centre - linear @ (dims / 2))
        return info


#**************************************************************************************************#
#                                            ItkReader                                             #
#**************************************************************************************************#
#                                                                                                  #
# ITK transforms map points of the fixed image to the moving image in LPS physical coordinates     #
# (mm). An affine (AffineTransform, MatrixOffsetTransformBase) has 9 matrix and 3 translation      #
# parameters and a centre c (FixedParameters): y = A (x - c) + t + c. The text format (.tfm) lists #
# one or more transforms; the HDF5 format (.h5, also ANTs composite transforms) holds them as      #
# groups of TransformGroup. Each affine becomes a Transform; ras_matrix is the same map in RAS (D  #
# M D with D flipping x and y). Non-affine transforms are counted in a note.                       #
#                                                                                                  #
#**************************************************************************************************#
class ItkReader(Reader):
    FORMATS = (Format.TRANSFORM_ITK,)
    PACKAGES = ("miox",)
    AFFINE = re.compile(r"^(AffineTransform|MatrixOffsetTransformBase)_(double|float)_3_3$")

    def read(self, sniff: Sniff) -> list[Transform]:
        items = self._h5(sniff.path) if sniff.path.suffix.lower() == ".h5" else \
            self._text(sniff.path)
        datasets, skipped = [], []
        for i, (kind, parameters, fixed) in enumerate(items):
            if not self.AFFINE.match(kind):
                skipped.append(kind)
                continue
            if len(parameters) != 12:
                raise ReaderError(f"{sniff.path}: {kind} with {len(parameters)} parameters")
            linear = np.reshape(parameters[:9], (3, 3))
            centre = np.asarray(fixed[:3] if len(fixed) >= 3 else np.zeros(3))
            matrix = affine(linear, np.asarray(parameters[9:]) + centre - linear @ centre)
            notes = (f"not affine, not read: {', '.join(sorted(set(skipped)))}",) if skipped \
                else ()
            meta = Meta({"type": kind, "parameters": list(parameters), "fixed": list(fixed)})
            datasets.append(Transform(
                "affine", "ITK LPS physical, fixed to moving", matrix, None, None,
                LPS @ matrix @ LPS, {}, {}, meta, self.provenance(sniff, notes),
                f"{sniff.path.stem}_{i}" if len(items) > 1 else sniff.path.stem,
            ))
        if not datasets:
            raise ReaderError(f"{sniff.path}: no affine transform ({', '.join(skipped)})")
        return datasets

    @staticmethod
    def _text(path: Path) -> list[tuple[str, list[float], list[float]]]:
        items, kind, parameters = [], None, None
        for line in path.read_text().splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "Transform":
                kind = value.strip()
            elif key.strip() == "Parameters":
                parameters = [float(v) for v in value.split()]
            elif key.strip() == "FixedParameters":
                items.append((kind, parameters or [], [float(v) for v in value.split()]))
                kind, parameters = None, None
        return items

    @staticmethod
    def _h5(path: Path) -> list[tuple[str, list[float], list[float]]]:
        items = []
        with h5py.File(path, "r") as fh:
            group = fh.get("TransformGroup")
            if group is None:
                raise ReaderError(f"{path}: no TransformGroup")
            for key in sorted(group, key=int):
                entry = group[key]
                kind = entry["TransformType"][0]
                kind = kind.decode() if isinstance(kind, bytes) else str(kind)
                if kind.startswith("CompositeTransform"):
                    continue
                parameters = entry["TransformParameters"][()].tolist() \
                    if "TransformParameters" in entry else []
                fixed = entry["TransformFixedParameters"][()].tolist() \
                    if "TransformFixedParameters" in entry else []
                items.append((kind, parameters, fixed))
        return items


#**************************************************************************************************#
#                                            AfniReader                                            #
#**************************************************************************************************#
#                                                                                                  #
# AFNI affine matrices: rows of 12 numbers, the 3 x 4 matrix row by row in DICOM (LPS)             #
# coordinates, one row per transform (3dvolreg -1Dmatrix_save, 3dAllineate .aff12.1D); "#" lines   #
# are comments. ras_matrix is the same map in RAS.                                                 #
#                                                                                                  #
#**************************************************************************************************#
class AfniReader(Reader):
    FORMATS = (Format.TRANSFORM_AFNI,)
    PACKAGES = ("miox",)

    def read(self, sniff: Sniff) -> list[Transform]:
        rows = [line.split() for line in sniff.path.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
        if not rows or any(len(r) != 12 for r in rows):
            raise ReaderError(f"{sniff.path}: rows of 12 numbers expected")
        datasets = []
        for i, row in enumerate(rows):
            values = np.array([float(v) for v in row]).reshape(3, 4)
            matrix = affine(values[:, :3], values[:, 3])
            datasets.append(Transform(
                "affine", "AFNI DICOM (LPS)", matrix, None, None, LPS @ matrix @ LPS, {}, {},
                Meta({}), self.provenance(sniff),
                f"{sniff.path.stem}_{i}" if len(rows) > 1 else sniff.path.stem,
            ))
        return datasets


#**************************************************************************************************#
#                                            FslReader                                             #
#**************************************************************************************************#
#                                                                                                  #
# An FSL FLIRT matrix: 4 x 4 numbers in FSL's scaled-voxel coordinates of the source and reference #
# images (voxel indices times voxel sizes, x flipped for images with a positive determinant). The  #
# file names neither image, so the RAS map cannot be derived from it alone: ras_matrix stays None. #
#                                                                                                  #
#**************************************************************************************************#
class FslReader(Reader):
    FORMATS = (Format.TRANSFORM_FSL,)
    PACKAGES = ("miox",)

    def read(self, sniff: Sniff) -> list[Transform]:
        rows = [line.split() for line in sniff.path.read_text().splitlines() if line.strip()]
        try:
            matrix = np.array([[float(v) for v in row] for row in rows])
        except ValueError as e:
            raise ReaderError(f"{sniff.path}: not an FSL matrix ({e})") from e
        if matrix.shape != (4, 4):
            raise ReaderError(f"{sniff.path}: matrix of shape {matrix.shape}, 4 x 4 expected")
        notes = ("FSL scaled-voxel coordinates: the RAS map needs the source and reference "
                 "images, which the file does not name",)
        return [Transform("affine", "FSL scaled voxel", matrix, None, None, None, {}, {},
                          Meta({}), self.provenance(sniff, notes), sniff.path.stem)]
