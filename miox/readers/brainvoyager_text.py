####################################################################################################
#                                       brainvoyager_text.py                                       #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 12/10/26                                                                                #
#                                                                                                  #
# Purpose: Readers for BrainVoyager text files: stimulation protocols (.prt), design matrices      #
#          (.sdm), volumes of interest (.voi), patches of interest (.poi), spatial                 #
#          transformations (.trf) and functional projects (.fmr with their slice time courses,     #
#          .stc).                                                                                  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

# own
from miox.data import MRImage, Protocol, Surface, Transform
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.readers.brainvoyager import bv_affine
from miox.sniff import Format, Sniff

NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
FIELD = re.compile(r"^([A-Za-z][\w ]*?):\s*(.*)$")
LPS = np.array([-1.0, -1.0, 1.0])  # DICOM LPS to RAS


def lines_of(source: Sniff | Path) -> list[str]:
    """The non-empty lines, stripped, tabs as spaces."""
    path = source if isinstance(source, Path) else source.path
    text = path.read_text(errors="replace")
    return [line.strip().replace("\t", " ") for line in text.splitlines() if line.strip()]


def field(line: str) -> tuple[str, str] | None:
    """A "Name: value" line as (name, value without quotes)."""
    match = FIELD.match(line)
    return (match[1].strip(), match[2].strip().strip('"')) if match else None


def plain(value: str):
    """A header value as int or float when it is one number."""
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


#**************************************************************************************************#
#                                  BrainVoyagerProtocolReader                                      #
#**************************************************************************************************#
#                                                                                                  #
# A PRT lists conditions, each with its intervals (start and stop, in milliseconds or volumes as   #
# ResolutionOfTime states; a third column of parametric weights when ParametricWeights is set) and #
# a colour. An SDM holds a design matrix: one column per predictor (name, colour), one row per     #
# data point, and which predictors are confounds. Both become a Protocol with everything in the    #
# vendor header.                                                                                   #
#                                                                                                  #
#**************************************************************************************************#
class BrainVoyagerProtocolReader(Reader):
    FORMATS = (Format.BRAINVOYAGER_PRT, Format.BRAINVOYAGER_SDM)
    PACKAGES = ("miox",)

    def read(self, sniff: Sniff) -> list[Protocol]:
        lines = lines_of(sniff)
        try:
            if sniff.format == Format.BRAINVOYAGER_PRT:
                return [self._prt(sniff, lines)]
            return [self._sdm(sniff, lines)]
        except (IndexError, ValueError) as e:
            raise ReaderError(f"{sniff.path}: not a readable BrainVoyager file ({e})") from e

    def _prt(self, sniff: Sniff, lines: list[str]) -> Protocol:
        header, i = {}, 0
        while "NrOfConditions" not in header:
            name, value = field(lines[i])
            header[name] = plain(value)
            i += 1
        weighted = bool(header.get("ParametricWeights", 0))
        conditions = []
        for _ in range(int(header["NrOfConditions"])):
            name, count = lines[i], int(lines[i + 1])
            rows = [[float(v) for v in NUMBER.findall(line)] for line in lines[i + 2:i + 2 + count]]
            condition = {"name": name, "start": [r[0] for r in rows], "stop": [r[1] for r in rows]}
            if weighted:
                condition["weight"] = [r[2] for r in rows]
            i += 2 + count
            if i < len(lines) and lines[i].lower().startswith("color"):
                condition["color"] = [int(v) for v in NUMBER.findall(lines[i])]
                i += 1
            conditions.append(condition)
        meta = Meta({"header": header, "conditions": conditions})
        meta.set("ProtocolTimeResolution", str(header.get("ResolutionOfTime", "")),
                 "PRT header: ResolutionOfTime", "header/ResolutionOfTime")
        name = str(header.get("Experiment") or sniff.path.stem)
        return Protocol(name, (), meta, self.provenance(sniff))

    def _sdm(self, sniff: Sniff, lines: list[str]) -> Protocol:
        header, i = {}, 0
        while (pair := field(lines[i])) is not None:
            header[pair[0]] = plain(pair[1])
            i += 1
        count = int(header["NrOfPredictors"])
        colours = [int(v) for v in NUMBER.findall(lines[i])]
        names = re.findall(r'"([^"]*)"', lines[i + 1])
        rows = [[float(v) for v in NUMBER.findall(line)] for line in lines[i + 2:]]
        matrix = np.array(rows)
        if len(names) != count or matrix.shape != (int(header["NrOfDataPoints"]), count):
            raise ReaderError(f"{sniff.path}: {len(names)} names and a {matrix.shape} matrix "
                              f"for {count} predictors")
        predictors = [{"name": n, "color": colours[3 * j:3 * j + 3]} for j, n in enumerate(names)]
        meta = Meta({"header": header, "predictors": predictors, "matrix": matrix.tolist()})
        return Protocol(sniff.path.stem, (), meta, self.provenance(sniff))


#**************************************************************************************************#
#                                   BrainVoyagerRegionReader                                       #
#**************************************************************************************************#
#                                                                                                  #
# A VOI file lists volumes of interest as voxels: in BV space as (X, Y, Z) voxel indices of the    #
# VMR's framing cube (its resolution, offsets and cube size stated in the header), in TAL or MNI   #
# space as (x, y, z) mm. They become one label volume (value = index of the VOI, 1-based) on the   #
# bounding box of all voxels, with BrainVoyager's internal axes and the framing-cube centre at 0   #
# (TAL/MNI: the 1 mm 256 cube, BV = 128 - TAL with the axes permuted). A POI file lists patches of #
# interest as vertex indices of a mesh: each becomes a label (1 inside) on that mesh's vertices.   #
#                                                                                                  #
#**************************************************************************************************#
class BrainVoyagerRegionReader(Reader):
    FORMATS = (Format.BRAINVOYAGER_VOI, Format.BRAINVOYAGER_POI)
    PACKAGES = ("miox",)

    def read(self, sniff: Sniff) -> list[MRImage | Surface]:
        lines = lines_of(sniff)
        try:
            if sniff.format == Format.BRAINVOYAGER_VOI:
                return [self._voi(sniff, lines)]
            return [self._poi(sniff, lines)]
        except (IndexError, ValueError, KeyError) as e:
            raise ReaderError(f"{sniff.path}: not a readable BrainVoyager file ({e})") from e

    def _voi(self, sniff: Sniff, lines: list[str]) -> MRImage:
        header, vois, i = {}, [], 0
        while i < len(lines):
            pair = field(lines[i])
            i += 1
            if pair is None:
                continue
            key, value = pair
            if key == "NameOfVOI":
                vois.append({"name": value})
            elif key == "ColorOfVOI":
                vois[-1]["color"] = [int(v) for v in NUMBER.findall(value)]
            elif key == "NrOfVoxels":
                count = int(value)
                vois[-1]["voxels"] = np.array([[int(float(v)) for v in lines[i + j].split()]
                                               for j in range(count)]).reshape(-1, 3)
                i += count
            elif key == "NrOfVOIVTCs":
                header[key] = int(value)
                header["VOIVTCs"] = lines[i:i + int(value)]
                i += int(value)
            elif not vois:
                header[key] = plain(value)
        space = str(header.get("ReferenceSpace", "BV")).upper()
        if space == "BV":
            voxel = tuple(float(header.get(f"OriginalVMRResolution{a}", 1.0)) for a in "XYZ")
            cube = int(header.get("OriginalVMRFramingCubeDim", 256))
            offset = np.array([int(header.get(f"OriginalVMROffset{a}", 0)) for a in "XYZ"])
            to_index = lambda v: v  # noqa: E731
        elif space in ("TAL", "MNI"):
            voxel, cube, offset = (1.0, 1.0, 1.0), 256, np.zeros(3, dtype=int)
            to_index = lambda v: 128 - v[:, [1, 2, 0]]  # noqa: E731  (x, y, z) mm -> (X, Y, Z)
        else:
            raise ReaderError(f"{sniff.path}: reference space {space} not known")
        convention = int(header.get("LeftRightConvention", 1))
        indices = [to_index(v["voxels"]) for v in vois]
        everything = np.concatenate(indices) if indices else np.zeros((0, 3), dtype=int)
        low = everything.min(axis=0) if len(everything) else np.zeros(3, dtype=int)
        size = (everything.max(axis=0) - low + 1) if len(everything) else np.ones(3, dtype=int)
        labels = np.zeros(tuple(size), dtype=np.int16)
        notes = []
        for n, voxels in enumerate(indices, start=1):
            local = voxels - low
            if labels[tuple(local.T)].any():
                notes.append(f"VOI {n} overlaps an earlier one: the overlap is labelled {n}")
            labels[tuple(local.T)] = n
        affine = bv_affine(tuple(low + offset), 1.0, cube, voxel, convention)
        notes.append(f"label volume on the voxels' bounding box, BrainVoyager {space} space, "
                     "framing-cube centre at 0")
        table = {n: {"name": v["name"], "color": v.get("color")} for n, v in enumerate(vois, 1)}
        meta = Meta({"header": header, "labels": table,
                     "voxel_counts": [len(v["voxels"]) for v in vois]})
        return MRImage(labels, affine, meta, self.provenance(sniff, tuple(notes)),
                       sniff.path.stem)

    def _poi(self, sniff: Sniff, lines: list[str]) -> Surface:
        header, pois, i = {}, [], 0
        while i < len(lines):
            pair = field(lines[i])
            i += 1
            if pair is None:
                continue
            key, value = pair
            if key == "NameOfPOI":
                pois.append({"name": value})
            elif key == "NrOfVertices" and pois:
                count = int(value)
                pois[-1]["vertices"] = np.array([int(lines[i + j]) for j in range(count)])
                i += count
            elif key == "NrOfPOIMTCs":
                header[key] = int(value)
                header["POIMTCs"] = lines[i:i + int(value)]
                i += int(value)
            elif pois:
                pois[-1][key] = plain(value)
            else:
                header[key] = plain(value)
        n = int(header["NrOfMeshVertices"])
        labels = {}
        for poi in pois:
            mask = np.zeros(n, dtype=np.int16)
            mask[poi["vertices"]] = 1
            colour = [int(v) / 255.0 for v in NUMBER.findall(str(poi.get("ColorOfPOI", "")))]
            labels[poi["name"]] = {"values": mask,
                                   "table": {1: (poi["name"], (*colour[:3], 1.0))}}
        vendor = {"header": header,
                  "pois": [{k: v for k, v in p.items() if k != "vertices"} for p in pois]}
        return Surface(None, None, "BrainVoyager mesh vertices", {}, labels, Meta(vendor),
                       self.provenance(sniff), sniff.path.stem)


#**************************************************************************************************#
#                                  BrainVoyagerTransformReader                                     #
#**************************************************************************************************#
#                                                                                                  #
# A TRF file states its fields as "Name: value" lines and its 4 x 4 matrices as rows of four       #
# numbers (DataFormat Matrix): one matrix, or two when ExtraVMRTransf is set (the FMR-VMR          #
# alignment, then the extra VMR transformation). The matrices act on BrainVoyager's coordinates of #
# the files named; no RAS map is derived.                                                          #
#                                                                                                  #
#**************************************************************************************************#
class BrainVoyagerTransformReader(Reader):
    FORMATS = (Format.BRAINVOYAGER_TRF,)
    PACKAGES = ("miox",)
    TYPES = {1: "rigid body (FMR-VMR alignment)", 2: "affine", 3: "MNI (12 parameters)",
             4: "Talairach"}

    def read(self, sniff: Sniff) -> list[Transform]:
        header, rows = {}, []
        for line in lines_of(sniff):
            pair = field(line)
            if pair is not None:
                header[pair[0]] = plain(pair[1])
            else:
                rows.append([float(v) for v in line.split()])
        if str(header.get("DataFormat", "Matrix")) != "Matrix":
            raise ReaderError(f"{sniff.path}: DataFormat {header['DataFormat']} is not read")
        if not rows or len(rows) % 4 or any(len(r) != 4 for r in rows):
            raise ReaderError(f"{sniff.path}: matrix rows of four numbers expected")
        kind = int(header.get("TransformationType", 0))
        convention = f"BrainVoyager TRF type {kind}: {self.TYPES.get(kind, 'unknown')}"
        notes = ("matrix on BrainVoyager coordinates; no RAS map derived",)
        matrices = [np.array(rows[i:i + 4]) for i in range(0, len(rows), 4)]
        roles = ["", "_extra_vmr"] if len(matrices) == 2 else [f"_{i}" for i in
                                                                range(len(matrices))]
        roles = [""] if len(matrices) == 1 else roles
        return [Transform("affine", convention, matrix, None, None, None,
                          {"file": header.get("SourceFile", "")},
                          {"file": header.get("TargetFile", "")}, Meta({"header": header}),
                          self.provenance(sniff, notes), sniff.path.stem + role)
                for matrix, role in zip(matrices, roles, strict=True)]


#**************************************************************************************************#
#                                     BrainVoyagerFmrReader                                        #
#**************************************************************************************************#
#                                                                                                  #
# An FMR is the text header of a functional project: volumes, slices, matrix, TR, voxel sizes,     #
# the PosInfos of the scan (slice centres and directions in DICOM LPS, as in the VMR), past        #
# transformations, the left-right convention and the slice timings; the samples sit in the STC    #
# file named by Prefix (DataStorageFormat 2: slice by slice, each slice volume by volume, rows,    #
# columns; 3: volume by volume; 4: time innermost; 1: one file per slice, not read), as int16 or   #
# float32 (DataType). The volume becomes an MRImage (columns, rows, slices, time); its affine      #
# comes from the PosInfos when they are verified, else the voxel sizes on the stored axes with    #
# the volume centre at 0.                                                                          #
#                                                                                                  #
#**************************************************************************************************#
class BrainVoyagerFmrReader(Reader):
    FORMATS = (Format.BRAINVOYAGER_FMR,)
    PACKAGES = ("miox",)
    DTYPES = {1: "i2", 2: "f4"}
    LAYOUTS = {  # DataStorageFormat: (file order of (slices, volumes, rows, columns), transpose)
        2: ((0, 1, 2, 3), (3, 2, 0, 1)),
        3: ((1, 0, 2, 3), (3, 2, 1, 0)),
        4: ((0, 2, 3, 1), (2, 1, 0, 3)),
    }
    MAIN = ("FileVersion", "NrOfVolumes", "NrOfSlices", "ResolutionX", "ResolutionY", "Prefix")

    def read(self, sniff: Sniff) -> list[MRImage]:
        paths = (sniff.path,) + sniff.companions
        fmr = next((p for p in paths if p.suffix.lower() == ".fmr"), None)
        stc = next((p for p in paths if p.suffix.lower() == ".stc"), None)
        if fmr is None:
            raise ReaderError(f"{sniff.path.name}: no FMR header names this STC file")
        try:
            h = self._header(lines_of(fmr))
            return [self._volume(sniff, fmr, stc, h)]
        except (IndexError, ValueError, KeyError) as e:
            raise ReaderError(f"{fmr}: not a readable FMR file ({e})") from e

    @staticmethod
    def _header(lines: list[str]) -> dict:
        header: dict = {}
        target, i = header, 0
        while i < len(lines):
            pair = field(lines[i])
            if pair is None:
                if lines[i] == "PositionInformationFromImageHeaders":
                    target = header.setdefault("PosInfos", {})
                i += 1
                continue
            name, value = pair
            if name == "NrOfPastSpatialTransformations":
                target = header
                header[name], header["PastTransformations"] = plain(value), []
            elif name == "NameOfSpatialTransformation":
                header["PastTransformations"].append({"Name": value})
            elif name in ("TypeOfSpatialTransformation", "AppliedToFileName"):
                header["PastTransformations"][-1][name] = plain(value)
            elif name == "NrOfTransformationValues":
                values: list[float] = []
                while len(values) < int(value):
                    i += 1
                    values += [float(v) for v in NUMBER.findall(lines[i])]
                header["PastTransformations"][-1]["Values"] = values
            elif name == "SliceTimingTableSize":
                header[name] = int(value)
                header["SliceTimings"] = [float(lines[i + k + 1]) for k in range(int(value))]
                i += int(value)
            else:
                target[name] = plain(value)
            i += 1
        header.setdefault("ResolutionX", header.get("NrOfColumns"))
        header.setdefault("ResolutionY", header.get("NrOfRows"))
        return header

    def _volume(self, sniff: Sniff, fmr: Path, stc: Path | None, h: dict) -> MRImage:
        nt, nz, nx, ny = (int(h[k]) for k in ("NrOfVolumes", "NrOfSlices", "ResolutionX",
                                                "ResolutionY"))
        storage = int(h.get("DataStorageFormat", 1))
        if storage not in self.LAYOUTS:
            raise ReaderError(f"{fmr.name}: DataStorageFormat {storage} (one STC file per slice) "
                              "is not read")
        if stc is None:
            raise ReaderError(f"{fmr.name}: slice time courses {h['Prefix']}.stc not found "
                              "beside it")
        dtype = self.DTYPES.get(int(h.get("DataType", 2)))
        if dtype is None:
            raise ReaderError(f"{fmr.name}: DataType {h.get('DataType')} unknown")
        values = np.fromfile(stc, dtype="<" + dtype)
        if values.size != nt * nz * ny * nx:
            raise ReaderError(f"{stc.name}: {values.size} values for {nt} volumes of "
                              f"{nz} x {ny} x {nx}")
        order, axes = self.LAYOUTS[storage]
        counts = (nz, nt, ny, nx)
        data = values.reshape([counts[a] for a in order]).transpose(axes)
        voxel = (float(h.get("InplaneResolutionX", 1.0)), float(h.get("InplaneResolutionY", 1.0)),
                 float(h.get("SliceThickness", 1.0)) + float(h.get("SliceGap", 0.0)))
        notes: tuple[str, ...] = ()
        if storage != 2:
            notes += (f"DataStorageFormat {storage} laid out as the developer guide describes, "
                      "unverified",)
        affine = self._scanner_affine(h.get("PosInfos", {}), voxel, (nx, ny, nz))
        if affine is None:
            affine = np.diag(voxel + (1.0,))
            affine[:3, 3] = -(np.array((nx, ny, nz)) - 1) / 2.0 * voxel
            notes += ("slice positions not verified: axes as stored, volume centre at 0",)
        else:
            notes += ("affine from the verified PosInfos (DICOM LPS turned to RAS)",)
        meta = Meta(h)
        meta.set("RepetitionTime", float(h["TR"]) / 1000.0, "FMR header: TR (ms)", "TR")
        if float(h.get("TE", 0)) > 0:
            meta.set("EchoTime", float(h["TE"]) / 1000.0, "FMR header: TE (ms)", "TE")
        provenance = self.provenance(sniff, notes)
        return MRImage(data, affine, meta, provenance, str(h.get("Prefix") or fmr.stem),
                       ("time",))

    @staticmethod
    def _scanner_affine(pos: dict, voxel: tuple, dims: tuple) -> np.ndarray | None:
        """Rows along RowDir, columns along ColDir, slices from the first to the last centre
        (all DICOM LPS), the first slice's centre being the centre of the first slice's plane;
        None unless the FMR marks its PosInfos as verified."""
        if pos.get("PosInfosVerified") != 1:
            return None
        row = np.array([pos.get(f"RowDir{a}", 0.0) for a in "XYZ"], dtype=float) * LPS
        column = np.array([pos.get(f"ColDir{a}", 0.0) for a in "XYZ"], dtype=float) * LPS
        first = np.array([pos.get(f"Slice1Center{a}", 0.0) for a in "XYZ"], dtype=float) * LPS
        last = np.array([pos.get(f"SliceNCenter{a}", 0.0) for a in "XYZ"], dtype=float) * LPS
        if not np.linalg.norm(row) or not np.linalg.norm(column):
            return None
        row, column = row / np.linalg.norm(row), column / np.linalg.norm(column)
        normal = last - first
        if np.linalg.norm(normal):
            normal = normal / np.linalg.norm(normal)
        else:
            normal = np.cross(row, column)
        nx, ny, _ = dims
        affine = np.eye(4)
        affine[:3, :3] = np.column_stack((row * voxel[0], column * voxel[1], normal * voxel[2]))
        affine[:3, 3] = first - affine[:3, 0] * (nx - 1) / 2.0 - affine[:3, 1] * (ny - 1) / 2.0
        return affine
