####################################################################################################
#                                         brainvoyager.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 12/10/26                                                                                #
#                                                                                                  #
# Purpose: Readers for BrainVoyager binary files: anatomical volumes (.vmr), volume time courses   #
#          (.vtc), GLM results (.glm), volume maps (.vmp), surfaces (.srf), surface maps (.smp),   #
#          mesh time courses (.mtc) and surface-to-surface mappings (.ssm); also gzipped.          #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import gzip
import struct

import numpy as np

# own
from miox.data import MRImage, Surface
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff

DEFAULT_CUBE = 256  # BrainVoyager's framing cube when a file does not name its VMR's
CONVENTIONS = {0: "unknown", 1: "radiological", 2: "neurological"}
SPACES = {0: "unknown", 1: "native", 2: "ACPC", 3: "Talairach", 4: "MNI"}
BOX = ("XStart", "XEnd", "YStart", "YEnd", "ZStart", "ZEnd")  # a grid as a box of VMR voxels


def read_bytes(sniff: Sniff) -> bytes:
    """The file's bytes, decompressed when it is gzipped (.vmr.gz, ...)."""
    raw = sniff.path.read_bytes()
    return gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw


def bv_affine(start: tuple, step: float, cube: float, voxel: tuple, convention: int) -> np.ndarray:
    """Voxel (i, j, k) on BrainVoyager's internal axes (X anterior -> posterior, Y superior ->
    inferior, Z right -> left in radiological and left -> right in neurological convention) to
    RAS mm, with the framing cube's centre at 0: the voxel's position in VMR voxels is
    start + step * index on each axis, and R, A, S = (cube / 2 - position) * voxel size."""
    sign = -1.0 if convention == 2 else 1.0
    centre = cube / 2.0
    (sx, sy, sz), (vx, vy, vz) = start, voxel
    return np.array([
        [0.0, 0.0, -sign * step * vz, sign * (centre - sz) * vz],
        [-step * vx, 0.0, 0.0, (centre - sx) * vx],
        [0.0, -step * vy, 0.0, (centre - sy) * vy],
        [0.0, 0.0, 0.0, 1.0],
    ])


#**************************************************************************************************#
#                                             BvStream                                             #
#**************************************************************************************************#
#                                                                                                  #
# Reads a BrainVoyager file front to back: little-endian numbers, NUL-terminated strings, RGB      #
# byte triples and arrays.                                                                         #
#                                                                                                  #
#**************************************************************************************************#
class BvStream:
    def __init__(self, raw: bytes):
        self.raw, self.position = raw, 0

    def read(self, fmt: str):
        values = struct.unpack_from("<" + fmt, self.raw, self.position)
        self.position += struct.calcsize("<" + fmt)
        return values[0] if len(values) == 1 else list(values)

    def string(self) -> str:
        end = self.raw.index(b"\0", self.position)
        text = self.raw[self.position:end].decode("utf-8", "replace")
        self.position = end + 1
        return text

    def rgb(self) -> list[int]:
        return list(self.read("3B"))

    def array(self, count: int, dtype: str) -> np.ndarray:
        dtype = np.dtype("<" + dtype)
        if self.position + count * dtype.itemsize > len(self.raw):
            raise ReaderError(f"{count} values of {dtype} run past the end of the file")
        values = np.frombuffer(self.raw, dtype, count, self.position)
        self.position += count * dtype.itemsize
        return values

    @property
    def left(self) -> int:
        return len(self.raw) - self.position


#**************************************************************************************************#
#                                     BrainVoyagerVolumeReader                                     #
#**************************************************************************************************#
#                                                                                                  #
# Volumes on BrainVoyager's internal axes, stored with X fastest and Z slowest; MIOX keeps the     #
# data as (X, Y, Z[, maps or time]) and puts the axes into the affine (see bv_affine). A VMR (one  #
# byte per voxel) states its voxel sizes, offsets in its framing cube (version 3+), the left-right #
# convention and reference space (version 4) and, after the data, the slice positions of the       #
# scan it came from (PosInfos, DICOM LPS; kept in the vendor header) and its past                  #
# transformations. VTC, GLM and VMP grids are boxes of VMR voxels (start, end, resolution          #
# multiplier); their voxels are taken as 1 mm VMR voxels in the cube of the VMR the file names     #
# (VMP states its VMR's dimensions; VTC and GLM do not, so the default 256 cube is assumed). A GLM #
# holds per voxel R2, SS, one beta and one SS_XiY per predictor, the mean time course value and    #
# 0-2 autoregression coefficients (RFX GLMs: R2 and the subjects' betas), as maps on a fourth      #
# axis named in the vendor header; GLMs on surface vertices become a Surface.                      #
#                                                                                                  #
#**************************************************************************************************#
class BrainVoyagerVolumeReader(Reader):
    FORMATS = (Format.BRAINVOYAGER_VMR, Format.BRAINVOYAGER_VTC, Format.BRAINVOYAGER_GLM,
               Format.BRAINVOYAGER_VMP)
    PACKAGES = ("miox",)
    VMP_MAGIC = 0xA1B2C3D4

    def read(self, sniff: Sniff) -> list[MRImage | Surface]:
        stream = BvStream(read_bytes(sniff))
        name = sniff.path.name.removesuffix(".gz").rsplit(".", 1)[0]
        try:
            if sniff.format == Format.BRAINVOYAGER_VMR:
                return [self._vmr(sniff, stream, name)]
            if sniff.format == Format.BRAINVOYAGER_VTC:
                return [self._vtc(sniff, stream, name)]
            if sniff.format == Format.BRAINVOYAGER_GLM:
                return [self._glm(sniff, stream, name)]
            return [self._vmp(sniff, stream, name)]
        except (struct.error, ValueError) as e:
            raise ReaderError(f"{sniff.path}: not a readable BrainVoyager file ({e})") from e

    def _vmr(self, sniff: Sniff, s: BvStream, name: str) -> MRImage:
        header = {"version": s.read("H"), "DimX": s.read("H"), "DimY": s.read("H"),
                  "DimZ": s.read("H")}
        dims = (header["DimX"], header["DimY"], header["DimZ"])
        data = s.array(int(np.prod(dims)), "u1").reshape(dims[::-1]).transpose(2, 1, 0)
        notes = []
        if header["version"] >= 2:
            try:
                header.update(self._vmr_post(s, header["version"]))
            except (struct.error, ValueError, ReaderError):
                notes.append("post-data header incomplete: geometry defaults used")
        offsets = [header.get(f"Offset{a}", 0) for a in "XYZ"]
        cube = header.get("FramingCubeDim") or max(dims)
        voxel = [header.get(f"VoxelSize{a}", 1.0) for a in "XYZ"]
        convention = header.get("LeftRightConvention", 0)
        if convention not in (1, 2):
            notes.append("left-right convention not stated: radiological assumed")
        space = SPACES.get(header.get("ReferenceSpaceVMR", 0), "unknown")
        notes.append(f"affine in BrainVoyager {space} space, framing-cube centre at 0")
        return MRImage(data, bv_affine(offsets, 1.0, cube, voxel, convention),
                       Meta(self.plain(header)),
                       self.provenance(sniff, tuple(notes)), name)

    @staticmethod
    def _vmr_post(s: BvStream, version: int) -> dict:
        """The post-data header of a VMR (version 2 and later)."""
        h: dict = {}
        if version >= 3:
            h.update(zip(("OffsetX", "OffsetY", "OffsetZ", "FramingCubeDim"), s.read("4h"),
                         strict=True))
        h["PosInfosVerified"], h["CoordinateSystem"] = s.read("2i")
        names = ("Slice1CenterX", "Slice1CenterY", "Slice1CenterZ", "SliceNCenterX",
                 "SliceNCenterY", "SliceNCenterZ", "RowDirX", "RowDirY", "RowDirZ", "ColDirX",
                 "ColDirY", "ColDirZ")
        h.update(zip(names, s.read("12f"), strict=True))
        h["NRows"], h["NCols"] = s.read("2i")
        h.update(zip(("FoVRows", "FoVCols", "SliceThickness", "GapThickness"), s.read("4f"),
                     strict=True))
        h["PastTransformations"] = []
        for _ in range(s.read("i")):
            past = {"Name": s.string(), "Type": s.read("i"), "SourceFileName": s.string()}
            past["Values"] = s.array(s.read("i"), "f4").tolist()
            h["PastTransformations"].append(past)
        h["LeftRightConvention"] = s.read("B")
        if version >= 4:
            h["ReferenceSpaceVMR"] = s.read("B")
        h.update(zip(("VoxelSizeX", "VoxelSizeY", "VoxelSizeZ"), s.read("3f"), strict=True))
        h["VoxelResolutionVerified"], h["VoxelResolutionInTALmm"] = s.read("2B")
        h.update(zip(("VMROrigV16MinValue", "VMROrigV16MeanValue", "VMROrigV16MaxValue"),
                     s.read("3i"), strict=True))
        return h

    def _vtc(self, sniff: Sniff, s: BvStream, name: str) -> MRImage:
        h: dict = {"version": s.read("h")}
        if h["version"] != 3:
            raise ReaderError(f"{sniff.path}: VTC version {h['version']} is not read (layout of "
                              "version 3 only)")
        h["SourceFMR"] = s.string()
        h["ProtocolAttached"] = s.read("h")
        h["Protocol"] = s.string() if h["ProtocolAttached"] > 0 else ""
        h["CurrentProtocolIndex"], h["DataType"], h["NrOfVolumes"], h["Resolution"] = s.read("4h")
        h.update(zip(BOX, s.read("6h"), strict=True))
        h["LeftRightConvention"], h["ReferenceSpace"] = s.read("2B")
        h["TR"] = s.read("f")
        dims = self._box(h, h["Resolution"])
        dtype = {1: "i2", 2: "f4"}.get(h["DataType"])
        if dtype is None:
            raise ReaderError(f"{sniff.path}: VTC data type {h['DataType']} unknown")
        values = s.array(int(np.prod(dims)) * h["NrOfVolumes"], dtype)
        data = values.reshape(*dims[::-1], h["NrOfVolumes"]).transpose(2, 1, 0, 3)
        meta = Meta(self.plain(h))
        meta.set("RepetitionTime", h["TR"] / 1000.0, "VTC header: TR (ms)")
        return MRImage(data, self._grid(h, h["Resolution"], DEFAULT_CUBE), meta,
                       self.provenance(sniff, self._grid_notes(h, DEFAULT_CUBE)), name,
                       ("time",))

    def _glm(self, sniff: Sniff, s: BvStream, name: str) -> MRImage | Surface:
        h: dict = {"version": s.read("h")}
        if h["version"] != 4:
            raise ReaderError(f"{sniff.path}: GLM version {h['version']} is not read (layout of "
                              "version 4 only)")
        h["Type"], h["RFX"] = s.read("2B")
        if h["RFX"]:
            h["NrOfSubjects"], h["NrOfSubjectPredictors"] = s.read("2i")
        h["NrOfTimePoints"], h["NrOfPredictors"], h["NrOfConfounds"], h["NrOfStudies"] = (
            s.read("4i"))
        if h["NrOfStudies"] > 1:
            h["NrOfConfoundsPerStudy"] = s.array(s.read("i"), "i4").tolist()
        h["SeparatePredictors"], h["TimeCourseNormalization"] = s.read("2B")
        h["Resolution"] = s.read("h")
        h["SerialCorrelation"] = s.read("B")
        h["SerialCorrelationBefore"], h["SerialCorrelationAfter"] = s.read("2f")
        if h["Type"] == 0:
            h.update(zip(("DimX", "DimY", "DimZ"), s.read("3h"), strict=True))
        elif h["Type"] == 1:
            h.update(zip(BOX, s.read("6h"), strict=True))
        elif h["Type"] == 2:
            h["NrOfVertices"] = s.read("i")
        else:
            raise ReaderError(f"{sniff.path}: GLM type {h['Type']} unknown")
        h["CortexMask"] = s.read("B")
        h["NrOfVoxelsInMask"] = s.read("i")
        h["CortexMaskName"] = s.string()
        h["Studies"] = []
        for _ in range(h["NrOfStudies"]):
            study = {"NrOfTimePoints": s.read("i"), "Data": s.string()}
            if h["Type"] == 2:
                study["SSM"] = s.string()
            study["SDM"] = s.string()
            h["Studies"].append(study)
        h["Predictors"] = [{"Name": s.string(), "CustomName": s.string(),
                            "Colors": [s.rgb() for _ in range(4)]}
                           for _ in range(h["NrOfPredictors"])]
        if not h["RFX"]:
            n, p = h["NrOfTimePoints"], h["NrOfPredictors"]
            h["DesignMatrix"] = s.array(n * p, "f4").reshape(n, p).tolist()
            h["InvertedXX"] = s.array(p * p, "f4").reshape(p, p).tolist()
        h["Maps"] = self._glm_maps(h)
        if h["Type"] == 2:
            values = s.array(len(h["Maps"]) * h["NrOfVertices"], "f4").reshape(len(h["Maps"]), -1)
            vertex_data = {m: values[i] for i, m in enumerate(h["Maps"])}
            return Surface(None, None, "BrainVoyager mesh vertices", vertex_data, {},
                           Meta(self.plain(h)), self.provenance(sniff), name)
        if h["Type"] == 0:
            dims = (h["DimX"], h["DimY"], h["DimZ"])
            affine, notes = self.NO_GEOMETRY.copy(), ("FMR-STC GLM: no geometry stated",)
        else:
            dims = self._box(h, h["Resolution"])
            affine = self._grid(h, h["Resolution"], DEFAULT_CUBE)
            notes = self._grid_notes(h, DEFAULT_CUBE)
        values = s.array(len(h["Maps"]) * int(np.prod(dims)), "f4")
        data = values.reshape(len(h["Maps"]), *dims[::-1]).transpose(3, 2, 1, 0)
        if s.left:
            notes += (f"{s.left} bytes after the maps",)
        return MRImage(data, affine, Meta(self.plain(h)), self.provenance(sniff, notes), name,
                       ("map",))

    @staticmethod
    def _glm_maps(h: dict) -> list[str]:
        """The names of the maps a GLM stores per voxel, in file order."""
        if h["RFX"]:
            return ["R2"] + [f"beta {i}" for i in range(h["NrOfSubjects"]
                                                        * h["NrOfSubjectPredictors"])]
        names = [p["CustomName"] or p["Name"] for p in h["Predictors"]]
        return (["R2", "SS"] + [f"beta {n}" for n in names] + [f"SS_XiY {n}" for n in names]
                + ["mean"] + [f"AR {i + 1}" for i in range(h["SerialCorrelation"])])

    def _vmp(self, sniff: Sniff, s: BvStream, name: str) -> MRImage:
        h: dict = {"magic": s.read("I")}
        h["version"], h["DocumentType"] = s.read("2h")
        if h["magic"] != self.VMP_MAGIC or h["version"] != 6:
            raise ReaderError(f"{sniff.path}: only NR-VMP version 6 files are read")
        h.update(zip(("NrOfSubMaps", "NrOfTimePoints", "NrOfComponentParams",
                      "ShowParamsRangeFrom", "ShowParamsRangeTo", "FingerprintParamsRangeFrom",
                      "FingerprintParamsRangeTo", "XStart", "XEnd", "YStart", "YEnd", "ZStart",
                      "ZEnd", "Resolution", "DimX", "DimY", "DimZ"), s.read("17i"), strict=True))
        h["VTC"], h["Protocol"], h["VOI"] = s.string(), s.string(), s.string()
        h["Maps"] = [self._vmp_map(s) for _ in range(h["NrOfSubMaps"])]
        if h["NrOfTimePoints"]:
            h["ComponentTimeCourses"] = s.array(h["NrOfSubMaps"] * h["NrOfTimePoints"],
                                                "f4").reshape(h["NrOfSubMaps"], -1).tolist()
        if h["NrOfComponentParams"]:
            params = [s.string() for _ in range(h["NrOfComponentParams"])]
            values = s.array(h["NrOfSubMaps"] * len(params), "f4").reshape(-1, len(params))
            h["ComponentParams"] = {p: values[:, i].tolist() for i, p in enumerate(params)}
        dims = self._box(h, h["Resolution"])
        values = s.array(h["NrOfSubMaps"] * int(np.prod(dims)), "f4")
        data = values.reshape(h["NrOfSubMaps"], *dims[::-1]).transpose(3, 2, 1, 0)
        cube = max(h["DimX"], h["DimY"], h["DimZ"])
        notes = self._grid_notes(h, cube, stated=True)
        if any(m["TypeOfMap"] == 3 for m in h["Maps"]):
            notes += ("cross-correlation maps as stored: integer part lag, fraction correlation",)
        return MRImage(data, self._grid(h, h["Resolution"], cube), Meta(self.plain(h)),
                       self.provenance(sniff, notes), name, ("map",))

    @staticmethod
    def _vmp_map(s: BvStream) -> dict:
        m = {"TypeOfMap": s.read("i")}
        m["MapThreshold"], m["UpperThreshold"] = s.read("2f")
        m["MapName"] = s.string()
        m.update(zip(("RGBPositiveMin", "RGBPositiveMax", "RGBNegativeMin", "RGBNegativeMax"),
                     (s.rgb() for _ in range(4)), strict=True))
        m["UseVMPColor"] = s.read("B")
        m["LUTFileName"] = s.string()
        m["TransparentColorFactor"] = s.read("f")
        if m["TypeOfMap"] == 3:
            m.update(zip(("NrOfLags", "DisplayMinLag", "DisplayMaxLag", "ShowCorrelationOrLag"),
                         s.read("4i"), strict=True))
        m["ClusterSizeThreshold"] = s.read("i")
        m["EnableClusterSizeThreshold"] = s.read("B")
        m["ShowValuesAboveUpperThreshold"], m["DF1"], m["DF2"] = s.read("3i")
        m["ShowPosNegValues"] = s.read("B")
        m["NrOfUsedVoxels"], size = s.read("2i")
        m["FDRTable"] = s.array(3 * size, "f4").reshape(size, 3).tolist()
        m["UseFDRTableIndex"] = s.read("i")
        return m

    @staticmethod
    def _box(h: dict, resolution: int) -> tuple[int, int, int]:
        """Grid dimensions (X, Y, Z) of a box of VMR voxels at a resolution multiplier."""
        if resolution <= 0:
            raise ReaderError(f"resolution multiplier {resolution}")
        return tuple((h[f"{a}End"] - h[f"{a}Start"]) // resolution for a in "XYZ")

    @staticmethod
    def _grid(h: dict, resolution: int, cube: int) -> np.ndarray:
        """Affine of a box grid: voxel i covers VMR voxels start + resolution * i onwards, its
        centre (resolution - 1) / 2 further."""
        start = tuple(h[f"{a}Start"] + (resolution - 1) / 2.0 for a in "XYZ")
        return bv_affine(start, resolution, cube, (1.0, 1.0, 1.0), h.get("LeftRightConvention", 1))

    @staticmethod
    def _grid_notes(h: dict, cube: int, stated: bool = False) -> tuple[str, ...]:
        space = SPACES.get(h.get("ReferenceSpace", 0), "unknown")
        notes = (f"affine in BrainVoyager {space} space on 1 mm VMR voxels, framing-cube centre "
                 "at 0",)
        if not stated:
            notes += (f"framing cube of the VMR not stated: {cube} assumed",)
        if h.get("LeftRightConvention", 1) not in (1, 2):
            notes += ("left-right convention not stated: radiological assumed",)
        return notes


#**************************************************************************************************#
#                                    BrainVoyagerSurfaceReader                                     #
#**************************************************************************************************#
#                                                                                                  #
# An SRF mesh holds vertex coordinates and normals on BrainVoyager's internal axes in VMR voxel    #
# mm, the mesh centre (for display), curvature colours, a colour code per vertex, neighbour lists, #
# triangles and triangle strips. Vertices become RAS mm with the framing-cube centre at 0 as for   #
# the VMR (R = c - Z as in the radiological convention, A = c - X, S = c - Y; c = 128 mm, the      #
# default cube, as the SRF does not name it); the normals are turned the same way;                 #
# neighbour lists are not kept (the triangles give them). SMP maps, MTC time courses and SSM       #
# vertex mappings are per-vertex data of a mesh they name (no vertices of their own).              #
#                                                                                                  #
#**************************************************************************************************#
class BrainVoyagerSurfaceReader(Reader):
    FORMATS = (Format.BRAINVOYAGER_SRF, Format.BRAINVOYAGER_SMP, Format.BRAINVOYAGER_MTC,
               Format.BRAINVOYAGER_SSM)
    PACKAGES = ("miox",)
    SPACE = "BrainVoyager system space, RAS mm, framing-cube centre at 0"
    VERTICES = "BrainVoyager mesh vertices"

    def read(self, sniff: Sniff) -> list[Surface]:
        stream = BvStream(read_bytes(sniff))
        name = sniff.path.name.removesuffix(".gz").rsplit(".", 1)[0]
        reader = {Format.BRAINVOYAGER_SRF: self._srf, Format.BRAINVOYAGER_SMP: self._smp,
                  Format.BRAINVOYAGER_MTC: self._mtc, Format.BRAINVOYAGER_SSM: self._ssm}
        try:
            return [reader[sniff.format](sniff, stream, name)]
        except (struct.error, ValueError) as e:
            raise ReaderError(f"{sniff.path}: not a readable BrainVoyager file ({e})") from e

    def _srf(self, sniff: Sniff, s: BvStream, name: str) -> Surface:
        h: dict = {"version": s.read("f"), "reserved": s.read("i")}
        h["NrOfVertices"], h["NrOfTriangles"] = s.read("2i")
        h["MeshCenter"] = s.read("3f")
        n = h["NrOfVertices"]
        coordinates = s.array(3 * n, "f4").reshape(3, n).T.astype(np.float64)
        normals = s.array(3 * n, "f4").reshape(3, n).T.astype(np.float64)
        h["ConvexCurvatureRGBA"], h["ConcaveCurvatureRGBA"] = s.read("4f"), s.read("4f")
        colours = s.array(n, "i4").copy()
        for _ in range(n):
            s.array(s.read("i"), "i4")
        faces = s.array(3 * h["NrOfTriangles"], "i4").reshape(-1, 3).astype(np.int64)
        h["NrOfStripElements"] = s.read("i")
        h["Strips"] = s.array(h["NrOfStripElements"], "i4").tolist()
        h["MTC"] = s.string() if s.left else ""
        vertices = (DEFAULT_CUBE / 2.0 - coordinates)[:, [2, 0, 1]]
        vertex_data = {"normal": -normals[:, [2, 0, 1]], "color_code": colours}
        notes = ("neighbour lists not kept (given by the triangles)",
                 "no left-right convention stated: radiological assumed",
                 f"framing cube of the VMR not stated: {DEFAULT_CUBE} assumed")
        return Surface(vertices, faces, self.SPACE, vertex_data, {}, Meta(self.plain(h)),
                       self.provenance(sniff, notes), name)

    def _smp(self, sniff: Sniff, s: BvStream, name: str) -> Surface:
        h: dict = {"version": s.read("h"), "NrOfVertices": s.read("i"), "NrOfMaps": s.read("h")}
        h["SRF"] = s.string()
        version, maps, vertex_data = h["version"], [], {}
        for _ in range(h["NrOfMaps"]):
            m = {"TypeOfMap": s.read("i")}
            if version >= 3 and m["TypeOfMap"] == 3:
                m.update(zip(("NrOfLags", "MinLag", "MaxLag", "Overlay"), s.read("4i"),
                             strict=True))
            m["ClusterSize"] = s.read("i")
            m["ClusterCheck"] = s.read("B")
            m["ThresholdMin"], m["ThresholdMax"] = s.read("2f")
            if version >= 4:
                m["IncludeGreaterThanMax"] = s.read("i")
            m["DF1"], m["DF2"] = s.read("2i")
            if version >= 5:
                m["ShowPositiveNegative"] = s.read("i")
            m["BonferroniValue"] = s.read("i")
            if version >= 2:
                m["RGBPositiveMin"], m["RGBPositiveMax"] = s.rgb(), s.rgb()
                if version >= 4:
                    m["RGBNegativeMin"], m["RGBNegativeMax"] = s.rgb(), s.rgb()
                m["UseRGB"] = s.read("B")
                if version >= 5:
                    m["LUTFile"] = s.string()
                m["Transparency"] = s.read("f")
            m["Name"] = s.string()
            key = m["Name"] if m["Name"] not in vertex_data else f"{m['Name']} ({len(maps)})"
            vertex_data[key] = s.array(h["NrOfVertices"], "f4").copy()
            maps.append(m)
        h["Maps"] = maps
        return Surface(None, None, self.VERTICES, vertex_data, {}, Meta(self.plain(h)),
                       self.provenance(sniff), name)

    def _mtc(self, sniff: Sniff, s: BvStream, name: str) -> Surface:
        h: dict = {"version": s.read("i"), "NrOfVertices": s.read("i"),
                   "NrOfTimePoints": s.read("i"), "VTC": s.string(), "Protocol": s.string()}
        h["HemodynamicDelay"] = s.read("i")
        h["TR"], h["HrfDelta"], h["HrfTau"] = s.read("3f")
        h["SegmentSize"], h["SegmentOffset"] = s.read("2i")
        h["DataType"] = s.read("B")
        if h["DataType"] != 1:
            raise ReaderError(f"{sniff.path}: MTC data type {h['DataType']} (only 1, float)")
        values = s.array(h["NrOfVertices"] * h["NrOfTimePoints"], "f4")
        meta = Meta(self.plain(h))
        meta.set("RepetitionTime", h["TR"] / 1000.0, "MTC header: TR (ms)")
        return Surface(None, None, self.VERTICES,
                       {"time_course": values.reshape(h["NrOfVertices"], -1)}, {}, meta,
                       self.provenance(sniff), name)

    def _ssm(self, sniff: Sniff, s: BvStream, name: str) -> Surface:
        h = {"version": s.read("h"), "NrOfVertices": s.read("i"),
             "NrOfTargetVertices": s.read("i")}
        mapping = s.array(h["NrOfVertices"], "i4").astype(np.int64)
        return Surface(None, None, self.VERTICES, {"target_vertex": mapping}, {},
                       Meta(self.plain(h)), self.provenance(sniff), name)
