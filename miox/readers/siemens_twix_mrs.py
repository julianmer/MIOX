####################################################################################################
#                                       siemens_twix_mrs.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 17/11/26                                                                                #
#                                                                                                  #
# Purpose: Siemens twix spectroscopy: the data streams pyMapVBVD reads, laid out per sequence,     #
#          with the voxel or grid geometry and the keys of the measurement header.                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# own
from miox import __version__
from miox.readers.base import ReaderError
from miox.readers.siemens_rda import SiemensRdaReader


#**************************************************************************************************#
#                                           TwixSpectrum                                           #
#**************************************************************************************************#
#                                                                                                  #
# One dataset of a twix stream: name, data (x, y, z, points[, ...]), dwell time, affine, keys as   #
# (value, field) and the dimension tags from the 5th on (a tag or (tag, info, per-index header)).  #
#                                                                                                  #
#**************************************************************************************************#
@dataclass
class TwixSpectrum:
    name: str
    data: np.ndarray
    dwelltime: float
    affine: np.ndarray
    values: dict
    tags: tuple = ()
    notes: list = field(default_factory=list)


#**************************************************************************************************#
#                                            SiemensGsl                                            #
#**************************************************************************************************#
#                                                                                                  #
# The slice classification and read/phase directions of Siemens' GSL library (fGSLClassOri,        #
# fGSLCalcPRS): a normal is sagittal, coronal or transverse by its largest component, ties going   #
# to transverse before coronal before sagittal; the phase direction is perpendicular to the normal #
# and to one patient axis (x for transverse slices, z for coronal and sagittal ones), turned about #
# the normal by the in-plane rotation, and read = normal x phase.                                  #
#                                                                                                  #
#**************************************************************************************************#
class SiemensGsl:
    SAGITTAL, CORONAL, TRANSVERSE = 0, 1, 2

    @classmethod
    def orientation(cls, normal) -> int:
        s, c, t = (abs(float(v)) for v in normal)
        sc, st, ct = np.isclose(s, c), np.isclose(s, t), np.isclose(c, t)
        if ((sc and st) or (sc and s < t) or (st and s > c) or (ct and c > s)
                or (s > c and s < t) or (s < c and c < t) or (s < t and t > c)
                or (c < t and t > s)):
            return cls.TRANSVERSE
        if ((sc and s > t) or (st and s < c) or (s < c and c > t) or (s > t and s < c)
                or (s < t and t < c)):
            return cls.CORONAL
        if ((ct and c < s) or (s > c and s > t) or (c > t and c < s) or (c < t and t < s)):
            return cls.SAGITTAL
        raise ReaderError(f"invalid slice orientation: normal {list(normal)}")

    @classmethod
    def directions(cls, normal, rotation: float) -> tuple[np.ndarray, np.ndarray]:
        """(phase, read) direction vectors of a slice normal and in-plane rotation (radians)."""
        gs = np.asarray(normal, dtype=float)
        orientation = cls.orientation(gs)
        gp = np.zeros(3)
        if orientation == cls.TRANSVERSE:
            norm = np.sqrt(1.0 / (gs[1] * gs[1] + gs[2] * gs[2]))
            gp[1], gp[2] = gs[2] * norm, -gs[1] * norm
        elif orientation == cls.CORONAL:
            norm = np.sqrt(1.0 / (gs[0] * gs[0] + gs[1] * gs[1]))
            gp[0], gp[1] = gs[1] * norm, -gs[0] * norm
        else:
            norm = np.sqrt(1.0 / (gs[0] * gs[0] + gs[1] * gs[1]))
            gp[0], gp[1] = -gs[1] * norm, gs[0] * norm
        gr = cls._cross(gs, gp)
        gp = np.cos(rotation) * gp - np.sin(rotation) * gr
        return gp, cls._cross(gs, gp)

    @staticmethod
    def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                         a[0] * b[1] - a[1] * b[0]])


#**************************************************************************************************#
#                                             TwixMrs                                              #
#**************************************************************************************************#
#                                                                                                  #
# Spectroscopy of one twix measurement. The measurement is a grid (MRSI) when the final matrix     #
# holds more than one voxel, unlocalised when the sequence is ``fid``, else a single voxel. A      #
# stream (pyMapVBVD, squeezed, oversampling kept) is conjugated (NIfTI-MRS stores the conjugate of #
# the scanner's FIDs) and tagged by its loop counters: Cha DIM_COIL, Ave, Set and Rep DIM_DYN, Eco #
# DIM_EDIT (VB: Set and Rep DIM_DYN, Lin, Phs and Seg spatial), any other DIM_USER_n. XA product   #
# sequences (svs_se, svs_st, svs_slaser) keep reference scans in the first Phs index: split off as #
# _ref, unsuppressed water (their all-zero transients left out). Sequences laid out by their own   #
# rules: MEGA, HERMES and HERCULES (mgs_svs_ed; smm_svs_herc on XA) split the transients into 2 or #
# 4 edit conditions by the sequence mode (WIP alFree[7]) with the edit pulses of WIP adFree[8-11]; #
# HYPER / ISTHMUS into its two water references, short-TE PRESS and HERCULES blocks (XA50/60       #
# smm_svs_isthmus_gls with its receiver phase corrected); svs_slaser(voi)_dkd(2) into the scan and #
# its RF-off reference blocks by lAutoRefScanMode. Geometry: the VoI (grids: the slice) normal,    #
# in-plane rotation, field of view and position plus the table offset, turned into DICOM           #
# directions by the GSL rules; grids as the scanner's DICOM orders them (readout mirrored for      #
# sagittal and transverse slices, read and phase swapped for transverse ones; the position at the  #
# grid corner, half a voxel in). Keys from the Meas buffer (VB to VE) or MeasYaps and Dicom (XA),  #
# times in s.                                                                                      #
#                                                                                                  #
#**************************************************************************************************#
class TwixMrs:
    TAGS = {
        "vb": {"Lin": "x", "Phs": "y", "Seg": "z", "Cha": "DIM_COIL", "Set": "DIM_DYN",
               "Rep": "DIM_DYN"},
        "vd": {"Lin": "x", "Par": "y", "Seg": "z", "Cha": "DIM_COIL", "Ave": "DIM_DYN",
               "Set": "DIM_DYN", "Rep": "DIM_DYN", "Eco": "DIM_EDIT"},
    }
    XA_PRODUCT = ("svs_se", "svs_st", "svs_slaser")
    NO_FOV = 10000.0  # mm: a voxel without stated size
    HYPER = (  # block, name suffix
        ("35ms Water Reference", "_ref_short_te"), ("80ms Water Reference", "_ref_edited"),
        ("35ms PRESS", "_short_te"), ("80ms HERCULES", "_edited"))
    HYPER_EDITED_TE = 0.080
    HYPER_TX_OFFSET = -1.7  # ppm, the metabolite blocks
    ISTHMUS_PHASE = [0, 0, 0, 0, 180, 180, 180, 180, 0, 0, 0, 0, 180, 180, 180, 180,
                     180, 180, 180, 180, 0, 0, 0, 0, 180, 180, 180, 180, 0, 0, 0, 0]  # degrees

    def __init__(self, twix, path: Path):
        self.twix, self.path = twix, path
        self.hdr = twix["hdr"]
        self.yaps = self.hdr["MeasYaps"]
        self.notes: list[str] = []

    # --- measurement -----------------------------------------------------------------------------

    @property
    def software(self) -> str:
        """'xa' or 'vx' (VA to VE)."""
        version = str(self.hdr["Dicom"]["SoftwareVersions"])
        if re.search(r"syngo MR XA\d{2}", version):
            return "xa"
        if re.search(r"syngo MR [A-Z]\d{2}", version):
            return "vx"
        raise ReaderError(f"unknown Siemens software version: {version}")

    @property
    def sequence(self) -> str:
        return str(self.hdr["Meas"]["tSequenceString"])

    @property
    def sequence_file(self) -> str:
        return str(self.hdr["Meas"]["tSequenceFileName"])

    def matrix(self, key: str) -> int:
        """lFinalMatrixSize<key>, 0 where unset."""
        value = self.hdr["Meas"].get(f"lFinalMatrixSize{key}")
        return int(value) if value else 0

    @property
    def kind(self) -> str:
        """'mrsi', 'fid' (unlocalised) or 'svs'."""
        if self.sequence == "fid":
            return "fid"
        phase, read, slices = (self.matrix(k) for k in ("Phase", "Read", "Slice"))
        voxels = slices * phase * read if phase and read else (slices or 1)
        return "mrsi" if voxels > 1 else "svs" if voxels == 1 else "fid"

    @property
    def grid(self) -> list[int]:
        """Final matrix (phase, read, slice), 1 where unset."""
        return [self.matrix(k) or 1 for k in ("Phase", "Read", "Slice")]

    def dwelltime(self) -> float:
        dwelltime = self.yaps[("sRXSPEC", "alDwellTime", "0")] / 1e9
        return dwelltime * 2 if self.kind == "mrsi" else dwelltime  # grids without oversampling

    # --- streams ---------------------------------------------------------------------------------

    def spectra(self, stream: str) -> list[TwixSpectrum]:
        """The datasets of one stream, named after the file (and ``_<stream>`` but for image)."""
        affine = self.affine()
        values = self.values(stream)
        if self.kind == "mrsi":
            data = np.zeros(self._grid_shape() + [int(self.hdr["Meas"]["lVectorSize"])],
                            dtype=complex)
            spectra = [TwixSpectrum("_empty", data, self.dwelltime(), affine, values)]
        else:
            spectra = self._single(stream, affine, values)
        suffix = "" if stream == "image" else f"_{stream}"
        for s in spectra:
            s.name = self.path.stem + s.name + suffix
            if len(s.tags) < s.data.ndim - 4:
                raise ReaderError(f"{s.name}: {s.data.ndim - 4} dimensions after the points, "
                                  f"{len(s.tags)} with a counter")
        return spectra

    def _grid_shape(self) -> list[int]:
        phase, read, slices = self.grid
        swapped = SiemensGsl.orientation(self._normal(True)) == SiemensGsl.TRANSVERSE
        return [read, phase, slices] if swapped else [phase, read, slices]

    def _single(self, stream: str, affine: np.ndarray, values: dict) -> list[TwixSpectrum]:
        """A single voxel or unlocalised stream: (points, counters...) laid out per sequence."""
        source = self.twix[stream]
        source.flagRemoveOS, source.squeeze = False, True
        data = source[""]
        dims = list(source.sqzDims)
        if dims[0] != "Col":
            raise ReaderError(f"stream '{stream}': Col is not the first dimension ({dims})")
        if self.kind == "svs" and self._isthmus_phase_error():
            data = self._isthmus_phase(data)
        data = data.conj()
        defaults = self.TAGS[source.softwareVersion]
        order = dims[1:]
        dwelltime = self.dwelltime()
        reference = None
        if self.software == "xa" and self.sequence in self.XA_PRODUCT and "Phs" in order:
            data, reference, order = self._xa_reference(data, order)
        tags, user = [], 0
        for counter in order:
            if counter in defaults:
                tags.append(defaults[counter])
            else:
                tags.append(f"DIM_USER_{user}")
                user += 1

        def one(name, array, keys, dim_tags) -> TwixSpectrum:
            return TwixSpectrum(name, array.reshape((1, 1, 1) + array.shape), dwelltime, affine,
                                keys, tuple(dim_tags[: array.ndim - 1]))

        ref = []
        if reference is not None:
            ref_tags = ["DIM_COIL", "DIM_DYN"] if reference.ndim > 2 else ["DIM_COIL"]
            ref = [one("_ref", reference, values | {"WaterSuppressed": (False, None)}, ref_tags)]
        if self.kind == "fid":
            return self._split(data, order, tags, one, values) + ref
        file = self.sequence_file
        if self.sequence == "mgs_svs_ed" or (self.software == "xa" and "smm_svs_herc" in file
                                             and not self._hyper()):
            data, values, tags = self._edited(data, values, tags)
        elif self._hyper():
            return ref + [one(suffix, block, block_values, block_tags)
                          for suffix, block, block_values, block_tags
                          in self._hyper_blocks(data, values, tags) if block.ndim <= 4]
        elif re.search(r"svs_slaser(voi)?_dkd", file, re.IGNORECASE):
            return ref + [TwixSpectrum(suffix, block, dwelltime, affine, values,
                                       tuple(tags[: block.ndim - 4]))
                          for suffix, block in self._slaser_dkd(data)]
        return ref + self._split(data, order, tags, one, values)

    @staticmethod
    def _split(data, order, tags, one, values) -> list[TwixSpectrum]:
        """One dataset, or one per index of the counters beyond the three NIfTI-MRS holds."""
        if data.ndim <= 4:
            return [one("", data, values, tags)]
        spectra = []
        for index in np.ndindex(data.shape[4:]):
            name = "".join(f"_{order[3 + i]}{n:03d}" for i, n in enumerate(index))
            spectra.append(one(name, data[(slice(None),) * 4 + index], values, tags))
        return spectra

    def _xa_reference(self, data: np.ndarray, order: list[str]) -> tuple:
        """XA product sequences: the first Phs index holds the reference scans."""
        axis = order.index("Phs") + 1
        reference = np.take(data, [0], axis=axis)
        rest = np.take(data, range(1, data.shape[axis]), axis=axis)
        if rest.shape[axis] == 1:
            order = [c for c in order if c != "Phs"]
            data = rest.squeeze()
        reference = reference.squeeze()
        zero = reference[0, :, :] == 0.0
        reference = reference[:, ~zero].reshape(reference.shape[0], reference.shape[1], -1)
        return data, reference, order

    # --- sequences -------------------------------------------------------------------------------

    def _wip(self, kind: str, index: int, fallback: Any = KeyError) -> Any:
        key = ("sWipMemBlock", kind, str(index))
        if fallback is KeyError:
            return self.hdr["Phoenix"][key]
        return self.hdr["Phoenix"].get(key, fallback)

    def _edited(self, data: np.ndarray, values: dict, tags: list) -> tuple:
        """MEGA (mode 0), HERMES (1, 2, 4) and HERCULES (3): the transients split into the edit
        conditions, the condition the faster index."""
        mode = self._wip("alFree", 7, 0.0)
        duration = self._wip("alFree", 12, None)
        duration = duration / 1e6 if duration else duration
        off = self._wip("adFree", 11, [])
        p1, p2, p3 = (self._wip("adFree", i) for i in (8, 9, 10))
        abcd = ["A", "B", "C", "D"]
        modes = {
            0.0: ("MEGA-EDITED j-difference editing, two conditions", ["ON", "OFF"],
                  [p1, off], duration),
            1.0: ("HERMES j-difference editing, GABA GSH, four conditions", abcd,
                  [p1, off, p2, [p1, p2]], 0.02),
            2.0: ("HERMES j-difference editing, GABA GSH EtOH, four conditions", abcd,
                  [[p1, p3], off, [p2, p3], [p1, p2]], 0.02),
            3.0: ("HERCULES j-difference editing, four conditions", abcd,
                  [[p1, p2], [off, p2], p1, off], 0.02),
            4.0: ("HERMES j-difference editing, GABA LAC, four conditions", abcd,
                  [p1, off, p2, [p1, p2]], 0.02),
        }
        if mode not in modes:
            raise ReaderError(f"unknown edit sequence mode {mode}")
        info, conditions, pulses, length = modes[mode]
        pulse = {c: {"PulseOffset": p, "PulseDuration": length}
                 for c, p in zip(conditions, pulses, strict=True)}
        data = self._conditions(data, len(conditions))
        tags = list(tags)
        tags.insert(data.ndim - 3, ("DIM_EDIT", info, {"EditCondition": conditions}))
        return data, values | {"EditPulse": (pulse, "Phoenix/sWipMemBlock.adFree[8]")}, tags

    @staticmethod
    def _conditions(data: np.ndarray, n: int) -> np.ndarray:
        """The last axis (transients) as (conditions, rest), the condition changing fastest."""
        shape = data.shape[:-1] + (n, -1)
        return data.T.reshape(shape[::-1]).T

    def _hyper(self) -> bool:
        return any(x in self.sequence_file for x in ("smm_svs_herc_hyper", "svs_isthmus"))

    def _hyper_blocks(self, data: np.ndarray, values: dict, tags: list) -> list[tuple]:
        """HYPER / ISTHMUS (transients last): water references every 66th transient from 33 (short
        TE) and from 0 (80 ms), PRESS transients 1 to 32, HERCULES the rest but every 33rd, its
        conditions A to D in turn."""
        short_te = self._wip("alFree", 6) / 1e6
        duration = self._wip("alFree", 12) / 1e6
        p1, p2, p4 = (self._wip("adFree", i) for i in (8, 9, 11))
        mask = np.ones(data.shape[-1], dtype=bool)
        mask[::33], mask[:33] = False, False
        edited = data[..., mask]
        complete = edited.shape[-1] // 4 * 4
        if complete < edited.shape[-1]:
            self.notes.append(f"HERCULES block: {edited.shape[-1] - complete} transients of an "
                              "incomplete set of four conditions left out")
        edited = edited[..., :complete]
        pulse = {"A": {"PulseOffset": [p1, p2], "PulseDuration": duration},
                 "B": {"PulseOffset": [p4, p2], "PulseDuration": duration},
                 "C": {"PulseOffset": p1, "PulseDuration": duration},
                 "D": {"PulseOffset": p4, "PulseDuration": duration}}
        edited = self._conditions(edited, 4)
        edited_tags = list(tags)
        edited_tags.insert(edited.ndim - 3, ("DIM_EDIT", f"Hyper - {self.HYPER[3][0]}",
                                             {"EditCondition": ["A", "B", "C", "D"]}))
        water = {"WaterSuppressed": (False, None), "TxOffset": (0.0, None)}
        metabolites = {"WaterSuppressed": (True, None), "TxOffset": (self.HYPER_TX_OFFSET, None)}
        te = "Phoenix/sWipMemBlock.alFree[6]"
        return [
            (self.HYPER[0][1], data[:, :, 33::66], values | water | {"EchoTime": (short_te, te)},
             tags),
            (self.HYPER[1][1], data[:, :, 0::66],
             values | water | {"EchoTime": (self.HYPER_EDITED_TE, None)}, tags),
            (self.HYPER[2][1], data[:, :, 1:33],
             values | metabolites | {"EchoTime": (short_te, te)}, tags),
            (self.HYPER[3][1], edited,
             values | metabolites | {"EchoTime": (self.HYPER_EDITED_TE, None),
                                     "EditPulse": (pulse, "Phoenix/sWipMemBlock.adFree[8]")},
             edited_tags),
        ]

    def _isthmus_phase_error(self) -> bool:
        """XA50/60 smm_svs_isthmus_gls acquires its HERCULES transients with a wrong receiver
        phase (fixed in smm_svs_isthmus_gls2)."""
        version = str(self.hdr["Dicom"]["SoftwareVersions"])
        name = re.split(r"[\\/]", self.sequence_file)[-1]
        return any(v in version for v in ("XA50", "XA60")) and name == "smm_svs_isthmus_gls"

    def _isthmus_phase(self, data: np.ndarray) -> np.ndarray:
        """(points, coils, transients): the HERCULES transients (34 to 263 but every 33rd) turned
        back by the receiver phase pattern."""
        phase = np.tile(self.ISTHMUS_PHASE, 224 // len(self.ISTHMUS_PHASE)) * (np.pi / 180)
        index = np.arange(34, 264)
        index = index[index % 33 != 0]
        data = data.copy()
        data[:, :, index] *= np.exp(-1j * phase)[None, None, :]
        return data

    def _slaser_dkd(self, data: np.ndarray) -> list[tuple[str, np.ndarray]]:
        """svs_slaser(voi)_dkd(2): scan mode 8 places lAutoRefScanNo RF-off and then RF-, gradient-
        and OVS-off references before and after the lAverages transients, mode 2 (dkd2) RF-off
        references with VAPOR and OVS off; mode 1 and others keep the stream whole."""
        mode = self.yaps[("sSpecPara", "lAutoRefScanMode")]
        name = self.sequence_file.lower()
        whole = [("", data.reshape((1, 1, 1) + data.shape))]
        if mode == 1.0:
            return whole
        dkd = re.search(r"svs_slaser(voi)?_dkd2?$", name)
        if not (dkd and mode == 8.0) and not (re.search(r"svs_slaser(voi)?_dkd2", name)
                                             and mode == 2.0):
            return whole
        refs = int(self.yaps[("sSpecPara", "lAutoRefScanNo")])
        transients = int(self.yaps[("lAverages",)])
        shape = (1, 1, 1) + data.shape[:2] + (-1,)
        if mode == 8.0:
            start = 2 * refs
            off = np.concatenate((np.arange(refs), np.arange(-2 * refs, -refs)))
            all_off = np.concatenate((np.arange(refs, 2 * refs), np.arange(-refs, 0)))
            return [("", data[..., start:start + transients].reshape(shape)),
                    ("_rf_off", data[..., off].reshape(shape)),
                    ("_rf_grads_ovs_off", data[..., all_off].reshape(shape))]
        off = np.concatenate((np.arange(refs), np.arange(-refs, 0)))
        return [("", data[..., refs:refs + transients].reshape(shape)),
                ("_vapor_ovs_rfoff", data[..., off].reshape(shape))]

    # --- geometry --------------------------------------------------------------------------------

    def _pick(self, grid: bool, slice_key: tuple, voi_key: tuple, default: Any) -> Any:
        """The slice's value (grids), else the VoI's, else ``default``."""
        slice_key = ("sSliceArray", "asSlice", "0") + slice_key
        if grid and slice_key in self.yaps:
            return self.yaps[slice_key]
        return self.yaps.get(("sSpecPara", "sVoI") + voi_key, default)

    def _normal(self, grid: bool) -> np.ndarray:
        normal = np.array([self._pick(grid, ("sNormal", d), ("sNormal", d), 0.0)
                           for d in ("dSag", "dCor", "dTra")], dtype=float)
        if not normal.any():  # unlocalised
            normal[0] = 1.0
        return normal

    def affine(self) -> np.ndarray:
        groups = self.yaps.get(("sGroupArray", "asGroup", "0", "nSize"))
        if groups is not None and groups != 1.0:
            raise ReaderError("slice-selective spectroscopy with more than one slice")
        grid = self.kind == "mrsi"
        normal = self._normal(grid)
        rotation = self._pick(grid, ("dInPlaneRot",), ("dInPlaneRot",), 0.0)
        slice_fov = ("sSliceArray", "asSlice", "0", "dReadoutFOV")
        if grid and slice_fov in self.yaps:
            read_fov = self.yaps[slice_fov]
            phase_fov = self.yaps[("sSliceArray", "asSlice", "0", "dPhaseFOV")]
        elif ("sSpecPara", "sVoI", "dReadoutFOV") in self.yaps:
            read_fov = self.yaps[("sSpecPara", "sVoI", "dReadoutFOV")]
            phase_fov = self.yaps[("sSpecPara", "sVoI", "dPhaseFOV")]
        else:
            read_fov = phase_fov = self.NO_FOV
        thickness = self._pick(grid, ("dThickness",), ("dThickness",), self.NO_FOV)
        position = np.array([self._pick(grid, ("sPosition", d), ("sPosition", d), 0.0)
                             for d in ("dSag", "dCor", "dTra")], dtype=float)
        position += [self.yaps.get((f"lScanRegionPos{d}",), 0.0) for d in ("Sag", "Cor", "Tra")]
        phase, read = SiemensGsl.directions(normal, rotation)
        if not grid:
            return SiemensRdaReader.grid_affine(np.concatenate([read, phase]), position,
                                                [phase_fov, read_fov, thickness], 1, False)
        n_phase, n_read, n_slice = self.grid
        if n_slice > 1:  # 3D: the first slice's centre, the slice spacing
            position = position - normal * (thickness / 2 - thickness / n_slice / 2)
            thickness = thickness / n_slice
        orientation = SiemensGsl.orientation(normal)
        if orientation == SiemensGsl.TRANSVERSE:
            read = -read
            row, column = read, phase
            spacing = [phase_fov / n_phase, read_fov / n_read]
            corner = position - row * read_fov / 2 - column * phase_fov / 2
        else:
            if orientation == SiemensGsl.SAGITTAL:
                read = -read
            row, column = phase, read
            spacing = [read_fov / n_read, phase_fov / n_phase]
            corner = position - row * phase_fov / 2 - column * read_fov / 2
        return SiemensRdaReader.grid_affine(np.concatenate([row, column]), corner,
                                            spacing + [thickness], n_slice, True)

    # --- keys ------------------------------------------------------------------------------------

    def values(self, stream: str) -> dict:
        """Standard keys: (value, header field)."""
        if self.software == "xa":
            values = self._values_xa()
        else:
            try:
                values = self._values_vx()
            except KeyError:  # some VX headers are written the XA way
                values = self._values_xa()
        meas, config = self.hdr["Meas"], self.hdr["Config"]
        dicom = self.hdr["Dicom"]
        values |= {
            "EchoTime": (self.hdr["Phoenix"][("alTE", "0")] * 1e-6, "Phoenix/alTE[0]"),
            "Manufacturer": (dicom["Manufacturer"], "Dicom/Manufacturer"),
            "ManufacturersModelName": (dicom["ManufacturersModelName"],
                                       "Dicom/ManufacturersModelName"),
            "DeviceSerialNumber": (self._text(dicom["DeviceSerialNumber"]),
                                   "Dicom/DeviceSerialNumber"),
            "SoftwareVersions": (dicom["SoftwareVersions"], "Dicom/SoftwareVersions"),
            "InstitutionName": (dicom["InstitutionName"], "Dicom/InstitutionName"),
            "InstitutionAddress": (dicom["InstitutionAddress"], "Dicom/InstitutionAddress"),
            "SequenceName": (meas["tSequenceString"], "Meas/tSequenceString"),
            "ProtocolName": (dicom["tProtocolName"], "Dicom/tProtocolName"),
            "PatientDoB": (self._text(meas["PatientBirthDay"]), "Meas/PatientBirthDay"),
            "ConversionMethod": (f"miox {__version__}", None),
            "ConversionTime": (datetime.now().isoformat(timespec="milliseconds"), None),
            "OriginalFile": ([Path(self.twix[stream].filename).name], None),
            "kSpace": ([False, False, False], None),
            "PulseSequenceFile": ({"Value": config["SequenceFileName"],
                                   "Description": "Sequence binary path."},
                                  "Config/SequenceFileName"),
            "IceProgramFile": ({"Value": meas["tICEProgramName"],
                                "Description": "Reconstruction binary path."},
                               "Meas/tICEProgramName"),
        }
        try:
            values["PatientWeight"] = (float(meas["flUsedPatientWeight"]),
                                       "Meas/flUsedPatientWeight")
        except ValueError:
            pass
        rx = (("sCoilSelectMeas", "aRxCoilSelectData", "0", "asList", "0", "sCoilElementID",
               "tCoilID"), ("asCoilSelectMeas", "0", "asList", "0", "sCoilElementID", "tCoilID"))
        key = next((k for k in rx if k in self.yaps), None)
        if key is not None:
            values["RxCoil"] = (str(self.yaps[key]).strip('"'), f"MeasYaps/{self._dotted(key)}")
        inversion = self.yaps.get(("alTI", "0"))
        if isinstance(inversion, (int, float)):
            values["InversionTime"] = (inversion * 1e-6, "MeasYaps/alTI[0]")
        return values

    def _values_vx(self) -> dict:
        meas = self.hdr["Meas"]
        tr = "TR_Time" if "TR_Time" in meas else "TR"
        offset = meas.get("dDeltaFrequency", 0.0)
        return {
            "SpectrometerFrequency": ([meas["Frequency"] / 1e6], "Meas/Frequency"),
            "ResonantNucleus": ([meas["ResonantNucleus"]], "Meas/ResonantNucleus"),
            "RepetitionTime": (float(meas[tr] / 1e6), f"Meas/{tr}"),
            "ExcitationFlipAngle": (float(meas["FlipAngle"]), "Meas/FlipAngle"),
            "TxOffset": (0.0 if offset == "" else offset, "Meas/dDeltaFrequency"),
            "PatientPosition": (meas["PatientPosition"], "Meas/PatientPosition"),
            "PatientName": (meas["PatientName"], "Meas/PatientName"),
            "PatientSex": ({1: "M", 2: "F"}.get(meas["PatientSex"], "O"), "Meas/PatientSex"),
        }

    def _values_xa(self) -> dict:
        meas = self.hdr["Meas"]
        nucleus = str(self.yaps[("sTXSPEC", "asNucleusInfo", "0", "tNucleus")]).strip('"')
        offset = self.yaps.get(("sSpecPara", "dDeltaFrequency"), 0.0)
        values = {
            "SpectrometerFrequency": ([self.hdr["Dicom"]["lFrequency"] / 1e6], "Dicom/lFrequency"),
            "ResonantNucleus": ([nucleus], "MeasYaps/sTXSPEC.asNucleusInfo[0].tNucleus"),
            "RepetitionTime": (float(self.yaps[("alTR", "0")] / 1e6), "MeasYaps/alTR[0]"),
            "ExcitationFlipAngle": (float(self.yaps[("adFlipAngleDegree", "0")]),
                                    "MeasYaps/adFlipAngleDegree[0]"),
            "TxOffset": (0.0 if offset == "" else offset, "MeasYaps/sSpecPara.dDeltaFrequency"),
            "PatientPosition": (meas["tPatientPosition"], "Meas/tPatientPosition"),
            "PatientName": (meas["tPatientName"], "Meas/tPatientName"),
        }
        sex = {1: "M", 2: "F", 3: "O"}.get(meas["PatientSex"])
        if sex is None:
            raise ReaderError(f"Meas/PatientSex is {meas['PatientSex']}, not 1, 2 or 3")
        values["PatientSex"] = (sex, "Meas/PatientSex")
        tx = ("sCoilSelectMeas", "aTxCoilSelectData", "0", "asList", "0", "sCoilElementID",
              "tCoilID")
        if tx in self.yaps:
            values["TxCoil"] = (str(self.yaps[tx]).strip('"'), f"MeasYaps/{self._dotted(tx)}")
        return values

    @staticmethod
    def _text(value: Any) -> str:
        """A header number as text: integers without the '.0' of a float."""
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    @staticmethod
    def _dotted(key: tuple) -> str:
        """Tuple keys in Siemens notation: ('alTE', '0') becomes alTE[0]."""
        return "".join(f"[{p}]" if p.isdigit() else f".{p}" for p in key).lstrip(".")
