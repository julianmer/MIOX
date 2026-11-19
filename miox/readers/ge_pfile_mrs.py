####################################################################################################
#                                         ge_pfile_mrs.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 16/11/26                                                                                #
#                                                                                                  #
# Purpose: GE P-file spectroscopy: the data block laid out per sequence, voxel geometry and keys,  #
#          from the rdb header fields (GeRdbHeader).                                               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# own
from miox import __version__
from miox.readers.base import ReaderError


#**************************************************************************************************#
#                                            GeSpectrum                                            #
#**************************************************************************************************#
#                                                                                                  #
# One dataset of a P-file: name suffix, data (x, y, z, points[, ...]), keys as (value, field) and  #
# dimension tags from the 5th on.                                                                  #
#                                                                                                  #
#**************************************************************************************************#
@dataclass
class GeSpectrum:
    suffix: str
    data: np.ndarray
    values: dict
    tags: tuple = ()
    notes: list = field(default_factory=list)


#**************************************************************************************************#
#                                          GePfileLayout                                           #
#**************************************************************************************************#
#                                                                                                  #
# Counts and geometry of the data block, as SIVIC's svkGEPFileMapper defines them (the layout      #
# spec2nii follows): coils from the receiver ranges of rh_dab, voxels from xcsi/ycsi/zcsi (one for #
# single voxel files; x and y swapped when the frequency direction is not 1), time points and      #
# dummy scans from the raw pass size. The voxel box: size from user8-10 (revision > 9; assigned to #
# the axis each direction cosine points along most) or roilen, centre user11-13, orientation from  #
# the corner points of the image header (third axis the negated cross product).                    #
#                                                                                                  #
#**************************************************************************************************#
class GePfileLayout:
    def __init__(self, hdr: Any, version: float):
        self.hdr, self.version = hdr, version

    @property
    def swap(self) -> bool:
        return self.hdr.rhi_freq_dir != 1

    @property
    def coils(self) -> int:
        coils = 0
        for i in range(4):
            start = getattr(self.hdr, f"rhr_rh_dab[{i}]_start_rcv")
            stop = getattr(self.hdr, f"rhr_rh_dab[{i}]_stop_rcv")
            if start != 0 or stop != 0:
                coils += stop - start + 1
        return int(coils) or 1

    @property
    def voxels(self) -> np.ndarray:
        if self.hdr.rhr_rh_file_contents == 0:
            n = np.array([1, 1, 1])
        else:
            n = np.array([int(self.hdr.rhr_xcsi), int(self.hdr.rhr_ycsi), int(self.hdr.rhr_zcsi)])
        if self.swap:
            n[0], n[1] = n[1], n[0]
        return n

    @property
    def kspace_points(self) -> int:
        return int(np.prod(self.voxels))

    @property
    def time_points(self) -> int:
        hdr = self.hdr
        per_frame = float(self.coils) * 2 * float(hdr.rhr_rh_point_size) * float(
            hdr.rhr_rh_frame_size)
        points = int(float(hdr.rhr_rh_raw_pass_size) / per_frame - 1) / float(self.kspace_points)
        return int(points) if points > 0 else 1

    @property
    def dummy_scans(self) -> int:
        hdr = self.hdr
        word = 2 * hdr.rhr_rh_point_size * hdr.rhr_rh_frame_size
        rest = hdr.rhr_rh_raw_pass_size - self.coils * self.time_points * self.kspace_points * word
        return int(rest / word)

    @property
    def chop(self) -> bool:
        return math.ceil(self.hdr.rhi_nex) * self.hdr.rhi_numecho <= 1

    @property
    def is_2d(self) -> bool:
        dims = self.hdr.rhr_csi_dims
        if dims == 0:
            dims = sum(int(v >= 0) for v in (self.hdr.rhr_xcsi, self.hdr.rhr_ycsi,
                                             self.hdr.rhr_zcsi))
        return dims == 2

    @property
    def fov(self) -> np.ndarray:
        hdr, n = self.hdr, self.voxels
        if self.version > 9:
            third = hdr.rhi_user10 if self.is_2d else hdr.rhi_scanspacing * hdr.rhr_zcsi
            fov = np.array([hdr.rhi_dfov, hdr.rhi_dfov, third], float)
        else:
            fov = np.array([hdr.rhr_rh_user7, hdr.rhr_rh_user8, hdr.rhr_rh_user9], float)
        if self.version > 9 and n[0] != n[1]:  # anisotropic: the next powers of two
            xdim = int(pow(2, math.ceil(math.log(n[0], 2))))
            ydim = int(pow(2, math.ceil(math.log(n[1], 2))))
            resolution = hdr.rhi_dfov / max(xdim, ydim)
            fov[0], fov[1] = resolution * xdim, resolution * ydim
        elif self.swap:
            fov[0], fov[1] = fov[1], fov[0]
        return fov

    def voxel_spacing(self, from_fov: bool = False) -> np.ndarray:
        """user19 (revision > 9) for every axis where it is set, else FOV over voxels; the probe-sl
        CSI layout always takes the FOV."""
        user19 = self.hdr.rhi_user19
        if not from_fov and user19 > 0 and self.version > 9:
            return np.array([user19] * 3, float)
        return self.fov / self.voxels

    @property
    def dcos(self) -> np.ndarray:
        hdr = self.hdr
        dcos = np.zeros((3, 3))
        dcos[0] = [hdr.rhi_trhc_R - hdr.rhi_tlhc_R, hdr.rhi_trhc_A - hdr.rhi_tlhc_A,
                   hdr.rhi_trhc_S - hdr.rhi_tlhc_S]
        dcos[0] /= np.sqrt(np.sum(dcos[0] ** 2))
        dcos[1] = [hdr.rhi_brhc_R - hdr.rhi_trhc_R, hdr.rhi_brhc_A - hdr.rhi_trhc_A,
                   hdr.rhi_brhc_S - hdr.rhi_trhc_S]
        dcos[1] /= np.sqrt(np.sum(dcos[1] ** 2))
        dcos[2][0] = - dcos[0][1] * dcos[1][2] + dcos[0][2] * dcos[1][1]
        dcos[2][1] = - dcos[0][2] * dcos[1][0] + dcos[0][0] * dcos[1][2]
        dcos[2][2] = - dcos[0][0] * dcos[1][1] + dcos[0][1] * dcos[1][0]
        return dcos

    @property
    def box_centre(self) -> np.ndarray:
        return np.array([self.hdr.rhi_user11, self.hdr.rhi_user12, self.hdr.rhi_user13])

    @property
    def box_size(self) -> np.ndarray:
        size = np.zeros(3)
        if self.version > 9:
            dcos = self.dcos
            index = [int(np.argmax([abs(dcos[i][axis]) for i in range(3)])) for axis in range(3)]
            for axis, value in enumerate((self.hdr.rhi_user8, self.hdr.rhi_user9,
                                          self.hdr.rhi_user10)):
                size[index[axis]] = value
        else:
            size[:] = self.hdr.rhr_roilenx, self.hdr.rhr_roileny, self.hdr.rhr_roilenz
            if self.swap:
                size[0], size[1] = size[1], size[0]
        return size

    def affine(self) -> np.ndarray:
        """The single voxel's affine: box size along the direction cosines at the box centre."""
        affine = np.eye(4)
        dcos = self.dcos.T
        dcos[dcos == 0.0] = 0.0  # no -0.0
        affine[:3, :3] = dcos @ np.diag(self.box_size)
        affine[:3, 3] = self.box_centre
        return affine

    def affine_grid(self, spacing: np.ndarray) -> np.ndarray:
        """The spectroscopic imaging grid's affine: first voxel from the box centre."""
        dcos, n = self.dcos, self.voxels
        origin = self.box_centre.astype(float)
        for i in range(3):
            for j in range(3):
                origin[i] -= dcos[j][i] * spacing[j] * (n[j] / 2.0 - 0.5)
        affine = np.eye(4)
        rotation = dcos.T
        rotation[rotation == 0.0] = 0.0
        affine[:3, :3] = rotation @ np.diag(spacing)
        affine[:3, 3] = origin
        return affine


#**************************************************************************************************#
#                                          GePfileReading                                          #
#**************************************************************************************************#
#                                                                                                  #
# The data block, by layout. Standard (SIVIC; PROBE-P, PRESS CSI, sLASER): per coil the time       #
# points of every k-space position, integers of rh_point_size bytes as complex, a dummy FID        #
# skipped where the dummy scans (as many as coils, or coils x time points) fall, every other       #
# spectrum negated when chopped (nex x echoes <= 1), x and y swapped (y reversed) when the         #
# frequency direction is swapped; water references are the first 16 / nex (sLASER: user19) frames, #
# metabolites the following ones. Edited (MEGA, HERMES, HERCULES, J-PRESS; as Gannet and Osprey    #
# read it): coils x (rows x echoes) frames x points after the data offset, frame 0 a noise         #
# readout, then user19 reference and user4 / nex metabolite frames per echo; scaled and sign-      #
# alternated as GELoad scales them; several echoes (or the edit waveform set) become the edit      #
# dimension.                                                                                       #
#                                                                                                  #
#**************************************************************************************************#
class GePfileReading:
    def __init__(self, path: Path, hdr: Any, version: float, endian: str):
        self.path, self.hdr, self.endian = path, hdr, endian
        self.layout = GePfileLayout(hdr, version)

    @property
    def data_offset(self) -> int:
        """Where the data start: stated from revision 11 on; before, the data fill the end of the
        file."""
        if hasattr(self.hdr, "rhr_rdb_hdr_off_data"):
            return self.hdr.rhr_rdb_hdr_off_data
        layout, hdr = self.layout, self.hdr
        spectrum = hdr.rhr_rh_frame_size * 2 * hdr.rhr_rh_point_size
        per_coil = layout.kspace_points * spectrum * layout.time_points
        per_coil += int(layout.dummy_scans * spectrum / layout.coils)
        return self.path.stat().st_size - per_coil * layout.coils

    def _words(self, count: int, handle) -> np.ndarray:
        dtype = {4: "i4", 2: "i2"}[self.hdr.rhr_rh_point_size]
        words = np.fromfile(handle, dtype=dtype, count=count)
        if self.endian != "little":
            words.byteswap(True)
        return words

    def standard(self, unsuppressed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(raw, unsuppressed, suppressed), each (x, y, z, time points, coils, points)."""
        layout, hdr = self.layout, self.hdr
        coils, times, points = layout.coils, layout.time_points, hdr.rhr_rh_frame_size
        fid = points * 2
        per_coil = layout.kspace_points * fid * hdr.rhr_rh_point_size * times
        per_coil += int(layout.dummy_scans * fid * hdr.rhr_rh_point_size / coils)
        n = layout.voxels
        data = np.zeros((n[0], n[1], n[2], times, coils, points), np.complex64)
        chop = -1 if layout.chop else 1
        position = 0
        with open(self.path, "rb") as handle:
            handle.seek(self.data_offset)
            for coil in range(coils):
                words = self._words(int(per_coil / hdr.rhr_rh_point_size), handle)
                values = words.astype(np.float32)
                offset = 0
                for time in range(times):
                    if self._dummy(position, coil, time):
                        offset += fid
                        position += fid
                    for index in range(layout.kspace_points):
                        x, y, z = self._xyz(index)
                        if layout.chop:
                            chop *= -1
                        data[x, y, z, time, coil, :] = chop * values[offset:offset + fid].view(
                            np.complex64)
                        offset += fid
                        position += fid
        return data, data[:, :, :, :unsuppressed], data[:, :, :, unsuppressed:]

    def _dummy(self, offset: int, coil: int, time: int) -> bool:
        layout = self.layout
        dummies, coils, times = layout.dummy_scans, layout.coils, layout.time_points
        fid = self.hdr.rhr_rh_frame_size * 2
        if dummies == coils:
            between = layout.kspace_points * fid * times
            offset -= coil * fid
            if time > 1:
                offset -= fid
        elif dummies == coils * times:
            between = layout.kspace_points * fid
            offset -= coil * fid + (coil + time) * fid
        elif dummies == 0:
            return False
        else:
            between = None
        return offset % between == 0

    def _xyz(self, index: int) -> tuple[int, int, int]:
        n = self.layout.voxels
        z = int(index / (n[0] * n[1]))
        if self.layout.swap:
            return int((index - z * n[0] * n[1]) / n[1]), n[1] - int(index % n[1]) - 1, z
        return int(index % n[0]), int((index - z * n[0] * n[1]) / n[0]), z

    def edited(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(raw, unsuppressed, suppressed): raw (1, 1, 1, points, frames, coils); the others
        (1, 1, 1, points, coils, frames[, conditions])."""
        hdr = self.hdr
        echoes, nex, frames = hdr.rhr_rh_nechoes, hdr.rhr_rh_navs, hdr.rhr_rh_nframes
        points, rows = hdr.rhr_rh_da_xres, hdr.rhr_rh_da_yres
        data_frames, ref_frames = hdr.rhr_rh_user4 / nex, int(hdr.rhr_rh_user19)
        coils, cv24 = self.layout.coils, hdr.rhi_user24
        total = rows * echoes
        with open(self.path, "rb") as handle:
            handle.seek(self.data_offset)
            words = self._words(int(points * 2 * total * coils), handle)
        values = words.astype(np.float32).view(np.complex64)
        raw = np.swapaxes(values.reshape((1, 1, 1, coils, total, points)), -1, -3)
        if echoes == 1:
            if int(data_frames + ref_frames) != frames:
                scale = 1
                data_frames *= nex
                ref_frames = int(frames - data_frames)
            else:
                scale = 1 / nex
            unsuppressed = raw[:, :, :, :, 1:ref_frames + 1, :] * scale
            suppressed = raw[:, :, :, :, ref_frames + 1:, :] * scale / 2
        else:
            if int(data_frames + ref_frames) != frames:
                scale, scale_water, alternate = nex / 2, nex, 1
                data_frames *= nex
                ref_frames = frames - data_frames
            else:
                scale, scale_water, alternate = 1 / nex, 1, 0
            if total != (data_frames + ref_frames + 1) * echoes:
                raise ReaderError("frames of the edited P-file do not add up: total != (data + "
                                  "reference + 1) x echoes")
            unsuppressed = self._frames(raw, ref_frames, 0, echoes, total, cv24, alternate,
                                        points, coils) * scale_water
            suppressed = self._frames(raw, data_frames, ref_frames, echoes, total, cv24,
                                      alternate, points, coils) * scale
        if echoes == 1 and hdr.rhi_user19 == 0:
            suppressed[:, :, :, :, 0::2, :] *= np.exp(1j * np.pi)
            unsuppressed[:, :, :, :, 0::2, :] *= np.exp(1j * np.pi)
            return (raw, np.moveaxis(unsuppressed, (4, 5), (5, 4)),
                    np.moveaxis(suppressed, (4, 5), (5, 4)))
        echoes = max(echoes, 2)
        split = []
        for block in (unsuppressed, suppressed):
            stacked = np.stack([block[:, :, :, :, e::echoes, :] for e in range(echoes)], axis=-1)
            split.append(np.moveaxis(stacked, (4, 5), (5, 4)))
        return raw, split[0], split[1]

    @staticmethod
    def _frames(raw, count, first, echoes, total, cv24, alternate, points, coils) -> np.ndarray:
        """``count`` frames per echo from frame 1 + ``first`` of each echo block, sign
        alternated unless cv24 >= 16384."""
        x1, x2 = np.meshgrid(np.arange(count), np.arange(echoes))
        x1, x2 = x1.T.ravel(), x2.T.ravel()
        sign = np.ones_like(x1) if cv24 >= 16384 else (-1) ** (alternate * x1)
        sign = np.moveaxis(np.broadcast_to(sign, (1, 1, 1, points, coils, sign.size)), -1, -2)
        index = (1 + first + (total / echoes) * x2 + x1).astype(int)
        return raw[:, :, :, :, index, :] * sign


#**************************************************************************************************#
#                                           GePfileMrs                                             #
#**************************************************************************************************#
#                                                                                                  #
# The datasets of a spectroscopy P-file by its sequence (psd name, lower case; a path ending in    #
# gaba is gaba; HBCD with a PRESS protocol is HBCD PRESS). PROBE-P and its kin: metabolites and    #
# water references (_ref), DIM_COIL x DIM_DYN; HBCD PRESS stores user19 water frames, then user4   #
# metabolite frames (checked by their amplitudes). sLASER: TxOffset -2 ppm (metabolites, eddy-     #
# current references) or 0 (8 references: the unsuppressed OVS-free ones, _ref_no_ovs). oSLASER    #
# one echo: water frames 0, 1, 4, 5 _quant, 2, 3, 6, 7 _ecc. Edited sequences: DIM_COIL x DIM_DYN  #
# x DIM_EDIT, two conditions with the edit pulses of user20-22 (offsets in Hz, duration in µs)     #
# unless the waveform user19 is 19 (HERMES, HERCULES). HBCD2 (ISTHMUS): after the noise readout,   #
# blocks of one reference and 32 transients: the first block's reference at 80 ms, then 35 ms and  #
# 80 ms alternating; the first block's transients short TE, the rest HERCULES (A to D in turn,     #
# whole sets of four). Spectroscopic imaging (PROBE-P, PROBE-sL, sLASER CNI, PRESS CSI): k-space   #
# Fourier transformed (shifted) along x, y, z. Keys from the header in s, Hz, degrees; the nucleus #
# from the frequency over the field strength (1H, 13C, 31P; else 1H).                              #
#                                                                                                  #
#**************************************************************************************************#
class GePfileMrs:
    STANDARD = ("probe-p", "probe-s", "probe-p_ach", "presscsi", "fidcsi", "ia/stable/fidcsi",
                "presscsi_nfl", "epsi_3d_24", "fidall", "mrs-press")
    SLASER = ("oslaser", "slaser_cni", "slaser", "probe-sl")
    EDITED = ("jpress", "jpress_ac", "gaba", "hbcd", "hbcd1", "hbcd2", "probe-p-mega_rml",
              "repress7")
    CSI = ("probe-p", "probe-sl", "slaser_cni", "presscsi")
    HERCULES_PULSES = {"A": {"PulseOffset": [4.58, 1.90], "PulseDuration": 0.02},
                       "B": {"PulseOffset": [4.18, 1.90], "PulseDuration": 0.02},
                       "C": {"PulseOffset": 4.58, "PulseDuration": 0.02},
                       "D": {"PulseOffset": 4.18, "PulseDuration": 0.02}}
    NUCLEI = ((42.57, "1H"), (10.7, "13C"), (17.2, "31P"))
    SEX = {1: "M", 2: "F"}
    ENCODINGS = (("utf-8", "strict"), ("ISO-8859-1", "strict"), ("ascii", "replace"))

    def __init__(self, path: Path, hdr: Any, version: float, endian: str):
        self.path, self.hdr = path, hdr
        self.reading = GePfileReading(path, hdr, version, endian)
        self.layout = self.reading.layout
        self.encoding = self.header_encoding(hdr)
        self.psdname = self.text(hdr.rhi_psdname)
        self.psd, self.hbcd_press = self.sequence(hdr)
        if not self.recognises(hdr):
            raise ReaderError(f"no layout for the pulse sequence {self.psdname}")

    @classmethod
    def recognises(cls, hdr: Any) -> bool:
        """Whether the sequence has a layout here: PROBE-P and its kin, the edited sequences,
        oSLASER and PROBE-sL with any echo count, the other sLASER sequences with one echo, and
        every HBCD variant."""
        psd, _ = cls.sequence(hdr)
        if psd in cls.STANDARD + cls.EDITED or psd in ("oslaser", "probe-sl"):
            return True
        return psd in cls.SLASER and hdr.rhi_numecho == 1 or psd.startswith("hbcd")

    @classmethod
    def sequence(cls, hdr: Any) -> tuple[str, bool]:
        """(sequence, HBCD PRESS): the psd name in lower case, a path ending in gaba as gaba, HBCD
        with a PRESS protocol as mrs-press."""
        encoding = cls.header_encoding(hdr)
        psd = hdr.rhi_psdname.decode(encoding, errors="replace").lower()
        protocol = hdr.rhs_se_desc.decode(encoding, errors="replace").lower()
        hbcd_press = psd == "hbcd" and "press" in protocol
        return "gaba" if psd.endswith("gaba") else "mrs-press" if hbcd_press else psd, hbcd_press

    @classmethod
    def header_encoding(cls, hdr: Any) -> str:
        """The first codec that decodes the psd name, protocol and patient name."""
        for encoding, errors in cls.ENCODINGS:
            try:
                for value in (hdr.rhi_psdname, hdr.rhs_se_desc, hdr.rhe_patname):
                    value.decode(encoding, errors)
            except UnicodeDecodeError:
                continue
            return encoding
        return "ascii"

    def text(self, value: bytes) -> str:
        return value.decode(self.encoding, errors="replace")

    @property
    def dwelltime(self) -> float:
        return 1 / self.hdr.rhr_spectral_width

    # --- datasets --------------------------------------------------------------------------------

    def spectra(self) -> tuple[list[GeSpectrum], np.ndarray]:
        """The datasets and the affine."""
        psd, numecho, hdr = self.psd, self.hdr.rhi_numecho, self.hdr
        single_voxel = np.prod(self.layout.voxels) == 1
        if psd in self.STANDARD or (psd == "probe-sl" and numecho > 1):
            raw, water, metab = self.reading.standard(int(16 / hdr.rhi_nex))
            single = single_voxel and raw.shape[3] >= 1 and int(hdr.rhi_user4 / hdr.rhi_nex) >= 1
        elif psd in self.SLASER and numecho == 1:
            raw, water, metab = self.reading.standard(int(hdr.rhr_rh_user19))
            single = single_voxel and raw.shape[3] >= 1 and int(hdr.rhr_rh_user4) >= 1
        elif psd in self.EDITED or psd == "oslaser":
            raw, water, metab = self.reading.edited()
            single = single_voxel
        else:
            raise ReaderError(f"no layout for {self.psdname} with {numecho} echoes")
        if not single:
            spacing = self.layout.voxel_spacing(from_fov=psd == "probe-sl" and numecho > 1)
            return self._noted(self._csi(raw)), self.layout.affine_grid(spacing)
        order = [0, 1, 2, 5, 4, 3]  # to (x, y, z, points, coils, frames)
        if psd in ("mrs-press", "probe-p", "probe-s", "probe-p_ach"):
            spectra = self._probe_p(raw, np.transpose(metab, order), np.transpose(water, order))
        elif psd in ("oslaser", "slaser_cni") and numecho == 1:
            spectra = self._oslaser(np.transpose(metab, order), np.transpose(water, order))
        elif psd in ("slaser", "probe-sl"):
            spectra = self._slaser(np.transpose(metab, order), np.transpose(water, order))
        elif psd == "hbcd2":
            spectra = self._isthmus(raw)
        elif psd in ("hbcd", "jpress", "jpress_ac", "gaba", "probe-p-mega_rml", "repress7",
                     "oslaser"):
            spectra = self._edited(metab, water)
        else:
            raise ReaderError(f"no single-voxel processing for {self.psdname}")
        return self._noted(spectra), self.layout.affine()

    def _noted(self, spectra: list[GeSpectrum]) -> list[GeSpectrum]:
        note = self.nucleus()[1]
        for spectrum in spectra if note else ():
            spectrum.notes.append(note)
        return spectra

    def _probe_p(self, raw, metab, water) -> list[GeSpectrum]:
        notes = []
        if self.hbcd_press:
            water_frames, metab_frames = int(self.hdr.rhr_rh_user19), int(self.hdr.rhr_rh_user4)
            order = [0, 1, 2, 5, 4, 3]
            if water_frames + metab_frames == raw.shape[3]:
                water = np.transpose(raw[:, :, :, :water_frames], order)
                metab = np.transpose(raw[:, :, :, water_frames:], order)
            else:
                notes.append(f"HBCD PRESS: {water_frames} water + {metab_frames} metabolite "
                             f"frames do not make the {raw.shape[3]} frames; split at 16 / nex")
        tags = ("DIM_COIL", "DIM_DYN")
        return [GeSpectrum("", metab, self.values(True), tags, notes),
                GeSpectrum("_ref", water, self.values(False), tags, list(notes))]

    def _oslaser(self, metab, water) -> list[GeSpectrum]:
        quant, ecc = water[..., [0, 1, 4, 5]], water[..., [2, 3, 6, 7]]
        return [GeSpectrum("_quant", quant, self.values(False), self._default_tags(quant)),
                GeSpectrum("_ecc", ecc, self.values(False), self._default_tags(ecc)),
                GeSpectrum("", metab, self.values(True), self._default_tags(metab))]

    def _slaser(self, metab, water) -> list[GeSpectrum]:
        values = dict(self.values(True), TxOffset=(-2.0, None))
        ecc = dict(self.values(False), TxOffset=(-2.0, None))
        spectra = [GeSpectrum("", metab, values, self._default_tags(metab))]
        if not np.isclose(self.hdr.rhr_rh_user19, 8.0):
            return spectra + [GeSpectrum("_ref_ecc", water, ecc, self._default_tags(water))]
        quant = dict(self.values(False), TxOffset=(0.0, None))
        return spectra + [
            GeSpectrum("_ref_ecc", water[..., [0, 1, 4, 5]], ecc, self._default_tags(water)),
            GeSpectrum("_ref_no_ovs", water[..., [2, 3, 6, 7]], quant, self._default_tags(water))]

    @staticmethod
    def _default_tags(data: np.ndarray) -> tuple:
        return ("DIM_COIL", "DIM_DYN", "DIM_INDIRECT_0")[:max(data.ndim - 4, 0)]

    def _edited(self, metab, water) -> list[GeSpectrum]:
        spectra = []
        for suffix, data, suppressed in (("", metab, True), ("_ref", water, False)):
            values, tags = self.values(suppressed), ("DIM_COIL", "DIM_DYN")
            if data.ndim == 7:
                edit, pulses = self._editing(data)
                tags += (edit,)
                if pulses is not None:
                    values["EditPulse"] = (pulses, None)
            spectra.append(GeSpectrum(suffix, data, values, tags))
        return spectra

    def _editing(self, data: np.ndarray) -> tuple:
        """The edit dimension's tag and, for two conditions, the edit pulses (user20-22)."""
        hdr = self.hdr
        if data.shape[-1] != 2 or hdr.rhi_user19 == 19.0:
            return "DIM_EDIT", None
        mhz = float(hdr.rhr_rh_ps_mps_freq * 1e-7)
        duration = hdr.rhi_user22 if hdr.rhi_user22 > 0 else 16000
        pulses = {"ON": {"PulseOffset": hdr.rhi_user20 / mhz, "PulseDuration": duration / 1e6},
                  "OFF": {"PulseOffset": hdr.rhi_user21 / mhz, "PulseDuration": duration / 1e6}}
        edit = ("DIM_EDIT", "MEGA-EDITED j-difference editing, two conditions",
                {"EditCondition": ["ON", "OFF"]})
        return edit, pulses

    def _isthmus(self, raw: np.ndarray) -> list[GeSpectrum]:
        """HBCD2: 80 ms HERCULES, 35 ms PRESS and the water references of each."""
        raw = raw[:, :, :, :, 1:, :]  # the first readout is noise
        mask = np.ones(raw.shape[4], dtype=bool)
        mask[::33] = False
        mask[:33] = False
        edited = raw[:, :, :, :, mask, :]
        notes = []
        if edited.shape[4] % 4 != 0:  # a scan that was stopped: whole sets of four only
            complete = edited.shape[4] // 4 * 4
            notes.append(f"80 ms HERCULES: {edited.shape[4]} averages cut to {complete}")
            edited = edited[:, :, :, :, :complete, :]
        shape = list(edited.shape)
        shape[4] //= 4
        conditions = np.zeros(shape + [4], dtype=np.complex128)
        for c in range(4):
            conditions[..., c] = edited[:, :, :, :, c::4, :]
        hercules = ("DIM_EDIT", None, {"EditCondition": ["A", "B", "C", "D"]})
        long_te, short_te = (0.080, None), (0.035, None)
        return [
            GeSpectrum("_edited", conditions, dict(
                self.values(True), EchoTime=long_te, EditPulse=(self.HERCULES_PULSES, None)),
                ("DIM_DYN", "DIM_COIL", hercules), notes),
            GeSpectrum("_short_te", raw[:, :, :, :, 1:33, :],
                       dict(self.values(True), EchoTime=short_te), ("DIM_DYN", "DIM_COIL")),
            GeSpectrum("_ref_edited", raw[:, :, :, :, 0::66, :],
                       dict(self.values(False), EchoTime=long_te), ("DIM_DYN", "DIM_COIL")),
            GeSpectrum("_ref_short_te", raw[:, :, :, :, 33::66, :],
                       dict(self.values(False), EchoTime=short_te), ("DIM_DYN", "DIM_COIL")),
        ]

    def _csi(self, raw: np.ndarray) -> list[GeSpectrum]:
        if self.psd not in self.CSI:
            raise ReaderError(f"no spectroscopic imaging layout for {self.psdname}")
        data = np.transpose(raw, [0, 1, 2, 5, 4, 3])
        if data.shape[5] == 1:
            data = data.squeeze(axis=5)
        for axis in range(3):
            data = np.fft.fftshift(np.fft.fft(data, axis=axis), axes=axis)
        notes = ["k-space Fourier transformed along x, y, z (fftshift); the grid's orientation is "
                 "not checked against another export of the scan"]
        return [GeSpectrum("", data, self.values(True), ("DIM_COIL",), notes)]

    # --- keys ------------------------------------------------------------------------------------

    def nucleus(self) -> tuple[str, str | None]:
        """The nucleus from the frequency over the field strength, and a note where it is taken
        as 1H for want of either."""
        hdr = self.hdr
        if not hdr.rhe_magstrength:
            return "1H", "field strength 0 in the header: nucleus taken as 1H"
        gamma = hdr.rhr_rh_ps_mps_freq * 1e-7 / (hdr.rhe_magstrength / 10000.0)
        nucleus = next((n for g, n in self.NUCLEI if abs(gamma - g) < 0.3), None)
        if nucleus is None:
            return "1H", f"{gamma:.2f} MHz/T matches no known nucleus: taken as 1H"
        return nucleus, None

    def values(self, water_suppressed: bool) -> dict:
        """Standard keys: (value, header field)."""
        hdr = self.hdr
        frequency = float(hdr.rhr_rh_ps_mps_freq) / 1e7
        nucleus = self.nucleus()[0]
        values = {
            "SpectrometerFrequency": ([frequency], "rhr_rh_ps_mps_freq"),
            "ResonantNucleus": ([nucleus], "rhe_magstrength"),
            "EchoTime": (float(hdr.rhi_te) / 1e6, "rhi_te"),
            "RepetitionTime": (float(hdr.rhi_tr) / 1e6, "rhi_tr"),
            "ExcitationFlipAngle": (float(hdr.rhi_mr_flip), "rhi_mr_flip"),
            "WaterSuppressed": (water_suppressed, None),
            "Manufacturer": ("GE", None),
            "DwellTime": (self.dwelltime, "rhr_spectral_width"),
            "ConversionMethod": (f"miox {__version__}", None),
            "ConversionTime": (datetime.now().isoformat(timespec="milliseconds"), None),
            "OriginalFile": ([self.path.name], None),
            "kSpace": ([False, False, False], None),
        }
        for key, name in (("ManufacturersModelName", "rhe_ex_sysid"),
                          ("DeviceSerialNumber", "rhe_uniq_sys_id"),
                          ("SoftwareVersions", "rhe_ex_verscre"),
                          ("InstitutionName", "rhe_hospname"), ("RxCoil", "rhi_cname"),
                          ("SequenceName", "rhi_psdname"), ("ProtocolName", "rhs_se_desc"),
                          ("PatientName", "rhe_patname"), ("PatientDoB", "rhe_dateofbirth")):
            values[key] = (self.text(getattr(hdr, name)), name)
        if hdr.rhi_ti:  # 0: no inversion
            values["InversionTime"] = (float(hdr.rhi_ti) / 1e6, "rhi_ti")
        if hdr.rhe_patsex in self.SEX:  # 0: not entered
            values["PatientSex"] = (self.SEX[hdr.rhe_patsex], "rhe_patsex")
        return values
