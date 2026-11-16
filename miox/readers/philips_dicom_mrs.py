####################################################################################################
#                                       philips_dicom_mrs.py                                       #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 16/11/26                                                                                #
#                                                                                                  #
# Purpose: Philips DICOM spectroscopy, enhanced (MR Spectroscopy Storage) and classic (Philips     #
#          private storage): spectra, voxel geometry and keys, read from the elements.             #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pydicom

# own
from miox import __version__
from miox.readers.base import ReaderError
from miox.readers.philips_spar_sdat import PhilipsSparSdatReader
from miox.sniff import open_dicom


#**************************************************************************************************#
#                                            Spectrum                                              #
#**************************************************************************************************#
#                                                                                                  #
# One dataset of a Philips DICOM file or series: data in NIfTI-MRS order (x, y, z, points[, ...]), #
# dwell time, affine, keys as (value, field) and the dimension tags from the 5th on.               #
#                                                                                                  #
#**************************************************************************************************#
@dataclass
class Spectrum:
    name: str
    data: np.ndarray
    dwelltime: float
    affine: np.ndarray
    values: dict
    tags: tuple = ()
    notes: list = field(default_factory=list)


#**************************************************************************************************#
#                                         PhilipsDicomMrs                                          #
#**************************************************************************************************#
#                                                                                                  #
# Enhanced files hold every frame's spectrum in SpectroscopyData (frames x points complex          #
# float32); classic files one spectrum in (2005,1270), (2005,1315) points of it. The private per-  #
# frame items (2005,140F) flag the water reference ((2005,1304), newer software (2005,1004)) and,  #
# in edited scans ((2005,1597) Y), the edit condition ((2005,1598) 1: ON); without flags a         #
# reference is the second of two frames when (0018,9199) states one. ON and OFF frames pair into   #
# DIM_DYN x DIM_EDIT (ON first); several frames otherwise are DIM_DYN. Spectra are conjugated      #
# (Philips stores the left-handed frame). The voxel: three localisation slabs give its orientation #
# (the first two slab normals, negated), centre (first slab's mid position) and size (slab         #
# thicknesses); one or two slabs the plane orientation (negated), plane position, pixel spacing    #
# and the slab thickness; classic files the box of the (2005,1085) item as the SPAR states it      #
# (angulations, sizes, off-centres), else the identity at the origin with pixel spacing and slice  #
# thickness; no localisation a 10 m voxel. Spectroscopic imaging (enhanced, more than one phase    #
# row or column): (frames, rows, columns, points), newer software (frames, columns, rows, points), #
# reference frames by the flag of their first voxel, the grid at the first voxel's plane position; #
# both layouts equal the SPAR/SDAT of the same scans, and NAA, Cr and Cho fall at 2.02, 3.03 and   #
# 3.23 ppm. Keys from the elements in s, Hz, degrees; the files of a series with one shape, voxel  #
# and dwell time are stacked along DIM_DYN, else each is its own dataset (_000, _001, ...).        #
#                                                                                                  #
#**************************************************************************************************#
class PhilipsDicomMrs:
    ENHANCED = "1.2.840.10008.5.1.4.1.1.4.2"
    CLASSIC = "1.3.46.670589.11.0.0.12.1"
    ITEMS = (0x2005, 0x140F)  # private per-frame items
    REFERENCE_FLAGS = ((0x2005, 0x1304), (0x2005, 0x1004))
    EDITED, EDIT_ON = (0x2005, 0x1597), (0x2005, 0x1598)
    CLASSIC_DATA, CLASSIC_POINTS = (0x2005, 0x1270), (0x2005, 0x1315)
    CLASSIC_SPECTRAL_WIDTH = (0x2005, 0x1357)
    CLASSIC_ECHO_TIME = (0x2005, 0x1310)  # ms
    CLASSIC_FREQUENCY, CLASSIC_NUCLEUS = (0x2001, 0x1083), (0x2001, 0x1087)
    CLASSIC_VOXEL = (0x2005, 0x1085)  # item with the SPAR's angulations, sizes and off-centres
    CLASSIC_BOX = {  # SPAR name: element of the classic voxel item
        "lr_angulation": (0x2005, 0x1056), "ap_angulation": (0x2005, 0x1054),
        "cc_angulation": (0x2005, 0x1055), "lr_size": (0x2005, 0x1059),
        "ap_size": (0x2005, 0x1057), "cc_size": (0x2005, 0x1058),
        "lr_off_center": (0x2005, 0x105C), "ap_off_center": (0x2005, 0x105A),
        "cc_off_center": (0x2005, 0x105B),
    }
    PER_FRAME = "dicom/PerFrameFunctionalGroupsSequence/0"
    SHARED = "dicom/SharedFunctionalGroupsSequence/0"

    # --- series ----------------------------------------------------------------------------------

    def read(self, paths: list[Path]) -> list[Spectrum]:
        """The datasets of one series."""
        first = self.dataset(paths[0])
        if self.csi(first):
            return self._csi(first, paths[0])
        name = str(first.get("SeriesDescription") or paths[0].stem)
        files = [self._svs(self.dataset(p) if i else first, p) for i, p in enumerate(paths)]
        mains = [f[0] for f in files]
        if len(files) == 1:
            spectra = [s for s in files[0] if s is not None]
            for spectrum, suffix in zip(spectra, ("", "_ref"), strict=False):
                spectrum.name = name + suffix
            return spectra
        same = all(len({repr(x) for x in values}) == 1 for values in (
            [m.data.shape for m in mains], [m.affine.round(9).tolist() for m in mains],
            [m.dwelltime for m in mains]))
        if not same:
            spectra = []
            for index, (main, ref) in enumerate(files):
                main.name = f"{name}_{index:03}"
                spectra.append(main)
                if ref is not None:
                    ref.name = f"{name}_ref_{index:03}"
                    spectra.append(ref)
            return spectra
        names = [p.name for p in paths]
        spectra = []
        for position, suffix in ((0, ""), (1, "_ref")):
            if files[0][position] is None:
                continue
            parts = [f[position] for f in files]
            stacked = parts[0]
            stacked.data = np.stack([p.data for p in parts], axis=-1)
            stacked.tags = stacked.tags + ("DIM_DYN",)
            stacked.values = dict(stacked.values, OriginalFile=(names, None))
            stacked.name = name + suffix
            spectra.append(stacked)
        return spectra

    @staticmethod
    def dataset(path: Path) -> pydicom.Dataset:
        with open_dicom(path) as fh:
            return pydicom.dcmread(fh, force=True)

    def enhanced(self, ds: pydicom.Dataset) -> bool:
        sop = str(ds.file_meta.get("MediaStorageSOPClassUID", ""))
        if sop not in (self.ENHANCED, self.CLASSIC):
            raise ReaderError(f"SOP class {sop} is not Philips spectroscopy")
        return sop == self.ENHANCED

    def csi(self, ds: pydicom.Dataset) -> bool:
        """Spectroscopic imaging (enhanced files); classic imaging files are refused."""
        if not self.enhanced(ds):
            items = ds.get(self.ITEMS)
            item = items.value[0] if items is not None and items.value else {}
            phases = [int(item[t].value or 1) if t in item else 1
                      for t in ((0x0018, 0x9234), (0x0018, 0x9095), (0x0018, 0x9159))]
            if np.prod(phases) > 1:
                raise ReaderError("classic Philips spectroscopic imaging is not read")
            return False
        rows = int(ds.get("SpectroscopyAcquisitionPhaseRows") or 1)
        columns = int(ds.get("SpectroscopyAcquisitionPhaseColumns") or 1)
        return rows * columns > 1

    # --- single voxel ----------------------------------------------------------------------------

    def _svs(self, ds: pydicom.Dataset, path: Path) -> tuple[Spectrum, Spectrum | None]:
        """The file's spectra and reference, each (1, 1, 1, points[, frames...])."""
        affine, notes = self.voxel(ds)
        dwelltime = self.dwelltime(ds)
        if not self.enhanced(ds):
            data = self._classic_spectrum(ds, path)
            main = Spectrum("", data.reshape(1, 1, 1, -1).conj(), dwelltime, affine,
                            self.values(ds, [path.name]), (), notes)
            return main, None
        frames = int(ds.get("NumberOfFrames") or 1)
        points = int(ds.SpectroscopyAcquisitionDataColumns)
        raw = np.frombuffer(ds.SpectroscopyData, dtype="<f4")
        if raw.size != 2 * frames * points:
            raise ReaderError(f"{path.name}: {raw.size // 2} points stored, not {frames} frames x "
                              f"{points} points")
        data = (raw[0::2] + 1j * raw[1::2]).reshape(frames, points).conj()
        items = self._items(ds, frames)
        reference = self._reference_frames(ds, items, frames)
        main, tags = self._frames(ds, data, items, reference, notes)
        values = self.values(ds, [path.name])
        spectrum = Spectrum("", main, dwelltime, affine, values, tags, notes)
        if not reference.any():
            return spectrum, None
        ref = data[reference].T.reshape(1, 1, 1, points, -1)
        ref, ref_tags = (ref[..., 0], ()) if ref.shape[-1] == 1 else (ref, ("DIM_DYN",))
        ref_values = self.values(ds, [path.name], water_suppressed=False)
        return spectrum, Spectrum("", ref, dwelltime, affine, ref_values, ref_tags, list(notes))

    def _items(self, ds: pydicom.Dataset, frames: int) -> list | None:
        items = ds.get(self.ITEMS)
        items = list(items.value) if items is not None else []
        return items if len(items) == frames else None

    def _reference_frames(self, ds: pydicom.Dataset, items: list | None, frames: int) -> np.ndarray:
        """Water reference frames: by the per-frame flag, else the second of two frames when the
        file states a reference ((0018,9199) YES)."""
        flag = next((t for t in self.REFERENCE_FLAGS if items and t in items[0]), None)
        if flag is not None and frames > 1:
            return np.array([int(item[flag].value) == 1 for item in items])
        stated = (0x0018, 0x9199) in ds and str(ds[0x0018, 0x9199].value) == "YES"
        if stated and frames == 2:
            return np.array([False, True])
        return np.zeros(frames, bool)

    def _frames(self, ds, data, items, reference, notes) -> tuple[np.ndarray, tuple]:
        """The metabolite frames: ON/OFF pairs of an edited scan, else frames as DIM_DYN."""
        points = data.shape[1]
        edited = self.EDITED in ds and str(ds[self.EDITED].value).strip() == "Y"
        if edited and items and all(self.EDIT_ON in item for item in items):
            on = np.array([str(item[self.EDIT_ON].value).strip() == "1" for item in items])
            on, off = on & ~reference, ~on & ~reference
            if on.sum() == off.sum() > 0:
                pairs = np.stack((data[on].T, data[off].T), axis=-1)
                edit = ("DIM_EDIT", None, {"EditCondition": ["ON", "OFF"]})
                return pairs.reshape((1, 1, 1) + pairs.shape), ("DIM_DYN", edit)
            warnings.warn(f"edit flags of {on.sum()} ON and {off.sum()} OFF frames do not pair: "
                          "the frames are kept as dynamics", stacklevel=2)
        main = data[~reference].T.reshape(1, 1, 1, points, -1)
        return (main[..., 0], ()) if main.shape[-1] == 1 else (main, ("DIM_DYN",))

    def _classic_spectrum(self, ds: pydicom.Dataset, path: Path) -> np.ndarray:
        value = ds.get(self.CLASSIC_DATA)
        if value is None or not isinstance(value.value, (bytes, bytearray)):
            raise ReaderError(f"{path.name}: no spectrum in (2005,1270)")
        points = int(ds[self.CLASSIC_POINTS].value)
        if len(value.value) < 8 * points:
            raise ReaderError(f"{path.name}: spectrum element (2005,1270) holds "
                              f"{len(value.value)} bytes, less than one spectrum of {points} "
                              "complex points")
        raw = np.frombuffer(value.value, dtype="<f4")
        return raw[0::2] + 1j * raw[1::2]

    # --- spectroscopic imaging -------------------------------------------------------------------

    def _csi(self, ds: pydicom.Dataset, path: Path) -> list[Spectrum]:
        rows = int(ds.SpectroscopyAcquisitionPhaseRows)
        columns = int(ds.SpectroscopyAcquisitionPhaseColumns)
        points, frames = int(ds.SpectroscopyAcquisitionDataColumns), int(ds.NumberOfFrames)
        raw = np.frombuffer(ds.SpectroscopyData, dtype="<f4")
        data = np.conj(raw[0::2] + 1j * raw[1::2])
        items = ds[self.ITEMS].value
        if self.REFERENCE_FLAGS[0] in items[0]:
            data = data.reshape(frames, rows, columns, points)
            flags = [int(item[self.REFERENCE_FLAGS[0]].value) for item in items]
        else:  # newer software: columns outer, the reference flag in (2005,1004)
            data = data.reshape(frames, columns, rows, points).transpose(0, 2, 1, 3)
            flags = [int(item[self.REFERENCE_FLAGS[1]].value) for item in items]
        is_ref = np.asarray(flags).reshape(frames, rows * columns)[:, 0] == 1
        orientation = np.asarray(self._plane(ds, "PlaneOrientationSequence",
                                             "ImageOrientationPatient"))
        position = self._plane(ds, "PlanePositionSequence", "ImagePositionPatient")
        spacing = self._plane(ds, "PixelMeasuresSequence", "PixelSpacing")
        normal = np.cross(orientation[:3], orientation[3:])
        slab = max(ds.VolumeLocalizationSequence,
                   key=lambda s: abs(np.dot(np.asarray(s.SlabOrientation, float), normal)))
        affine = self.affine(orientation, position, [spacing[0], spacing[1], slab.SlabThickness])
        name = str(ds.get("SeriesDescription") or path.stem)
        spectra = []
        for reference, suffix in ((False, ""), (True, "_ref")):
            selected = data[is_ref == reference]
            if not selected.size:
                continue
            grid = selected.transpose(2, 1, 3, 0)[:, :, None]  # columns, rows, 1, points, frames
            grid = grid[..., 0] if grid.shape[-1] == 1 else grid
            values = self.values(ds, [path.name], water_suppressed=not reference)
            tags = ("DIM_DYN",) if grid.ndim == 5 else ()
            spectra.append(Spectrum(name + suffix, grid, self.dwelltime(ds), affine, values, tags))
        return spectra

    # --- geometry --------------------------------------------------------------------------------

    def voxel(self, ds: pydicom.Dataset) -> tuple[np.ndarray, list[str]]:
        """The single voxel's affine, and a note where it comes from other than three slabs."""
        if not self.enhanced(ds):
            return self._classic_voxel(ds)
        slabs = ds.get("VolumeLocalizationSequence", [])
        if len(slabs) >= 3:
            orientation = -np.concatenate([np.asarray(s.SlabOrientation, float)
                                           for s in slabs[:2]])
            sizes = [float(slabs[1].SlabThickness), float(slabs[0].SlabThickness),
                     float(slabs[2].SlabThickness)]  # as PixelSpacing: across rows first
            return self.affine(orientation, slabs[0].MidSlabPosition, sizes), []
        if slabs:
            orientation = -np.asarray(self._plane(ds, "PlaneOrientationSequence",
                                                  "ImageOrientationPatient"))
            position = self._plane(ds, "PlanePositionSequence", "ImagePositionPatient")
            spacing = self._plane(ds, "PixelMeasuresSequence", "PixelSpacing")
            sizes = [spacing[0], spacing[1], float(slabs[0].SlabThickness)]
            return self.affine(orientation, position, sizes), [
                "one localisation slab: voxel from the plane position, orientation and pixel "
                "spacing"]
        if str(ds.get("VolumeLocalizationTechnique", "")) == "NONE":
            return PhilipsSparSdatReader.NO_VOXEL.copy(), ["no localisation: a 10 m voxel"]
        raise ReaderError("no localisation slabs: the voxel is not known")

    def _classic_voxel(self, ds: pydicom.Dataset) -> tuple[np.ndarray, list[str]]:
        items = ds.get(self.CLASSIC_VOXEL)
        item = items.value[0] if items is not None and items.value else None
        if item is not None and all(t in item for t in self.CLASSIC_BOX.values()):
            box = {name: float(item[t].value) for name, t in self.CLASSIC_BOX.items()}
            box["volume_selection_enable"] = "yes"
            return PhilipsSparSdatReader.voxel(box), []
        spacing = [float(v) for v in ds.PixelSpacing]
        sizes = [spacing[0], spacing[1], float(ds.SliceThickness)]
        return self.affine([1, 0, 0, 0, 1, 0], [0, 0, 0], sizes), [
            "no voxel item (2005,1085): identity orientation at the origin"]

    @staticmethod
    def affine(orientation, position, sizes) -> np.ndarray:
        """NIfTI affine of one voxel from DICOM direction cosines (row, column), its position and
        sizes (row spacing, column spacing, thickness): LPS turned to RAS."""
        cosines = np.asarray(orientation, float).reshape(2, 3)
        norms = np.linalg.norm(cosines, axis=1, keepdims=True)
        cosines = cosines / np.where(norms == 0, 1.0, norms)
        axes = np.vstack([cosines, np.cross(cosines[0], cosines[1])]).T
        affine = np.eye(4)
        affine[:3, :3] = axes @ np.diag([float(sizes[1]), float(sizes[0]), float(sizes[2])])
        affine[:3, 3] = [float(p) for p in position]
        affine[:2, :] *= -1
        return affine

    @staticmethod
    def _plane(ds: pydicom.Dataset, sequence: str, keyword: str) -> list[float]:
        """A value from the first frame's functional groups, else the shared ones."""
        for groups in ("PerFrameFunctionalGroupsSequence", "SharedFunctionalGroupsSequence"):
            group = ds.get(groups, [None])[0]
            if group is not None and sequence in group and keyword in group[sequence].value[0]:
                return [float(v) for v in group[sequence].value[0].get(keyword)]
        raise ReaderError(f"no {keyword} in the functional groups")

    # --- keys ------------------------------------------------------------------------------------

    def dwelltime(self, ds: pydicom.Dataset) -> float:
        if self.enhanced(ds):
            return 1.0 / float(ds.SpectralWidth)
        return 1.0 / float(ds[self.CLASSIC_SPECTRAL_WIDTH].value)

    def dwell_field(self, ds: pydicom.Dataset) -> str:
        return "dicom/SpectralWidth" if self.enhanced(ds) else "dicom/(2005,1357)"

    def values(self, ds: pydicom.Dataset, files: list[str], water_suppressed: bool = True) -> dict:
        """Standard keys: (value, vendor field)."""
        values = {}
        for key, choices in (
            ("SpectrometerFrequency", (("TransmitterFrequency", "TransmitterFrequency"),
                                       (self.CLASSIC_FREQUENCY, "(2001,1083)"))),
            ("ResonantNucleus", (("ResonantNucleus", "ResonantNucleus"),
                                 (self.CLASSIC_NUCLEUS, "(2001,1087)"))),
        ):
            tag, name = next(((t, n) for t, n in choices if t in ds), choices[0])
            value = float(ds[tag].value) if key == "SpectrometerFrequency" else str(ds[tag].value)
            values[key] = ([value], f"dicom/{name}")
        timing = "MRTimingAndRelatedParametersSequence"
        echo = self._group(ds, "MREchoSequence", "EffectiveEchoTime", shared=False)
        if echo is None and self.CLASSIC_ECHO_TIME in ds:
            echo = (ds[self.CLASSIC_ECHO_TIME].value, "dicom/(2005,1310)")
        for key, found, scale in (
            ("EchoTime", echo, 1e-3),
            ("RepetitionTime", self._group(ds, timing, "RepetitionTime"), 1e-3),
            ("ExcitationFlipAngle", self._group(ds, timing, "FlipAngle"), 1.0),
        ):
            if found is not None:
                values[key] = (float(found[0]) * scale, found[1])
        values["WaterSuppressed"] = (water_suppressed, None)
        for key, keyword in (("Manufacturer", "Manufacturer"),
                             ("ManufacturersModelName", "ManufacturerModelName"),
                             ("DeviceSerialNumber", "DeviceSerialNumber"),
                             ("SoftwareVersions", "SoftwareVersions"),
                             ("InstitutionName", "InstitutionName"),
                             ("InstitutionAddress", "InstitutionAddress"),
                             ("SequenceName", "PulseSequenceName"),
                             ("ProtocolName", "ProtocolName"),
                             ("PatientPosition", "PatientPosition")):
            if keyword in ds:
                values[key] = (str(ds.data_element(keyword).value), f"dicom/{keyword}")
        if ds.get("PatientName"):
            values["PatientName"] = (str(ds.PatientName.family_name), "dicom/PatientName")
        if ds.get("PatientWeight"):
            values["PatientWeight"] = (float(ds.PatientWeight), "dicom/PatientWeight")
        for key, keyword in (("PatientDoB", "PatientBirthDate"), ("PatientSex", "PatientSex")):
            if ds.get(keyword):
                values[key] = (str(ds.data_element(keyword).value), f"dicom/{keyword}")
        values["DwellTime"] = (self.dwelltime(ds), self.dwell_field(ds))
        values["ConversionMethod"] = (f"miox {__version__}", None)
        values["ConversionTime"] = (datetime.now().isoformat(timespec="milliseconds"), None)
        values["kSpace"] = ([False, False, False], None)
        values["OriginalFile"] = (list(files), None)
        return values

    def _group(self, ds: pydicom.Dataset, sequence: str, keyword: str,
               shared: bool = True) -> tuple[float, str] | None:
        """A value of the first frame's functional groups, else the shared ones, else the
        top-level element: (value, field)."""
        groups = [("PerFrameFunctionalGroupsSequence", self.PER_FRAME)]
        if shared:
            groups.append(("SharedFunctionalGroupsSequence", self.SHARED))
        for name, prefix in groups:
            group = ds.get(name, [None])[0]
            if group is not None and sequence in group and keyword in group[sequence].value[0]:
                return group[sequence].value[0].get(keyword), f"{prefix}/{sequence}/0/{keyword}"
        if keyword != "EffectiveEchoTime" and keyword in ds:
            return ds.data_element(keyword).value, f"dicom/{keyword}"
        return None
