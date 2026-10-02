####################################################################################################
#                                          dicom_image.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for DICOM MR images: classic files (one slice each, or a Siemens mosaic) and     #
#          enhanced multi-frame files, from one file or a folder of them. The vendor header is the #
#          first file's header (CSA headers and Siemens protocol included).                        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import dataclasses
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pydicom

with warnings.catch_warnings():  # nibabel.nicom warns on import that it is experimental
    warnings.simplefilter("ignore", UserWarning)
    from nibabel.nicom import csareader

# own
from miox.data import MRImage
from miox.meta.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.readers.dicom_mrs import DicomMrsReader
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                              Frame                                               #
#**************************************************************************************************#
#                                                                                                  #
# One 2D image: a classic file, a tile set of a mosaic file, or one frame of an enhanced file,     #
# with what assembly needs. ``order`` sorts the frames of one slice position in time.              #
#                                                                                                  #
#**************************************************************************************************#
@dataclass
class Frame:
    ds: pydicom.Dataset
    path: Path
    index: int | None  # frame of an enhanced file
    position: np.ndarray  # ImagePositionPatient, mm (LPS)
    orientation: np.ndarray  # ImageOrientationPatient, 6 values
    spacing: tuple[float, float]  # PixelSpacing: row spacing, column spacing
    thickness: float
    echo: float | None  # echo time, ms
    kind: tuple[str, ...]  # image type from its third value on (M, P, FIELD_MAP, ...)
    order: tuple
    group: tuple = field(default=())


#**************************************************************************************************#
#                                         DicomImageReader                                         #
#**************************************************************************************************#
#                                                                                                  #
# Frames are grouped into images by series, echo time, image type (from its third value: M, P,     #
# FIELD_MAP, ...), orientation and matrix size, as dcm2niix splits them; each image's slices are   #
# sorted along the slice normal and its volumes, per position, by acquisition number, temporal     #
# position, time and instance number. Data are indexed (column, row, slice, volume); the affine is #
# the DICOM one (row and column cosines times the pixel spacing, the step between the first and    #
# last slice, the first position) with x and y negated for RAS. Values are rescaled (slope,        #
# intercept); Philips values as floating point, (stored * slope + intercept) / (slope * private    #
# MRScaleSlope), as dcm2niix writes them. Siemens mosaics are cut into their                       #
# NumberOfImagesInMosaic tiles; the first tile's position is corrected from the mosaic's corner    #
# and the slice direction is the CSA SliceNormalVector. Enhanced files are read frame by frame     #
# from their functional groups (private sequences, which repeat classic tags, are skipped).        #
# Multi-valued text is joined with a backslash, as DICOM writes it; COLUMN (enhanced DICOM) is     #
# written COL. Checked against dcm2niix: data in world space and every key both state, on the      #
# dcm_qa sets. An image that cannot be read (e.g. pixel data no installed decoder reads) is named  #
# in the notes of the others.                                                                      #
#                                                                                                  #
#**************************************************************************************************#
class DicomImageReader(Reader):
    FORMATS = (Format.DICOM_IMAGE,)
    PACKAGES = Reader.PACKAGES + ("pydicom",)
    SOURCE = "DICOM"
    LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])
    PHILIPS_SCALE = (0x2005, 0x100E)  # MRScaleSlope
    PHILIPS_FRAMES = (0x2005, 0x140F)  # private per-frame items
    KIND_SUFFIXES = (
        ({"P", "PHASE"}, "_ph"),
        ({"FIELD_MAP", "B0 MAP", "B0"}, "_fieldmap"),
        ({"R", "REAL"}, "_real"),
        ({"I", "IMAGINARY"}, "_imaginary"),
    )
    MS = 1e-3
    KEYS = {  # standard key: (DICOM keyword, scale or None)
        "Modality": ("Modality", None),
        "Manufacturer": ("Manufacturer", None),
        "ManufacturersModelName": ("ManufacturerModelName", None),
        "DeviceSerialNumber": ("DeviceSerialNumber", None),
        "StationName": ("StationName", None),
        "SoftwareVersions": ("SoftwareVersions", None),
        "InstitutionName": ("InstitutionName", None),
        "InstitutionAddress": ("InstitutionAddress", None),
        "InstitutionalDepartmentName": ("InstitutionalDepartmentName", None),
        "StudyDescription": ("StudyDescription", None),
        "ProcedureStepDescription": ("PerformedProcedureStepDescription", None),
        "SeriesDescription": ("SeriesDescription", None),
        "ProtocolName": ("ProtocolName", None),
        "SeriesNumber": ("SeriesNumber", None),
        "AcquisitionNumber": ("AcquisitionNumber", None),
        "ImageType": ("ImageType", None),
        "BodyPart": ("BodyPartExamined", None),
        "PatientPosition": ("PatientPosition", None),
        "SequenceName": ("SequenceName", None),
        "ScanningSequence": ("ScanningSequence", None),
        "SequenceVariant": ("SequenceVariant", None),
        "ScanOptions": ("ScanOptions", None),
        "MRAcquisitionType": ("MRAcquisitionType", None),
        "RepetitionTime": ("RepetitionTime", MS),
        "InversionTime": ("InversionTime", MS),
        "ExcitationFlipAngle": ("FlipAngle", None),
        "FieldStrength": ("MagneticFieldStrength", None),
        "SliceThickness": ("SliceThickness", None),
        "SpacingBetweenSlices": ("SpacingBetweenSlices", None),
        "PixelBandwidth": ("PixelBandwidth", None),
        "EchoTrainLength": ("EchoTrainLength", None),
        "PercentPhaseFOV": ("PercentPhaseFieldOfView", None),
        "PercentSampling": ("PercentSampling", None),
        "InPlanePhaseEncodingDirectionDICOM": ("InPlanePhaseEncodingDirection", None),
        "PatientName": ("PatientName", None),
        "PatientID": ("PatientID", None),
        "PatientSex": ("PatientSex", None),
        "PatientWeight": ("PatientWeight", None),
        "PatientDoB": ("PatientBirthDate", None),
    }

    def read(self, sniff: Sniff) -> list[MRImage]:
        paths = self._files(sniff)
        frames = []
        for path in paths:
            ds = pydicom.dcmread(path)
            if "PixelData" not in ds:
                continue
            frames += self._frames(ds, path)
        if not frames:
            raise ReaderError(f"{sniff.path.name}: no DICOM image data")
        groups = defaultdict(list)
        for frame in frames:
            groups[frame.group].append(frame)
        names = self._names(list(groups))
        header = DicomMrsReader().header(paths[0])
        images, failed = [], []
        for group, name in zip(groups.values(), names, strict=True):
            try:
                images.append(self._image(sniff, group, header, name))
            except ReaderError as e:
                failed.append(f"image {name} not read: {e}")
        if not images:
            raise ReaderError("; ".join(failed))
        for image in images:
            image.provenance = dataclasses.replace(
                image.provenance, notes=image.provenance.notes + tuple(failed)
            )
        return images

    # --- frames ----------------------------------------------------------------------------------

    def _frames(self, ds: pydicom.Dataset, path: Path) -> list[Frame]:
        if "PerFrameFunctionalGroupsSequence" in ds:
            return [self._frame(ds, path, i) for i in range(int(ds.NumberOfFrames))]
        return [self._frame(ds, path, None)]

    def _frame(self, ds: pydicom.Dataset, path: Path, index: int | None) -> Frame:
        orientation = np.asarray(self._value(ds, "ImageOrientationPatient", index), float)
        position = np.asarray(self._value(ds, "ImagePositionPatient", index), float)
        spacing = tuple(float(v) for v in self._value(ds, "PixelSpacing", index))
        thickness = float(self._value(ds, "SliceThickness", index) or 0)
        echo = self._value(ds, "EffectiveEchoTime", index)
        echo = self._value(ds, "EchoTime", index) if echo is None else echo
        echo = None if echo is None else float(echo)
        image_type = self._value(ds, "FrameType", index) or ds.get("ImageType") or []
        kind = tuple(str(v).upper() for v in list(image_type)[2:])
        temporal = self._value(ds, "TemporalPositionIndex", index)
        order = (
            int(ds.get("AcquisitionNumber") or 0),
            int(temporal or 0),
            str(ds.get("AcquisitionTime") or ""),
            int(ds.get("InstanceNumber") or 0),
            index or 0,
        )
        group = (
            str(ds.get("SeriesInstanceUID")),
            int(ds.get("SeriesNumber") or 0),
            None if echo is None else round(echo, 3),
            kind,
            tuple(np.round(orientation, 4)),
            (int(ds.Rows), int(ds.Columns)),
        )
        return Frame(ds, path, index, position, orientation, spacing, thickness, echo, kind,
                     order, group)

    @staticmethod
    def _value(ds: pydicom.Dataset, keyword: str, index: int | None) -> Any:
        """A value from the frame's functional groups, the shared ones, else the dataset; private
        sequences (e.g. Philips (2005,140F), which repeats classic tags) are not searched."""
        sequences = []
        if index is not None and "PerFrameFunctionalGroupsSequence" in ds:
            sequences.append(ds.PerFrameFunctionalGroupsSequence[index])
        if "SharedFunctionalGroupsSequence" in ds:
            sequences.append(ds.SharedFunctionalGroupsSequence[0])
        for item in sequences:
            for element in item:
                if element.tag.is_private or element.VR != "SQ" or not element.value:
                    continue
                if keyword in element.value[0]:
                    return element.value[0].get(keyword)
        return ds.get(keyword)

    # --- assembly --------------------------------------------------------------------------------

    def _image(self, sniff: Sniff, frames: list[Frame], header: dict, name: str) -> MRImage:
        first = frames[0]
        notes = []
        if self._is_mosaic(first.ds):
            data, affine = self._mosaic(frames, notes)
        else:
            data, affine = self._volume(frames, notes)
        meta = Meta(header)
        for key, (keyword, scale) in self.KEYS.items():
            value = self._value(first.ds, keyword, first.index)
            if value in (None, ""):
                continue
            value = self.plain(DicomMrsReader()._value(value))
            if isinstance(value, list) and key != "ImageType":  # DICOM multi-value
                value = "\\".join(str(v) for v in value)
            if scale is not None:
                value = float(value) * scale
            source = f"{self.SOURCE}: {keyword}"
            if key == "InPlanePhaseEncodingDirectionDICOM" and value == "COLUMN":
                value, source = "COL", f"{source}; COLUMN (enhanced DICOM) written as COL"
            meta.set(key, value, source, f"dicom/{keyword}")
        if first.echo is not None:
            source = f"{self.SOURCE}: echo time (ms)"
            meta.set("EchoTime", first.echo * self.MS, source, "dicom/EchoTime")
        frequency = self._value(first.ds, "ImagingFrequency", first.index)
        if frequency:
            source = f"{self.SOURCE}: ImagingFrequency"
            meta.set("SpectrometerFrequency", [float(frequency)], source, "dicom/ImagingFrequency")
        meta.set("ImageOrientationPatientDICOM", [float(v) for v in first.orientation],
                 f"{self.SOURCE}: ImageOrientationPatient", "dicom/ImageOrientationPatient")
        dims = ("volume",) if data.ndim == 4 else ()
        provenance = self.provenance(sniff, tuple(notes))
        return MRImage(data, affine, meta, provenance, name, dims)

    def _volume(self, frames: list[Frame], notes: list[str]) -> tuple[np.ndarray, np.ndarray]:
        first = frames[0]
        row, column = first.orientation[:3], first.orientation[3:]
        normal = np.cross(row, column)
        positions = defaultdict(list)
        for frame in frames:
            positions[round(float(frame.position @ normal), 3)].append(frame)
        keys = sorted(positions)
        volumes = {len(v) for v in positions.values()}
        if len(volumes) != 1:
            raise ReaderError(f"{first.path.name}: slice positions hold {sorted(volumes)} frames")
        nvol = volumes.pop()
        rows, columns = int(first.ds.Rows), int(first.ds.Columns)
        data = np.empty((columns, rows, len(keys), nvol), dtype=np.float64)
        for k, key in enumerate(keys):
            for v, frame in enumerate(sorted(positions[key], key=lambda f: f.order)):
                data[:, :, k, v] = self._pixels(frame).T
        if len(keys) > 1:
            last = positions[keys[-1]][0].position
            step = (last - positions[keys[0]][0].position) / (len(keys) - 1)
        else:
            step = normal * (float(first.ds.get("SpacingBetweenSlices") or first.thickness or 1))
        affine = self._affine(first, positions[keys[0]][0].position, step)
        return self._narrow(data[..., 0] if nvol == 1 else data), affine

    def _mosaic(self, frames: list[Frame], notes: list[str]) -> tuple[np.ndarray, np.ndarray]:
        first = frames[0]
        csa = csareader.get_csa_header(first.ds, "image")["tags"]
        count = int(csa["NumberOfImagesInMosaic"]["items"][0])
        tiles = math.ceil(math.sqrt(count))
        rows, columns = int(first.ds.Rows) // tiles, int(first.ds.Columns) // tiles
        volumes = []
        for frame in sorted(frames, key=lambda f: f.order):
            mosaic = self._pixels(frame)
            slices = [
                mosaic[(k // tiles) * rows : (k // tiles + 1) * rows,
                       (k % tiles) * columns : (k % tiles + 1) * columns].T
                for k in range(count)
            ]
            volumes.append(np.stack(slices, axis=-1))
        data = np.stack(volumes, axis=-1)
        row, column = first.orientation[:3], first.orientation[3:]
        row_spacing, column_spacing = first.spacing
        corner = (
            first.position
            + row * column_spacing * (int(first.ds.Columns) - columns) / 2
            + column * row_spacing * (int(first.ds.Rows) - rows) / 2
        )
        normal = np.asarray(csa["SliceNormalVector"]["items"], float)
        spacing = float(first.ds.get("SpacingBetweenSlices") or first.thickness)
        affine = self._affine(first, corner, normal * spacing)
        notes.append(f"Siemens mosaic of {count} slices")
        return self._narrow(data[..., 0] if data.shape[-1] == 1 else data), affine

    def _affine(self, first: Frame, position: np.ndarray, step: np.ndarray) -> np.ndarray:
        row_spacing, column_spacing = first.spacing
        lps = np.eye(4)
        lps[:3, 0] = first.orientation[:3] * column_spacing
        lps[:3, 1] = first.orientation[3:] * row_spacing
        lps[:3, 2] = step
        lps[:3, 3] = position
        return self.LPS_TO_RAS @ lps

    def _pixels(self, frame: Frame) -> np.ndarray:
        """The frame's stored values, rescaled; Philips as floating point values."""
        try:
            pixels = frame.ds.pixel_array
        except Exception as e:
            syntax = frame.ds.file_meta.TransferSyntaxUID
            raise ReaderError(f"{frame.path.name}: pixel data ({syntax.name}) not decoded") from e
        pixels = pixels if frame.index is None else pixels[frame.index]
        slope = float(self._value(frame.ds, "RescaleSlope", frame.index) or 1)
        intercept = float(self._value(frame.ds, "RescaleIntercept", frame.index) or 0)
        values = pixels * slope + intercept
        private = self._philips_scale(frame)
        if private:
            values = values / (slope * private)
        return values

    def _philips_scale(self, frame: Frame) -> float | None:
        ds = frame.ds
        if "philips" not in str(ds.get("Manufacturer", "")).lower():
            return None
        if frame.index is not None:  # in the frame's functional groups
            group = ds.PerFrameFunctionalGroupsSequence[frame.index]
            item = group[self.PHILIPS_FRAMES].value[0] if self.PHILIPS_FRAMES in group else group
            value = item.get(self.PHILIPS_SCALE)
        else:
            value = ds.get(self.PHILIPS_SCALE)
        value = None if value is None else float(value.value)
        return value if value else None

    @staticmethod
    def _narrow(data: np.ndarray) -> np.ndarray:
        """Integers stay integers when no scaling made fractions."""
        if np.all(np.mod(data, 1) == 0) and data.min() >= -(2**31) and data.max() < 2**31:
            return data.astype(np.int32 if data.min() < 0 or data.max() > 32767 else np.int16)
        return data.astype(np.float32)

    # --- naming and files ------------------------------------------------------------------------

    @staticmethod
    def _is_mosaic(ds: pydicom.Dataset) -> bool:
        return "MOSAIC" in [str(v).upper() for v in ds.get("ImageType", [])]

    def _names(self, groups: list[tuple]) -> list[str]:
        """Series number, _e<n> when one image type has several echoes, a suffix for phase and
        field maps; a counter where names would repeat."""
        echoes = defaultdict(set)
        for uid, _, echo, kind, *_ in groups:
            echoes[(uid, kind)].add(echo)
        names = []
        for uid, number, echo, kind, *_ in groups:
            name = str(number)
            ordered = sorted(e for e in echoes[(uid, kind)] if e is not None)
            if len(ordered) > 1 and echo is not None:
                name += f"_e{ordered.index(echo) + 1}"
            name += next((s for k, s in self.KIND_SUFFIXES if set(kind) & k), "")
            names.append(name)
        seen = defaultdict(int)
        unique = []
        for name in names:
            seen[name] += 1
            unique.append(name if names.count(name) == 1 else f"{name}_{seen[name]}")
        return unique

    @staticmethod
    def _files(sniff: Sniff) -> list[Path]:
        if not sniff.path.is_dir():
            return [sniff.path]
        files = []
        for path in sniff.companions:
            with open(path, "rb") as fh:
                if fh.read(132)[128:] == b"DICM":
                    files.append(path)
        return sorted(files)
