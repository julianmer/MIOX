####################################################################################################
#                                            ismrmrd.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 05/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for ISMRMRD (MRD version 1) HDF5 files: the XML header, the acquisitions with    #
#          their headers and trajectories, the stored waveforms and the reconstructed images.      #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import h5py
import numpy as np

# own
from miox.data import MRImage, RawData
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                          IsmrmrdReader                                           #
#**************************************************************************************************#
#                                                                                                  #
# Every group of the file with a ``data`` table becomes one RawData (one row per readout:          #
# AcquisitionHeader, trajectory, interleaved float32 samples), with the ``waveforms`` table        #
# attached; every ``image_*`` subgroup becomes one MRImage. The group's XML header, if any, is     #
# kept whole as the vendor header (repeated elements as lists, numbers as numbers); the standard   #
# keys are read from it and from the first acquisition header. Times of the header are in ms,      #
# frequencies in Hz, the sample time in µs. Image headers give the image centre and the read,      #
# phase and slice directions in the DICOM patient frame (LPS); the affine maps to RAS.             #
#                                                                                                  #
#**************************************************************************************************#
class IsmrmrdReader(Reader):
    FORMATS = (Format.ISMRMRD,)
    PACKAGES = ("miox", "h5py")
    SOURCE = "ISMRMRD header"
    TABLES = ("xml", "data", "waveforms", "config", "config_file")
    GAMMA_H1 = 42.577478518  # MHz/T

    # standard key: (header path, divisor to the key's unit); None keeps the value as it is
    KEYS = {
        "PatientName": ("subjectInformation/patientName", None),
        "PatientWeight": ("subjectInformation/patientWeight_kg", None),
        "PatientHeight": ("subjectInformation/patientHeight_m", None),
        "PatientID": ("subjectInformation/patientID", None),
        "PatientDoB": ("subjectInformation/patientBirthdate", None),
        "PatientSex": ("subjectInformation/patientGender", None),
        "StudyID": ("studyInformation/studyID", None),
        "AccessionNumber": ("studyInformation/accessionNumber", None),
        "ReferringPhysicianName": ("studyInformation/referringPhysicianName", None),
        "StudyDescription": ("studyInformation/studyDescription", None),
        "StudyInstanceUID": ("studyInformation/studyInstanceUID", None),
        "BodyPart": ("studyInformation/bodyPartExamined", None),
        "ProtocolName": ("measurementInformation/protocolName", None),
        "SequenceName": ("measurementInformation/sequenceName", None),
        "SeriesDescription": ("measurementInformation/seriesDescription", None),
        "SeriesNumber": ("measurementInformation/initialSeriesNumber", None),
        "PatientPosition": ("measurementInformation/patientPosition", None),
        "FrameOfReferenceUID": ("measurementInformation/frameOfReferenceUID", None),
        "Manufacturer": ("acquisitionSystemInformation/systemVendor", None),
        "ManufacturersModelName": ("acquisitionSystemInformation/systemModel", None),
        "FieldStrength": ("acquisitionSystemInformation/systemFieldStrength_T", None),
        "NumberOfReceiveChannels": ("acquisitionSystemInformation/receiverChannels", None),
        "InstitutionName": ("acquisitionSystemInformation/institutionName", None),
        "StationName": ("acquisitionSystemInformation/stationName", None),
        "DeviceSerialNumber": ("acquisitionSystemInformation/deviceSerialNumber", None),
        "H1ResonanceFrequency": ("experimentalConditions/H1resonanceFrequency_Hz", 1e6),
        "RepetitionTime": ("sequenceParameters/TR", 1000),
        "EchoTime": ("sequenceParameters/TE", 1000),
        "InversionTime": ("sequenceParameters/TI", 1000),
        "ExcitationFlipAngle": ("sequenceParameters/flipAngle_deg", None),
        "PulseSequenceType": ("sequenceParameters/sequence_type", None),
        "VendorReportedEchoSpacing": ("sequenceParameters/echo_spacing", 1000),
        "DiffusionScheme": ("sequenceParameters/diffusionScheme", None),
    }
    ENCODING_KEYS = {
        "EncodedMatrixSize": ("encodedSpace/matrixSize", None),
        "EncodedFieldOfView": ("encodedSpace/fieldOfView_mm", None),
        "ReconMatrixSize": ("reconSpace/matrixSize", None),
        "ReconFieldOfView": ("reconSpace/fieldOfView_mm", None),
        "KSpaceTrajectory": ("trajectory", None),
        "EchoTrainLength": ("echoTrainLength", None),
        "ParallelReductionFactorInPlane":
            ("parallelImaging/accelerationFactor/kspace_encoding_step_1", None),
        "ParallelReductionFactorOutOfPlane":
            ("parallelImaging/accelerationFactor/kspace_encoding_step_2", None),
    }
    TEXT_KEYS = {  # keys the header schema types as text, even when the text is a number
        "PatientName", "PatientID", "StudyID", "ReferringPhysicianName", "StudyDescription",
        "StudyInstanceUID", "BodyPart", "ProtocolName", "SequenceName", "SeriesDescription",
        "FrameOfReferenceUID", "Manufacturer", "ManufacturersModelName", "InstitutionName",
        "StationName", "DeviceSerialNumber", "PulseSequenceType", "DiffusionScheme",
    }
    MULTIBAND_FACTOR = "multiband_factor"
    IMAGE_TYPES = {1: "MAGNITUDE", 2: "PHASE", 3: "REAL", 4: "IMAG", 5: "COMPLEX"}
    NOISE_BIT = 19  # ACQ_IS_NOISE_MEASUREMENT
    ACQUISITION_FLAGS = {  # bit number (from 1): name, as in ismrmrd.h
        **{2 * i + 1: f"FIRST_IN_{d}" for i, d in enumerate(
            ("ENCODE_STEP1", "ENCODE_STEP2", "AVERAGE", "SLICE", "CONTRAST", "PHASE",
             "REPETITION", "SET", "SEGMENT"))},
        **{2 * i + 2: f"LAST_IN_{d}" for i, d in enumerate(
            ("ENCODE_STEP1", "ENCODE_STEP2", "AVERAGE", "SLICE", "CONTRAST", "PHASE",
             "REPETITION", "SET", "SEGMENT"))},
        19: "IS_NOISE_MEASUREMENT", 20: "IS_PARALLEL_CALIBRATION",
        21: "IS_PARALLEL_CALIBRATION_AND_IMAGING", 22: "IS_REVERSE",
        23: "IS_NAVIGATION_DATA", 24: "IS_PHASECORR_DATA", 25: "LAST_IN_MEASUREMENT",
        26: "IS_HPFEEDBACK_DATA", 27: "IS_DUMMYSCAN_DATA", 28: "IS_RTFEEDBACK_DATA",
        29: "IS_SURFACECOILCORRECTIONSCAN_DATA", 30: "IS_PHASE_STABILIZATION_REFERENCE",
        31: "IS_PHASE_STABILIZATION",
        **{52 + i: f"COMPRESSION{i}" for i in range(1, 5)},
        **{56 + i: f"USER{i}" for i in range(1, 9)},
    }
    LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])

    def read(self, sniff: Sniff) -> list[RawData | MRImage]:
        datasets: list[RawData | MRImage] = []
        notes: list[str] = []
        with h5py.File(sniff.path, "r") as fh:
            for name, group in fh.items():
                if not isinstance(group, h5py.Group):
                    continue
                header = self.header(group)
                if "data" in group and "xml" in group:
                    datasets.append(self._raw(sniff, group, header, name, notes))
                for sub, image in group.items():
                    if isinstance(image, h5py.Group) and "data" in image and "header" in image:
                        datasets.append(self._image(sniff, image, header, f"{name}/{sub}", notes))
        if not datasets:
            raise ReaderError(f"{sniff.path.name}: no ISMRMRD acquisitions or images")
        return datasets

    # --- header ----------------------------------------------------------------------------------

    def header(self, group: h5py.Group) -> dict:
        """The group's XML header as a nested dict; empty when the group has none."""
        if "xml" not in group:
            return {}
        text = group["xml"][0]
        text = text.decode() if isinstance(text, bytes) else str(text)
        root = ET.fromstring(text)
        return self._element(root)

    @classmethod
    def _element(cls, node: ET.Element) -> Any:
        """An element as a dict of its children (a tag that repeats becomes a list), a leaf as
        its text (numbers as numbers)."""
        children = list(node)
        if not children:
            return cls._scalar(node.text)
        tags = [child.tag.rsplit("}", 1)[-1] for child in children]
        result: dict[str, Any] = {}
        for tag, child in zip(tags, children, strict=True):
            value = cls._element(child)
            if tags.count(tag) > 1:
                result.setdefault(tag, []).append(value)
            else:
                result[tag] = value
        return result

    @staticmethod
    def _scalar(text: str | None) -> Any:
        if text is None:
            return None
        text = text.strip()
        try:
            number = int(text)
            return number if str(number) == text else text  # leading zeros stay text
        except ValueError:
            pass
        try:
            return float(text)
        except ValueError:
            return text

    @staticmethod
    def _lookup(header: dict, path: str) -> Any:
        node: Any = header
        for part in path.split("/"):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node

    def _meta(self, header: dict, notes: list[str]) -> Meta:
        meta = Meta(dict(header))  # a copy: datasets of one header add their own entries
        for key, (path, divisor) in self.KEYS.items():
            self._set(meta, key, header, path, divisor)
        encodings = header.get("encoding")
        encoding = encodings[0] if isinstance(encodings, list) else encodings
        if isinstance(encoding, dict):
            prefix = "encoding/"
            if isinstance(encodings, list):
                notes.append(f"{len(encodings)} encoding spaces; the keys describe the first")
                prefix = "encoding/0/"
            for key, (path, divisor) in self.ENCODING_KEYS.items():
                self._set(meta, key, encoding, path, divisor, prefix=prefix)
            limits = encoding.get("encodingLimits")
            if isinstance(limits, dict):
                field = f"{prefix}encodingLimits"
                meta.set("EncodingLimits", self._limits(limits), self.SOURCE, field)
            multiband = self._lookup(encoding, "parallelImaging/multiband")
            if isinstance(multiband, dict) and self.MULTIBAND_FACTOR in multiband:
                field = f"{prefix}parallelImaging/multiband/{self.MULTIBAND_FACTOR}"
                meta.set("MultibandAccelerationFactor", multiband[self.MULTIBAND_FACTOR],
                         self.SOURCE, field)
        self._datetime(meta, header)
        self._diffusion(meta, header, notes)
        self._table_position(meta, header)
        if "FieldStrength" not in meta and "H1ResonanceFrequency" in meta:
            field = meta["H1ResonanceFrequency"] / self.GAMMA_H1
            meta.set("FieldStrength", field, "H1ResonanceFrequency / 42.577478518 MHz/T")
        return meta

    @staticmethod
    def _limits(limits: dict) -> dict:
        return limits

    def _set(self, meta: Meta, key: str, header: dict, path: str, divisor, prefix: str = ""):
        value = self._lookup(header, path)
        if value is None or isinstance(value, (dict, list)) and not value:
            return
        if key in self.TEXT_KEYS and not isinstance(value, str):
            value = str(value)
        if isinstance(value, dict):  # x, y, z triplets
            if set(value) == {"x", "y", "z"}:
                value = [value["x"], value["y"], value["z"]]
            else:
                return
        if divisor is not None:
            numbers = value if isinstance(value, list) else [value]
            if not all(isinstance(v, (int, float)) for v in numbers):
                return  # text where the schema has a number; the vendor header keeps it
            value = [v / divisor for v in value] if isinstance(value, list) else value / divisor
        if isinstance(value, list) and len(value) == 1 and key not in self.ENCODING_KEYS:
            value = value[0]
        meta.set(key, self.plain(value), f"{self.SOURCE}: {prefix}{path}", prefix + path)

    def _datetime(self, meta: Meta, header: dict) -> None:
        for key, group, prefix in (("AcquisitionDateTime", "measurementInformation", "series"),
                                   ("StudyDateTime", "studyInformation", "study")):
            info = header.get(group)
            if not isinstance(info, dict):
                continue
            date, time = info.get(f"{prefix}Date"), info.get(f"{prefix}Time")
            if date is not None and time is not None:
                field = f"{group}/{prefix}Date, {prefix}Time"
                meta.set(key, f"{date}T{time}", f"{self.SOURCE}: {field}", field)

    def _diffusion(self, meta: Meta, header: dict, notes: list[str]) -> None:
        entries = self._lookup(header, "sequenceParameters/diffusion")
        if entries is None:
            return
        entries = entries if isinstance(entries, list) else [entries]
        values = [float(e["bvalue"]) for e in entries if isinstance(e, dict) and "bvalue" in e]
        if values:
            field = "sequenceParameters/diffusion/bvalue"
            meta.set("DiffusionBValue", values, f"{self.SOURCE}: {field}", field)
            notes.append(
                "diffusion gradient directions (rl, ap, fh) kept in the header only: their sign "
                "convention is not stated"
            )

    def _table_position(self, meta: Meta, header: dict) -> None:
        value = self._lookup(header, "measurementInformation/relativeTablePosition")
        if isinstance(value, dict) and set(value) == {"x", "y", "z"}:
            field = "measurementInformation/relativeTablePosition"
            position = [value["x"], value["y"], value["z"]]
            meta.set("TablePosition", self.plain(position), f"{self.SOURCE}: {field}", field)

    # --- acquisitions ----------------------------------------------------------------------------

    def _raw(
        self, sniff: Sniff, group: h5py.Group, header: dict, name: str, notes: list[str]
    ) -> RawData:
        table = group["data"][()].reshape(-1)  # older writers store the table as (n, 1)
        heads = np.ascontiguousarray(table["head"])
        n = len(heads)
        if n == 0:
            raise ReaderError(f"{sniff.path.name}: {name}/data holds no acquisitions")
        samples = heads["number_of_samples"].astype(int)
        channels = heads["active_channels"].astype(int)
        data = np.zeros((n, int(channels.max()), int(samples.max())), dtype=np.complex64)
        for i, row in enumerate(table["data"]):
            values = np.asarray(row, dtype=np.float32).view(np.complex64)
            data[i, : channels[i], : samples[i]] = values.reshape(channels[i], samples[i])
        trajectory = self._trajectory(table["traj"], heads, samples, notes)
        if (samples != samples[0]).any():
            notes.append(f"readouts of {len(set(samples.tolist()))} lengths, zero-padded")
        if (channels != channels[0]).any():
            notes.append("readouts with differing channel counts, zero-padded")
        meta = self._meta(header, notes)
        noise = (heads["flags"] >> np.uint64(self.NOISE_BIT - 1)) & np.uint64(1) == 1
        first = heads[np.argmin(noise)]  # the first readout that is not a noise measurement
        dwell = float(first["sample_time_us"]) * 1e-6
        if dwell > 0:
            field = "sample_time_us"
            meta.set("DwellTime", dwell, f"first data AcquisitionHeader: {field}", field)
        if (samples == samples[0]).all():
            field = "number_of_samples"
            meta.set("NumberOfPoints", int(samples[0]), f"AcquisitionHeader: {field}", field)
        waveforms = self._waveforms(group)
        provenance = self.provenance(sniff, tuple(notes))
        notes.clear()
        encoding = self._encoding(header)
        return RawData(
            data, heads, meta, provenance, name, trajectory, samples, encoding, waveforms,
            self.ACQUISITION_FLAGS,
        )

    @staticmethod
    def _trajectory(
        rows: np.ndarray, heads: np.ndarray, samples: np.ndarray, notes: list[str]
    ) -> np.ndarray | None:
        dims = heads["trajectory_dimensions"].astype(int)
        if dims.max() == 0:
            return None
        trajectory = np.zeros((len(heads), int(samples.max()), int(dims.max())), dtype=np.float32)
        for i, row in enumerate(rows):
            if dims[i] == 0:
                continue
            values = np.asarray(row, dtype=np.float32)
            if values.size != samples[i] * dims[i]:
                notes.append(f"readout {i}: trajectory of {values.size} values does not match")
                continue
            trajectory[i, : samples[i], : dims[i]] = values.reshape(samples[i], dims[i])
        return trajectory

    def _encoding(self, header: dict) -> dict:
        encodings = header.get("encoding")
        if encodings is None:
            return {}
        return self.plain(encodings if isinstance(encodings, list) else [encodings])

    def _waveforms(self, group: h5py.Group) -> list[dict]:
        if "waveforms" not in group:
            return []
        waveforms = []
        table = group["waveforms"][()].reshape(-1)
        for head, row in zip(table["head"], table["data"], strict=True):
            fields = {name: self.plain(head[name]) for name in head.dtype.names}
            data = np.asarray(row, dtype=np.uint32)
            channels, count = int(head["channels"]), int(head["number_of_samples"])
            if data.size == channels * count:
                data = data.reshape(channels, count)
            waveforms.append({**fields, "data": data})
        return waveforms

    # --- images ----------------------------------------------------------------------------------

    def _image(
        self, sniff: Sniff, group: h5py.Group, header: dict, name: str, notes: list[str]
    ) -> MRImage:
        heads = np.ascontiguousarray(group["header"][()]).reshape(-1)
        raw = group["data"][()]
        data = self._complex(raw)  # (n, channels, z, y, x)
        if data.ndim != 5:
            raise ReaderError(f"{sniff.path.name}: {name}/data has {data.ndim} dimensions, not 5")
        data = np.transpose(data, (4, 3, 2, 1, 0))  # (x, y, z, channels, n)
        dims: tuple[str, ...] = ("channel", "image")
        if data.shape[3] == 1:
            data, dims = data[:, :, :, 0, :], ("image",)
        if data.shape[-1] == 1:
            data, dims = data[..., 0], ()
        first = heads[0]
        affine = self._affine(first)
        meta = self._meta(header, notes)
        self._image_keys(meta, heads, group)
        if len(heads) > 1 and not all(
            np.array_equal(h["position"], first["position"]) for h in heads[1:]
        ):
            notes.append("images at differing positions; the affine is the first image's")
        provenance = self.provenance(sniff, tuple(notes))
        notes.clear()
        return MRImage(data, affine, meta, provenance, name, dims)

    @staticmethod
    def _complex(raw: np.ndarray) -> np.ndarray:
        if raw.dtype.names and set(raw.dtype.names) >= {"real", "imag"}:
            return raw["real"].astype(np.float64) + 1j * raw["imag"].astype(np.float64)
        return raw

    def _affine(self, head: np.ndarray) -> np.ndarray:
        """Voxel centre (i, j, k) to RAS mm: the header's position is the image centre, the
        directions and field of view are in the DICOM patient frame."""
        matrix = np.maximum(head["matrix_size"].astype(float), 1)
        steps = head["field_of_view"].astype(float) / matrix
        directions = np.stack(
            [head["read_dir"], head["phase_dir"], head["slice_dir"]], axis=1
        ).astype(float)
        affine = np.eye(4)
        affine[:3, :3] = directions * steps
        affine[:3, 3] = head["position"].astype(float) - affine[:3, :3] @ ((matrix - 1) / 2)
        return self.LPS_TO_RAS @ affine

    def _image_keys(self, meta: Meta, heads: np.ndarray, group: h5py.Group) -> None:
        kinds = sorted({self.IMAGE_TYPES.get(int(t), str(t)) for t in heads["image_type"]})
        meta.set("ImageType", kinds, "ImageHeader: image_type", "image_type")
        series = sorted({int(s) for s in heads["image_series_index"]})
        if len(series) == 1:
            field = "image_series_index"
            meta.set("SeriesNumber", series[0], f"ImageHeader: {field}", field)
        fields = {name: self.plain(heads[name]) for name in heads.dtype.names}
        meta.vendor["image_headers"] = fields
        if "attributes" in group:
            attributes = group["attributes"][()]
            meta.vendor["image_attributes"] = [
                a.decode() if isinstance(a, bytes) else str(a) for a in attributes
            ]
