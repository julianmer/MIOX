####################################################################################################
#                                              mrd.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for MRD (version 2) files, binary or NDJSON: the header, the acquisitions with   #
#          their waveforms, the images and the Pulseq sequence items of a stream, and noise        #
#          covariance matrices.                                                                    #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

# own
from miox.data import MRImage, NoiseCovariance, PulseSequence, RawData
from miox.meta import Meta
from miox.readers.base import ReaderError
from miox.readers.ismrmrd import IsmrmrdReader
from miox.readers.pulseq import PulseqReader
from miox.sniff import Format, Sniff
from miox.yardl import YardlError, YardlFile


#**************************************************************************************************#
#                                            MrdReader                                             #
#**************************************************************************************************#
#                                                                                                  #
# An MRD file is the yardl protocol "Mrd": an optional header, then a stream of items. All         #
# acquisitions become one RawData (the waveforms attached), the images one MRImage per (image      #
# type, series, image type, size), the Pulseq items one PulseSequence. The header is kept whole as #
# the vendor header (dates and times as ISO text, enums as their symbols); its fields carry the    #
# names of the ISMRMRD header in camel case, so the keys are read as for ISMRMRD files, with the   #
# same units (times in ms, frequencies in Hz). The acquisition headers are mapped into the         #
# ISMRMRD layout as the MRD converters map them: absent optional values become 0, time stamps in   #
# ns become ns // 10^6 (the converters take the ISMRMRD stamps as ms), sample times in ns µs; the  #
# exact ns values, phase arrays, channel orders and user values beyond eight are kept in the       #
# vendor header. Trajectories are (basis, samples) per readout; a trajectory stored (samples,      #
# basis), as ismrmrd_to_mrd writes it, is recognised by its sample count and transposed. A file of #
# the protocol "MrdNoiseCovariance" becomes one NoiseCovariance.                                   #
#                                                                                                  #
#**************************************************************************************************#
class MrdReader(IsmrmrdReader):
    FORMATS = (Format.MRD,)
    PACKAGES = ("miox",)
    SOURCE = "MRD header"
    MULTIBAND_FACTOR = "multibandFactor"
    KEYS = {
        "PatientName": ("subjectInformation/patientName", None),
        "PatientWeight": ("subjectInformation/patientWeightKg", None),
        "PatientHeight": ("subjectInformation/patientHeightM", None),
        "PatientID": ("subjectInformation/patientID", None),
        "PatientDoB": ("subjectInformation/patientBirthdate", None),
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
        "FrameOfReferenceUID": ("measurementInformation/frameOfReferenceUID", None),
        "Manufacturer": ("acquisitionSystemInformation/systemVendor", None),
        "ManufacturersModelName": ("acquisitionSystemInformation/systemModel", None),
        "FieldStrength": ("acquisitionSystemInformation/systemFieldStrengthT", None),
        "NumberOfReceiveChannels": ("acquisitionSystemInformation/receiverChannels", None),
        "InstitutionName": ("acquisitionSystemInformation/institutionName", None),
        "StationName": ("acquisitionSystemInformation/stationName", None),
        "DeviceSerialNumber": ("acquisitionSystemInformation/deviceSerialNumber", None),
        "H1ResonanceFrequency": ("experimentalConditions/h1resonanceFrequencyHz", 1e6),
        "RepetitionTime": ("sequenceParameters/tR", 1000),
        "EchoTime": ("sequenceParameters/tE", 1000),
        "InversionTime": ("sequenceParameters/tI", 1000),
        "ExcitationFlipAngle": ("sequenceParameters/flipAngleDeg", None),
        "PulseSequenceType": ("sequenceParameters/sequenceType", None),
        "VendorReportedEchoSpacing": ("sequenceParameters/echoSpacing", 1000),
        "DiffusionScheme": ("sequenceParameters/diffusionScheme", None),
    }
    ENCODING_KEYS = {
        "EncodedMatrixSize": ("encodedSpace/matrixSize", None),
        "EncodedFieldOfView": ("encodedSpace/fieldOfViewMm", None),
        "ReconMatrixSize": ("reconSpace/matrixSize", None),
        "ReconFieldOfView": ("reconSpace/fieldOfViewMm", None),
        "KSpaceTrajectory": ("trajectory", None),
        "EchoTrainLength": ("echoTrainLength", None),
        "ParallelReductionFactorInPlane":
            ("parallelImaging/accelerationFactor/kspaceEncodingStep1", None),
        "ParallelReductionFactorOutOfPlane":
            ("parallelImaging/accelerationFactor/kspaceEncodingStep2", None),
    }
    UPPER_CASE = {  # enum keys whose MRD symbols are the ISMRMRD values in camel case
        "PatientSex": "subjectInformation/patientGender",
        "PatientPosition": "measurementInformation/patientPosition",
    }
    COUNTERS = {  # MRD encoding counter: ISMRMRD encoding counter
        "kspaceEncodeStep1": "kspace_encode_step_1", "kspaceEncodeStep2": "kspace_encode_step_2",
        **{name: name for name in (
            "average", "slice", "contrast", "phase", "repetition", "set", "segment")},
    }
    OPTIONAL = {  # ISMRMRD header field: MRD header field (absent values become 0)
        "scan_counter": "scanCounter", "discard_pre": "discardPre", "discard_post": "discardPost",
        "center_sample": "centerSample", "encoding_space_ref": "encodingSpaceRef",
    }
    PULSEQ = ("pulseqDefinitions", "pulseqBlocks", "pulseqRfEvent", "pulseqArbitraryGradient",
              "pulseqTrapezoidalGradient", "pulseqAdcEvent", "pulseqShape")
    RF_USE = {"undefined": "u", "excitation": "e", "refocusing": "r", "inversion": "i",
              "saturation": "s", "preparation": "p", "other": "o"}  # Pulseq's one-letter codes

    def read(self, sniff: Sniff) -> list[RawData | MRImage | PulseSequence | NoiseCovariance]:
        try:
            stream = YardlFile(sniff.path)
        except YardlError as error:
            raise ReaderError(f"{sniff.path.name}: {error}") from error
        if stream.schema.name == "MrdNoiseCovariance":
            return [self._noise_covariance(sniff, stream)]
        if stream.schema.name != "Mrd":
            raise ReaderError(f"{sniff.path.name}: yardl protocol {stream.schema.name}, not Mrd")
        header = self._header(stream.steps.get("header"))
        items: dict[str, list] = {}
        images: list[tuple[str, dict]] = []  # in stream order
        for tag, value in stream.steps.get("data") or []:
            items.setdefault(tag, []).append(value)
            if tag.startswith("image"):
                images.append((tag, value))
        notes = [f"MRD stream, {stream.encoding} encoding"]
        if not header:
            notes.append("no MRD header in the stream")
        skipped = {tag: len(v) for tag, v in items.items()
                   if tag not in ("acquisition", "waveformUint32") and not tag.startswith("image")
                   and tag not in self.PULSEQ}
        if skipped:
            notes.append("stream items not read: " + ", ".join(
                f"{count} {tag}" for tag, count in skipped.items()))
        datasets: list[RawData | MRImage | PulseSequence] = []
        acquisitions, waveforms = items.get("acquisition", []), items.get("waveformUint32", [])
        if acquisitions or waveforms:
            datasets.append(self._raw(sniff, acquisitions, waveforms, header, list(notes)))
        datasets.extend(self._images(sniff, images, header, list(notes)))
        if any(tag in items for tag in self.PULSEQ):
            datasets.append(self._sequence(sniff, items, list(notes)))
        if not datasets:
            raise ReaderError(f"{sniff.path.name}: no acquisitions, images or sequence items")
        return datasets

    # --- noise covariance ------------------------------------------------------------------------

    def _noise_covariance(self, sniff: Sniff, stream: YardlFile) -> NoiseCovariance:
        value = stream.steps["noiseCovariance"]
        matrix = value["matrix"]
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ReaderError(f"{sniff.path.name}: noise covariance of shape {matrix.shape}")
        labels = [{"number": c["coilNumber"], "name": c["coilName"]} for c in value["coilLabels"]]
        notes = [f"MRD noise covariance, {stream.encoding} encoding"]
        if labels and len(labels) != matrix.shape[0]:
            notes.append(f"{len(labels)} coil labels for {matrix.shape[0]} channels")
        meta = Meta({k: self._plain_value(v) for k, v in value.items() if k != "matrix"})
        meta.set("NumberOfReceiveChannels", int(matrix.shape[0]), "noise covariance: matrix size",
                 "matrix")
        dwell = value["noiseDwellTimeNs"] / 1e9 or None
        if dwell:
            meta.set("DwellTime", dwell, "noise covariance: noiseDwellTimeNs", "noiseDwellTimeNs")
        provenance = self.provenance(sniff, tuple(notes))
        return NoiseCovariance(
            matrix, labels, value["receiverNoiseBandwidth"], dwell, value["sampleCount"], meta,
            provenance, sniff.path.stem,
        )

    # --- header ----------------------------------------------------------------------------------

    @classmethod
    def _header(cls, value: Any) -> dict:
        """The MRD header as plain Python: dates as YYYY-MM-DD, times as HH:MM:SS with the
        fraction the file holds."""
        if value is None:
            return {}
        return cls._plain_value(value)

    @classmethod
    def _plain_value(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: cls._plain_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._plain_value(v) for v in value]
        if isinstance(value, np.timedelta64):
            return cls._time(int(value.astype("timedelta64[ns]").astype(np.int64)))
        if isinstance(value, np.datetime64):
            return str(value)
        return cls.plain(value)

    @staticmethod
    def _time(ns: int) -> str:
        seconds, fraction = divmod(ns, 10**9)
        minutes, second = divmod(seconds, 60)
        hour, minute = divmod(minutes, 60)
        text = f"{hour:02d}:{minute:02d}:{second:02d}"
        return text + f".{fraction:09d}".rstrip("0") if fraction else text

    @staticmethod
    def _limits(limits: dict) -> dict:
        """The encoding limits under the ISMRMRD counter names, absent ones left out."""
        names = {"kspaceEncodingStep0": "kspace_encoding_step_0",
                 "kspaceEncodingStep1": "kspace_encoding_step_1",
                 "kspaceEncodingStep2": "kspace_encoding_step_2",
                 **{f"user{i}": f"user_{i}" for i in range(8)}}
        return {names.get(k, k): v for k, v in limits.items() if v is not None}

    def _meta(self, header: dict, notes: list[str]) -> Meta:
        meta = super()._meta(header, notes)
        for key, path in self.UPPER_CASE.items():
            value = self._lookup(header, path)
            if isinstance(value, str):
                meta.set(key, value.upper(), f"{self.SOURCE}: {path} (upper case)", path)
        return meta

    # --- acquisitions ----------------------------------------------------------------------------

    def _raw(
        self, sniff: Sniff, acquisitions: list[dict], waveforms: list[dict], header: dict,
        notes: list[str],
    ) -> RawData:
        n = len(acquisitions)
        coils = np.array([a["data"].shape[0] for a in acquisitions], dtype=int)
        samples = np.array([a["data"].shape[1] for a in acquisitions], dtype=int)
        data = np.zeros((n, coils.max(initial=0), samples.max(initial=0)), dtype=np.complex64)
        heads = np.zeros(n, dtype=RawData.HEADER)
        heads["version"] = 1
        heads["number_of_samples"] = samples
        heads["available_channels"] = heads["active_channels"] = coils
        trajectories, swapped = [], 0
        columns: dict[str, list] = {name: [] for name in (
            "flags", "measurement_uid", "acquisition_time_stamp", "sample_time_us", "position",
            "read_dir", "phase_dir", "slice_dir", "patient_table_position",
            *self.OPTIONAL)}
        counters: dict[str, list] = {name: [] for name in self.COUNTERS}
        absent: Counter = Counter()
        extra: dict[str, list] = {"acquisition_time_stamp_ns": [], "sample_time_ns": [],
                                  "physiology_time_stamp_ns": [], "channel_order": [],
                                  "phase": [], "user_int": [], "user_float": [], "idx_user": []}
        for i, acquisition in enumerate(acquisitions):
            head = acquisition["head"]
            data[i, : coils[i], : samples[i]] = acquisition["data"]
            columns["flags"].append(head["flags"])
            columns["measurement_uid"].append(head["measurementUid"])
            stamp = head["acquisitionTimeStampNs"]
            extra["acquisition_time_stamp_ns"].append(stamp)
            columns["acquisition_time_stamp"].append(0 if stamp is None else stamp // 10**6)
            dwell = head["sampleTimeNs"]
            extra["sample_time_ns"].append(dwell)
            columns["sample_time_us"].append(0.0 if dwell is None else dwell / 1000)
            for name in ("position", "read_dir", "phase_dir", "slice_dir"):
                columns[name].append(head[self._camel(name)])
            columns["patient_table_position"].append(head["patientTablePosition"])
            for name, field in self.OPTIONAL.items():
                columns[name].append(head[field] or 0)
                absent[field] += head[field] is None
            for name in self.COUNTERS:
                counters[name].append(head["idx"][name] or 0)
                absent[f"idx.{name}"] += head["idx"][name] is None
            physiology = head["physiologyTimeStampNs"]
            extra["physiology_time_stamp_ns"].append(physiology)
            heads["physiology_time_stamp"][i, : min(len(physiology), 3)] = [
                p // 10**6 for p in physiology[:3]]
            extra["channel_order"].append(head["channelOrder"])
            for name, field, target in (("user_int", "userInt", heads["user_int"]),
                                        ("user_float", "userFloat", heads["user_float"]),
                                        ("idx_user", None, heads["idx"]["user"])):
                values = head["idx"]["user"] if field is None else head[field]
                extra[name].append(values)
                target[i, : min(len(values), 8)] = values[:8]
            extra["phase"].append(acquisition["phase"])
            trajectory, flipped = self._trajectory_of(acquisition["trajectory"], samples[i])
            trajectories.append(trajectory)
            swapped += flipped
        for name, values in columns.items():
            heads[name] = values
        for name, values in counters.items():
            heads["idx"][self.COUNTERS[name]] = values
        dims = np.array([t.shape[0] for t in trajectories], dtype=int)
        heads["trajectory_dimensions"] = dims
        trajectory = None
        if dims.max(initial=0) > 0:
            trajectory = np.zeros((n, samples.max(), dims.max()), dtype=np.float32)
            for i, t in enumerate(trajectories):
                trajectory[i, : t.shape[1], : t.shape[0]] = t.T
        if swapped:
            notes.append(f"{swapped} trajectories stored (samples, basis), transposed")
        if (samples != samples[0]).any():
            notes.append(f"readouts of {len(set(samples.tolist()))} lengths, zero-padded")
        if (coils != coils[0]).any():
            notes.append("readouts with differing channel counts, zero-padded")
        unset = [f"{field} ({count})" for field, count in absent.items() if count]
        if unset:
            notes.append("absent in the stream, 0 in the ISMRMRD layout: " + ", ".join(unset))
        meta = self._meta(header, notes)
        self._acquisition_keys(meta, heads, samples, extra["sample_time_ns"])
        self._keep(meta, extra, n, notes)
        provenance = self.provenance(sniff, tuple(notes))
        return RawData(
            data, heads, meta, provenance, "data", trajectory, samples, self._encoding(header),
            [self._waveform(w) for w in waveforms], self.ACQUISITION_FLAGS,
        )

    @staticmethod
    def _camel(name: str) -> str:
        first, *rest = name.split("_")
        return first + "".join(part.title() for part in rest)

    @staticmethod
    def _trajectory_of(trajectory: np.ndarray, samples: int) -> tuple[np.ndarray, bool]:
        if trajectory.size == 0:
            return np.zeros((0, 0), dtype=np.float32), False
        if trajectory.shape[1] != samples and trajectory.shape[0] == samples:
            return trajectory.T, True
        return trajectory, False

    def _acquisition_keys(
        self, meta: Meta, heads: np.ndarray, samples: np.ndarray, sample_time_ns: list,
    ) -> None:
        if len(heads) == 0:
            return
        noise = (heads["flags"] >> np.uint64(self.NOISE_BIT - 1)) & np.uint64(1) == 1
        first = sample_time_ns[int(np.argmin(noise))]  # the first readout that is not noise
        dwell = 0 if first is None else first / 1e9
        if dwell > 0:
            source = "first data AcquisitionHeader: sampleTimeNs"
            meta.set("DwellTime", dwell, source, "sampleTimeNs")
        if (samples == samples[0]).all():
            meta.set("NumberOfPoints", int(samples[0]), "Acquisition data: samples", "samples")

    @staticmethod
    def _keep(meta: Meta, extra: dict[str, list], n: int, notes: list[str]) -> None:
        """The per-readout values the ISMRMRD layout cannot hold, in the vendor header."""
        kept = {
            "acquisition_time_stamp_ns": extra["acquisition_time_stamp_ns"],
            "sample_time_ns": extra["sample_time_ns"],
        }
        if any(extra["physiology_time_stamp_ns"]):
            kept["physiology_time_stamp_ns"] = extra["physiology_time_stamp_ns"]
        if any(order != list(range(len(order))) for order in extra["channel_order"]):
            kept["channel_order"] = extra["channel_order"]
        if any(p is not None for p in extra["phase"]):
            kept["phase"] = extra["phase"]
        for name in ("user_int", "user_float", "idx_user"):
            if any(len(v) > 8 for v in extra[name]):
                kept[name] = extra[name]
                notes.append(f"{name} beyond eight values kept in the vendor header only")
        meta.vendor["acquisitions"] = {"count": n, **kept}

    @staticmethod
    def _waveform(waveform: dict) -> dict:
        """A waveform under the ISMRMRD waveform header names, plus the ns values."""
        data = waveform["data"]
        return {
            "flags": waveform["flags"], "measurement_uid": waveform["measurementUid"],
            "scan_counter": waveform["scanCounter"],
            "time_stamp": waveform["timeStampNs"] // 10**6,
            "time_stamp_ns": waveform["timeStampNs"],
            "sample_time_us": waveform["sampleTimeNs"] / 1000,
            "sample_time_ns": waveform["sampleTimeNs"],
            "waveform_id": waveform["waveformId"],
            "channels": int(data.shape[0]), "number_of_samples": int(data.shape[1]),
            "data": data,
        }

    # --- images ----------------------------------------------------------------------------------

    def _images(
        self, sniff: Sniff, images: list[tuple[str, dict]], header: dict, notes: list[str],
    ) -> list[MRImage]:
        groups: list[tuple[tuple, list[dict]]] = []
        current: dict[tuple, int] = {}  # key: position of its open group in ``groups``
        for tag, image in images:
            head = image["head"]
            key = (tag, head["imageSeriesIndex"], head["imageType"], image["data"].shape)
            members = groups[current[key]][1] if key in current else None
            index = head["imageIndex"]
            last = None if members is None else members[-1]["head"]["imageIndex"]
            if members is None or (index is not None and last is not None and index < last):
                # a decreasing image index starts a new set of the same series
                current[key] = len(groups)
                groups.append((key, []))
            groups[current[key]][1].append(image)
        names = []
        for (tag, series, kind, _), _ in groups:
            name = tag if series is None else f"{tag}/series_{series}"
            names.append(name if kind == "magnitude" else f"{name}/{kind}")
        results = []
        for i, ((_, members), name) in enumerate(zip(groups, names, strict=True)):
            if names.count(name) > 1:
                name += f"/{names[:i].count(name) + 1}"
            results.append(self._image(sniff, name, members, header, list(notes)))
        return results

    def _image(
        self, sniff: Sniff, name: str, images: list[dict], header: dict, notes: list[str],
    ) -> MRImage:
        data = np.stack([image["data"] for image in images])  # (n, channel, z, y, x)
        data = np.transpose(data, (4, 3, 2, 1, 0))  # (x, y, z, channel, n)
        dims: tuple[str, ...] = ("channel", "image")
        if data.shape[3] == 1:
            data, dims = data[:, :, :, 0, :], ("image",)
        if data.shape[-1] == 1:
            data, dims = data[..., 0], ()
        heads = [image["head"] for image in images]
        first = heads[0]
        x, y, z = images[0]["data"].shape[3], images[0]["data"].shape[2], images[0]["data"].shape[1]
        affine = self._affine({
            "matrix_size": np.array([x, y, z]), "field_of_view": first["fieldOfView"],
            "read_dir": first["colDir"], "phase_dir": first["lineDir"],
            "slice_dir": first["sliceDir"], "position": first["position"],
        })
        if any(not np.array_equal(h["position"], first["position"]) for h in heads[1:]):
            notes.append("images at differing positions; the affine is the first image's")
        meta = self._meta(header, notes)
        meta.set("ImageType", [first["imageType"].upper()], "ImageHeader: imageType", "imageType")
        if first["imageSeriesIndex"] is not None:
            field = "imageSeriesIndex"
            meta.set("SeriesNumber", first["imageSeriesIndex"], f"ImageHeader: {field}", field)
        meta.vendor["image_headers"] = {
            field: self._plain_value([h[field] for h in heads]) for field in first
        }
        meta.vendor["image_meta"] = [self._plain_value(self._meta_values(image["meta"]))
                                     for image in images]
        provenance = self.provenance(sniff, tuple(notes))
        return MRImage(data, affine, meta, provenance, name, dims)

    @staticmethod
    def _meta_values(values: dict) -> dict:
        """Image meta attributes: each a list of (type, value) entries, kept as the values (one
        value as itself)."""
        result = {}
        for key, entries in values.items():
            plain = [entry[1] for entry in entries]
            result[key] = plain[0] if len(plain) == 1 else plain
        return result

    # --- sequence --------------------------------------------------------------------------------

    def _sequence(self, sniff: Sniff, items: dict[str, list], notes: list[str]) -> PulseSequence:
        definitions: dict[str, Any] = {}
        given = items.get("pulseqDefinitions", [])
        if len(given) > 1:
            notes.append(f"{len(given)} Pulseq definitions items; the last is used")
        if given:
            d = given[-1]
            definitions = {
                "AdcRasterTime": d["adcRasterTime"],
                "BlockDurationRaster": d["blockDurationRaster"],
                "GradientRasterTime": d["gradientRasterTime"],
                "RadiofrequencyRasterTime": d["radiofrequencyRasterTime"],
            }
            if d["name"] is not None:
                definitions["Name"] = d["name"]
            if d["fov"] is not None:
                definitions["FOV"] = [d["fov"]["x"], d["fov"]["y"], d["fov"]["z"]]
            if d["totalDuration"] is not None:
                definitions["TotalDuration"] = d["totalDuration"]
            for key, text in d["custom"].items():
                values = [PulseqReader._scalar(v) for v in text.split()]
                definitions[key] = values[0] if len(values) == 1 else values
        rows = [block for blocks in items.get("pulseqBlocks", []) for block in blocks]
        columns = PulseqReader.BLOCK_COLUMNS["duration"]
        blocks = np.array([tuple(row[c] for c in columns) for row in rows],
                          dtype=[(c, "<i8") for c in columns])
        events = {
            "rf": {e["id"]: {
                "amp": e["amp"], "mag_id": e["magId"], "phase_id": e["phaseId"],
                "time_id": e["timeId"], "center": e["center"], "delay": e["delay"],
                "freq_ppm": e["freqPPM"], "phase_ppm": e["phasePPM"], "freq": e["freqOffset"],
                "phase": e["phaseOffset"], "use": self.RF_USE.get(e["use"], e["use"]),
            } for e in items.get("pulseqRfEvent", [])},
            "gradients": {e["id"]: {
                "amp": e["amp"], "first": e["first"], "last": e["last"],
                "shape_id": e["shapeId"], "time_id": e["timeId"], "delay": e["delay"],
            } for e in items.get("pulseqArbitraryGradient", [])},
            "trap": {e["id"]: {
                "amp": e["amp"], "rise": e["rise"], "flat": e["flat"], "fall": e["fall"],
                "delay": e["delay"],
            } for e in items.get("pulseqTrapezoidalGradient", [])},
            "adc": {e["id"]: {
                "num": e["num"], "dwell": e["dwell"], "delay": e["delay"],
                "freq_ppm": e["freqPPM"], "phase_ppm": e["phasePPM"], "freq": e["freq"],
                "phase": e["phase"], "phase_shape_id": e["phaseShapeId"],
            } for e in items.get("pulseqAdcEvent", [])},
        }
        shapes = {}
        for shape in items.get("pulseqShape", []):
            values = np.asarray(shape["data"], dtype=np.float64)
            if len(values) != shape["numSamples"]:
                values = PulseqReader.decompress(values)
                if len(values) != shape["numSamples"]:
                    notes.append(f"shape {shape['id']}: {len(values)} samples decompressed, "
                                 f"{shape['numSamples']} declared")
            shapes[shape["id"]] = values
        notes.append("the MRD stream does not state the Pulseq revision; events in the columns "
                     "of revision 1.5")
        meta = PulseqReader()._meta(definitions, "", {})
        provenance = self.provenance(sniff, tuple(notes))
        return PulseSequence(
            "", definitions, blocks, events, {}, shapes, {}, meta, provenance, sniff.path.stem,
        )
