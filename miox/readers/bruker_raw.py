####################################################################################################
#                                          bruker_raw.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/10/26                                                                                #
#                                                                                                  #
# Purpose: Raw k-space of Bruker ParaVision imaging scans: the fid file of ParaVision 5 and 6 and  #
#          the rawdata.jobN files of ParaVision 360, as readouts with their encoding counters.     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

# own
from miox.data import Provenance, RawData
from miox.meta import Meta
from miox.readers.base import ReaderError
from miox.readers.bruker_2dseq import Bruker2dseq


#**************************************************************************************************#
#                                            BrukerRaw                                             #
#**************************************************************************************************#
#                                                                                                  #
# A fid file holds one block per acquired readout group: the samples of every receive channel in   #
# turn (ACQ_size[0] values per channel, real and imaginary interleaved), the block padded to a     #
# multiple of 1024 bytes when GO_block_size is Standard_KBlock_Format; the word size and byte      #
# order come from GO_raw_data_format and BYTORDA. A rawdata.jobN file of ParaVision 360 holds one  #
# scan after the other without padding: ACQ_jobs gives the values per scan, ACQ_word_size the word #
# size; the channels of a scan follow each other. The blocks are in acquisition order: for         #
# Cartesian scans the echoes of a train (ACQ_phase_factor) change fastest, then the objects (NI,   #
# slices times echo images, in the order of ACQ_obj_order), then the phase-encoding trains, then   #
# the second encoding direction of a 3D scan, then the repetitions (NR), as brukerapi lays them    #
# out; the phase-encoding line of a train position is PVM_EncSteps1 (and PVM_EncSteps2), counted   #
# from its minimum for the ISMRMRD encoding steps. Radial scans (UTE, ZTE) count the projections   #
# in the first encoding step. Other pulse programs keep the readouts in file order with the block  #
# index as the scan counter and a note. Nothing is reordered or corrected; the trajectory tables   #
# of radial scans stay in the vendor header.                                                       #
#                                                                                                  #
#**************************************************************************************************#
class BrukerRaw:
    SOURCE = "Bruker acqp/method"
    FORMATS = {
        "GO_32BIT_SGN_INT": "i4", "GO_16BIT_SGN_INT": "i2", "GO_32BIT_FLOAT": "f4",
        "_32_BIT": "i4", "_16_BIT": "i2",
    }
    RADIAL = ("UTE.ppg", "UTE3D.ppg", "ZTE.ppg")
    MS = 1e-3
    KEYS = {  # standard key: (file, parameter, divisor to the key's unit)
        "SpectrometerFrequency": ("acqp", "BF1", None),
        "ResonantNucleus": ("acqp", "NUCLEUS", None),
        "EchoTime": ("method", "PVM_EchoTime", 1e3),
        "RepetitionTime": ("method", "PVM_RepetitionTime", 1e3),
        "InversionTime": ("method", "PVM_InversionTime", 1e3),
        "ExcitationFlipAngle": ("acqp", "ACQ_flip_angle", None),
        "NumberOfAverages": ("method", "PVM_NAverages", None),
        "EchoTrainLength": ("method", "PVM_RareFactor", None),
        "SliceThickness": ("method", "PVM_SliceThick", None),
        "SoftwareVersions": ("acqp", "ACQ_sw_version", None),
        "ProtocolName": ("acqp", "ACQ_protocol_name", None),
        "SequenceName": ("acqp", "ACQ_method", None),
        "PulseSequenceName": ("acqp", "PULPROG", None),
        "ManufacturersModelName": ("acqp", "ACQ_station", None),
        "InstitutionName": ("acqp", "ACQ_institution", None),
        "PatientName": ("subject", "SUBJECT_name_string", None),
        "PatientID": ("subject", "SUBJECT_id", None),
        "PatientDoB": ("subject", "SUBJECT_dbirth", None),
        "PatientWeight": ("subject", "SUBJECT_weight", None),
    }
    TEXT_KEYS = ("ResonantNucleus", "SoftwareVersions", "ProtocolName", "SequenceName",
                 "PulseSequenceName", "ManufacturersModelName", "InstitutionName", "PatientName",
                 "PatientID", "PatientDoB")

    def __init__(
        self, path: Path, header: dict[str, dict], provenance: Provenance,
        geometry: Bruker2dseq | None = None,
    ):
        self.path = path
        self.header = header
        self.provenance = provenance
        self.geometry = geometry
        self.acqp = header.get("acqp", {})
        self.method = header.get("method", {})
        self.pv360 = "360" in str(self.acqp.get("ACQ_sw_version", ""))

    def datasets(self) -> list[RawData]:
        notes: list[str] = list(self.provenance.notes)
        data = self._scans(notes)  # (blocks, channels, points)
        headers, encoding = self._headers(data.shape, notes)
        self._positions(headers, encoding, notes)
        meta = self._meta(data.shape)
        provenance = Provenance(
            self.provenance.paths, self.provenance.format, self.provenance.reader,
            self.provenance.versions, tuple(notes),
        )
        name = f"{self.path.parent.name}/{self.path.name}"
        return [RawData(data, headers, meta, provenance, name, None, None, encoding, [], {})]

    # --- samples ---------------------------------------------------------------------------------

    def _channels(self) -> int:
        count = self.method.get("PVM_EncNReceivers")
        if isinstance(count, (int, float)) and count > 0:
            return int(count)
        select = self.acqp.get("ACQ_ReceiverSelect")
        if isinstance(select, list):
            return max(1, sum(1 for v in select if str(v) == "Yes"))
        return 1

    def _dtype(self) -> np.dtype:
        code = self.acqp.get("ACQ_word_size") if self.pv360 else self.acqp.get("GO_raw_data_format")
        kind = self.FORMATS.get(str(code))
        if kind is None:
            raise ReaderError(f"{self.path}: unknown sample format {code!r}")
        order = ">" if str(self.acqp.get("BYTORDA", "little")).lower() == "big" else "<"
        return np.dtype(order + kind)

    def _scans(self, notes: list[str]) -> np.ndarray:
        dtype = self._dtype()
        channels = self._channels()
        values = np.fromfile(self.path, dtype=dtype)
        if self.pv360 and self.path.name.startswith("rawdata.job"):
            jobs = self.acqp.get("ACQ_jobs")
            job = jobs[0] if isinstance(jobs, list) and jobs and isinstance(jobs[0], list) else jobs
            per_scan = int(job[0]) if isinstance(job, list) and job else 0
            if per_scan <= 0:
                raise ReaderError(f"{self.path}: ACQ_jobs states no scan size")
            block = per_scan * channels
            padded = block
        else:
            size = self.acqp.get("ACQ_size")
            per_scan = int(size[0] if isinstance(size, list) else size)
            block = per_scan * channels
            padded = block
            if str(self.acqp.get("GO_block_size", "")) == "Standard_KBlock_Format":
                padded = int(np.ceil(block * dtype.itemsize / 1024) * 1024 // dtype.itemsize)
        n = values.size // padded
        if n == 0:
            raise ReaderError(f"{self.path}: {values.size} values, less than a block of {padded}")
        if values.size % padded:
            notes.append(f"{values.size % padded} values at the end of {self.path.name} unused")
        blocks = values[: n * padded].reshape(n, padded)[:, :block]
        blocks = blocks.reshape(n, channels, per_scan // 2, 2).astype(np.float64)
        return blocks[..., 0] + 1j * blocks[..., 1]

    # --- counters --------------------------------------------------------------------------------

    def _headers(self, shape: tuple[int, ...], notes: list[str]) -> tuple[np.ndarray, dict]:
        n, channels, points = shape
        headers = np.zeros(n, dtype=RawData.HEADER)
        headers["scan_counter"] = np.arange(n)
        headers["number_of_samples"] = points
        headers["available_channels"] = channels
        headers["active_channels"] = channels
        headers["center_sample"] = points // 2
        encoding: dict[str, Any] = {
            "blocks": n, "channels": channels, "points": points, "readout_order": "file order",
        }
        pulprog = str(self.acqp.get("PULPROG", "")).strip("<>")
        dim = self.acqp.get("ACQ_dim")
        steps1 = self._ints("PVM_EncSteps1")
        if pulprog in self.RADIAL or self.method.get("PVM_TrajKx") is not None:
            self._radial(headers, n, encoding, notes)
        elif steps1 and dim in (2, 3):
            self._cartesian(headers, n, steps1, encoding, notes)
        else:
            notes.append(f"readout counters not derived for {pulprog or 'this pulse program'}")
        return headers, encoding

    def _ints(self, name: str) -> list[int]:
        values = self.method.get(name)
        if isinstance(values, (int, float)):
            return [int(values)]
        if isinstance(values, list) and all(isinstance(v, (int, float)) for v in values):
            return [int(v) for v in values]
        return []

    def _loops(self, n: int) -> tuple[int, int, int, int, list[int]]:
        """Echoes per train, objects, trains per object, repetitions and the object order."""
        trains_per_object = int(self.acqp.get("ACQ_phase_factor") or 1)
        objects = int(self.acqp.get("NI") or 1)
        repetitions = int(self.acqp.get("NR") or 1)
        order = self.acqp.get("ACQ_obj_order")
        order = [int(v) for v in order] if isinstance(order, list) else list(range(objects))
        return trains_per_object, objects, repetitions, n, order

    def _cartesian(self, headers, n: int, steps1: list[int], encoding: dict, notes: list[str]):
        echoes, objects, repetitions, _, order = self._loops(n)
        steps2 = self._ints("PVM_EncSteps2") or [0]
        trains = len(steps1) // echoes if echoes else len(steps1)
        if echoes * trains * objects * len(steps2) * repetitions != n:
            notes.append(
                f"{n} blocks do not match {echoes} echoes x {trains} trains x {objects} objects x "
                f"{len(steps2)} partitions x {repetitions} repetitions; counters not derived"
            )
            return
        index = np.arange(n)
        echo = index % echoes
        obj = index // echoes % objects
        train = index // (echoes * objects) % trains
        partition = index // (echoes * objects * trains) % len(steps2)
        repetition = index // (echoes * objects * trains * len(steps2))
        line = echo + echoes * train
        ky = np.asarray(steps1)[line] - min(steps1)
        kz = np.asarray(steps2)[partition] - min(steps2)
        n_echo_images = int(self.acqp.get("ACQ_n_echo_images") or 1)
        headers["idx"]["kspace_encode_step_1"] = ky
        headers["idx"]["kspace_encode_step_2"] = kz
        headers["idx"]["repetition"] = repetition
        headers["idx"]["segment"] = echo
        if n_echo_images > 1:
            headers["idx"]["contrast"] = obj % n_echo_images
            slot = obj // n_echo_images
            notes.append("objects split into slices and echo images by ACQ_n_echo_images")
        else:
            slot = obj
        lookup = np.asarray(order) if len(order) >= slot.max() + 1 else np.arange(slot.max() + 1)
        headers["idx"]["slice"] = lookup[slot]
        headers["idx"]["user"][:, 0] = obj
        encoding.update(
            {
                "readout_order": "echo in train, object, train, partition, repetition",
                "echoes_per_train": echoes, "objects": objects, "trains": trains,
                "partitions": len(steps2), "repetitions": repetitions,
                "object_order": order, "phase_encoding_steps": steps1,
            }
        )

    def _radial(self, headers, n: int, encoding: dict, notes: list[str]):
        echoes, objects, repetitions, _, order = self._loops(n)
        projections = n // (echoes * objects * repetitions) if echoes * objects * repetitions else 0
        if projections * echoes * objects * repetitions != n:
            notes.append(f"{n} blocks do not divide into {objects} objects x {repetitions} reps")
            return
        index = np.arange(n)
        echo = index % echoes
        obj = index // echoes % objects
        train = index // (echoes * objects) % projections
        repetition = index // (echoes * objects * projections)
        headers["idx"]["kspace_encode_step_1"] = echo + echoes * train
        headers["idx"]["repetition"] = repetition
        lookup = np.asarray(order) if len(order) >= objects else np.arange(objects)
        headers["idx"]["slice"] = lookup[obj]
        headers["idx"]["user"][:, 0] = obj
        encoding.update(
            {
                "readout_order": "projection, object, repetition", "projections": projections,
                "objects": objects, "repetitions": repetitions, "object_order": order,
                "trajectory": "radial; PVM_TrajKx/Ky in the vendor header",
            }
        )

    def _positions(self, headers: np.ndarray, encoding: dict, notes: list[str]) -> None:
        """Slice centres and directions (LPS) from the reconstruction's visu_pars: the frame of
        each geometric slice gives the position of its first voxel, the directions of its image
        axes and whether the image is transposed (then the readout is the second axis)."""
        if self.geometry is None:
            notes.append("no slice geometry: no pdata/1/visu_pars beside the raw file")
            return
        d = self.geometry
        groups = d.groups()
        names = [n for n, _ in groups]
        index = names.index("FG_SLICE") if "FG_SLICE" in names else 0
        stride = int(np.prod([c for _, c in groups[:index]])) if index else 1
        spacing = [s for s, kind in zip(d.spacing(), d.dim_desc, strict=True) if kind == "spatial"]
        size = [n for n, kind in zip(d.size, d.dim_desc, strict=True) if kind == "spatial"]
        if len(spacing) < 2 or len(size) < 2:
            notes.append("no slice geometry: the reconstruction is not a 2D image")
            return
        for header in headers:
            frame = int(header["idx"]["slice"]) * stride
            if frame >= d.frame_count:
                continue
            axes = d.orientation(frame)
            centre = d.position(frame)
            for k in (0, 1):
                centre = centre + axes[k] * spacing[k] * (size[k] - 1) / 2
            header["position"] = centre
            read, phase = (1, 0) if d.transposed(frame) else (0, 1)
            header["read_dir"], header["phase_dir"] = axes[read], axes[phase]
            header["slice_dir"] = axes[2]
        encoding["geometry"] = "pdata/1/visu_pars: slice centres and axes of the reconstruction"

    # --- keys ------------------------------------------------------------------------------------

    def _meta(self, shape: tuple[int, ...]) -> Meta:
        meta = Meta(dict(self.header))
        field = "acqp/ACQ_sw_version"
        meta.set("Manufacturer", "Bruker", f"{self.SOURCE}: ParaVision scan folder", field)
        for key, (file, name, divisor) in self.KEYS.items():
            value = self.header.get(file, {}).get(name)
            if value is None or isinstance(value, (list, dict)):
                continue
            if key in self.TEXT_KEYS:
                value = str(value).strip("<>").strip()
                if not value:
                    continue
            elif isinstance(value, (int, float)):
                if divisor:
                    value = value / divisor
            else:
                continue
            if key == "ResonantNucleus" or key == "SpectrometerFrequency":
                value = [value]
            meta.set(key, value, f"{self.SOURCE}: {file}/{name}", f"{file}/{name}")
        for file, name in (("acqp", "SW_h"), ("method", "PVM_EffSWh")):
            sweep = self.header.get(file, {}).get(name)
            if isinstance(sweep, (int, float)) and sweep > 0:
                meta.set("DwellTime", 1 / sweep, f"{self.SOURCE}: 1 / {name}", f"{file}/{name}")
                break
        n, channels, points = shape
        field = "method/PVM_EncNReceivers"
        meta.set("NumberOfReceiveChannels", channels, f"{self.SOURCE}: receive channels", field)
        meta.set("NumberOfPoints", points, f"{self.SOURCE}: ACQ_size[0] / 2", "acqp/ACQ_size")
        matrix = self._ints("PVM_EncMatrix")
        fov = self.method.get("PVM_Fov")
        thickness = self.method.get("PVM_SliceThick")
        if matrix:
            field = "method/PVM_EncMatrix"
            meta.set("EncodedMatrixSize", (matrix + [1, 1])[:3], f"{self.SOURCE}: {field}", field)
        recon = self._ints("PVM_Matrix")
        if recon:
            field = "method/PVM_Matrix"
            meta.set("ReconMatrixSize", (recon + [1, 1])[:3], f"{self.SOURCE}: {field}", field)
        if isinstance(fov, list) and all(isinstance(v, (int, float)) for v in fov):
            fov = [float(v) for v in fov]
            if len(fov) == 2 and isinstance(thickness, (int, float)):
                fov = fov + [float(thickness)]
            if len(fov) == 3:
                text = f"{self.SOURCE}: PVM_Fov (and PVM_SliceThick)"
                meta.set("EncodedFieldOfView", fov, text, "method/PVM_Fov")
                meta.set("ReconFieldOfView", fov, text, "method/PVM_Fov")
        separation = self.acqp.get("ACQ_slice_sepn")
        if isinstance(separation, list) and separation:
            separation = separation[0]
        if isinstance(separation, (int, float)):
            field = "acqp/ACQ_slice_sepn"
            meta.set("SpacingBetweenSlices", float(separation), f"{self.SOURCE}: {field}", field)
        when = str(self.acqp.get("ACQ_time", "")).strip("<>")
        if re.match(r"\d{4}-\d\d-\d\dT", when):
            field = "acqp/ACQ_time"
            when = when.replace(",", ".")
            meta.set("AcquisitionDateTime", when, f"{self.SOURCE}: {field}", field)
        limits = {}
        steps1 = self._ints("PVM_EncSteps1")
        if steps1:
            span = max(steps1) - min(steps1)
            centre = -min(steps1)
            limits["kspace_encoding_step_1"] = {"minimum": 0, "maximum": span, "center": centre}
        objects = self.acqp.get("NI")
        if isinstance(objects, int) and objects > 0:
            limits["slice"] = {"minimum": 0, "maximum": objects - 1, "center": 0}
        if limits:
            text = f"{self.SOURCE}: PVM_EncSteps1 range (centre at step 0) and NI"
            meta.set("EncodingLimits", limits, text, "method/PVM_EncSteps1")
        return meta
