####################################################################################################
#                                       siemens_twix_raw.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 05/10/26                                                                                #
#                                                                                                  #
# Purpose: Raw k-space of a Siemens twix measurement: every data stream (image, noise, refscan,    #
#          phasecor, ...) as readouts with their complete MDH scan headers.                        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

# own
from miox.data import Provenance, RawData
from miox.meta import Meta


#**************************************************************************************************#
#                                          SiemensTwixRaw                                          #
#**************************************************************************************************#
#                                                                                                  #
# Turns the streams of a pymapvbvd measurement into RawData. The readouts come from the parser in  #
# file order with oversampling kept; the scan headers (MDH, 128 bytes in VB, 192 in VD and later)  #
# are read again from the file at the positions the parser recorded, so every field is kept: the   #
# evaluation mask, loop counters, time stamps, cut-offs, k-space centre, ICE and free parameters,  #
# slice position and orientation quaternion, table position. The headers are stored in the         #
# ISMRMRD acquisition layout with the same field mapping as siemens_to_ismrmrd (Siemens            #
# quaternion (w, x, y, z) to the read, phase and slice directions; sLC counters to idx), so the    #
# flags keep the Siemens evaluation-mask names. Protocol times are in µs, the dwell in ns.         #
#                                                                                                  #
#**************************************************************************************************#
class SiemensTwixRaw:
    SOURCE = "twix header"
    COUNTERS = (  # sLC order in the MDH: ISMRMRD idx field
        ("Line", "kspace_encode_step_1"), ("Acquisition", "average"), ("Slice", "slice"),
        ("Partition", "kspace_encode_step_2"), ("Echo", "contrast"), ("Phase", "phase"),
        ("Repetition", "repetition"), ("Set", "set"), ("Seg", "segment"),
        ("Ida", "user0"), ("Idb", "user1"), ("Idc", "user2"), ("Idd", "user3"), ("Ide", "user4"),
    )  # idx.user[5] and [6] hold the k-space centre line and partition, as siemens_to_ismrmrd
    SEX = {1: "M", 2: "F", 3: "O"}
    TEXT_KEYS = ("PatientDoB", "DeviceSerialNumber")
    INTEGER_KEYS = (
        "BaseResolution", "PhaseEncodingSteps", "PhaseEncodingStepsOutOfPlane",
        "ParallelReductionFactorInPlane", "ParallelReductionFactorOutOfPlane", "RefLinesPE",
        "EchoTrainLength",
    )
    EVAL_INFO = {  # bit number (from 1): MDH evaluation mask name
        1: "ACQEND", 2: "RTFEEDBACK", 3: "HPFEEDBACK", 4: "ONLINE", 5: "OFFLINE", 6: "SYNCDATA",
        9: "LASTSCANINCONCAT", 11: "RAWDATACORRECTION", 12: "LASTSCANINMEAS", 13: "SCANSCALEFACTOR",
        14: "2NDHADAMARPULSE", 15: "REFPHASESTABSCAN", 16: "PHASESTABSCAN", 17: "D3FFT",
        18: "SIGNREV", 19: "PHASEFFT", 20: "SWAPPED", 21: "POSTSHAREDLINE", 22: "PHASCOR",
        23: "PATREFSCAN", 24: "PATREFANDIMASCAN", 25: "REFLECT", 26: "NOISEADJSCAN",
        27: "SHARENOW", 28: "LASTMEASUREDLINE", 29: "FIRSTSCANINSLICE", 30: "LASTSCANINSLICE",
        31: "TREFFECTIVEBEGIN", 32: "TREFFECTIVEEND", 33: "MDS_REF_POSITION", 34: "SLC_AVERAGED",
        35: "TAGFLAG1", 36: "CT_NORMALIZE", 37: "SCAN_FIRST", 38: "SCAN_LAST",
        41: "FIRST_SCAN_IN_BLADE", 42: "LAST_SCAN_IN_BLADE", 43: "LAST_BLADE_IN_TR", 45: "PACE",
        46: "RETRO_LASTPHASE", 47: "RETRO_ENDOFMEAS", 48: "RETRO_REPEATTHISHEARTBEAT",
        49: "RETRO_REPEATPREVHEARTBEAT", 50: "RETRO_ABORTSCANNOW", 51: "RETRO_LASTHEARTBEAT",
        52: "RETRO_DUMMYSCAN", 53: "RETRO_ARRDETDISABLED", 54: "B1_CONTROLLOOP",
        55: "SKIP_ONLINE_PHASCOR", 56: "SKIP_REGRIDDING", 57: "MDH_VOP",
        63: "IMASCAN",
    }
    HEADER = np.dtype(  # the ISMRMRD AcquisitionHeader layout
        [
            ("version", "<u2"), ("flags", "<u8"), ("measurement_uid", "<u4"),
            ("scan_counter", "<u4"), ("acquisition_time_stamp", "<u4"),
            ("physiology_time_stamp", "<u4", (3,)), ("number_of_samples", "<u2"),
            ("available_channels", "<u2"), ("active_channels", "<u2"),
            ("channel_mask", "<u8", (16,)), ("discard_pre", "<u2"), ("discard_post", "<u2"),
            ("center_sample", "<u2"), ("encoding_space_ref", "<u2"),
            ("trajectory_dimensions", "<u2"), ("sample_time_us", "<f4"), ("position", "<f4", (3,)),
            ("read_dir", "<f4", (3,)), ("phase_dir", "<f4", (3,)), ("slice_dir", "<f4", (3,)),
            ("patient_table_position", "<f4", (3,)),
            ("idx", [
                ("kspace_encode_step_1", "<u2"), ("kspace_encode_step_2", "<u2"),
                ("average", "<u2"), ("slice", "<u2"), ("contrast", "<u2"), ("phase", "<u2"),
                ("repetition", "<u2"), ("set", "<u2"), ("segment", "<u2"), ("user", "<u2", (8,)),
            ]),
            ("user_int", "<i4", (8,)), ("user_float", "<f4", (8,)),
        ]
    )
    # MDH scan header fields: name, struct format, byte offset in VB and in VD/VE/XA
    MDH = (
        ("flags_and_dma", "<I", 0, 0), ("meas_uid", "<i", 4, 4), ("scan_counter", "<I", 8, 8),
        ("time_stamp", "<I", 12, 12), ("pmu_time_stamp", "<I", 16, 16),
        ("system_type", "<H", None, 20), ("ptab_pos_delay", "<H", None, 22),
        ("ptab_pos", "<3i", None, 24), ("eval_info_mask", "<Q", 20, 40),
        ("samples_in_scan", "<H", 28, 48), ("used_channels", "<H", 30, 50),
        ("loop_counters", "<14H", 32, 52), ("cut_off", "<2H", 60, 80),
        ("kspace_centre_column", "<H", 64, 84), ("coil_select", "<H", 66, 86),
        ("readout_offcentre", "<f", 68, 88), ("time_since_last_rf", "<I", 72, 92),
        ("kspace_centre_line", "<H", 76, 96), ("kspace_centre_partition", "<H", 78, 98),
        ("ice_program_para", "<4H", 80, None), ("free_para", "<4H", 88, None),
        ("slice_pos", "<3f", 96, 100), ("quaternion", "<4f", 108, 112),
        ("ice_program_para", "<24H", None, 128), ("reserved_para", "<4H", None, 176),
        ("application_counter", "<H", None, 184), ("application_mask", "<H", None, 186),
        ("channel_id", "<H", 124, None), ("ptab_pos_neg", "<H", 126, None),
    )
    MDH_BYTES = {"vb": 128, "vd": 192}
    KEYS = {  # standard key: (buffer, field in Siemens notation, divisor to the key's unit)
        "EchoTime": ("Phoenix", "alTE[0]", 1e6),
        "RepetitionTime": ("Phoenix", "alTR[0]", 1e6),
        "InversionTime": ("Phoenix", "alTI[0]", 1e6),
        "ExcitationFlipAngle": ("Phoenix", "adFlipAngleDegree[0]", None),
        "DwellTime": ("Phoenix", "sRXSPEC.alDwellTime[0]", 1e9),
        "SpectrometerFrequency": ("Phoenix", "sTXSPEC.asNucleusInfo[0].lFrequency", 1e6),
        "ResonantNucleus": ("Phoenix", "sTXSPEC.asNucleusInfo[0].tNucleus", None),
        "ProtocolName": ("Dicom", "tProtocolName", None),
        "PulseSequenceName": ("Phoenix", "tSequenceFileName", None),
        "SequenceName": ("Meas", "tSequenceString", None),
        "Manufacturer": ("Dicom", "Manufacturer", None),
        "ManufacturersModelName": ("Dicom", "ManufacturersModelName", None),
        "DeviceSerialNumber": ("Dicom", "DeviceSerialNumber", None),
        "SoftwareVersions": ("Dicom", "SoftwareVersions", None),
        "InstitutionName": ("Dicom", "InstitutionName", None),
        "InstitutionAddress": ("Dicom", "InstitutionAddress", None),
        "FieldStrength": ("Dicom", "flMagneticFieldStrength", None),
        "PatientPosition": ("Meas", "PatientPosition", None),
        "PatientName": ("Meas", "PatientName", None),
        "PatientWeight": ("Meas", "flUsedPatientWeight", None),
        "PatientDoB": ("Meas", "PatientBirthDay", None),
        "PatientSex": ("Meas", "PatientSex", None),
        "BaseResolution": ("Phoenix", "sKSpace.lBaseResolution", None),
        "PhaseEncodingSteps": ("Phoenix", "sKSpace.lPhaseEncodingLines", None),
        "PhaseEncodingStepsOutOfPlane": ("Phoenix", "sKSpace.lPartitions", None),
        "ParallelReductionFactorInPlane": ("Phoenix", "sPat.lAccelFactPE", None),
        "ParallelReductionFactorOutOfPlane": ("Phoenix", "sPat.lAccelFact3D", None),
        "RefLinesPE": ("Phoenix", "sPat.lRefLinesPE", None),
        "EchoTrainLength": ("Phoenix", "sFastImaging.lEchoTrainLength", None),
        "PhaseOversampling": ("Phoenix", "sKSpace.dPhaseOversamplingForDialog", None),
        "SliceThickness": ("Phoenix", "sSliceArray.asSlice[0].dThickness", None),
    }
    QUOTED = ("tNucleus", "tProtocolName", "tSequenceFileName")

    def __init__(self, path: Path, twix, vendor: dict, provenance: Provenance):
        self.path = path
        self.twix = twix
        self.vendor = vendor
        self.provenance = provenance
        self.version = "vb" if twix["image"].softwareVersion == "vb" else "vd"

    def datasets(self, streams: list[str]) -> list[RawData]:
        with open(self.path, "rb") as fh:
            return [self._stream(fh, stream) for stream in streams]

    def _stream(self, fh, stream: str) -> RawData:
        obj = self.twix[stream]
        obj.flagRemoveOS = False
        obj.flagDoRawDataCorrect = False
        data = np.ascontiguousarray(np.transpose(obj.unsorted(), (2, 1, 0)))  # (N, Cha, Col)
        headers = self._headers(fh, np.asarray(obj.memPos).astype(np.int64), data.shape)
        meta = self._meta(headers, data.shape)
        matrix = [int(data.shape[-1]), int(obj.NLin), int(obj.NPar)]
        meta.set("EncodedMatrixSize", matrix, "readout length and the Lin and Par extents")
        name = self.path.stem if stream == "image" else f"{self.path.stem}_{stream}"
        encoding = self._encoding()
        return RawData(
            data, headers, meta, self.provenance, name, None, None, encoding, [], self.EVAL_INFO
        )

    # --- scan headers ----------------------------------------------------------------------------

    def _headers(self, fh, positions: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        n, channels, samples = shape
        headers = np.zeros(n, dtype=self.HEADER)
        size = self.MDH_BYTES[self.version]
        offset_index = 2 if self.version == "vb" else 3
        for i, position in enumerate(positions):
            fh.seek(int(position))
            raw = fh.read(size)
            fields = {}
            for name, fmt, *offsets in self.MDH:
                offset = offsets[offset_index - 2]
                if offset is None:
                    continue
                value = struct.unpack_from(fmt, raw, offset)
                fields[name] = value[0] if len(value) == 1 else value
            self._fill(headers[i], fields, channels, samples)
        return headers

    def _fill(self, header: np.void, fields: dict[str, Any], channels: int, samples: int) -> None:
        header["flags"] = fields["eval_info_mask"]
        header["measurement_uid"] = fields["meas_uid"] & 0xFFFFFFFF
        header["scan_counter"] = fields["scan_counter"]
        header["acquisition_time_stamp"] = fields["time_stamp"]
        header["physiology_time_stamp"] = (fields["pmu_time_stamp"], 0, 0)
        header["number_of_samples"] = fields["samples_in_scan"]
        header["available_channels"] = channels
        header["active_channels"] = fields["used_channels"]
        header["discard_pre"], header["discard_post"] = fields["cut_off"]
        header["center_sample"] = fields["kspace_centre_column"]
        header["position"] = fields["slice_pos"]
        w, x, y, z = fields["quaternion"]
        header["phase_dir"], header["read_dir"], header["slice_dir"] = self.directions(x, y, z, w)
        if "ptab_pos" in fields:
            header["patient_table_position"] = fields["ptab_pos"]
        counters = fields["loop_counters"]
        for (_, field), value in zip(self.COUNTERS, counters, strict=True):
            if field.startswith("user"):
                header["idx"]["user"][int(field[4:])] = value
            else:
                header["idx"][field] = value
        header["idx"]["user"][5] = fields["kspace_centre_line"]
        header["idx"]["user"][6] = fields["kspace_centre_partition"]
        ice = fields["ice_program_para"]
        header["user_int"] = list(ice[:8]) + [0] * (8 - len(ice[:8]))
        header["user_float"][:2] = (fields["readout_offcentre"], fields["time_since_last_rf"])

    @staticmethod
    def directions(a: float, b: float, c: float, d: float) -> tuple[tuple, tuple, tuple]:
        """The three direction cosines of a unit quaternion (a, b, c, d), as ISMRMRD derives
        them; called with the Siemens quaternion reordered to (x, y, z, w), the first result is
        the phase direction and the second the read direction."""
        first = (1 - 2 * (b * b + c * c), 2 * (a * b + c * d), 2 * (a * c - b * d))
        second = (2 * (a * b - c * d), 1 - 2 * (a * a + c * c), 2 * (b * c + a * d))
        third = (2 * (a * c + b * d), 2 * (b * c - a * d), 1 - 2 * (a * a + b * b))
        return first, second, third

    # --- metadata --------------------------------------------------------------------------------

    def _meta(self, headers: np.ndarray, shape: tuple[int, ...]) -> Meta:
        meta = Meta(self.vendor)
        for key, (buffer, field, divisor) in self.KEYS.items():
            value = self.vendor.get(buffer, {}).get(field)
            if value is None or value == "":
                continue
            if isinstance(value, str) and field.split(".")[-1] in self.QUOTED:
                value = value.strip('"')
            if divisor is not None:
                if not isinstance(value, (int, float)):
                    continue
                value = value / divisor
            if key in self.INTEGER_KEYS and isinstance(value, float) and value.is_integer():
                value = int(value)
            if key in self.TEXT_KEYS:
                value = str(int(value)) if isinstance(value, float) else str(value)
            if key == "PatientSex":
                value = self.SEX.get(int(value), str(value))
            meta.set(key, value, f"{self.SOURCE}: {buffer}/{field}", f"{buffer}/{field}")
        n, channels, samples = shape
        meta.set("NumberOfPoints", int(samples), "readout length, oversampling included")
        meta.set("NumberOfReceiveChannels", int(channels), "channels per readout")
        if "DwellTime" in meta:
            headers["sample_time_us"] = meta["DwellTime"] * 1e6
        return meta

    def _encoding(self) -> dict:
        obj = self.twix["image"]
        return {
            "dims": list(obj.dataDims),
            "size": [int(v) for v in obj.fullSize],
            "software_version": obj.softwareVersion,
        }
