####################################################################################################
#                                        philips_raw_lab.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for Philips raw k-space exports: the .lab label file (one 64-byte label per      #
#          acquired or control event), the .raw data file (the samples of every data label, bit-   #
#          packed on recent systems) and the .sin text file (the reconstruction parameters).       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Any

import numpy as np

# own
from miox.data import RawData
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.readers.ismrmrd import IsmrmrdReader
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                           MiraDecoder                                            #
#**************************************************************************************************#
#                                                                                                  #
# Unpacks the bit-packed ("MIRA") data blocks of a .raw file, as philips_to_ismrmrd decodes them.  #
# A block is a sequence of chunks; a chunk starts with its decoded size in bytes, its encoded size #
# in bytes and the byte offset of its values in the decoded block, then holds an MSB-first bit     #
# stream over little-endian 32-bit words: groups of up to 16 values, each group preceded by a      #
# 5-bit bit resolution and a 5-bit shift, the values two's-complement integers of that resolution, #
# restored as (value << shift) + (1 << shift >> 1). The block ends when the decoded sizes add up   #
# to the label's data size; 8 bytes follow every block in the file.                                #
#                                                                                                  #
#**************************************************************************************************#
class MiraDecoder:
    CHUNK = struct.Struct("<HHI")
    TRAILER = 8
    POWERS = {n: 1 << np.arange(n - 1, -1, -1, dtype=np.int64) for n in range(1, 33)}

    @classmethod
    def block(cls, raw: np.ndarray, start: int, data_size: int) -> tuple[np.ndarray, int]:
        """The values of the block at ``start`` and the position after its chunks."""
        out = np.zeros(data_size // 4, dtype=np.int32)
        pos, decoded = start, 0
        while decoded < data_size:
            size, coded, offset = cls.CHUNK.unpack_from(raw, pos)
            pos += cls.CHUNK.size
            out[offset // 4 : (offset + size) // 4] = cls.chunk(raw[pos : pos + coded], size // 4)
            pos += coded
            decoded += size
        return out, pos

    @classmethod
    def chunk(cls, coded: np.ndarray, n: int) -> np.ndarray:
        padded = np.concatenate([coded, np.zeros(-len(coded) % 4 + 4, dtype=np.uint8)])
        bits = np.unpackbits(padded.view("<u4").byteswap().view(np.uint8))
        out = np.zeros(n, dtype=np.int64)
        pos = idx = 0
        while idx < n:
            resolution = int(bits[pos : pos + 5] @ cls.POWERS[5])
            shift = int(bits[pos + 5 : pos + 10] @ cls.POWERS[5])
            pos += 10
            count = min(16, n - idx)
            if resolution:
                values = bits[pos : pos + count * resolution].reshape(count, resolution)
                values = values @ cls.POWERS[resolution]
                negative = values >= 1 << (resolution - 1)
                values = np.where(negative, values - (1 << resolution), values)
                pos += count * resolution
            else:
                values = np.zeros(count, dtype=np.int64)
            out[idx : idx + count] = (values << shift) + (1 << shift >> 1)
            idx += count
        return out

    @classmethod
    def valid(cls, raw: np.ndarray, start: int, data_size: int, coded_size: int) -> bool:
        """Whether a chunk sequence of the stated sizes starts at ``start``."""
        pos, decoded = start, 0
        while decoded < data_size:
            if pos + cls.CHUNK.size > len(raw):
                return False
            size, coded, offset = cls.CHUNK.unpack_from(raw, pos)
            if size == 0 or size % 4 or offset % 4 or offset + size > data_size:
                return False
            pos += cls.CHUNK.size + coded
            decoded += size
            if pos - start > coded_size:
                return False
        return pos - start + cls.TRAILER == coded_size


#**************************************************************************************************#
#                                       PhilipsRawLabReader                                        #
#**************************************************************************************************#
#                                                                                                  #
# The .sin file lists parameters as "i j k: name : values" (the indices count the mix, echo and    #
# location, or channels, from 1); it is kept whole in the vendor header and gives the standard     #
# keys. The .lab file holds 64-byte labels in acquisition order, in the layout of                  #
# philips_to_ismrmrd's new_label (old_label when the .sin has an enable_pda parameter: those files #
# hold plain 16-bit samples); every label is kept in the vendor header. The .raw file holds, from  #
# byte 512, one block per label with a coded data size, in label order; bit-packed blocks (raw     #
# formats 4 and 6) are located by their chunk structure, so a block whose size on disk differs     #
# from its label (the DC-fixed data of the sample file) is skipped over and noted. Every control   #
# type with decodable data becomes one RawData (the normal data under the file's name, then        #
# _frc_noise_data, _echo_phase_data, ...), its readouts in label order with the samples of each    #
# channel in turn; 32-bit samples are kept as complex128. Headers follow philips_to_ismrmrd: e1    #
# and e2 profile numbers as the encoding steps, echo as contrast, location as slice, dynamic scan  #
# as repetition, cardiac phase as phase, measurement as average, row as set, the sequence number   #
# as the scan counter, rtop offset and RR interval as physiology time stamps, the noise flag on    #
# noise data; mix, extra attribute, e3 profile, RF echo and gradient echo in the user counters;    #
# control, random phase, measurement phase and sign, gain index, raw format, active channels and   #
# encoding time in the user integers, the normalization factor in the first user float. Nothing is #
# corrected; a reconstruction needs the random phase r (a signed 16-bit number, though the label   #
# stores it unsigned), which Philips' own reader removes from sample s of a readout as             #
# exp(i·π·r·(s + 10)/327680) - a phase ramp along the readout, not the constant π·r/32767 the      #
# converter applies - and the chop (every other e1 profile negated).                               #
# The slice position comes from the location's offcentres (AP, RL, FH to LPS), its normal from the #
# image orientation rows; which image axis is the readout is not stated, so the read and phase     #
# directions stay zero.                                                                            #
#                                                                                                  #
#**************************************************************************************************#
class PhilipsRawLabReader(Reader):
    FORMATS = (Format.PHILIPS_RAW_LAB,)
    SOURCE = "sin header"
    FLAGS = IsmrmrdReader.ACQUISITION_FLAGS
    NOISE_FLAG = 19
    RAW_OFFSET = 512
    CODED_FORMATS = (4, 6)
    SEARCH_LIMIT = 1 << 22  # bytes searched for the next bit-packed block after a size mismatch
    SIN_LINE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s*:\s*(\w+)\s*:(.*)$")
    LABEL_TYPES = {
        0x7F01: "standard", 0x7F02: "igeo", 0x7F03: "igeo_pmc", 0x7F04: "coil_pos",
        0x7F05: "non_lin",
    }
    CONTROLS = {
        0: "normal_data", 1: "dc_offset_data", 2: "junk_data", 3: "echo_phase_data", 4: "no_data",
        5: "next_phase", 6: "suspend", 7: "resume", 8: "total_end", 9: "invalidation",
        10: "type_nr_end", 11: "validation", 12: "no_operation", 13: "dyn_scan_info",
        14: "selective_end", 15: "frc_ch_data", 16: "frc_noise_data", 17: "reference_data",
        18: "dc_fixed_data", 19: "navigator_data", 20: "flush", 21: "recon_end",
        22: "image_status", 23: "tracking", 24: "fluoroscopy_toggle", 25: "rejected_data",
        26: "progress_info", 27: "end_prep_phase", 28: "channel_definition",
    }
    NOISE_CONTROL = 16
    COUNTERS = (  # label field: ISMRMRD encoding counter, as philips_to_ismrmrd
        ("e1_profile_nr", "kspace_encode_step_1"), ("e2_profile_nr", "kspace_encode_step_2"),
        ("measurement_nr", "average"), ("location_nr", "slice"), ("echo_nr", "contrast"),
        ("cardiac_phase_nr", "phase"), ("dynamic_scan_nr", "repetition"), ("row_nr", "set"),
    )
    USER_COUNTERS = ("mix_nr", "extra_attr_nr", "e3_profile_nr", "rf_echo_nr", "grad_echo_nr")
    USER_INTS = (
        "control", "random_phase", "measurement_phase", "measurement_sign",
        "gain_setting_index", "raw_format", "channels_active", "enc_time",
    )
    COMMON = [  # label fields shared by both layouts after the type-specific head
        ("control", "u1"), ("monitoring_flag", "u1"), ("measurement_phase", "i1"),
        ("measurement_sign", "i1"), ("gain_setting_index", "u1"),
    ]
    TAIL = [
        ("progress_cnt", "<u2"), ("mix_nr", "<u2"), ("dynamic_scan_nr", "<u2"),
        ("cardiac_phase_nr", "<u2"), ("echo_nr", "<u2"), ("location_nr", "<u2"),
        ("row_nr", "<u2"), ("extra_attr_nr", "<u2"), ("measurement_nr", "<u2"),
        ("e1_profile_nr", "<u2"), ("e2_profile_nr", "<u2"), ("e3_profile_nr", "<u2"),
        ("rf_echo_nr", "<u2"), ("grad_echo_nr", "<u2"), ("enc_time", "<u2"),
        ("random_phase", "<u2"), ("rr_interval", "<u2"), ("rtop_offset", "<u2"),
        ("channels_active", "<u4"),
    ]
    NEW_LABEL = np.dtype(
        [("data_size", "<u4"), ("coded_data_size", "<u4"), ("normalization_factor", "<f4"),
         ("seq_nr", "<u2"), ("label_type", "<u2")] + COMMON + [("raw_format", "u1"),
         ("spare_1", "<u2")] + TAIL
    )
    OLD_LABEL = np.dtype(
        [("data_size", "<u4"), ("coded_data_size", "<u4"), ("src_code", "<u2"),
         ("dst_code", "<u2"), ("seq_nr", "<u2"), ("label_type", "<u2")] + COMMON
        + [("spare_1", "i1"), ("spare_2", "<u2")] + TAIL
    )
    TEXT_KEYS = ("ProtocolName", "ReceiveCoilName")  # all words of the line, joined
    KEYS = {  # standard key: sin parameter, divisor to the key's unit, whether zero means unset
        "ProtocolName": ("scan_name", None, False),
        "RepetitionTime": ("repetition_times", 1000, False),
        "EchoTime": ("echo_times", 1000, False),
        "InversionTime": ("inversion_delays", 1000, True),
        "ExcitationFlipAngle": ("flip_angles", None, False),
        "DwellTime": ("sample_time_interval", 1e6, True),
        "NumberOfReceiveChannels": ("nr_measured_channels", None, True),
        "ReceiveCoilName": ("receiving_coils", None, False),
        "SliceThickness": ("slice_thickness", None, True),
        "SpacingBetweenSlices": ("spacing_between_slices_arr", None, True),
        "NumberOfAverages": ("nr_measurements", None, True),
        "EchoTrainLength": ("turbo_factor", None, True),
    }
    LIMITS = (  # ISMRMRD counter: sin count parameter
        ("slice", "nr_locations"), ("set", "nr_rows"), ("phase", "nr_cardiac_phases"),
        ("repetition", "nr_dynamic_scans"), ("contrast", "nr_echoes"),
        ("average", "nr_measurements"),
    )
    MONTHS = {m: i + 1 for i, m in enumerate(
        ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
    )}

    def read(self, sniff: Sniff) -> list[RawData]:
        files = {p.suffix.lower(): p for p in (sniff.path,) + sniff.companions}
        for suffix in (".lab", ".raw", ".sin"):
            if suffix not in files:
                raise ReaderError(f"{sniff.path.name}: no {suffix} file beside it")
        sin = self.sin(files[".sin"])
        old = "enable_pda" in sin
        labels = np.fromfile(files[".lab"], dtype=self.OLD_LABEL if old else self.NEW_LABEL)
        if len(labels) == 0:
            raise ReaderError(f"{files['.lab'].name}: no labels")
        notes: list[str] = []
        blocks = self._blocks(files[".raw"], labels, old, notes)
        if not blocks:
            raise ReaderError(f"{files['.raw'].name}: no data blocks could be read")
        channels = self._channels(sin)
        vendor = {
            "sin": sin, "labels": {name: labels[name].tolist() for name in labels.dtype.names},
            "label_layout": "old" if old else "new",
        }
        meta = self._meta(sin, vendor)
        datasets = []
        for control in sorted({int(labels[i]["control"]) for i in blocks}, key=self._order):
            rows = [i for i in blocks if labels[i]["control"] == control]
            name = sniff.path.stem
            if control:
                name += "_" + self.CONTROLS.get(control, f"control_{control}")
            provenance = self.provenance(sniff, tuple(notes))
            datasets.append(
                self._stream(rows, labels, blocks, channels, old, meta, sin, provenance, name)
            )
        return datasets

    @staticmethod
    def _order(control: int) -> tuple[int, int]:
        return (control != 0, control)

    # --- sin -------------------------------------------------------------------------------------

    @classmethod
    def sin(cls, path: Path) -> dict[str, list[dict]]:
        """Every parameter line: name -> [{"index": [i, j, k], "values": [...]}, ...]."""
        params: dict[str, list[dict]] = {}
        for line in path.read_text(errors="replace").splitlines():
            match = cls.SIN_LINE.match(line)
            if match is None:
                continue
            i, j, k, name, rest = match.groups()
            values = [cls._scalar(v) for v in rest.split()]
            entry = {"index": [int(i), int(j), int(k)], "values": values}
            params.setdefault(name, []).append(entry)
        return params

    @staticmethod
    def _scalar(text: str) -> Any:
        try:
            return int(text)
        except ValueError:
            pass
        try:
            return float(text)
        except ValueError:
            return text

    @staticmethod
    def first(sin: dict, name: str) -> list:
        entries = sin.get(name)
        return entries[0]["values"] if entries else []

    def _channels(self, sin: dict) -> int:
        for name in ("nr_measured_channels", "max_measured_channels"):
            values = self.first(sin, name)
            if values and isinstance(values[0], int) and values[0] > 0:
                return values[0]
        return 1

    # --- raw blocks ------------------------------------------------------------------------------

    def _blocks(
        self, path: Path, labels: np.ndarray, old: bool, notes: list[str]
    ) -> dict[int, np.ndarray]:
        raw = np.fromfile(path, dtype=np.uint8)
        blocks: dict[int, np.ndarray] = {}
        pos = self.RAW_OFFSET
        for i, label in enumerate(labels):
            coded, size = int(label["coded_data_size"]), int(label["data_size"])
            if coded == 0:
                continue
            packed = not old and int(label["raw_format"]) in self.CODED_FORMATS
            if packed:
                start = self._locate(raw, pos, size, coded)
                if start is None:
                    notes.append(f"no bit-packed block found for label {i} after byte {pos}")
                    break
                if start != pos:
                    notes.append(f"{start - pos} bytes skipped before label {i}'s block")
                blocks[i], _ = MiraDecoder.block(raw, start, size)
                pos = start + coded
            elif old or int(label["control"]) in (0, self.NOISE_CONTROL):
                if pos + size > len(raw):
                    notes.append(f"label {i}: {size} bytes of samples expected, file ends")
                    break
                dtype = "<i2" if old else "<i4"
                blocks[i] = raw[pos : pos + size].view(dtype)
                pos += coded
            else:
                pos += coded
        return blocks

    def _locate(self, raw: np.ndarray, pos: int, size: int, coded: int) -> int | None:
        for start in range(pos, min(pos + self.SEARCH_LIMIT, len(raw)), 2):
            if MiraDecoder.valid(raw, start, size, coded):
                return start
        return None

    # --- streams ---------------------------------------------------------------------------------

    def _stream(
        self, rows: list[int], labels: np.ndarray, blocks: dict[int, np.ndarray], channels: int,
        old: bool, meta: Meta, sin: dict, provenance, name: str,
    ) -> RawData:
        shapes = [self._shape(blocks[i], labels[i], channels) for i in rows]
        max_channels = max(c for c, _ in shapes)
        max_samples = max(s for _, s in shapes)
        complex_type = np.complex64 if old else np.complex128
        data = np.zeros((len(rows), max_channels, max_samples), dtype=complex_type)
        headers = np.zeros(len(rows), dtype=RawData.HEADER)
        samples = np.zeros(len(rows), dtype=int)
        for r, (i, (c, s)) in enumerate(zip(rows, shapes, strict=True)):
            values = blocks[i].astype(np.float32 if old else np.float64).reshape(c, s, 2)
            data[r, :c, :s] = values[..., 0] + 1j * values[..., 1]
            samples[r] = s
            self._header(headers[r], labels[i], c, s, old)
        stream_meta = Meta(meta.vendor)
        stream_meta.update(meta)
        stream_meta.source.update(meta.source)
        stream_meta.fields.update(meta.fields)
        if (samples == samples[0]).all():
            field = "labels/data_size"
            stream_meta.set("NumberOfPoints", int(samples[0]), f"label: {field}", field)
        self._positions(headers, sin)
        encoding = {
            "readout_order": "label order",
            "controls": sorted({int(labels[i]["control"]) for i in rows}),
            "channels": int(max_channels), "samples": int(max_samples),
            "sample_bytes": 2 if old else 4,
        }
        return RawData(
            data, headers, stream_meta, provenance, name, None, samples, encoding, [], self.FLAGS
        )

    def _shape(self, block: np.ndarray, label: np.void, channels: int) -> tuple[int, int]:
        """Channels and samples of a block: the label's active channel count when it is set and
        divides the values (the echo-phase data of the sample file states 2 of 12 channels), else
        the measured channel count, else one channel."""
        values = block.size // 2
        active = int(label["channels_active"])
        for c in (active, channels, 1):
            if c > 0 and values % c == 0:
                return c, values // c
        return 1, values

    def _header(self, header: np.void, label: np.void, channels: int, samples: int, old: bool):
        header["scan_counter"] = label["seq_nr"]
        header["number_of_samples"] = samples
        header["available_channels"] = channels
        header["active_channels"] = channels
        header["center_sample"] = samples // 2
        header["physiology_time_stamp"] = (label["rtop_offset"], label["rr_interval"], 0)
        for field, counter in self.COUNTERS:
            header["idx"][counter] = label[field]
        for k, field in enumerate(self.USER_COUNTERS):
            header["idx"]["user"][k] = label[field]
        for k, field in enumerate(self.USER_INTS):
            value = int(label[field]) if field in label.dtype.names else 0
            header["user_int"][k] = min(value, 2**31 - 1)
        if not old:
            header["user_float"][0] = label["normalization_factor"]
        if int(label["control"]) == self.NOISE_CONTROL:
            header["flags"] = 1 << (self.NOISE_FLAG - 1)

    def _positions(self, headers: np.ndarray, sin: dict) -> None:
        """Slice centres (LPS, mm) and normals per location from the sin geometry."""
        offcentres = {e["index"][2]: e["values"] for e in sin.get("loc_ap_rl_fh_offcentres", [])}
        rows = {e["index"][2]: e["values"] for e in sin.get("loc_ap_rl_fh_row_image_oris", [])}
        cols = {e["index"][2]: e["values"] for e in sin.get("loc_ap_rl_fh_col_image_oris", [])}
        for header in headers:
            location = int(header["idx"]["slice"]) + 1
            centre = offcentres.get(location)
            if centre and len(centre) == 3:
                header["position"] = self._lps(centre)
            row, col = rows.get(location), cols.get(location)
            if row and col and len(row) == 3 and len(col) == 3:
                header["slice_dir"] = np.cross(self._lps(row), self._lps(col))

    @staticmethod
    def _lps(ap_rl_fh: list) -> np.ndarray:
        """Philips (AP, RL, FH) components as LPS: L = RL, P = AP, S = FH."""
        return np.array([ap_rl_fh[1], ap_rl_fh[0], ap_rl_fh[2]], dtype=float)

    # --- standard keys ---------------------------------------------------------------------------

    def _meta(self, sin: dict, vendor: dict) -> Meta:
        meta = Meta(vendor)
        meta.set("Manufacturer", "Philips", "sin file", "sin")
        for key, (name, divisor, zero_unset) in self.KEYS.items():
            values = self.first(sin, name)
            if not values:
                continue
            if key in self.TEXT_KEYS:
                value: Any = " ".join(str(v) for v in values)
            elif not all(isinstance(v, (int, float)) for v in values):
                continue
            elif zero_unset and not any(values):
                continue
            else:
                scaled = [v / divisor if divisor else v for v in values]
                value = scaled[0] if len(scaled) == 1 else scaled
            meta.set(key, value, f"{self.SOURCE}: {name}", f"sin/{name}/0/values")
        when = self._datetime(self.first(sin, "start_scan_date_time"))
        if when:
            meta.set(
                "AcquisitionDateTime", when, f"{self.SOURCE}: start_scan_date_time",
                "sin/start_scan_date_time/0/values",
            )
        self._matrix_keys(meta, sin)
        limits = self._limits(sin)
        if limits:
            self._put(
                meta, "EncodingLimits", limits,
                "min/max_encoding_numbers and the counts per dimension, as philips_to_ismrmrd",
                ("max_encoding_numbers",),
            )
        orientation = self._orientation(sin)
        if orientation:
            self._put(
                meta, "ImageOrientationPatientDICOM", orientation,
                "row and column image orientations of the first location, as LPS",
                ("loc_ap_rl_fh_row_image_oris", "loc_ap_rl_fh_col_image_oris"),
            )
        return meta

    def _put(self, meta: Meta, key: str, value: Any, text: str, names: tuple) -> None:
        fields = ", ".join(f"sin/{name}/0/values" for name in names)
        meta.set(key, value, f"{self.SOURCE}: {text}", fields)

    def _matrix_keys(self, meta: Meta, sin: dict) -> None:
        """Matrix sizes and fields of view as philips_to_ismrmrd derives them: the recon
        resolutions, times the oversample factors for the encoded space, times the voxel sizes
        for the fields of view."""
        recon = self.first(sin, "recon_resolutions")[:3]
        over = self.first(sin, "oversample_factors")[:3]
        voxel = self.first(sin, "voxel_sizes")[:3]
        if len(recon) != 3 or not all(isinstance(v, (int, float)) for v in recon + over + voxel):
            return
        self._put(meta, "ReconMatrixSize", [int(v) for v in recon], "recon_resolutions",
                  ("recon_resolutions",))
        if len(over) == 3:
            encoded = [int(round(r * o)) for r, o in zip(recon, over, strict=True)]
            self._put(meta, "EncodedMatrixSize", encoded, "recon_resolutions x oversample_factors",
                      ("recon_resolutions", "oversample_factors"))
        if len(voxel) == 3:
            fov = [v * r for v, r in zip(voxel, recon, strict=True)]
            self._put(meta, "ReconFieldOfView", fov, "voxel_sizes x recon_resolutions",
                      ("voxel_sizes", "recon_resolutions"))
            if len(over) == 3:
                encoded_fov = [f * o for f, o in zip(fov, over, strict=True)]
                self._put(meta, "EncodedFieldOfView", encoded_fov,
                          "voxel_sizes x recon_resolutions x oversample_factors",
                          ("voxel_sizes", "recon_resolutions", "oversample_factors"))

    def _limits(self, sin: dict) -> dict:
        limits: dict[str, dict] = {}
        low, high = self.first(sin, "min_encoding_numbers"), self.first(sin, "max_encoding_numbers")
        for k, counter in ((1, "kspace_encoding_step_1"), (2, "kspace_encoding_step_2")):
            if len(low) <= k or len(high) <= k:
                continue
            if not all(isinstance(v, int) for v in (low[k], high[k])):
                continue
            span = high[k] - low[k]
            limits[counter] = {"minimum": 0, "maximum": span, "center": span // 2}
        for counter, name in self.LIMITS:
            values = self.first(sin, name)
            if values and isinstance(values[0], int) and values[0] > 0:
                limits[counter] = {"minimum": 0, "maximum": values[0] - 1, "center": 0}
        return limits

    def _orientation(self, sin: dict) -> list[float] | None:
        row = self.first(sin, "loc_ap_rl_fh_row_image_oris")
        col = self.first(sin, "loc_ap_rl_fh_col_image_oris")
        if len(row) == 3 and len(col) == 3:
            return self._lps(row).tolist() + self._lps(col).tolist()
        return None

    def _datetime(self, values: list) -> str | None:
        """"17-JUN-2025 21:06:21.61" as ISO 8601."""
        if len(values) < 2:
            return None
        try:
            day, month, year = str(values[0]).split("-")
            return f"{int(year):04d}-{self.MONTHS[month.upper()]:02d}-{int(day):02d}T{values[1]}"
        except (ValueError, KeyError):
            return None
