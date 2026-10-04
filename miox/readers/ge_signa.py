####################################################################################################
#                                            ge_signa.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 04/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for GE Signa images before DICOM: Genesis 5.x files (with or without the "IMGF"  #
#          pixel header) and Signa 4.x files, one slice per file, assembled per series and echo.   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import struct
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# own
from miox.data import MRImage
from miox.meta.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                            GeHeader                                              #
#**************************************************************************************************#
#                                                                                                  #
# The header of one GE Signa file as dictionaries per block (suite, exam, series, image), plus     #
# the values every variant provides: the matrix, the pixel sizes, the three corners of the image   #
# in patient RAS millimetres, the slice thickness and gap, the series and echo keys, the modality  #
# and where the big-endian 16-bit pixels start. Genesis 5.x headers are big-endian IEEE values     #
# at fixed byte offsets within each block; the blocks follow the 114-byte suite header, or are     #
# located by the pointers of the "IMGF" pixel header. Signa 4.x headers are blocks of 16-bit       #
# words whose floats are in Data General format. Offsets follow ITK's GE readers.                  #
#                                                                                                  #
#**************************************************************************************************#
class GeHeader:
    IMGF = 0x494D4746
    SUITE_LENGTH, EXAM_LENGTH, SERIES_LENGTH, IMAGE_LENGTH = 114, 1024, 1020, 1022
    # Genesis block fields: name -> (format, offset[, length]); format 's' is a string
    SUITE = {"suite_id": ("s", 0, 4), "product_id": ("s", 7, 13)}
    EXAM = {
        "exam_number": ("H", 8), "hospital_name": ("s", 10, 33), "magnet_strength_gauss": ("i", 80),
        "patient_id": ("s", 84, 13), "patient_name": ("s", 97, 25), "patient_age": ("h", 122),
        "patient_age_notation": ("h", 124), "patient_sex": ("h", 126),
        "patient_weight_g": ("i", 128),
        "history": ("s", 134, 61), "requisition_number": ("s", 195, 13),
        "exam_datetime": ("t", 208), "referring_physician": ("s", 212, 33),
        "radiologist": ("s", 245, 33), "operator": ("s", 278, 4),
        "exam_description": ("s", 282, 23),
        "exam_type": ("s", 305, 3), "system_id": ("s", 318, 9), "unique_system_id": ("s", 438, 16),
        "service_id": ("s", 454, 16), "mobile_location": ("s", 470, 4),
    }
    SERIES = {
        "series_number": ("h", 10), "series_datetime": ("t", 12), "actual_datetime": ("t", 16),
        "series_description": ("s", 20, 30), "primary_receiver_suite": ("s", 50, 9),
        "archiver_suite": ("s", 59, 9), "series_type": ("h", 68), "source": ("h", 70),
        "plane": ("h", 72), "scan_type": ("h", 74), "patient_position": ("i", 76),
        "patient_entry": ("i", 80), "anatomical_reference": ("s", 84, 3),
        "landmark_horizontal_offset": ("f", 88), "protocol": ("s", 92, 25), "contrast": ("h", 118),
        "start_ras": ("s", 120, 1), "start_location": ("f", 122), "end_ras": ("s", 126, 1),
        "end_location": ("f", 128), "pulse_sequence": ("h", 132), "sort_order": ("h", 134),
        "acquisitions": ("h", 140),
    }
    IMAGE = {  # the part shared by MR and CT images
        "image_number": ("h", 12), "image_datetime": ("t", 14), "actual_datetime": ("t", 18),
        "scan_duration_us": ("f", 22), "slice_thickness": ("f", 26), "matrix_x": ("h", 30),
        "matrix_y": ("h", 32), "display_fov_x": ("f", 34), "display_fov_y": ("f", 38),
        "acquisition_dim_x": ("f", 42), "acquisition_dim_y": ("f", 46), "pixel_size_x": ("f", 50),
        "pixel_size_y": ("f", 54), "pixel_data_id": ("s", 58, 14), "iv_contrast": ("s", 72, 17),
        "oral_contrast": ("s", 89, 17), "contrast_mode": ("h", 106), "plane": ("h", 114),
        "scan_spacing": ("f", 116), "compression": ("h", 120), "scout_type": ("h", 122),
        "location_ras": ("s", 124, 1), "location": ("f", 126), "centre_r": ("f", 130),
        "centre_a": ("f", 134), "centre_s": ("f", 138), "normal_r": ("f", 142),
        "normal_a": ("f", 146), "normal_s": ("f", 150), "tlhc_r": ("f", 154), "tlhc_a": ("f", 158),
        "tlhc_s": ("f", 162), "trhc_r": ("f", 166), "trhc_a": ("f", 170), "trhc_s": ("f", 174),
        "brhc_r": ("f", 178), "brhc_a": ("f", 182), "brhc_s": ("f", 186),
    }
    MR_IMAGE = {
        "tr_us": ("i", 194), "ti_us": ("i", 198), "te_us": ("i", 202), "te2_us": ("i", 206),
        "echoes": ("h", 210), "echo_number": ("h", 212), "table_delta": ("f", 214),
        "nex": ("f", 218),
        "contiguous": ("h", 222), "heart_rate": ("h", 224), "trigger_delay_us": ("i", 226),
        "sar_average": ("f", 230), "sar_peak": ("f", 234), "sar_monitored": ("h", 238),
        "trigger_window": ("h", 240), "cardiac_repetition_time": ("f", 242),
        "images_per_cardiac_cycle": ("h", 246), "transmit_gain": ("h", 248),
        "receive_gain_1": ("h", 250), "receive_gain_2": ("h", 252), "flip_angle": ("h", 254),
        "minimum_delay": ("i", 256), "cardiac_phases": ("h", 260),
        "swap_phase_frequency": ("h", 262),
        "pause_interval": ("h", 264), "pause_time": ("f", 266), "oblique_plane": ("i", 270),
        "slice_location_fov": ("f", 274), "transmit_frequency_0_1hz": ("i", 278),
        "auto_transmit_frequency": ("i", 282), "auto_transmit_gain": ("h", 286),
        "prescan_r1": ("h", 288), "prescan_r2": ("h", 290), "user_bitmap": ("i", 292),
        "centre_frequency": ("h", 296), "imaging_mode": ("h", 298), "imaging_options": ("i", 300),
        "pulse_sequence": ("h", 304), "pulse_sequence_mode": ("h", 306), "psd_name": ("s", 308, 33),
        "psd_datetime": ("t", 342), "psd_internal_name": ("s", 346, 13), "coil_type": ("h", 360),
        "coil_name": ("s", 362, 17), "surface_coil_type": ("h", 380),
        "surface_coil_extremity": ("h", 382),
        "raw_run_number": ("i", 384), "calibrated_field_strength": ("i", 388),
        "suppression_technique": ("h", 392), "variable_bandwidth_khz": ("f", 394),
        "slices": ("h", 398),
        "graphic_prescription": ("h", 400), "inter_image_delay": ("f", 402),
        "user_variables": ("f", 406, 25), "sat_bitmap": ("h", 552),
        "intensity_correction": ("h", 554), "flow_axis": ("h", 574), "venc": ("h", 576),
        "image_type": ("h", 584), "echo_train_length": ("h", 640), "fractional_echo": ("h", 642),
        "preparatory_pulse": ("h", 644), "cardiac_phase_number": ("h", 646),
        "variable_echo": ("h", 648), "frequency_direction": ("h", 742), "vascular_mode": ("h", 744),
    }
    # Genesis version 2 pixel headers place the blocks at fixed offsets ...
    V2_POINTERS = {"suite": 124, "exam": 240, "series": 1280, "image": 2308}
    # ... and shift these image fields (ITK's VOff table); the others are not read for version 2
    V2_IMAGE = {
        "slice_thickness": 28, "matrix_x": 32, "matrix_y": 34, "display_fov_x": 36,
        "display_fov_y": 40, "acquisition_dim_x": 44, "acquisition_dim_y": 48, "pixel_size_x": 52,
        "pixel_size_y": 56, "plane": 116, "location": 132, "centre_r": 136, "centre_a": 140,
        "centre_s": 144, "normal_r": 146, "normal_a": 152, "normal_s": 156, "tlhc_r": 160,
        "tlhc_a": 164, "tlhc_s": 168, "trhc_r": 172, "trhc_a": 176, "trhc_s": 180, "brhc_r": 184,
        "brhc_a": 188, "brhc_s": 192, "tr_us": 200, "ti_us": 204, "te_us": 208, "te2_us": 212,
        "echoes": 216, "echo_number": 218, "nex": 224, "flip_angle": 260, "psd_name": 320,
        "slices": 416,
    }
    V2_EXAM = {"patient_id": 88, "patient_name": 101, "exam_type": 309}

    # Signa 4.x blocks, in 16-bit words from the start of the file
    STUDY_START, SERIES_START, IMAGE_START = 1536, 2048, 2560
    STUDY_4X = {
        "study_number": ("s", 32, 6), "system_generation_id": ("s", 37, 4),
        "study_date": ("s", 39, 10),
        "study_time": ("s", 47, 8), "patient_name": ("s", 54, 32), "patient_id": ("s", 70, 12),
        "patient_age": ("s", 78, 3), "patient_sex": ("s", 80, 1), "patient_weight_g": ("i", 81),
        "referring_physician": ("s", 83, 32), "radiologist": ("s", 99, 32),
        "operator": ("s", 115, 32),
        "study_description": ("s", 131, 60), "history": ("s", 161, 120),
        "site_name": ("s", 223, 32),
        "requisition_number": ("s", 240, 6),
    }
    SERIES_4X = {
        "series_number": ("s", 31, 3), "series_date": ("s", 37, 10), "series_time": ("s", 45, 8),
        "series_description": ("s", 52, 120), "series_type": ("h", 112), "coil_type": ("h", 113),
        "coil_name": ("s", 114, 16), "contrast": ("s", 122, 32), "plane_type": ("h", 138),
        "plane_name": ("s", 139, 16), "image_mode": ("h", 147), "field_strength_gauss": ("h", 148),
        "pulse_sequence": ("h", 149), "pulse_sequence_subtype": ("h", 150), "fov_mm": ("f", 151),
        "centre_rl": ("f", 153), "centre_ap": ("f", 155), "centre_si": ("f", 157),
        "orientation": ("h", 159), "position": ("h", 160), "longitudinal_reference": ("s", 161, 32),
        "vertical_reference": ("s", 177, 32), "vertical_landmark": ("f", 193),
        "horizontal_landmark": ("f", 195), "table_location": ("f", 197),
        "scan_matrix_x": ("h", 199),
        "scan_matrix_y": ("h", 200), "image_matrix": ("h", 201), "images_allocated": ("h", 202),
        "gating_type": ("h", 203), "pulse_sequence_mode": ("h", 204), "psd_name": ("s", 205, 12),
        "protocol": ("s", 213, 20), "surface_coil_type": ("h", 223),
        "suppression_technique": ("h", 224),
        "vascular_mode": ("h", 236), "flow_axis": ("h", 237),
    }
    IMAGE_4X = {
        "image_date": ("s", 29, 10), "image_time": ("s", 37, 8), "image_number": ("s", 44, 3),
        "series_number": ("s", 46, 3), "location_r_min": ("f", 52), "location_r_max": ("f", 54),
        "location_a_min": ("f", 56), "location_a_max": ("f", 58), "location_s_min": ("f", 60),
        "location_s_max": ("f", 62), "slice_location": ("f", 73), "table_position": ("f", 75),
        "slice_thickness": ("f", 77), "slice_spacing": ("f", 79), "tr_us": ("f", 82),
        "scan_time_us": ("f", 84), "te_us": ("f", 86), "ti_us": ("f", 88), "echoes": ("h", 98),
        "echo_number": ("h", 99), "slices": ("h", 100), "averages": ("h", 101),
        "research_mode": ("h", 102), "psd_file": ("s", 103, 32), "image_shape": ("h", 136),
        "matrix_x": ("h", 137), "matrix_y": ("h", 138), "pixel_size": ("f", 139),
        "compression": ("h", 141), "bits_per_pixel": ("h", 142), "nex": ("f", 146),
        "sar_peak": ("f", 148), "sar_average": ("f", 150), "contiguous_slices": ("h", 153),
        "heart_rate": ("h", 154), "trigger_delay_ms": ("f", 155),
        "transmit_attenuation_0_1db": ("h", 162),
        "receive_attenuation_0_1db": ("h", 163), "field_strength_10ugauss": ("i", 164),
        "psd_name": ("s", 169, 12), "flip_angle": ("h", 175), "centre_r": ("f", 197),
        "centre_a": ("f", 199), "centre_s": ("f", 201), "normal_r": ("f", 203),
        "normal_a": ("f", 205),
        "normal_s": ("f", 207), "tlhc_r": ("f", 209), "tlhc_a": ("f", 211), "tlhc_s": ("f", 213),
        "trhc_r": ("f", 215), "trhc_a": ("f", 217), "trhc_s": ("f", 219), "blhc_r": ("f", 221),
        "blhc_a": ("f", 223), "blhc_s": ("f", 225), "te2_us": ("f", 230),
        "swap_phase_frequency": ("h", 232), "user_variables": ("f", 237, 10),
        "oblique_plane": ("h", 257),
        "contrast_agent": ("s", 259, 10), "transmit_frequency_0_1hz": ("i", 268),
        "receive_frequency_0_1hz": ("i", 270), "image_type": ("h", 289), "venc": ("h", 292),
        "fractional_echo": ("h", 296), "echo_train_length": ("h", 297),
        "cardiac_phase_number": ("h", 300),
        "acquisitions": ("h", 301),
    }

    def __init__(self, path: Path):
        self.path = path
        self.blob = path.read_bytes()
        self.blocks: dict[str, dict[str, Any]] = {}
        self.notes: list[str] = []
        if struct.unpack(">i", self.blob[:4])[0] == self.IMGF:
            self.variant = "Genesis 5.x with pixel header"
            self._genesis_with_pixel_header()
        elif self.blob[7:12] == b"SIGNA":
            self.variant = "Genesis 5.x"
            self._genesis(0, self.SUITE_LENGTH, self.SUITE_LENGTH + self.EXAM_LENGTH,
                          self.SUITE_LENGTH + self.EXAM_LENGTH + self.SERIES_LENGTH, 3)
            self.offset = len(self.blob) - self.matrix[0] * self.matrix[1] * 2
        else:
            self.variant = "Signa 4.x"
            self._signa4()

    # --- Genesis 5.x -----------------------------------------------------------------------------

    def _genesis_with_pixel_header(self) -> None:
        pixel = dict(zip(
            ("magic", "header_length", "width", "height", "depth", "compression", "window", "level",
             "background_shade", "overflow", "underflow", "top_offset", "bottom_offset"),
            struct.unpack(">13i", self.blob[:52]), strict=True,
        ))
        pixel["version"], pixel["checksum"] = struct.unpack(">hH", self.blob[52:56])
        pointers = struct.unpack(">8i", self.blob[124:156])
        self.blocks["pixel"] = pixel
        if pixel["version"] == 2:
            suite, exam, series, image = self.V2_POINTERS.values()
        else:
            suite, exam, series, image = pointers[0], pointers[2], pointers[4], pointers[6]
        self._genesis(suite, exam, series, image, pixel["version"])
        if pixel["compression"] != 1:
            code = pixel["compression"]
            raise ReaderError(f"{self.path.name}: GE compressed pixels (code {code}) are not read")
        if pixel["depth"] != 16:
            raise ReaderError(f"{self.path.name}: {pixel['depth']}-bit GE pixels are not read")
        self.offset = pixel["header_length"]
        self.matrix = (pixel["width"], pixel["height"])

    def _genesis(self, suite: int, exam: int, series: int, image: int, version: int) -> None:
        self.blocks["suite"] = self._fields(self.SUITE, suite)
        self.blocks["exam"] = self._fields(self.EXAM, exam, self.V2_EXAM if version == 2 else {})
        self.blocks["series"] = self._fields(self.SERIES, series)
        self.modality = self.blocks["exam"].get("exam_type", "")
        fields = dict(self.IMAGE)
        if self.modality != "CT":
            fields.update(self.MR_IMAGE)
        if version == 2:
            fields = {k: v for k, v in fields.items() if k in self.V2_IMAGE or v[1] < 26}
            self.notes.append("Genesis version 2 header: only the fields ITK reads are taken")
        shifts = self.V2_IMAGE if version == 2 else {}
        self.blocks["image"] = img = self._fields(fields, image, shifts)
        self.matrix = (img["matrix_x"], img["matrix_y"])
        self.pixel_size = (img["pixel_size_x"], img["pixel_size_y"])
        self.corners = self._corners(img, ("tlhc", "trhc", "brhc"))
        self.thickness, self.gap = img["slice_thickness"], img.get("scan_spacing", 0.0)
        exam, series = self.blocks["exam"], self.blocks["series"]
        self.series_key = (exam.get("exam_number"), series.get("series_number"))
        self.echo = img.get("echo_number", 0) if self.modality != "CT" else 0
        self.rows_are_second_corner_axis = False  # brhc: the row axis runs from trhc to brhc

    @staticmethod
    def _corners(img: dict[str, Any], names: tuple[str, str, str]) -> tuple[np.ndarray, ...]:
        return tuple(np.array([img[f"{c}_r"], img[f"{c}_a"], img[f"{c}_s"]], float) for c in names)

    def _fields(self, table: dict, base: int, shifts: dict[str, int] | None = None) -> dict:
        out: dict[str, Any] = {}
        shifts = shifts or {}
        for name, spec in table.items():
            fmt, offset = spec[0], base + shifts.get(name, spec[1])
            if fmt == "s":
                raw = self.blob[offset : offset + spec[2]]
                out[name] = raw.split(b"\0")[0].decode("latin-1", "replace").strip()
            elif fmt == "t":
                (stamp,) = struct.unpack(">i", self.blob[offset : offset + 4])
                out[name] = self._timestamp(stamp)
            else:
                n = spec[2] if len(spec) == 3 else 1
                raw = self.blob[offset : offset + n * struct.calcsize(fmt)]
                values = struct.unpack(f">{n}{fmt}", raw)
                out[name] = list(values) if len(spec) == 3 else values[0]
        return out

    @staticmethod
    def _timestamp(seconds: int) -> str:
        if seconds <= 0:
            return ""
        return datetime.fromtimestamp(seconds, UTC).strftime("%Y-%m-%dT%H:%M:%S")

    # --- Signa 4.x -------------------------------------------------------------------------------

    def _signa4(self) -> None:
        plane = self.blob[2 * (self.SERIES_START + 139) :][:16]
        if self.blob[2 * self.STUDY_START :][:5] != b"STUDY" or not any(
            p in plane for p in (b"AXIAL", b"CORONAL", b"SAGITTAL", b"OBLIQUE")
        ):
            raise ReaderError(f"{self.path.name}: not a GE Signa image")
        self.blocks["study"] = self._words(self.STUDY_4X, self.STUDY_START)
        self.blocks["series"] = self._words(self.SERIES_4X, self.SERIES_START)
        self.blocks["image"] = img = self._words(self.IMAGE_4X, self.IMAGE_START)
        self.modality = "MR"
        self.matrix = (img["matrix_x"], img["matrix_y"])
        self.pixel_size = (img["pixel_size"], img["pixel_size"])
        self.corners = self._corners(img, ("tlhc", "trhc", "blhc"))
        self.thickness, self.gap = img["slice_thickness"], img["slice_spacing"]
        study, series = self.blocks["study"], self.blocks["series"]
        self.series_key = (study.get("study_number"), series.get("series_number"))
        self.echo = img.get("echo_number", 0)
        self.rows_are_second_corner_axis = True  # blhc: the row axis runs from tlhc to blhc
        self.offset = len(self.blob) - self.matrix[0] * self.matrix[1] * 2

    def _words(self, table: dict, start: int) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, spec in table.items():
            fmt, offset = spec[0], 2 * (start + spec[1])
            if fmt == "s":
                raw = self.blob[offset : offset + spec[2]]
                out[name] = raw.split(b"\0")[0].decode("latin-1", "replace").strip()
            elif fmt == "f":
                n = spec[2] if len(spec) == 3 else 1
                words = struct.unpack(f">{n}I", self.blob[offset : offset + 4 * n])
                values = [self.data_general_float(w) for w in words]
                out[name] = values if n > 1 else values[0]
            elif fmt == "i":
                (out[name],) = struct.unpack(">i", self.blob[offset : offset + 4])
            else:
                (out[name],) = struct.unpack(">h", self.blob[offset : offset + 2])
        return out

    @staticmethod
    def data_general_float(word: int) -> float:
        """A 32-bit Data General float (sign, 7-bit excess-64 exponent of 16, 24-bit mantissa)
        as a Python float, as ITK's MvtSunf converts it."""
        sign = -1.0 if word & 0x80000000 else 1.0
        exponent = (word >> 24) & 0x7F
        mantissa = word & 0x00FFFFFF
        if mantissa == 0:
            return 0.0
        return sign * mantissa * 16.0 ** (exponent - 64) / 2**24

    # --- derived geometry ------------------------------------------------------------------------

    def axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unit vectors (RAS) of the column, row and slice axes; the slice axis is their cross
        product, so the three form a right-handed set."""
        tl, tr, third = self.corners
        ex = tr - tl
        ey = third - tl if self.rows_are_second_corner_axis else third - tr
        ex, ey = ex / np.linalg.norm(ex), ey / np.linalg.norm(ey)
        ez = np.cross(ex, ey)
        return ex, ey, ez

    def first_voxel_centre(self) -> np.ndarray:
        """The top-left corner is the edge of the field of view: half a pixel in along both axes."""
        ex, ey, _ = self.axes()
        return self.corners[0] + 0.5 * self.pixel_size[0] * ex + 0.5 * self.pixel_size[1] * ey

    def pixels(self) -> np.ndarray:
        count = self.matrix[0] * self.matrix[1]
        raw = self.blob[self.offset : self.offset + 2 * count]
        if len(raw) != 2 * count:
            raise ReaderError(f"{self.path.name}: {len(raw)} pixel bytes, expected {2 * count}")
        return np.frombuffer(raw, ">i2").reshape(self.matrix[1], self.matrix[0]).T  # (column, row)


#**************************************************************************************************#
#                                          GeSignaReader                                           #
#**************************************************************************************************#
#                                                                                                  #
# Assembles the files of a folder into one image per series and echo (a single file is read with   #
# the siblings of its series and echo), slices ordered along the slice axis. Image corners are in  #
# patient RAS millimetres already; the top-left corner marks the edge of the field of view, so     #
# the first voxel centre lies half a pixel inside. Checked against ITK's GE readers (whose origin  #
# is that corner) and their MetaImage baselines.                                                   #
#                                                                                                  #
#**************************************************************************************************#
class GeSignaReader(Reader):
    FORMATS = (Format.GE_SIGNA,)
    SOURCE = "GE Signa header"
    POSITIONS = {1: "S", 2: "P", 4: "DL", 8: "DR"}  # Genesis patient position codes
    ENTRIES = {1: "HF", 2: "FF"}  # Genesis patient entry codes
    SEXES = {1: "M", 2: "F"}
    PLANES = {2: "AXIAL", 4: "SAGITTAL", 8: "CORONAL", 16: "OBLIQUE"}

    def read(self, sniff: Sniff) -> list[MRImage]:
        path = sniff.path
        folder = path if path.is_dir() else path.parent
        headers = []
        for f in sorted(folder.iterdir()):
            if not f.is_file() or f.name.startswith("."):
                continue
            try:
                headers.append(GeHeader(f))
            except (ReaderError, struct.error, KeyError, IndexError, ValueError):
                continue
        if not headers:
            raise ReaderError(f"{path.name}: no GE Signa image")
        groups: dict[tuple, list[GeHeader]] = {}
        for h in headers:
            groups.setdefault((h.series_key, h.echo, h.matrix, h.pixel_size), []).append(h)
        if not path.is_dir():
            own = GeHeader(path)
            mine = (own.series_key, own.echo, own.matrix, own.pixel_size)
            groups = {k: v for k, v in groups.items() if k == mine}
        ordered = sorted(groups.items(), key=lambda kv: str(kv[0]))
        return [self._image(sniff, group) for _, group in ordered]

    def _image(self, sniff: Sniff, group: list[GeHeader]) -> MRImage:
        first = group[0]
        ex, ey, ez = first.axes()
        ranked = sorted(group, key=lambda h: float(np.dot(h.first_voxel_centre(), ez)))
        positions = [float(np.dot(h.first_voxel_centre(), ez)) for h in ranked]
        notes = list(first.notes)
        if len(ranked) > 1:
            steps = np.diff(positions)
            step = float(np.mean(steps))
            if not np.allclose(steps, step, atol=1e-2):
                notes.append("uneven slice positions; mean spacing used")
        else:
            step = first.thickness + first.gap
        data = np.stack([h.pixels() for h in ranked], axis=2)
        affine = np.eye(4)
        affine[:3, 0] = ex * first.pixel_size[0]
        affine[:3, 1] = ey * first.pixel_size[1]
        affine[:3, 2] = ez * step
        affine[:3, 3] = ranked[0].first_voxel_centre()
        meta = Meta(dict(first.blocks))
        self._keys(meta, first, len(ranked), step)
        notes.append(f"{first.variant}, {len(ranked)} slice file(s)")
        if first.modality == "CT":
            notes.append("CT image: the MR fields of the image header are not read")
        name = f"exam{first.series_key[0]}_series{first.series_key[1]}"
        if first.blocks["image"].get("echoes", 1) > 1:
            name += f"_echo{first.echo}"
        return MRImage(data, affine, meta, self.provenance(sniff, tuple(notes)), name)

    def _keys(self, meta: Meta, h: GeHeader, nslices: int, step: float) -> None:
        def put(key: str, value: Any, block: str, field: str) -> None:
            if value not in (None, "", []):
                meta.set(key, value, f"{self.SOURCE}: {block} {field}", f"{block}/{field}")

        exam = h.blocks.get("exam") or h.blocks.get("study", {})
        series, image = h.blocks["series"], h.blocks["image"]
        exam_block = "exam" if "exam" in h.blocks else "study"
        put("Modality", h.modality, exam_block, "exam_type")
        put("Manufacturer", "GE", exam_block, "product_id")
        put("PatientName", exam.get("patient_name"), exam_block, "patient_name")
        put("PatientID", exam.get("patient_id"), exam_block, "patient_id")
        put("InstitutionName", exam.get("hospital_name") or exam.get("site_name"), exam_block,
            "hospital_name" if "hospital_name" in exam else "site_name")
        put("StudyDescription", exam.get("exam_description") or exam.get("study_description"),
            exam_block, "exam_description" if "exam_description" in exam else "study_description")
        put("StationName", exam.get("system_id") if exam_block == "exam" else None, exam_block,
            "system_id")
        sex = exam.get("patient_sex")
        put("PatientSex", self.SEXES.get(sex, sex if isinstance(sex, str) else None), exam_block,
            "patient_sex")
        if (weight := exam.get("patient_weight_g")) and weight > 0:
            put("PatientWeight", weight / 1000, exam_block, "patient_weight_g")
        put("SeriesDescription", series.get("series_description"), "series", "series_description")
        number = series.get("series_number")
        if str(number).strip().isdigit():
            put("SeriesNumber", int(number), "series", "series_number")
        put("ProtocolName", series.get("protocol"), "series", "protocol")
        position = self.ENTRIES.get(series.get("patient_entry"), "") + self.POSITIONS.get(
            series.get("patient_position"), ""
        )
        if len(position) > 2:
            put("PatientPosition", position, "series", "patient_entry, patient_position")
        if gauss := exam.get("magnet_strength_gauss"):
            put("FieldStrength", gauss / 10000, exam_block, "magnet_strength_gauss")
        elif gauss := series.get("field_strength_gauss"):
            put("FieldStrength", gauss / 10000, "series", "field_strength_gauss")
        put("SliceThickness", h.thickness, "image", "slice_thickness")
        put("SpacingBetweenSlices", step if nslices > 1 else h.thickness + h.gap, "image",
            "slice positions" if nslices > 1 else "slice_thickness + scan_spacing")
        stamp = image.get("actual_datetime") or image.get("image_datetime")
        put("AcquisitionDateTime", stamp, "image", "actual_datetime")
        if image.get("image_date") and image.get("image_time"):
            stamp = self._signa4_datetime(image["image_date"], image["image_time"])
            put("AcquisitionDateTime", stamp, "image", "image_date, image_time")
        if (plane := image.get("plane") or series.get("plane")) in self.PLANES:
            put("ImageOrientationText", self.PLANES[plane], "image", "plane")
        elif series.get("plane_name"):
            put("ImageOrientationText", series["plane_name"], "series", "plane_name")
        if h.modality == "CT":
            return
        for key, field, per_unit in (
            ("RepetitionTime", "tr_us", 1e6), ("EchoTime", "te_us", 1e6),
            ("InversionTime", "ti_us", 1e6), ("ExcitationFlipAngle", "flip_angle", 1),
            ("NumberOfAverages", "nex", 1), ("EchoTrainLength", "echo_train_length", 1),
            ("SAR", "sar_average", 1), ("AcquisitionDuration", "scan_duration_us", 1e6),
        ):
            value = image.get(field)
            if value not in (None, 0, 0.0):
                put(key, value / per_unit if per_unit != 1 else value, "image", field)
        put("EchoNumber", image.get("echo_number"), "image", "echo_number")
        put("PulseSequenceName", image.get("psd_name") or series.get("psd_name"), "image",
            "psd_name")
        put("InternalPulseSequenceName", image.get("psd_internal_name"), "image",
            "psd_internal_name")
        put("RxCoil", image.get("coil_name") or series.get("coil_name"), "image", "coil_name")
        if (freq := image.get("transmit_frequency_0_1hz")) and freq > 1e6:
            put("SpectrometerFrequency", [freq / 1e7], "image", "transmit_frequency_0_1hz")
        if h.modality == "MR":
            put("ResonantNucleus", ["1H"], exam_block, "exam_type")
        dim_x, dim_y = image.get("acquisition_dim_x"), image.get("acquisition_dim_y")
        if dim_x and dim_y:
            put("FrequencyEncodingSteps", int(dim_x), "image", "acquisition_dim_x")
            put("AcquisitionMatrixPE", int(dim_y), "image", "acquisition_dim_y")
        elif series.get("scan_matrix_x") and series.get("scan_matrix_y"):
            put("FrequencyEncodingSteps", series["scan_matrix_x"], "series", "scan_matrix_x")
            put("AcquisitionMatrixPE", series["scan_matrix_y"], "series", "scan_matrix_y")

    @staticmethod
    def _signa4_datetime(date: str, time: str) -> str:
        try:
            stamp = datetime.strptime(f"{date} {time}", "%d-%b-%y %H:%M:%S")
            return stamp.strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return f"{date} {time}"
