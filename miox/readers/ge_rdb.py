####################################################################################################
#                                            ge_rdb.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 06/10/26                                                                                #
#                                                                                                  #
# Purpose: GE raw-data (rdb) header: a P-file's leading bytes or the pool header a ScanArchive     #
#          carries. The named fields of the revision's layout (ge_rdb_fields), the data            #
#          acquisition table with the corner points of every slice, and the standard keys.         #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

# own
from miox.meta import Meta
from miox.readers import ge_rdb_fields
from miox.readers.base import ReaderError


#**************************************************************************************************#
#                                           GeRdbHeader                                            #
#**************************************************************************************************#
#                                                                                                  #
# The revision (rdbm.h, the first four bytes; little-endian, or big-endian on the SGI consoles     #
# before revision 11), the fields of its layout as read (``hdr``: texts as bytes) and as plain     #
# values by their GE names (``values``: texts decoded to the first NUL), the receive coil count,   #
# the slice table and the geometry of every slice as ge_to_ismrmrd derives it (read direction from #
# the first to the second corner point, phase direction from the first to the third, slice         #
# direction their cross product, the centre between the second and third point; the points are    #
# rotated into LPS for the patient entry and position and shifted so that the first slice's centre #
# equals the image header's centre, which the table's points lack), the patient position as a      #
# DICOM code and the standard keys.                                                                #
#                                                                                                  #
#**************************************************************************************************#
class GeRdbHeader:
    SOURCE = "rdb header"
    HEAD_BYTES = 262144  # longer than any header; the data start is stated in the header
    ROTATIONS = {  # (entry, position): matrix from the scanner frame to LPS, as ge_to_ismrmrd
        ("HF", "S"): ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
        ("HF", "P"): ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ("HF", "DL"): ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
        ("HF", "DR"): ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
        ("FF", "S"): ((1, 0, 0), (0, -1, 0), (0, 0, -1)),
        ("FF", "P"): ((-1, 0, 0), (0, 1, 0), (0, 0, -1)),
        ("FF", "DL"): ((0, -1, 0), (-1, 0, 0), (0, 0, -1)),
        ("FF", "DR"): ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
    }
    PATIENT_ENTRY = {1: "HF", 2: "FF"}
    PATIENT_POSITION = {1: "S", 2: "P", 4: "DL", 8: "DR"}
    PHASE_DIRECTION = {1: "COL", 2: "ROW"}  # rhi_freq_dir 1: frequency encoding along the rows
    SLICE_ENTRY = "hh9fhh"  # pass, slice in pass (from 1), three corner points, transpose, rotate
    SLICE_STRIDES = (44, 48, 52, 56, 64, 72, 80, 96)  # entry sizes tried for the slice table
    TEXT_KEYS = (  # standard key: field, stored when the text is not empty
        ("PatientName", "rhe_patname"), ("PatientID", "rhe_patid"),
        ("PatientDoB", "rhe_dateofbirth"), ("StudyDescription", "rhe_ex_desc"),
        ("InstitutionName", "rhe_hospname"), ("StationName", "rhe_ex_sysid"),
        ("DeviceSerialNumber", "rhe_uniq_sys_id"), ("SoftwareVersions", "rhe_ex_verscre"),
        ("SeriesDescription", "rhs_se_desc"), ("ProtocolName", "rhs_prtcl"),
        ("PulseSequenceName", "rhi_psdname"), ("ReceiveCoilName", "rhi_cname"),
    )
    NUMBER_KEYS = (  # standard key: field, divisor to the key's unit, whether zero means unset
        ("SeriesNumber", "rhs_se_no", 1, False), ("PatientAge", "rhe_patage", 1, True),
        ("PatientWeight", "rhe_patweight", 1000, True),
        ("ExcitationFlipAngle", "rhi_mr_flip", 1, False), ("NumberOfAverages", "rhi_nex", 1, True),
        ("SliceThickness", "rhi_slthick", 1, True), ("ScanDuration", "rhi_sctime", 1e6, True),
        ("EchoTime", "rhi_te", 1e6, False), ("RepetitionTime", "rhi_tr", 1e6, False),
        ("InversionTime", "rhi_ti", 1e6, True), ("FieldStrength", "rhe_magstrength", 1e4, True),
        ("SpectrometerFrequency", "rhr_rh_ps_mps_freq", 1e7, True),
    )
    INTEGER_KEYS = ("SeriesNumber", "PatientAge", "NumberOfAverages")  # whole numbers as int

    def __init__(self, raw: bytes):
        self.raw = raw
        self.version, self.endian = self.revision(raw)
        self.prefix = ">" if self.endian == "big" else "<"
        self.hdr = SimpleNamespace(**self._fields())
        self.values = {"version": self.version}
        self.values.update((name, self.plain(value)) for name, value in vars(self.hdr).items())

    @classmethod
    def from_file(cls, path: Path) -> GeRdbHeader:
        with open(path, "rb") as fh:
            return cls(fh.read(cls.HEAD_BYTES))

    # --- fields ----------------------------------------------------------------------------------

    @staticmethod
    def revision(raw: bytes) -> tuple[float, str]:
        """(rdbm.h revision, byte order): the first four bytes as a little-endian float, or as a
        big-endian one when that gives a revision below 11."""
        for prefix, endian in (("<", "little"), (">", "big")):
            version = round(struct.unpack_from(prefix + "f", raw, 0)[0], 3)
            if ge_rdb_fields.layout(version) and (endian == "little" or version < 11):
                return version, endian
        (little,) = struct.unpack_from("<f", raw, 0)
        raise ReaderError(f"no layout for the rdb header revision {little:g}")

    def _fields(self) -> dict[str, Any]:
        fields = {}
        for name, offset, fmt in ge_rdb_fields.layout(self.version):
            if offset + struct.calcsize(fmt) <= len(self.raw):
                (value,) = struct.unpack_from(self.prefix + fmt, self.raw, offset)
                fields[name] = value.split(b"\0")[0] if isinstance(value, bytes) else value
        return fields

    @staticmethod
    def plain(value: Any) -> Any:
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                return value.decode("latin-1")
        return value

    @property
    def coils(self) -> int:
        """Receive channels from the DAB table: one entry per board, start and stop receiver."""
        count = 0
        for i in range(4):
            start = self.values.get(f"rhr_rh_dab[{i}]_start_rcv", 0)
            stop = self.values.get(f"rhr_rh_dab[{i}]_stop_rcv", 0)
            if start or stop:
                count += stop - start + 1
        return count or 1

    @property
    def data_offset(self) -> int | None:
        return self.values.get("rhr_rdb_hdr_off_data")

    # --- geometry --------------------------------------------------------------------------------

    def slice_table(self) -> list[dict]:
        """The data acquisition table in its (spatial) order: pass number, slice in pass (the
        acquisition number of the slice, from 1), the three corner points (scanner frame, mm) and
        the transpose and rotate codes of every slice; empty when the header does not locate the
        table or no entry size fits."""
        base = self.values.get("rhr_rdb_hdr_off_data_acq_tab") or 0
        nslices = self.values.get("rhr_rh_nslices") or 0
        if base <= 0 or nslices < 1 or base >= len(self.raw):
            return []
        for stride in self.SLICE_STRIDES:
            entries = self._slice_entries(base, stride, nslices)
            if entries is not None:
                return entries
        return []

    def _slice_entries(self, base: int, stride: int, nslices: int) -> list[dict] | None:
        fmt = self.prefix + self.SLICE_ENTRY
        passes = self.values.get("rhr_rh_npasses") or 1
        entries, seen, centres = [], set(), set()
        for k in range(nslices):
            offset = base + k * stride
            if offset + struct.calcsize(fmt) > len(self.raw):
                return None
            values = struct.unpack_from(fmt, self.raw, offset)
            pass_number, slice_in_pass, points = values[0], values[1], values[2:11]
            if not (0 <= pass_number < passes and 1 <= slice_in_pass <= nslices):
                return None
            if (pass_number, slice_in_pass) in seen:
                return None
            corners = np.array(points, dtype=float).reshape(3, 3)
            if not np.isfinite(corners).all():
                return None
            edges = corners[1] - corners[0], corners[2] - corners[0]
            if not all(np.linalg.norm(edge) for edge in edges):
                return None
            seen.add((pass_number, slice_in_pass))
            centres.add(tuple(np.round((corners[1] + corners[2]) / 2, 3)))
            entries.append(
                {
                    "pass_number": pass_number, "slice_in_pass": slice_in_pass,
                    "corners": corners.tolist(), "transpose": values[11], "rotate": values[12],
                }
            )
        if len(centres) < nslices:
            return None
        return entries

    def geometric_slices(self) -> dict[tuple[int, int], int]:
        """(pass number, acquired slice index from 0): index of the slice in the table, which
        lists the slices in spatial order."""
        return {
            (e["pass_number"], e["slice_in_pass"] - 1): k for k, e in enumerate(self.slice_table())
        }

    def patient_codes(self) -> tuple[str, str] | None:
        entry = self.PATIENT_ENTRY.get(self.values.get("rhs_entry"))
        position = self.PATIENT_POSITION.get(self.values.get("rhs_position"))
        return None if entry is None or position is None else (entry, position)

    def patient_position(self) -> str | None:
        codes = self.patient_codes()
        return None if codes is None else codes[0] + codes[1]

    def geometry(self) -> list[dict] | None:
        """Per slice of the table: centre (LPS, mm) and the read, phase and slice direction
        cosines."""
        table = self.slice_table()
        if not table:
            return None
        rotation = np.array(self.ROTATIONS.get(self.patient_codes(), self.ROTATIONS["HF", "S"]))
        rotated = [np.array(entry["corners"]) @ rotation.T for entry in table]
        v = self.values
        centre = np.array([-v["rhi_ctr_R"], -v["rhi_ctr_A"], v["rhi_ctr_S"]])
        shift = centre - (rotated[0][1] + rotated[0][2]) / 2
        slices = []
        for points in rotated:
            read = self._unit(points[1] - points[0], (1.0, 0.0, 0.0))
            phase = self._unit(points[2] - points[0], (0.0, 1.0, 0.0))
            normal = self._unit(np.cross(read, phase), (0.0, 0.0, 1.0))
            position = (points[1] + points[2]) / 2 + shift
            slices.append(
                {
                    "position": position.tolist(), "read_dir": read.tolist(),
                    "phase_dir": phase.tolist(), "slice_dir": normal.tolist(),
                }
            )
        return slices

    @staticmethod
    def _unit(vector: np.ndarray, fallback: tuple[float, float, float]) -> np.ndarray:
        length = np.linalg.norm(vector)
        return vector / length if length else np.array(fallback)

    # --- standard keys ---------------------------------------------------------------------------

    def meta(self) -> Meta:
        v = self.values
        meta = Meta(dict(v))
        for key, field in self.TEXT_KEYS:
            text = v.get(field)
            if isinstance(text, str) and text.strip():
                self._set(meta, key, text.strip(), field)
        for key, field, divisor, zero_unset in self.NUMBER_KEYS:
            value = v.get(field)
            if not isinstance(value, (int, float)) or (zero_unset and value == 0):
                continue
            value = value / divisor if divisor != 1 else value
            if key in self.INTEGER_KEYS and isinstance(value, float) and value.is_integer():
                value = int(value)
            self._set(meta, key, value, field)
        if "rhe_ex_no" in v:
            self._set(meta, "StudyID", str(v["rhe_ex_no"]), "rhe_ex_no")
        if "GE" in str(v.get("rhr_rh_logo", "")):
            self._set(meta, "Manufacturer", "GE", "rhr_rh_logo")
        position = self.patient_position()
        if position:
            self._set(meta, "PatientPosition", position, "rhs_entry, rhs_position")
        phase = self.PHASE_DIRECTION.get(v.get("rhi_freq_dir"))
        if phase:
            self._set(meta, "InPlanePhaseEncodingDirectionDICOM", phase, "rhi_freq_dir")
        thickness, spacing = v.get("rhi_slthick"), v.get("rhi_scanspacing")
        if isinstance(thickness, float) and thickness > 0 and isinstance(spacing, float):
            meta.set(
                "SpacingBetweenSlices", thickness + spacing,
                f"{self.SOURCE}: rhi_slthick + rhi_scanspacing", "rhi_slthick, rhi_scanspacing",
            )
        when = self._scan_datetime()
        if when:
            self._set(meta, "AcquisitionDateTime", when, "rhr_rh_scan_date, rhr_rh_scan_time")
        meta.set("NumberOfReceiveChannels", self.coils, f"{self.SOURCE}: rhr_rh_dab", "rhr_rh_dab")
        samples, frames = v.get("rhr_rh_frame_size"), v.get("rhr_rh_nframes")
        if samples:
            self._set(meta, "NumberOfPoints", int(samples), "rhr_rh_frame_size")
        if samples and frames:
            meta.set(
                "EncodedMatrixSize", [int(samples), int(frames), 1],
                f"{self.SOURCE}: rhr_rh_frame_size, rhr_rh_nframes; the slices in the counter",
                "rhr_rh_frame_size, rhr_rh_nframes",
            )
        fov = [v.get("rhi_dfov"), v.get("rhi_dfov_rect"), thickness]
        if all(isinstance(x, float) and x > 0 for x in fov):
            fields = "rhi_dfov, rhi_dfov_rect, rhi_slthick"
            self._set(meta, "EncodedFieldOfView", fov, fields)
            self._set(meta, "ReconFieldOfView", fov, fields)
        recon = [v.get("rhr_rh_rc_xres"), v.get("rhr_rh_rc_yres")]
        if all(isinstance(x, int) and x > 0 for x in recon):
            recon.append(v.get("rhr_rh_rc_zres") or 1)
            fields = "rhr_rh_rc_xres, rhr_rh_rc_yres, rhr_rh_rc_zres"
            self._set(meta, "ReconMatrixSize", recon, fields)
        limits = {}
        for counter, field in (
            ("kspace_encoding_step_1", "rhr_rh_nframes"), ("slice", "rhr_rh_nslices"),
            ("contrast", "rhr_rh_nechoes"),
        ):
            count = v.get(field)
            if isinstance(count, int) and count > 0:
                limits[counter] = {"minimum": 0, "maximum": count - 1, "center": count // 2}
        if limits:
            meta.set(
                "EncodingLimits", limits,
                f"{self.SOURCE}: frame, slice and echo counts; the centre is half the count "
                "rounded down, as ge_to_ismrmrd sets it",
                "rhr_rh_nframes, rhr_rh_nslices, rhr_rh_nechoes",
            )
        return meta

    def _set(self, meta: Meta, key: str, value: Any, field: str) -> None:
        meta.set(key, value, f"{self.SOURCE}: {field}", field)

    def _scan_datetime(self) -> str | None:
        """Scan date "mm/dd/yyy" (years since 1900) and time "hh:mm" as ISO 8601 to the minute."""
        date, time = self.values.get("rhr_rh_scan_date"), self.values.get("rhr_rh_scan_time")
        try:
            month, day, year = (int(part) for part in str(date).split("/"))
            hour, minute = (int(part) for part in str(time).split(":")[:2])
        except (ValueError, AttributeError):
            return None
        return f"{1900 + year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}"
