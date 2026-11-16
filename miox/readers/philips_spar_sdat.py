####################################################################################################
#                                       philips_spar_sdat.py                                       #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for Philips SPAR/SDAT pairs. Either file of the pair can be given; the SPAR is   #
#          kept whole as the vendor header.                                                        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import warnings
from ast import literal_eval
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# own
from miox import __version__
from miox.data import MRSData
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                      PhilipsSparSdatReader                                       #
#**************************************************************************************************#
#                                                                                                  #
# Reads one SPAR/SDAT pair. The SPAR is parsed line by line (comment lines may start with a space; #
# values some exports write as a .NET type name instead of numbers, System.Collections..., are     #
# left out with a note). The SDAT holds rows x samples complex VAX single floats, stored in the    #
# left-handed frame: they are conjugated. A single voxel (volume_selection_enable yes) gets the    #
# box's affine: sizes lr, ap, cc, turned by the angulations (-lr, -ap, cc; intrinsic x, y, z) and  #
# centred at the off-centres (-lr, -ap, cc); without it a 10 m voxel. Spectroscopic imaging rows   #
# are put on their grid (dim2 x, dim3 y, dim4 z points, x fastest) with the SI geometry: in-plane  #
# voxels of phase_encoding_fov / dim2 points, slice_thickness, centred at the si_*_off_center      #
# offsets and turned by the si_*_angulation angles, x and y running from right to left and front   #
# to back (equal to the scanner's DICOM of the same scan). Several rows of a single voxel are      #
# DIM_DYN; all-zero indices of a higher dimension are left out. HYPER (scan_id HYPER) splits as    #
# the sequence acquires: the act file's first 32 rows are the short-TE block (35 ms), the rest the #
# HERCULES edit conditions A to D in turn (80 ms); the ref file's water references are every 32nd  #
# row, alternating 80 and 35 ms. Keys and units as spec2nii writes them, with the inversion time   #
# in s and the patient position as a DICOM code.                                                   #
#                                                                                                  #
#**************************************************************************************************#
class PhilipsSparSdatReader(Reader):
    FORMATS = (Format.PHILIPS_SPAR_SDAT,)
    SOURCE = "SPAR"
    PLACEHOLDER = "System.Collections."  # what some exports write instead of a value
    NO_VOXEL = np.diag([1e4, 1e4, 1e4, 1.0])  # a 10 m voxel, as spec2nii
    MS = 1e-3
    HYPER_SHORT_TE = 0.035  # s; HYPER acquires its short-TE block at 35 ms, the edited one at 80 ms
    HYPER_EDITED_TE = 0.08
    HYPER_SHORT_ROWS = 32  # short-TE transients before the edited ones; every 32nd ref row is kept
    HERCULES = ("A", "B", "C", "D")
    HERCULES_EDIT_PULSES = {  # ppm, 20 ms pulses (Oeltzschner et al. 2019)
        "A": {"PulseOffset": [1.9, 4.58], "PulseDuration": 0.02},
        "B": {"PulseOffset": [1.9, 4.18], "PulseDuration": 0.02},
        "C": {"PulseOffset": 4.58, "PulseDuration": 0.02},
        "D": {"PulseOffset": 4.18, "PulseDuration": 0.02},
    }
    POSITIONS = {"head_first": "HF", "feet_first": "FF"}
    ORIENTATIONS = {"supine": "S", "prone": "P"}

    def read(self, sniff: Sniff) -> list[MRSData]:
        spar, sdat = self._pair(sniff)
        header = self.read_spar(spar)
        notes: list[str] = []
        data = self.read_sdat(sdat, int(header["samples"]), int(header["rows"])).conj()
        data, affine, grid_notes = self._geometry(data, header)
        notes += grid_notes
        values = self._values(header, sdat, notes)
        vendor = {key: self.plain(value) for key, value in header.items()}
        name = sdat.stem
        args = (sniff, vendor, notes, affine)
        if str(header.get("scan_id", "")).lower() == "hyper":
            if "_ref" in sdat.stem:
                return self._hyper_ref(data, values, name, *args)
            return self._hyper(data, values, name, *args)
        tags = ("DIM_DYN",) if data.ndim == 5 else ()
        return [self._dataset(data, values, name, tags, *args)]

    # --- files -----------------------------------------------------------------------------------

    @classmethod
    def read_spar(cls, path: Path) -> dict:
        """The SPAR's parameters: ``key : value`` lines, values as Python literals where they are
        one (numbers, quoted text), else the text."""
        params, unwritten = {}, []
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("!") or ":" not in line:
                continue
            key, value = (part.strip() for part in line.split(":", 1))
            if value.startswith(cls.PLACEHOLDER):
                unwritten.append(key)
                continue
            try:
                params[key] = literal_eval(value)
            except (ValueError, SyntaxError):
                params[key] = value or None
        if unwritten:
            warnings.warn(f"{path.name}: the export wrote no value for {', '.join(unwritten)}",
                          stacklevel=2)
        return params

    @staticmethod
    def read_sdat(path: Path, samples: int, rows: int) -> np.ndarray:
        """(samples, rows) complex from VAX F single floats: per float the bytes hold the fraction's
        middle, sign and exponent, then its low and high bytes; value (0.5 + fraction / 2^24) x
        2^(exponent - 128), zero for exponent 0."""
        b = np.frombuffer(path.read_bytes(), dtype=np.uint8).reshape(-1, 4).astype(np.int64)
        sign = np.where(b[:, 1] & 0x80, -1.0, 1.0)
        exponent = ((b[:, 1] & 0x7F) << 1) + ((b[:, 0] & 0x80) >> 7)
        fraction = ((b[:, 0] & 0x7F) << 16) + (b[:, 3] << 8) + b[:, 2]
        values = sign * (0.5 + fraction / 16777216.0) * np.exp2(exponent - 128.0)
        values[exponent == 0] = 0.0
        data = (values[0::2] + 1j * values[1::2]).astype(np.complex64)
        return data.reshape(rows, samples).T

    def _pair(self, sniff: Sniff) -> tuple[Path, Path]:
        files = (sniff.path,) + sniff.companions
        spar = next((f for f in files if f.suffix.lower() == ".spar"), None)
        sdat = next((f for f in files if f.suffix.lower() == ".sdat"), None)
        if spar is None or sdat is None:
            raise ReaderError(f"incomplete SPAR/SDAT pair: {sniff.path.name}")
        return spar, sdat

    # --- geometry --------------------------------------------------------------------------------

    def _geometry(self, data: np.ndarray, header: dict) -> tuple[np.ndarray, np.ndarray, list]:
        """(x, y, z, points[, rows]) and the affine: the spectroscopic imaging grid, else the
        single voxel."""
        grid = self._grid(data, header)
        if grid is not None:
            return grid
        if header.get("volume_selection_enable") == "yes":
            angles = [-header["lr_angulation"], -header["ap_angulation"], header["cc_angulation"]]
            sizes = [header["lr_size"], header["ap_size"], header["cc_size"]]
            centre = [-header["lr_off_center"], -header["ap_off_center"],
                      header["cc_off_center"]]
            affine = self.box(angles, sizes, centre)
        else:
            affine = self.NO_VOXEL
        shape = (1, 1, 1) + data.shape if data.shape[1] > 1 else (1, 1, 1, data.shape[0])
        return data.reshape(shape), affine, []

    @staticmethod
    def box(angles: list[float], sizes: list[float], centre: list[float]) -> np.ndarray:
        """Affine of a box: diagonal sizes turned by intrinsic x, y, z angles (degrees)."""
        affine = np.eye(4)
        rotation = Rotation.from_euler("XYZ", [float(a) for a in angles], degrees=True)
        affine[:3, :3] = rotation.as_matrix() @ np.diag([float(s) for s in sizes])
        affine[:3, 3] = [float(c) for c in centre]
        return affine

    def _grid(self, data: np.ndarray, header: dict) -> tuple[np.ndarray, np.ndarray, list] | None:
        """Spectroscopic imaging rows on their grid, with its geometry."""
        if header.get("phase_encoding_enable") != "yes" or data.shape[1] < 2:
            return None
        sizes = {header.get(f"dim{n}_direction"): int(header.get(f"dim{n}_pnts") or 1)
                 for n in (2, 3, 4)}
        nx, ny, nz = (sizes.get(axis, 1) for axis in ("x", "y", "z"))
        if nx * ny * nz != data.shape[1]:
            return None
        grid = np.transpose(data.T.reshape(nz, ny, nx, -1), (2, 1, 0, 3))
        note = f"{nx * ny * nz} rows put on their {nx} x {ny} x {nz} grid (SPAR dim2-dim4)"
        fields = ("phase_encoding_fov", "slice_thickness", "si_lr_off_angulation",
                  "si_ap_off_angulation", "si_cc_off_angulation", "si_lr_off_center",
                  "si_ap_off_center", "si_cc_off_center")
        if any(not isinstance(header.get(key), (int, float)) for key in fields):
            return grid, np.eye(4), [f"{note}; the SPAR states no imaging geometry: voxel "
                                     "positions unknown"]
        size = float(header["phase_encoding_fov"]) / nx
        angles = [-header["si_lr_off_angulation"], -header["si_ap_off_angulation"],
                  header["si_cc_off_angulation"]]
        centre = [-header["si_lr_off_center"], -header["si_ap_off_center"],
                  header["si_cc_off_center"]]
        affine = self.box(angles, [size, size, float(header["slice_thickness"])], centre)
        affine[:3, :2] *= -1  # x from right to left, y from front to back
        affine[:3, 3] -= affine[:3, :3] @ ((np.array([nx, ny, nz]) - 1) / 2)
        return grid, affine, [note]

    # --- keys ------------------------------------------------------------------------------------

    def _values(self, header: dict, sdat: Path, notes: list[str]) -> dict:
        """Standard keys: (value, SPAR field)."""
        frequency = float(header["synthesizer_frequency"]) / 1e6
        values = {
            "SpectrometerFrequency": ([frequency], "synthesizer_frequency"),
            "ResonantNucleus": ([str(header["nucleus"])], "nucleus"),
            "EchoTime": (float(header["echo_time"]) * self.MS, "echo_time"),
            "RepetitionTime": (float(header["repetition_time"]) / 1e3, "repetition_time"),
            "TxOffset": (float(header["offset_frequency"]) / frequency, "offset_frequency"),
            "Manufacturer": ("Philips", None),
            "kSpace": ([False, False, False], None),
            "ConversionMethod": (f"miox {__version__}", None),
            "ConversionTime": (datetime.now().isoformat(timespec="milliseconds"), None),
            "OriginalFile": ([sdat.name], None),
        }
        inversion = header.get("spectrum_inversion_time")
        if isinstance(inversion, (int, float)) and inversion > 0:
            values["InversionTime"] = (float(inversion) * self.MS, "spectrum_inversion_time")
        for key, field in (("SoftwareVersions", "equipment_sw_verions"),
                           ("ProtocolName", "scan_id"), ("PatientName", "patient_name"),
                           ("PatientDoB", "patient_birth_date")):
            if header.get(field) is not None:
                values[key] = (str(header[field]), field)
        position = self._position(header, notes)
        if position:
            values["PatientPosition"] = position
        return values

    def _position(self, header: dict, notes: list[str]) -> tuple[str, str] | None:
        """The DICOM code of patient_position and patient_orientation (head_first supine: HFS)."""
        entry, side = header.get("patient_position"), header.get("patient_orientation")
        if entry is None or side is None:
            return None
        code = (self.POSITIONS.get(str(entry)), self.ORIENTATIONS.get(str(side)))
        if None in code:
            notes.append(f"patient position '{entry} {side}' has no DICOM code here: kept as text")
            return f"{entry} {side}", "patient_position"
        return "".join(code), "patient_position"

    # --- datasets --------------------------------------------------------------------------------

    def _dataset(self, data, values, name, tags, sniff, vendor, notes, affine) -> MRSData:
        dwelltime = 1.0 / float(vendor["sample_frequency"])
        values = dict(values, DwellTime=(dwelltime, "sample_frequency"))
        notes = list(notes)
        data, tags = self._without_zero_indices(data, tags, notes)
        return self.dataset(sniff, data, dwelltime, values, vendor, name, notes, self.SOURCE,
                            affine, tags)

    @staticmethod
    def _without_zero_indices(data: np.ndarray, tags: tuple, notes: list[str]) -> tuple:
        """Indices of dimension 5 and up that are zero everywhere left out (unless all are), and
        their entries of a per-index header with them."""
        tags = list(tags)
        for axis in range(4, data.ndim):
            others = tuple(a for a in range(data.ndim) if a != axis)
            zero = np.all(data == 0, axis=others)
            if not zero.any() or zero.all():
                continue
            keep = np.flatnonzero(~zero)
            notes.append(f"all-zero indices of dimension {axis + 1} left out: "
                         f"{np.flatnonzero(zero).tolist()}")
            data = np.take(data, keep, axis=axis)
            tag = tags[axis - 4] if axis - 4 < len(tags) else None
            if isinstance(tag, tuple) and tag[2]:
                header = {k: [v[i] for i in keep] for k, v in tag[2].items()}
                tags[axis - 4] = (tag[0], tag[1], header)
        return data, tuple(tags)

    def _hyper(self, data, values, name, *args) -> list[MRSData]:
        """Short-TE block (first 32 rows) and the HERCULES conditions A to D in turn."""
        short, edited = data[..., : self.HYPER_SHORT_ROWS], data[..., self.HYPER_SHORT_ROWS:]
        complete = edited.shape[-1] // 4 * 4
        notes = []
        if complete < edited.shape[-1]:
            notes.append(f"edited block: {edited.shape[-1] - complete} transients of an "
                         "incomplete set of four conditions left out")
        edited = edited[..., :complete]
        edited = edited.reshape(edited.shape[:4] + (complete // 4, 4)).swapaxes(4, 5)
        short_values = dict(values, EchoTime=(self.HYPER_SHORT_TE, None),
                            WaterSuppressed=(True, None))
        edited_values = dict(values, EchoTime=(self.HYPER_EDITED_TE, None),
                             WaterSuppressed=(True, None),
                             EditPulse=(self.HERCULES_EDIT_PULSES, None))
        conditions = ("DIM_EDIT", "HERCULES j-difference editing, four conditions",
                      {"EditCondition": list(self.HERCULES)})
        sniff, vendor, base_notes, affine = args
        args = (sniff, vendor, base_notes + notes, affine)
        return [self._dataset(short, short_values, f"{name}_hyper_short_te", ("DIM_DYN",), *args),
                self._dataset(edited, edited_values, f"{name}_hyper_edited",
                              (conditions, "DIM_DYN"), *args)]

    def _hyper_ref(self, data, values, name, *args) -> list[MRSData]:
        """Water references: every 32nd row, the first at the edited (80 ms) echo time, then
        alternating with the short (35 ms) one."""
        references = data[..., :: self.HYPER_SHORT_ROWS]
        edited, short = references[..., 0::2], references[..., 1::2]
        short_values = dict(values, EchoTime=(self.HYPER_SHORT_TE, None),
                            WaterSuppressed=(False, None))
        edited_values = dict(values, EchoTime=(self.HYPER_EDITED_TE, None),
                             WaterSuppressed=(False, None))
        return [self._dataset(short, short_values, f"{name}_hyper_ref_short_te", ("DIM_DYN",),
                              *args),
                self._dataset(edited, edited_values, f"{name}_hyper_ref_edited", ("DIM_DYN",),
                              *args)]
