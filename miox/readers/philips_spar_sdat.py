####################################################################################################
#                                       philips_spar_sdat.py                                       #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for Philips SPAR/SDAT pairs. Either file of the pair can be given; spec2nii      #
#          converts and parses the SPAR, which is kept whole as the vendor header.                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import contextlib
import dataclasses
import warnings
from ast import literal_eval
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from nifti_mrs.nifti_mrs import NIFTI_MRS
from spec2nii.nifti_orientation import calc_affine
from spec2nii.Philips import philips, philips_data_list
from spec2nii.Philips.philips import read_sdat_spar_pair
from spec2nii.utils import detect_file_encoding

# own
from miox.data import MRSData
from miox.readers.base import ReaderError, Spec2niiReader
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                      PhilipsSparSdatReader                                       #
#**************************************************************************************************#
#                                                                                                  #
# Reads one SPAR/SDAT pair with spec2nii's defaults (dimension tags chosen from the data). The     #
# SPAR is parsed here for spec2nii too: comment lines may start with a space, and values some      #
# exports write as a .NET type name instead of numbers (System.Collections...) are left out, with  #
# a note. spec2nii leaves the rows of spectroscopic imaging as one dimension; here they are put on #
# their grid (dim2 x, dim3 y, dim4 z points, x fastest) with the SI geometry: in-plane voxels of   #
# phase_encoding_fov / dim2 points, slice_thickness, centred at the si_*_off_center offsets and    #
# turned by the si_*_angulation angles as spec2nii turns the single-voxel box, x and y running     #
# from right to left and front to back (equal to the scanner's DICOM of the same scan).            #
#                                                                                                  #
#**************************************************************************************************#
class PhilipsSparSdatReader(Spec2niiReader):
    FORMATS = (Format.PHILIPS_SPAR_SDAT,)
    CONVERTER = "philips"
    REQUIRED_BY_SPEC2NII = {"spectrum_inversion_time": 0}  # 0: no inversion (spec2nii skips it)
    HYPER_SHORT_TE = 0.035  # s; HYPER acquires its short-TE block at 35 ms, the edited one at 80 ms
    HYPER_EDITED_TE = 0.08
    PLACEHOLDER = "System.Collections."  # what some exports write instead of a value
    SOURCES = {
        "SpectrometerFrequency": "synthesizer_frequency",
        "ResonantNucleus": "nucleus",
        "EchoTime": "echo_time",
        "RepetitionTime": "repetition_time",
        "InversionTime": "spectrum_inversion_time",
        "TxOffset": "offset_frequency",
        "SoftwareVersions": "equipment_sw_verions",
        "ProtocolName": "scan_id",
        "PatientPosition": "patient_position",
        "PatientName": "patient_name",
        "PatientDoB": "patient_birth_date",
        "DwellTime": "sample_frequency",
    }

    def convert(self, sniff: Sniff) -> tuple[list[NIFTI_MRS], list[str], dict]:
        spar, sdat = self._pair(sniff)
        header = self.read_spar(spar)
        with self.defaults(header):
            images, names = read_sdat_spar_pair(sdat, spar, None, [None, None, None], None, None)
        return list(images), list(names), {key: self.plain(v) for key, v in header.items()}

    @classmethod
    def read_spar(cls, path: Path) -> dict:
        """The SPAR's parameters, as spec2nii's read_spar parses them."""
        params, unwritten = {}, []
        with open(path, encoding=detect_file_encoding(path)) as f:
            for line in f:
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

    @contextlib.contextmanager
    def defaults(self, header: dict) -> Iterator[None]:
        """spec2nii reads the SPAR as here and sees the keys it reads unconditionally, if this
        SPAR lacks them."""
        missing = {k: v for k, v in self.REQUIRED_BY_SPEC2NII.items() if k not in header}
        for key in missing:
            warnings.warn(f"SPAR has no {key}; read as {missing[key]!r}", stacklevel=2)
        originals = philips.read_spar, philips_data_list.read_spar
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # noted once, when the header was read
            philips.read_spar = philips_data_list.read_spar = lambda path: {
                **missing, **self.read_spar(Path(path))}
            try:
                yield
            finally:
                philips.read_spar, philips_data_list.read_spar = originals

    def correct(self, d: MRSData) -> None:
        """The SPAR inversion time unscaled (ms) and the HYPER short-TE reference set to 80 ms."""
        super().correct(d)
        self._grid(d)
        if d.meta.fields.get("InversionTime") == "spectrum_inversion_time":
            self.fix(d, "InversionTime", d.meta["InversionTime"] * 1e-3, "SPAR value in ms")
        if d.name.endswith("_hyper_ref_short_te"):
            self.fix(d, "EchoTime", self.HYPER_SHORT_TE, "HYPER short-TE reference is 35 ms")

    def _grid(self, d: MRSData) -> None:
        """Spectroscopic imaging rows on their grid, with its geometry."""
        v = d.meta.vendor_fields()
        if v.get("phase_encoding_enable") != "yes" or d.data.ndim != 5:
            return
        sizes = {v.get(f"dim{n}_direction"): int(v.get(f"dim{n}_pnts") or 1) for n in (2, 3, 4)}
        nx, ny, nz = (sizes.get(axis, 1) for axis in ("x", "y", "z"))
        if nx * ny * nz != d.data.shape[4] or nx * ny * nz == 1:
            return
        rows = np.moveaxis(np.asarray(d.data)[0, 0, 0], 0, -1)  # (rows, points)
        d.data = np.transpose(rows.reshape(nz, ny, nx, -1), (2, 1, 0, 3))
        d.dim_tags, d.dim_info = (None, None, None), {}
        geometry = ("phase_encoding_fov", "slice_thickness", "si_lr_off_angulation",
                    "si_ap_off_angulation", "si_cc_off_angulation", "si_lr_off_center",
                    "si_ap_off_center", "si_cc_off_center")
        if any(not isinstance(v.get(key), (int, float)) for key in geometry):
            d.affine = np.diag([1.0, 1.0, 1.0, 1.0])
            d.provenance = dataclasses.replace(d.provenance, notes=d.provenance.notes + (
                f"{nx * ny * nz} rows put on their {nx} x {ny} x {nz} grid (SPAR dim2-dim4); the "
                "SPAR states no imaging geometry: voxel positions unknown",))
            return
        size = float(v["phase_encoding_fov"]) / nx
        angles = [-float(v["si_lr_off_angulation"]), -float(v["si_ap_off_angulation"]),
                  float(v["si_cc_off_angulation"])]
        centre = [-float(v["si_lr_off_center"]), -float(v["si_ap_off_center"]),
                  float(v["si_cc_off_center"])]
        affine = calc_affine(angles, [size, size, float(v["slice_thickness"])], centre)
        affine[:3, :2] *= -1  # x from right to left, y from front to back
        affine[:3, 3] -= affine[:3, :3] @ ((np.array([nx, ny, nz]) - 1) / 2)
        d.affine = affine
        d.provenance = dataclasses.replace(d.provenance, notes=d.provenance.notes + (
            f"{nx * ny * nz} rows put on their {nx} x {ny} x {nz} grid (SPAR dim2-dim4)",))

    def _pair(self, sniff: Sniff) -> tuple[Path, Path]:
        files = (sniff.path,) + sniff.companions
        spar = next((f for f in files if f.suffix.lower() == ".spar"), None)
        sdat = next((f for f in files if f.suffix.lower() == ".sdat"), None)
        if spar is None or sdat is None:
            raise ReaderError(f"incomplete SPAR/SDAT pair: {sniff.path.name}")
        return spar, sdat
