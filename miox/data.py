####################################################################################################
#                                             data.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: The objects every reader returns: spectroscopy (complex FIDs in NIfTI-MRS dimension     #
#          order, dwell time, geometry, tagged higher dimensions), images, raw k-space readouts,   #
#          basis sets, priors, fit results, pulse sequences, noise covariances, physiological      #
#          recordings and measurement protocols, each with metadata and the provenance of the      #
#          read.                                                                                   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from mrs_tools.constants import GYRO_MAG_RATIO
from nifti_mrs.nifti_mrs import NIFTI_MRS

# own
from miox.meta.meta import Meta
from miox.sniff import Format


#**************************************************************************************************#
#                                            Provenance                                            #
#**************************************************************************************************#
#                                                                                                  #
# How a dataset was read: the input files, their format, the reader and the package versions.      #
#                                                                                                  #
#**************************************************************************************************#
@dataclass(frozen=True)
class Provenance:
    paths: tuple[Path, ...]
    format: Format
    reader: str
    versions: dict[str, str]
    notes: tuple[str, ...] = ()


#**************************************************************************************************#
#                                             MRSData                                              #
#**************************************************************************************************#
#                                                                                                  #
# One spectroscopy dataset. ``data`` holds the complex FIDs as (x, y, z, t) plus up to three       #
# tagged dimensions (coils, dynamics, edit conditions, ...), following the NIfTI-MRS dimension and #
# conjugation conventions. ``dim_tags`` names dimensions 5 to 7, ``dim_info`` holds their          #
# description and per-index header values. A file may state no frequency, nucleus or dwell time:   #
# ``missing`` names those gaps and ``supply`` fills them from values given by the caller.          #
#                                                                                                  #
#**************************************************************************************************#
class MRSData:
    DIM_KEY = re.compile(r"^dim_([5-7])(?:_(info|header))?$")  # dimension keys of the header
    REQUIRED = ("SpectrometerFrequency", "ResonantNucleus", "DwellTime")  # NIfTI-MRS needs these
    LIST_KEYS = ("SpectrometerFrequency", "ResonantNucleus")  # one value per nucleus

    def __init__(
        self,
        data: np.ndarray,
        dwelltime: float | None,
        affine: np.ndarray,
        meta: Meta,
        provenance: Provenance,
        dim_tags: tuple[str | None, ...] = (None, None, None),
        dim_info: dict[int, dict] | None = None,
        name: str = "",
    ):
        self.data = data
        self.dwelltime = dwelltime
        self.affine = affine
        self.meta = meta
        self.provenance = provenance
        self.dim_tags = tuple(dim_tags)
        self.dim_info = dim_info if dim_info is not None else {}
        self.name = name

    @classmethod
    def from_nifti_mrs(
        cls,
        image: NIFTI_MRS,
        provenance: Provenance,
        source: str,
        vendor: dict | None = None,
        name: str = "",
    ) -> MRSData:
        """Build from an in-memory NIfTI-MRS image; ``source`` names where its header came from."""
        meta = Meta(vendor)
        dim_info = {}
        for key, value in image.hdr_ext.to_dict().items():
            match = cls.DIM_KEY.match(key)
            if match is None:
                meta.set(key, value, source)
            elif match[2]:
                dim_info.setdefault(int(match[1]), {})[match[2]] = value
        dwelltime = float(image.dwelltime)
        meta.set("DwellTime", dwelltime, source)
        meta.set("NumberOfPoints", int(image.shape[3]), "data shape")
        if "SpectralWidth" not in meta:
            meta.set("SpectralWidth", 1 / dwelltime, "derived: 1 / DwellTime")
        field = float(image.field_strength)
        if not np.isnan(field):
            meta.set(
                "FieldStrength", field, "derived: SpectrometerFrequency / gyromagnetic ratio"
            )
        return cls(
            image.image[:],  # as stored; ``image[:]`` would conjugate to FSL-MRS's convention
            dwelltime,
            image.getAffine("voxel", "world"),
            meta,
            provenance,
            dim_tags=tuple(image.dim_tags),
            dim_info=dim_info,
            name=name,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def dims(self) -> tuple[str, ...]:
        """Names of the data dimensions: x, y, z, t, then the tags of dimensions 5 to 7."""
        return ("x", "y", "z", "t") + tuple(str(t) for t in self.dim_tags[: self.data.ndim - 4])

    @property
    def spectrometer_frequency(self) -> float | None:
        """MHz; None when not stated."""
        value = self.meta.get("SpectrometerFrequency")
        return None if value is None else float(value[0])

    @property
    def nucleus(self) -> str | None:
        value = self.meta.get("ResonantNucleus")
        return None if value is None else value[0]

    @property
    def spectral_width(self) -> float | None:
        """Hz; None when the dwell time is not stated."""
        return None if self.dwelltime is None else 1 / self.dwelltime

    @property
    def n_points(self) -> int:
        return self.data.shape[3]

    @property
    def time_axis(self) -> np.ndarray | None:
        """Seconds from the first point; None when the dwell time is not stated."""
        return None if self.dwelltime is None else np.arange(self.n_points) * self.dwelltime

    @property
    def missing(self) -> tuple[str, ...]:
        """The required keys (frequency, nucleus, dwell time) the file did not state."""
        return tuple(key for key in self.REQUIRED if key not in self.meta)

    def supply(self, values: dict[str, Any], source: str = "given") -> None:
        """Fill the standard keys in ``values`` that are not stated; stated values stay.

        The dwell time may come as DwellTime or SpectralWidth; the other one and FieldStrength
        are derived.
        """
        for key, value in values.items():
            if key in self.meta:
                continue
            if key in self.LIST_KEYS and not isinstance(value, list):
                value = [value]
            self.meta.set(key, value, source)
        if self.dwelltime is None and "DwellTime" not in self.meta and "SpectralWidth" in self.meta:
            dwelltime = 1 / float(self.meta["SpectralWidth"])
            self.meta.set("DwellTime", dwelltime, "derived: 1 / SpectralWidth")
        if self.dwelltime is None and "DwellTime" in self.meta:
            self.dwelltime = float(self.meta["DwellTime"])
            if "SpectralWidth" not in self.meta:
                self.meta.set("SpectralWidth", 1 / self.dwelltime, "derived: 1 / DwellTime")
        frequency, nucleus = self.spectrometer_frequency, self.nucleus
        if "FieldStrength" not in self.meta and frequency is not None and nucleus in GYRO_MAG_RATIO:
            field = frequency / GYRO_MAG_RATIO[nucleus]
            self.meta.set(
                "FieldStrength", field, "derived: SpectrometerFrequency / gyromagnetic ratio"
            )

    def acquisition(self) -> str:
        """Nucleus, frequency and dwell time in words, "unknown" where not stated."""
        nucleus = self.nucleus or "unknown nucleus"
        frequency = self.spectrometer_frequency
        frequency = "unknown frequency" if frequency is None else f"{frequency:.4f} MHz"
        dwell = self.dwelltime
        dwell = "unknown dwell time" if dwell is None else f"dwell {dwell * 1e3:.4g} ms"
        return f"{nucleus} at {frequency}, {dwell}"

    def to_dict(self) -> dict:
        """JSON-ready summary: everything except the data array and the vendor header itself."""
        return {
            "name": self.name,
            "format": self.provenance.format,
            "reader": self.provenance.reader,
            "paths": [str(p) for p in self.provenance.paths],
            "shape": list(self.shape),
            "dims": list(self.dims),
            "dwelltime": self.dwelltime,
            "affine": self.affine.tolist(),
            "dim_info": {str(k): v for k, v in self.dim_info.items()},
            "meta": dict(self.meta),
            "source": dict(self.meta.source),
            "vendor_fields": len(self.meta.vendor_fields()),
            "notes": list(self.provenance.notes),
            "versions": self.provenance.versions,
        }

    def __repr__(self) -> str:
        name = f" {self.name}" if self.name else ""
        dims = " ".join(f"{d}={n}" for d, n in zip(self.dims, self.shape, strict=True))
        return f"<MRSData{name} {self.provenance.format} {dims}, {self.acquisition()}>"


#**************************************************************************************************#
#                                             BasisSet                                             #
#**************************************************************************************************#
#                                                                                                  #
# A basis set for fitting: one FID per metabolite (or macromolecule) along dimension 5, the names  #
# in that dimension's header under "Metabolite". Edited sub-spectra, when present, follow along a  #
# further tagged dimension.                                                                        #
#                                                                                                  #
#**************************************************************************************************#
class BasisSet(MRSData):
    @property
    def metabolites(self) -> list[str]:
        return list(self.dim_info.get(5, {}).get("header", {}).get("Metabolite", []))

    def fid(self, metabolite: str) -> np.ndarray:
        """The FID(s) of one metabolite: time first, then any further dimensions."""
        index = self.metabolites.index(metabolite)
        return self.data[0, 0, 0, :, index] if self.data.ndim > 4 else self.data[0, 0, 0, :]

    def __repr__(self) -> str:
        return (
            f"<BasisSet {self.name} {self.provenance.format}: {len(self.metabolites)} metabolites, "
            f"{self.n_points} points, {self.acquisition()}>"
        )



#**************************************************************************************************#
#                                             PriorSet                                             #
#**************************************************************************************************#
#                                                                                                  #
# Prior knowledge for fitting, as peak lists instead of sampled signals: one dict per line, with   #
# at least "metabolite" and "ppm", and what the format adds ("area" or "amplitude", "phase" in     #
# degrees, "linewidth" in Hz, bounds, loop indices). Values a format gives as expressions linking  #
# lines (e.g. "BATP/2") stay strings.                                                              #
#                                                                                                  #
#**************************************************************************************************#
class PriorSet:
    def __init__(self, lines: list[dict], meta: Meta, provenance: Provenance, name: str = ""):
        self.lines = lines
        self.meta = meta
        self.provenance = provenance
        self.name = name

    @property
    def metabolites(self) -> list[str]:
        return list(dict.fromkeys(line["metabolite"] for line in self.lines))

    def lines_of(self, metabolite: str) -> list[dict]:
        return [line for line in self.lines if line["metabolite"] == metabolite]

    def to_dict(self) -> dict:
        """JSON-ready summary: everything except the vendor header itself."""
        return {
            "name": self.name,
            "format": self.provenance.format,
            "reader": self.provenance.reader,
            "paths": [str(p) for p in self.provenance.paths],
            "metabolites": self.metabolites,
            "lines": self.lines,
            "meta": dict(self.meta),
            "source": dict(self.meta.source),
            "vendor_fields": len(self.meta.vendor_fields()),
            "notes": list(self.provenance.notes),
            "versions": self.provenance.versions,
        }

    def __repr__(self) -> str:
        return (
            f"<PriorSet {self.name} {self.provenance.format}: {len(self.metabolites)} metabolites, "
            f"{len(self.lines)} lines>"
        )


#**************************************************************************************************#
#                                            FitResult                                             #
#**************************************************************************************************#
#                                                                                                  #
# The result of fitting one spectrum (one voxel): a row per metabolite with "concentration" and    #
# "sd_percent" (the tool's units; CRLB or %SD) when the format gives them and every other column   #
# under the format's own name; fit diagnostics under the format's names; and, when the format      #
# holds them, curves on a ppm axis ("ppm", "data", "fit", "baseline" and one per metabolite).      #
#                                                                                                  #
#**************************************************************************************************#
class FitResult:
    def __init__(
        self,
        rows: list[dict],
        diagnostics: dict,
        curves: dict[str, np.ndarray],
        meta: Meta,
        provenance: Provenance,
        name: str = "",
        voxel: tuple[int, ...] | None = None,
    ):
        self.rows = rows
        self.diagnostics = diagnostics
        self.curves = curves
        self.meta = meta
        self.provenance = provenance
        self.name = name
        self.voxel = voxel

    @property
    def metabolites(self) -> list[str]:
        return [row["metabolite"] for row in self.rows]

    def row(self, metabolite: str) -> dict:
        return self.rows[self.metabolites.index(metabolite)]

    def to_dict(self) -> dict:
        """JSON-ready summary: everything except the curves and the vendor header itself."""
        return {
            "name": self.name,
            "format": self.provenance.format,
            "reader": self.provenance.reader,
            "paths": [str(p) for p in self.provenance.paths],
            "voxel": self.voxel,
            "rows": self.rows,
            "diagnostics": self.diagnostics,
            "curves": {name: len(values) for name, values in self.curves.items()},
            "meta": dict(self.meta),
            "source": dict(self.meta.source),
            "vendor_fields": len(self.meta.vendor_fields()),
            "notes": list(self.provenance.notes),
            "versions": self.provenance.versions,
        }

    def __repr__(self) -> str:
        voxel = f" voxel {self.voxel}" if self.voxel else ""
        curves = f", curves {list(self.curves)[:4]}..." if self.curves else ""
        return (
            f"<FitResult {self.name}{voxel} {self.provenance.format}: "
            f"{len(self.rows)} metabolites{curves}>"
        )


#**************************************************************************************************#
#                                             MRImage                                              #
#**************************************************************************************************#
#                                                                                                  #
# One MR image (volume or series): ``data`` indexed (i, j, k, further dimensions) as the file      #
# stores the voxels, ``affine`` mapping (i, j, k) to RAS+ millimetres as in NIfTI. ``dims`` names  #
# the dimensions after the third (e.g. "time", "echo").                                            #
#                                                                                                  #
#**************************************************************************************************#
class MRImage:
    def __init__(
        self,
        data: np.ndarray,
        affine: np.ndarray,
        meta: Meta,
        provenance: Provenance,
        name: str = "",
        dims: tuple[str, ...] = (),
    ):
        self.data = data
        self.affine = affine
        self.meta = meta
        self.provenance = provenance
        self.name = name
        names = ("i", "j", "k") + tuple(dims)
        names += tuple(f"dim{n}" for n in range(len(names), data.ndim))
        self.dims = names[: data.ndim]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def voxel_sizes(self) -> tuple[float, float, float]:
        """mm, the lengths of the affine's first three columns."""
        return tuple(float(v) for v in np.linalg.norm(self.affine[:3, :3], axis=0))

    def to_dict(self) -> dict:
        """JSON-ready summary: everything except the data array and the vendor header itself."""
        return {
            "name": self.name,
            "format": self.provenance.format,
            "reader": self.provenance.reader,
            "paths": [str(p) for p in self.provenance.paths],
            "shape": list(self.shape),
            "dims": list(self.dims),
            "dtype": str(self.data.dtype),
            "affine": self.affine.tolist(),
            "meta": dict(self.meta),
            "source": dict(self.meta.source),
            "vendor_fields": len(self.meta.vendor_fields()),
            "notes": list(self.provenance.notes),
            "versions": self.provenance.versions,
        }

    def __repr__(self) -> str:
        name = f" {self.name}" if self.name else ""
        sizes = " x ".join(f"{v:.3g}" for v in self.voxel_sizes)
        kind = f"{self.provenance.format} {self.shape} {self.data.dtype}"
        return f"<MRImage{name} {kind}, {sizes} mm>"


#**************************************************************************************************#
#                                             RawData                                              #
#**************************************************************************************************#
#                                                                                                  #
# Raw k-space as acquired: ``data`` holds the readouts as (readouts, coils, samples) complex       #
# values, ``headers`` one row per readout (a structured array: encoding counters, flags, time      #
# stamps, position and directions of the readout, ...), ``trajectory`` the k-space coordinates of  #
# every sample as (readouts, samples, dimensions) when the file states them. Readouts of unequal   #
# length are padded with zeros to the longest; ``samples`` gives each one's own length.            #
# ``encoding`` holds the file's encoding description (matrix sizes, fields of view in mm, limits   #
# of the counters, trajectory type), ``waveforms`` physiological or other recordings stored with   #
# the readouts (one dict per waveform: header fields and the samples), ``flag_names`` the meaning  #
# of the bits of the readouts' ``flags`` field (bit number from 1: name). Nothing is               #
# reconstructed.                                                                                   #
#                                                                                                  #
#**************************************************************************************************#
class RawData:
    HEADER = np.dtype(  # the ISMRMRD AcquisitionHeader layout of ``headers``
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

    def __init__(
        self,
        data: np.ndarray,
        headers: np.ndarray,
        meta: Meta,
        provenance: Provenance,
        name: str = "",
        trajectory: np.ndarray | None = None,
        samples: np.ndarray | None = None,
        encoding: dict | None = None,
        waveforms: list[dict] | None = None,
        flag_names: dict[int, str] | None = None,
    ):
        self.data = data
        self.headers = headers
        self.meta = meta
        self.provenance = provenance
        self.name = name
        self.trajectory = trajectory
        self.samples = samples if samples is not None else np.full(data.shape[0], data.shape[-1])
        self.encoding = encoding if encoding is not None else {}
        self.waveforms = waveforms if waveforms is not None else []
        self.flag_names = flag_names if flag_names is not None else {}

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def n_readouts(self) -> int:
        return self.data.shape[0]

    @property
    def n_coils(self) -> int:
        return self.data.shape[1]

    def counters(self, name: str) -> np.ndarray:
        """The encoding counter ``name`` (e.g. "kspace_encode_step_1", "slice") of every readout."""
        return np.asarray(self.headers["idx"][name])

    def flagged(self, name: str) -> np.ndarray:
        """Whether each readout carries the flag ``name`` (a key of ``flag_names``)."""
        bit = next(b for b, n in self.flag_names.items() if n == name)
        return (np.asarray(self.headers["flags"]).astype(np.uint64) >> np.uint64(bit - 1)) & 1 == 1

    def flags_of(self, index: int) -> list[str]:
        """The names of the flags set on readout ``index``."""
        value = int(self.headers["flags"][index])
        return [name for bit, name in self.flag_names.items() if value >> (bit - 1) & 1]

    def to_dict(self) -> dict:
        """JSON-ready summary: everything except the data, headers and the vendor header itself."""
        return {
            "name": self.name,
            "format": self.provenance.format,
            "reader": self.provenance.reader,
            "paths": [str(p) for p in self.provenance.paths],
            "shape": list(self.shape),
            "dtype": str(self.data.dtype),
            "trajectory": None if self.trajectory is None else list(self.trajectory.shape),
            "encoding": self.encoding,
            "waveforms": len(self.waveforms),
            "meta": dict(self.meta),
            "source": dict(self.meta.source),
            "vendor_fields": len(self.meta.vendor_fields()),
            "notes": list(self.provenance.notes),
            "versions": self.provenance.versions,
        }

    def __repr__(self) -> str:
        name = f" {self.name}" if self.name else ""
        trajectory = ", with trajectory" if self.trajectory is not None else ""
        return (
            f"<RawData{name} {self.provenance.format}: {self.n_readouts} readouts x "
            f"{self.n_coils} coils x {self.data.shape[-1]} samples{trajectory}>"
        )


#**************************************************************************************************#
#                                          PulseSequence                                           #
#**************************************************************************************************#
#                                                                                                  #
# A pulse sequence as a Pulseq file describes it: ``version`` (major.minor.revision),              #
# ``definitions`` (the key/value pairs, FOV in metres, times in seconds), ``blocks`` (a structured #
# array with the block id, its duration in units of BlockDurationRaster or its delay event id, and #
# the rf, gx, gy, gz, adc and ext event ids), ``events`` (per table name "rf", "gradients",        #
# "trap", "adc", "delays": the events by id with the columns named as the specification names      #
# them), ``extensions`` (the linked list and the specifications by extension name), ``shapes``     #
# (the decompressed samples by shape id) and ``signature``. ``duration`` is the sum of the block   #
# durations in seconds when the file states them.                                                  #
#                                                                                                  #
#**************************************************************************************************#
class PulseSequence:
    def __init__(
        self,
        version: str,
        definitions: dict,
        blocks: np.ndarray,
        events: dict[str, dict[int, dict]],
        extensions: dict,
        shapes: dict[int, np.ndarray],
        signature: dict,
        meta: Meta,
        provenance: Provenance,
        name: str = "",
    ):
        self.version = version
        self.definitions = definitions
        self.blocks = blocks
        self.events = events
        self.extensions = extensions
        self.shapes = shapes
        self.signature = signature
        self.meta = meta
        self.provenance = provenance
        self.name = name

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    @property
    def duration(self) -> float | None:
        """Total duration in seconds from the block durations, None for files before 1.4."""
        raster = self.definitions.get("BlockDurationRaster")
        if "duration" not in self.blocks.dtype.names or not isinstance(raster, (int, float)):
            return None
        return float(self.blocks["duration"].sum() * raster)

    def to_dict(self) -> dict:
        """JSON-ready summary: counts, definitions and keys, no tables."""
        return {
            "name": self.name,
            "format": self.provenance.format,
            "reader": self.provenance.reader,
            "paths": [str(p) for p in self.provenance.paths],
            "version": self.version,
            "n_blocks": self.n_blocks,
            "duration": self.duration,
            "events": {name: len(table) for name, table in self.events.items()},
            "extensions": sorted(self.extensions.get("specifications", {})),
            "shapes": len(self.shapes),
            "definitions": self.definitions,
            "meta": dict(self.meta),
            "source": dict(self.meta.source),
            "notes": list(self.provenance.notes),
            "versions": self.provenance.versions,
        }

    def __repr__(self) -> str:
        name = f" {self.name}" if self.name else ""
        duration = f", {self.duration:.4g} s" if self.duration is not None else ""
        return f"<PulseSequence{name} v{self.version}: {self.n_blocks} blocks{duration}>"


#**************************************************************************************************#
#                                         NoiseCovariance                                          #
#**************************************************************************************************#
#                                                                                                  #
# The noise covariance of the receive channels: ``matrix`` (channels x channels, complex), the     #
# ``coil_labels`` of its rows (number and name, as far as the file names them), the relative       #
# receiver noise ``bandwidth``, the ``dwell_time`` of the noise samples in seconds and the         #
# ``sample_count`` the matrix was estimated from.                                                  #
#                                                                                                  #
#**************************************************************************************************#
class NoiseCovariance:
    def __init__(
        self,
        matrix: np.ndarray,
        coil_labels: list[dict],
        bandwidth: float | None,
        dwell_time: float | None,
        sample_count: int | None,
        meta: Meta,
        provenance: Provenance,
        name: str = "",
    ):
        self.matrix = matrix
        self.coil_labels = coil_labels
        self.bandwidth = bandwidth
        self.dwell_time = dwell_time
        self.sample_count = sample_count
        self.meta = meta
        self.provenance = provenance
        self.name = name

    @property
    def n_coils(self) -> int:
        return self.matrix.shape[0]

    def to_dict(self) -> dict:
        """JSON-ready summary: everything except the matrix and the vendor header itself."""
        return {
            "name": self.name,
            "format": self.provenance.format,
            "reader": self.provenance.reader,
            "paths": [str(p) for p in self.provenance.paths],
            "shape": list(self.matrix.shape),
            "coil_labels": self.coil_labels,
            "bandwidth": self.bandwidth,
            "dwell_time": self.dwell_time,
            "sample_count": self.sample_count,
            "meta": dict(self.meta),
            "source": dict(self.meta.source),
            "notes": list(self.provenance.notes),
            "versions": self.provenance.versions,
        }

    def __repr__(self) -> str:
        name = f" {self.name}" if self.name else ""
        return (
            f"<NoiseCovariance{name} {self.provenance.format}: {self.n_coils} x {self.n_coils}, "
            f"{self.sample_count} samples>"
        )


#**************************************************************************************************#
#                                            PhysioData                                            #
#**************************************************************************************************#
#                                                                                                  #
# A physiological or other continuous recording at one sampling frequency: ``data`` (samples x     #
# channels, as recorded), ``channels`` (their names, as BIDS names them where it does: cardiac,    #
# respiratory, trigger, ...), ``sampling_frequency`` in Hz, ``start_time`` in seconds against the  #
# clock ``time_reference`` names (e.g. seconds since midnight on the scanner clock), and           #
# ``events`` (per event name the indices of the samples at or right after each event: triggers,    #
# volume markers). Either may be None when the file does not state it.                             #
#                                                                                                  #
#**************************************************************************************************#
class PhysioData:
    def __init__(
        self,
        data: np.ndarray,
        channels: tuple[str, ...],
        sampling_frequency: float | None,
        start_time: float | None,
        time_reference: str,
        events: dict[str, np.ndarray],
        meta: Meta,
        provenance: Provenance,
        name: str = "",
    ):
        self.data = data
        self.channels = tuple(channels)
        self.sampling_frequency = sampling_frequency
        self.start_time = start_time
        self.time_reference = time_reference
        self.events = events
        self.meta = meta
        self.provenance = provenance
        self.name = name

    @property
    def n_samples(self) -> int:
        return self.data.shape[0]

    @property
    def duration(self) -> float | None:
        """Seconds covered by the samples, None without a sampling frequency."""
        if not self.sampling_frequency:
            return None
        return self.n_samples / self.sampling_frequency

    def times(self) -> np.ndarray | None:
        """The time of every sample in seconds (from 0 without a start time)."""
        if not self.sampling_frequency:
            return None
        return (self.start_time or 0.0) + np.arange(self.n_samples) / self.sampling_frequency

    def to_dict(self) -> dict:
        """JSON-ready summary: everything except the samples and the vendor header itself."""
        return {
            "name": self.name,
            "format": self.provenance.format,
            "reader": self.provenance.reader,
            "paths": [str(p) for p in self.provenance.paths],
            "shape": list(self.data.shape),
            "channels": list(self.channels),
            "sampling_frequency": self.sampling_frequency,
            "start_time": self.start_time,
            "time_reference": self.time_reference,
            "events": {name: len(indices) for name, indices in self.events.items()},
            "meta": dict(self.meta),
            "source": dict(self.meta.source),
            "notes": list(self.provenance.notes),
            "versions": self.provenance.versions,
        }

    def __repr__(self) -> str:
        name = f" {self.name}" if self.name else ""
        rate = f" at {self.sampling_frequency:g} Hz" if self.sampling_frequency else ""
        return (
            f"<PhysioData{name} {self.provenance.format}: {self.n_samples} samples x "
            f"{', '.join(self.channels)}{rate}>"
        )


#**************************************************************************************************#
#                                             Protocol                                             #
#**************************************************************************************************#
#                                                                                                  #
# One measurement protocol as a protocol file (an exported exam tree, a single protocol, a printed #
# protocol) holds it: its ``name``, its ``location`` in the exam tree (directories, program), the  #
# standard keys in ``meta`` and every parameter in the vendor header.                              #
#                                                                                                  #
#**************************************************************************************************#
class Protocol:
    def __init__(
        self, name: str, location: tuple[str, ...], meta: Meta, provenance: Provenance,
    ):
        self.name = name
        self.location = tuple(location)
        self.meta = meta
        self.provenance = provenance

    def to_dict(self) -> dict:
        """JSON-ready summary: everything except the vendor header itself."""
        return {
            "name": self.name,
            "location": list(self.location),
            "format": self.provenance.format,
            "reader": self.provenance.reader,
            "paths": [str(p) for p in self.provenance.paths],
            "meta": dict(self.meta),
            "source": dict(self.meta.source),
            "notes": list(self.provenance.notes),
            "versions": self.provenance.versions,
        }

    def __repr__(self) -> str:
        where = f" in {' / '.join(self.location)}" if self.location else ""
        return f"<Protocol {self.name}{where} {self.provenance.format}>"


#**************************************************************************************************#
#                                             Phantom                                              #
#**************************************************************************************************#
#                                                                                                  #
# A simulation phantom as a set of spins: ``positions`` (spins, 3) in metres, ``properties`` one   #
# array per spin property by name (rho, T1, T2, T2s in seconds, ...) with their ``units``, the     #
# ``motion`` definitions as the file gives them, and ``grid`` when the phantom is defined on a     #
# voxel grid (shape, resolution and offset in metres, and the grid index of every spin).           #
#                                                                                                  #
#**************************************************************************************************#
class Phantom:
    def __init__(
        self,
        positions: np.ndarray,
        properties: dict[str, np.ndarray],
        units: dict[str, str],
        motion: list[dict],
        meta: Meta,
        provenance: Provenance,
        name: str = "",
        grid: dict | None = None,
    ):
        self.positions = positions
        self.properties = properties
        self.units = units
        self.motion = motion
        self.meta = meta
        self.provenance = provenance
        self.name = name
        self.grid = grid

    @property
    def n_spins(self) -> int:
        return self.positions.shape[0]

    def to_dict(self) -> dict:
        """JSON-ready summary: everything except the spin arrays and the vendor header itself."""
        return {
            "name": self.name,
            "format": self.provenance.format,
            "reader": self.provenance.reader,
            "paths": [str(p) for p in self.provenance.paths],
            "n_spins": self.n_spins,
            "properties": list(self.properties),
            "units": self.units,
            "motion": len(self.motion),
            "grid": None if self.grid is None else {
                k: v for k, v in self.grid.items() if k != "indices"},
            "meta": dict(self.meta),
            "notes": list(self.provenance.notes),
            "versions": self.provenance.versions,
        }

    def __repr__(self) -> str:
        name = f" {self.name}" if self.name else ""
        return (f"<Phantom{name} {self.provenance.format}: {self.n_spins} spins, "
                f"{', '.join(self.properties)}>")
