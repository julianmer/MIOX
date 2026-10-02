####################################################################################################
#                                             data.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: The canonical spectroscopy object every reader returns: complex FIDs in NIfTI-MRS       #
#          dimension order, dwell time, geometry, tagged higher dimensions, metadata and the       #
#          provenance of the read.                                                                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
# description and per-index header values.                                                         #
#                                                                                                  #
#**************************************************************************************************#
class MRSData:
    DIM_KEY = re.compile(r"^dim_([5-7])(?:_(info|header))?$")  # dimension keys of the header

    def __init__(
        self,
        data: np.ndarray,
        dwelltime: float,
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
    def spectrometer_frequency(self) -> float:
        """MHz."""
        return float(self.meta["SpectrometerFrequency"][0])

    @property
    def nucleus(self) -> str:
        return self.meta["ResonantNucleus"][0]

    @property
    def spectral_width(self) -> float:
        """Hz."""
        return 1 / self.dwelltime

    @property
    def n_points(self) -> int:
        return self.data.shape[3]

    @property
    def time_axis(self) -> np.ndarray:
        """Seconds from the first point."""
        return np.arange(self.n_points) * self.dwelltime

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
        return (
            f"<MRSData{name} {self.provenance.format} {dims}, {self.nucleus} at "
            f"{self.spectrometer_frequency:.3f} MHz, dwell {self.dwelltime * 1e3:.4g} ms>"
        )


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
            f"{self.n_points} points, {self.nucleus} at {self.spectrometer_frequency:.3f} MHz, "
            f"dwell {self.dwelltime * 1e3:.4g} ms>"
        )

