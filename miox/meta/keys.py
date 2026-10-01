####################################################################################################
#                                             keys.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 01/10/26                                                                                #
#                                                                                                  #
# Purpose: Key table v0: the standard metadata vocabulary. The NIfTI-MRS required and standard     #
#          keys are taken at import time from the nifti-mrs package so the two never drift, plus a #
#          few MIOX keys the standard does not define. Each entry carries the SI unit the value #
#          is stored in, the MRS-BIDS sidecar name where one exists, and the BasisREMY sheet name  #
#          and unit where the sheet uses it. Vendor aliases and derivation rules are not part of   #
#          v0.                                                                                     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

from dataclasses import dataclass

from nifti_mrs.definitions import dimension_tags, nifti_mrs_version, required, standard_defined

NIFTI_MRS_VERSION = tuple(nifti_mrs_version)
DIM_TAGS = tuple(dimension_tags)


@dataclass(frozen=True)
class Key:
    name: str
    unit: str | None
    """Unit the value is stored in under ``meta[name]`` (SI where the standard says so)."""
    type: tuple
    """Python types accepted, as in ``nifti_mrs.definitions`` (``(list, float)`` for arrays)."""
    doc: str
    origin: str
    """``nifti-mrs:required``, ``nifti-mrs:standard`` or ``miox``."""
    anon: bool = False
    """Removed by anonymisation."""
    bids: str | None = None
    """MRS-BIDS sidecar field name (BIDS schema ``rules/sidecars/mrs.yaml``)."""
    sheet: str | None = None
    """BasisREMY sheet name."""
    sheet_unit: str | None = None
    sheet_factor: float | None = None
    """``sheet value = meta value * sheet_factor``; ``None`` means a derivation rule is needed."""


# MRS-BIDS sidecar names for NIfTI-MRS keys whose BIDS name is different or the same.
# Only names present in the BIDS schema (rules/sidecars/mrs.yaml, checked 1 Oct 2026) are listed.
_BIDS = {
    "SpectrometerFrequency": "SpectrometerFrequency",
    "ResonantNucleus": "ResonantNucleus",
    "SpectralWidth": "SpectralWidth",
    "EchoTime": "EchoTime",
    "MixingTime": "MixingTime",
    "ExcitationFlipAngle": "FlipAngle",
    "WaterSuppressed": "WaterSuppression",
    "WaterSuppressionType": "WaterSuppressionTechnique",
    "Manufacturer": "Manufacturer",
    "ManufacturersModelName": "ManufacturersModelName",
    "DeviceSerialNumber": "DeviceSerialNumber",
    "SoftwareVersions": "SoftwareVersions",
    "InstitutionName": "InstitutionName",
    "InstitutionAddress": "InstitutionAddress",
    "RxCoil": "ReceiveCoilName",
    "SequenceName": "SequenceName",
    "EditCondition": "EditCondition",
    "EditPulse": "EditPulse",
}

# BasisREMY sheet names and units (sheet times are in ms, the standard stores seconds).
_SHEET = {
    "EchoTime": ("TE", "ms", 1e3),
    "MixingTime": ("TM", "ms", 1e3),
    "SpectralWidth": ("Bandwidth", "Hz", 1.0),
    "ResonantNucleus": ("Nucleus", None, 1.0),
}


def _from_nifti_mrs(table: dict, origin: str) -> dict[str, Key]:
    keys = {}
    for name, field in table.items():
        sheet, sheet_unit, factor = _SHEET.get(name, (None, None, None))
        keys[name] = Key(
            name=name,
            unit=field.units,
            type=field.type,
            doc=field.doc,
            origin=origin,
            anon=field.anon,
            bids=_BIDS.get(name),
            sheet=sheet,
            sheet_unit=sheet_unit,
            sheet_factor=factor,
        )
    return keys


_MIOX = (
    Key(
        "FieldStrength",
        "T",
        (float,),
        "Static field strength, derived from SpectrometerFrequency and the gyromagnetic ratio of "
        "ResonantNucleus unless the file states it.",
        "miox",
        bids="MagneticFieldStrength",
        sheet="Bfield",
        sheet_unit="T",
        sheet_factor=1.0,
    ),
    Key(
        "DwellTime",
        "s",
        (float,),
        "Time between consecutive complex samples of the FID; inverse of SpectralWidth.",
        "miox",
    ),
    Key(
        "NumberOfPoints",
        None,
        (int,),
        "Number of complex samples per FID.",
        "miox",
        bids="NumberOfSpectralPoints",
        sheet="Samples",
        sheet_unit=None,
        sheet_factor=1.0,
    ),
    Key(
        "NumberOfAverages",
        None,
        (int,),
        "Number of acquired averages (transients) per spectrum, before any combination.",
        "miox",
    ),
)

KEYS: dict[str, Key] = {
    **_from_nifti_mrs(required, "nifti-mrs:required"),
    **_from_nifti_mrs(standard_defined, "nifti-mrs:standard"),
    **{k.name: k for k in _MIOX},
}
