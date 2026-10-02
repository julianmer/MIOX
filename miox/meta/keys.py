####################################################################################################
#                                             keys.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 01/10/26                                                                                #
#                                                                                                  #
# Purpose: Key table: the standard metadata vocabulary. The NIfTI-MRS required and standard keys   #
#          are taken from the nifti-mrs package so the two never drift, plus the keys the standard #
#          does not define. Each entry carries the unit its value is stored in, the MRS-BIDS       #
#          sidecar name where one exists, and its sheet name and unit.                             #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

from dataclasses import dataclass

from nifti_mrs.definitions import dimension_tags, nifti_mrs_version, required, standard_defined


#**************************************************************************************************#
#                                               Key                                                #
#**************************************************************************************************#
#                                                                                                  #
# One standard metadata key: its unit and type, where it comes from, and its names in MRS-BIDS and #
# in the sheet view.                                                                               #
#                                                                                                  #
#**************************************************************************************************#
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
    """Short name in the sheet view."""
    sheet_unit: str | None = None
    sheet_factor: float | None = None
    """``sheet value = meta value * sheet_factor``; ``None`` means a derivation rule is needed."""


#**************************************************************************************************#
#                                             KeyTable                                             #
#**************************************************************************************************#
#                                                                                                  #
# All standard keys by name: the NIfTI-MRS required and standard keys, then the keys the standard  #
# does not define. Also holds the NIfTI-MRS version and dimension tags it was built from.          #
#                                                                                                  #
#**************************************************************************************************#
class KeyTable(dict):
    # NIfTI-MRS key -> MRS-BIDS sidecar name, only names in the BIDS schema
    BIDS = {
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

    # NIfTI-MRS key -> sheet name, unit and factor (sheet times are in ms, the standard stores s)
    SHEET = {
        "EchoTime": ("TE", "ms", 1e3),
        "MixingTime": ("TM", "ms", 1e3),
        "SpectralWidth": ("Bandwidth", "Hz", 1.0),
        "ResonantNucleus": ("Nucleus", None, 1.0),
    }

    # keys the NIfTI-MRS standard does not define
    MIOX = (
        Key(
            "FieldStrength",
            "T",
            (float,),
            "Static field strength, derived from SpectrometerFrequency and the gyromagnetic ratio "
            "of ResonantNucleus unless the file states it.",
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

    def __init__(self):
        super().__init__()
        self.nifti_mrs_version = tuple(nifti_mrs_version)
        self.dim_tags = tuple(dimension_tags)
        self._add_nifti_mrs(required, "nifti-mrs:required")
        self._add_nifti_mrs(standard_defined, "nifti-mrs:standard")
        self.update({key.name: key for key in self.MIOX})

    def _add_nifti_mrs(self, definitions: dict, origin: str) -> None:
        for name, definition in definitions.items():
            sheet, sheet_unit, sheet_factor = self.SHEET.get(name, (None, None, None))
            self[name] = Key(
                name=name,
                unit=definition.units,
                type=definition.type,
                doc=definition.doc,
                origin=origin,
                anon=definition.anon,
                bids=self.BIDS.get(name),
                sheet=sheet,
                sheet_unit=sheet_unit,
                sheet_factor=sheet_factor,
            )
