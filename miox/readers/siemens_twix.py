####################################################################################################
#                                         siemens_twix.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for Siemens twix raw data (.dat, VB to XA). pyMapVBVD parses the file, spec2nii  #
#          assembles the data; the vendor header keeps all header buffers of the measurement.      #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import warnings

from mapvbvd import mapVBVD
from nifti_mrs.nifti_mrs import NIFTI_MRS
from spec2nii.Siemens.twixfunctions import process_twix, xa_or_vx

# own
from miox.data import MRSData
from miox.readers.base import ReaderError, Spec2niiReader
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                        SiemensTwixReader                                         #
#**************************************************************************************************#
#                                                                                                  #
# Converts every data stream (evalinfo flag) of the measurement: ``image`` holds the scan itself;  #
# the others (water reference, noise, phase correction, ...) become further datasets named after   #
# their flag, and a stream spec2nii cannot convert is noted instead of failing the read. Header    #
# keys of the protocol buffers are written in Siemens notation, e.g. ``sTXSPEC.asNucleusInfo[0].   #
# tNucleus``.                                                                                      #
#                                                                                                  #
#**************************************************************************************************#
class SiemensTwixReader(Spec2niiReader):
    FORMATS = (Format.SIEMENS_TWIX,)
    CONVERTER = "twix"
    MAIN_STREAM = "image"
    MULTIRAID = 2  # in a multi-measurement file the second holds the scan (spec2nii default)
    NO_OVERRIDES = {"dims": (None, None, None), "tags": (None, None, None)}

    # where spec2nii reads each key, the same for XA and for VB to VE
    SOURCES_COMMON = {
        "EchoTime": "Phoenix/alTE[0]",
        "DwellTime": "MeasYaps/sRXSPEC.alDwellTime[0]",
        "Manufacturer": "Dicom/Manufacturer",
        "ManufacturersModelName": "Dicom/ManufacturersModelName",
        "DeviceSerialNumber": "Dicom/DeviceSerialNumber",
        "SoftwareVersions": "Dicom/SoftwareVersions",
        "InstitutionName": "Dicom/InstitutionName",
        "InstitutionAddress": "Dicom/InstitutionAddress",
        "RxCoil": (
            "MeasYaps/sCoilSelectMeas.aRxCoilSelectData[0].asList[0].sCoilElementID.tCoilID",
            "MeasYaps/asCoilSelectMeas[0].asList[0].sCoilElementID.tCoilID",
        ),
        "SequenceName": "Meas/tSequenceString",
        "ProtocolName": "Dicom/tProtocolName",
        "PatientWeight": "Meas/flUsedPatientWeight",
        "PatientDoB": "Meas/PatientBirthDay",
        "PatientSex": "Meas/PatientSex",
        "PulseSequenceFile": "Config/SequenceFileName",
        "IceProgramFile": "Meas/tICEProgramName",
    }
    SOURCES_XA = SOURCES_COMMON | {
        "SpectrometerFrequency": "Dicom/lFrequency",
        "ResonantNucleus": "MeasYaps/sTXSPEC.asNucleusInfo[0].tNucleus",
        "RepetitionTime": "MeasYaps/alTR[0]",
        "InversionTime": "MeasYaps/alTI[0]",
        "ExcitationFlipAngle": "MeasYaps/adFlipAngleDegree[0]",
        "TxOffset": "MeasYaps/sSpecPara.dDeltaFrequency",
        "TxCoil": "MeasYaps/sCoilSelectMeas.aTxCoilSelectData[0].asList[0].sCoilElementID.tCoilID",
        "PatientPosition": "Meas/tPatientPosition",
        "PatientName": "Meas/tPatientName",
    }
    SOURCES_VX = SOURCES_COMMON | {
        "SpectrometerFrequency": "Meas/Frequency",
        "ResonantNucleus": "Meas/ResonantNucleus",
        "RepetitionTime": ("Meas/TR_Time", "Meas/TR"),
        "InversionTime": "Meas/TI_Time",
        "ExcitationFlipAngle": "Meas/FlipAngle",
        "TxOffset": "Meas/dDeltaFrequency",
        "PatientPosition": "Meas/PatientPosition",
        "PatientName": "Meas/PatientName",
    }

    def convert(self, sniff: Sniff) -> tuple[list[NIFTI_MRS], list[str], dict]:
        twix = mapVBVD(str(sniff.path), quiet=True)
        if isinstance(twix, list):
            twix = twix[min(self.MULTIRAID, len(twix)) - 1]
        streams = [key for key in twix.keys() if key != "hdr"]
        if self.MAIN_STREAM not in streams:
            raise ReaderError(f"no '{self.MAIN_STREAM}' data in {sniff.path.name}: {streams}")
        images, names = self._stream(twix, sniff, self.MAIN_STREAM)
        for stream in streams:
            if stream == self.MAIN_STREAM:
                continue
            try:
                more, more_names = self._stream(twix, sniff, stream)
            except Exception as e:
                message = f"stream '{stream}' not converted: {type(e).__name__}: {e}"
                warnings.warn(message, stacklevel=2)
                continue
            images += more
            names += [f"{name}_{stream}" for name in more_names]
        return images, names, self._vendor(twix, streams)

    def sources(self, vendor: dict) -> dict[str, str | tuple[str, ...]]:
        return self.SOURCES_XA if xa_or_vx(vendor) == "xa" else self.SOURCES_VX

    def correct(self, d: MRSData) -> None:
        """spec2nii stores the inversion time unscaled (XA) or not at all (VB to VE)."""
        super().correct(d)
        inversion = d.meta.vendor["MeasYaps"].get("alTI[0]")
        if isinstance(inversion, (int, float)):
            d.meta.set(
                "InversionTime",
                inversion * 1e-6,
                "MeasYaps/alTI[0] in µs, converted to s",
                "MeasYaps/alTI[0]",
            )

    def _stream(self, twix, sniff: Sniff, stream: str) -> tuple[list[NIFTI_MRS], list[str]]:
        images, names = process_twix(
            twix, None, sniff.path.name, stream, self.NO_OVERRIDES, quiet=True, remove_os=False
        )
        return list(images), list(names)

    def _vendor(self, twix, streams: list[str]) -> dict:
        header = twix["hdr"]
        vendor = {
            buffer: {self.protocol_key(k): self.plain(v) for k, v in header[buffer].items()}
            for buffer in header.keys()
        }
        vendor["streams"] = {
            stream: {
                "dims": list(twix[stream].dataDims),
                "size": self.plain(twix[stream].fullSize),
            }
            for stream in streams
        }
        return vendor

    @staticmethod
    def protocol_key(key) -> str:
        """Tuple keys in Siemens notation: ``('alTE', '0')`` becomes ``alTE[0]``."""
        if not isinstance(key, tuple):
            return str(key)
        return "".join(f"[{p}]" if p.isdigit() else f".{p}" for p in key).lstrip(".")
