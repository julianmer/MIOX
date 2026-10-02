####################################################################################################
#                                           dicom_mrs.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for DICOM spectroscopy from Siemens, Philips and UIH, one file or a series       #
#          folder. spec2nii converts with the vendor's converter; the vendor header holds every    #
#          element of the first file (sequences nested, private tags by number, bulk data          #
#          summarised by size) and, for Siemens, both CSA headers and the protocol.                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import warnings
from pathlib import Path

import pydicom
from mapvbvd.read_twix_hdr import parse_buffer
from nifti_mrs.nifti_mrs import NIFTI_MRS

with warnings.catch_warnings():  # nibabel.nicom warns on import that it is experimental
    warnings.simplefilter("ignore", UserWarning)
    from nibabel.nicom import csareader
    from spec2nii.Philips.philips_dcm import multi_file_dicom as philips_dicom
    from spec2nii.Siemens.dicomfunctions import multi_file_dicom as siemens_dicom
    from spec2nii.uih import multi_file_dicom as uih_dicom

# own
from miox.readers.base import ReaderError, Spec2niiReader
from miox.readers.siemens_twix import SiemensTwixReader
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                          DicomMrsReader                                          #
#**************************************************************************************************#
#                                                                                                  #
# A folder is read as one series set: all its DICOM files go to the converter together, which      #
# groups them by series. The source tables follow spec2nii's extraction per vendor and variant     #
# (Siemens VB-VE with CSA headers or XA enhanced; Philips enhanced or classic; UIH).               #
#                                                                                                  #
#**************************************************************************************************#
class DicomMrsReader(Spec2niiReader):
    FORMATS = (Format.DICOM_MRS,)
    CONVERTERS = {"siemens": "dicom", "philips": "philips_dicom", "uih": "uih"}
    TAGS = {"dicom": "DIM_DYN", "philips_dicom": "DIM_DYN", "uih": None}  # spec2nii defaults
    BINARY_VRS = {"OB", "OW", "OF", "OD", "OL", "OV", "UN"}
    BULK = 1024  # binary values longer than this are summarised by their size
    ENHANCED_MRS = "1.2.840.10008.5.1.4.1.1.4.2"
    SHARED = "dicom/SharedFunctionalGroupsSequence/0"
    PER_FRAME = "dicom/PerFrameFunctionalGroupsSequence/0"

    SOURCES_COMMON = {
        "Manufacturer": "dicom/Manufacturer",
        "ManufacturersModelName": "dicom/ManufacturerModelName",
        "DeviceSerialNumber": "dicom/DeviceSerialNumber",
        "SoftwareVersions": "dicom/SoftwareVersions",
        "InstitutionName": "dicom/InstitutionName",
        "InstitutionAddress": "dicom/InstitutionAddress",
        "ProtocolName": "dicom/ProtocolName",
        "PatientPosition": "dicom/PatientPosition",
        "PatientName": "dicom/PatientName",
        "PatientWeight": "dicom/PatientWeight",
        "PatientDoB": "dicom/PatientBirthDate",
        "PatientSex": "dicom/PatientSex",
    }
    SOURCES_SIEMENS_VX = SOURCES_COMMON | {
        "SpectrometerFrequency": "csa_image/ImagingFrequency",
        "ResonantNucleus": "csa_image/ImagedNucleus",
        "EchoTime": "MrPhoenixProtocol/alTE[0]",
        "RepetitionTime": "csa_image/RepetitionTime",
        "InversionTime": "csa_image/InversionTime",
        "ExcitationFlipAngle": "csa_image/FlipAngle",
        "TxCoil": "csa_image/TransmittingCoil",
        "RxCoil": ("csa_image/ReceivingCoil", "csa_image/ImaCoilString"),
        "SequenceName": "csa_image/SequenceName",
        "PulseSequenceFile": "csa_image/SequenceName",
        "DwellTime": "csa_image/RealDwellTime",
    }
    SOURCES_SIEMENS_XA = SOURCES_COMMON | {
        "SpectrometerFrequency": "dicom/TransmitterFrequency",
        "ResonantNucleus": "dicom/ResonantNucleus",
        "EchoTime": f"{SHARED}/MREchoSequence/0/EffectiveEchoTime",
        "RepetitionTime": f"{SHARED}/MRTimingAndRelatedParametersSequence/0/RepetitionTime",
        "ExcitationFlipAngle": f"{SHARED}/MRTimingAndRelatedParametersSequence/0/FlipAngle",
        "TxCoil": f"{SHARED}/MRTransmitCoilSequence/0/TransmitCoilName",
        "RxCoil": f"{SHARED}/MRReceiveCoilSequence/0/ReceiveCoilName",
        "SequenceName": "dicom/PulseSequenceName",
        "PatientID": "dicom/PatientID",
        "DwellTime": "dicom/SpectralWidth",
    }
    SOURCES_PHILIPS_ENHANCED = SOURCES_COMMON | {
        "SpectrometerFrequency": "dicom/TransmitterFrequency",
        "ResonantNucleus": "dicom/ResonantNucleus",
        "EchoTime": f"{PER_FRAME}/MREchoSequence/0/EffectiveEchoTime",
        "RepetitionTime": (
            f"{PER_FRAME}/MRTimingAndRelatedParametersSequence/0/RepetitionTime",
            f"{SHARED}/MRTimingAndRelatedParametersSequence/0/RepetitionTime",
        ),
        "ExcitationFlipAngle": (
            f"{PER_FRAME}/MRTimingAndRelatedParametersSequence/0/FlipAngle",
            f"{SHARED}/MRTimingAndRelatedParametersSequence/0/FlipAngle",
        ),
        "SequenceName": "dicom/PulseSequenceName",
        "DwellTime": "dicom/SpectralWidth",
    }
    SOURCES_PHILIPS_CLASSIC = SOURCES_COMMON | {
        "SpectrometerFrequency": "dicom/(2001,1083)",
        "ResonantNucleus": "dicom/(2001,1087)",
        "EchoTime": "dicom/(2005,1310)",
        "RepetitionTime": "dicom/RepetitionTime",
        "DwellTime": "dicom/(2005,1357)",
    }
    SOURCES_UIH = SOURCES_COMMON | {
        "SpectrometerFrequency": "dicom/TransmitterFrequency",
        "ResonantNucleus": "dicom/ResonantNucleus",
        "EchoTime": "dicom/EchoTime",
        "RepetitionTime": "dicom/RepetitionTime",
        "InversionTime": "dicom/InversionTime",
        "ExcitationFlipAngle": "dicom/FlipAngle",
        "SequenceName": "dicom/SequenceName",
        "DwellTime": "dicom/SpectralWidth",
    }

    def converter(self, sniff: Sniff) -> str:
        if sniff.vendor not in self.CONVERTERS:
            raise ReaderError(f"no spec2nii converter for {sniff.vendor} DICOM spectroscopy")
        return self.CONVERTERS[sniff.vendor]

    def convert(self, sniff: Sniff) -> tuple[list[NIFTI_MRS], list[str], dict]:
        converter = self.converter(sniff)
        files = self._files(sniff)
        function = {"dicom": siemens_dicom, "philips_dicom": philips_dicom, "uih": uih_dicom}
        images, names = function[converter](files, None, self.TAGS[converter], False)
        return list(images), list(names), self._header(files[0])

    def sources(self, vendor: dict) -> dict[str, str | tuple[str, ...]]:
        manufacturer = str(vendor["dicom"].get("Manufacturer", "")).lower()
        if "siemens" in manufacturer:
            return self.SOURCES_SIEMENS_VX if "csa_image" in vendor else self.SOURCES_SIEMENS_XA
        if "philips" in manufacturer:
            enhanced = vendor["file_meta"].get("MediaStorageSOPClassUID") == self.ENHANCED_MRS
            return self.SOURCES_PHILIPS_ENHANCED if enhanced else self.SOURCES_PHILIPS_CLASSIC
        return self.SOURCES_UIH

    def _files(self, sniff: Sniff) -> list[Path]:
        if not sniff.path.is_dir():
            return [sniff.path]
        files = []
        for path in sniff.companions:
            with open(path, "rb") as fh:
                if fh.read(132)[128:] == b"DICM":
                    files.append(path)
        return sorted(files)

    def _header(self, path: Path) -> dict:
        ds = pydicom.dcmread(path)
        header = {"file_meta": self._dataset(ds.file_meta), "dicom": self._dataset(ds)}
        for kind in ("image", "series"):
            csa = csareader.get_csa_header(ds, kind)
            if csa:
                header[f"csa_{kind}"] = {
                    name: self._csa_value(tag) for name, tag in csa["tags"].items()
                }
        protocol = header.get("csa_series", {}).get("MrPhoenixProtocol")
        if isinstance(protocol, str):
            header["MrPhoenixProtocol"] = {
                SiemensTwixReader.protocol_key(k): self.plain(v)
                for k, v in parse_buffer(protocol).items()
            }
        return header

    def _dataset(self, ds: pydicom.Dataset) -> dict:
        elements = {}
        for element in ds:
            key = element.keyword or f"({element.tag.group:04X},{element.tag.element:04X})"
            if element.VR == "SQ":
                elements[key] = [self._dataset(item) for item in element.value]
            elif element.VR in self.BINARY_VRS and len(element.value or b"") > self.BULK:
                elements[key] = f"<{len(element.value)} bytes>"
            else:
                elements[key] = self._value(element.value)
        return elements

    def _value(self, value):
        if isinstance(value, (pydicom.multival.MultiValue, list, tuple)):
            return [self._value(v) for v in value]
        if isinstance(value, (pydicom.valuerep.DSfloat, pydicom.valuerep.DSdecimal)):
            return float(value)
        if isinstance(value, pydicom.valuerep.IS):
            return int(value)
        value = self.plain(value)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    def _csa_value(self, tag: dict):
        items = [self.plain(item) for item in tag["items"]]
        if not items:
            return None
        return items[0] if len(items) == 1 else items
