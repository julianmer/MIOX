####################################################################################################
#                                           dicom_mrs.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for DICOM spectroscopy from Siemens, Philips and UIH, one file or a series       #
#          folder. MIOX reads Philips, spec2nii converts Siemens and UIH; the vendor header holds  #
#          every element of the first file (sequences nested, private tags by number, bulk data    #
#          summarised by size) and, for Siemens, both CSA headers and the protocol.                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import contextlib
import dataclasses
import re
import warnings
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pydicom
from mapvbvd.read_twix_hdr import parse_buffer
from nifti_mrs.create_nmrs import gen_nifti_mrs_hdr_ext
from nifti_mrs.hdr_ext import Hdr_Ext
from nifti_mrs.nifti_mrs import NIFTI_MRS
from spec2nii.dcm2niiOrientation.orientationFuncs import dcm_to_nifti_orientation

with warnings.catch_warnings():  # nibabel.nicom warns on import that it is experimental
    warnings.simplefilter("ignore", UserWarning)
    from nibabel.nicom import csareader
    from spec2nii.Siemens import dicomfunctions as siemens_dcm
    from spec2nii.Siemens.dicomfunctions import multi_file_dicom as siemens_dicom
    from spec2nii.uih import multi_file_dicom as uih_dicom

# own
from miox.data import MRSData
from miox.readers.base import ReaderError, Spec2niiReader
from miox.readers.philips_dicom_mrs import PhilipsDicomMrs
from miox.readers.siemens_twix import SiemensTwixReader
from miox.sniff import Format, Sniff, open_dicom


#**************************************************************************************************#
#                                          DicomMrsReader                                          #
#**************************************************************************************************#
#                                                                                                  #
# A folder is read series by series (SeriesInstanceUID). Philips series are read by MIOX           #
# (PhilipsDicomMrs). Siemens and UIH series go to spec2nii's converter together, which splits off  #
# references; the converters expect one series at a time and stack or misname several. The source  #
# tables follow spec2nii's extraction per vendor and variant (Siemens VB-VE with CSA headers or XA #
# enhanced; UIH). Siemens VA single-voxel files (CSA headers in (0029,12xx), protocol              #
# "MrProtocol"), which spec2nii does not read, are converted here: data in (7FE1,1010) as stored,  #
# the dwell time RealDwellTime (the ADC dwell of the protocol is oversampled), the voxel from the  #
# protocol's sVoI (ImagePositionPatient is its corner there) and the CSA ImageOrientationPatient.  #
# The MR Spectroscopy module's Chemical Shift Reference (ppm at the transmitter frequency) is      #
# ReferencePPM. Keys the file states in top-level elements that spec2nii does not read (series,    #
# station, acquisition date and duration, localisation technique, ...) are added to every dataset  #
# of a DICOM file; the stated field strength replaces the one derived from the frequency.          #
#                                                                                                  #
#**************************************************************************************************#
class DicomMrsReader(Spec2niiReader):
    FORMATS = (Format.DICOM_MRS,)
    CONVERTERS = {"siemens": "dicom", "uih": "uih"}
    REFERENCE = "dicom/ChemicalShiftReference"
    TAGS = {"dicom": "DIM_DYN", "uih": None}  # spec2nii defaults
    PHILIPS_SOURCE = "DICOM"
    BINARY_VRS = {"OB", "OW", "OF", "OD", "OL", "OV", "UN"}
    XA_SERIES_HEADER = (0x0021, 0x1019)  # Siemens XA: XProtocol text with the ASCCONV block
    BULK = 1024  # binary values longer than this are summarised by their size
    SHARED = "dicom/SharedFunctionalGroupsSequence/0"

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
    SOURCES_SIEMENS_VA = SOURCES_COMMON | {
        "SpectrometerFrequency": "csa_image/ImagingFrequency",
        "ResonantNucleus": "csa_image/ImagedNucleus",
        "EchoTime": "MrProtocol/alTE[0]",
        "RepetitionTime": "csa_image/RepetitionTime",
        "ExcitationFlipAngle": "csa_image/FlipAngle",
        "TxCoil": "csa_image/TransmittingCoil",
        "SequenceName": "csa_image/SequenceName",
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

    HEADER_KEYS = {  # top-level elements
        "Modality": (("dicom/Modality",), str),
        "StationName": (("dicom/StationName",), str),
        "InstitutionalDepartmentName": (("dicom/InstitutionalDepartmentName",), str),
        "StudyDescription": (("dicom/StudyDescription",), str),
        "ProcedureStepDescription": (("dicom/PerformedProcedureStepDescription",), str),
        "SeriesDescription": (("dicom/SeriesDescription",), str),
        "SeriesNumber": (("dicom/SeriesNumber",), int),
        "AcquisitionNumber": (("dicom/AcquisitionNumber",), int),
        "ImageType": (("dicom/ImageType",), list),
        "BodyPart": (("dicom/BodyPartExamined",), str),
        "ScanOptions": (("dicom/ScanOptions",), str),
        "PatientID": (("dicom/PatientID",), str),
        "FieldStrength": (("dicom/MagneticFieldStrength",), float),
        "AcquisitionDuration": (("dicom/AcquisitionDuration",), float),
        "EchoTrainLength": (("dicom/EchoTrainLength",), int),
        "VolumeLocalizationTechnique": (("dicom/VolumeLocalizationTechnique",), str),
        "MRSpectroscopyAcquisitionType": (("dicom/MRSpectroscopyAcquisitionType",), str),
        "Decoupling": (("dicom/Decoupling",), str),
    }

    def read(self, sniff: Sniff) -> list[MRSData]:
        series = self._series(sniff)
        parts = [sniff] if len(series) == 1 else [
            dataclasses.replace(sniff, companions=tuple(files)) for files in series.values()]
        read = self._philips if sniff.vendor == "philips" else super().read
        datasets = [d for part in parts for d in read(part)]
        for d in datasets:
            self._acquisition_time(d)
        return datasets

    def _philips(self, sniff: Sniff) -> list[MRSData]:
        """One Philips series, read by MIOX (PhilipsDicomMrs)."""
        files = self._files(sniff)
        header = self.header(files[0])
        notes: list[str] = []
        with self.captured(notes):
            spectra = PhilipsDicomMrs().read(files)
        datasets = []
        for s in spectra:
            spectrum_notes = notes + s.notes
            data, tags = self.without_zero_indices(s.data, s.tags, spectrum_notes)
            d = self.dataset(sniff, data, s.dwelltime, s.values, header, s.name, spectrum_notes,
                             self.PHILIPS_SOURCE, s.affine, tags)
            self._reference(d)
            self._header_keys(d)
            datasets.append(d)
        return datasets

    @staticmethod
    def _acquisition_time(d: MRSData) -> None:
        """AcquisitionDateTime in ISO 8601, from the element or from AcquisitionDate and Time."""
        dicom = d.meta.vendor["dicom"]
        stamp = str(dicom.get("AcquisitionDateTime") or "")
        source, field = "AcquisitionDateTime", "dicom/AcquisitionDateTime"
        if not stamp and dicom.get("AcquisitionDate") and dicom.get("AcquisitionTime"):
            stamp = f"{dicom['AcquisitionDate']}{dicom['AcquisitionTime']}"
            source, field = "AcquisitionDate + AcquisitionTime", "dicom/AcquisitionTime"
        if len(stamp) >= 14 and "AcquisitionDateTime" not in d.meta:
            clock = re.split(r"[+-]", stamp[8:])[0]  # without a UTC offset
            value = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}T{clock[:2]}:{clock[2:4]}:{clock[4:]}"
            d.meta.set("AcquisitionDateTime", value, f"DICOM: {source}", field)

    def _series(self, sniff: Sniff) -> dict[str, list[Path]]:
        """The DICOM files of a folder grouped by SeriesInstanceUID, in folder order."""
        series: dict[str, list[Path]] = {}
        for path in self._files(sniff):
            with open_dicom(path) as fh:
                tags = ["SeriesInstanceUID"]
                ds = pydicom.dcmread(fh, stop_before_pixels=True, specific_tags=tags)
            series.setdefault(str(ds.get("SeriesInstanceUID", "")), []).append(path)
        return series

    def converter(self, sniff: Sniff) -> str:
        if sniff.vendor not in self.CONVERTERS:
            raise ReaderError(f"no spec2nii converter for {sniff.vendor} DICOM spectroscopy")
        return self.CONVERTERS[sniff.vendor]

    def convert(self, sniff: Sniff) -> tuple[list[NIFTI_MRS], list[str], dict]:
        converter = self.converter(sniff)
        files = self._files(sniff)
        header = self.header(files[0])
        if converter == "dicom" and "MrProtocol" in header:
            images = [self._siemens_va(f, self.header(f)) for f in files]
            names = [str(header["dicom"].get("SeriesDescription") or f.stem) for f in files]
            return images, names, header
        function = {"dicom": siemens_dicom, "uih": uih_dicom}
        with self._siemens_orientation():
            images, names = function[converter](files, None, self.TAGS[converter], False)
        return list(images), list(names), header

    def sources(self, vendor: dict) -> dict[str, str | tuple[str, ...]]:
        manufacturer = str(vendor["dicom"].get("Manufacturer", "")).lower()
        if "siemens" in manufacturer:
            if "MrProtocol" in vendor:
                return self.SOURCES_SIEMENS_VA
            return self.SOURCES_SIEMENS_VX if "csa_image" in vendor else self.SOURCES_SIEMENS_XA
        return self.SOURCES_UIH

    def correct(self, d: MRSData) -> None:
        """The inversion time spec2nii leaves in ms (Siemens CSA)."""
        super().correct(d)
        fields = d.meta.fields
        if fields.get("InversionTime") == "csa_image/InversionTime":
            self.fix(d, "InversionTime", d.meta["InversionTime"] * 1e-3, "CSA value in ms")

    def _files(self, sniff: Sniff) -> list[Path]:
        if not sniff.path.is_dir():
            return [sniff.path]
        files = []
        for path in sniff.companions:
            with open(path, "rb") as fh:
                if fh.read(132)[128:] == b"DICM":
                    files.append(path)
        return sorted(files)

    def header(self, path: Path) -> dict:
        """Complete header of one DICOM file: elements, CSA headers and Siemens protocol (from
        the CSA series header, or from the XA series header (0021,1019) of enhanced files)."""
        with open_dicom(path) as fh:
            ds = pydicom.dcmread(fh, force=True)  # also files without preamble (ACR-NEMA)
        header = {"file_meta": self._dataset(ds.file_meta), "dicom": self._dataset(ds)}
        for kind in ("image", "series"):
            csa = csareader.get_csa_header(ds, kind)
            if csa:
                header[f"csa_{kind}"] = {
                    name: self._csa_value(tag) for name, tag in csa["tags"].items()
                }
        for name in ("MrPhoenixProtocol", "MrProtocol"):  # VB and later, VA
            protocol = header.get("csa_series", {}).get(name)
            if isinstance(protocol, str):
                header[name] = self._protocol(protocol)
        if "MrPhoenixProtocol" not in header:
            for element in ds.iterall():
                value = element.value if element.tag == self.XA_SERIES_HEADER else None
                if isinstance(value, bytes) and b"### ASCCONV BEGIN" in value:
                    header["MrPhoenixProtocol"] = self._protocol(value.decode("latin-1"))
                    break
        return header

    def _protocol(self, text: str) -> dict:
        """The ASCCONV block of a Siemens protocol as a flat dict."""
        return {
            SiemensTwixReader.protocol_key(k): self.plain(v) for k, v in parse_buffer(text).items()
        }

    @contextlib.contextmanager
    def _siemens_orientation(self) -> Iterator[None]:
        """spec2nii's Siemens enhanced orientation, for files whose ImageOrientationPatient does
        not hold six direction cosines (one syngo B15 export stores "1"): identity, noted."""
        original = siemens_dcm._get_enhanced_dcm_img_orientation_pat
        warned = []

        def orientation(img):
            try:
                return original(img)
            except ValueError as e:  # the stated values do not reshape to two rows of three
                if "reshape" not in str(e):
                    raise
            if not warned:
                warnings.warn("ImageOrientationPatient does not hold six values: identity "
                              "orientation assumed, the orientation is not known", stacklevel=2)
                warned.append(True)
            return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        siemens_dcm._get_enhanced_dcm_img_orientation_pat = orientation
        try:
            yield
        finally:
            siemens_dcm._get_enhanced_dcm_img_orientation_pat = original

    def _siemens_va(self, path: Path, header: dict) -> NIFTI_MRS:
        """One Siemens VA single-voxel file as NIfTI-MRS, with the keys spec2nii sets for VB."""
        values = np.frombuffer(pydicom.dcmread(path)[0x7FE1, 0x1010].value, dtype="<f4")
        fid = (values[0::2] + 1j * values[1::2]).reshape(1, 1, 1, -1)
        csa, protocol, dicom = header["csa_image"], header["MrProtocol"], header["dicom"]
        voi = "sSpecPara.sVoI."
        position = [float(protocol.get(f"{voi}sPosition.d{a}", 0)) for a in ("Sag", "Cor", "Tra")]
        size = [float(protocol[f"{voi}d{k}"]) for k in ("PhaseFOV", "ReadoutFOV", "Thickness")]
        orientation = np.asarray(csa["ImageOrientationPatient"], dtype=float).reshape(2, 3)
        affine = dcm_to_nifti_orientation(orientation, position, np.asarray(size), (1, 1, 1)).Q44
        hdr = Hdr_Ext(float(csa["ImagingFrequency"]), csa["ImagedNucleus"])
        echo = sum(v for k, v in protocol.items() if re.fullmatch(r"alTE\[\d+\]", k))
        hdr.set_standard_def("EchoTime", echo * 1e-6)
        hdr.set_standard_def("RepetitionTime", float(csa["RepetitionTime"]) * 1e-3)
        hdr.set_standard_def("ExcitationFlipAngle", float(csa["FlipAngle"]))
        hdr.set_standard_def("TxCoil", csa["TransmittingCoil"])
        hdr.set_standard_def("SequenceName", csa["SequenceName"])
        for key, name in (
            ("Manufacturer", "Manufacturer"),
            ("ManufacturersModelName", "ManufacturerModelName"),
            ("DeviceSerialNumber", "DeviceSerialNumber"),
            ("SoftwareVersions", "SoftwareVersions"),
            ("InstitutionName", "InstitutionName"),
            ("ProtocolName", "ProtocolName"),
            ("PatientPosition", "PatientPosition"),
            ("PatientSex", "PatientSex"),
        ):
            if dicom.get(name) not in (None, ""):
                hdr.set_standard_def(key, str(dicom[name]))
        dwelltime = float(csa["RealDwellTime"]) * 1e-9
        return gen_nifti_mrs_hdr_ext(fid, dwelltime, hdr, affine, no_conj=True)

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
