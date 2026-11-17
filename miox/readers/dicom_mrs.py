####################################################################################################
#                                           dicom_mrs.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for DICOM spectroscopy from Siemens, Philips and UIH, one file or a series       #
#          folder; the vendor header holds every element of the first file (sequences nested,      #
#          private tags by number, bulk data summarised by size) and, for Siemens, both CSA        #
#          headers and the protocol.                                                               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import dataclasses
import re
import warnings
from pathlib import Path

import pydicom
from mapvbvd.read_twix_hdr import parse_buffer

with warnings.catch_warnings():  # nibabel.nicom warns on import that it is experimental
    warnings.simplefilter("ignore", UserWarning)
    from nibabel.nicom import csareader

# own
from miox.data import MRSData
from miox.readers.base import Reader, ReaderError
from miox.readers.philips_dicom_mrs import PhilipsDicomMrs
from miox.readers.siemens_dicom_mrs import SiemensDicomMrs
from miox.readers.siemens_twix import SiemensTwixReader
from miox.readers.uih_dicom_mrs import UihDicomMrs
from miox.sniff import Format, Sniff, open_dicom


#**************************************************************************************************#
#                                          DicomMrsReader                                          #
#**************************************************************************************************#
#                                                                                                  #
# A folder is read series by series (SeriesInstanceUID), each by its vendor's reader:              #
# PhilipsDicomMrs, SiemensDicomMrs or UihDicomMrs. The MR Spectroscopy module's Chemical Shift     #
# Reference (ppm at the transmitter frequency) is ReferencePPM. Keys the file states in top-level  #
# elements (series, station, acquisition date and duration, localisation technique, ...) are       #
# added to every dataset of a DICOM file; the stated field strength replaces the one derived from  #
# the frequency.                                                                                   #
#                                                                                                  #
#**************************************************************************************************#
class DicomMrsReader(Reader):
    FORMATS = (Format.DICOM_MRS,)
    PACKAGES = Reader.PACKAGES + ("pydicom",)
    VENDORS = {"philips": PhilipsDicomMrs, "siemens": SiemensDicomMrs, "uih": UihDicomMrs}
    SOURCE = "DICOM"
    REFERENCE = "dicom/ChemicalShiftReference"
    BINARY_VRS = {"OB", "OW", "OF", "OD", "OL", "OV", "UN"}
    XA_SERIES_HEADER = (0x0021, 0x1019)  # Siemens XA: XProtocol text with the ASCCONV block
    BULK = 1024  # binary values longer than this are summarised by their size

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
        if sniff.vendor not in self.VENDORS:
            raise ReaderError(f"no reader for {sniff.vendor} DICOM spectroscopy")
        series = self._series(sniff)
        parts = [sniff] if len(series) == 1 else [
            dataclasses.replace(sniff, companions=tuple(files)) for files in series.values()]
        datasets = [d for part in parts for d in self._one_series(part)]
        for d in datasets:
            self._acquisition_time(d)
        return datasets

    def _one_series(self, sniff: Sniff) -> list[MRSData]:
        """One series, read by its vendor's reader."""
        files = self._files(sniff)
        header = self.header(files[0])
        notes: list[str] = []
        vendor = self.VENDORS[sniff.vendor]()
        with self.captured(notes):
            spectra = vendor.read(files)
        notes += getattr(vendor, "notes", [])
        datasets = []
        for s in spectra:
            spectrum_notes = notes + s.notes
            data, tags = self.without_zero_indices(s.data, s.tags, spectrum_notes)
            d = self.dataset(sniff, data, s.dwelltime, s.values, header, s.name, spectrum_notes,
                             self.SOURCE, s.affine, tags)
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
