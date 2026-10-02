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

import contextlib
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
    from nibabel.nicom.dicomwrappers import wrapper_from_file
    from spec2nii.Philips import philips_dcm
    from spec2nii.Philips.philips_dcm import multi_file_dicom as philips_dicom
    from spec2nii.Siemens.dicomfunctions import multi_file_dicom as siemens_dicom
    from spec2nii.uih import multi_file_dicom as uih_dicom

# own
from miox.data import MRSData
from miox.readers.base import ReaderError, Spec2niiReader
from miox.readers.siemens_twix import SiemensTwixReader
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                          DicomMrsReader                                          #
#**************************************************************************************************#
#                                                                                                  #
# A folder is read as one series set: all its DICOM files go to the converter together, which      #
# groups them by series. The source tables follow spec2nii's extraction per vendor and variant     #
# (Siemens VB-VE with CSA headers or XA enhanced; Philips enhanced or classic; UIH). Siemens VA    #
# single-voxel files (CSA headers in (0029,12xx), protocol "MrProtocol"), which spec2nii does not  #
# read, are converted here: data in (7FE1,1010) as stored, the dwell time RealDwellTime (the ADC   #
# dwell of the protocol is oversampled), the voxel from the protocol's sVoI (ImagePositionPatient  #
# is its corner there) and the CSA ImageOrientationPatient. Philips enhanced files with one        #
# localisation slab (spec2nii needs three) take the voxel from the plane position (the slab        #
# centre), the plane orientation negated as spec2nii negates the slab orientations, PixelSpacing   #
# in plane and the slab thickness along the normal. Philips enhanced CSI, which spec2nii does not  #
# read: SpectroscopyData as (frames, rows, columns, points), the frames split into metabolite and  #
# reference by (2005,1304) of the private per-frame items, conjugated as spec2nii conjugates       #
# Philips SVS (checked: NAA, Cr, Cho at 2.02, 3.03, 3.23 ppm), the grid from the plane position    #
# (first voxel), orientation and pixel spacing. The MR Spectroscopy module's Chemical Shift        #
# Reference (ppm at the transmitter frequency) is ReferencePPM.                                    #
#                                                                                                  #
#**************************************************************************************************#
class DicomMrsReader(Spec2niiReader):
    FORMATS = (Format.DICOM_MRS,)
    CONVERTERS = {"siemens": "dicom", "philips": "philips_dicom", "uih": "uih"}
    REFERENCE = "dicom/ChemicalShiftReference"
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
        "EchoTime": (f"{PER_FRAME}/MREchoSequence/0/EffectiveEchoTime", "dicom/(2005,1310)"),
        "RepetitionTime": (
            f"{PER_FRAME}/MRTimingAndRelatedParametersSequence/0/RepetitionTime",
            "dicom/RepetitionTime",
        ),
        "ExcitationFlipAngle": (  # spec2nii reads the repetition time when per-frame data exist
            f"{PER_FRAME}/MRTimingAndRelatedParametersSequence/0/RepetitionTime",
            "dicom/FlipAngle",
        ),
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
        header = self.header(files[0])
        if converter == "philips_dicom":
            self._check_philips_classic(files)
        if converter == "philips_dicom" and self._is_philips_csi(header):
            return self._philips_csi(files[0], header)
        if converter == "dicom" and "MrProtocol" in header:
            images = [self._siemens_va(f, self.header(f)) for f in files]
            names = [str(header["dicom"].get("SeriesDescription") or f.stem) for f in files]
            return images, names, header
        function = {"dicom": siemens_dicom, "philips_dicom": philips_dicom, "uih": uih_dicom}
        with self._one_slab():
            images, names = function[converter](files, None, self.TAGS[converter], False)
        return list(images), list(names), header

    def sources(self, vendor: dict) -> dict[str, str | tuple[str, ...]]:
        manufacturer = str(vendor["dicom"].get("Manufacturer", "")).lower()
        if "siemens" in manufacturer:
            if "MrProtocol" in vendor:
                return self.SOURCES_SIEMENS_VA
            return self.SOURCES_SIEMENS_VX if "csa_image" in vendor else self.SOURCES_SIEMENS_XA
        if "philips" in manufacturer:
            enhanced = vendor["file_meta"].get("MediaStorageSOPClassUID") == self.ENHANCED_MRS
            return self.SOURCES_PHILIPS_ENHANCED if enhanced else self.SOURCES_PHILIPS_CLASSIC
        return self.SOURCES_UIH

    def correct(self, d: MRSData) -> None:
        """Units spec2nii leaves in ms, and the flip angle it reads from the wrong field."""
        super().correct(d)
        fields = d.meta.fields
        if fields.get("InversionTime") == "csa_image/InversionTime":
            self.fix(d, "InversionTime", d.meta["InversionTime"] * 1e-3, "CSA value in ms")
        if self.sources(d.meta.vendor) is not self.SOURCES_PHILIPS_CLASSIC:
            return
        if fields.get("EchoTime") == "dicom/(2005,1310)":
            self.fix(d, "EchoTime", d.meta["EchoTime"] * 1e-3, "private tag in ms")
        if fields.get("RepetitionTime") == "dicom/RepetitionTime":
            self.fix(d, "RepetitionTime", d.meta["RepetitionTime"] * 1e-3, "DICOM value in ms")
        if fields.get("ExcitationFlipAngle", "").endswith("/RepetitionTime"):
            field = fields["ExcitationFlipAngle"].removesuffix("RepetitionTime") + "FlipAngle"
            flip = d.meta.vendor_fields().get(field)
            if flip is None:
                field, flip = "dicom/FlipAngle", d.meta.vendor["dicom"].get("FlipAngle")
            d.meta.set("ExcitationFlipAngle", flip, f"{field}; spec2nii read RepetitionTime", field)

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
        """Complete header of one DICOM file: elements, CSA headers and Siemens protocol."""
        ds = pydicom.dcmread(path)
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
                header[name] = {
                    SiemensTwixReader.protocol_key(k): self.plain(v)
                    for k, v in parse_buffer(protocol).items()
                }
        return header

    @contextlib.contextmanager
    def _one_slab(self) -> Iterator[None]:
        """spec2nii's Philips enhanced geometry, with the one-slab case added."""
        original = philips_dcm._enhanced_dcm_svs_to_orientation

        def orientation(img, verbose=False):
            ds = img.dcm_data
            slabs = len(ds.get("VolumeLocalizationSequence", []))
            if not philips_dcm._is_new_format(img) or not 0 < slabs < 3:
                return original(img, verbose)
            iop = self._plane(ds, "PlaneOrientationSequence", "ImageOrientationPatient")
            iop = -np.asarray(iop)
            ipp = np.asarray(self._plane(ds, "PlanePositionSequence", "ImagePositionPatient"))
            spacing = self._plane(ds, "PixelMeasuresSequence", "PixelSpacing")
            thickness = float(ds.VolumeLocalizationSequence[0].SlabThickness)
            size = np.asarray([float(spacing[1]), float(spacing[0]), thickness])
            warnings.warn("one localisation slab: voxel from the plane position, orientation and "
                          "pixel spacing", stacklevel=2)
            return dcm_to_nifti_orientation(iop.reshape(2, 3), ipp, size, (1, 1, 1), verbose)

        philips_dcm._enhanced_dcm_svs_to_orientation = orientation
        try:
            yield
        finally:
            philips_dcm._enhanced_dcm_svs_to_orientation = original

    @staticmethod
    def _check_philips_classic(files: list[Path]) -> None:
        """Classic Philips files hold the spectrum in (2005,1270), (2005,1315) points of it."""
        for path in files:
            ds = pydicom.dcmread(path)
            if (0x2005, 0x1270) not in ds or (0x2005, 0x1315) not in ds:
                continue
            size, points = len(ds[0x2005, 0x1270].value), int(ds[0x2005, 0x1315].value)
            if size < 8 * points:
                raise ReaderError(
                    f"{path.name}: spectrum element (2005,1270) holds {size} bytes, "
                    f"less than one spectrum of {points} complex points"
                )

    @staticmethod
    def _is_philips_csi(header: dict) -> bool:
        dicom = header["dicom"]
        rows = int(dicom.get("SpectroscopyAcquisitionPhaseRows") or 1)
        columns = int(dicom.get("SpectroscopyAcquisitionPhaseColumns") or 1)
        return "SpectroscopyData" in dicom and rows * columns > 1

    def _philips_csi(self, path: Path, header: dict) -> tuple[list[NIFTI_MRS], list[str], dict]:
        img = wrapper_from_file(path)
        ds = img.dcm_data
        rows = int(ds.SpectroscopyAcquisitionPhaseRows)
        columns = int(ds.SpectroscopyAcquisitionPhaseColumns)
        points, frames = int(ds.SpectroscopyAcquisitionDataColumns), int(ds.NumberOfFrames)
        values = np.frombuffer(ds.SpectroscopyData, dtype="<f4")
        data = np.conj(values[0::2] + 1j * values[1::2]).reshape(frames, rows, columns, points)
        flags = [int(item[0x2005, 0x1304].value) for item in ds[0x2005, 0x140F].value]
        is_ref = np.asarray(flags).reshape(frames, rows * columns)[:, 0] == 1
        iop = np.asarray(self._plane(ds, "PlaneOrientationSequence", "ImageOrientationPatient"))
        ipp = np.asarray(self._plane(ds, "PlanePositionSequence", "ImagePositionPatient"))
        spacing = self._plane(ds, "PixelMeasuresSequence", "PixelSpacing")
        normal = np.cross(iop[:3], iop[3:])
        slab = max(ds.VolumeLocalizationSequence,
                   key=lambda s: abs(np.dot(np.asarray(s.SlabOrientation, float), normal)))
        size = np.asarray([spacing[1], spacing[0], float(slab.SlabThickness)])
        affine = dcm_to_nifti_orientation(iop.reshape(2, 3), ipp, size, (columns, rows, 1)).Q44
        dwelltime = 1 / float(ds.SpectralWidth)
        name = str(ds.get("SeriesDescription") or path.stem)
        images, names = [], []
        for reference, suffix in ((False, ""), (True, "_ref")):
            selected = data[is_ref == reference]
            if not selected.size:
                continue
            grid = selected.transpose(2, 1, 3, 0)[:, :, None]  # columns, rows, 1, points, frames
            grid = grid[..., 0] if grid.shape[-1] == 1 else grid
            meta = philips_dcm._extractDicomMetadata_new(img, water_suppressed=not reference)
            if grid.ndim == 5:
                meta.set_dim_info(0, "DIM_DYN")
            images.append(gen_nifti_mrs_hdr_ext(grid, dwelltime, meta, affine, no_conj=True))
            names.append(name + suffix)
        return images, names, header

    @staticmethod
    def _plane(ds: pydicom.Dataset, sequence: str, keyword: str) -> list[float]:
        """A value from the first frame's functional groups, else the shared ones."""
        for groups in ("PerFrameFunctionalGroupsSequence", "SharedFunctionalGroupsSequence"):
            group = ds.get(groups, [None])[0]
            if group is not None and sequence in group and keyword in group[sequence].value[0]:
                return [float(v) for v in group[sequence].value[0].get(keyword)]
        raise ReaderError(f"no {keyword} in the functional groups")

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
