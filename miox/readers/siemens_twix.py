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

import re
import warnings

import numpy as np
import scipy.fft
import spec2nii.GSL.gslfunctions as GSL
from mapvbvd import mapVBVD
from nifti_mrs.create_nmrs import gen_nifti_mrs_hdr_ext
from nifti_mrs.nifti_mrs import NIFTI_MRS
from spec2nii.Siemens.twixfunctions import id_sequence_type, process_twix, xa_or_vx

# own
from miox.data import MRSData, RawData
from miox.readers.base import ReaderError, Spec2niiReader
from miox.readers.siemens_twix_raw import SiemensTwixRaw
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                        SiemensTwixReader                                         #
#**************************************************************************************************#
#                                                                                                  #
# A spectroscopy measurement (the protocol states a spectral vector size) is converted by          #
# spec2nii, every data stream (evalinfo flag): ``image`` holds the scan itself; the others (water  #
# reference, noise, phase correction, ...) become further datasets named after their flag, and a   #
# stream spec2nii cannot convert is noted instead of failing the read. Any other measurement is    #
# imaging raw data: its streams become RawData with their scan headers (SiemensTwixRaw). Header    #
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

    def read(self, sniff: Sniff) -> list[MRSData | RawData]:
        notes: list[str] = []
        with self.captured(notes):
            twix = self._measurement(sniff)
        if self.is_spectroscopy(twix):
            return super().read(sniff)
        streams = [key for key in twix.keys() if key != "hdr"]
        if self.MAIN_STREAM not in streams:
            raise ReaderError(f"no '{self.MAIN_STREAM}' data in {sniff.path.name}: {streams}")
        provenance = self.provenance(sniff, tuple(notes))
        raw = SiemensTwixRaw(sniff.path, twix, self._vendor(twix, streams), provenance)
        return raw.datasets([self.MAIN_STREAM] + [s for s in streams if s != self.MAIN_STREAM])

    def _measurement(self, sniff: Sniff):
        twix = mapVBVD(str(sniff.path), quiet=True)
        if isinstance(twix, list):
            twix = twix[min(self.MULTIRAID, len(twix)) - 1]
        return twix

    @staticmethod
    def is_spectroscopy(twix) -> bool:
        """A spectral vector size in the protocol marks a spectroscopy measurement."""
        value = twix["hdr"]["Phoenix"].get(("sSpecPara", "lVectorSize"))
        return isinstance(value, (int, float)) and value > 0

    def convert(self, sniff: Sniff) -> tuple[list[NIFTI_MRS], list[str], dict]:
        twix = self._measurement(sniff)
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
        images, names = list(images), list(names)
        if stream == self.MAIN_STREAM and id_sequence_type(twix["hdr"]) == "mrsi":
            # spec2nii leaves MRSI empty: the data, reconstructed, in its geometry
            csi = TwixCsi(twix)
            if not csi.verified:
                warnings.warn(TwixCsi.UNVERIFIED, stacklevel=2)
                return images, names
            empty = images[0]
            empty.hdr_ext.set_dim_info(0, "DIM_COIL")
            images[0] = gen_nifti_mrs_hdr_ext(csi.reconstruct(), empty.dwelltime, empty.hdr_ext,
                                              empty.getAffine("voxel", "world"))
            names[0] = names[0].removesuffix("_empty")
            warnings.warn(TwixCsi.NOTE, stacklevel=2)
        return images, names

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


#**************************************************************************************************#
#                                             TwixCsi                                              #
#**************************************************************************************************#
#                                                                                                  #
# Cartesian CSI from the k-space of a twix file, as the scanner reconstructs it (checked against   #
# its RDA of the same scans: VD and later, 2D and 3D transverse; VB and 3D sagittal or coronal     #
# grids do not match yet and stay empty): the phase encodes placed by their counters (Lin phase,   #
# Seg readout, Par partitions), the middle lVectorSize points after the oversampling is removed, a #
# Hamming filter over the k-space, zero filling to the final matrix and the spatial Fourier        #
# transform; then spec2nii's geometry (the readout mirrored for sagittal and transverse slices,    #
# read and phase swapped for transverse ones). Coils are kept (x, y, z, points, coils), as the     #
# FIDs stand before NIfTI-MRS conjugation.                                                         #
#                                                                                                  #
#**************************************************************************************************#
class TwixCsi:
    NOTE = ("MRSI reconstructed by MIOX (spec2nii leaves it empty): Hamming filter, zero filling "
            "to the final matrix, spatial FFT, coils kept")
    UNVERIFIED = ("MRSI not reconstructed: spec2nii leaves it empty (zeros), and MIOX reconstructs "
                  "only what it has checked against the scanner (VD and later; 2D, 3D transverse)")
    SAGITTAL, CORONAL, TRANSVERSE = 0, 1, 2  # spec2nii's main orientations

    def __init__(self, twix):
        self.twix = twix
        self.protocol = twix["hdr"]["MeasYaps"]

    @staticmethod
    def hamming(n: int) -> np.ndarray:
        return 0.54 + 0.46 * np.cos(2 * np.pi * (np.arange(n) - n // 2) / n)

    @property
    def verified(self) -> bool:
        """Checked against the scanner: VD and later, 2D, and 3D with transverse slices."""
        version = str(self.twix["hdr"]["Dicom"].get("SoftwareVersions", ""))  # "syngo MR B17"
        vb = re.search(r"\bB\d", version) is not None
        partitions = int(self.protocol.get(("sKSpace", "lPartitions"), 1) or 1)
        return not vb and (partitions == 1 or self.orientation() == self.TRANSVERSE)

    def orientation(self) -> int:
        normal = [float(self.protocol.get(("sSpecPara", "sVoI", "sNormal", d), 0.0) or 0.0)
                  for d in ("dSag", "dCor", "dTra")]
        return GSL.class_ori(*normal, False)

    def kspace(self) -> np.ndarray:
        """(coils, phase, readout, partition, points): Lin, Seg and Par as stored (the encodes
        centred in their counters), other counters averaged."""
        image = self.twix["image"]
        image.flagRemoveOS, image.squeeze = True, False
        data, dims = image[""], list(image.dataDims)
        encodes = ["Cha", "Lin", "Seg", "Par"]
        rest = [d for d in dims if d not in encodes + ["Col"]]
        data = np.transpose(data, [dims.index(d) for d in encodes + rest + ["Col"]])
        data = data.reshape(data.shape[:4] + (-1, data.shape[-1])).mean(axis=4)
        points = int(self.protocol[("sSpecPara", "lVectorSize")])
        start = (data.shape[-1] - points) // 2  # the ADC holds a few points more on both sides
        return data[..., start:start + points]

    def reconstruct(self) -> np.ndarray:
        k = self.kspace()
        for axis in (1, 2, 3):
            shape = [1] * k.ndim
            shape[axis] = k.shape[axis]
            k = k * self.hamming(k.shape[axis]).reshape(shape)
        final = [max(1, int(self.protocol.get(("sSpecPara", f"lFinalMatrixSize{d}"), 1) or 1))
                 for d in ("Phase", "Read", "Slice")]
        grid = np.zeros((k.shape[0], *final, k.shape[-1]), dtype=np.complex128)
        grid[(slice(None),) + tuple(slice(m // 2 - n // 2, m // 2 - n // 2 + n)
                                    for n, m in zip(k.shape[1:4], final, strict=True))] = k
        axes = (1, 2, 3)
        image = np.fft.fftshift(scipy.fft.fftn(np.fft.ifftshift(grid, axes=axes), axes=axes,
                                               workers=-1), axes=axes)
        orientation = self.orientation()
        if orientation in (self.SAGITTAL, self.TRANSVERSE):
            image = np.flip(image, 2)  # the readout mirrored
        if orientation == self.TRANSVERSE:
            image = np.swapaxes(image, 1, 2)  # read before phase
        return np.moveaxis(image, 0, -1)  # (x, y, z, points, coils)
