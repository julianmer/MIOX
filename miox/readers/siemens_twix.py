####################################################################################################
#                                         siemens_twix.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for Siemens twix raw data (.dat, VB to XA). pyMapVBVD parses the file, MIOX      #
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
from mapvbvd import mapVBVD

# own
from miox.data import MRSData, RawData
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.readers.dicom_derived import DerivedKeys
from miox.readers.siemens_twix_mrs import SiemensGsl, TwixMrs, TwixSpectrum
from miox.readers.siemens_twix_raw import SiemensTwixRaw
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                        SiemensTwixReader                                         #
#**************************************************************************************************#
#                                                                                                  #
# pyMapVBVD parses the file. A spectroscopy measurement (the protocol states a spectral vector     #
# size) is laid out by TwixMrs, every data stream (evalinfo flag): ``image`` holds the scan        #
# itself; the others (water reference, noise, phase correction, ...) become further datasets named #
# after their flag, and a stream that cannot be laid out is noted instead of failing the read.     #
# Grids are reconstructed where TwixCsi is checked against the scanner, else left empty            #
# (``_empty``). Any other measurement is imaging raw data: its streams become RawData with their   #
# scan headers (SiemensTwixRaw). Header keys of the protocol buffers are written in Siemens        #
# notation, e.g. ``sTXSPEC.asNucleusInfo[0].tNucleus``.                                            #
#                                                                                                  #
#**************************************************************************************************#
class SiemensTwixReader(Reader):
    FORMATS = (Format.SIEMENS_TWIX,)
    PACKAGES = Reader.PACKAGES + ("pymapvbvd",)
    SOURCE = "twix header"
    MAIN_STREAM = "image"
    MULTIRAID = 2  # in a multi-measurement file the second holds the scan
    # standard key: (header fields, the first one found; type)
    HEADER_KEYS = {
        "FieldStrength": (("Dicom/flMagneticFieldStrength",), float),
        "TxRefAmp": (("Phoenix/sTXSPEC.asNucleusInfo[0].flReferenceAmplitude",), float),
        "ScanDuration": (("Meas/lTotalScanTimeSec",), float),
        "Modality": (("Dicom/Modality",), str),
        "BodyPart": (("Dicom/tBodyPartExamined",), str),
        "ScanningSequence": (("Dicom/tScanningSequence",), str),
        "SequenceVariant": (("Dicom/tSequenceVariant",), str),
        "MRAcquisitionType": (("Dicom/tMRAcquisitionType",), str),
        "ScanOptions": (("Dicom/tScanOptions",), str),
        "EchoTrainLength": (("Dicom/EchoTrainLength",), int),
        "ConsistencyInfo": (("Phoenix/sProtConsistencyInfo.tMeasuredBaselineString",
                             "Phoenix/sProtConsistencyInfo.tBaselineString"), str),
    }
    RF_PULSE = re.compile(r"Phoenix/sTXSPEC\.aRFPULSE\[(\d+)\]\.tName")

    def read(self, sniff: Sniff) -> list[MRSData | RawData]:
        notes: list[str] = []
        with self.captured(notes):
            twix = self._measurement(sniff)
        streams = [key for key in twix.keys() if key != "hdr"]
        if self.MAIN_STREAM not in streams:
            raise ReaderError(f"no '{self.MAIN_STREAM}' data in {sniff.path.name}: {streams}")
        vendor = self._vendor(twix, streams)
        if not self.is_spectroscopy(twix):
            provenance = self.provenance(sniff, tuple(notes))
            raw = SiemensTwixRaw(sniff.path, twix, vendor, provenance)
            return raw.datasets([self.MAIN_STREAM] + [s for s in streams if s != self.MAIN_STREAM])
        mrs = TwixMrs(twix, sniff.path)
        with self.captured(notes):
            spectra = self._main(mrs, twix)
            for stream in streams:
                if stream == self.MAIN_STREAM:
                    continue
                try:
                    spectra += mrs.spectra(stream)
                except Exception as e:  # noqa: BLE001  a stream the layout does not fit
                    warnings.warn(f"stream '{stream}' not converted: {type(e).__name__}: {e}",
                                  stacklevel=2)
        notes += mrs.notes
        header = self._header_values(Meta(vendor).vendor_fields())
        datasets = []
        for s in spectra:
            spectrum_notes = notes + s.notes
            data, tags = self.without_zero_indices(s.data, s.tags, spectrum_notes)
            d = self.dataset(sniff, data, s.dwelltime, s.values | header, vendor, s.name,
                             spectrum_notes, self.SOURCE, s.affine, tags)
            self._protocol_keys(d)
            datasets.append(d)
        return datasets

    def _main(self, mrs: TwixMrs, twix) -> list[TwixSpectrum]:
        """The image stream; a grid reconstructed where TwixCsi is checked."""
        spectra = mrs.spectra(self.MAIN_STREAM)
        if mrs.kind != "mrsi":
            return spectra
        csi = TwixCsi(twix)
        if not csi.verified:
            warnings.warn(TwixCsi.UNVERIFIED, stacklevel=2)
            return spectra
        empty = spectra[0]
        warnings.warn(TwixCsi.NOTE, stacklevel=2)
        return [TwixSpectrum(empty.name.removesuffix("_empty"), csi.reconstruct().conj(),
                             empty.dwelltime, empty.affine, empty.values, ("DIM_COIL",))]

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

    def _header_values(self, fields: dict) -> dict:
        """The HEADER_KEYS the header states: (value, field)."""
        values = {}
        for key, (alternatives, kind) in self.HEADER_KEYS.items():
            field = next((f for f in alternatives if fields.get(f) not in (None, "", [])), None)
            if field is None:
                continue
            value = fields[field]
            if kind is str:
                value = str(value).strip().strip('"')
            else:
                try:
                    value = kind(float(str(value).replace(",", ".")))
                except ValueError:
                    continue
            values[key] = (value, field)
        return values

    def _protocol_keys(self, d: MRSData) -> None:
        """Shim setting and RF pulse names from the protocol."""
        fields = d.meta.vendor_fields()
        shim = [next((fields[f"Phoenix/{f}"] for f in options if f"Phoenix/{f}" in fields), 0.0)
                for options in DerivedKeys.SIEMENS_SHIM]
        if all(isinstance(v, (int, float)) for v in shim) and shim[0]:
            shim = [int(v) if float(v).is_integer() else v for v in shim]
            d.meta.set("ShimSetting", shim, "twix header: Phoenix sGRADSPEC offsets and shim "
                       "currents", "Phoenix/sGRADSPEC.asGPAData[0].lOffsetX")
        pulses = {int(m[1]): str(v).strip('"') for f, v in fields.items()
                  if (m := self.RF_PULSE.fullmatch(f))}
        if pulses:
            d.meta.set("RFPulseName", [pulses[i] for i in sorted(pulses)],
                       "twix header: Phoenix/sTXSPEC.aRFPULSE[n].tName", "Phoenix/sTXSPEC.aRFPULSE")

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
# transform; then the grid order of TwixMrs (the readout mirrored for sagittal and transverse      #
# slices, read and phase swapped for transverse ones). Coils are kept (x, y, z, points, coils), as #
# the FIDs stand before NIfTI-MRS conjugation.                                                     #
#                                                                                                  #
#**************************************************************************************************#
class TwixCsi:
    NOTE = ("MRSI reconstructed by MIOX: Hamming filter, zero filling to the final matrix, "
            "spatial FFT, coils kept")
    UNVERIFIED = ("MRSI not reconstructed: the grid stays empty (zeros); MIOX reconstructs only "
                  "what it has checked against the scanner (VD and later; 2D, 3D transverse)")
    SAGITTAL, CORONAL, TRANSVERSE = SiemensGsl.SAGITTAL, SiemensGsl.CORONAL, SiemensGsl.TRANSVERSE

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
        return SiemensGsl.orientation(normal)

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
