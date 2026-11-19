####################################################################################################
#                                         uih_dicom_mrs.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 17/11/26                                                                                #
#                                                                                                  #
# Purpose: UIH DICOM spectroscopy (MR Spectroscopy Storage): spectra, voxel or grid geometry and   #
#          keys of one series.                                                                     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pydicom

# own
from miox import __version__
from miox.readers.philips_dicom_mrs import Spectrum
from miox.readers.siemens_rda import SiemensRdaReader
from miox.sniff import open_dicom


#**************************************************************************************************#
#                                           UihDicomMrs                                            #
#**************************************************************************************************#
#                                                                                                  #
# A single voxel file holds the spectrum in SpectroscopyData and its water reference in            #
# (0065,FF07); svs_mega scans also the OFF (0065,FF11) and ON (0065,FF10) sums, the spectrum then  #
# being their difference (_diff). A grid (more than one row or column) is (columns, rows, frames,  #
# points) in the file, rows and columns swapped on reading. Geometry from ImageOrientationPatient, #
# ImagePositionPatient, PixelSpacing and SpacingBetweenSlices (else SliceThickness), half a voxel  #
# in from the position in-plane; grids also half a voxel along the slice axis, which points the    #
# way the sign of (0065,FF06)'s third value says. Files of a series that give one dataset each of  #
# one shape, geometry and dwell time are stacked; otherwise each dataset stands alone (_000, ...). #
#                                                                                                  #
#**************************************************************************************************#
class UihDicomMrs:
    DATA = (0x5600, 0x0020)
    WATER = (0x0065, 0xFF07)
    MEGA_OFF, MEGA_ON = (0x0065, 0xFF11), (0x0065, 0xFF10)
    SLICE_SIGN = (0x0065, 0xFF06)
    SEQUENCE = (0x0018, 0x0024)

    def __init__(self):
        self.notes: list[str] = []

    def read(self, paths: list[Path]) -> list[Spectrum]:
        """The datasets of one series."""
        spectra, suffixes = [], []
        name = ""
        for index, path in enumerate(paths):
            ds = self.dataset(path)
            if int(ds.Rows) * int(ds.Columns) > 1:
                found, found_suffixes = [self._csi(ds, path)], []
            else:
                found, found_suffixes = self._svs(ds, path)
            spectra += found
            suffixes += found_suffixes
            if index == 0:
                name = str(ds.SeriesDescription)

        def same(items: list) -> bool:
            return items[:-1] == items[1:]

        combine = (same([s.data.shape for s in spectra]) and same([s.affine.tolist() for s in
                   spectra]) and same([s.dwelltime for s in spectra]) and same(suffixes))
        if combine:
            first = spectra[0]
            data = np.stack([s.data for s in spectra], axis=-1) if len(spectra) > 1 else first.data
            values = dict(first.values, OriginalFile=([p.name for p in paths], None))
            return [Spectrum(name, data, first.dwelltime, first.affine, values, first.tags)]
        for index, (spectrum, suffix) in enumerate(zip(spectra, suffixes, strict=False)):
            spectrum.name = f"{name}_{index:03}{suffix}" if len(paths) > 1 else name + suffix
        return spectra[: len(suffixes)]

    @staticmethod
    def dataset(path: Path) -> pydicom.Dataset:
        with open_dicom(path) as fh:
            return pydicom.dcmread(fh, force=True)

    @staticmethod
    def _complex(ds: pydicom.Dataset, tag: tuple[int, int]) -> np.ndarray:
        floats = np.frombuffer(ds[tag].value, dtype=np.single)
        return floats[0::2] + 1j * floats[1::2]

    def _svs(self, ds: pydicom.Dataset, path: Path) -> tuple[list[Spectrum], list[str]]:
        affine = self._affine(ds, 1)
        dwelltime = 1.0 / ds.SpectralWidth
        values = self.values(ds, path)
        main = Spectrum("", self._complex(ds, self.DATA), dwelltime, affine, values)
        water = Spectrum("", self._complex(ds, self.WATER), dwelltime, affine,
                         values | {"WaterSuppressed": (False, None)})
        spectra, suffixes = [main, water], ["", "_wref"]
        if ds[self.SEQUENCE].value == "svs_mega":
            suffixes[0] = "_diff"
            off_on = np.stack([self._complex(ds, self.MEGA_OFF), self._complex(ds, self.MEGA_ON)],
                              axis=-1)
            spectra.append(Spectrum("", off_on, dwelltime, affine,
                                    values | {"WaterSuppressed": (True, None)},
                                    (("DIM_EDIT", None, {"EditCondition": ["OFF", "ON"]}),)))
            suffixes.append("_off_on")
        for s in spectra:
            s.data = s.data.reshape((1, 1, 1) + s.data.shape)
        return spectra, suffixes

    def _csi(self, ds: pydicom.Dataset, path: Path) -> Spectrum:
        self.notes.append("The orientation of UIH MRSI is weakly tested")
        shape = (ds.Columns, ds.Rows, int(ds.NumberOfFrames),
                 ds.SpectroscopyAcquisitionDataColumns)
        data = np.swapaxes(self._complex(ds, self.DATA).reshape(shape), 0, 1)
        affine = self._affine(ds, data.shape[2])
        if float(ds[self.SLICE_SIGN].value[2]) < 0.0:
            affine[:3, 2] *= -1
        affine[:3, 3] += affine[:3, :3] @ [0.0, 0.0, 0.5]
        return Spectrum("", data, 1.0 / ds.SpectralWidth, affine, self.values(ds, path))

    @staticmethod
    def _affine(ds: pydicom.Dataset, slices: int) -> np.ndarray:
        thickness = ds.get("SpacingBetweenSlices")
        if thickness is None:
            thickness = ds.get("SliceThickness")
        if thickness is None or thickness == "":
            thickness = 1.0
        sizes = [float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]), float(thickness)]
        return SiemensRdaReader.grid_affine(np.asarray(ds.ImageOrientationPatient, dtype=float),
                                            ds.ImagePositionPatient, sizes, slices, True)

    def values(self, ds: pydicom.Dataset, path: Path) -> dict:
        """Standard keys: (value, element)."""
        values = {
            "SpectrometerFrequency": ([float(ds.TransmitterFrequency)],
                                      "dicom/TransmitterFrequency"),
            "ResonantNucleus": ([str(ds.ResonantNucleus)], "dicom/ResonantNucleus"),
            "EchoTime": (float(ds.EchoTime) * 1e-3, "dicom/EchoTime"),
            "RepetitionTime": (float(ds.RepetitionTime) / 1e3, "dicom/RepetitionTime"),
            "ExcitationFlipAngle": (float(ds.FlipAngle), "dicom/FlipAngle"),
            "Manufacturer": (str(ds.Manufacturer), "dicom/Manufacturer"),
            "ManufacturersModelName": (str(ds.ManufacturerModelName),
                                       "dicom/ManufacturerModelName"),
            "DeviceSerialNumber": (str(ds.DeviceSerialNumber), "dicom/DeviceSerialNumber"),
            "SoftwareVersions": (str(ds.SoftwareVersions), "dicom/SoftwareVersions"),
            "InstitutionName": (str(ds.InstitutionName), "dicom/InstitutionName"),
            "InstitutionAddress": (str(ds.InstitutionAddress), "dicom/InstitutionAddress"),
            "SequenceName": (str(ds.SequenceName), "dicom/SequenceName"),
            "ProtocolName": (str(ds.ProtocolName), "dicom/ProtocolName"),
            "PatientPosition": (str(ds.PatientPosition), "dicom/PatientPosition"),
            "ConversionMethod": (f"miox {__version__}", None),
            "ConversionTime": (datetime.now().isoformat(timespec="milliseconds"), None),
            "OriginalFile": ([path.name], None),
            "kSpace": ([False, False, False], None),
            "DwellTime": (1.0 / ds.SpectralWidth, "dicom/SpectralWidth"),
        }
        if float(ds.get("InversionTime") or 0.0):  # 0: no inversion
            values["InversionTime"] = (float(ds.InversionTime) * 1e-3, "dicom/InversionTime")
        if ds.get("PatientName"):
            values["PatientName"] = (str(ds.PatientName.family_name), "dicom/PatientName")
        if ds.get("PatientWeight"):
            values["PatientWeight"] = (float(ds.PatientWeight), "dicom/PatientWeight")
        for key, keyword in (("PatientDoB", "PatientBirthDate"), ("PatientSex", "PatientSex")):
            if ds.get(keyword):
                values[key] = (str(ds.data_element(keyword).value), f"dicom/{keyword}")
        return values
