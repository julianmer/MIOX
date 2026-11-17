####################################################################################################
#                                          siemens_rda.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for Siemens RDA spectroscopy exports. The vendor header keeps every line of the  #
#          text header.                                                                            #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

# own
from miox import __version__
from miox.data import MRSData
from miox.readers.base import Reader, ReaderError
from miox.readers.philips_dicom_mrs import PhilipsDicomMrs
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                         SiemensRdaReader                                         #
#**************************************************************************************************#
#                                                                                                  #
# Reads one RDA file: a text header of ``key: value`` lines (UTF-8, else Latin-1; decimal commas   #
# where the scanner's locale writes them), then complex pairs of float64, stored as NIfTI-MRS      #
# holds them. More than one CSIMatrixSize point is spectroscopic imaging: z, y, x, points in the   #
# file, the affine at PositionVector shifted by half a voxel in-plane and, across several slices,  #
# the slice axis turned to point along its main patient axis. A single voxel sits at its           #
# VOIPosition (else PositionVector). Directions from RowVector and ColumnVector, the voxel size    #
# from PixelSpacingCol along the rows, PixelSpacingRow along the columns and SliceThickness.       #
# Header values stay the strings written in the file.                                              #
#                                                                                                  #
#**************************************************************************************************#
class SiemensRdaReader(Reader):
    FORMATS = (Format.SIEMENS_RDA,)
    SOURCE = "RDA header"
    HEADER_START = ">>> Begin of header <<<"
    HEADER_END = ">>> End of header <<<"
    MS = 1e-3
    # standard key: (header field, type); every one the header states
    KEYS = {
        "EchoTime": ("TE", "ms"),
        "RepetitionTime": ("TR", "ms"),
        "InversionTime": ("TI", "ms"),
        "MixingTime": ("TM", "ms"),
        "ExcitationFlipAngle": ("FlipAngle", float),
        "ManufacturersModelName": ("ModelName", str),
        "DeviceSerialNumber": ("DeviceSerialNumber", str),
        "SoftwareVersions": ("SoftwareVersion[0]", str),
        "InstitutionName": ("InstitutionName", str),
        "TxCoil": ("TransmitCoil", str),
        "SequenceName": ("SequenceName", str),
        "ProtocolName": ("ProtocolName", str),
        "PatientPosition": ("PatientPosition", str),
        "PatientName": ("PatientName", str),
        "PatientID": ("PatientID", str),
        "PatientWeight": ("PatientWeight", float),
        "PatientDoB": ("PatientBirthDate", str),
        "PatientSex": ("PatientSex", str),
        "StudyDescription": ("StudyDescription", str),
        "SeriesDescription": ("SeriesDescription", str),
        "SeriesNumber": ("SeriesNumber", int),
        "StationName": ("StationName", str),
        "AcquisitionNumber": ("AcquisitionNumber", int),
        "EchoNumber": ("EchoNumber", int),
        "FieldStrength": ("MagneticFieldStrength", float),
        "TxRefAmp": ("TransmitRefAmplitude[1H]", float),
        "SliceThickness": ("SliceThickness", float),
    }

    def read(self, sniff: Sniff) -> list[MRSData]:
        header, stored = self.read_rda(sniff.path)
        notes: list[str] = []
        try:
            grid = [int(header[f"CSIMatrixSize[{i}]"]) for i in range(3)]
            points = int(header["VectorSize"])
            dwelltime = self.number(header["DwellTime"]) / 1e6
            frequency = self.number(header["MRFrequency"])
            nucleus = header["Nucleus"]
        except (KeyError, ValueError) as e:
            raise ReaderError(f"{sniff.path.name}: header incomplete ({e})") from e
        size = int(np.prod(grid)) * points * 16  # complex pairs of float64
        if len(stored) < size:
            raise ReaderError(f"{sniff.path.name}: {len(stored)} bytes of data, the header states "
                              f"{grid} x {points} points")
        if len(stored) > size:
            notes.append(f"{len(stored) - size} bytes after the data left out")
        floats = np.frombuffer(stored, dtype="<f8", count=size // 8)
        data = floats[0::2] + 1j * floats[1::2]
        imaging = np.prod(grid) > 1
        if imaging:  # z, y, x, points in the file
            data = data.reshape(grid[2], grid[1], grid[0], points).transpose(2, 1, 0, 3)
        else:
            data = data.reshape(1, 1, 1, points)
        values = {
            "SpectrometerFrequency": ([frequency], "MRFrequency"),
            "ResonantNucleus": ([nucleus], "Nucleus"),
            "DwellTime": (dwelltime, "DwellTime"),
            "Manufacturer": ("Siemens", None),
            "kSpace": ([False, False, False], None),
            "ConversionMethod": (f"miox {__version__}", None),
            "ConversionTime": (datetime.now().isoformat(timespec="milliseconds"), None),
            "OriginalFile": ([sniff.path.name], None),
        }
        values |= self._values(header, notes)
        affine = self.affine(header, data.shape[:3], imaging)
        return [self.dataset(sniff, data, dwelltime, values, header, sniff.path.stem, notes,
                             self.SOURCE, affine)]

    # --- file ------------------------------------------------------------------------------------

    @classmethod
    def read_rda(cls, path: Path) -> tuple[dict[str, str], bytes]:
        """The header's ``key: value`` lines and the bytes after it."""
        header = {}
        with open(path, "rb") as fh:
            for raw in fh:
                try:
                    line = raw.decode("utf-8")
                except UnicodeDecodeError:
                    line = raw.decode("latin-1")
                line = line.rstrip("\r\n")
                if line.startswith(cls.HEADER_START):
                    continue
                if line.startswith(cls.HEADER_END):
                    break
                key, _, value = line.partition(": ")
                header[key] = value
            stored = fh.read()
        return header, stored

    @staticmethod
    def number(text: str) -> float:
        """A header number; a decimal comma where the scanner's locale writes one."""
        return float(text.replace(",", "."))

    # --- keys ------------------------------------------------------------------------------------

    def _values(self, header: dict[str, str], notes: list[str]) -> dict:
        """The KEYS the header states: (value, header field)."""
        values = {}
        for key, (field, kind) in self.KEYS.items():
            text = header.get(field, "").strip()
            if not text:
                continue
            try:
                if kind == "ms":
                    value = self.number(text) * self.MS
                elif kind is str:
                    value = text
                else:
                    value = kind(self.number(text))
            except ValueError:
                notes.append(f"{field} '{text}' is not a number: left out")
                continue
            values[key] = (value, field)
        return values

    # --- geometry --------------------------------------------------------------------------------

    def affine(self, header: dict[str, str], shape: tuple[int, ...], imaging: bool) -> np.ndarray:
        """NIfTI affine of the grid or the single voxel."""
        fields = [f"PositionVector[{i}]" for i in range(3)]
        voi = ("VOIPositionSag", "VOIPositionCor", "VOIPositionTra")
        if not imaging and all(f in header for f in voi):
            fields = list(voi)
        position = [self.number(header[f]) for f in fields]
        orientation = [self.number(header[f"{vector}[{i}]"])
                       for vector in ("RowVector", "ColumnVector") for i in range(3)]
        sizes = [self.number(header[f]) for f in ("PixelSpacingRow", "PixelSpacingCol",
                                                  "SliceThickness")]
        affine = PhilipsDicomMrs.affine(orientation, position, sizes)
        if shape[2] > 1:  # the slice axis along its main patient axis, as dcm2niix orders slices
            slice_axis = affine[:3, 2] * [-1, -1, 1]  # in the DICOM frame
            main = 2 - int(np.argmax(np.abs(slice_axis)[::-1]))  # ties to the later axis
            if slice_axis[main] < 0:
                affine[:3, 2] *= -1
        if imaging:  # half a voxel in-plane from PositionVector
            affine[:3, 3] += affine[:3, :3] @ [0.5, 0.5, 0.0]
        return affine
