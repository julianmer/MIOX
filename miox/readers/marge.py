####################################################################################################
#                                             marge.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 10/11/26                                                                                #
#                                                                                                  #
# Purpose: Readers for MaRGE (MaRCoS Graphical Environment) acquisitions: the .mat file a sequence #
#          saves (RARE and Localizer k-space lines with the stored reconstruction, Noise readouts) #
#          and the .csv of its parameters beside it.                                               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import ast
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

# own
from miox.data import MRImage, Protocol, RawData
from miox.matfile import MatFile
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.readers.ismrmrd import IsmrmrdReader
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                           MargeReader                                            #
#**************************************************************************************************#
#                                                                                                  #
# The .mat holds every parameter of the sequence (ms, us, cm for the FOV, metres for its offset    #
# dfov, MHz for the Larmor frequency and the RARE bandwidth, kHz for the Noise bandwidth) beside   #
# the data. A RARE or Localizer stores dataFull (scans, slices, phase lines, readout points): the  #
# ADC stream decimated by the oversampling factor, sorted into k-space order by sweepOrder and     #
# cropped to the readout matrix around the echo; every line becomes one readout with the scan as   #
# its average and the phase and slice encodes as its counters. kSpace3D is the mean over scans     #
# times exp(-2 pi i k . dfov) and image3D its centred inverse 3-D FFT, both checked against the    #
# stored arrays; image3D becomes an MRImage with the voxels placed on the scanner's x, y, z axes   #
# by axesOrientation. The oversampled stream, the noise and the dummy readouts stay in the vendor  #
# header as shapes. A Noise acquisition stores one readout of complex samples, flagged as a noise  #
# measurement. The echo time of a RARE is the echo of the train that holds the central phase line  #
# (sweepOrder) times the echo spacing. The .csv names every parameter; its names join the vendor   #
# header when it lies beside the .mat.                                                             #
#                                                                                                  #
#**************************************************************************************************#
class MargeReader(Reader):
    FORMATS = (Format.MARGE_MAT,)
    PACKAGES = Reader.PACKAGES + ("mat-io",)
    SOURCE = "MaRGE"
    GAMMA_H1 = 42.577478518  # MHz/T
    NUCLEUS = "1H"
    NOISE_FLAG = 1 << (IsmrmrdReader.NOISE_BIT - 1)

    def read(self, sniff: Sniff) -> list[RawData | MRImage]:
        values = MatFile.load(sniff.path)
        if "seqName" not in values:
            raise ReaderError(f"{sniff.path.name}: no seqName")
        vendor = MatFile.header(values)
        tables = [p for p in sniff.companions if p.suffix.lower() == ".csv" and p.is_file()]
        if tables:
            vendor["parameter_names"] = self.table(tables[0])[0]
        if "kSpace3D" in values:
            return self._imaging(sniff, values, vendor)
        if "data" in values:
            return [self._noise(sniff, values, vendor)]
        raise ReaderError(f"{sniff.path.name}: {values['seqName']} holds neither kSpace3D nor data")

    @staticmethod
    def table(path: Path) -> tuple[dict[str, str], dict]:
        """The parameter names and values of a MaRGE .csv: a row of keys, a row of names, a row
        of values (lists and numbers as Python writes them)."""
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            rows = [row for row in csv.reader(fh) if row]
        if len(rows) < 3:
            raise ReaderError(f"{path.name}: {len(rows)} rows, 3 expected (keys, names, values)")
        keys, names, texts = rows[0], rows[1], rows[2]
        values = {}
        for key, text in zip(keys, texts, strict=False):
            try:
                values[key] = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                values[key] = text
        return dict(zip(keys, names, strict=False)), values

    def _imaging(self, sniff: Sniff, values: dict, vendor: dict) -> list[RawData | MRImage]:
        points = self._points(values)
        if len(points) != 3:
            raise ReaderError(f"{sniff.path.name}: nPoints {points} is not (readout, phase, slice)")
        nrd, nph, nsl = points
        scans = int(self._number(values, "nScans"))
        lines = np.asarray(values["dataFull"]).reshape(scans, nsl, nph, nrd)
        data = np.ascontiguousarray(lines.reshape(scans * nsl * nph, 1, nrd))
        headers = np.zeros(data.shape[0], dtype=RawData.HEADER)
        scan, sl, ph = np.meshgrid(np.arange(scans), np.arange(nsl), np.arange(nph), indexing="ij")
        headers["idx"]["average"] = scan.ravel()
        headers["idx"]["kspace_encode_step_2"] = sl.ravel()
        headers["idx"]["kspace_encode_step_1"] = ph.ravel()
        headers["scan_counter"] = np.arange(data.shape[0])
        headers["number_of_samples"] = nrd
        headers["available_channels"] = headers["active_channels"] = 1
        headers["center_sample"] = nrd // 2
        headers["sample_time_us"] = 1 / self._number(values, "bw")  # bw in MHz
        axes = [int(v) for v in np.ravel(values["axesOrientation"])]
        for name, axis in zip(("read_dir", "phase_dir", "slice_dir"), axes, strict=True):
            headers[name][:, axis] = 1.0
        headers["position"] = self._dfov_mm(values)
        fov = self._fov_mm(values, axes)
        encoding = [{
            "trajectory": "cartesian",
            "encodedSpace": {"matrixSize": dict(zip("xyz", points, strict=True)),
                             "fieldOfView_mm": dict(zip("xyz", fov, strict=True))},
            "reconSpace": {"matrixSize": dict(zip("xyz", points, strict=True)),
                           "fieldOfView_mm": dict(zip("xyz", fov, strict=True))},
            "encodingLimits": self._limits(points, scans),
            "echoTrainLength": int(self._number(values, "etl")),
        }]
        notes = (
            "readouts as MaRGE decimated, sorted and cropped them (dataFull), one per scan; "
            "kSpace3D is their mean over scans times exp(-2 pi i k . dfov)",
            "the oversampled ADC stream (overData), the noise and the dummy readouts stay in the "
            "vendor header as shapes",
        )
        raw = RawData(
            data, headers, self._meta(values, dict(vendor)), self.provenance(sniff, notes),
            sniff.path.stem, None, None, encoding, [], IsmrmrdReader.ACQUISITION_FLAGS,
        )
        return [raw, self._image(sniff, values, vendor, points, axes)]

    def _image(self, sniff: Sniff, values: dict, vendor: dict, points: list[int],
               axes: list[int]) -> MRImage:
        nrd, nph, nsl = points
        volume = np.asarray(values["image3D"]).reshape(nsl, nph, nrd)
        xyz = np.transpose(np.transpose(volume, (2, 1, 0)), np.argsort(axes))
        voxel = np.ones(3)
        size = np.ones(3, dtype=int)
        for axis, extent, count in zip(axes, self._fov_mm(values, axes), points, strict=True):
            voxel[axis] = extent / count
            size[axis] = count
        affine = np.diag([*voxel, 1.0])
        affine[:3, 3] = -(size // 2) * voxel + self._dfov_mm(values)
        meta = self._meta(values, dict(vendor))
        meta.set("ImageType", ["COMPLEX"], f"{self.SOURCE}: image3D is complex", "image3D")
        notes = (
            "image3D as stored: the centred inverse 3-D FFT of kSpace3D",
            "voxels placed on the scanner's x, y, z by axesOrientation, the volume centred at "
            "dfov; the axis directions are not verified against MaRGE's DICOM export",
        )
        name = f"{sniff.path.stem}/image3D"
        return MRImage(xyz, affine, meta, self.provenance(sniff, notes), name)

    def _noise(self, sniff: Sniff, values: dict, vendor: dict) -> RawData:
        samples = np.asarray(values["data"]).ravel()
        data = np.ascontiguousarray(samples.reshape(1, 1, -1))
        headers = np.zeros(1, dtype=RawData.HEADER)
        headers["flags"] = self.NOISE_FLAG
        headers["number_of_samples"] = samples.size
        headers["available_channels"] = headers["active_channels"] = 1
        headers["sample_time_us"] = 1e3 / self._number(values, "bw")  # bw in kHz
        notes = ("one noise readout, no encoding",)
        return RawData(
            data, headers, self._meta(values, vendor), self.provenance(sniff, notes),
            sniff.path.stem, None, None, {}, [], IsmrmrdReader.ACQUISITION_FLAGS,
        )

    def _meta(self, values: dict, vendor: dict) -> Meta:
        """The standard keys a MaRGE parameter set states (.mat values or the .csv row)."""
        meta = Meta(vendor)
        noise = str(values.get("seqName", "")) == "Noise"

        def put(key: str, value, how: str, field: str) -> None:
            meta.set(key, value, f"{self.SOURCE}: {how}", field)

        if values.get("seqName"):
            put("SequenceName", str(values["seqName"]), "seqName", "seqName")
        if "repetitionTime" in values:
            put("RepetitionTime", self._number(values, "repetitionTime") * 1e-3,
                "repetitionTime (ms)", "repetitionTime")
        if "inversionTime" in values and self._number(values, "inversionTime") > 0:
            put("InversionTime", self._number(values, "inversionTime") * 1e-3,
                "inversionTime (ms)", "inversionTime")
        if "rfExFA" in values:
            put("ExcitationFlipAngle", self._number(values, "rfExFA"), "rfExFA (degrees)",
                "rfExFA")
        if "etl" in values:
            put("EchoTrainLength", int(self._number(values, "etl")), "etl", "etl")
        if "nScans" in values:
            put("NumberOfAverages", int(self._number(values, "nScans")), "nScans", "nScans")
        if "larmorFreq" in values:
            offset = self._number(values, "freqOffset") * 1e-3 if "freqOffset" in values else 0.0
            frequency = self._number(values, "larmorFreq") + offset
            put("SpectrometerFrequency", [frequency], "larmorFreq (MHz) + freqOffset (kHz)",
                "larmorFreq")
            put("FieldStrength", frequency / self.GAMMA_H1,
                f"SpectrometerFrequency / {self.GAMMA_H1} MHz/T", "larmorFreq")
            meta.set("ResonantNucleus", [self.NUCLEUS], f"assumed: {self.NUCLEUS}")
        if "bw" in values:
            unit, scale = ("kHz", 1e3) if noise else ("MHz", 1e6)
            put("DwellTime", 1 / (self._number(values, "bw") * scale), f"1 / bw ({unit})", "bw")
        points = self._points(values) if "nPoints" in values else []
        if points:
            put("NumberOfPoints", points[0], "nPoints[0] (readout)", "nPoints")
        if len(points) == 3:
            put("EncodedMatrixSize", points, "nPoints (readout, phase, slice)", "nPoints")
            put("ReconMatrixSize", points, "nPoints: image3D has the encoded size", "nPoints")
            scans = int(self._number(values, "nScans")) if "nScans" in values else 1
            put("EncodingLimits", self._limits(points, scans), "nPoints and nScans", "nPoints")
            put("KSpaceTrajectory", "cartesian", "RARE lines on a Cartesian grid", "nPoints")
            if "fov" in values and "axesOrientation" in values:
                axes = [int(v) for v in np.ravel(values["axesOrientation"])]
                fov = self._fov_mm(values, axes)
                put("EncodedFieldOfView", fov, "fov (cm) along axesOrientation, in mm", "fov")
                put("ReconFieldOfView", fov, "fov (cm) along axesOrientation, in mm", "fov")
            if {"echoSpacing", "sweepOrder", "etl"} <= set(values):
                order = [int(v) for v in np.ravel(values["sweepOrder"])]
                if points[1] // 2 in order:
                    echo = order.index(points[1] // 2) % int(self._number(values, "etl")) + 1
                    put("EchoTime", self._number(values, "echoSpacing") * echo * 1e-3,
                        f"echoSpacing (ms) x echo {echo}, which holds the central phase line "
                        "(sweepOrder)", "echoSpacing")
        put("NumberOfReceiveChannels", 1, "one receiver", "rxChannel")
        if "Scan Time (s)" in values:
            put("ScanDuration", self._number(values, "Scan Time (s)"), "Scan Time (s)",
                "Scan Time (s)")
        if values.get("name_string"):
            when = datetime.strptime(str(values["name_string"]), "%Y.%m.%d.%H.%M.%S.%f")
            put("AcquisitionDateTime", when.isoformat(timespec="milliseconds"), "name_string",
                "name_string")
        return meta

    @staticmethod
    def _number(values: dict, key: str) -> float:
        return float(np.ravel(values[key])[0])

    @staticmethod
    def _points(values: dict) -> list[int]:
        return [int(v) for v in np.ravel(values["nPoints"])]

    @staticmethod
    def _fov_mm(values: dict, axes: list[int]) -> list[float]:
        """The field of view along readout, phase and slice in mm: fov (cm) is stored along
        the scanner's x, y, z and axesOrientation names the scanner axis of each."""
        fov = [float(v) * 10 for v in np.ravel(values["fov"])]
        return [fov[axis] for axis in axes]

    @staticmethod
    def _dfov_mm(values: dict) -> np.ndarray:
        """The offset of the field of view along the scanner's x, y, z in mm (dfov in metres)."""
        if "dfov" not in values:
            return np.zeros(3)
        return np.asarray([float(v) * 1e3 for v in np.ravel(values["dfov"])])

    @staticmethod
    def _limits(points: list[int], scans: int) -> dict:
        def limit(count: int) -> dict:
            return {"minimum": 0, "maximum": count - 1, "center": count // 2}

        return {"kspace_encoding_step_1": limit(points[1]),
                "kspace_encoding_step_2": limit(points[2]), "average": limit(scans)}


#**************************************************************************************************#
#                                       MargeProtocolReader                                        #
#**************************************************************************************************#
#                                                                                                  #
# The .csv on its own: one Protocol named by the sequence, the standard keys from the values and   #
# every parameter with its name in the vendor header.                                              #
#                                                                                                  #
#**************************************************************************************************#
class MargeProtocolReader(MargeReader):
    FORMATS = (Format.MARGE_CSV,)
    PACKAGES = Reader.PACKAGES

    def read(self, sniff: Sniff) -> list[Protocol]:
        names, values = self.table(sniff.path)
        vendor = {"parameters": values, "parameter_names": names}
        meta = self._meta(values, vendor)
        name = str(values.get("seqName") or sniff.path.stem)
        return [Protocol(name, (), meta, self.provenance(sniff))]
