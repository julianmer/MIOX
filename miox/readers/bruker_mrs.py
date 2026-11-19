####################################################################################################
#                                          bruker_mrs.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 17/11/26                                                                                #
#                                                                                                  #
# Purpose: Bruker ParaVision spectroscopy: fid, fid_proc.64 and rawdata.job0 laid out per scheme,  #
#          complex 2dseq reconstructions turned back into FIDs, voxel geometry and keys.           #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# own
from miox import __version__
from miox.readers.base import ReaderError
from miox.readers.bruker_2dseq import Bruker2dseq
from miox.readers.philips_dicom_mrs import Spectrum


#**************************************************************************************************#
#                                            BrukerMrs                                             #
#**************************************************************************************************#
#                                                                                                  #
# A fid holds blocks of interleaved real and imaginary samples (acqp GO_raw_data_format and        #
# BYTORDA up to ParaVision 7; on PV-360 32-bit integers, fid_proc.64 64-bit floats), each block    #
# padded to 1 KiB where GO_block_size is Standard_KBlock_Format; a block holds one transient       #
# (single voxel, NR blocks) or one phase-encoding step (CSI.ppg, the steps in k-space order). A    #
# rawdata.job0 holds its job's samples per receiver and transient. The points before the echo      #
# (PVM_DigShift; PV-360: acqp ACQ_RxFilterInfo) are left out, a working offset                     #
# (PVM_FrqWorkOffsetPpm) is turned back by a linear phase, and the FIDs are conjugated to NIfTI-   #
# MRS's convention. A complex 2dseq spectrum is the reconstruction's: conjugated, inverse FFT,     #
# without the digital filter's points, which the reconstruction moves to the end. Single voxels    #
# sit in the method's voxel (PVM_VoxArr*, the axes taken as 1, 2, 0); CSI reconstructions on the   #
# visu_pars grid turned by the subject position. Keys and layouts as ParaVision states them, with  #
# the dwell time from SW_h (else the job's sample rate) or, for a 2dseq, VisuAcqPixelBandwidth.    #
#                                                                                                  #
#**************************************************************************************************#
class BrukerMrs:
    CLASSIC = ("<PV 5.1>", "<PV 6.0>", "<PV 6.0.1>", "<PV-7.0.0>")
    FORMATS = {"GO_32BIT_SGN_INT": "i4", "GO_16BIT_SGN_INT": "i2", "GO_32BIT_FLOAT": "f4"}
    MEGA = "mt_MEGA_sLASER_V35.ppg"
    TAGS = {"repetition": "DIM_DYN", "channel": "DIM_COIL"}
    POSITIONS = {  # VisuSubjectPosition: rotation of the visu_pars frame (BrkRaw's recipe)
        "Head_Supine": np.pi, "Head_Prone": 0.0, "Head_Left": np.pi / 2, "Head_Right": -np.pi / 2,
    }

    def __init__(self, path: Path, header: dict):
        self.path = path
        self.header = header
        self.acqp = header.get("acqp", {})
        self.method = header.get("method", {})
        self.visu = header.get("visu_pars", {})
        self.version = str(self.acqp.get("ACQ_sw_version", ""))
        self.pv360 = self.version.startswith("<PV-360.")
        self.pulprog = str(self.acqp.get("PULPROG", "")).strip("<>")
        self.notes: list[str] = []

    @property
    def kind(self) -> str:
        if self.path.name == "fid_proc.64":
            return "fid_proc"
        return "rawdata" if self.path.name.startswith("rawdata.") else self.path.name

    def spectrum(self) -> Spectrum:
        if self.kind == "2dseq":
            return self._reconstruction()
        data, dims = self._rawdata() if self.kind == "rawdata" else self._fid()
        description = self.acqp.get("ACQ_dim_desc")
        svs = description == "Spectroscopic" or description == ["Spectroscopic"]
        csi = description == ["Spectroscopic", "Spatial", "Spatial"]
        if svs or csi:
            data = data[self.points_before_echo():, ...]
        if svs:
            offset = self._first(self.method.get("PVM_FrqWorkOffsetPpm"))
            if offset:
                hertz = offset * self._first(self.method.get("PVM_FrqRef"))
                time = np.arange(data.shape[0]) * (1.0 / self.method["PVM_SpecSWH"])
                phase = np.exp(1j * time * hertz * 2 * np.pi)
                data = data * phase.reshape((-1,) + (1,) * (data.ndim - 1))
            data = data.conj().reshape((1, 1, 1) + data.shape)
        elif csi:
            data = np.expand_dims(np.moveaxis(data.conj(), 0, -1), axis=2)
        tags = self._tags(data, dims)
        while data.ndim > 4 and data.shape[-1] == 1:  # trailing length-1 dimensions, as NIfTI
            data, tags = data[..., 0], tags[: data.ndim - 5]
        values = self._values(self.acqp.get("BF1"), self.acqp.get("NUCLEUS"),
                              self.acqp.get("ACQ_sw_version"), self.method.get("Method"),
                              ("acqp/BF1", "acqp/NUCLEUS", "acqp/ACQ_sw_version", "method/Method"))
        if self.method.get("PVM_EchoTime") is not None:  # none stated by pulse-acquire methods
            values["EchoTime"] = (float(self.method["PVM_EchoTime"] * 1e-3),
                                  "method/PVM_EchoTime")
        if self.method.get("PVM_RepetitionTime") is not None:
            values["RepetitionTime"] = (float(self.method["PVM_RepetitionTime"] / 1e3),
                                        "method/PVM_RepetitionTime")
        dwelltime, field = self._dwelltime()
        values["DwellTime"] = (dwelltime, field)
        return Spectrum(self.name(), data, dwelltime, self.voxel(), values, tags)

    # --- fid -------------------------------------------------------------------------------------

    def _fid(self) -> tuple[np.ndarray, list[str]]:
        """(points, transients) or, for CSI.ppg, (points, steps 1, steps 2), as encoded."""
        dtype = self._dtype()
        size = self._first(self.acqp["ACQ_size"])
        receivers = 1 if self._dim_desc_first() == "Spectroscopic" else \
            self.method.get("PVM_EncNReceivers")
        jobs = self._job()
        if not self.pv360:
            block = int(size) * receivers
            if str(self.acqp.get("GO_block_size")) == "Standard_KBlock_Format":
                block = int(np.ceil(block * dtype.itemsize / 1024.0) * 1024.0 / dtype.itemsize)
                length = int(size) * receivers
            else:
                length = int(2 * self.method["PVM_DigNp"] * receivers)
        elif self.pulprog == self.MEGA:
            block = length = int(2 * self._first(self.method["PVM_SpecMatrix"]))
        else:
            block = length = int(jobs[0])
        csi = self.pulprog == "CSI.ppg"
        repetitions = self.acqp["NR"]
        if csi and not self.pv360:
            sizes = self.acqp["ACQ_size"]
            count = sizes[1] * sizes[2]
            encoding = [sizes[0] // 2, sizes[1], sizes[2]]
        elif csi and self.version == "<PV-360.1.1>":
            count = self.acqp["ACQ_spatial_size_0"] * self.acqp["ACQ_spatial_size_1"]
            encoding = [int(jobs[0]) // 2, self.acqp["ACQ_spatial_size_0"],
                        self.acqp["ACQ_spatial_size_1"]]
        elif csi:
            raise ReaderError(f"{self.path}: CSI fid of {self.version} has no known layout")
        else:
            if receivers != 1 and not self.pv360:
                raise ReaderError(f"{self.path}: a fid of {receivers} receivers that is not CSI "
                                  "has no known layout")
            count = repetitions
            encoding = [length // 2, repetitions]  # the block's samples as complex points
        stored = self._stored(dtype, (block, int(count)))
        stored = stored[:length]
        data = stored[0::2] + 1j * stored[1::2]
        data = data.reshape([int(n) for n in encoding], order="F")
        data = self._reorder_lines(data)
        dims = ["k_space_encode_step_0", "k_space_encode_step_1", "k_space_encode_step_2"] \
            if csi else ["k_space_encode_step_0", "repetition"]
        return data, dims[1:]

    def _reorder_lines(self, data: np.ndarray) -> np.ndarray:
        """Lines (axis 1) in the order of PVM_EncSteps1, where the method states it."""
        steps = self.method.get("PVM_EncSteps1")
        if steps is None:
            return data
        order = np.argsort(steps)
        if np.array_equal(order, steps):
            return data
        data = data.copy()
        for index in np.ndindex(data.shape[2:]):
            where = (slice(None), slice(None)) + index
            data[where] = data[where][:, order]
        return data

    def _dtype(self) -> np.dtype:
        order = "<" if str(self.acqp.get("BYTORDA")) == "little" else ">"
        if self.pv360:
            if self.kind == "fid_proc":
                return np.dtype("<f8")
            if str(self.acqp.get("ACQ_word_size")) == "_32_BIT":
                return np.dtype(order + "i4")
            raise ReaderError(f"{self.path}: unknown sample format")
        kind = self.FORMATS.get(str(self.acqp.get("GO_raw_data_format")))
        if kind is None or self.version not in self.CLASSIC:
            raise ReaderError(f"{self.path}: unknown sample format")
        return np.dtype(order + kind)

    def _stored(self, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
        """The file as an array of ``shape`` (first axis fastest)."""
        size = self.path.stat().st_size
        if size != int(np.prod(shape)) * dtype.itemsize:
            raise ReaderError(f"{self.path}: {size} bytes, the parameters state {shape} samples "
                              f"of {dtype.itemsize} bytes")
        return np.fromfile(self.path, dtype=dtype).reshape(shape, order="F")

    # --- rawdata ---------------------------------------------------------------------------------

    def _rawdata(self) -> tuple[np.ndarray, list[str]]:
        """(points, receivers, transients), receivers left out where there is one (but CSI)."""
        jobs = self._job()
        receivers = self.method["PVM_EncNReceivers"]
        dtype = self._dtype()
        csi = self.pulprog == "CSI.ppg"
        if csi or receivers > 1:
            shape = (int(jobs[0]), receivers, int(jobs[3]))
            dims = ["channel", "repetition"]
        else:
            shape = (int(jobs[0]), int(jobs[3]))
            dims = ["repetition"]
        stored = self._stored(dtype, shape)
        return stored[0::2] + 1j * stored[1::2], dims

    def _job(self) -> list:
        """The first job of ACQ_jobs (PV-360: without its label)."""
        jobs = self.acqp.get("ACQ_jobs")
        job = jobs[0] if isinstance(jobs, list) and jobs and isinstance(jobs[0], list) else jobs
        if not isinstance(job, list):
            return []
        return job[:-1] if self.pv360 else job

    # --- 2dseq -----------------------------------------------------------------------------------

    def _reconstruction(self) -> Spectrum:
        """A complex 2dseq spectrum turned back into its FID."""
        dataset = Bruker2dseq(self.path, self.visu)
        types = [t.upper() for t in dataset.frame_types]
        if "REAL_IMAGE" not in types or "IMAGINARY_IMAGE" not in types:
            raise ReaderError(f"{self.path}: a spectroscopic 2dseq without real and imaginary "
                              f"frames ({sorted(set(types))})")
        frames = dataset.frames()
        spectrum = frames[types.index("REAL_IMAGE")] + 1j * frames[types.index("IMAGINARY_IMAGE")]
        shift = self.method.get("PVM_DigShift", 0)
        shift = shift if isinstance(shift, int) else 0
        fid = np.fft.ifft(np.fft.ifftshift(spectrum.conj(), axes=-1), axis=-1)
        fid = np.roll(fid, -shift, axis=-1)[..., : fid.shape[-1] - shift]
        if fid.ndim == 1:
            data, affine = fid.reshape(1, 1, 1, -1), self.voxel()
        else:  # (y, x, points) -> (x, y, 1, points)
            data, affine = np.swapaxes(fid, 0, 1)[:, :, np.newaxis, :], self._grid()
        self.notes.append("complex 2dseq spectrum turned into a FID by inverse FFT, without the "
                          f"digital filter's {shift} points")
        visu = self.visu
        values = self._values(visu.get("VisuAcqImagingFrequency"),
                              visu.get("VisuAcqImagedNucleus"),
                              visu.get("VisuAcqSoftwareVersion"),
                              visu.get("VisuAcquisitionProtocol"),
                              ("visu_pars/VisuAcqImagingFrequency",
                               "visu_pars/VisuAcqImagedNucleus",
                               "visu_pars/VisuAcqSoftwareVersion",
                               "visu_pars/VisuAcquisitionProtocol"))
        for key, field, scale, method_field in (
                ("EchoTime", "VisuAcqEchoTime", 1e-3, "PVM_EchoTime"),
                ("RepetitionTime", "VisuAcqRepetitionTime", 1e-3, "PVM_RepetitionTime")):
            if visu.get(field) is not None:
                values[key] = (float(visu[field] * scale), f"visu_pars/{field}")
            elif self.method.get(method_field) is not None:
                values[key] = (float(self.method[method_field] * scale), f"method/{method_field}")
        name = visu.get("VisuSubjectName")
        if name is not None:
            values["PatientName"] = (str(name)[1:-1], "visu_pars/VisuSubjectName")
        dwelltime = 1.0 / (visu["VisuAcqPixelBandwidth"] * self._first(visu["VisuCoreSize"]))
        values["DwellTime"] = (dwelltime, "visu_pars/VisuAcqPixelBandwidth")
        return Spectrum(self.name(), data, dwelltime, affine, values)

    def _grid(self) -> np.ndarray:
        """Affine of a CSI reconstruction: visu_pars orientation turned by the subject position,
        voxels of the method's voxel extent over the grid (ParaVision 5.1: the extent is one
        voxel's, the slice VisuCoreFrameThickness), at the method's voxel position."""
        visu, extent = self.visu, np.asarray(self.method["PVM_VoxArrSize"], float).reshape(-1, 3)
        sizes = visu["VisuCoreSize"]
        creator = str(visu.get("VisuCreatorVersion"))
        if creator == "<5.1>":
            resolution = np.array([extent[0, 0], extent[0, 1], visu["VisuCoreFrameThickness"]])
        else:
            resolution = np.array([extent[0, 0] / sizes[1], extent[0, 1] / sizes[2],
                                   extent[0, 2]])
            if creator != "<6.0.1>":
                self.notes.append(f"CSI grid of ParaVision {creator} laid out as checked for "
                                  "6.0.1")
        angle = self.POSITIONS.get(str(visu.get("VisuSubjectPosition")))
        if angle is None:
            raise ReaderError(f"{self.path}: subject position {visu.get('VisuSubjectPosition')} "
                              "has no grid orientation here")
        turn = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0],
                         [0, 0, 1]])
        orientation = np.reshape(np.asarray(visu["VisuCoreOrientation"], float).ravel()[:9],
                                 (3, 3))
        affine = np.eye(4)
        affine[:3, :3] = turn @ orientation.T.dot(np.diag(resolution))
        affine[:3, 3] = np.asarray(self.method["PVM_VoxArrPosition"], float).reshape(-1, 3)[0]
        return affine

    # --- geometry, keys --------------------------------------------------------------------------

    def voxel(self) -> np.ndarray | None:
        """The method's voxel: PVM_VoxArrGradOrient, Position and Size, axes taken as 1, 2, 0;
        None for a method without localisation, which states no voxel."""
        fields = ("PVM_VoxArrGradOrient", "PVM_VoxArrPosition", "PVM_VoxArrSize")
        if any(self.method.get(field) is None for field in fields):
            self.notes.append("no voxel in the method file (PVM_VoxArr*): no geometry")
            return None
        orientation, position, size = (np.squeeze(np.asarray(self.method[field], float))
                                       for field in fields)
        reorder = [1, 2, 0]
        affine = np.zeros((4, 4))
        affine[3, 3] = 1
        affine[:3, :3] = orientation[reorder, :].T * size[reorder]
        affine[:3, 3] = position[reorder]
        return affine

    def points_before_echo(self) -> int:
        shift = self.method.get("PVM_DigShift")
        if shift is not None:
            return int(shift)
        if self.pv360:
            return int(np.round(self.acqp["ACQ_RxFilterInfo"][0][0]))
        return 0

    def _dwelltime(self) -> tuple[float, str]:
        if self.kind == "rawdata":
            rate = self.acqp.get("SW_h")
            if rate:
                return 1.0 / rate, "acqp/SW_h"
            return 1.0 / self._job()[5], "acqp/ACQ_jobs"
        if self.acqp.get("SW_h"):
            return 1.0 / self.acqp["SW_h"], "acqp/SW_h"
        return 1.0 / self._job()[5], "acqp/ACQ_jobs"

    def _tags(self, data: np.ndarray, dims: list[str]) -> tuple:
        """Tags of dimensions 5 and up: transients DIM_DYN, receivers DIM_COIL (rawdata: where the
        size is the number of repetitions or receivers), else DIM_USER_n; length 1 untagged."""
        tags: list[str | None] = []
        user = 0
        for index, dim in enumerate(dims):
            if index >= data.ndim - 4:
                break
            size = data.shape[index + 4]
            if size <= 1:
                tags.append(None)
                continue
            known = dim in self.TAGS
            if known and self.kind == "rawdata":
                expected = self.method.get("PVM_EncNReceivers") if dim == "channel" else \
                    self.method.get("PVM_NRepetitions")
                known = size == expected
            if known:
                tags.append(self.TAGS[dim])
            else:
                tags.append(f"DIM_USER_{user}")
                user += 1
        return tuple(tags)

    def _values(self, frequency, nucleus, software, sequence, fields) -> dict:
        """The keys of every spectrum; a value the parameter files do not state is left out."""
        offset = self._first(self.method.get("PVM_FrqWorkOffsetPpm"))
        values = {
            "Manufacturer": ("Bruker", None),
            "ConversionMethod": (f"miox {__version__}", None),
            "ConversionTime": (datetime.now().isoformat(timespec="milliseconds"), None),
            "OriginalFile": ([self.original_file()], None),
            "kSpace": ([False, False, False], None),
        }
        if frequency is not None:
            values["SpectrometerFrequency"] = ([frequency], fields[0])
        if nucleus is not None and str(nucleus).strip("<>"):
            values["ResonantNucleus"] = ([str(nucleus)[1:-1]], fields[1])
        if offset is not None:
            values["TxOffset"] = (float(offset), "method/PVM_FrqWorkOffsetPpm")
        for key, value, field in (("SoftwareVersions", software, fields[2]),
                                  ("SequenceName", sequence, fields[3])):
            if value is not None:
                values[key] = (value, field)
        return values

    def original_file(self) -> str:
        """The data file from its scan folder on, e.g. 10/pdata/2/2dseq."""
        scan = self.path.parents[2] if self.kind in ("2dseq", "fid_proc") else self.path.parent
        return str(self.path.relative_to(scan.parent))

    def name(self) -> str:
        """FID_<scan>[_<processing>], Rawdata_<scan> or 2DSEQ_<scan>_<processing>_<subject>_<study>
        (the study number)."""
        if self.kind == "fid":
            return f"FID_{self.path.parent.name}"
        if self.kind == "fid_proc":
            return f"FID_{self.path.parents[2].name}_{self.path.parent.name}"
        if self.kind == "rawdata":
            job = self.path.name.removeprefix("rawdata.")
            prefix = "Rawdata" if job == "job0" else f"Rawdata_{job}"
            return f"{prefix}_{self.path.parent.name}"
        subject = str(self.visu.get("VisuSubjectName", "<>"))[1:-1]
        study = self.visu.get("VisuStudyNumber", "")
        return f"2DSEQ_{self.path.parents[2].name}_{self.path.parent.name}_{subject}_{study}" \
            .rstrip("_")

    def _dim_desc_first(self) -> Any:
        description = self.acqp.get("ACQ_dim_desc")
        return description[0] if description else description

    @staticmethod
    def _first(value: Any) -> Any:
        return value[0] if isinstance(value, list) else value
