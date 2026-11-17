####################################################################################################
#                                            bruker.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for Bruker ParaVision scans (fid, fid_proc.64, 2dseq, rawdata); the vendor       #
#          header holds every parameter of every parameter file of the scan and its processing     #
#          folder.                                                                                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from brukerapi.jcampdx import JCAMPDX
from mrs_tools.constants import GYRO_MAG_RATIO

# own
from miox.data import MRImage, MRSData
from miox.meta.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.readers.bruker_2dseq import Bruker2dseq
from miox.readers.bruker_mrs import BrukerMrs
from miox.readers.bruker_raw import BrukerRaw
from miox.readers.dicom_image import DicomImageReader
from miox.sniff import Format, Kind, Sniff, Sniffer


#**************************************************************************************************#
#                                           BrukerReader                                           #
#**************************************************************************************************#
#                                                                                                  #
# A data file is read in its own mode; a scan folder or a parameter file reads the scan's fid,     #
# else the 2dseq or rawdata next to it. The fid and rawdata of a CSI scan hold k-space, which is   #
# not reconstructed here: they say so, and its folder reads the scanner's reconstruction instead   #
# (pdata/<n>/2dseq, complex spectra first). Spectroscopy is laid out by BrukerMrs; the frequency   #
# is acqp BF1 (reconstructions: VisuAcqImagingFrequency) and the method's PVM_FrqWorkPpm is        #
# ReferencePPM. Transients stored one by one (rawdata: averages x repetitions) are DIM_DYN.        #
# Processed images (2dseq with spatial dimensions only) are assembled from visu_pars, one MRImage  #
# per slice pack; magnitude spectra (MAGNITUDE_IMAGE frames over a spectroscopic dimension) hold   #
# no phase and become MRImages with a frequency axis. Image geometry and values equal the vendor's #
# own DICOM export. Spectroscopy reconstructions take the echo and repetition time from the method #
# file where visu_pars states 0 or none (as for STEAM), as the fid of the same scan reads them. A  #
# scan folder without a file named method takes its one file named <name>_method instead, with a   #
# note.                                                                                            #
#                                                                                                  #
#**************************************************************************************************#
class BrukerReader(Reader):
    FORMATS = (Format.BRUKER,)
    PACKAGES = Reader.PACKAGES + ("brukerapi",)
    REFERENCE = "method/PVM_FrqWorkPpm"
    MODES = ("fid", "fid_proc.64", "2dseq", "rawdata.job0")
    PROCESSED = ("2dseq", "fid_proc.64")  # stored in <study>/<scan>/pdata/<n>/
    PARAMETER_FILES = ("acqp", "method", "visu_pars", "reco", "subject")
    RENAMED_METHOD = "_method"  # a method file renamed <name>_method
    SOURCE = "Bruker visu_pars"
    SPECTROSCOPY_SOURCE = "Bruker parameters"
    LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])
    MS = 1e-3
    IMAGE_KEYS = {  # standard key: (visu_pars field, factor or None)
        "Manufacturer": ("VisuManufacturer", None),
        "ManufacturersModelName": ("VisuStation", None),
        "DeviceSerialNumber": ("VisuSystemOrderNumber", None),
        "InstitutionName": ("VisuInstitution", None),
        "Modality": ("VisuInstanceModality", None),
        "SpectrometerFrequency": ("VisuAcqImagingFrequency", None),
        "ResonantNucleus": ("VisuAcqImagedNucleus", None),
        "EchoTime": ("VisuAcqEchoTime", MS),
        "RepetitionTime": ("VisuAcqRepetitionTime", MS),
        "InversionTime": ("VisuAcqInversionTime", MS),
        "ExcitationFlipAngle": ("VisuAcqFlipAngle", None),
        "SequenceName": ("VisuAcqSequenceName", None),
        "ProtocolName": ("VisuAcquisitionProtocol", None),
        "SeriesDescription": ("VisuSeriesComment", None),
        "StudyDescription": ("VisuStudyDescription", None),
        "PatientName": ("VisuSubjectName", None),
        "PatientID": ("VisuSubjectId", None),
        "PatientWeight": ("VisuSubjectWeight", None),
        "PatientDoB": ("VisuSubjectBirthDate", None),
        "StationName": ("VisuStation", None),
        "SliceThickness": ("VisuCoreFrameThickness", None),
        "PixelBandwidth": ("VisuAcqPixelBandwidth", None),
        "EchoTrainLength": ("VisuAcqEchoTrainLength", None),
        "NumberOfAverages": ("VisuAcqNumberOfAverages", None),
        "PhaseEncodingSteps": ("VisuAcqPhaseEncSteps", None),
    }
    LIST_KEYS = ("SpectrometerFrequency", "ResonantNucleus")
    SEX = {"MALE": "M", "FEMALE": "F", "UNKNOWN": "O"}  # DICOM PatientSex codes
    POSITIONS = {  # VisuSubjectPosition -> DICOM PatientPosition
        "Head_Supine": "HFS", "Head_Prone": "HFP", "Head_Left": "HFDL", "Head_Right": "HFDR",
        "Feet_Supine": "FFS", "Feet_Prone": "FFP", "Feet_Left": "FFDL", "Feet_Right": "FFDR",
    }
    SPOILING = {
        "GradientSpoiled": "GRADIENT", "RFSpoiled": "RF", "RFandGradientSpoiled": "COMBINED"
    }
    GROUP_DIMS = {"FG_SLICE": "slice", "FG_ECHO": "echo", "FG_CYCLE": "cycle", "FG_MOVIE": "volume",
                  "FG_DIFFUSION": "diffusion", "FG_COMPLEX": "complex", "FG_ISA": "isa"}

    # --- dispatch --------------------------------------------------------------------------------

    def read(self, sniff: Sniff) -> list:
        path = self._data_file(sniff.path)
        if path.name == "2dseq":
            header = self._header(path)
            if "visu_pars" not in header:
                raise ReaderError(f"{path}: no visu_pars next to the 2dseq file")
            dataset = Bruker2dseq(path, header["visu_pars"])
            if dataset.spectral_axis() is None:
                return self._images(sniff, dataset, header)
            if set(dataset.frame_types) <= {"MAGNITUDE_IMAGE"}:
                return self._spectra(sniff, dataset, header)
        elif self._imaging_raw(path, sniff):
            header = self._header(path)
            processed = path.parent / "pdata" / "1"
            geometry = None
            if (processed / "visu_pars").is_file():
                visu = self._parameters(processed / "visu_pars")
                geometry = Bruker2dseq(processed / "2dseq", visu)
            return BrukerRaw(path, header, self.provenance(sniff), geometry).datasets()
        return self._spectroscopy(sniff)

    def _imaging_raw(self, path: Path, sniff: Sniff) -> bool:
        """A fid or rawdata.jobN file of an imaging method, or any job beyond the first."""
        if path.name.startswith("rawdata.job") and path.name != "rawdata.job0":
            return True
        if path.name not in ("fid", "rawdata.job0"):
            return False
        header = self._header(path)
        method = header.get("method", {}).get("Method") or sniff.detail.get("method")
        return Sniffer().bruker_scan_kind(method) == Kind.IMAGING

    # --- spectroscopy ---------------------------------------------------------------------------

    def _spectroscopy(self, sniff: Sniff) -> list[MRSData]:
        path = self._data_file(sniff.path)
        header = self._header(path)
        if "method" not in header:
            raise ReaderError(f"{path}: the scan folder has no method file")
        notes: list[str] = []
        scan = path.parents[2] if path.name in self.PROCESSED else path.parent
        method = self.method_file(scan)
        if method is not None and method.name != "method":
            notes.append(f"method file read from {method.name}")
        if self.raw_csi(path, header):
            notes.append(self._kspace_note(path))
        mrs = BrukerMrs(path, header)
        with self.captured(notes):
            spectrum = mrs.spectrum()
        notes += mrs.notes
        tags = self._tag_transients(spectrum.data, spectrum.tags, header)
        data, tags = self.without_zero_indices(spectrum.data, tags, notes)
        d = self.dataset(sniff, data, spectrum.dwelltime, spectrum.values, header, spectrum.name,
                         notes, self.SPECTROSCOPY_SOURCE, spectrum.affine, tags)
        self._reference(d)
        return [d]

    @staticmethod
    def _tag_transients(data: np.ndarray, tags: tuple, header: dict) -> tuple:
        """Transients stored one by one (rawdata: averages x repetitions) are DIM_DYN; a dimension
        that long is otherwise DIM_USER_n."""
        averages = header.get("method", {}).get("PVM_NAverages")
        repetitions = header.get("acqp", {}).get("NR")
        if not isinstance(averages, int) or not isinstance(repetitions, int) or "DIM_DYN" in tags:
            return tags
        tags = list(tags)
        for index, tag in enumerate(tags):
            if isinstance(tag, str) and tag.startswith("DIM_USER") and \
                    data.shape[4 + index] == averages * repetitions:
                tags[index] = ("DIM_DYN", "transients: averages (method PVM_NAverages) x "
                               "repetitions (acqp NR), as acquired", None)
        return tuple(tags)

    # --- images ----------------------------------------------------------------------------------

    def _images(self, sniff: Sniff, dataset: Bruker2dseq, header: dict) -> list[MRImage]:
        frames = dataset.frames()
        groups = dataset.groups()
        counts = [c for _, c in groups]
        arr = frames.reshape(tuple(counts[::-1]) + frames.shape[1:])  # last group first
        spatial = len(frames.shape) - 1
        # spatial axes to (cols, rows[, z]), groups after them, first group first
        spatial_axes = list(range(len(counts), len(counts) + spatial))[::-1]
        order = spatial_axes + list(range(len(counts)))[::-1]
        arr = np.transpose(arr, order)
        names = [self.GROUP_DIMS.get(name, name.lower().removeprefix("fg_")) for name, _ in groups]
        slice_axis = names.index("slice") if "slice" in names else None
        images = []
        packs = dataset.slice_packs() if slice_axis is not None else [(0, 1)]
        for number, (start, count) in enumerate(packs):
            if slice_axis is not None:
                pack = np.take(arr, range(start, start + count), axis=spatial + slice_axis)
                pack = np.moveaxis(pack, spatial + slice_axis, 2)  # slices become the third axis
                dims = tuple(n for n in names if n != "slice")
            else:
                pack = arr if spatial == 3 else arr[:, :, np.newaxis]
                dims = tuple(names)
            first = self._first_frame(dataset, start, slice_axis, counts)
            geometry: list[str] = []
            affine = self._affine(dataset, first, count if slice_axis is not None else None,
                                  geometry)
            name = f"{self._scan(sniff)}_{dataset.path.parent.name}"
            if len(packs) > 1:
                name += f"_{number + 1}"
            meta = self._image_meta(header, dataset, first)
            if slice_axis is not None and count > 1:
                self._slice_spacing(affine, meta)
            notes = [f"Bruker 2dseq, frame groups {groups}" if groups else "Bruker 2dseq"]
            notes += geometry
            if len(packs) > 1:
                notes.append(f"slice pack {number + 1} of {len(packs)}")
            provenance = self.provenance(sniff, tuple(notes))
            data = DicomImageReader._narrow(np.ascontiguousarray(pack))
            images.append(MRImage(data, affine, meta, provenance, name, dims))
        return images

    @staticmethod
    def _first_frame(
        dataset: Bruker2dseq, start: int, slice_axis: int | None, counts: list[int]
    ) -> int:
        """Number of the frame at slice ``start`` with every other group at its first index."""
        if slice_axis is None:
            return 0
        stride = int(np.prod(counts[:slice_axis])) if slice_axis else 1
        return start * stride

    def _affine(self, dataset: Bruker2dseq, first: int, slices: int | None,
                notes: list[str] | None = None) -> np.ndarray:
        """The slice step from the first and last slice positions; when that step does not run
        along the slice normal of VisuCoreOrientation (a contradiction in the file), the normal
        times the frame thickness, noted."""
        orientation = dataset.orientation(first)
        dims = zip(dataset.spacing(), dataset.dim_desc, strict=True)
        spacing = [s for s, d in dims if d == "spatial"]
        lps = np.eye(4)
        lps[:3, 0] = orientation[0] * spacing[0]
        lps[:3, 1] = orientation[1] * spacing[1]
        if slices is not None and slices > 1:
            last = dataset.position(first + (slices - 1) * self._slice_stride(dataset))
            step = (last - dataset.position(first)) / (slices - 1)
            length = np.linalg.norm(step)
            if length and abs(step @ orientation[2]) < self.ALONG_NORMAL * length:
                lps[:3, 2] = orientation[2] * dataset.thickness()
                if notes is not None:
                    notes.append("slice positions do not advance along the slice normal of "
                                 "VisuCoreOrientation: the normal times VisuCoreFrameThickness "
                                 "is used")
            else:
                lps[:3, 2] = step
        elif len(spacing) > 2:
            lps[:3, 2] = orientation[2] * spacing[2]
        else:
            lps[:3, 2] = orientation[2] * dataset.thickness()
        lps[:3, 3] = dataset.position(first)
        return self.LPS_TO_RAS @ lps

    ALONG_NORMAL = 0.99  # cosine between slice step and normal below which they contradict

    def _slice_spacing(self, affine: np.ndarray, meta: Meta) -> None:
        meta.set("SpacingBetweenSlices", float(np.linalg.norm(affine[:3, 2])),
                 "derived: distance between slice centres")

    @staticmethod
    def _slice_stride(dataset: Bruker2dseq) -> int:
        groups = dataset.groups()
        names = [n for n, _ in groups]
        index = names.index("FG_SLICE") if "FG_SLICE" in names else 0
        return int(np.prod([c for _, c in groups[:index]])) if index else 1

    def _image_meta(self, header: dict, dataset: Bruker2dseq, first: int) -> Meta:
        meta = Meta(header)
        visu = header["visu_pars"]
        meta.set("Manufacturer", "Bruker BioSpin", "Bruker ParaVision data set")
        for key, (field, factor) in self.IMAGE_KEYS.items():
            value = visu.get(field)
            if value is None or value == "":
                continue
            value = self.plain(value)
            if isinstance(value, str):
                value = value.strip("<>")
            if factor is not None and isinstance(value, (int, float)):
                value = value * factor
            if key in self.LIST_KEYS and not isinstance(value, list):
                value = [value]
            meta.set(key, value, f"{self.SOURCE}: {field}", f"visu_pars/{field}")
        orientation = dataset.orientation(first)
        meta.set("ImageOrientationPatientDICOM", [float(v) for v in orientation[:2].ravel()],
                 f"{self.SOURCE}: VisuCoreOrientation", "visu_pars/VisuCoreOrientation")
        self._field_strength(meta)
        self._coded_values(meta, visu, dataset, first)
        return meta

    def _coded_values(self, meta: Meta, visu: dict, dataset: Bruker2dseq, first: int) -> None:
        """Values with a DICOM vocabulary or form: sex, position, software, time, phase encoding
        (on the stored axes, which a transposed frame swaps), acquisition type, spoiling and
        magnetisation transfer."""
        def text(field: str) -> str:
            return str(visu.get(field, "")).strip("<>")

        def set_text(key: str, value: Any, field: str) -> None:
            meta.set(key, value, f"{self.SOURCE}: {field}", f"visu_pars/{field}")

        if sex := text("VisuSubjectSex"):
            set_text("PatientSex", self.SEX.get(sex, sex), "VisuSubjectSex")
        if position := text("VisuSubjectPosition"):
            code = self.POSITIONS.get(position, position)
            set_text("PatientPosition", code, "VisuSubjectPosition")
        acquisition, creator = text("VisuAcqSoftwareVersion"), text("VisuCreatorVersion")
        if acquisition or creator:
            value = f"ParaVision Acquisition {acquisition}\\ParaVision {creator or acquisition}"
            source = f"{self.SOURCE}: VisuAcqSoftwareVersion, VisuCreatorVersion"
            meta.set("SoftwareVersions", value, source, "visu_pars/VisuAcqSoftwareVersion")
        stamp = text("VisuAcqDate")  # 2020-12-08T11:32:08,077+0100
        if len(stamp) >= 19 and stamp[10] == "T":
            fraction = stamp[20:23] if stamp[19:20] == "," else "000"
            clock = stamp[11:19] + "." + fraction.ljust(6, "0")
            set_text("AcquisitionTime", clock, "VisuAcqDate")
            set_text("AcquisitionDateTime", stamp[:10] + "T" + clock, "VisuAcqDate")
        spatial = dataset.spatial_dims
        if spatial in (2, 3):
            set_text("MRAcquisitionType", f"{spatial}D", "VisuCoreDimDesc")
        encoding = [str(v) for v in np.atleast_1d(visu.get("VisuAcqGradEncoding", []))]
        if "phase_enc" in encoding[:2]:
            axis = encoding.index("phase_enc")
            stored = 1 - axis if dataset.transposed(first) else axis
            fields = "VisuAcqGradEncoding, VisuCoreTransposition"
            set_text("InPlanePhaseEncodingDirectionDICOM", "ROW" if stored == 0 else "COL", fields)
            set_text("PhaseEncodingAxis", "ij"[stored], fields)
            set_text("ReconMatrixPE", int(dataset.size[stored]), "VisuCoreSize")
            matrix = [int(v) for v in np.atleast_1d(visu.get("VisuAcqSize", []))]
            if len(matrix) > axis:
                set_text("AcquisitionMatrixPE", matrix[axis], "VisuAcqSize")
            factors = (
                ("VisuAcqPartialFourier", "PartialFourier"),
                ("VisuAcqParallelReductionFactor", "ParallelReductionFactorInPlane"),
            )
            for field, key in factors:
                values = [float(v) for v in np.atleast_1d(visu.get(field, []))]
                if len(values) > axis and (key != "PartialFourier" or values[axis] < 1):
                    set_text(key, values[axis], field)
        if spoiling := text("VisuAcqSpoiling"):
            set_text("SpoilingState", spoiling != "NoSpoiling", "VisuAcqSpoiling")
            if spoiling in self.SPOILING:
                set_text("SpoilingType", self.SPOILING[spoiling], "VisuAcqSpoiling")
        if transfer := text("VisuAcqMagnetizationTransfer"):
            set_text("MTState", transfer.lower() == "yes", "VisuAcqMagnetizationTransfer")

    @staticmethod
    def _field_strength(meta: Meta) -> None:
        frequency, nucleus = meta.get("SpectrometerFrequency"), meta.get("ResonantNucleus")
        if frequency and nucleus and nucleus[0] in GYRO_MAG_RATIO:
            field = float(frequency[0]) / GYRO_MAG_RATIO[nucleus[0]]
            meta.set("FieldStrength", field, "derived: SpectrometerFrequency / gyromagnetic ratio")

    @staticmethod
    def _scan(sniff: Sniff) -> str:
        path = sniff.path if sniff.path.is_dir() else sniff.path.parent
        while path.name in ("pdata",) or path.parent.name == "pdata":
            path = path.parent
        return path.name

    # --- magnitude spectra -----------------------------------------------------------------------

    def _spectra(self, sniff: Sniff, dataset: Bruker2dseq, header: dict) -> list[MRImage]:
        frames = dataset.frames()  # (frame, [y, x,] points)
        groups = dataset.groups()
        counts = [c for _, c in groups]
        names = [self.GROUP_DIMS.get(name, name.lower().removeprefix("fg_")) for name, _ in groups]
        arr = frames.reshape(tuple(counts[::-1]) + frames.shape[1:])
        spatial = frames.ndim - 2
        if spatial == 0:
            data = arr.reshape((1, 1, 1) + arr.shape)  # (1, 1, 1, groups..., points)
            data = np.moveaxis(data, -1, 3)  # points first after space
            affine = self._voxel_affine(header)
            notes = ["magnitude spectrum of a single voxel: no phase, so no FID"]
        else:  # (groups..., y, x, points) -> (x, y, 1, points, groups...)
            data = np.moveaxis(arr, -1, 0)  # points first
            data = np.moveaxis(data, -1, 0)  # x
            data = np.moveaxis(data, -1, 1)  # y
            data = data[:, :, np.newaxis]
            affine = self._affine(dataset, 0, None)
            notes = ["magnitude spectra of a CSI grid: no phase, so no FID"]
        dims = ("frequency",) + tuple(reversed(names))
        meta = self._image_meta(header, dataset, 0)
        self._method_timing(meta, header)
        axis = dataset.spectral_axis()
        points = dataset.size[axis]
        width_ppm = dataset.extent[axis]
        frequency = meta.get("SpectrometerFrequency", [None])[0]
        meta.set("NumberOfPoints", int(points), "data shape")
        width, source, field = 0.0, f"{self.SOURCE}: VisuCoreExtent (ppm) * frequency", None
        if frequency and width_ppm > 0:
            width = width_ppm * float(frequency)
        elif header.get("method", {}).get("PVM_SpecSWH"):
            width = float(np.atleast_1d(header["method"]["PVM_SpecSWH"])[0])
            source, field = "method: PVM_SpecSWH", "method/PVM_SpecSWH"
        if width > 0:
            meta.set("SpectralWidth", width, source, field)
            meta.set("DwellTime", 1 / width, "derived: 1 / SpectralWidth")
        else:
            notes.append("no spectral width stated (VisuCoreExtent 0, no PVM_SpecSWH)")
        reference = header.get("method", {}).get("PVM_FrqWorkPpm")
        if reference is not None:
            value = float(np.atleast_1d(reference)[0])
            meta.set("ReferencePPM", value, "method: PVM_FrqWorkPpm", "method/PVM_FrqWorkPpm")
        notes.append("frequency increases with the index; the centre point lies at ReferencePPM")
        provenance = self.provenance(sniff, tuple(notes))
        name = f"{self._scan(sniff)}_{dataset.path.parent.name}"
        return [MRImage(data.astype(np.float32), affine, meta, provenance, name, dims)]

    def _voxel_affine(self, header: dict) -> np.ndarray:
        """The spectroscopy voxel: PVM_VoxArr* of the method file, as the fid reads them."""
        method = header.get("method", {})
        try:
            orientation = np.asarray(method["PVM_VoxArrGradOrient"], float).reshape(3, 3)
            position = np.asarray(method["PVM_VoxArrPosition"], float).ravel()[:3]
            size = np.asarray(method["PVM_VoxArrSize"], float).ravel()[:3]
        except (KeyError, ValueError):
            return self.NO_GEOMETRY
        reorder = [1, 2, 0]
        affine = np.eye(4)
        affine[:3, :3] = orientation[reorder, :].T * size[reorder]
        affine[:3, 3] = position[reorder]
        return affine

    # --- files -----------------------------------------------------------------------------------

    def correct(self, d: MRSData) -> None:
        super().correct(d)
        if "visu_pars" in d.meta.vendor:
            self._method_timing(d.meta, d.meta.vendor)

    def _method_timing(self, meta: Meta, header: dict) -> None:
        """Spectroscopy reconstructions: visu_pars states the echo time as 0 and no repetition
        time; the method file states the acquisition's, as the fid of the same scan reads them."""
        method = header.get("method", {})
        for key, field in (("EchoTime", "PVM_EchoTime"), ("RepetitionTime", "PVM_RepetitionTime")):
            value = method.get(field)
            if isinstance(value, (int, float)) and value > 0 and not meta.get(key):
                source = f"method: {field} (ms); visu_pars states none or 0"
                meta.set(key, value * self.MS, source, f"method/{field}")

    def _data_file(self, path: Path) -> Path:
        if path.name in self.MODES:
            return path
        folder = path if path.is_dir() else path.parent
        for name in self.MODES:
            if (folder / name).is_file():
                reconstructions = self.reconstructions(folder / name)
                return reconstructions[0] if reconstructions else folder / name
        seqs = sorted(folder.glob("pdata/*/2dseq"), key=lambda p: (len(p.parent.name),
                                                                   p.parent.name))
        if seqs:  # a scan shared without its raw data: the first reconstruction
            return seqs[0]
        raise ReaderError(f"no Bruker data file (fid, 2dseq, rawdata) in {folder}")

    # --- CSI k-space -----------------------------------------------------------------------------

    def raw_csi(self, path: Path, header: dict | None = None) -> bool:
        """A fid or rawdata.job0 of a CSI scan (pulse program CSI.ppg, brukerapi's CSI scheme):
        it holds k-space: its grid dimensions are phase-encoding steps, not voxels."""
        if path.name not in ("fid", "rawdata.job0"):
            return False
        header = self._header(path) if header is None else header
        return str(header.get("acqp", {}).get("PULPROG", "")).strip("<>") == "CSI.ppg"

    def reconstructions(self, path: Path) -> list[Path]:
        """The scanner's reconstructions (pdata/<n>/2dseq) of a raw CSI file, complex spectra
        before magnitude spectra, else in processing order."""
        if not self.raw_csi(path):
            return []
        found = []
        for seq in sorted(path.parent.glob("pdata/*/2dseq"), key=lambda p: (len(p.parent.name),
                                                                              p.parent.name)):
            if (seq.parent / "visu_pars").is_file():
                dataset = Bruker2dseq(seq, self._parameters(seq.parent / "visu_pars"))
                types = {t.upper() for t in dataset.frame_types}
                found.append((not {"REAL_IMAGE", "IMAGINARY_IMAGE"} <= types, seq))
        return [seq for _, seq in sorted(found, key=lambda f: f[0])]

    def _kspace_note(self, path: Path) -> str:
        names = [str(p.relative_to(path.parent)) for p in self.reconstructions(path)]
        where = (f"the scanner's reconstruction is {' or '.join(names)}" if names
                 else "the scan holds no reconstruction (pdata/<n>/2dseq)")
        return (f"k-space, not reconstructed: the spatial dimensions of {path.name} hold "
                f"phase-encoding steps, not voxels; {where}")

    def _header(self, data_file: Path) -> dict:
        """All parameters of the files that describe ``data_file``, by file name."""
        if data_file.name in self.PROCESSED:
            folders = (data_file.parent, data_file.parents[2], data_file.parents[3])
        else:  # <study>/<scan>/fid
            folders = (data_file.parent, data_file.parents[1])
        header = {}
        for folder in folders:
            for name in self.PARAMETER_FILES:
                file = self.method_file(folder) if name == "method" else folder / name
                if name not in header and file is not None and file.is_file():
                    header[name] = self._parameters(file)
        return header

    @classmethod
    def method_file(cls, scan: Path) -> Path | None:
        """The scan folder's method file: ``method``, or else its one file named <name>_method."""
        if (scan / "method").is_file():
            return scan / "method"
        renamed = [p for p in scan.iterdir() if p.is_file() and not p.name.startswith("._")
                   and p.name.endswith(cls.RENAMED_METHOD)]
        return renamed[0] if len(renamed) == 1 else None

    def _parameters(self, path: Path) -> dict[str, Any]:
        parameters = {}
        for key, parameter in JCAMPDX(path).params.items():
            try:
                parameters[key] = self.plain(parameter.value)
            except Exception:
                parameters[key] = parameter.val_str
        return parameters
