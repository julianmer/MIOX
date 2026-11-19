####################################################################################################
#                                       siemens_dicom_mrs.py                                       #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 17/11/26                                                                                #
#                                                                                                  #
# Purpose: Siemens DICOM spectroscopy, syngo VA to VE (CSA headers) and XA (MR Spectroscopy        #
#          Storage): spectra, voxel or grid geometry and keys of one series.                       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from mapvbvd.read_twix_hdr import parse_buffer

with warnings.catch_warnings():  # nibabel.nicom warns on import that it is experimental
    warnings.simplefilter("ignore", UserWarning)
    from nibabel.nicom import csareader

# own
from miox import __version__
from miox.readers.base import ReaderError
from miox.readers.philips_dicom_mrs import Spectrum
from miox.readers.siemens_rda import SiemensRdaReader
from miox.sniff import open_dicom


#**************************************************************************************************#
#                                         SiemensDicomMrs                                          #
#**************************************************************************************************#
#                                                                                                  #
# syngo VB to VE files (Siemens private storage) hold the spectra in (7FE1,1010) and their         #
# parameters in the CSA image header; XA files (MR Spectroscopy Storage) in SpectroscopyData, as   #
# the conjugate (the imaginary part is negated on reading). More than one voxel is a grid: (slices #
# or frames, rows, columns, points) in the file, x the columns; VB to VE grids at the CSA          #
# ImagePositionPatient shifted half a voxel in-plane, XA grids at the plane position; their VoI    #
# box is VOI. A single voxel sits at the CSA VoiPosition, sized VoiPhaseFoV x VoiReadoutFoV x      #
# VoiThickness (XA: plane position and pixel measures). syngo VA files (protocol MrProtocol) keep  #
# the voxel in the protocol's sVoI and the dwell time in RealDwellTime (the protocol's is          #
# oversampled). The files of a series are ordered by series and instance number and stacked along  #
# DIM_DYN; the CMRR svs_slaser(VOI)_dkd(2) and dkd_svs_(m)slaser_msspnav / moconav sequences keep  #
# reference transients at the start and end of the series (lAutoRefScanMode, lAutoRefScanNo),      #
# which become their own datasets, and their metabolite-cycled (and MEGA-edited) transients are    #
# laid out as DIM_METCYCLE (and DIM_EDIT). Keys from the CSA header and protocol (VB to VE: echo   #
# time the sum of alTE) or the functional groups (XA), times in s; OriginalFile the file names.    #
#                                                                                                  #
#**************************************************************************************************#
class SiemensDicomMrs:
    VX = "1.3.12.2.1107.5.9.1"  # Siemens private spectroscopy storage
    XA = "1.2.840.10008.5.1.4.1.1.4.2"  # MR Spectroscopy Storage
    MR_IMAGE = "1.2.840.10008.5.1.4.1.1.4"
    VX_DATA = (0x7FE1, 0x1010)
    XA_DATA = (0x5600, 0x0020)
    VIENNA_CRT = re.compile(r"_ViennaCrt_")
    IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def __init__(self):
        self.notes: list[str] = []

    # --- series ----------------------------------------------------------------------------------

    def read(self, paths: list[Path]) -> list[Spectrum]:
        """The datasets of one series."""
        files = []
        for index, path in enumerate(paths):
            ds = self.dataset(path)
            try:
                kind = self.kind(ds)
            except ReaderError as e:
                if len(paths) == 1:
                    raise
                self.notes.append(f"{path.name} skipped: {e}")
                continue
            if "MrProtocol" in self._series_protocol_names(ds):
                spectrum = self._va(ds, path)
            elif kind == "csi":
                spectrum = self._csi(ds, path)
            else:
                spectrum = self._svs(ds, path)
            series = ds.get("SeriesNumber")
            series = int(series) if series is not None else int(
                str(ds.SeriesInstanceUID).replace(".", ""))
            instance = ds.get("InstanceNumber")
            reference, suffix = self.reference(ds, instance)
            files.append((series, int(instance) if instance is not None else index, reference,
                          suffix, spectrum, path, ds))
        if not files:
            raise ReaderError("no Siemens spectroscopy file in the series")
        first = files[0][6]
        name = str(first.get("SeriesDescription") or first.get("SeriesInstanceUID") or "")
        files.sort(key=lambda f: (f[0], f[1]))
        spectra = []
        for series in sorted({f[0] for f in files}):
            for reference in sorted({f[2] for f in files}):
                group = [f for f in files if f[0] == series and f[2] == reference]
                if group:
                    spectra.append(self._combine(group, name))
        names = [s.name for s in spectra]
        if len(set(names)) < len(names):
            counts: dict[str, int] = {}
            for s in spectra:
                count = counts.get(s.name, 0)
                counts[s.name] = count + 1
                s.name = f"{s.name}_{count:03}"
        return spectra

    def _combine(self, group: list[tuple], name: str) -> Spectrum:
        """One group of files (one series, one reference kind) as one dataset."""
        spectrum = group[0][4]
        last = group[-1][6]
        data = np.stack([f[4].data for f in group], axis=-1) if len(group) > 1 else spectrum.data
        if len(group) > 1 and self.VIENNA_CRT.search(str(last.get("ProtocolName", ""))):
            data = np.moveaxis(data, (0, 1, 2, 3, 4), (0, 1, 4, 3, 2))
        values = dict(spectrum.values, OriginalFile=([f[5].name for f in group], None))
        data, tags = self._sequence_layout(data, last, group[0][2])
        return Spectrum(name + group[0][3], data, spectrum.dwelltime, spectrum.affine, values,
                        tags)

    def _sequence_layout(self, data: np.ndarray, ds: pydicom.Dataset, reference: int) -> tuple:
        """Metabolite cycling (and MEGA editing) of the dkd_svs_(m)slaser navigator sequences;
        otherwise the stacked files are DIM_DYN."""
        name = self._sequence_file(ds)
        if reference == 0 and re.search(r"dkd_svs_mslaser_(msspnav|moconav)$", name):
            data = data.reshape(data.shape[:4] + (2, -1, 2))
            return data, ("DIM_EDIT", "DIM_DYN", "DIM_METCYCLE")
        if reference == 0 and re.search(r"dkd_svs_slaser_(msspnav|moconav)$", name):
            return data.reshape(data.shape[:4] + (-1, 2)), ("DIM_DYN", "DIM_METCYCLE")
        return data, ("DIM_DYN",) if data.ndim > 4 else ()

    # --- files -----------------------------------------------------------------------------------

    @staticmethod
    def dataset(path: Path) -> pydicom.Dataset:
        with open_dicom(path) as fh:
            return pydicom.dcmread(fh, force=True)

    def software(self, ds: pydicom.Dataset) -> str:
        """'vx' (syngo VA to VE) or 'xa'."""
        sop = str(ds.get("SOPClassUID", ""))
        if sop == self.VX:
            return "vx"
        if sop == self.XA:
            return "xa"
        if sop == self.MR_IMAGE:
            raise ReaderError(f"SOP class {sop} holds MR images, not spectroscopy")
        raise ReaderError(f"SOP class {sop} is not Siemens spectroscopy")

    def kind(self, ds: pydicom.Dataset) -> str:
        """'csi' when the grid holds more than one voxel, else 'svs'."""
        if self.software(ds) == "vx":
            csa = self.csa(ds)
            if "MrProtocol" in self._series_protocol_names(ds):
                return "svs"
            sizes = [csa["Rows"][0], csa["Columns"][0], csa["NumberOfFrames"][0]]
        else:
            geometry = ds.SharedFunctionalGroupsSequence[0].MRSpectroscopyFOVGeometrySequence[0]
            sizes = [geometry.SpectroscopyAcquisitionPhaseRows,
                     geometry.SpectroscopyAcquisitionPhaseColumns,
                     geometry.SpectroscopyAcquisitionOutOfPlanePhaseSteps]
        return "csi" if np.prod([float(s) for s in sizes]) > 1 else "svs"

    @staticmethod
    def csa(ds: pydicom.Dataset, kind: str = "image") -> dict[str, list]:
        """The items of every CSA header element."""
        header = csareader.get_csa_header(ds, kind)
        return {name: tag["items"] for name, tag in header["tags"].items()} if header else {}

    def _series_protocol_names(self, ds: pydicom.Dataset) -> list[str]:
        return [n for n in ("MrPhoenixProtocol", "MrProtocol") if self.csa(ds, "series").get(n)]

    def protocol(self, ds: pydicom.Dataset) -> dict | None:
        """The series' MrPhoenixProtocol (syngo VB to VE), parsed."""
        items = self.csa(ds, "series").get("MrPhoenixProtocol")
        return parse_buffer(items[0]) if items else None

    def _sequence_file(self, ds: pydicom.Dataset) -> str:
        protocol = self.protocol(ds)
        return str(protocol[("tSequenceFileName",)]).strip('"').lower() if protocol else ""

    # --- spectra ---------------------------------------------------------------------------------

    def _svs(self, ds: pydicom.Dataset, path: Path) -> Spectrum:
        if self.software(ds) == "vx":
            csa = self.csa(ds)
            floats = np.frombuffer(ds[self.VX_DATA].value, dtype=np.single)
            data = floats[0::2] + 1j * floats[1::2]
            orientation = np.asarray(csa["ImageOrientationPatient"], dtype=float)
            sizes = [csa["VoiPhaseFoV"][0], csa["VoiReadoutFoV"][0], csa["VoiThickness"][0]]
            affine = SiemensRdaReader.grid_affine(orientation, csa["VoiPosition"], sizes, 1, False)
            dwelltime = csa["RealDwellTime"][0] * 1e-9
            values = self._values_vx(ds, csa)
        else:
            floats = np.frombuffer(ds[self.XA_DATA].value, dtype=np.single)
            data = floats[0::2] - 1j * floats[1::2]
            frame = ds.PerFrameFunctionalGroupsSequence[0]
            measures = ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0]
            sizes = [float(measures.PixelSpacing[0]), float(measures.PixelSpacing[1]),
                     float(measures.SliceThickness)]
            affine = SiemensRdaReader.grid_affine(
                self._xa_orientation(ds), frame.PlanePositionSequence[0].ImagePositionPatient,
                sizes, 1, False)
            dwelltime = 1 / ds.SpectralWidth
            values = self._values_xa(ds)
        values["DwellTime"] = (dwelltime, "csa_image/RealDwellTime" if self.software(ds) == "vx"
                               else "dicom/SpectralWidth")
        return Spectrum("", data.reshape(1, 1, 1, -1), dwelltime, affine, values)

    def _csi(self, ds: pydicom.Dataset, path: Path) -> Spectrum:
        if self.software(ds) == "vx":
            csa = self.csa(ds)
            floats = np.frombuffer(ds[self.VX_DATA].value, dtype=np.single)
            data = floats[0::2] + 1j * floats[1::2]
            rows, columns = csa["Rows"][0], csa["Columns"][0]
            slices, points = csa["NumberOfFrames"][0], csa["DataPointColumns"][0]
            if self.VIENNA_CRT.search(str(ds.get("ProtocolName", ""))):
                slices = 1
            data = np.moveaxis(data.reshape(slices, rows, columns, points), (0, 1, 2), (2, 1, 0))
            orientation = np.asarray(csa["ImageOrientationPatient"], dtype=float)
            sizes = [csa["PixelSpacing"][0], csa["PixelSpacing"][1], csa["SliceThickness"][0]]
            affine = SiemensRdaReader.grid_affine(orientation, csa["ImagePositionPatient"], sizes,
                                                  data.shape[2], True)
            dwelltime = csa["RealDwellTime"][0] * 1e-9
            values = self._values_vx(ds, csa)
            voi = [csa["VoiPhaseFoV"][0], csa["VoiReadoutFoV"][0], csa["VoiThickness"][0]]
            box = SiemensRdaReader.grid_affine(orientation, csa["VoiPosition"], voi, 1, False)
            values["VOI"] = (box.tolist(), "csa_image/VoiPosition")
            values["DwellTime"] = (dwelltime, "csa_image/RealDwellTime")
        else:
            self.notes.append("The orientation of CSI stored in enhanced DICOM is weakly tested")
            floats = np.frombuffer(ds[self.XA_DATA].value, dtype=np.single)
            data = floats[0::2] - 1j * floats[1::2]
            shape = (int(ds.NumberOfFrames), ds.Rows, ds.Columns, ds.DataPointColumns)
            data = np.moveaxis(data.reshape(shape), (0, 1, 2), (2, 1, 0))
            frame = ds.PerFrameFunctionalGroupsSequence[0]
            measures = ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0]
            sizes = [float(measures.PixelSpacing[0]), float(measures.PixelSpacing[1]),
                     float(measures.SliceThickness)]
            orientation = self._xa_orientation(ds)
            affine = SiemensRdaReader.grid_affine(
                orientation, frame.PlanePositionSequence[0].ImagePositionPatient, sizes,
                data.shape[2], False)
            dwelltime = 1 / ds.SpectralWidth
            values = self._values_xa(ds)
            slabs = ds.get("VolumeLocalizationSequence")
            if slabs:
                voi = [slabs[1].SlabThickness, slabs[2].SlabThickness, slabs[0].SlabThickness]
                box = SiemensRdaReader.grid_affine(orientation, slabs[0].MidSlabPosition, voi, 1,
                                                   False)
                values["VOI"] = (box.tolist(), "dicom/VolumeLocalizationSequence")
            values["DwellTime"] = (dwelltime, "dicom/SpectralWidth")
        return Spectrum("", data, dwelltime, affine, values)

    def _va(self, ds: pydicom.Dataset, path: Path) -> Spectrum:
        """syngo VA single voxel: data as stored, the voxel from the protocol's sVoI."""
        csa = self.csa(ds)
        protocol = parse_buffer(self.csa(ds, "series")["MrProtocol"][0])
        floats = np.frombuffer(ds[self.VX_DATA].value, dtype="<f4")
        data = (floats[0::2] + 1j * floats[1::2]).reshape(1, 1, 1, -1)
        voi = ("sSpecPara", "sVoI")
        position = [float(protocol.get(voi + ("sPosition", f"d{a}"), 0)) for a in
                    ("Sag", "Cor", "Tra")]
        sizes = [float(protocol[voi + (f"d{k}",)]) for k in ("PhaseFOV", "ReadoutFOV",
                                                              "Thickness")]
        orientation = np.asarray(csa["ImageOrientationPatient"], dtype=float)
        affine = SiemensRdaReader.grid_affine(orientation, position, sizes, 1, False)
        echo = sum(v for k, v in protocol.items() if len(k) == 2 and k[0] == "alTE")
        values = {
            "SpectrometerFrequency": ([float(csa["ImagingFrequency"][0])],
                                      "csa_image/ImagingFrequency"),
            "ResonantNucleus": ([str(csa["ImagedNucleus"][0])], "csa_image/ImagedNucleus"),
            "EchoTime": (echo * 1e-6, "MrProtocol/alTE[0]"),
            "RepetitionTime": (float(csa["RepetitionTime"][0]) * 1e-3,
                               "csa_image/RepetitionTime"),
            "ExcitationFlipAngle": (float(csa["FlipAngle"][0]), "csa_image/FlipAngle"),
            "TxCoil": (str(csa["TransmittingCoil"][0]), "csa_image/TransmittingCoil"),
            "SequenceName": (str(csa["SequenceName"][0]), "csa_image/SequenceName"),
        }
        for key, keyword in (("Manufacturer", "Manufacturer"),
                             ("ManufacturersModelName", "ManufacturerModelName"),
                             ("DeviceSerialNumber", "DeviceSerialNumber"),
                             ("SoftwareVersions", "SoftwareVersions"),
                             ("InstitutionName", "InstitutionName"),
                             ("ProtocolName", "ProtocolName"),
                             ("PatientPosition", "PatientPosition"),
                             ("PatientSex", "PatientSex")):
            if ds.get(keyword) not in (None, ""):
                values[key] = (str(ds.data_element(keyword).value), f"dicom/{keyword}")
        dwelltime = float(csa["RealDwellTime"][0]) * 1e-9
        values |= self._conversion()
        values["DwellTime"] = (dwelltime, "csa_image/RealDwellTime")
        return Spectrum("", data, dwelltime, affine, values)

    def _xa_orientation(self, ds: pydicom.Dataset) -> np.ndarray:
        """Direction cosines of the first frame, else the shared ones; identity, noted, where the
        file does not hold six values."""
        frame, shared = ds.PerFrameFunctionalGroupsSequence[0], ds.SharedFunctionalGroupsSequence[0]
        for groups in (frame, shared):
            if "PlaneOrientationSequence" in groups:
                cosines = np.asarray(groups.PlaneOrientationSequence[0].ImageOrientationPatient,
                                     dtype=float)
                break
        else:
            raise ReaderError("no PlaneOrientationSequence")
        if cosines.size != 6:
            self.notes.append("ImageOrientationPatient does not hold six values: identity "
                              "orientation assumed, the orientation is not known")
            return self.IDENTITY.ravel()
        return cosines

    # --- references ------------------------------------------------------------------------------

    def reference(self, ds: pydicom.Dataset, instance: Any) -> tuple[int, str]:
        """Integrated reference transients of the CMRR sequences (VB to VE): group (0 the scan)
        and name suffix of an instance number."""
        if self.software(ds) == "xa":
            return 0, ""
        protocol = self.protocol(ds)
        if protocol is None:
            return 0, ""
        name = str(protocol[("tSequenceFileName",)]).strip('"').lower()
        mode = protocol.get(("sSpecPara", "lAutoRefScanMode"))
        if mode not in (2.0, 8.0):
            return 0, ""
        refs = int(protocol[("sSpecPara", "lAutoRefScanNo")])
        transients = int(protocol[("lAverages",)])
        if re.search(r"svs_slaser(voi)?_dkd2?$", name) and mode == 8.0:
            total = transients + refs * 4
            if instance <= refs or (total - 2 * refs) < instance <= (total - refs):
                return 1, "_rf_off"
            if instance <= refs * 2 or (total - refs) < instance <= total:
                return 2, "_rf_grads_ovs_off"
            return 0, ""
        if re.search(r"svs_slaser(voi)?_dkd2", name) and mode == 2.0:
            total = transients + refs * 2
            return (1, "_vapor_ovs_rfoff") if instance <= refs or instance > total - refs \
                else (0, "")
        if re.search(r"dkd_svs_mslaser_msspnav$", name) and mode == 8.0:
            total = transients * 2 + refs * 4  # MEGA on
            if instance <= refs:
                return 1, "_ws_ovs_off"
            if transients + refs < instance < transients + refs * 4:
                return 2, "_empty_data"
            return (1, "_ws_ovs_off") if instance > total - refs else (0, "")
        if re.search(r"dkd_svs_slaser_moconav$", name) and mode == 8.0:
            total = transients + refs * 2
            return (1, "_wref") if instance <= refs or instance > total - refs else (0, "")
        return 0, ""

    # --- keys ------------------------------------------------------------------------------------

    def _values_vx(self, ds: pydicom.Dataset, csa: dict) -> dict:
        protocol = self.protocol(ds)
        echo = 0
        for index in range(128):
            if ("alTE", str(index)) not in protocol:
                break
            echo += protocol[("alTE", str(index))]
        values = {
            "SpectrometerFrequency": ([csa["ImagingFrequency"][0]], "csa_image/ImagingFrequency"),
            "ResonantNucleus": ([csa["ImagedNucleus"][0]], "csa_image/ImagedNucleus"),
            "EchoTime": (float(echo * 1e-6), "MrPhoenixProtocol/alTE[0]"),
            "RepetitionTime": (float(csa["RepetitionTime"][0] / 1e3), "csa_image/RepetitionTime"),
            "ExcitationFlipAngle": (float(csa["FlipAngle"][0]), "csa_image/FlipAngle"),
            "TxCoil": (csa["TransmittingCoil"][0], "csa_image/TransmittingCoil"),
            "SequenceName": (csa["SequenceName"][0], "csa_image/SequenceName"),
            "PulseSequenceFile": ({"Value": csa["SequenceName"][0],
                                   "Description": "Sequence binary path."},
                                  "csa_image/SequenceName"),
        }
        if csa.get("InversionTime"):
            values["InversionTime"] = (float(csa["InversionTime"][0]) * 1e-3,
                                       "csa_image/InversionTime")
        if csa.get("ReceivingCoil"):
            values["RxCoil"] = (csa["ReceivingCoil"][0], "csa_image/ReceivingCoil")
        else:
            values["RxCoil"] = (csa["ImaCoilString"][0], "csa_image/ImaCoilString")
        return values | self._dicom_values(ds, patient_id=False)

    def _values_xa(self, ds: pydicom.Dataset) -> dict:
        shared = ds.SharedFunctionalGroupsSequence[0]
        timing = shared.MRTimingAndRelatedParametersSequence[0]
        group = "dicom/SharedFunctionalGroupsSequence/0"
        values = {
            "SpectrometerFrequency": ([float(ds.TransmitterFrequency)],
                                      "dicom/TransmitterFrequency"),
            "ResonantNucleus": ([str(ds.ResonantNucleus)], "dicom/ResonantNucleus"),
            "EchoTime": (float(shared.MREchoSequence[0].EffectiveEchoTime * 1e-3),
                         f"{group}/MREchoSequence/0/EffectiveEchoTime"),
            "RepetitionTime": (float(timing.RepetitionTime) / 1e3,
                               f"{group}/MRTimingAndRelatedParametersSequence/0/RepetitionTime"),
            "ExcitationFlipAngle": (float(timing.FlipAngle),
                                    f"{group}/MRTimingAndRelatedParametersSequence/0/FlipAngle"),
            "TxCoil": (str(shared.MRTransmitCoilSequence[0].TransmitCoilName),
                       f"{group}/MRTransmitCoilSequence/0/TransmitCoilName"),
            "RxCoil": (str(shared.MRReceiveCoilSequence[0].ReceiveCoilName),
                       f"{group}/MRReceiveCoilSequence/0/ReceiveCoilName"),
        }
        if "PulseSequenceName" in ds:
            values["SequenceName"] = (str(ds.PulseSequenceName), "dicom/PulseSequenceName")
        return values | self._dicom_values(ds, patient_id=True)

    def _dicom_values(self, ds: pydicom.Dataset, patient_id: bool) -> dict:
        """Scanner, protocol and patient keys of the top-level elements the file has."""
        values = {}
        for key, keyword in (("Manufacturer", "Manufacturer"),
                             ("ManufacturersModelName", "ManufacturerModelName"),
                             ("DeviceSerialNumber", "DeviceSerialNumber"),
                             ("SoftwareVersions", "SoftwareVersions"),
                             ("InstitutionName", "InstitutionName"),
                             ("InstitutionAddress", "InstitutionAddress"),
                             ("ProtocolName", "ProtocolName"),
                             ("PatientPosition", "PatientPosition"),
                             ("PatientID", "PatientID"),
                             ("PatientDoB", "PatientBirthDate"),
                             ("PatientSex", "PatientSex")):
            if keyword in ds and (patient_id or key != "PatientID"):
                value = ds.data_element(keyword).value
                value = str(value) if key == "DeviceSerialNumber" else self.plain(value)
                values[key] = (value, f"dicom/{keyword}")
        if "PatientName" in ds:
            values["PatientName"] = (str(ds.PatientName.family_name), "dicom/PatientName")
        if "PatientWeight" in ds:
            try:
                values["PatientWeight"] = (float(ds.PatientWeight), "dicom/PatientWeight")
            except (TypeError, ValueError):
                pass
        return values | self._conversion()

    @staticmethod
    def _conversion() -> dict:
        return {"ConversionMethod": (f"miox {__version__}", None),
                "ConversionTime": (datetime.now().isoformat(timespec="milliseconds"), None),
                "kSpace": ([False, False, False], None)}

    @staticmethod
    def plain(value: Any) -> Any:
        """A DICOM value as text, a number or a list of them."""
        if isinstance(value, pydicom.multival.MultiValue):
            return [SiemensDicomMrs.plain(v) for v in value]
        if isinstance(value, (pydicom.valuerep.DSfloat, pydicom.valuerep.DSdecimal)):
            return float(value)
        if isinstance(value, pydicom.valuerep.IS):
            return int(value)
        return value if isinstance(value, (int, float)) else str(value)
