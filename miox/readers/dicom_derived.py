####################################################################################################
#                                         dicom_derived.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 04/10/26                                                                                #
#                                                                                                  #
# Purpose: Standard keys derived from a DICOM image's header: acquisition matrix, acceleration,    #
#          readout timing, phase-encoding direction, slice timing, sequence type and the vendor    #
#          fields of Siemens, Philips, GE and UIH, as dcm2niix derives them.                       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import math
import re
import struct
import zlib
from typing import TYPE_CHECKING, Any

import numpy as np
import pydicom

# own
from miox.meta.meta import Meta

if TYPE_CHECKING:
    from miox.readers.dicom_image import Frame


Tag = str | tuple[int, int] | tuple[int, str, int]  # keyword, (group, element), (group, creator, o)


def find_tag(ds: pydicom.Dataset, index: int | None, tag: Tag) -> Any:
    """Value of a public (keyword), private (group, element) or private-creator-relative
    (group, creator, offset) tag: in frame ``index``'s functional groups, the shared ones
    (private sequences included), else the dataset."""
    items = []
    if index is not None and "PerFrameFunctionalGroupsSequence" in ds:
        items.append(ds.PerFrameFunctionalGroupsSequence[index])
    if "SharedFunctionalGroupsSequence" in ds:
        items.append(ds.SharedFunctionalGroupsSequence[0])
    for item in items:
        for element in item:
            if element.VR != "SQ" or not element.value:
                continue
            found = _element(element.value[0], tag)
            if found is not None:
                return found.value
    found = _element(ds, tag)
    return None if found is None else found.value


def _element(item: pydicom.Dataset, tag: Tag) -> pydicom.DataElement | None:
    if isinstance(tag, str):
        return item.data_element(tag) if tag in item else None
    if len(tag) == 2:
        return item.get(tag)
    group, creator, offset = tag
    try:
        block = item.private_block(group, creator)
    except KeyError:
        creators = [e for e in item if e.tag.group == group and e.tag.is_private_creator]
        return None if creators else item.get((group, 0x1000 | offset))  # no creator at all
    return block[offset] if offset in block else None


def seconds_of_day(text: str) -> float | None:
    """A DICOM time (HHMMSS.FFFFFF) or date-time (YYYYMMDDHHMMSS.FFFFFF) as seconds."""
    text = str(text or "").strip()
    if len(text) > 13 and text[:8].isdigit():
        text = text[8:]
    if len(text) < 6:
        return None
    try:
        return int(text[:2]) * 3600 + int(text[2:4]) * 60 + float(text[4:])
    except ValueError:
        return None


#**************************************************************************************************#
#                                           DerivedKeys                                            #
#**************************************************************************************************#
#                                                                                                  #
# Works on one assembled image: its first frame (``first``), the frames by slice and volume        #
# (``grid[k][v]``), the vendor header of its first file and the data shape. Values come from the   #
# frame's functional groups, the shared ones or the dataset (private sequences included), from     #
# the Siemens CSA headers and protocol, from the GE protocol block or from UIH's per-slice items.  #
# Directions and slice times refer to the image's own voxel axes (columns, rows, slices along the  #
# normal); a negative phase-encoding polarity reads as "-". Every key is written with the DICOM    #
# field or formula it came from.                                                                   #
#                                                                                                  #
#**************************************************************************************************#
class DerivedKeys:
    SOURCE = "DICOM"
    MR_HEADER, SDS, SDI = "SIEMENS MR HEADER", "SIEMENS MR SDS 01", "SIEMENS MR SDI 02"
    SIEMENS_PE_POSITIVE = (0x0021, SDI, 0x1C)  # XA, per frame
    SIEMENS_DWELL = ((0x0019, MR_HEADER, 0x18), (0x0021, SDI, 0x42))  # ns
    SIEMENS_BANDWIDTH_PE = ((0x0019, MR_HEADER, 0x28), (0x0021, SDI, 0x53))  # Hz per pixel, PE
    SIEMENS_COIL_ELEMENTS = (0x0021, SDI, 0x4F)
    SIEMENS_PAT_TEXT = ((0x0051, MR_HEADER, 0x11), (0x0021, SDS, 0x09))  # "p2 s4": PE, slice
    SIEMENS_ORIENTATION_TEXT = (0x0051, MR_HEADER, 0x0E)
    SIEMENS_IMAGE_TYPE_TEXT = (0x0021, SDI, 0x75)
    SIEMENS_ICE_DIMS = (0x0021, SDI, 0x06)  # "X_4_1_1_1_1_160_1_1_1_1_1_277": coil, echo, ...
    SIEMENS_SCANNING_SEQUENCE = (0x0021, SDS, 0x5A)
    SIEMENS_SEQUENCE_VARIANT = (0x0021, SDS, 0x5B)
    SIEMENS_SCAN_OPTIONS = (0x0021, SDS, 0x5C)
    SIEMENS_TIME_AFTER_START = (0x0021, SDI, 0x04)  # s, per frame (XA)
    SIEMENS_B_VALUE = (0x0019, MR_HEADER, 0x0C)  # XA classic
    SIEMENS_GRADIENT = (0x0019, MR_HEADER, 0x0E)  # XA classic, patient (LPS) coordinates
    SIEMENS_PARTIAL_FOURIER = {1: 0.5, 2: 0.625, 4: 0.75, 8: 0.875}  # sKSpace.ucPhasePartialFourier
    SIEMENS_PAT_MODE = {1: "SENSE", 2: "GRAPPA"}
    SIEMENS_COMBINE = {1: "Sum of Squares", 2: "Adaptive Combine"}
    SIEMENS_DIFFUSION_SCHEME = {1: "Bipolar", 2: "Monopolar"}
    SIEMENS_SHIM = (
        ("sGRADSPEC.asGPAData[0].lOffsetX", "sGRADSPEC.lOffsetX"),
        ("sGRADSPEC.asGPAData[0].lOffsetY", "sGRADSPEC.lOffsetY"),
        ("sGRADSPEC.asGPAData[0].lOffsetZ", "sGRADSPEC.lOffsetZ"),
        ("sGRADSPEC.alShimCurrent[0]",),
        ("sGRADSPEC.alShimCurrent[1]",),
        ("sGRADSPEC.alShimCurrent[2]",),
        ("sGRADSPEC.alShimCurrent[3]",),
        ("sGRADSPEC.alShimCurrent[4]",),
    )
    SIEMENS_PROTOCOL = (  # key, protocol field, factor
        ("BaseResolution", "sKSpace.lBaseResolution", None),
        ("TxRefAmp", "sTXSPEC.asNucleusInfo[0].flReferenceAmplitude", None),
        ("PhaseResolution", "sKSpace.dPhaseResolution", None),
        ("PhaseOversampling", "sKSpace.dPhaseOversamplingForDialog", None),
        ("VendorReportedEchoSpacing", "sFastImaging.lEchoSpacing", 1e-6),
        ("DelayTime", "lDelayTimeInTR", 1e-6),
        ("RefLinesPE", "sPat.lRefLinesPE", None),
        ("Interpolation2D", "sKSpace.uc2DInterpolation", None),
        ("PulseSequenceDetails", "tSequenceFileName", None),
        ("WipMemBlock", "sWipMemBlock.tFree", None),
        ("ReceiveCoilActiveElements", "sCoilSelectMeas.sCoilStringForConversion", None),
    )
    GE_ACQU, GE_RELA = "GEMS_ACQU_01", "GEMS_RELA_01"
    GE_SERS, GE_PARM = "GEMS_SERS_01", "GEMS_PARM_01"
    GE_PULSE_SEQUENCE = (0x0019, GE_ACQU, 0x9C)
    GE_INTERNAL_SEQUENCE = (0x0019, GE_ACQU, 0x9E)
    GE_ACQUISITION_DURATION = (0x0019, GE_ACQU, 0x5A)  # µs
    GE_LOCATIONS = (0x0021, GE_RELA, 0x4F)  # slice locations in the acquisition
    GE_PROTOCOL_BLOCK = (0x0025, GE_SERS, 0x1B)
    GE_SHIM = ((0x0043, GE_PARM, 0x02), (0x0043, GE_PARM, 0x03), (0x0043, GE_PARM, 0x04))
    GE_USER_DATA = (0x0043, GE_PARM, 0x2A)
    GE_ECHO_SPACING = (0x0043, GE_PARM, 0x2C)  # µs
    GE_GROUP_DELAY = (0x0043, GE_PARM, 0x7C)  # s
    GE_ASSET = (0x0043, GE_PARM, 0x83)  # 1 / in-plane factor, 1 / slice factor
    GE_PRESCAN = (0x0043, GE_PARM, 0x95)
    GE_MULTIBAND = (0x0043, GE_PARM, 0xB6)
    GE_B_VALUE = (0x0043, GE_PARM, 0x39)  # first value; newer software adds 1e9
    GE_DIRECTIONS = ((0x0019, GE_ACQU, 0xBB), (0x0019, GE_ACQU, 0xBC), (0x0019, GE_ACQU, 0xBD))
    GE_FLIPS = {1: (1, 1, 0), 2: (0, 1, 1), 3: (0, 0, 1)}  # sagittal, coronal, axial stacks
    GE_BLOCK_INTEGERS = ("SLICEORDER", "VIEWORDER", "MBACCEL", "NOSLC")
    GE_BLOCK_KEYS = GE_BLOCK_INTEGERS + ("DELACQ", "DELACQNOAV", "PSEQ", "IOPT")
    GE_VERSION = re.compile(r"([A-Za-z]{2})(\d+)\.(\d+)(?:_([A-Za-z]{2})(\d+))?")
    PHILIPS_IMAGING, PHILIPS_MR = "Philips Imaging DD 001", "Philips MR Imaging DD 001"
    PHILIPS_SCALE_SLOPE = (0x2005, PHILIPS_MR, 0x0E)
    PHILIPS_B_VALUE = (0x2001, PHILIPS_IMAGING, 0x03)
    PHILIPS_GRADIENT = (  # RL, AP, FH
        (0x2005, PHILIPS_MR, 0xB0),
        (0x2005, PHILIPS_MR, 0xB1),
        (0x2005, PHILIPS_MR, 0xB2),
    )
    PHILIPS_EPI_FACTOR = (0x2001, PHILIPS_IMAGING, 0x13)
    PHILIPS_PARTIAL_MATRIX = (0x2001, PHILIPS_IMAGING, 0x19)
    PHILIPS_WATER_FAT_SHIFT = (0x2001, PHILIPS_IMAGING, 0x22)
    PHILIPS_FAT_SHIFT_PPM = 3.4
    UIH_IMAGE = "Image Private Header"
    UIH_PE_DISPLAYED = (0x0065, UIH_IMAGE, 0x05)
    UIH_PARALLEL = (0x0065, UIH_IMAGE, 0x0D)  # "F:2S"
    UIH_GRID_COUNT = (0x0065, UIH_IMAGE, 0x50)  # NumberOfImagesInGrid
    UIH_GRID_ITEMS = (0x0065, UIH_IMAGE, 0x51)  # one item per grid image: position, time, ...
    UIH_PE_POSITIVE = (0x0065, UIH_IMAGE, 0x58)
    UIH_B_VALUE = (0x0065, UIH_IMAGE, 0x09)
    UIH_GRADIENT = (0x0065, UIH_IMAGE, 0x37)  # along the image's column, row and slice axes
    LPS_TO_RAS = np.array([-1.0, -1.0, 1.0])
    MAX_SLICES = 1024
    MS = 1e-3

    def __init__(
        self,
        first: Frame,
        grid: list[list[Frame]],
        header: dict,
        shape: tuple[int, ...],
        slice_times: list[float] | None = None,
    ):
        self.first = first
        self.ds = first.ds
        self.index = first.index
        self.grid = grid
        self.header = header
        self.protocol: dict = header.get("MrPhoenixProtocol", {})
        self.csa_image: dict = header.get("csa_image", {})
        self.columns, self.rows, self.nslices = shape[0], shape[1], shape[2]
        self.nvol = len(grid[0]) if grid and grid[0] else 1
        self.slice_times = slice_times  # UIH grid items, s of day
        manufacturer = str(self.ds.get("Manufacturer", "")).lower()
        self.siemens = "siemens" in manufacturer
        self.philips = "philips" in manufacturer
        self.ge = manufacturer.startswith("ge") or " ge " in f" {manufacturer} "
        self.uih = "uih" in manufacturer or "united imaging" in manufacturer
        self.is_3d = str(self._tag("MRAcquisitionType") or "").upper() == "3D"
        self.simultaneous = 1  # slices sharing the earliest slice time

    def apply(self, meta: Meta) -> None:
        self._acquisition_time(meta)
        self._slice_timing(meta)
        self._sequence(meta)
        self._siemens_protocol(meta)
        self._matrix(meta)
        self._acceleration(meta)
        self._readout(meta)
        self._phase_encoding(meta)
        self._diffusion(meta)
        self._philips(meta)
        self._ge(meta)
        self._misc(meta)

    # --- lookups ---------------------------------------------------------------------------------

    def _tag(self, tag: Tag) -> Any:
        return find_tag(self.ds, self.index, tag)

    def _number(self, tag: Tag) -> float | None:
        value = self._tag(tag)
        if isinstance(value, (list, pydicom.multival.MultiValue)):
            value = value[0] if value else None
        try:
            return None if value in (None, "") else float(value)
        except (TypeError, ValueError):
            return None

    def _text(self, tag: Tag) -> str:
        value = self._tag(tag)
        if isinstance(value, bytes):
            value = value.decode("latin-1", "replace")
        if isinstance(value, (list, pydicom.multival.MultiValue)):
            return "\\".join(str(v) for v in value)
        return "" if value is None else str(value).strip()

    def _texts(self, tag: Tag) -> list[str]:
        value = self._tag(tag)
        if value in (None, ""):
            return []
        if isinstance(value, (list, pydicom.multival.MultiValue)):
            return [str(v) for v in value]
        return [str(value)]

    def _prot(self, key: str) -> Any:
        """A protocol value; strings lose their doubled quotes."""
        value = self.protocol.get(key)
        if isinstance(value, str):
            return value.strip().strip('"')
        return value

    def _prot_number(self, key: str) -> float | None:
        """Protocol numbers; the uc... switches are hexadecimal text (0x1)."""
        value = self._prot(key)
        if isinstance(value, str) and value.lower().startswith("0x"):
            try:
                return float(int(value, 16))
            except ValueError:
                return None
        return float(value) if isinstance(value, (int, float)) else None

    def _set(self, meta: Meta, key: str, value: Any, source: str, field: str | None = None) -> None:
        meta.set(key, value, f"{self.SOURCE}: {source}", field)

    def _set_tag(self, meta: Meta, key: str, value: Any, tag: Tag) -> None:
        """Set ``key`` from a DICOM tag, naming the tag as source and field."""
        if isinstance(tag, str):
            name = tag
        elif len(tag) == 2:
            name = f"({tag[0]:04X},{tag[1]:04X})"
        else:
            name = f"({tag[0]:04X},xx{tag[2]:02X}) {tag[1]}"
        meta.set(key, value, f"{self.SOURCE}: {name}", f"dicom/{name}")

    def _set_protocol(self, meta: Meta, key: str, value: Any, field: str) -> None:
        meta.set(key, value, f"{self.SOURCE}: protocol {field}", f"MrPhoenixProtocol/{field}")

    # --- keys ------------------------------------------------------------------------------------

    def _acquisition_time(self, meta: Meta) -> None:
        """The earliest AcquisitionTime among the files of the first volume."""
        date = self._text("AcquisitionDate")
        if not date and (stamp := self._text("AcquisitionDateTime")):
            date = stamp[:8]
        times = []
        for column in self.grid:
            if column:
                ds = column[0].ds
                stamp = ds.get("AcquisitionTime") or ds.get("AcquisitionDateTime") or ""
                times.append(seconds_of_day(stamp))
        times = [t for t in times if t is not None]
        if not times or len(date) != 8:
            return
        earliest = min(times)
        hours, rest = divmod(earliest, 3600)
        minutes, seconds = divmod(rest, 60)
        clock = f"{int(hours):02d}:{int(minutes):02d}:{seconds:09.6f}"
        self._set_tag(meta, "AcquisitionTime", clock, "AcquisitionTime")
        stamp = f"{date[:4]}-{date[4:6]}-{date[6:]}T{clock}"
        self._set(meta, "AcquisitionDateTime", stamp, "AcquisitionDate + AcquisitionTime")

    def _sequence(self, meta: Meta) -> None:
        if self.siemens:  # XA files state these only in the private series header
            for key, tag in (
                ("ScanningSequence", self.SIEMENS_SCANNING_SEQUENCE),
                ("SequenceVariant", self.SIEMENS_SEQUENCE_VARIANT),
                ("ScanOptions", self.SIEMENS_SCAN_OPTIONS),
            ):
                if key not in meta and (text := self._text(tag)):
                    self._set_tag(meta, key, text, tag)
        pulse = self._text("PulseSequenceName")
        if not pulse and self.ge:
            pulse = self._text(self.GE_PULSE_SEQUENCE)
        if pulse and "SequenceName" not in meta:
            self._set(meta, "SequenceName", pulse, "PulseSequenceName (no SequenceName)")
        elif pulse:
            self._set(meta, "PulseSequenceName", pulse, "PulseSequenceName")
        if self.ge and (internal := self._text(self.GE_INTERNAL_SEQUENCE)):
            self._set_tag(meta, "InternalPulseSequenceName", internal, self.GE_INTERNAL_SEQUENCE)
        scanning = str(meta.get("ScanningSequence", ""))
        variant = str(meta.get("SequenceVariant", ""))
        spoiled = re.search(r"(^|\\)SP($|\\)", variant) is not None
        kind = None
        if "EP" in scanning:
            kind = "Multiband " if max(self._multiband(), self.simultaneous) > 1 else ""
            kind += ("Spin" if "SE" in scanning else "Gradient") + " Echo EPI"
        elif "GR" in scanning:
            if "MP" in variant and "IR" in scanning:
                kind = "MPRAGE"
            else:
                kind = "Spoiled Gradient Echo" if spoiled else "Gradient Echo"
        elif "SE" in scanning:
            kind = "Inversion Recovery Spin Echo" if "IR" in scanning else "Spin Echo"
        if kind:
            self._set(meta, "PulseSequenceType", kind, "ScanningSequence and SequenceVariant")
        self._spoiling(meta, spoiled)
        transfer = self._text("MagnetizationTransfer").upper()
        if transfer.startswith("OF"):
            self._set_tag(meta, "MTState", True, "MagnetizationTransfer")
        elif transfer.startswith(("ON", "NO")):
            self._set_tag(meta, "MTState", False, "MagnetizationTransfer")
        elif self._prot_number("sPrepPulses.ucMTC") == 1:
            self._set_protocol(meta, "MTState", True, "sPrepPulses.ucMTC")
        echo = self._number("EchoNumbers")
        echoes = {frame.echo for column in self.grid for frame in column}
        if echo is not None and (echo > 1 or len(echoes) > 1):
            self._set_tag(meta, "EchoNumber", int(echo), "EchoNumbers")
        types = [str(v).upper() for v in (meta.get("ImageType") or [])]
        text = self._texts(self.SIEMENS_IMAGE_TYPE_TEXT) if self.siemens else []
        if text:
            self._set_tag(meta, "ImageTypeText", text, self.SIEMENS_IMAGE_TYPE_TEXT)
            types += [v.upper() for v in text]
        if any(v in ("DIS2D", "DIS3D") for v in types):
            self._set(meta, "NonlinearGradientCorrection", True, "ImageType DIS2D/DIS3D")
        elif "ND" in types:
            self._set(meta, "NonlinearGradientCorrection", False, "ImageType ND")
        if self.siemens and (orientation := self._text(self.SIEMENS_ORIENTATION_TEXT)):
            self._set_tag(meta, "ImageOrientationText", orientation, self.SIEMENS_ORIENTATION_TEXT)
        if self.uih and (displayed := self._text(self.UIH_PE_DISPLAYED)):
            self._set_tag(meta, "PhaseEncodingDirectionDisplayed", displayed, self.UIH_PE_DISPLAYED)

    def _spoiling(self, meta: Meta, variant_spoiled: bool) -> None:
        spoiling = self._text("Spoiling").upper()
        if not spoiling:
            if variant_spoiled:
                self._set_tag(meta, "SpoilingState", True, "SequenceVariant")
            return
        if "RF" in spoiling and "GRADIENT" in spoiling:
            kind = "COMBINED"
        elif "RF" in spoiling:
            kind = "RF"
        elif "GRADIENT" in spoiling:
            kind = "GRADIENT"
        else:
            kind = None
        self._set_tag(meta, "SpoilingState", kind is not None, "Spoiling")
        if kind:
            self._set_tag(meta, "SpoilingType", kind, "Spoiling")

    def _siemens_protocol(self, meta: Meta) -> None:
        if not self.siemens or not self.protocol:
            return
        for key, field, factor in self.SIEMENS_PROTOCOL:
            value = self._prot(field)
            if value in (None, ""):
                continue
            if isinstance(value, str) and value.lower().startswith("0x"):
                value = self._prot_number(field)
            if isinstance(value, float):
                if factor:
                    value = value * factor
                elif value.is_integer():
                    value = int(value)
            if value is not None:
                self._set_protocol(meta, key, value, field)
        shim = []
        for fields in self.SIEMENS_SHIM:
            field = next((f for f in fields if f in self.protocol), None)
            value = self._prot_number(field) if field else None
            shim.append(0.0 if value is None else value)
        if shim[0]:
            shim = [int(v) if v.is_integer() else v for v in shim]
            self._set(meta, "ShimSetting", shim, "protocol sGRADSPEC offsets and shim currents")
        coil = self._prot("sCoilElementID.tCoilID")
        if coil is None:
            keys = sorted(k for k in self.protocol if k.endswith(".sCoilElementID.tCoilID"))
            coil = self._prot(keys[0]) if keys else None
        if coil:
            self._set_protocol(meta, "RxCoil", coil, "sCoilElementID.tCoilID")
        combine = self._prot_number("ucCoilCombineMode")
        if combine is not None and int(combine) in self.SIEMENS_COMBINE:
            method = self.SIEMENS_COMBINE[int(combine)]
            self._set_protocol(meta, "CoilCombinationMethod", method, "ucCoilCombineMode")
        if "B_value" in self.csa_image:
            field, scheme = "sDiffusion.dsScheme", self._prot_number("sDiffusion.dsScheme")
            if not scheme:  # older software: ucReadOutMode 1 is monopolar, 2 bipolar
                field, mode = "ucReadOutMode", self._prot_number("ucReadOutMode")
                scheme = 3 - int(mode) if mode in (1, 2) else None
            if scheme is not None and int(scheme) in self.SIEMENS_DIFFUSION_SCHEME:
                name = self.SIEMENS_DIFFUSION_SCHEME[int(scheme)]
                self._set_protocol(meta, "DiffusionScheme", name, field)
        for field in (
            "sProtConsistencyInfo.tMeasuredBaselineString",
            "sProtConsistencyInfo.tBaselineString",
        ):
            if info := self._prot(field):
                self._set_protocol(meta, "ConsistencyInfo", info, field)
                break
        partial = self._prot_number("sKSpace.ucPhasePartialFourier")
        if partial is not None and int(partial) in self.SIEMENS_PARTIAL_FOURIER:
            fraction = self.SIEMENS_PARTIAL_FOURIER[int(partial)]
            self._set_protocol(meta, "PartialFourier", fraction, "sKSpace.ucPhasePartialFourier")
        table = self.csa_image.get("ImaRelTablePosition")
        if isinstance(table, list) and len(table) == 3:
            position = [float(table[0]), float(table[1]), -float(table[2])]
            source = f"{self.SOURCE}: CSA ImaRelTablePosition (z negated)"
            meta.set("TablePosition", position, source, "csa_image/ImaRelTablePosition")
        if "RxCoil" not in meta and (coil := self._text("ReceiveCoilName")):
            self._set_tag(meta, "RxCoil", coil, "ReceiveCoilName")
        if "ReceiveCoilActiveElements" not in meta:
            if elements := self._text(self.SIEMENS_COIL_ELEMENTS):
                tag = self.SIEMENS_COIL_ELEMENTS
                self._set_tag(meta, "ReceiveCoilActiveElements", elements, tag)
        name, tag = self._text("ReceiveCoilName"), "ReceiveCoilName"
        ice = self._text(self.SIEMENS_ICE_DIMS).split("_")
        if ice and ice[0].isdigit() and int(ice[0]) > 0:  # one coil element's image
            name, tag = str(int(ice[0])), self.SIEMENS_ICE_DIMS
        if name and name != meta.get("ReceiveCoilActiveElements"):
            self._set_tag(meta, "CoilString", name, tag)

    def _matrix(self, meta: Meta) -> None:
        matrix = self._tag("AcquisitionMatrix")
        lines = 0
        if isinstance(matrix, (list, pydicom.multival.MultiValue)) and len(matrix) == 4:
            lines = int(matrix[3]) or int(matrix[2])
        if lines > 0:
            self._set_tag(meta, "AcquisitionMatrixPE", lines, "AcquisitionMatrix")
        direction = self._text("InPlanePhaseEncodingDirection").upper()[:1]
        if self.rows == self.columns or direction == "C":
            recon = self.rows
        elif direction == "R":
            recon = self.columns
        else:
            recon = lines
        if recon > 0:
            source = "matrix size along the phase-encoding direction"
            self._set(meta, "ReconMatrixPE", int(recon), source)
        steps = self._number("NumberOfPhaseEncodingSteps")
        steps = self._number("MRAcquisitionPhaseEncodingStepsInPlane") if steps is None else steps
        if steps and steps > 0:
            scanned = str(self._tag(self.PHILIPS_PARTIAL_MATRIX) or "").upper()
            if self.philips and scanned.startswith("Y"):
                self._set_tag(meta, "PartialFourierEnabled", "YES", self.PHILIPS_PARTIAL_MATRIX)
                source = "NumberOfPhaseEncodingSteps"
                self._set(meta, "PhaseEncodingStepsNoPartialFourier", int(steps), source)
            else:
                self._set(meta, "PhaseEncodingSteps", int(steps), "NumberOfPhaseEncodingSteps")
        for key, tag in (
            ("FrequencyEncodingSteps", "MRAcquisitionFrequencyEncodingSteps"),
            ("PhaseEncodingStepsOutOfPlane", "MRAcquisitionPhaseEncodingStepsOutOfPlane"),
            ("NumberOfKSpaceTrajectories", "NumberOfKSpaceTrajectories"),
        ):
            value = self._number(tag)
            if value and value > 0:
                self._set_tag(meta, key, int(value), tag)

    def _acceleration(self, meta: Meta) -> None:
        in_plane = self._number("ParallelReductionFactorInPlane")
        source = "ParallelReductionFactorInPlane"
        out_of_plane = self._number("ParallelReductionFactorOutOfPlane")
        source_oop = "ParallelReductionFactorOutOfPlane"
        if self.siemens:
            pat = self._prot_number("sPat.lAccelFactPE")
            if self.csa_image and pat and pat > 0:
                in_plane, source = pat, "protocol sPat.lAccelFactPE"
                mode = int(self._prot_number("sPat.ucPATMode") or 0)
                coil_mode = self.SIEMENS_PAT_MODE.get(mode, "None")
                self._set_protocol(meta, "MatrixCoilMode", coil_mode, "sPat.ucPATMode")
            elif self.csa_image and self.protocol:
                self._set(meta, "MatrixCoilMode", "None", "protocol: no sPat.lAccelFactPE")
            text = self._pat_text()
            if in_plane is None and (match := re.search(r"p(\d+)", text)):
                in_plane, source = float(match.group(1)), "PAT text (0051,1011)"
            accel_3d = self._prot_number("sPat.lAccelFact3D")
            if self.csa_image and accel_3d and accel_3d > 0:
                out_of_plane, source_oop = accel_3d, "protocol sPat.lAccelFact3D"
        if self.uih and (text := self._text(self.UIH_PARALLEL)):
            if match := re.search(r":(\d+(?:\.\d+)?)", text):
                in_plane, source = float(match.group(1)), "(0065,100D)"
        if self.ge:
            asset = self._tag(self.GE_ASSET)
            if isinstance(asset, (list, pydicom.multival.MultiValue)) and len(asset) >= 1:
                if float(asset[0]) > 0:
                    in_plane, source = 1 / float(asset[0]), "(0043,1083): 1 / ASSET phase factor"
                if len(asset) > 1 and float(asset[1]) > 0:
                    out_of_plane = 1 / float(asset[1])
                    source_oop = "(0043,1083): 1 / ASSET slice factor"
        if in_plane is not None and in_plane >= 1:
            self._set(meta, "ParallelReductionFactorInPlane", in_plane, source)
        if out_of_plane is not None and out_of_plane >= 1:
            self._set(meta, "ParallelReductionFactorOutOfPlane", out_of_plane, source_oop)
        if technique := self._text("ParallelAcquisitionTechnique"):
            key = "ParallelAcquisitionTechnique"
            self._set_tag(meta, key, technique, key)
        multiband = self._multiband()
        if multiband > 1:
            self._set(meta, "MultibandAccelerationFactor", multiband, "PAT text or GE (0043,10B6)")
        elif self.simultaneous > 1:
            source = "slices sharing the earliest slice time"
            self._set(meta, "MultibandAccelerationFactor", self.simultaneous, source)
        if direction := self._text("PartialFourierDirection"):
            self._set_tag(meta, "PartialFourierDirection", direction, "PartialFourierDirection")
        if self.siemens and not self.csa_image and not self.is_3d and "PartialFourier" not in meta:
            lines = meta.get("AcquisitionMatrixPE") or meta.get("PhaseEncodingSteps") or 0
            train = int(self._number("EchoTrainLength") or 0)
            if lines > train > 1:
                accel = meta.get("ParallelReductionFactorInPlane") or 1
                fraction = train / (lines / accel if accel > 1 else lines)
                if fraction < 1:
                    source = "EchoTrainLength / phase-encoding lines (estimate)"
                    self._set(meta, "PartialFourier", fraction, source)

    def _pat_text(self) -> str:
        return next((self._text(t) for t in self.SIEMENS_PAT_TEXT if self._text(t)), "")

    def _multiband(self) -> int:
        if self.siemens and (match := re.search(r"s(\d+)", self._pat_text())):
            return int(match.group(1))
        if self.ge:
            value = self._number(self.GE_MULTIBAND)
            if value and value > 1:
                return int(value)
        return 1

    def _readout(self, meta: Meta) -> None:
        bandwidth = None
        if self.siemens:
            values = (self._number(t) for t in self.SIEMENS_BANDWIDTH_PE)
            bandwidth = next((v for v in values if v), None)
            if bandwidth is None and self.csa_image.get("BandwidthPerPixelPhaseEncode"):
                bandwidth = float(self.csa_image["BandwidthPerPixelPhaseEncode"])
            dwell = next((self._number(t) for t in self.SIEMENS_DWELL if self._number(t)), None)
            if dwell:
                self._set(meta, "DwellTime", dwell * 1e-9, "(0019,1018) / (0021,1142) in ns")
        recon = int(meta.get("ReconMatrixPE") or 0)
        spacing = 0.0
        if bandwidth and bandwidth > 0:
            source = "(0019,1028) / (0021,1153) / CSA"
            self._set(meta, "BandwidthPerPixelPhaseEncode", bandwidth, source)
            if recon > 0:
                spacing = 1 / (bandwidth * recon)
        if self.philips:
            self._philips_readout(meta, spacing, recon)
        if self.ge:
            spacing = self._ge_spacing(meta, recon)
        if spacing > 0:
            if self.ge:
                source = "(0043,102C) echo spacing over the acquired lines"
            else:
                source = "1 / (BandwidthPerPixelPhaseEncode * ReconMatrixPE)"
            self._set(meta, "EffectiveEchoSpacing", spacing, source)
            total = spacing * (recon - 1)
            self._set(meta, "TotalReadoutTime", total, "EffectiveEchoSpacing * (ReconMatrixPE - 1)")
        elif self.uih and (duration := self._number("AcquisitionDuration")):
            source = f"{self.SOURCE}: AcquisitionDuration (ms)"
            meta.set("TotalReadoutTime", duration / 1000, source, "dicom/AcquisitionDuration")
        if bandwidth and bandwidth > 0 and recon > 0:
            factor = 1.0
            accel = meta.get("ParallelReductionFactorInPlane") or 0
            if accel > 1:
                factor /= accel
            factor *= 1 + (meta.get("PhaseOversampling") or 0)
            source = ("1 / (BandwidthPerPixelPhaseEncode * ReconMatrixPE / acceleration "
                      "* (1 + PhaseOversampling))")
            spacing = 1 / (bandwidth * factor * recon)
            self._set(meta, "DerivedVendorReportedEchoSpacing", spacing, source)

    def _philips_readout(self, meta: Meta, spacing: float, recon: int) -> None:
        shift = self._number(self.PHILIPS_WATER_FAT_SHIFT)
        if shift:
            self._set_tag(meta, "WaterFatShift", shift, self.PHILIPS_WATER_FAT_SHIFT)
        frequency = self._number("ImagingFrequency") or 0
        epi = self._number("EchoTrainLength") or self._number(self.PHILIPS_EPI_FACTOR) or 0
        if spacing == 0 and shift and frequency > 0 and epi > 0 and recon > 1:
            actual = np.float32(shift / (frequency * self.PHILIPS_FAT_SHIFT_PPM * (epi + 1)))
            total = np.float32(actual * epi)
            formula = ("WaterFatShift / (ImagingFrequency * 3.4 * (EchoTrainLength + 1)) "
                       "* EchoTrainLength")
            effective = float(np.float32(total / (recon - 1)))
            source = f"{formula} / (ReconMatrixPE - 1)"
            self._set(meta, "EstimatedEffectiveEchoSpacing", effective, source)
            self._set(meta, "EstimatedTotalReadoutTime", float(total), formula)

    def _ge_spacing(self, meta: Meta, recon: int) -> float:
        esp = self._number(self.GE_ECHO_SPACING)
        lines = int(meta.get("AcquisitionMatrixPE") or 0)
        accel = meta.get("ParallelReductionFactorInPlane") or 0
        if not esp or esp <= 0 or recon <= 1 or accel <= 0 or lines <= 1:
            return 0.0
        partial = str(self._tag("PartialFourier") or "").upper() == "YES"
        partial = partial or "PFF" in self._text("ScanOptions")
        rounding = 4.0 if partial else 2.0
        acquired = math.ceil(lines / accel / rounding) * rounding
        total = (acquired - 1) * esp * 1e-6
        return total / (lines - 1)

    def _phase_encoding(self, meta: Meta) -> None:
        direction = self._text("InPlanePhaseEncodingDirection").upper()[:1]
        axis = {"R": "i", "C": "j"}.get(direction)
        positive = self._polarity(meta)
        scanning = str(meta.get("ScanningSequence", ""))
        skip = self.is_3d
        train = int(self._number("EchoTrainLength") or 0)
        if (train > 1 or meta.get("BandwidthPerPixelPhaseEncode", 0) > 0) and "SE" not in scanning:
            skip = False
        if axis is None or skip:
            return
        if positive is None:
            source = "InPlanePhaseEncodingDirection (polarity unknown)"
            self._set(meta, "PhaseEncodingAxis", axis, source)
            return
        source = "InPlanePhaseEncodingDirection and the vendor's polarity flag, on these axes"
        self._set(meta, "PhaseEncodingDirection", axis + ("" if positive else "-"), source)

    def _polarity(self, meta: Meta) -> bool | None:
        """True when phase encoding runs along the increasing voxel index, None when unknown."""
        if self.siemens:
            value = self.csa_image.get("PhaseEncodingDirectionPositive")
            value = self._number(self.SIEMENS_PE_POSITIVE) if value in (None, "") else value
            return None if value is None else bool(int(float(value)))
        if self.uih:
            value = self._number(self.UIH_PE_POSITIVE)
            return None if value is None else int(value) == 0
        if self.ge:
            flipped = self._ge_flipped()
            if flipped is not None:
                polarity = "Flipped" if flipped else "Unflipped"
                source = "(0018,9034) or (0043,102A) flags"
                self._set(meta, "PhaseEncodingPolarityGE", polarity, source)
                return not flipped
        return None

    def _ge_flipped(self) -> bool | None:
        reordering = self._text("RectilinearPhaseEncodeReordering").upper()
        if reordering.startswith("L"):
            return True
        if reordering.startswith("R"):
            return False
        data = self._tag(self.GE_USER_DATA)
        if not isinstance(data, bytes) or len(data) < 128:
            return None
        offset = struct.unpack_from("<H", data, 24)[0]
        if len(data) < offset + 916:
            return None
        version = struct.unpack_from("<f", data, offset)[0]
        if not 5.0 <= version <= 40.0:
            return None
        epi_check, polarity_offset, order_offset = 0x3A, 0x30, 0x394
        if version >= 25.002:
            offset += 0x4C
            order_offset -= 0x8C
        if len(data) < offset + order_offset + 2:
            return None
        if not struct.unpack_from("<H", data, offset + epi_check)[0] & 0x800:
            return None
        flipped = bool(struct.unpack_from("<H", data, offset + polarity_offset)[0] & 0x0004)
        if struct.unpack_from("<H", data, offset + order_offset)[0] == 2:  # bottom-up slices
            flipped = not flipped
        return flipped

    def _slice_timing(self, meta: Meta) -> None:
        if self.is_3d or self.nslices < 2 or self.nslices > self.MAX_SLICES:
            return
        times, source = None, ""
        if self.siemens and self.csa_image:
            if int(self.csa_image.get("NumberOfImagesInMosaic") or 0) > 1:
                times = self._plausible(self._mosaic_times, meta)
                source = "CSA MosaicRefAcqTimes"
                if times is None:
                    times = self._siemens_protocol_times(meta)
                    source = "protocol sSliceArray.ucMode and TR"
            else:
                times = self._plausible(self._frame_times, meta)
                source = "AcquisitionTime of the slices of one volume"
        elif self.siemens and self.nvol >= 2:
            times = [self.grid[k][1].after_start for k in range(self.nslices)]
            times = None if any(t is None for t in times) else [float(t) for t in times]
            source = "(0021,1104) time after start of the slices of the second volume"
        elif self.uih and self.nvol >= 2:
            if self.slice_times is not None:
                times, source = list(self.slice_times), "AcquisitionTime of the grid items"
            else:
                times = self._frame_times(0)
                source = "AcquisitionTime of the slices of the first volume"
        elif self.ge:
            times = self._ge_times(meta)
            source = "GE protocol block: slice order, multiband factor and TR"
        if not times or any(t is None for t in times):
            return
        times = [t - min(times) for t in times]
        if len({round(t, 6) for t in times}) < 2:
            return
        self.simultaneous = sum(1 for t in times if abs(t) < 1e-6)
        self._set(meta, "SliceTiming", [float(f"{t:.6g}") for t in times], source)

    def _plausible(self, times_of, meta: Meta) -> list[float] | None:
        """The first volume's slice times, or the second volume's when the first span more than
        TR (the first volume of some multiband series carries corrupt times)."""
        repetition = meta.get("RepetitionTime") or math.inf
        for volume in range(min(self.nvol, 2)):
            times = times_of(volume)
            if times is None:
                continue
            if max(times) - min(times) <= repetition:
                return times
        return None

    def _mosaic_times(self, volume: int) -> list[float] | None:
        """The CSA MosaicRefAcqTimes (ms) of one volume's mosaic file, in seconds."""
        from nibabel.nicom import csareader

        column = self.grid[0]
        if volume >= len(column):
            return None
        tags = csareader.get_csa_header(column[volume].ds, "image")["tags"]
        items = tags.get("MosaicRefAcqTimes", {}).get("items", [])
        if len(items) != self.nslices:
            return None
        return [float(t) * self.MS for t in items]

    def _frame_times(self, volume: int) -> list[float] | None:
        times = []
        for k in range(self.nslices):
            column = self.grid[k]
            times.append(column[volume].time if volume < len(column) else None)
        return None if any(t is None for t in times) else [float(t) for t in times]

    def _siemens_protocol_times(self, meta: Meta) -> list[float] | None:
        """Equidistant times in the protocol's slice order, for mosaics without acquisition times
        (ascending 1, descending 2, interleaved 4)."""
        mode = self._prot_number("sSliceArray.ucMode")
        repetition = meta.get("RepetitionTime")
        if mode not in (1, 2, 4) or not repetition or self._multiband() > 1:
            return None
        delay = (self._prot_number("lDelayTimeInTR") or 0) * 1e-6
        n = self.nslices
        slot = [(repetition - delay) / n * i for i in range(n)]
        if mode == 1:
            return slot
        if mode == 2:
            return slot[::-1]
        odd, even = 0, (n + 1) // 2
        if n % 2 == 0:
            odd, even = even, 0
        times = []
        for i in range(n):
            if i % 2 == 0:
                times.append(slot[odd])
                odd += 1
            else:
                times.append(slot[even])
                even += 1
        return times

    def _ge_times(self, meta: Meta) -> list[float] | None:
        if self.nvol < 2 or "RepetitionTime" not in meta:
            return None
        block = self._ge_block()
        if not block:
            return None
        version = self._ge_version()
        multiband = max(self._multiband(), int(block.get("MBACCEL") or 0), 1)
        sequence = self._text(self.GE_PULSE_SEQUENCE)
        options = str(block.get("IOPT", ""))
        repetition = float(meta["RepetitionTime"])
        delay = self._number(self.GE_GROUP_DELAY) or 0
        interleaved = True
        if self._tag(self.GE_GROUP_DELAY) is not None or "epiRT" in sequence or "FMRI" in options:
            interleaved = block.get("SLICEORDER") != 0
        elif self._text(self.GE_INTERNAL_SEQUENCE) == "EPI" or "MPh" in options:
            if "MPhVar" in options:
                return None
            block_delay = float(block.get("DELACQNOAV") or 0)
            if "MPh" in options and block.get("DELACQ") == "Minimum":
                block_delay = 0.0
            if block_delay > 0:
                repetition += block_delay
                delay = block_delay
                source = "RepetitionTime + protocol block DELACQNOAV delay between volumes"
                self._set(meta, "RepetitionTime", repetition, source, "dicom/RepetitionTime")
        else:
            return None
        if multiband > 1 and (version is None or version[0] < 26.0):
            return None
        excitations = math.ceil(self.nslices / multiband)
        is_27r3 = version is not None and version[1]
        if multiband > 1 and not is_27r3 and excitations % 2 == 0:
            excitations += 1
        per_slice = (repetition - delay) / excitations
        if not interleaved:
            times = [i * per_slice for i in range(excitations)]
        else:
            odd = (excitations - 1) // 2
            times = []
            for i in range(excitations):
                slot = i // 2 if i % 2 == 0 else odd + (i + 1) // 2
                times.append(slot * per_slice)
            if multiband > 1 and is_27r3 and excitations > 2 and excitations % 2 == 0:
                times[-1], times[-3] = times[-3], times[-1]
        return [times[rank % excitations] for rank in self._instance_ranks()]

    def _instance_ranks(self) -> list[int]:
        """For each slice, its rank in the acquisition order (instance numbers) of the first
        volume; a descending acquisition stored ascending ranks its slices in reverse."""
        def instance(k: int) -> tuple[int, int]:
            frame = self.grid[k][0]
            return int(frame.ds.get("InstanceNumber") or 0), frame.index or 0

        ordered = sorted(range(self.nslices), key=instance)
        ranks = [0] * self.nslices
        for rank, k in enumerate(ordered):
            ranks[k] = rank
        return ranks

    def _ge_block(self) -> dict | None:
        """Key-value lines of the compressed protocol block (0025,101B)."""
        raw = self._tag(self.GE_PROTOCOL_BLOCK)
        if not isinstance(raw, bytes):
            return None
        start = raw.find(b"\x1f\x8b\x08")
        if start < 0:
            return None
        try:
            text = zlib.decompressobj(31).decompress(raw[start:]).decode("latin-1", "replace")
        except zlib.error:
            return None
        block = {}
        for key in self.GE_BLOCK_KEYS:
            match = re.search(rf"^{key}\s+\"?([^\"\n]*)\"?\s*$", text, re.M)
            if match is None:
                continue
            value = match.group(1).strip()
            try:
                if key in self.GE_BLOCK_INTEGERS:
                    block[key] = int(value)
                elif key == "DELACQNOAV":
                    block[key] = float(value)
                else:
                    block[key] = value
            except ValueError:
                block[key] = value
        return block

    def _ge_version(self) -> tuple[float, bool] | None:
        """Major.minor version and whether it is 27 release 3 or later, from SoftwareVersions."""
        text = self._text("SoftwareVersions")
        if "release" in text:
            text = text[text.find("release") + 8:]
        else:
            text = text.split("\\")[-1]
        match = self.GE_VERSION.search(text)
        if match is None:
            return None
        major, minor = int(match.group(2)), int(match.group(3))
        release = int(match.group(5)) if match.group(5) and match.group(4) != "EA" else 0
        version = major + 0.1 * minor
        return version, version >= 27.1 or (major == 27 and release >= 3)

    def _diffusion(self, meta: Meta) -> None:
        """B-value and gradient direction of every volume, the direction as a unit vector in
        RAS world coordinates. Siemens, Philips and enhanced files state patient (LPS)
        coordinates, UIH the image's axes, GE its own image-space convention."""
        values, vectors = [], []
        for frame in self.grid[0]:
            b_value, vector = self._diffusion_of(frame)
            values.append(b_value)
            vectors.append(vector)
        if not any(b and b > 0 for b in values):
            return  # not diffusion weighted (GE writes a b-value of 0 on every image)
        if self.ge:
            values, vectors = self._ge_diffusion(values, vectors)
        values = [0.0 if b is None else float(b) for b in values]
        self._set(meta, "DiffusionBValue", values, "b-value of each volume")
        if all(v is None for b, v in zip(values, vectors, strict=True) if b > 0):
            return  # directions stored in a form not read (e.g. in image comments)
        directions = []
        for b_value, vector in zip(values, vectors, strict=True):
            vector = np.zeros(3) if vector is None else np.asarray(vector, float)
            norm = float(np.linalg.norm(vector))
            if b_value <= 0 or norm < 1e-6:
                directions.append([0.0, 0.0, 0.0])
            else:
                directions.append([float(f"{v:.6f}") for v in vector / norm])
        source = "gradient direction of each volume, as a unit vector in RAS world coordinates"
        self._set(meta, "DiffusionGradientOrientation", directions, source)

    def _diffusion_of(self, frame: Frame) -> tuple[float | None, np.ndarray | None]:
        other = DerivedKeys.__new__(DerivedKeys)
        other.ds, other.index = frame.ds, frame.index
        if self.ge:  # image-space components, turned into world vectors by _ge_diffusion
            b_value = other._number(self.GE_B_VALUE)
            b_value = None if b_value is None else float(int(b_value) % 1_000_000_000)
            parts = [other._number(t) for t in self.GE_DIRECTIONS]
            vector = None if any(p is None for p in parts) else np.asarray(parts, float)
            return b_value, vector
        if self.siemens and self.csa_image:
            from nibabel.nicom import csareader

            tags = csareader.get_csa_header(frame.ds, "image")["tags"]
            b_value = tags.get("B_value", {}).get("items") or [None]
            vector = tags.get("DiffusionGradientDirection", {}).get("items") or []
            lps = np.asarray(vector, float) if len(vector) == 3 else None
            return (None if b_value[0] in (None, "") else float(b_value[0])), self._ras(lps)
        b_value = other._number("DiffusionBValue")
        vector = other._nested("DiffusionGradientOrientation")
        vector = other._tag("DiffusionGradientOrientation") if vector is None else vector
        if self.siemens:
            b_value = other._number(self.SIEMENS_B_VALUE) if b_value is None else b_value
            vector = other._tag(self.SIEMENS_GRADIENT) if vector is None else vector
        elif self.philips:
            b_value = other._number(self.PHILIPS_B_VALUE) if b_value is None else b_value
            if vector is None:
                parts = [other._number(t) for t in self.PHILIPS_GRADIENT]
                vector = None if any(p is None for p in parts) else parts
        elif self.uih:
            b_value = other._number(self.UIH_B_VALUE) if b_value is None else b_value
            image = other._tag(self.UIH_GRADIENT)
            if image is not None and len(image) == 3:
                row, column = self.first.orientation[:3], self.first.orientation[3:]
                normal = np.cross(row, column)
                image = np.asarray(image, float)
                vector = image[0] * row + image[1] * column + image[2] * normal
        lps = None if vector is None or len(vector) != 3 else np.asarray(vector, float)
        return b_value, self._ras(lps)

    def _ras(self, lps: np.ndarray | None) -> np.ndarray | None:
        return None if lps is None else lps * self.LPS_TO_RAS

    def _ge_diffusion(
        self, values: list[float | None], vectors: list[np.ndarray | None]
    ) -> tuple[list[float | None], list[np.ndarray | None]]:
        """GE stores directions along its own image axes, with signs that depend on the slice
        orientation and the stacking direction, and encodes lower b-values by shorter vectors.
        The components are turned into world vectors the way dcm2niix validated them."""
        row, column = self.first.orientation[:3], self.first.orientation[3:]
        normal = np.cross(row, column)
        axis = int(np.argmax(np.abs(normal))) + 1  # 1 sagittal, 2 coronal, 3 axial
        stack = self._stack_sign(normal)
        flips = list(self.GE_FLIPS[axis])
        if stack < 0:
            flips[2] = 1 - flips[2]
        column_encoded = self._text("InPlanePhaseEncodingDirection").upper().startswith("C")
        axes = np.array([row, -column, stack * normal])  # dcm2niix's voxel axes, LPS
        out_values, out_vectors = [], []
        for b_value, vector in zip(values, vectors, strict=True):
            if b_value is None or vector is None or not b_value > 0:
                out_values.append(b_value)
                out_vectors.append(None)
                continue
            vector = np.asarray(vector, float)
            length = float(np.linalg.norm(vector))
            if length < 1e-6:
                out_values.append(b_value)
                out_vectors.append(None)
                continue
            if 0.03 < length < 0.97:  # a shorter vector encodes a lower b-value
                scaled = b_value * length * length
                nominal = b_value
                b_value = 5.0 if 0 < scaled < 5 else float(int((scaled + 2.5) / 5) * 5)
                if b_value:
                    vector = vector * math.sqrt(nominal / b_value)
            vector = np.array([-v if f else v for v, f in zip(vector, flips, strict=True)])
            vector[1] = -vector[1]
            if not column_encoded:
                vector[0], vector[1] = vector[1], vector[0]
                vector[0] = -vector[0]
            vector = -vector
            world = vector @ axes
            out_values.append(b_value)
            out_vectors.append(self._ras(world))
        return out_values, out_vectors

    def _stack_sign(self, normal: np.ndarray) -> int:
        """+1 when the files were stacked along the slice normal, -1 when against it."""
        first, last = self.grid[0][0], self.grid[-1][0]
        a = int(first.ds.get("InstanceNumber") or 0), first.index or 0
        b = int(last.ds.get("InstanceNumber") or 0), last.index or 0
        return 1 if a <= b else -1

    def _nested(self, keyword: str) -> Any:
        """A value nested two sequences deep in the frame's or shared functional groups."""
        items = []
        if self.index is not None and "PerFrameFunctionalGroupsSequence" in self.ds:
            items.append(self.ds.PerFrameFunctionalGroupsSequence[self.index])
        if "SharedFunctionalGroupsSequence" in self.ds:
            items.append(self.ds.SharedFunctionalGroupsSequence[0])
        for item in items:
            for element in item:
                if element.VR != "SQ" or not element.value:
                    continue
                for inner in element.value[0]:
                    if inner.VR == "SQ" and inner.value and keyword in inner.value[0]:
                        return inner.value[0].data_element(keyword).value
        return None

    def _philips(self, meta: Meta) -> None:
        if not self.philips:
            return
        slope = self._number("RescaleSlope")
        intercept = self._number("RescaleIntercept")
        scale = self._number(self.PHILIPS_SCALE_SLOPE)
        if scale and slope is not None and not self._scaling_varies():
            self._set_tag(meta, "PhilipsRescaleSlope", slope, "RescaleSlope")
            self._set_tag(meta, "PhilipsRescaleIntercept", intercept or 0.0, "RescaleIntercept")
            self._set_tag(meta, "PhilipsScaleSlope", scale, self.PHILIPS_SCALE_SLOPE)
            source = "values divided by RescaleSlope * (2005,100E)"
            self._set(meta, "UsePhilipsFloatNotDisplayScaling", 1, source)
        real_slope = self._number("RealWorldValueSlope")
        if real_slope:
            self._set_tag(meta, "PhilipsRWVSlope", real_slope, "RealWorldValueSlope")
            intercept = self._number("RealWorldValueIntercept") or 0.0
            self._set_tag(meta, "PhilipsRWVIntercept", intercept, "RealWorldValueIntercept")

    def _scaling_varies(self) -> bool:
        """Enhanced Philips frames may be scaled differently."""
        if self.index is None:
            return False
        seen = set()
        for column in self.grid:
            for frame in column:
                other = DerivedKeys.__new__(DerivedKeys)
                other.ds, other.index = frame.ds, frame.index
                seen.add((
                    other._number("RescaleSlope"),
                    other._number("RescaleIntercept"),
                    other._number(self.PHILIPS_SCALE_SLOPE),
                ))
                if len(seen) > 1:
                    return True
        return False

    def _ge(self, meta: Meta) -> None:
        if not self.ge:
            return
        shim = [self._number(t) for t in self.GE_SHIM]
        if shim[0] is not None and shim[0] > -33333:
            values = [int(v) for v in shim if v is not None]
            self._set(meta, "ShimSetting", values, "(0043,1002-1004)")
        if prescan := self._text(self.GE_PRESCAN):
            self._set_tag(meta, "PrescanReuseString", prescan, self.GE_PRESCAN)
        duration = self._number(self.GE_ACQUISITION_DURATION)
        if duration and duration > 0 and "AcquisitionDuration" not in meta:
            source = f"{self.SOURCE}: (0019,105A) in µs"
            meta.set("AcquisitionDuration", duration * 1e-6, source, "dicom/(0019,105A)")
        block = self._ge_block()
        if block and block.get("PSEQ"):
            step = str(meta.get("ProcedureStepDescription") or "") + str(block["PSEQ"])
            source = "PerformedProcedureStepDescription + protocol block PSEQ"
            self._set(meta, "ProcedureStepDescription", step, source)

    def _misc(self, meta: Meta) -> None:
        sar = self._number("SAR")
        if sar and sar > 0:
            self._set_tag(meta, "SAR", sar, "SAR")
        duration = self._number("AcquisitionDuration")
        if duration and duration > 0 and not self.uih and "AcquisitionDuration" not in meta:
            self._set_tag(meta, "AcquisitionDuration", duration, "AcquisitionDuration")
        if comments := self._text("ImageComments"):
            self._set_tag(meta, "ImageComments", comments, "ImageComments")
