####################################################################################################
#                                             sniff.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 01/10/26                                                                                #
#                                                                                                  #
# Purpose: Format sniffer. Says what a file or directory is from magic bytes and structure without #
#          reading it whole: every rule reads at most 4 KB, and the confidence says whether bytes  #
#          or only the file name decided.                                                          #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import dataclasses
import gzip
import json
import re
import struct
import warnings
import zlib
from dataclasses import KW_ONLY, dataclass, field
from enum import StrEnum
from pathlib import Path

import nibabel as nib
import pydicom

# own
from miox.matfile import MatFile


#**************************************************************************************************#
#                                              Format                                              #
#**************************************************************************************************#
#                                                                                                  #
# The file formats the sniffer recognises.                                                         #
#                                                                                                  #
#**************************************************************************************************#
class Format(StrEnum):
    SIEMENS_TWIX = "siemens_twix"
    SIEMENS_RDA = "siemens_rda"
    PHILIPS_SPAR_SDAT = "philips_spar_sdat"
    PHILIPS_DATA_LIST = "philips_data_list"
    GE_PFILE = "ge_pfile"
    DICOM_MRS = "dicom_mrs"
    DICOM_IMAGE = "dicom_image"
    DICOM_OTHER = "dicom_other"
    NIFTI_MRS = "nifti_mrs"
    NIFTI = "nifti"
    BRUKER = "bruker"
    VARIAN = "varian"
    LCMODEL_RAW = "lcmodel_raw"
    LCMODEL_BASIS = "lcmodel_basis"
    LCMODEL_CONTROL = "lcmodel_control"
    LCMODEL_COORD = "lcmodel_coord"
    LCMODEL_TABLE = "lcmodel_table"
    LCMODEL_CSV = "lcmodel_csv"
    LCMODEL_PRINT = "lcmodel_print"
    LCMODEL_PS = "lcmodel_ps"
    TARQUIN_DPT = "tarquin_dpt"
    TARQUIN_RESULTS = "tarquin_results"
    TARQUIN_FIT = "tarquin_fit"
    VASF = "vasf"
    VESPA_VIFF = "vespa_viff"
    SIVIC_DDF = "sivic_ddf"
    SIVIC_IDF = "sivic_idf"
    FSL_MRS_BASIS = "fsl_mrs_basis"
    SPINWIZARD_BASIS = "spinwizard_basis"
    OSPREY_BASIS = "osprey_basis"
    INSPECTOR_BASIS = "inspector_basis"
    MARSS_BASIS = "marss_basis"
    PROFIT_BASIS = "profit_basis"
    FIDA_MAT = "fida_mat"
    BASIS_FOLDER = "basis_folder"
    JMRUI_TEXT = "jmrui_text"
    JMRUI_BINARY = "jmrui_binary"
    ISMRMRD = "ismrmrd"
    UNKNOWN = "unknown"


#**************************************************************************************************#
#                                               Kind                                               #
#**************************************************************************************************#
#                                                                                                  #
# What the data is, as far as the sniffer can tell.                                                #
#                                                                                                  #
#**************************************************************************************************#
class Kind(StrEnum):
    SPECTROSCOPY = "spectroscopy"
    IMAGING = "imaging"
    RAW = "raw"  # k-space / unprocessed ADC lines (twix, P-file, Bruker fid)
    BASIS_SET = "basis_set"  # one signal per metabolite, for fitting
    FIT_RESULT = "fit_result"  # concentrations, fitted curves
    PROTOCOL = "protocol"  # acquisition or analysis settings
    REPORT = "report"  # graphics for reading, no data
    UNKNOWN = "unknown"


#**************************************************************************************************#
#                                              Sniff                                               #
#**************************************************************************************************#
#                                                                                                  #
# The result of sniffing one path: format, kind, vendor, how sure, why, and the files that belong  #
# with it.                                                                                         #
#                                                                                                  #
#**************************************************************************************************#
@dataclass(frozen=True)
class Sniff:
    path: Path
    format: Format
    kind: Kind = Kind.UNKNOWN
    _: KW_ONLY
    vendor: str | None = None
    confidence: float = 0.0
    evidence: tuple[str, ...] = ()
    companions: tuple[Path, ...] = ()
    detail: dict = field(default_factory=dict)

    def __str__(self) -> str:
        vendor = f" {self.vendor}" if self.vendor else ""
        return f"{self.path}: {self.format}{vendor} ({self.kind}, {self.confidence:.1f})"

    def to_dict(self) -> dict:
        """JSON-ready record, paths as strings."""
        record = dataclasses.asdict(self)
        record["path"] = str(self.path)
        record["companions"] = [str(c) for c in self.companions]
        return record


#**************************************************************************************************#
#                                             Sniffer                                              #
#**************************************************************************************************#
#                                                                                                  #
# Identifies files and directories. A file goes through the rules in order, most specific first;   #
# each rule looks at the first bytes and returns a Sniff or None. A directory is decided by its    #
# contents (Bruker scan, Varian .fid folder, DICOM series).                                        #
#                                                                                                  #
#**************************************************************************************************#
class Sniffer:
    HEAD_BYTES = 4096  # bytes read from the start of a file
    DIR_PROBE_LIMIT = 50  # files probed for DICOM in a directory

    DICOM_SOP_MRS = {
        "1.2.840.10008.5.1.4.1.1.4.2": "MR Spectroscopy Storage",
        "1.3.12.2.1107.5.9.1": "Siemens CSA Non-Image Storage",
        "1.3.46.670589.11.0.0.12.1": "Philips Private MR Spectrum Storage",
        "1.3.46.670589.11.0.0.12.2": "Philips Private MR Series",
    }
    DICOM_SOP_IMAGE = {
        "1.2.840.10008.5.1.4.1.1.4": "MR Image Storage",
        "1.2.840.10008.5.1.4.1.1.4.1": "Enhanced MR Image Storage",
        "1.2.840.10008.5.1.4.1.1.4.3": "Enhanced MR Color Image Storage",
        "1.2.840.10008.5.1.4.1.1.4.4": "Legacy Converted Enhanced MR Image Storage",
    }
    VENDORS = (  # vendor name and the pattern that finds it in a Manufacturer string
        ("siemens", re.compile(r"siemens", re.I)),
        ("philips", re.compile(r"philips", re.I)),
        ("ge", re.compile(r"^ge\b|ge medical", re.I)),
        ("bruker", re.compile(r"bruker", re.I)),
        ("uih", re.compile(r"uih|united imaging", re.I)),
        ("canon", re.compile(r"canon|toshiba", re.I)),
        ("hyperfine", re.compile(r"hyperfine", re.I)),
        ("varian", re.compile(r"varian|agilent", re.I)),
    )
    NIFTI_MRS_INTENT = re.compile(r"^mrs_v(\d+)_(\d+)$")  # intent_name, e.g. mrs_v0_2
    NIFTI_MRS_DIM_TAG = re.compile(r"^dim_[5-7]$")  # header extension keys of the tagged dims
    SPAR_KEYS = (b"examination_name", b"scan_id", b"synthesizer_frequency", b"samples")
    TWIX_VB_BUFFERS = (b"Config", b"Dicom", b"Meas", b"MeasYaps", b"Phoenix", b"Spice")
    PFILE_REVISIONS = (7, 8, 9, 10, 11, 12, 14, 15, 16, 20, 24, 26, 27, 28, 30)
    PFILE_NAME = re.compile(r"^P\d{5}\.7$", re.I)
    BRUKER_PARAM_FILES = ("acqp", "method", "visu_pars", "reco", "subject")
    BRUKER_DATA_FILES = re.compile(
        r"^(fid|ser|2dseq|rawdata\.job\d+|fid\.refscan|rawdata\.Navigator"
        r"|fid_proc\.64|fid_refscan\.64)$"
    )
    BRUKER_PROCESSED = ("2dseq", "fid_proc.64", "fid_refscan.64")  # in <scan>/pdata/<n>/
    BRUKER_ACQP_FIELDS = {  # detail key -> acqp parameter
        "paravision_version": "ACQ_sw_version",
        "scan_name": "ACQ_scan_name",
        "method": "ACQ_method",
    }
    TEXT_MARKERS = (  # (markers, format, kind, evidence); first match wins, upper-case markers
        ((b"$BASIS", b"&BASIS"), Format.LCMODEL_BASIS, Kind.BASIS_SET, "LCModel $BASIS namelist"),
        ((b"$LCMODL", b"&LCMODL"), Format.LCMODEL_CONTROL, Kind.PROTOCOL, "LCModel $LCMODL"),
        ((b"$NMALL", b"$NMEACH"), Format.LCMODEL_CONTROL, Kind.PROTOCOL, "MakeBasis $NMALL"),
        ((b"$NMID", b"$SEQPAR"), Format.LCMODEL_RAW, Kind.SPECTROSCOPY, "LCModel $NMID/$SEQPAR"),
    )
    TEXT_STARTS = (  # (start of the file, format, kind, evidence)
        (b"Dangerplot_version", Format.TARQUIN_DPT, Kind.SPECTROSCOPY, "TARQUIN dangerplot"),
        (b"Signal     Conc (au)", Format.TARQUIN_RESULTS, Kind.FIT_RESULT, "TARQUIN results"),
        (b"Row : ", Format.TARQUIN_FIT, Kind.FIT_RESULT, "TARQUIN fit"),
        (b"Row, Col,", Format.LCMODEL_CSV, Kind.FIT_RESULT, "LCModel CSV"),
        (b"[FILE INFORMATION]", Format.VASF, Kind.SPECTROSCOPY, "VASF header"),
        (b"DATA DESCRIPTOR FILE", Format.SIVIC_DDF, Kind.SPECTROSCOPY, "UCSF DDF header"),
        (b"IMAGE DESCRIPTOR FILE", Format.SIVIC_IDF, Kind.IMAGING, "UCSF IDF header"),
    )
    MATLAB_VARIABLES = (  # (variable name, format, kind); first match wins
        (re.compile(r"^BASIS$"), Format.OSPREY_BASIS, Kind.BASIS_SET),
        (re.compile(r"^lcmBasis$"), Format.INSPECTOR_BASIS, Kind.BASIS_SET),
        (re.compile(r"^exptDat$"), Format.MARSS_BASIS, Kind.BASIS_SET),
        (re.compile(r"^basis$"), Format.PROFIT_BASIS, Kind.BASIS_SET),
        (re.compile(r"^out"), Format.FIDA_MAT, Kind.SPECTROSCOPY),  # out, outw, outA, ...
    )
    BASIS_MEMBERS = (
        Format.LCMODEL_RAW,
        Format.JMRUI_TEXT,
        Format.FSL_MRS_BASIS,
        Format.NIFTI_MRS,
        Format.FIDA_MAT,
        Format.MARSS_BASIS,
    )
    METABOLITES = (  # lower-case abbreviations and names used for basis-set files
        "naa", "naag", "cr", "pcr", "tcr", "cre", "creatine", "cho", "choline", "gpc", "pch",
        "pcho", "tcho", "ins", "mi", "mins", "myo", "si", "sins", "scyllo", "glu", "glutamate",
        "gln", "glutamine", "glx", "gaba", "gsh", "lac", "ala", "asp", "asc", "ace", "tau", "glc",
        "gly", "pe", "ser", "cit", "2hg", "lip", "mm", "mac", "bhb", "etoh", "suc", "thr", "val",
        "pyr", "h2o", "water", "his", "phe", "trp", "tyr", "ala", "ac", "acetate", "lys",
    )
    COMPANIONS = (  # (data suffix, header suffix, format)
        (".rsd", ".rsp", Format.VASF),
        (".cmplx", ".ddf", Format.SIVIC_DDF),
        (".real", ".idf", Format.SIVIC_IDF),
        (".int2", ".idf", Format.SIVIC_IDF),
        (".byt", ".idf", Format.SIVIC_IDF),
    )
    BRUKER_MRS_METHODS = (
        "press", "steam", "csi", "svs", "slaser", "laser", "isis", "fid", "nspect"
    )

    def __init__(self):
        self.rules = (
            self.dicom,
            self.nifti,
            self.rda,
            self.matlab,
            self.companion_binary,
            self.analysis_text,
            self.vespa,
            self.spar_sdat,
            self.data_list,
            self.twix,
            self.pfile,
            self.bruker,
            self.varian,
            self.text_formats,
            self.hdf5,
            self.extension_only,
            self.dicom_no_preamble,
        )

    def sniff(self, path: str | Path) -> Sniff:
        """Identify the format of ``path`` (a file or a directory)."""
        path = Path(path)
        if path.is_dir():
            return self.directory(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        head = self._head(path)
        for rule in self.rules:
            result = rule(path, head)
            if result is not None:
                return result
        return Sniff(path, Format.UNKNOWN)

    def directory(self, path: Path) -> Sniff:
        """A Bruker scan, a Varian .fid folder, a basis set of one file per metabolite, or a
        DICOM series."""
        names = {p.name for p in path.iterdir()}
        if names & set(self.BRUKER_PARAM_FILES):
            return self._bruker_scan(path, path, "directory with Bruker parameter files")
        if "procpar" in names:
            return self._varian_scan(path, path, "directory with procpar")
        files = [c for c in sorted(path.iterdir()) if c.is_file() and c.name != ".DS_Store"]
        if "BasisSetParameters.txt" in names:
            evidence = ("SpinWizard BasisSetParameters.txt",)
            return Sniff(
                path,
                Format.SPINWIZARD_BASIS,
                Kind.BASIS_SET,
                confidence=1.0,
                evidence=evidence,
                companions=tuple(files),
            )
        if basis := self._basis_folder(path, files):
            return basis
        for child in files[: self.DIR_PROBE_LIMIT]:
            if not self._is_dicom(self._head(child)):
                continue
            first = self._read_dicom(child)
            return dataclasses.replace(
                first,
                path=path,
                evidence=("directory; first DICOM file decides",) + first.evidence,
                companions=(child,) + tuple(c for c in files if c != child),
            )
        return Sniff(path, Format.UNKNOWN, evidence=("directory with no recognised content",))

    # --- file rules ------------------------------------------------------------------------------

    def dicom(self, path: Path, head: bytes) -> Sniff | None:
        if self._is_dicom(head):
            return self._read_dicom(path)
        return None

    def nifti(self, path: Path, head: bytes) -> Sniff | None:
        gzipped = head[:2] == b"\x1f\x8b"
        if gzipped:
            try:
                head = zlib.decompressobj(31).decompress(head, 1024)
            except zlib.error:
                return None
        if len(head) < 348:
            return None
        if head[344:348] in (b"n+1\0", b"ni1\0"):
            version, header_class = 1, nib.Nifti1Header
        elif head[4:8] in (b"n+2\0", b"ni2\0"):
            version, header_class = 2, nib.Nifti2Header
        else:
            return None
        with (gzip.open if gzipped else open)(path, "rb") as fh:
            header = header_class.from_fileobj(fh, check=False)
        detail = {"nifti_version": version, "shape": tuple(int(n) for n in header.get_data_shape())}
        extension = next((e for e in header.extensions if e.get_code() == 44), None)
        if extension is None:
            return Sniff(
                path,
                Format.NIFTI,
                Kind.IMAGING,
                confidence=1.0,
                evidence=(f"NIfTI-{version} magic",),
                detail=detail,
            )
        intent_name = header["intent_name"].item().decode("ascii", "replace")
        intent = self.NIFTI_MRS_INTENT.match(intent_name)
        detail["nifti_mrs_version"] = f"{intent[1]}.{intent[2]}" if intent else ""
        mrs = json.loads(extension.get_content())
        detail["dim_tags"] = {k: v for k, v in mrs.items() if self.NIFTI_MRS_DIM_TAG.match(k)}
        return Sniff(
            path,
            Format.NIFTI_MRS,
            Kind.SPECTROSCOPY,
            confidence=1.0,
            evidence=(f"NIfTI-{version} magic", "header extension ecode 44"),
            detail=detail,
        )

    def rda(self, path: Path, head: bytes) -> Sniff | None:
        if not head.startswith(b">>> Begin of header <<<"):
            return None
        return Sniff(
            path,
            Format.SIEMENS_RDA,
            Kind.SPECTROSCOPY,
            vendor="siemens",
            confidence=1.0,
            evidence=("RDA header marker",),
        )

    def spar_sdat(self, path: Path, head: bytes) -> Sniff | None:
        if head.startswith(b"!---") or sum(k in head for k in self.SPAR_KEYS) >= 2:
            sdat = self._sibling(path, ".sdat")
            return Sniff(
                path,
                Format.PHILIPS_SPAR_SDAT,
                Kind.SPECTROSCOPY,
                vendor="philips",
                confidence=1.0 if sdat else 0.9,
                evidence=(
                    "SPAR text header",
                    "SDAT companion found" if sdat else "no SDAT companion",
                ),
                companions=(sdat,) if sdat else (),
            )
        if path.suffix.lower() == ".sdat":
            spar = self._sibling(path, ".spar")
            return Sniff(
                path,
                Format.PHILIPS_SPAR_SDAT,
                Kind.SPECTROSCOPY,
                vendor="philips",
                confidence=1.0 if spar else 0.5,
                evidence=(
                    ".sdat extension",
                    "SPAR companion found" if spar else "no SPAR companion",
                ),
                companions=(spar,) if spar else (),
            )
        return None

    def data_list(self, path: Path, head: bytes) -> Sniff | None:
        if head.startswith(b"# === DATA DESCRIPTION FILE"):
            data = self._sibling(path, ".data")
            return Sniff(
                path,
                Format.PHILIPS_DATA_LIST,
                Kind.RAW,
                vendor="philips",
                confidence=1.0 if data else 0.9,
                evidence=(
                    "LIST header marker",
                    "DATA companion found" if data else "no DATA companion",
                ),
                companions=(data,) if data else (),
            )
        if path.suffix.lower() == ".data":
            lst = self._sibling(path, ".list")
            if lst:
                return Sniff(
                    path,
                    Format.PHILIPS_DATA_LIST,
                    Kind.RAW,
                    vendor="philips",
                    confidence=1.0,
                    evidence=(".data extension", "LIST companion found"),
                    companions=(lst,),
                )
        return None

    def twix(self, path: Path, head: bytes) -> Sniff | None:
        if len(head) < 16:
            return None
        size = path.stat().st_size
        first, second = struct.unpack_from("<II", head, 0)
        if first == 0 and 1 <= second <= 64:
            # VD/VE/XA: 0, n_measurements, then per measurement (MeasID, FileID, offset, length)
            offset, length = struct.unpack_from("<QQ", head, 16)
            if offset < size and 0 < length <= size:
                return Sniff(
                    path,
                    Format.SIEMENS_TWIX,
                    Kind.RAW,
                    vendor="siemens",
                    confidence=1.0,
                    evidence=("VD/VE/XA multi-measurement header",),
                    detail={"twix_layout": "VD", "n_measurements": second},
                )
        # VB: header length, number of buffers, then the first buffer name
        name = head[8:40].split(b"\0")[0]
        if first < size and 1 <= second <= 64 and name in self.TWIX_VB_BUFFERS:
            return Sniff(
                path,
                Format.SIEMENS_TWIX,
                Kind.RAW,
                vendor="siemens",
                confidence=1.0,
                evidence=("VB header: length + buffer names",),
                detail={"twix_layout": "VB", "n_buffers": second},
            )
        return None

    def pfile(self, path: Path, head: bytes) -> Sniff | None:
        if len(head) < 4:
            return None
        if self._is_text(head):
            return None
        (revision,) = struct.unpack_from("<f", head, 0)
        named = path.suffix == ".7" or self.PFILE_NAME.match(path.name) is not None
        known = any(abs(revision - r) < 0.01 for r in self.PFILE_REVISIONS)
        exact = abs(revision - round(revision, 3)) < 1e-5  # revisions are like 20.006
        if not exact or not (known or (named and 5.0 <= revision <= 40.0)):
            return None
        return Sniff(
            path,
            Format.GE_PFILE,
            Kind.RAW,
            vendor="ge",
            confidence=1.0 if named else 0.8,
            evidence=(f"P-file revision {revision:g} at offset 0",)
            + ((".7 name",) if named else ()),
            detail={"pfile_revision": round(revision, 2)},
        )

    def bruker(self, path: Path, head: bytes) -> Sniff | None:
        if head.startswith(b"##TITLE=") and b"##JCAMP" in head:
            if path.name in self.BRUKER_PARAM_FILES:
                return self._bruker_scan(path, path.parent, f"JCAMP-DX parameter file {path.name}")
            if b"##$" in head:
                return self._bruker_scan(path, path.parent, "JCAMP-DX file with Bruker parameters")
        if self.BRUKER_DATA_FILES.match(path.name):
            folder = path.parent
            scan_dir = folder.parent.parent if path.name in self.BRUKER_PROCESSED else folder
            if (folder / "visu_pars").is_file() or any(
                (scan_dir / f).is_file() for f in ("acqp", "method")
            ):
                return self._bruker_scan(
                    path, scan_dir, f"Bruker data file {path.name} with parameters"
                )
        return None

    def varian(self, path: Path, head: bytes) -> Sniff | None:
        if path.name == "procpar" or (path.name == "fid" and (path.parent / "procpar").is_file()):
            return self._varian_scan(path, path.parent, f"{path.name} with procpar")
        return None

    def text_formats(self, path: Path, head: bytes) -> Sniff | None:
        if b"jMRUI Data Textfile" in head:
            return Sniff(
                path,
                Format.JMRUI_TEXT,
                Kind.SPECTROSCOPY,
                confidence=1.0,
                evidence=("jMRUI text marker",),
            )
        return None

    def companion_binary(self, path: Path, head: bytes) -> Sniff | None:
        """Headerless binary data next to the text header that describes it."""
        for suffix, header, fmt in self.COMPANIONS:
            if path.suffix.lower() == suffix and (text := self._sibling(path, header)):
                return Sniff(
                    path,
                    fmt,
                    Kind.SPECTROSCOPY if fmt != Format.SIVIC_IDF else Kind.IMAGING,
                    confidence=0.9,
                    evidence=(f"{suffix} data with {text.name}",),
                    companions=(text,),
                )
        return None

    def analysis_text(self, path: Path, head: bytes) -> Sniff | None:
        """Text formats of LCModel, TARQUIN, VASF and SIVIC, recognised by their own markers."""
        upper = head.upper()
        stripped = head.lstrip()
        if stripped.startswith(b"{") and b'"basis_re"' in head:
            evidence = ("FSL-MRS basis JSON",)
            fmt = Format.FSL_MRS_BASIS
            return Sniff(path, fmt, Kind.BASIS_SET, confidence=1.0, evidence=evidence)
        if stripped.startswith(b"%!PS") and b"Creator: LCModel" in head:
            evidence = ("LCModel PostScript",)
            return Sniff(path, Format.LCMODEL_PS, Kind.REPORT, confidence=1.0, evidence=evidence)
        for markers, fmt, kind, evidence in self.TEXT_MARKERS:
            if any(m in upper for m in markers):
                companions = self._text_companions(path, fmt)
                return Sniff(
                    path, fmt, kind, confidence=1.0, evidence=(evidence,), companions=companions
                )
        if stripped.startswith(b"LCModel (Version"):
            if b"$$CONC" in head:
                fmt = Format.LCMODEL_TABLE
            elif b"lines in following" in head:
                fmt = Format.LCMODEL_COORD
            else:
                fmt = Format.LCMODEL_PRINT
            return Sniff(path, fmt, Kind.FIT_RESULT, confidence=0.9, evidence=("LCModel banner",))
        for start, fmt, kind, evidence in self.TEXT_STARTS:
            if stripped.startswith(start):
                companions = self._text_companions(path, fmt)
                return Sniff(
                    path, fmt, kind, confidence=0.9, evidence=(evidence,), companions=companions
                )
        return None

    def vespa(self, path: Path, head: bytes) -> Sniff | None:
        if head[:2] == b"\x1f\x8b":
            try:
                head = zlib.decompressobj(31).decompress(head, 4096)
            except zlib.error:
                return None
        if b"<vespa_export" not in head:
            return None
        kind = Kind.UNKNOWN
        if b"<dataset" in head:
            kind = Kind.SPECTROSCOPY
        elif b"<experiment" in head or b"<metabolite" in head:
            kind = Kind.BASIS_SET
        return Sniff(path, Format.VESPA_VIFF, kind, confidence=1.0, evidence=("<vespa_export>",))

    def _basis_folder(self, path: Path, files: list[Path]) -> Sniff | None:
        """Several files of one spectroscopy format, most named after metabolites."""
        members = {}
        for child in files[: self.DIR_PROBE_LIMIT]:
            fmt = self.sniff(child).format
            if fmt in self.BASIS_MEMBERS:
                members.setdefault(fmt, []).append(child)
        if not members:
            return None
        fmt, paths = max(members.items(), key=lambda item: len(item[1]))
        named = sum(self.is_metabolite(p.name.split(".")[0]) for p in paths)
        if len(paths) < 2 or named < len(paths) / 2:
            return None
        evidence = (f"{len(paths)} {fmt} files, {named} named after metabolites",)
        detail = {"member_format": str(fmt)}
        return Sniff(
            path,
            Format.BASIS_FOLDER,
            Kind.BASIS_SET,
            confidence=0.8,
            evidence=evidence,
            companions=tuple(paths),
            detail=detail,
        )

    @classmethod
    def is_metabolite(cls, name: str) -> bool:
        """``name`` starts with a metabolite or macromolecule abbreviation (NAA, Cr391, MM09)."""
        name = name.lower()
        for known in cls.METABOLITES:
            if name == known or (name.startswith(known) and len(known) >= 3):
                return True
            if name.startswith(known) and not name[len(known)].isalpha():
                return True
        return False

    def _text_companions(self, path: Path, fmt: Format) -> tuple[Path, ...]:
        for suffix, header, companion_fmt in self.COMPANIONS:
            if companion_fmt == fmt and path.suffix.lower() == header:
                data = self._sibling(path, suffix)
                return (data,) if data else ()
        return ()

    def matlab(self, path: Path, head: bytes) -> Sniff | None:
        """MATLAB files, by the variable names the tools save."""
        if not MatFile.is_mat(head):
            return None
        try:
            variables = MatFile.variables(path)
        except Exception as e:
            return Sniff(path, Format.UNKNOWN, evidence=(f"MATLAB file, unreadable: {e}",))
        for pattern, fmt, kind in self.MATLAB_VARIABLES:
            names = [name for name in variables if pattern.match(name)]
            if names:
                return Sniff(
                    path,
                    fmt,
                    kind,
                    confidence=0.8,
                    evidence=(f"MATLAB variable {names[0]}",),
                    detail={"variables": variables},
                )
        return Sniff(path, Format.UNKNOWN, evidence=(f"MATLAB file with {sorted(variables)}",))

    def hdf5(self, path: Path, head: bytes) -> Sniff | None:
        if head[:8] != b"\x89HDF\r\n\x1a\n":
            return None
        if path.suffix.lower() in (".h5", ".mrd", ".hdf5"):
            return Sniff(
                path,
                Format.ISMRMRD,
                Kind.RAW,
                confidence=0.7,
                evidence=("HDF5 magic", f"{path.suffix} extension"),
            )
        return Sniff(path, Format.UNKNOWN, evidence=("HDF5 container, unknown layout",))

    def extension_only(self, path: Path, head: bytes) -> Sniff | None:
        if path.suffix.lower() == ".mrui":
            return Sniff(
                path,
                Format.JMRUI_BINARY,
                Kind.SPECTROSCOPY,
                confidence=0.5,
                evidence=(".mrui extension only",),
            )
        return None

    def dicom_no_preamble(self, path: Path, head: bytes) -> Sniff | None:
        """Last resort: some exports drop the 128-byte preamble; let pydicom try."""
        if path.suffix.lower() not in (".dcm", ".ima", ""):
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return self._read_dicom(path, force=True)
        except Exception:
            return None

    # --- helpers ---------------------------------------------------------------------------------

    def _head(self, path: Path) -> bytes:
        with open(path, "rb") as fh:
            return fh.read(self.HEAD_BYTES)

    @staticmethod
    def _is_text(head: bytes) -> bool:
        """Printable ASCII and whitespace only, as in a text header or document."""
        sample = head[:1024]
        return bool(sample) and all(32 <= b < 127 or b in (9, 10, 13) for b in sample)

    @staticmethod
    def _is_dicom(head: bytes) -> bool:
        return head[128:132] == b"DICM"

    @staticmethod
    def _sibling(path: Path, suffix: str) -> Path | None:
        """Companion file with the same stem and ``suffix`` in any letter case."""
        for candidate in path.parent.iterdir():
            if candidate.stem == path.stem and candidate.suffix.lower() == suffix:
                if candidate != path:
                    return candidate
        return None

    def _vendor(self, manufacturer: str) -> str | None:
        for name, pattern in self.VENDORS:
            if pattern.search(manufacturer):
                return name
        return manufacturer.strip().lower() or None

    def _read_dicom(self, path: Path, force: bool = False) -> Sniff:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=force)
        sop = str(ds.get("SOPClassUID", ""))
        if not sop and force:
            raise ValueError("not DICOM")
        manufacturer = str(ds.get("Manufacturer", ""))
        detail = {
            "sop_class_uid": sop,
            "modality": str(ds.get("Modality", "")),
            "manufacturer": manufacturer,
            "software_versions": str(ds.get("SoftwareVersions", "")),
            "protocol_name": str(ds.get("ProtocolName", "")),
        }
        if sop in self.DICOM_SOP_MRS:
            fmt, kind, confidence = Format.DICOM_MRS, Kind.SPECTROSCOPY, 1.0
            sop_name = self.DICOM_SOP_MRS[sop]
        elif sop in self.DICOM_SOP_IMAGE:
            fmt, kind, confidence = Format.DICOM_IMAGE, Kind.IMAGING, 1.0
            sop_name = self.DICOM_SOP_IMAGE[sop]
        else:
            fmt, kind, confidence = Format.DICOM_OTHER, Kind.UNKNOWN, 0.8
            sop_name = sop or "missing"
        return Sniff(
            path,
            fmt,
            kind,
            vendor=self._vendor(manufacturer),
            confidence=confidence,
            evidence=(
                "parsed as DICOM without preamble" if force else "DICM magic",
                f"SOP class {sop_name}",
            ),
            detail=detail,
        )

    def _bruker_scan(self, path: Path, scan_dir: Path, evidence: str) -> Sniff:
        companions = tuple(
            p for p in (scan_dir / f for f in self.BRUKER_PARAM_FILES) if p.is_file()
        )
        detail = {}
        acqp = scan_dir / "acqp"
        if acqp.is_file():
            text = acqp.read_text(errors="replace")
            for key, param in self.BRUKER_ACQP_FIELDS.items():
                match = re.search(rf"##\${param}=\(\s*\d+\s*\)\s*<([^>]*)>", text)
                if match:
                    detail[key] = match.group(1)
        if self.BRUKER_DATA_FILES.match(path.name) and path.name not in self.BRUKER_PROCESSED:
            kind = Kind.RAW
        else:
            kind = self._bruker_scan_kind(detail.get("method"))
        return Sniff(
            path,
            Format.BRUKER,
            kind,
            vendor="bruker",
            confidence=1.0,
            evidence=(evidence,),
            companions=companions,
            detail=detail,
        )

    def _bruker_scan_kind(self, method: str | None) -> Kind:
        if method is None:
            return Kind.UNKNOWN
        if any(m in method.lower() for m in self.BRUKER_MRS_METHODS):
            return Kind.SPECTROSCOPY
        return Kind.IMAGING

    def _varian_scan(self, path: Path, folder: Path, evidence: str) -> Sniff:
        companions = tuple(
            p for p in (folder / "fid", folder / "procpar") if p.is_file() and p != path
        )
        return Sniff(
            path,
            Format.VARIAN,
            Kind.RAW,
            vendor="varian",
            confidence=1.0,
            evidence=(evidence,),
            companions=companions,
        )


#**************************************************************************************************#
#                                              sniff                                               #
#**************************************************************************************************#
#                                                                                                  #
# Identify one path with a fresh Sniffer.                                                          #
#                                                                                                  #
#**************************************************************************************************#
def sniff(path: str | Path) -> Sniff:
    return Sniffer().sniff(path)
