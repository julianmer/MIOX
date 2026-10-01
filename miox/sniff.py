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
#          or only the file name decided. Each format id is also the name of its (future) reader   #
#          module.                                                                                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

HEAD_BYTES = 4096
_DIR_PROBE_LIMIT = 50  # files probed for DICOM in a directory

# Format ids. Each is also the name of the (future) reader module.
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
JMRUI_TEXT = "jmrui_text"
JMRUI_BINARY = "jmrui_binary"
ISMRMRD = "ismrmrd"
UNKNOWN = "unknown"

# What the data is, as far as the sniffer can tell.
SPECTROSCOPY = "spectroscopy"
IMAGING = "imaging"
RAW = "raw"  # k-space / unprocessed ADC lines (twix, P-file, Bruker fid)
KIND_UNKNOWN = "unknown"

_DICOM_SOP_MRS = {
    "1.2.840.10008.5.1.4.1.1.4.2": "MR Spectroscopy Storage",
    "1.3.12.2.1107.5.9.1": "Siemens CSA Non-Image Storage",
    "1.3.46.670589.11.0.0.12.1": "Philips Private MR Spectrum Storage",
    "1.3.46.670589.11.0.0.12.2": "Philips Private MR Series",
}
_DICOM_SOP_IMAGE = {
    "1.2.840.10008.5.1.4.1.1.4": "MR Image Storage",
    "1.2.840.10008.5.1.4.1.1.4.1": "Enhanced MR Image Storage",
    "1.2.840.10008.5.1.4.1.1.4.3": "Enhanced MR Color Image Storage",
    "1.2.840.10008.5.1.4.1.1.4.4": "Legacy Converted Enhanced MR Image Storage",
}
_VENDOR_PATTERNS = (
    ("siemens", re.compile(r"siemens", re.I)),
    ("philips", re.compile(r"philips", re.I)),
    ("ge", re.compile(r"^ge\b|ge medical", re.I)),
    ("bruker", re.compile(r"bruker", re.I)),
    ("uih", re.compile(r"uih|united imaging", re.I)),
    ("canon", re.compile(r"canon|toshiba", re.I)),
    ("hyperfine", re.compile(r"hyperfine", re.I)),
    ("varian", re.compile(r"varian|agilent", re.I)),
)
_TWIX_VB_BUFFERS = (b"Config", b"Dicom", b"Meas", b"MeasYaps", b"Phoenix", b"Spice")
_BRUKER_PARAM_FILES = ("acqp", "method", "visu_pars", "reco", "subject")
_BRUKER_DATA_FILES = re.compile(
    r"^(fid|ser|2dseq|rawdata\.job\d+|fid\.refscan|rawdata\.Navigator)$"
)
_PFILE_NAME = re.compile(r"^P\d{5}\.7$", re.I)


@dataclass(frozen=True)
class Sniff:
    """Result of sniffing one path."""

    path: Path
    format: str
    kind: str = KIND_UNKNOWN
    vendor: str | None = None
    confidence: float = 0.0
    evidence: tuple[str, ...] = ()
    companions: tuple[Path, ...] = ()
    detail: dict = field(default_factory=dict)

    def __str__(self) -> str:
        vendor = f" {self.vendor}" if self.vendor else ""
        return f"{self.path}: {self.format}{vendor} ({self.kind}, {self.confidence:.1f})"


def sniff(path: str | Path) -> Sniff:
    """Identify the format of ``path`` (a file or a directory)."""
    path = Path(path)
    if path.is_dir():
        return _sniff_dir(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path, "rb") as fh:
        head = fh.read(HEAD_BYTES)
    size = path.stat().st_size
    for rule in _FILE_RULES:
        result = rule(path, head, size)
        if result is not None:
            return result
    return Sniff(path, UNKNOWN)


# --- directories --------------------------------------------------------------------------------


def _sniff_dir(path: Path) -> Sniff:
    names = {p.name for p in path.iterdir()}
    if names & set(_BRUKER_PARAM_FILES):
        return _bruker(path, path, evidence="directory with Bruker parameter files")
    if "procpar" in names:
        return _varian(path, path, evidence="directory with procpar")
    files = [c for c in sorted(path.iterdir()) if c.is_file() and c.name != ".DS_Store"]
    for child in files[:_DIR_PROBE_LIMIT]:
        with open(child, "rb") as fh:
            if not _is_dicom(fh.read(HEAD_BYTES)):
                continue
        result = _dicom(child, child.stat().st_size)
        others = tuple(c for c in files if c != child)
        return Sniff(
            path,
            result.format,
            result.kind,
            result.vendor,
            result.confidence,
            ("directory; first DICOM file decides",) + result.evidence,
            (child,) + others,
            result.detail,
        )
    return Sniff(path, UNKNOWN, evidence=("directory with no recognised content",))


# --- file rules, most specific first ------------------------------------------------------------


def _is_dicom(head: bytes) -> bool:
    return head[128:132] == b"DICM"


def _rule_dicom(path: Path, head: bytes, size: int) -> Sniff | None:
    if _is_dicom(head):
        return _dicom(path, size)
    return None


def _rule_dicom_no_preamble(path: Path, head: bytes, size: int) -> Sniff | None:
    """Last resort: some exports drop the 128-byte preamble; let pydicom try."""
    if path.suffix.lower() not in (".dcm", ".ima", ""):
        return None
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _dicom(path, size, force=True)
    except Exception:
        return None


def _dicom(path: Path, size: int, force: bool = False) -> Sniff:
    import pydicom

    ds = pydicom.dcmread(path, stop_before_pixels=True, force=force)
    sop = str(ds.get("SOPClassUID", ""))
    if not sop and force:
        raise ValueError("not DICOM")
    vendor = _vendor(str(ds.get("Manufacturer", "")))
    detail = {
        "sop_class_uid": sop,
        "modality": str(ds.get("Modality", "")),
        "manufacturer": str(ds.get("Manufacturer", "")),
        "software_versions": str(ds.get("SoftwareVersions", "")),
        "protocol_name": str(ds.get("ProtocolName", "")),
    }
    evidence = ["DICM magic" if not force else "parsed as DICOM without preamble"]
    if sop in _DICOM_SOP_MRS:
        evidence.append(f"SOP class {_DICOM_SOP_MRS[sop]}")
        return Sniff(path, DICOM_MRS, SPECTROSCOPY, vendor, 1.0, tuple(evidence), (), detail)
    if sop in _DICOM_SOP_IMAGE:
        evidence.append(f"SOP class {_DICOM_SOP_IMAGE[sop]}")
        return Sniff(path, DICOM_IMAGE, IMAGING, vendor, 1.0, tuple(evidence), (), detail)
    evidence.append(f"SOP class {sop or 'missing'}")
    return Sniff(path, DICOM_OTHER, KIND_UNKNOWN, vendor, 0.8, tuple(evidence), (), detail)


def _rule_nifti(path: Path, head: bytes, size: int) -> Sniff | None:
    if head[:2] == b"\x1f\x8b":
        try:
            head = zlib.decompressobj(31).decompress(head, 1024)
        except zlib.error:
            return None
        gzipped = True
    else:
        gzipped = False
    if len(head) < 348:
        return None
    if head[344:348] in (b"n+1\0", b"ni1\0"):
        version = 1
    elif head[4:8] in (b"n+2\0", b"ni2\0"):
        version = 2
    else:
        return None
    import gzip

    import nibabel as nib

    header_class = nib.Nifti1Header if version == 1 else nib.Nifti2Header
    opener = gzip.open if gzipped else open
    with opener(path, "rb") as fh:
        hdr = header_class.from_fileobj(fh, check=False)
    detail = {"nifti_version": version, "shape": tuple(int(n) for n in hdr.get_data_shape())}
    ext = [e for e in hdr.extensions if e.get_code() == 44]
    if ext:
        import json

        hdr = json.loads(ext[0].get_content())
        detail["nifti_mrs_version"] = (
            hdr.get("NIfTI_MRS_version") or hdr.get("nifti_mrs_version") or ""
        )
        detail["dim_tags"] = {
            k: v for k, v in hdr.items() if k.startswith("dim_") and k.endswith("tag")
        }
        return Sniff(
            path,
            NIFTI_MRS,
            SPECTROSCOPY,
            None,
            1.0,
            (f"NIfTI-{version} magic", "header extension ecode 44"),
            (),
            detail,
        )
    return Sniff(path, NIFTI, IMAGING, None, 1.0, (f"NIfTI-{version} magic",), (), detail)


def _rule_rda(path: Path, head: bytes, size: int) -> Sniff | None:
    if head.startswith(b">>> Begin of header <<<"):
        return Sniff(
            path, SIEMENS_RDA, SPECTROSCOPY, "siemens", 1.0, ("RDA header marker",), (), {}
        )
    return None


_SPAR_KEYS = (b"examination_name", b"scan_id", b"synthesizer_frequency", b"samples")


def _rule_spar_sdat(path: Path, head: bytes, size: int) -> Sniff | None:
    suffix = path.suffix.lower()
    if head.startswith(b"!---") or sum(k in head for k in _SPAR_KEYS) >= 2:
        sdat = _sibling(path, ".sdat")
        companions = (sdat,) if sdat else ()
        ev = ("SPAR text header",) + (("SDAT companion found",) if sdat else ("no SDAT companion",))
        return Sniff(
            path, PHILIPS_SPAR_SDAT, SPECTROSCOPY, "philips", 1.0 if sdat else 0.9, ev, companions
        )
    if suffix == ".sdat":
        spar = _sibling(path, ".spar")
        if spar:
            return Sniff(
                path,
                PHILIPS_SPAR_SDAT,
                SPECTROSCOPY,
                "philips",
                1.0,
                (".sdat extension", "SPAR companion found"),
                (spar,),
            )
        return Sniff(
            path,
            PHILIPS_SPAR_SDAT,
            SPECTROSCOPY,
            "philips",
            0.5,
            (".sdat extension", "no SPAR companion"),
        )
    return None


def _rule_data_list(path: Path, head: bytes, size: int) -> Sniff | None:
    if head.startswith(b"# === DATA DESCRIPTION FILE"):
        data = _sibling(path, ".data")
        return Sniff(
            path,
            PHILIPS_DATA_LIST,
            RAW,
            "philips",
            1.0 if data else 0.9,
            ("LIST header marker",)
            + (("DATA companion found",) if data else ("no DATA companion",)),
            (data,) if data else (),
        )
    if path.suffix.lower() == ".data":
        lst = _sibling(path, ".list")
        if lst:
            return Sniff(
                path,
                PHILIPS_DATA_LIST,
                RAW,
                "philips",
                1.0,
                (".data extension", "LIST companion found"),
                (lst,),
            )
    return None


def _rule_twix(path: Path, head: bytes, size: int) -> Sniff | None:
    if len(head) < 16:
        return None
    first, second = struct.unpack_from("<II", head, 0)
    if first == 0 and 1 <= second <= 64:
        # VD/VE/XA: 0, n_measurements, then per measurement (MeasID, FileID, offset, length)
        offset, length = struct.unpack_from("<QQ", head, 16)
        if offset < size and 0 < length <= size:
            return Sniff(
                path,
                SIEMENS_TWIX,
                RAW,
                "siemens",
                1.0,
                ("VD/VE/XA multi-measurement header",),
                (),
                {"twix_layout": "VD", "n_measurements": second},
            )
    if first < size and 1 <= second <= 64 and head[8:40].split(b"\0")[0] in _TWIX_VB_BUFFERS:
        return Sniff(
            path,
            SIEMENS_TWIX,
            RAW,
            "siemens",
            1.0,
            ("VB header: length + buffer names",),
            (),
            {"twix_layout": "VB", "n_buffers": second},
        )
    return None


_PFILE_REVISIONS = (7, 8, 9, 10, 11, 12, 14, 15, 16, 20, 24, 26, 27, 28, 30)


def _rule_pfile(path: Path, head: bytes, size: int) -> Sniff | None:
    if len(head) < 4:
        return None
    (rev,) = struct.unpack_from("<f", head, 0)
    named = path.suffix == ".7" or _PFILE_NAME.match(path.name) is not None
    if any(abs(rev - r) < 0.01 for r in _PFILE_REVISIONS) or (named and 5.0 <= rev <= 40.0):
        return Sniff(
            path,
            GE_PFILE,
            RAW,
            "ge",
            1.0 if named else 0.8,
            (f"P-file revision {rev:g} at offset 0",) + ((".7 name",) if named else ()),
            (),
            {"pfile_revision": round(rev, 2)},
        )
    return None


def _rule_bruker(path: Path, head: bytes, size: int) -> Sniff | None:
    if head.startswith(b"##TITLE=") and b"##JCAMP" in head:
        if path.name in _BRUKER_PARAM_FILES:
            return _bruker(path, path.parent, evidence=f"JCAMP-DX parameter file {path.name}")
        if b"##$" in head:
            return _bruker(path, path.parent, evidence="JCAMP-DX file with Bruker parameters")
    if _BRUKER_DATA_FILES.match(path.name):
        folder = path.parent
        if path.name == "2dseq":
            scan_dir = folder.parent.parent  # <scan>/pdata/<n>/2dseq
        else:
            scan_dir = folder
        if (folder / "visu_pars").is_file() or any(
            (scan_dir / f).is_file() for f in ("acqp", "method")
        ):
            return _bruker(path, scan_dir, evidence=f"Bruker data file {path.name} with parameters")
    return None


def _bruker(path: Path, scan_dir: Path, evidence: str) -> Sniff:
    companions = tuple(p for p in (scan_dir / f for f in _BRUKER_PARAM_FILES) if p.is_file())
    detail = {}
    acqp = scan_dir / "acqp"
    if acqp.is_file():
        text = acqp.read_text(errors="replace")
        m = re.search(r"##\$ACQ_sw_version=\(\s*\d+\s*\)\s*<([^>]*)>", text)
        if m:
            detail["paravision_version"] = m.group(1)
        m = re.search(r"##\$ACQ_scan_name=\(\s*\d+\s*\)\s*<([^>]*)>", text)
        if m:
            detail["scan_name"] = m.group(1)
        m = re.search(r"##\$ACQ_method=\(\s*\d+\s*\)\s*<([^>]*)>", text)
        if m:
            detail["method"] = m.group(1)
    if _BRUKER_DATA_FILES.match(path.name) and path.name != "2dseq":
        kind = RAW
    else:
        kind = _bruker_scan_kind(detail.get("method"))
    return Sniff(path, BRUKER, kind, "bruker", 1.0, (evidence,), companions, detail)


_BRUKER_MRS_METHODS = ("press", "steam", "csi", "svs", "slaser", "laser", "isis", "fid", "nspect")


def _bruker_scan_kind(method: str | None) -> str:
    if method is None:
        return KIND_UNKNOWN
    if any(k in method.lower() for k in _BRUKER_MRS_METHODS):
        return SPECTROSCOPY
    return IMAGING


def _rule_varian(path: Path, head: bytes, size: int) -> Sniff | None:
    if path.name == "procpar" or (path.name == "fid" and (path.parent / "procpar").is_file()):
        return _varian(path, path.parent, evidence=f"{path.name} with procpar")
    return None


def _varian(path: Path, folder: Path, evidence: str) -> Sniff:
    companions = tuple(p for p in (folder / "fid", folder / "procpar") if p.is_file() and p != path)
    return Sniff(path, VARIAN, RAW, "varian", 1.0, (evidence,), companions)


def _rule_text_formats(path: Path, head: bytes, size: int) -> Sniff | None:
    if b"jMRUI Data Textfile" in head:
        return Sniff(path, JMRUI_TEXT, SPECTROSCOPY, None, 1.0, ("jMRUI text marker",))
    if b"$NMID" in head or b"$SEQPAR" in head:
        return Sniff(
            path, LCMODEL_RAW, SPECTROSCOPY, None, 1.0, ("LCModel $NMID/$SEQPAR namelist",)
        )
    return None


def _rule_hdf5(path: Path, head: bytes, size: int) -> Sniff | None:
    if head[:8] != b"\x89HDF\r\n\x1a\n":
        return None
    if path.suffix.lower() in (".h5", ".mrd", ".hdf5"):
        return Sniff(path, ISMRMRD, RAW, None, 0.7, ("HDF5 magic", f"{path.suffix} extension"))
    return Sniff(path, UNKNOWN, evidence=("HDF5 container, unknown layout",))


def _rule_extension_only(path: Path, head: bytes, size: int) -> Sniff | None:
    if path.suffix.lower() == ".mrui":
        return Sniff(path, JMRUI_BINARY, SPECTROSCOPY, None, 0.5, (".mrui extension only",))
    return None


_FILE_RULES = (
    _rule_dicom,
    _rule_nifti,
    _rule_rda,
    _rule_spar_sdat,
    _rule_data_list,
    _rule_twix,
    _rule_pfile,
    _rule_bruker,
    _rule_varian,
    _rule_text_formats,
    _rule_hdf5,
    _rule_extension_only,
    _rule_dicom_no_preamble,
)


# --- helpers ------------------------------------------------------------------------------------


def _sibling(path: Path, suffix: str) -> Path | None:
    """Companion file with the same stem and ``suffix`` in any letter case."""
    for candidate in path.parent.iterdir():
        if candidate.stem == path.stem and candidate.suffix.lower() == suffix and candidate != path:
            return candidate
    return None


def _vendor(manufacturer: str) -> str | None:
    for name, pattern in _VENDOR_PATTERNS:
        if pattern.search(manufacturer):
            return name
    return manufacturer.strip().lower() or None
