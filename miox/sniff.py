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

import bz2
import dataclasses
import gzip
import json
import lzma
import mmap
import re
import struct
import warnings
import zipfile
import zlib
from dataclasses import KW_ONLY, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import BinaryIO

import h5py
import nibabel as nib
import pydicom

# own
from miox.matfile import MatFile
from miox.yardl import YardlFile

GZIP_MAGIC = b"\x1f\x8b"
SIGNA_4X_STUDY_START = 1536  # GE Signa 4.x: study header block, in 16-bit words
SIGNA_4X_SERIES_START = 2048  # series header block, in words
SIGNA_4X_PLANE_NAME = 139  # the plane name in that block, in words


def open_dicom(path: Path) -> BinaryIO:
    """The file for reading, decompressed when it is a gzipped export (.dcm.gz)."""
    with open(path, "rb") as fh:
        gzipped = fh.read(2) == GZIP_MAGIC
    return gzip.open(path, "rb") if gzipped else open(path, "rb")


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
    PHILIPS_RAW_LAB = "philips_raw_lab"
    GE_PFILE = "ge_pfile"
    DICOM_MRS = "dicom_mrs"
    DICOM_IMAGE = "dicom_image"
    DICOM_OTHER = "dicom_other"
    NIFTI_MRS = "nifti_mrs"
    NIFTI = "nifti"
    NIFTI_SIDECAR_MRS = "nifti_sidecar_mrs"  # complex NIfTI with MRS values in a BIDS sidecar
    ANALYZE = "analyze"
    MGH = "mgh"
    MINC1 = "minc1"
    MINC2 = "minc2"
    PARREC = "parrec"
    AFNI = "afni"
    NRRD = "nrrd"
    METAIMAGE = "metaimage"
    JNIFTI = "jnifti"  # JNIfTI in JSON text (.jnii)
    VISTA = "vista"  # Vista images (.v; Lipsia)
    NIFTI_ZARR = "nifti_zarr"  # NIfTI-Zarr store (.nii.zarr folder)
    STIMULATE = "stimulate"  # Stimulate .spr header with .sdt data
    VTK_IMAGE = "vtk_image"  # VTK legacy STRUCTURED_POINTS
    VTK_IMAGE_XML = "vtk_image_xml"  # VTK XML ImageData (.vti)
    JNIFTI_BINARY = "jnifti_binary"  # JNIfTI in binary BJData (.bnii)
    MRTRIX = "mrtrix"
    VARIAN_FDF = "varian_fdf"  # Varian/Agilent VnmrJ image
    GE_SIGNA = "ge_signa"  # GE Signa 4.x, Genesis 5.x and Advantage Windows images
    BRUKER = "bruker"
    TOPSPIN = "topspin"  # Bruker TopSpin experiment: fid or ser with acqus
    TOPSPIN_PROCESSED = "topspin_processed"  # TopSpin processed spectra: pdata/<n> with procs
    JCAMP_NMR = "jcamp_nmr"  # JCAMP-DX NMR FIDs and spectra
    NMRML = "nmrml"  # nmrML XML files
    JEOL_JDF = "jeol_jdf"  # JEOL Delta data files
    TECMAG_TNT = "tecmag_tnt"  # Tecmag TNMR data files
    NMRPIPE = "nmrpipe"  # NMRPipe data files
    SPARKY = "sparky"  # Sparky (UCSF) spectra
    SIMPSON = "simpson"  # SIMPSON simulation output
    CSDM = "csdm"  # Core Scientific Dataset Model (.csdf, .csdfe)
    SPINSOLVE = "spinsolve"  # Magritek Spinsolve (Prospa) data files and experiment folders
    RS2D = "rs2d"  # RS2D experiment folders (header.xml + data.dat)
    RNMRTK = "rnmrtk"  # Rowland NMR Toolkit .sec data with .par parameters
    TRANSFORM_LTA = "transform_lta"  # FreeSurfer linear transform array
    TRANSFORM_ITK = "transform_itk"  # ITK transform (.tfm text, .h5)
    TRANSFORM_AFNI = "transform_afni"  # AFNI affine matrices (12 numbers per row)
    TRANSFORM_FSL = "transform_fsl"  # FSL FLIRT 4 x 4 matrix
    TRK = "trk"  # TrackVis streamlines
    TCK = "tck"  # MRtrix streamlines
    TRX = "trx"  # TRX streamlines (zip archive or folder)
    VTK_POLYDATA = "vtk_polydata"  # VTK legacy POLYDATA: lines (streamlines) or polygons
    DSI_STUDIO_TT = "dsi_studio_tt"  # DSI Studio streamlines (.tt, gzipped MATLAB v4)
    TSF = "tsf"  # MRtrix track scalar file: one value per streamline point
    MZ3 = "mz3"  # Surfice / NiiVue mesh (.mz3, often gzipped)
    DSI_STUDIO_SRC = "dsi_studio_src"  # DSI Studio source: diffusion images and b-table
    DSI_STUDIO_FIB = "dsi_studio_fib"  # DSI Studio fibre file: reconstructed maps
    BRAINVOYAGER_VMR = "brainvoyager_vmr"  # BrainVoyager anatomical volume
    BRAINVOYAGER_VTC = "brainvoyager_vtc"  # BrainVoyager volume time course
    BRAINVOYAGER_GLM = "brainvoyager_glm"  # BrainVoyager GLM results
    BRAINVOYAGER_VMP = "brainvoyager_vmp"  # BrainVoyager volume maps
    BRAINVOYAGER_VOI = "brainvoyager_voi"  # BrainVoyager volumes of interest
    BRAINVOYAGER_SRF = "brainvoyager_srf"  # BrainVoyager surface mesh
    BRAINVOYAGER_SMP = "brainvoyager_smp"  # BrainVoyager surface maps
    BRAINVOYAGER_MTC = "brainvoyager_mtc"  # BrainVoyager mesh time courses
    BRAINVOYAGER_SSM = "brainvoyager_ssm"  # BrainVoyager surface-to-surface mapping
    BRAINVOYAGER_POI = "brainvoyager_poi"  # BrainVoyager patches of interest
    BRAINVOYAGER_PRT = "brainvoyager_prt"  # BrainVoyager stimulation protocol
    BRAINVOYAGER_SDM = "brainvoyager_sdm"  # BrainVoyager design matrix
    BRAINVOYAGER_TRF = "brainvoyager_trf"  # BrainVoyager spatial transformation
    GIFTI = "gifti"  # GIFTI surfaces and per-vertex data (.gii, also gzip/bzip2)
    FREESURFER_SURFACE = "freesurfer_surface"  # FreeSurfer triangle or quad surface
    FREESURFER_MORPH = "freesurfer_morph"  # FreeSurfer per-vertex values (curv, thickness, ...)
    FREESURFER_ANNOT = "freesurfer_annot"  # FreeSurfer annotation (.annot)
    FREESURFER_LABEL = "freesurfer_label"  # FreeSurfer label (.label text)
    ZIP_ARCHIVE = "zip_archive"  # zip archive of files MIOX reads
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
    VESPA_PRIOR = "vespa_prior"
    GAVA_PRIOR = "gava_prior"
    MIDAS_PRIOR = "midas_prior"
    PYAMARES_PRIOR = "pyamares_prior"
    PYAMARES_RESULT = "pyamares_result"
    DMIWIZARD_MAP = "dmiwizard_map"  # DMIWizard metabolic or ROI map: one text column
    SPANT_RDS = "spant_rds"  # spant spectra, basis sets or fit results saved as R data
    OSPREY_MRSCONT = "osprey_mrscont"
    BASIS_FOLDER = "basis_folder"
    JMRUI_TEXT = "jmrui_text"
    JMRUI_BINARY = "jmrui_binary"
    TEXT_FID = "text_fid"  # headerless columns of real and imaginary parts
    ISMRMRD = "ismrmrd"
    FASTMRI_H5 = "fastmri_h5"
    GE_SCANARCHIVE = "ge_scanarchive"
    PULSEQ = "pulseq"
    BART_CFL = "bart_cfl"
    MRD = "mrd"
    SIEMENS_PMU = "siemens_pmu"
    CMRR_PHYSIO = "cmrr_physio"
    PHILIPS_PHYSLOG = "philips_physlog"
    GE_PHYSIO = "ge_physio"
    BIDS_PHYSIO = "bids_physio"
    HCP_PHYSIO = "hcp_physio"  # Human Connectome Project *_Physio_log.txt
    YARRA_MODE = "yarra_mode"  # Yarra reconstruction mode (.mode)
    YARRA_MODULE = "yarra_module"  # Yarra module manifest (.ymf)
    BIOPAC_ACQ = "biopac_acq"
    KOMA_PHANTOM = "koma_phantom"
    JEMRIS_PHANTOM = "jemris_phantom"
    JEMRIS_SIGNAL = "jemris_signal"
    JEMRIS_MOTION = "jemris_motion"  # JEMRIS motion trajectory (.dat text)
    TOPPE_ENTRY = "toppe_entry"
    TOPPE_MODULES = "toppe_modules"
    TOPPE_LOOP = "toppe_loop"
    TOPPE_MOD = "toppe_mod"
    MRILAB_PHANTOM = "mrilab_phantom"
    SIEMENS_PRO = "siemens_pro"
    SIEMENS_EXAR = "siemens_exar"
    SIEMENS_PRINTPROT = "siemens_printprot"
    PHILIPS_EXAMCARD = "philips_examcard"
    PHILIPS_DICOM_RAW = "philips_dicom_raw"
    SIEMENS_DICOM_RAW = "siemens_dicom_raw"
    SIEMENS_PHOENIX_REPORT = "siemens_phoenix_report"
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
    SEQUENCE = "sequence"  # pulse sequence timing: RF, gradients, ADC events
    PHYSIO = "physio"  # physiological recordings: cardiac, respiratory, triggers
    PHANTOM = "phantom"  # simulation phantoms: spins and their properties
    TRACTOGRAM = "tractogram"  # streamlines from diffusion tractography
    SURFACE = "surface"  # surface meshes and per-vertex data
    TRANSFORM = "transform"  # spatial transforms between image spaces
    REPORT = "report"  # graphics for reading, no data
    ARCHIVE = "archive"  # a container of other files (zip)
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
    STUDY_LIMIT = 100_000  # DICOM files gathered from a study folder

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
    RAW_DATA_STORAGE = "1.2.840.10008.5.1.4.1.1.66"
    STRUCTURED_REPORT = "1.2.840.10008.5.1.4.1.1.88."  # the SR storage classes
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
    SIN_LINE = re.compile(rb"^\s*\d+\s+\d+\s+\d+\s*:\s*[a-z_]+\s*:", re.M)  # Philips .sin parameter
    TWIX_VB_BUFFERS = (b"Config", b"Dicom", b"Meas", b"MeasYaps", b"Phoenix", b"Spice")
    PFILE_REVISIONS = (7, 8, 9, 10, 11, 12, 14, 15, 16, 20, 24, 26, 27, 28, 30)
    PFILE_NAME = re.compile(r"^P\d{5}\.7$", re.I)
    PFILE_MIN_SIZE = 39984  # header bytes of revision 7, the smallest P-file header
    ZARR_MARKERS = (".zarray", ".zgroup", "zarr.json")  # metadata files of a zarr store
    ZARR_DEPTH = 6  # parent folders searched for them
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
        ((b";SIMULATION SPECTRAL RESULTS",), Format.GAVA_PRIOR, Kind.BASIS_SET, "GAVA prior"),
    )
    PYAMARES = re.compile(rb"^Index,[^\r\n]*\r?\nInitial Values,", re.M)  # prior table rows
    SPANT_CLASSES = ((b"fit_result", Kind.FIT_RESULT), (b"mrs_data", Kind.SPECTROSCOPY),
                     (b"basis_set", Kind.BASIS_SET))  # first found gives the kind
    DMIWIZARD_MAP = re.compile(r"^(MetabolicMap|ROIMap)_(\d+)x(\d+)_([XYZ])slice(\d+)_(.+)$")
    TEXT_STARTS = (  # (start of the file, format, kind, evidence)
        (b"Dangerplot_version", Format.TARQUIN_DPT, Kind.SPECTROSCOPY, "TARQUIN dangerplot"),
        (b"Signal     Conc (au)", Format.TARQUIN_RESULTS, Kind.FIT_RESULT, "TARQUIN results"),
        (b"Row : ", Format.TARQUIN_FIT, Kind.FIT_RESULT, "TARQUIN fit"),
        (b"Row, Col,", Format.LCMODEL_CSV, Kind.FIT_RESULT, "LCModel CSV"),
        (b"name,amplitude,sd,CRLB(%)", Format.PYAMARES_RESULT, Kind.FIT_RESULT, "PyAMARES result"),
        (b"[FILE INFORMATION]", Format.VASF, Kind.SPECTROSCOPY, "VASF header"),
        (b"DATA DESCRIPTOR FILE", Format.SIVIC_DDF, Kind.SPECTROSCOPY, "UCSF DDF header"),
        (b"IMAGE DESCRIPTOR FILE", Format.SIVIC_IDF, Kind.IMAGING, "UCSF IDF header"),
    )
    PAR_LINE = re.compile(rb"^\.\s+[A-Za-z]", re.M)  # ".    Patient name :"; LIST: ".    0    0"
    NUMBER = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")  # one numeric text field
    FIELD_SEPARATOR = re.compile(r"[,\s]+")
    TEXT_FID_LINES = 8  # complete numeric lines a headerless text file must start with
    TEXT_FID_MAX_COLUMNS = 8  # more numeric columns than that is a table or an image, not FIDs
    NIFTI_SIDECARS = (".json", ".bval", ".bvec", ".img")  # BIDS sidecar, diffusion tables, pair
    ANALYZE_SIZES = (b"\x00\x00\x01\x5c", b"\x5c\x01\x00\x00")  # sizeof_hdr 348, either byte order
    BRIK = re.compile(r"\.brik(\.gz)?$", re.I)
    MATLAB_VARIABLES = (  # (variable name, format, kind); first match wins
        (re.compile(r"^MRSCont$"), Format.OSPREY_MRSCONT, Kind.SPECTROSCOPY),
        (re.compile(r"^VObj$"), Format.MRILAB_PHANTOM, Kind.PHANTOM),
        (re.compile(r"^BASIS$"), Format.OSPREY_BASIS, Kind.BASIS_SET),
        (re.compile(r"^lcmBasis$"), Format.INSPECTOR_BASIS, Kind.BASIS_SET),
        (re.compile(r"^exptDat$"), Format.MARSS_BASIS, Kind.BASIS_SET),
        (re.compile(r"^basis$"), Format.PROFIT_BASIS, Kind.BASIS_SET),
        (re.compile(r"^out"), Format.FIDA_MAT, Kind.SPECTROSCOPY),  # out, outw, outA, ...
    )
    BASIS_MEMBERS = (
        Format.LCMODEL_RAW,
        Format.JMRUI_TEXT,
        Format.TEXT_FID,
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
        (".sdt", ".spr", Format.STIMULATE),
    )
    BRUKER_MRS_METHODS = (
        "press", "steam", "csi", "svs", "slaser", "laser", "isis", "fid", "nspect"
    )
    JCAMP_NMR_TYPE = re.compile(rb"##DATA\s*TYPE=\s*(n?D\s*)?NMR\s*(FID|SPECTRUM)", re.I)
    JCAMP_LINK = re.compile(rb"##DATA\s*TYPE=\s*LINK", re.I)
    TOPSPIN_SPECTRA = re.compile(r"^(1[ri]|2[ri]{2}|3[ri]{3}|4[ri]{4})$")  # 1r, 2rr, 3rrr, ...
    PARAVISION_FILES = ("acqp", "method", "visu_pars", "reco")  # beside a ParaVision scan's data

    def __init__(self):
        self.rules = (
            self.dicom,
            self.nifti,
            self.physio,
            self.toppe,
            self.protocols,
            self.image_files,
            self.rda,
            self.matlab,
            self.companion_binary,
            self.analysis_text,
            self.vespa,
            self.spar_sdat,
            self.raw_lab_sin,
            self.data_list,
            self.twix,
            self.pfile,
            self.jcamp,
            self.nmrml,
            self.jeol,
            self.tecmag,
            self.nmrpipe,
            self.sparky,
            self.simpson,
            self.csdm,
            self.rds,
            self.spinsolve,
            self.rs2d,
            self.rnmrtk,
            self.jemris_motion,
            self.brainvoyager,
            self.tractograms,
            self.surfaces,
            self.transforms,
            self.topspin,
            self.bruker,
            self.varian,
            self.pulseq,
            self.bart,
            self.yardl,
            self.text_formats,
            self.hdf5,
            self.archive,
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
        if store := self._zarr_store(path):
            return Sniff(path, Format.UNKNOWN, evidence=(f"inside the zarr store {store.name}",))
        head = self._head(path)
        for rule in self.rules:
            result = rule(path, head)
            if result is not None:
                return result
        return Sniff(path, Format.UNKNOWN)

    def _zarr_store(self, path: Path) -> Path | None:
        """The zarr array or group folder ``path`` lies in: its files are chunks and metadata of
        the store, not files of their own."""
        for folder in path.parents[:self.ZARR_DEPTH]:
            if any((folder / marker).is_file() for marker in self.ZARR_MARKERS):
                return folder
        return None

    def directory(self, path: Path) -> Sniff:
        """A NIfTI-Zarr store, a TRX tractogram folder, a Bruker scan, a Varian .fid folder, a
        Spinsolve experiment, a basis set of one file per metabolite, a DICOM series, or a study
        folder of DICOM series folders (a DICOMDIR at its top is passed over)."""
        names = {p.name for p in path.iterdir()}
        if names & {".zgroup", "zarr.json"} and (path / "nifti").is_dir():
            return self._image(path, Format.NIFTI_ZARR, "zarr group with a nifti header array")
        if "header.json" in names and self._trx(path):
            return Sniff(path, Format.TRX, Kind.TRACTOGRAM, confidence=0.9,
                         evidence=("folder with header.json, positions and offsets",))
        if names & set(self.BRUKER_PARAM_FILES):
            return self._bruker_scan(path, path, "directory with Bruker parameter files")
        if "acqus" in names:
            return self._topspin(path, Format.TOPSPIN, "directory with acqus")
        if "procs" in names and not self._paravision(path):
            return self._topspin(path, Format.TOPSPIN_PROCESSED, "directory with procs")
        experiments = [c for c in sorted(path.iterdir()) if (c / "acqus").is_file()]
        if experiments and not any(self._paravision(e) for e in experiments):
            return Sniff(path, Format.TOPSPIN, Kind.SPECTROSCOPY, vendor="bruker",
                         confidence=1.0, evidence=(f"{len(experiments)} TopSpin experiments",),
                         companions=tuple(experiments))
        if fdfs := sorted(p for p in path.iterdir() if p.suffix.lower() == ".fdf"):
            return self._image(path, Format.VARIAN_FDF, f"directory with {len(fdfs)} fdf files")
        if "procpar" in names:
            return self._varian_scan(path, path, "directory with procpar")
        if self._rs2d_folder(path):
            return Sniff(path, Format.RS2D, Kind.SPECTROSCOPY, vendor="rs2d", confidence=1.0,
                         evidence=("RS2D header.xml with data.dat",))
        if "acqu.par" in names and names & {"data.1d", "fid.1d", "spectrum.1d"}:
            return Sniff(path, Format.SPINSOLVE, Kind.SPECTROSCOPY, confidence=1.0,
                         evidence=("Spinsolve experiment folder (acqu.par with .1d data)",))
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
        dicoms = [c for c in files[: self.DIR_PROBE_LIMIT] if self._is_dicom_file(c)]
        subdirs = any(c.is_dir() for c in path.iterdir())
        if dicoms:
            first, child = self._deciding_dicom(dicoms)
            if first.format != Format.DICOM_OTHER or not subdirs:
                return dataclasses.replace(
                    first,
                    path=path,
                    evidence=("directory; first DICOM data file decides",) + first.evidence,
                    companions=(child,) + tuple(c for c in files if c != child),
                )
        for child in files[: self.DIR_PROBE_LIMIT]:
            if signa := self._ge_signa(child, self._head(child)):
                return self._image(path, Format.GE_SIGNA, f"directory of GE Signa files ({signa})")
        if subdirs and (study := self._study(path)):
            first, _ = self._deciding_dicom(study)
            return dataclasses.replace(
                first,
                path=path,
                evidence=("study folder; first DICOM data file decides",) + first.evidence,
                companions=tuple(study),
            )
        return Sniff(path, Format.UNKNOWN, evidence=("directory with no recognised content",))

    def _deciding_dicom(self, paths: list[Path]) -> tuple[Sniff, Path]:
        """The sniff of the first image or spectroscopy file among ``paths`` (presentation
        states, reports and raw data are passed over), else of the first file."""
        first = None
        for child in paths[: self.DIR_PROBE_LIMIT]:
            sniff = self._read_dicom(child)
            first = first or (sniff, child)
            if sniff.format != Format.DICOM_OTHER:
                return sniff, child
        return first

    def _study(self, path: Path) -> list[Path]:
        """DICOM files anywhere under a folder whose own DICOM files, if any, hold no data (a
        study, with or without a DICOMDIR)."""
        found = []
        for child in sorted(path.rglob("*")):
            if len(found) >= self.STUDY_LIMIT:
                break
            if child.is_file() and not child.name.startswith("."):
                if self._is_dicom_file(child):
                    found.append(child)
        return found

    # --- file rules ------------------------------------------------------------------------------

    def dicom(self, path: Path, head: bytes) -> Sniff | None:
        if head[:2] == GZIP_MAGIC:
            try:
                head = zlib.decompressobj(31).decompress(head, 132)
            except zlib.error:
                return None
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
        stem = path.name.removesuffix(".gz").removesuffix(".nii").removesuffix(".hdr")
        sidecars = [path.with_name(stem + s) for s in self.NIFTI_SIDECARS]
        sidecars = tuple(s for s in sidecars if s.is_file())
        if extension is None:
            if header.get_data_dtype().kind == "c" and self._mrs_sidecar(sidecars):
                return Sniff(
                    path,
                    Format.NIFTI_SIDECAR_MRS,
                    Kind.SPECTROSCOPY,
                    confidence=0.9,
                    evidence=(f"NIfTI-{version} magic", "complex data", "MRS values in sidecar"),
                    companions=sidecars,
                    detail=detail,
                )
            return Sniff(
                path,
                Format.NIFTI,
                Kind.IMAGING,
                confidence=1.0,
                evidence=(f"NIfTI-{version} magic",),
                companions=sidecars,
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
            companions=tuple(s for s in sidecars if s.suffix == ".json"),
            detail=detail,
        )

    @staticmethod
    def _mrs_sidecar(sidecars: tuple[Path, ...]) -> bool:
        """A BIDS sidecar that states the spectrometer frequency and nucleus."""
        sidecar = next((s for s in sidecars if s.suffix == ".json"), None)
        if sidecar is None:
            return False
        try:
            values = json.loads(sidecar.read_text())
        except ValueError:
            return False
        return "SpectrometerFrequency" in values and "ResonantNucleus" in values

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

    def raw_lab_sin(self, path: Path, head: bytes) -> Sniff | None:
        """Philips raw export: .lab labels (type 0x7f01 to 0x7f05 at byte 14), the .sin parameter
        text, or a .raw with a .lab beside it."""
        suffix = path.suffix.lower()
        if suffix == ".lab" and len(head) >= 16 and self._label_type(head):
            what = "label types at byte 14"
        elif suffix == ".sin" and self._is_text(head) and self.SIN_LINE.search(head):
            what = "parameter lines"
        elif suffix == ".raw" and (lab := self._sibling(path, ".lab")):
            with open(lab, "rb") as fh:
                label = fh.read(16)
            if len(label) < 16 or not self._label_type(label):
                return None
            what = ".lab companion with label types"
        else:
            return None
        companions = tuple(
            c for s in (".lab", ".raw", ".sin") if s != suffix and (c := self._sibling(path, s))
        )
        return Sniff(
            path,
            Format.PHILIPS_RAW_LAB,
            Kind.RAW,
            vendor="philips",
            confidence=1.0 if len(companions) == 2 else 0.7,
            evidence=(what, f"{len(companions)} of 2 companions found"),
            companions=companions,
        )

    @staticmethod
    def _label_type(label: bytes) -> bool:
        """Whether the 64-byte Philips label starts with a known label type at byte 14."""
        return 0x7F00 < struct.unpack_from("<H", label, 14)[0] < 0x7F06

    def data_list(self, path: Path, head: bytes) -> Sniff | None:
        if head.startswith(b"# === DATA DESCRIPTION FILE") and self.PAR_LINE.search(head):
            rec = self._sibling(path, ".rec")
            return Sniff(
                path,
                Format.PARREC,
                Kind.IMAGING,
                vendor="philips",
                confidence=1.0 if rec else 0.9,
                evidence=("PAR header markers", "REC companion found" if rec else "no REC"),
                companions=(rec,) if rec else (),
            )
        if path.suffix.lower() == ".rec" and (par := self._sibling(path, ".par")):
            return dataclasses.replace(self.sniff(par), path=path, companions=(par,))
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
        if len(head) < 4 or path.stat().st_size < self.PFILE_MIN_SIZE:
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

    def jcamp(self, path: Path, head: bytes) -> Sniff | None:
        """JCAMP-DX files with an NMR FID or spectrum (DATA TYPE NMR FID, NMR SPECTRUM, nD NMR
        ..., also within a compound LINK file)."""
        if not head.lstrip().startswith(b"##TITLE"):
            return None
        if self.JCAMP_NMR_TYPE.search(head):
            linked = bool(self.JCAMP_LINK.search(head))
            what = "compound file with an NMR block" if linked else "NMR data type"
            return Sniff(path, Format.JCAMP_NMR, Kind.SPECTROSCOPY, confidence=1.0,
                         evidence=(f"JCAMP-DX {what}",))
        return None

    def nmrml(self, path: Path, head: bytes) -> Sniff | None:
        """nmrML files: an XML document whose root element is nmrML (any namespace prefix)."""
        if b"<nmrML" not in head or not head.lstrip().startswith(b"<"):
            return None
        return Sniff(path, Format.NMRML, Kind.SPECTROSCOPY, confidence=1.0,
                     evidence=("XML root element nmrML",))

    def jeol(self, path: Path, head: bytes) -> Sniff | None:
        """JEOL Delta data files: the file identifier JEOL.NMR at the start."""
        if not head.startswith(b"JEOL.NMR"):
            return None
        return Sniff(path, Format.JEOL_JDF, Kind.SPECTROSCOPY, confidence=1.0,
                     evidence=("JEOL.NMR file identifier",))

    def tecmag(self, path: Path, head: bytes) -> Sniff | None:
        """Tecmag TNMR data files: the magic TNT1.nnn at the start."""
        if not re.match(rb"TNT1\.\d{3}", head[:8]):
            return None
        return Sniff(path, Format.TECMAG_TNT, Kind.SPECTROSCOPY, confidence=1.0,
                     evidence=("TNT1 magic",))

    def nmrpipe(self, path: Path, head: bytes) -> Sniff | None:
        """NMRPipe files: a 2048-byte header of floats whose third (FDFLTORDER) is 2.345 in the
        byte order of the file and whose first (FDMAGIC) is 0."""
        if len(head) < 2048:
            return None
        for order in "<>":
            magic, _, check = struct.unpack(f"{order}3f", head[:12])
            if magic == 0.0 and abs(check - 2.345) < 1e-6:
                return Sniff(path, Format.NMRPIPE, Kind.SPECTROSCOPY, confidence=1.0,
                             evidence=("NMRPipe header (FDFLTORDER 2.345)",))
        return None

    def sparky(self, path: Path, head: bytes) -> Sniff | None:
        """Sparky spectra: the identifier UCSF NMR at the start of the 180-byte header."""
        if not head.startswith(b"UCSF NMR"):
            return None
        return Sniff(path, Format.SPARKY, Kind.SPECTROSCOPY, confidence=1.0,
                     evidence=("UCSF NMR identifier",))

    def simpson(self, path: Path, head: bytes) -> Sniff | None:
        """SIMPSON output (fsave): a first line SIMP and an NP= line in the header."""
        lines = head.split(b"\n", 12)
        if lines[0].strip() != b"SIMP" or not any(x.strip().startswith(b"NP=") for x in lines):
            return None
        return Sniff(path, Format.SIMPSON, Kind.SPECTROSCOPY, confidence=1.0,
                     evidence=("SIMP header with NP",))

    def rds(self, path: Path, head: bytes) -> Sniff | None:
        """spant objects saved by saveRDS: the class names among the serialized strings."""
        if path.suffix.lower() != ".rds":
            return None
        data = path.read_bytes()
        for decompress in (gzip.decompress, bz2.decompress, lzma.decompress):
            try:
                data = decompress(data)
                break
            except (OSError, ValueError, lzma.LZMAError, EOFError):
                continue
        if not data.startswith(b"X\n"):
            return None
        for name, kind in self.SPANT_CLASSES:
            if len(name).to_bytes(4, "big") + name in data:
                return Sniff(path, Format.SPANT_RDS, kind, confidence=0.9,
                             evidence=("R serialization", f"spant class {name.decode()}"))
        return None

    def csdm(self, path: Path, head: bytes) -> Sniff | None:
        """Core Scientific Dataset Model: a JSON object whose first key is "csdm"."""
        if not re.match(rb'\s*\{\s*"csdm"\s*:', head):
            return None
        return Sniff(path, Format.CSDM, Kind.SPECTROSCOPY, confidence=1.0,
                     evidence=('JSON object starting with "csdm"',))

    def spinsolve(self, path: Path, head: bytes) -> Sniff | None:
        """Spinsolve data: Prospa data files (magic SORPATAD; plot files .pt1 differ)."""
        if not head.startswith(b"SORPATAD"):
            return None
        return Sniff(path, Format.SPINSOLVE, Kind.SPECTROSCOPY, confidence=1.0,
                     evidence=("Prospa data magic",))

    def rs2d(self, path: Path, head: bytes) -> Sniff | None:
        """The header.xml and data.dat of an RS2D experiment (or of its Proc/<n> folders)."""
        if path.name not in ("header.xml", "data.dat") or not self._rs2d_folder(path.parent):
            return None
        return Sniff(path, Format.RS2D, Kind.SPECTROSCOPY, vendor="rs2d", confidence=1.0,
                     evidence=(f"RS2D {path.name} beside its partner",))

    @staticmethod
    def _rs2d_folder(folder: Path) -> bool:
        """header.xml (root header with params) and data.dat in ``folder``."""
        header = folder / "header.xml"
        if not (header.is_file() and (folder / "data.dat").is_file()):
            return False
        with open(header, "rb") as fh:
            head = fh.read(512)
        return b"<header>" in head and b"<params>" in head

    def rnmrtk(self, path: Path, head: bytes) -> Sniff | None:
        """RNMRTK data: a .sec file and its .par (Dom and Layout lines), either of the two."""
        if path.suffix.lower() not in (".sec", ".par"):
            return None
        sec, par = path.with_suffix(".sec"), path.with_suffix(".par")
        if not (sec.is_file() and par.is_file()):
            return None
        with open(par, "rb") as fh:
            text = fh.read(4096)
        if b"\nDom" not in text or b"\nLayout" not in text:
            return None
        return Sniff(path, Format.RNMRTK, Kind.SPECTROSCOPY, confidence=1.0,
                     evidence=(f"{path.suffix} with its RNMRTK partner",))

    JEMRIS_MOTION_ROWS = re.compile(rb"\A\s*(\S+\s+){2}\S+\s*\n(\s*(\S+\s+){6}\S+\s*\n){2}")

    def jemris_motion(self, path: Path, head: bytes) -> Sniff | None:
        """JEMRIS motion trajectories: a .dat file named motion..., one line of three numbers
        (the rotation origin) and then lines of seven (time, translation, rotation)."""
        if path.suffix.lower() != ".dat" or "motion" not in path.name.lower():
            return None
        if not self.JEMRIS_MOTION_ROWS.match(head):
            return None
        return Sniff(path, Format.JEMRIS_MOTION, Kind.PHANTOM, confidence=0.9,
                     evidence=("motion .dat: 3 numbers, then rows of 7",))

    BRAINVOYAGER_TEXT = {  # extension: (format, kind, word the header holds)
        ".prt": (Format.BRAINVOYAGER_PRT, Kind.PROTOCOL, b"NrOfConditions"),
        ".sdm": (Format.BRAINVOYAGER_SDM, Kind.PROTOCOL, b"NrOfPredictors"),
        ".voi": (Format.BRAINVOYAGER_VOI, Kind.IMAGING, b"NrOfVOIs"),
        ".poi": (Format.BRAINVOYAGER_POI, Kind.SURFACE, b"NrOfPOIs"),
        ".trf": (Format.BRAINVOYAGER_TRF, Kind.TRANSFORM, b"DataFormat"),
    }
    BRAINVOYAGER_BINARY = {
        ".vmr": (Format.BRAINVOYAGER_VMR, Kind.IMAGING), ".vtc": (Format.BRAINVOYAGER_VTC,
                                                                  Kind.IMAGING),
        ".glm": (Format.BRAINVOYAGER_GLM, Kind.IMAGING), ".vmp": (Format.BRAINVOYAGER_VMP,
                                                                  Kind.IMAGING),
        ".srf": (Format.BRAINVOYAGER_SRF, Kind.SURFACE), ".smp": (Format.BRAINVOYAGER_SMP,
                                                                  Kind.SURFACE),
        ".mtc": (Format.BRAINVOYAGER_MTC, Kind.SURFACE), ".ssm": (Format.BRAINVOYAGER_SSM,
                                                                  Kind.SURFACE),
    }

    def brainvoyager(self, path: Path, head: bytes) -> Sniff | None:
        """BrainVoyager files by extension (also gzipped) and a check of their first bytes: text
        files hold "FileVersion" and the field naming their content; VMR dimensions fit the
        size, VMP starts with its magic number, the others with a plausible version."""
        name = path.name.lower().removesuffix(".gz")
        suffix = "." + name.rsplit(".", 1)[-1] if "." in name else ""
        if suffix not in self.BRAINVOYAGER_TEXT and suffix not in self.BRAINVOYAGER_BINARY:
            return None
        if head[:2] == GZIP_MAGIC:
            try:
                head = zlib.decompressobj(31).decompress(head, self.HEAD_BYTES)
            except zlib.error:
                return None
        if suffix in self.BRAINVOYAGER_TEXT:
            fmt, kind, word = self.BRAINVOYAGER_TEXT[suffix]
            if b"FileVersion" not in head or word not in head:
                return None
            evidence = f"{suffix} with FileVersion and {word.decode()}"
        else:
            fmt, kind = self.BRAINVOYAGER_BINARY[suffix]
            if not self._brainvoyager_binary(suffix, head, path):
                return None
            evidence = f"{suffix} with a fitting header"
        return Sniff(path, fmt, kind, vendor="brainvoyager", confidence=0.9,
                     evidence=(evidence,))

    @staticmethod
    def _brainvoyager_binary(suffix: str, head: bytes, path: Path) -> bool:
        if len(head) < 16:
            return False
        if suffix == ".vmr":
            version, *dims = struct.unpack_from("<4H", head)
            plain = path.name.lower().endswith(".vmr")
            return 1 <= version <= 10 and all(dims) and (
                not plain or path.stat().st_size >= 8 + dims[0] * dims[1] * dims[2])
        if suffix == ".vmp":
            return struct.unpack_from("<I", head)[0] == 0xA1B2C3D4
        if suffix == ".srf":
            version, _, vertices, triangles = struct.unpack_from("<fiii", head)
            return 1.0 <= version < 10.0 and vertices > 0 and triangles > 0
        if suffix == ".mtc":
            return 0 < struct.unpack_from("<i", head)[0] < 10
        version = struct.unpack_from("<h", head)[0]  # VTC, GLM, SMP, SSM: a short version first
        return 0 < version < 10

    TRX_ARRAY = re.compile(r"^(positions\.3|offsets)\.(u?int|float)\d+$")

    def tractograms(self, path: Path, head: bytes) -> Sniff | None:
        """Streamlines: TrackVis .trk ("TRACK" magic, header size 1000 at byte 996), MRtrix .tck
        ("mrtrix tracks" first line) and their scalars .tsf, TRX zip archives (header.json,
        positions and offsets arrays), DSI Studio .tt (MATLAB v4), VTK legacy POLYDATA (its kind
        from the cells it holds: LINES or POLYGONS)."""
        found = None
        if head.startswith(b"TRACK") and len(head) >= 1000 and 1000 in struct.unpack_from(
                "<i", head, 996) + struct.unpack_from(">i", head, 996):
            found = Format.TRK, Kind.TRACTOGRAM, "TRACK magic, header size 1000"
        elif head.startswith(b"mrtrix tracks"):
            found = Format.TCK, Kind.TRACTOGRAM, "mrtrix tracks first line"
        elif head.startswith(b"mrtrix track scalars"):
            found = Format.TSF, Kind.TRACTOGRAM, "mrtrix track scalars first line"
        elif path.name.lower().endswith((".tt", ".tt.gz")) and self._matlab_v4_track(head):
            found = Format.DSI_STUDIO_TT, Kind.TRACTOGRAM, "MATLAB v4 variables of a .tt file"
        elif head.startswith(b"PK\x03\x04") and self._trx(path):
            found = Format.TRX, Kind.TRACTOGRAM, "zip with header.json, positions and offsets"
        elif head.startswith(b"# vtk DataFile Version") and re.search(
                rb"^DATASET\s+POLYDATA", head, re.M | re.I):
            with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as m:
                lines = re.search(rb"\nLINES\s", m) is not None
            found = (Format.VTK_POLYDATA, Kind.TRACTOGRAM if lines else Kind.SURFACE,
                     "VTK legacy POLYDATA" + (" with LINES" if lines else ""))
        if found is None:
            return None
        return Sniff(path, found[0], found[1], confidence=0.9, evidence=(found[2],))

    @staticmethod
    def _mz3(head: bytes) -> bool:
        if head[:2] == GZIP_MAGIC:
            try:
                head = zlib.decompressobj(31).decompress(head, 16)
            except zlib.error:
                return False
        return head[:2] == b"MZ"

    @staticmethod
    def _matlab_v4_track(head: bytes) -> bool:
        """A (gzipped) MATLAB version 4 file: a small type code, then the first variable's
        name, as DSI Studio writes .tt, .src and .fib files."""
        if head[:2] == GZIP_MAGIC:
            try:
                head = zlib.decompressobj(31).decompress(head, 64)
            except zlib.error:
                return False
        if len(head) < 24:
            return False
        kind, rows, columns, imaginary, length = struct.unpack_from("<5i", head)
        return 0 <= kind < 100 and rows > 0 and columns > 0 and imaginary in (0, 1) and (
            0 < length < 64)

    def _trx(self, path: Path) -> bool:
        """A zip archive or folder with header.json and TRX positions and offsets arrays."""
        if path.is_dir():
            names = {p.name for p in path.iterdir()}
        else:
            try:
                with zipfile.ZipFile(path) as archive:
                    names = set(archive.namelist())
            except zipfile.BadZipFile:
                return False
        return "header.json" in names and sum(bool(self.TRX_ARRAY.match(n)) for n in names) == 2

    SURFACE_MAGICS = {b"\xff\xff\xfe": "triangle", b"\xff\xff\xff": "quad",
                      b"\xff\xff\xfd": "new quad"}  # FreeSurfer surfaces (curv shares 0xffffff)

    def surfaces(self, path: Path, head: bytes) -> Sniff | None:
        """Surface meshes and per-vertex data: GIFTI (a <GIFTI root, also inside .gii.gz and
        .gii.bz2); FreeSurfer surfaces by magic number and a size that holds the vertices and
        faces they state, new-format curv files (magic 0xffffff, exactly 15 + 4 x vertices bytes),
        .annot files (vertex count, then (vertex, value) pairs), .label text files, MZ3 meshes."""
        name, size = path.name.lower(), path.stat().st_size
        found = None
        if name.endswith((".gii.gz", ".gii.bz2")):
            opener = gzip.open if name.endswith(".gz") else bz2.open
            try:
                with opener(path, "rb") as fh:
                    head = fh.read(self.HEAD_BYTES)
            except (OSError, EOFError):
                return None
        if re.search(rb"^\s*(<\?xml[^>]*>\s*)?(<!DOCTYPE[^>]*>\s*)?<GIFTI[\s>]", head):
            found = Format.GIFTI, "GIFTI root element"
        elif head[:3] in self.SURFACE_MAGICS and len(head) >= 15:
            found = self._freesurfer_binary(head, size)
        elif name.endswith(".annot") and len(head) >= 4:
            count = struct.unpack(">i", head[:4])[0]
            if 0 < count and size >= 4 + 8 * count:
                found = Format.FREESURFER_ANNOT, ".annot with room for its vertex count"
        elif name.endswith(".label") and head.startswith(b"#!ascii label"):
            found = Format.FREESURFER_LABEL, "#!ascii label first line"
        elif name.endswith(".mz3") and self._mz3(head):
            found = Format.MZ3, "MZ3 magic 0x5A4D"
        if found is None:
            return None
        return Sniff(path, found[0], Kind.SURFACE, confidence=0.9, evidence=(found[1],))

    def _freesurfer_binary(self, head: bytes, size: int) -> tuple[Format, str] | None:
        """A FreeSurfer curv or surface file from its magic number and its size."""
        magic = self.SURFACE_MAGICS[head[:3]]
        if magic == "quad" and size == 15 + 4 * struct.unpack(">i", head[3:7])[0]:
            return Format.FREESURFER_MORPH, "curv magic 0xffffff, 15 + 4 x vertices bytes"
        if magic == "triangle":
            end = head.find(b"\n\n", 3)  # creation line, ended by two newlines
            if end < 0 or len(head) < end + 10:
                return None
            vertices, faces = struct.unpack(">ii", head[end + 2:end + 10])
            needed = end + 10 + 12 * vertices + 12 * faces
        else:  # three-byte counts, int16 (quad) or float32 (new quad) coordinates
            vertices = int.from_bytes(head[3:6], "big")
            faces = int.from_bytes(head[6:9], "big")
            needed = 9 + (6 if magic == "quad" else 12) * vertices + 12 * faces
        if vertices <= 0 or faces <= 0 or size < needed:
            return None
        return Format.FREESURFER_SURFACE, f"{magic} surface magic, size fits the counts"

    def transforms(self, path: Path, head: bytes) -> Sniff | None:
        """Linear transforms: FreeSurfer LTA (type and nxforms lines), ITK (.tfm text header,
        or HDF5 with TransformGroup), AFNI (non-comment rows of 12 numbers), FSL FLIRT (four
        rows of four numbers, the last 0 0 0 1 to
        single precision)."""
        found = None
        if b"nxforms" in head and re.search(rb"^\s*type\s*=\s*\d", head, re.M):
            found = Format.TRANSFORM_LTA, "LTA type and nxforms lines"
        elif head.startswith(b"#Insight Transform File"):
            found = Format.TRANSFORM_ITK, "ITK transform file header"
        elif head.startswith(b"\x89HDF") and path.suffix.lower() == ".h5":
            try:
                with h5py.File(path, "r") as fh:
                    if "TransformGroup" in fh:
                        found = Format.TRANSFORM_ITK, "HDF5 with TransformGroup"
            except OSError:
                return None
        elif self._is_text(head) and len(head) < self.HEAD_BYTES:
            rows = [line.split() for line in head.decode("ascii", "replace").splitlines()
                    if line.strip() and not line.lstrip().startswith("#")]
            numbers = all(self.NUMBER.fullmatch(v) for row in rows for v in row) and rows
            if numbers and all(len(row) == 12 for row in rows):
                found = Format.TRANSFORM_AFNI, "rows of 12 numbers"
            elif numbers and len(rows) == 4 and all(len(row) == 4 for row in rows) and all(
                    abs(float(v) - w) < 1e-4 for v, w in zip(rows[3], (0, 0, 0, 1), strict=True)):
                found = Format.TRANSFORM_FSL, "4 x 4 matrix ending 0 0 0 1"
        if found is None:
            return None
        return Sniff(path, found[0], Kind.TRANSFORM, confidence=0.9, evidence=(found[1],))

    def topspin(self, path: Path, head: bytes) -> Sniff | None:
        """Bruker TopSpin files outside ParaVision scans: fid or ser and the JCAMP-DX parameter
        files (acqu, acqus, ...) of an experiment folder with acqus; processed spectra (1r, 2rr,
        ...) and the parameter files of a pdata/<n> folder with procs."""
        name, folder = path.name, path.parent
        if self._paravision(folder):
            return None
        parameters = head.startswith(b"##TITLE=")
        if (folder / "acqus").is_file() and (name in ("fid", "ser") or parameters):
            what = f"{name} with acqus" if not parameters else f"JCAMP-DX parameter file {name}"
            return self._topspin(path, Format.TOPSPIN, what)
        if (folder / "procs").is_file() and (self.TOPSPIN_SPECTRA.match(name) or parameters):
            what = f"{name} with procs" if not parameters else f"JCAMP-DX parameter file {name}"
            return self._topspin(path, Format.TOPSPIN_PROCESSED, what)
        return None

    def varian(self, path: Path, head: bytes) -> Sniff | None:
        if path.name == "procpar" or (path.name == "fid" and (path.parent / "procpar").is_file()):
            return self._varian_scan(path, path.parent, f"{path.name} with procpar")
        return None

    def pulseq(self, path: Path, head: bytes) -> Sniff | None:
        if not self._is_text(head):
            return None
        if head.startswith(b"# Pulseq") or (b"[VERSION]" in head and b"major" in head):
            return Sniff(
                path,
                Format.PULSEQ,
                Kind.SEQUENCE,
                confidence=1.0,
                evidence=("[VERSION] section" if b"[VERSION]" in head else "Pulseq comment",),
            )
        return None

    def bart(self, path: Path, head: bytes) -> Sniff | None:
        """BART arrays: a .hdr starting with "# Dimensions" beside a .cfl, a .cfl beside such a
        .hdr, or a .ra file with the rawarray magic."""
        suffix = path.suffix.lower()
        if suffix == ".ra" and head[:8] == b"rawarray":
            return Sniff(
                path, Format.BART_CFL, Kind.IMAGING, confidence=1.0, evidence=("rawarray magic",)
            )
        if suffix == ".hdr" and head.startswith(b"# Dimensions"):
            cfl = self._sibling(path, ".cfl")
            return Sniff(
                path,
                Format.BART_CFL,
                Kind.IMAGING,
                confidence=1.0 if cfl else 0.7,
                evidence=("# Dimensions line", "CFL companion found" if cfl else "no CFL"),
                companions=(cfl,) if cfl else (),
            )
        if suffix == ".cfl" and (hdr := self._sibling(path, ".hdr")):
            with open(hdr, "rb") as fh:
                first = fh.read(12)
            if first.startswith(b"# Dimensions"):
                return Sniff(
                    path,
                    Format.BART_CFL,
                    Kind.IMAGING,
                    confidence=1.0,
                    evidence=(".cfl extension", "HDR companion with Dimensions"),
                    companions=(hdr,),
                )
        return None

    def physio(self, path: Path, head: bytes) -> Sniff | None:
        """Physiological logs: Siemens PMU files (by extension and their four header values),
        CMRR physio logs (LogDataType and UUID lines), Philips SCANPHYSLOG files (the "# v1raw"
        column line), GE PPGData/RESPData/ECGData files (by name, numbers only), BIOPAC
        AcqKnowledge files (.acq with a plausible file revision and channel count), HCP physio
        logs (*_Physio_log.txt, three integer columns) and BIDS physio tables (a JSON sidecar
        with SamplingFrequency and Columns)."""
        suffix, name = path.suffix.lower(), path.name
        if suffix in (".puls", ".resp", ".ecg", ".ext", ".ext2") and re.match(
            rb"\s*\d+ \d+ \d+ \d+\s", head
        ):
            return Sniff(path, Format.SIEMENS_PMU, Kind.PHYSIO, vendor="siemens", confidence=1.0,
                         evidence=(f"{suffix} file", "PMU header values"))
        if suffix == ".log" and b"LogDataType" in head and b"UUID" in head:
            return Sniff(path, Format.CMRR_PHYSIO, Kind.PHYSIO, vendor="siemens", confidence=1.0,
                         evidence=("LogDataType and UUID lines",))
        if head.startswith(b"##") and re.search(rb"^# v1raw", head, re.M):
            return Sniff(path, Format.PHILIPS_PHYSLOG, Kind.PHYSIO, vendor="philips",
                         confidence=1.0, evidence=("# v1raw column line",))
        if re.match(r"(PPG|RESP|ECG)Data_", name) and re.fullmatch(
            rb"[\s\d-]*", head[: head.rfind(b"\n") + 1] if b"\n" in head else head
        ):
            return Sniff(path, Format.GE_PHYSIO, Kind.PHYSIO, vendor="ge", confidence=0.9,
                         evidence=("GE physio file name", "numbers only"))
        if suffix == ".acq" and len(head) >= 12 and any(
            30 <= struct.unpack_from(f"{o}I", head, 2)[0] <= 1000
            and 0 < struct.unpack_from(f"{o}h", head, 10)[0] <= 1024 for o in "<>"
        ):
            return Sniff(path, Format.BIOPAC_ACQ, Kind.PHYSIO, vendor="biopac", confidence=1.0,
                         evidence=(".acq file", "AcqKnowledge graph header"))
        if name.endswith("_Physio_log.txt") and re.match(rb"(\d+\s+\d+\s+\d+\s*\n){3}", head):
            return Sniff(path, Format.HCP_PHYSIO, Kind.PHYSIO, vendor="siemens", confidence=1.0,
                         evidence=("HCP physio file name", "three integer columns"))
        if name.endswith((".tsv", ".tsv.gz")):
            sidecar = path.with_name(name.removesuffix(".gz").removesuffix(".tsv") + ".json")
            if sidecar.is_file():
                try:
                    info = json.loads(sidecar.read_text())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return None
                if isinstance(info, dict) and {"SamplingFrequency", "Columns"} <= set(info):
                    return Sniff(path, Format.BIDS_PHYSIO, Kind.PHYSIO, confidence=1.0,
                                 evidence=("JSON sidecar with SamplingFrequency and Columns",),
                                 companions=(sidecar,))
        return None

    def protocols(self, path: Path, head: bytes) -> Sniff | None:
        """Siemens protocols: XProtocol text (a .pro file), exam trees (.exar1, SQLite) and
        printed protocols (PrintOut XML or its text form); Philips ExamCards (SOAP XML); Yarra
        reconstruction modes (.mode) and module manifests (.ymf)."""
        suffix = path.suffix.lower()
        if suffix == ".mode" and re.search(rb"^\[(ClientConfig|Reconstruction)\]", head, re.M):
            return Sniff(path, Format.YARRA_MODE, Kind.PROTOCOL, confidence=1.0,
                         evidence=(".mode file with Yarra sections",))
        if suffix == ".ymf" and re.search(rb"^\[YarraModule\]", head, re.M):
            return Sniff(path, Format.YARRA_MODULE, Kind.PROTOCOL, confidence=1.0,
                         evidence=(".ymf file with [YarraModule]",))
        if head.lstrip().startswith(b"<XProtocol>"):
            return Sniff(path, Format.SIEMENS_PRO, Kind.PROTOCOL, vendor="siemens",
                         confidence=1.0, evidence=("XProtocol text",))
        text = head.removeprefix(b"\xef\xbb\xbf")
        if text.lstrip().startswith(b"<SOAP-ENV:Envelope") and b"ExamCards" in text:
            return Sniff(path, Format.PHILIPS_EXAMCARD, Kind.PROTOCOL, vendor="philips",
                         confidence=1.0, evidence=("SOAP envelope of an ExamCard",))
        printed_xml = b"<PrintOut>" in text and (b"SIEMENS" in text or b"Siemens" in text)
        printed_text = b"SIEMENS MAGNETOM" in text[:400] and b"-----" in text[:800]
        if printed_xml or printed_text:
            return Sniff(path, Format.SIEMENS_PRINTPROT, Kind.PROTOCOL, vendor="siemens",
                         confidence=0.9, evidence=("printed protocol",))
        if head.startswith(b"SQLite format 3\0") and path.suffix.lower().startswith(".exar"):
            return Sniff(path, Format.SIEMENS_EXAR, Kind.PROTOCOL, vendor="siemens",
                         confidence=1.0, evidence=("SQLite database", f"{path.suffix} extension"))
        return None

    def toppe(self, path: Path, head: bytes) -> Sniff | None:
        """TOPPE files: the module list (first line "Total number of unique cores"), the scan
        loop (first line "nt<tab>maxslice"), toppe<N>.entry files (by name, the module list on
        the second line) and .mod files (a big-endian description length, the description and
        the "b1max:" line)."""
        lines = head.splitlines()
        first = lines[0].strip() if lines else b""
        if first.startswith(b"Total number of unique cores"):
            return Sniff(path, Format.TOPPE_MODULES, Kind.SEQUENCE, vendor="ge", confidence=1.0,
                         evidence=("Total number of unique cores line",))
        if first.startswith(b"nt\tmaxslice"):
            return Sniff(path, Format.TOPPE_LOOP, Kind.SEQUENCE, vendor="ge", confidence=1.0,
                         evidence=("nt, maxslice, ... header line",))
        if re.fullmatch(r"toppe\w*\.entry", path.name, re.I) and len(lines) > 2 and lines[
                1].strip().endswith(b".txt"):
            return Sniff(path, Format.TOPPE_ENTRY, Kind.SEQUENCE, vendor="ge", confidence=1.0,
                         evidence=("toppe entry file name", "module list on line 2"))
        if path.suffix.lower() == ".mod" and len(head) > 2:
            size = struct.unpack_from(">h", head, 0)[0]
            if 0 < size < len(head) - 8 and b"b1max:" in head[2 + size : 2 + size + 32]:
                return Sniff(path, Format.TOPPE_MOD, Kind.SEQUENCE, vendor="ge", confidence=1.0,
                             evidence=(".mod file", "description and b1max line"))
        return None

    def yardl(self, path: Path, head: bytes) -> Sniff | None:
        """MRD (version 2) files: yardl binary ("yardl" magic) or NDJSON (a first line
        {"yardl": ...}) whose embedded schema names the protocol Mrd (a stream) or
        MrdNoiseCovariance (a noise covariance matrix)."""
        protocol = YardlFile.protocol_name(head)
        if protocol not in ("Mrd", "MrdNoiseCovariance"):
            return None
        encoding = "binary" if head.startswith(YardlFile.MAGIC) else "NDJSON"
        return Sniff(
            path, Format.MRD, Kind.RAW, confidence=1.0,
            evidence=(f"yardl {encoding}", f"protocol {protocol}"),
        )

    def text_formats(self, path: Path, head: bytes) -> Sniff | None:
        if head.startswith(b"NRRD000"):
            where = "detached" if b"data file:" in head or b"datafile:" in head else "attached"
            return self._image(path, Format.NRRD, f"NRRD magic, data {where}")
        if head.startswith(b"mrtrix image"):
            return self._image(path, Format.MRTRIX, "mrtrix image magic")
        if head.startswith(b"ObjectType") and b"NDims" in head and b"ElementType" in head:
            return self._image(path, Format.METAIMAGE, "MetaImage ObjectType header")
        if head.startswith(b"#!/usr/local/fdf/startup"):
            return self._image(path, Format.VARIAN_FDF, "fdf startup line")
        if b"jMRUI Data Textfile" in head:
            return Sniff(
                path,
                Format.JMRUI_TEXT,
                Kind.SPECTROSCOPY,
                confidence=1.0,
                evidence=("jMRUI text marker",),
            )
        if self._is_text(head) and (columns := self._numeric_columns(head)):
            return Sniff(
                path,
                Format.TEXT_FID,
                Kind.SPECTROSCOPY,
                confidence=0.5,
                evidence=(f"headerless text, {columns} numeric columns",),
                detail={"columns": columns},
            )
        return None

    def _numeric_columns(self, head: bytes) -> int:
        """Columns of a text file that starts with nothing but numbers in an even number of
        columns (pairs of real and imaginary parts), 0 otherwise."""
        lines = head.decode("ascii", "replace").splitlines()
        if len(head) >= self.HEAD_BYTES:
            lines = lines[:-1]  # cut off
        lines = [line for line in lines if line.strip()]
        if len(lines) < self.TEXT_FID_LINES:
            return 0
        counts = set()
        for line in lines:
            fields = [f for f in self.FIELD_SEPARATOR.split(line) if f]
            if not all(self.NUMBER.fullmatch(f) for f in fields):
                return 0
            counts.add(len(fields))
        columns = counts.pop() if len(counts) == 1 else 0
        if columns > self.TEXT_FID_MAX_COLUMNS:
            return 0
        return columns if columns >= 2 and columns % 2 == 0 else 0

    def companion_binary(self, path: Path, head: bytes) -> Sniff | None:
        """Headerless binary data next to the text header that describes it."""
        for suffix, header, fmt in self.COMPANIONS:
            if path.suffix.lower() == suffix and (text := self._sibling(path, header)):
                return Sniff(
                    path,
                    fmt,
                    Kind.IMAGING if fmt in (Format.SIVIC_IDF, Format.STIMULATE)
                    else Kind.SPECTROSCOPY,
                    confidence=0.9,
                    evidence=(f"{suffix} data with {text.name}",),
                    companions=(text,),
                )
        return None

    def analysis_text(self, path: Path, head: bytes) -> Sniff | None:
        """Text formats of LCModel, TARQUIN, VASF, SIVIC and DMIWizard, recognised by their own
        markers (DMIWizard's maps by their names)."""
        upper = head.upper()
        stripped = head.lstrip()
        if stripped.startswith(b"{") and b'"basis_re"' in head:
            evidence = ("FSL-MRS basis JSON",)
            fmt = Format.FSL_MRS_BASIS
            return Sniff(path, fmt, Kind.BASIS_SET, confidence=1.0, evidence=evidence)
        if b"<FITT_Generic_XML" in head and b"PRIOR_METABOLITE_INFORMATION" in path.read_bytes():
            evidence = ("MIDAS FITT_Generic_XML prior",)
            fmt = Format.MIDAS_PRIOR
            return Sniff(path, fmt, Kind.BASIS_SET, confidence=1.0, evidence=evidence)
        if self.PYAMARES.search(head):
            evidence = ("PyAMARES prior table (Index, Initial Values)",)
            fmt = Format.PYAMARES_PRIOR
            return Sniff(path, fmt, Kind.BASIS_SET, confidence=0.9, evidence=evidence)
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
        if (match := self.DMIWIZARD_MAP.match(path.name)) and self.NUMBER.fullmatch(
                head.split(b"\n", 1)[0].strip().decode("ascii", "replace")):
            evidence = (f"DMIWizard {match[1]} name", "a column of numbers")
            return Sniff(path, Format.DMIWIZARD_MAP, Kind.IMAGING, confidence=0.9,
                         evidence=evidence)
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
        elif b"<prior" in head:
            evidence = ("<vespa_export> with <prior>",)
            fmt = Format.VESPA_PRIOR
            return Sniff(path, fmt, Kind.BASIS_SET, confidence=1.0, evidence=evidence)
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

    def image_files(self, path: Path, head: bytes) -> Sniff | None:
        """Image formats nibabel reads: MGH, MINC 1 and 2, AFNI, Analyze (and their data halves);
        gzipped MRtrix images; GE Signa images; JNIfTI (JSON or BJData); Vista; Stimulate; VTK
        images (legacy STRUCTURED_POINTS, XML ImageData); DSI Studio .src and .fib files."""
        name, suffix = path.name.lower(), path.suffix.lower()
        if name.endswith(".mif.gz") and head[:2] == GZIP_MAGIC:
            try:
                if zlib.decompressobj(31).decompress(head, 32).startswith(b"mrtrix image"):
                    return self._image(path, Format.MRTRIX, "gzipped mrtrix image magic")
            except zlib.error:
                pass
        if signa := self._ge_signa(path, head):
            return self._image(path, Format.GE_SIGNA, signa)
        if name.endswith((".mgh", ".mgz")):
            if head[:2] == b"\x1f\x8b":
                head = zlib.decompressobj(31).decompress(head, 16)
            if head[:4] == b"\x00\x00\x00\x01":
                return self._image(path, Format.MGH, "MGH version 1")
        if suffix == ".mnc" and head[:4] in (b"CDF\x01", b"CDF\x02"):
            return self._image(path, Format.MINC1, "NetCDF .mnc")
        if suffix == ".mnc" and head[:8] == b"\x89HDF\r\n\x1a\n":
            return self._image(path, Format.MINC2, "HDF5 .mnc")
        if suffix == ".head" and b"name" in head and b"type" in head:
            stem = path.name[:-5]
            briks = [path.with_name(stem + s) for s in (".BRIK", ".BRIK.gz", ".brik")]
            brik = next((b for b in briks if b.is_file()), None)
            return self._image(path, Format.AFNI, "AFNI attributes", brik, "BRIK")
        if name.endswith((".brik", ".brik.gz")):
            header = path.with_name(self.BRIK.sub(".HEAD", path.name))
            if header.is_file():
                return dataclasses.replace(self.sniff(header), path=path, companions=(header,))
        if suffix == ".spr" and self._is_text(head) and b"dim:" in head and b"dataType" in head:
            sdt = self._sibling(path, ".sdt")
            return self._image(path, Format.STIMULATE, "Stimulate .spr header", sdt, "SDT")
        for ending, fmt in ((".src", Format.DSI_STUDIO_SRC), (".fib", Format.DSI_STUDIO_FIB)):
            if name.endswith((ending, ending + ".gz")) and self._matlab_v4_track(head):
                return self._image(path, fmt, f"MATLAB v4 variables of a {ending} file")
        if head.startswith(b"# vtk DataFile Version") and re.search(
                rb"^DATASET\s+STRUCTURED_POINTS", head, re.M | re.I):
            return self._image(path, Format.VTK_IMAGE, "VTK legacy STRUCTURED_POINTS")
        if b"<VTKFile" in head[:512] and re.search(rb'<VTKFile[^>]*type="ImageData"', head):
            return self._image(path, Format.VTK_IMAGE_XML, "VTK XML ImageData")
        if head.startswith(b"V-data 2 {") and b"image" in head:
            return self._image(path, Format.VISTA, "V-data header with an image")
        if head[:1] == b"{" and b"NIFTIHeader" in head:
            binary = head[1:2] not in b' \t\r\n"}'  # a BJData key length marker
            fmt = Format.JNIFTI_BINARY if binary else Format.JNIFTI
            return self._image(path, fmt, "JData object with a NIFTIHeader")
        if suffix == ".hdr" and head[:4] in self.ANALYZE_SIZES:
            img = self._sibling(path, ".img")
            return self._image(path, Format.ANALYZE, "Analyze header size 348", img, "IMG")
        if suffix == ".img" and (hdr := self._sibling(path, ".hdr")):
            return dataclasses.replace(self.sniff(hdr), path=path, companions=(hdr,))
        return None

    @staticmethod
    def _image(
        path: Path, fmt: Format, evidence: str, companion: Path | None = None, part: str = ""
    ) -> Sniff:
        """An image format; with ``part`` the format needs a companion data file."""
        found = companion is not None or not part
        notes = (evidence,)
        if part:
            notes += (f"{part} companion {'found' if found else 'missing'}",)
        return Sniff(
            path,
            fmt,
            Kind.IMAGING,
            confidence=1.0 if found else 0.8,
            evidence=notes,
            companions=(companion,) if companion else (),
        )

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
        """ISMRMRD: a group with an ``xml`` header and a ``data`` table (raw) or ``image_*``
        subgroups; fastMRI: ``kspace`` and ``ismrmrd_header`` at the root; GE ScanArchive:
        ``DownloadData.xml`` and an ``Acquisition`` group at the root; JEMRIS signals
        (``signal/channels`` and ``signal/times``) and phantoms (``sample/data``); KomaMRI
        phantoms (``position`` and ``contrast`` groups, or x, y, z and ρ at the root)."""
        if head[:8] != b"\x89HDF\r\n\x1a\n":
            return None
        try:
            with h5py.File(path, "r") as fh:
                names = list(fh.keys())
                groups = {
                    name: list(fh[name].keys())
                    for name in names
                    if isinstance(fh[name], h5py.Group)
                }
        except OSError as e:
            return Sniff(path, Format.UNKNOWN, evidence=(f"HDF5 container, unreadable: {e}",))
        for name, members in groups.items():
            images = [m for m in members if m.startswith("image_")]
            if "xml" in members and "data" in members:
                kind, what = Kind.RAW, f"ISMRMRD xml header and {name}/data"
            elif images:
                kind, what = Kind.IMAGING, f"ISMRMRD image groups in {name}: {len(images)}"
            else:
                continue
            return Sniff(
                path,
                Format.ISMRMRD,
                kind,
                confidence=1.0,
                evidence=("HDF5 magic", what),
                detail={"groups": groups},
            )
        if "DownloadData.xml" in names and "Acquisition" in groups:
            return Sniff(
                path,
                Format.GE_SCANARCHIVE,
                Kind.RAW,
                vendor="ge",
                confidence=1.0,
                evidence=("HDF5 magic", "DownloadData.xml and the Acquisition group"),
                detail={"names": names},
            )
        if "signal" in groups and {"channels", "times"} <= set(groups["signal"]):
            return Sniff(path, Format.JEMRIS_SIGNAL, Kind.RAW, confidence=1.0,
                         evidence=("HDF5 magic", "signal/channels and signal/times"),
                         detail={"names": names})
        if "sample" in groups and "data" in groups["sample"]:
            return Sniff(path, Format.JEMRIS_PHANTOM, Kind.PHANTOM, confidence=1.0,
                         evidence=("HDF5 magic", "sample/data"), detail={"names": names})
        if {"position", "contrast"} <= set(groups) or {"x", "y", "z", "ρ"} <= set(names):
            what = "position and contrast groups" if "position" in groups else "x, y, z and ρ"
            return Sniff(path, Format.KOMA_PHANTOM, Kind.PHANTOM, confidence=1.0,
                         evidence=("HDF5 magic", what), detail={"names": names})
        if "kspace" in names and "ismrmrd_header" in names:
            return Sniff(
                path,
                Format.FASTMRI_H5,
                Kind.RAW,
                confidence=1.0,
                evidence=("HDF5 magic", "kspace and ismrmrd_header datasets"),
                detail={"names": names},
            )
        return Sniff(
            path,
            Format.UNKNOWN,
            evidence=("HDF5 container, unknown layout",),
            detail={"names": names},
        )

    def archive(self, path: Path, head: bytes) -> Sniff | None:
        """Zip archives (local file header or, when empty, end-of-directory record first) that
        no format rule claimed: their members are sniffed when the archive is loaded. Office
        documents (Office Open XML, OpenDocument) are zips too and are passed over."""
        if not head.startswith((b"PK\x03\x04", b"PK\x05\x06")):
            return None
        try:
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
        except (zipfile.BadZipFile, OSError):
            return None
        if "[Content_Types].xml" in names or names[:1] == ["mimetype"]:
            return None
        return Sniff(path, Format.ZIP_ARCHIVE, Kind.ARCHIVE, confidence=1.0,
                     evidence=("zip signature",))

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
    def _ge_signa(path: Path, head: bytes) -> str | None:
        """Evidence of a GE Signa image: the Genesis 5.x "IMGF" pixel header or suite header
        with the product "SIGNA", or a Signa 4.x study header block whose series block names the
        plane."""
        if head[:4] == b"IMGF":
            return "GE Genesis IMGF magic"
        if head[7:12] == b"SIGNA" and head[114 + 305 : 114 + 307] in (b"MR", b"CT"):
            return "GE Genesis suite header"
        if head[2 * SIGNA_4X_STUDY_START :][:5] == b"STUDY":
            with open(path, "rb") as fh:
                fh.seek(2 * (SIGNA_4X_SERIES_START + SIGNA_4X_PLANE_NAME))
                plane = fh.read(16)
            if any(p in plane for p in (b"AXIAL", b"CORONAL", b"SAGITTAL", b"OBLIQUE")):
                return "GE Signa 4.x study and series headers"
        return None

    @staticmethod
    def _is_dicom_file(path: Path) -> bool:
        """DICM magic after the preamble, in a plain or gzipped file."""
        with open_dicom(path) as fh:
            return fh.read(132)[128:] == b"DICM"

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
        with open_dicom(path) as fh:
            ds = pydicom.dcmread(fh, stop_before_pixels=True, force=force)
        sop = str(ds.get("SOPClassUID", "") or ds.file_meta.get("MediaStorageSOPClassUID", ""))
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
        if "PHYSIO" in [str(v) for v in (ds.get("ImageType") or [])]:
            fmt, kind, confidence = Format.CMRR_PHYSIO, Kind.PHYSIO, 1.0
            sop_name = f"{sop} with image type PHYSIO"
        elif sop in self.DICOM_SOP_MRS:
            fmt, kind, confidence = Format.DICOM_MRS, Kind.SPECTROSCOPY, 1.0
            sop_name = self.DICOM_SOP_MRS[sop]
        elif sop in self.DICOM_SOP_IMAGE:
            fmt, kind, confidence = Format.DICOM_IMAGE, Kind.IMAGING, 1.0
            sop_name = self.DICOM_SOP_IMAGE[sop]
        elif sop == self.RAW_DATA_STORAGE and self._vendor(manufacturer) == "philips":
            fmt, kind, confidence = Format.PHILIPS_DICOM_RAW, Kind.PROTOCOL, 1.0
            sop_name = "Raw Data Storage (Philips scan parameters)"
        elif sop == self.RAW_DATA_STORAGE and self._vendor(manufacturer) == "siemens":
            fmt, kind, confidence = Format.SIEMENS_DICOM_RAW, Kind.PROTOCOL, 1.0
            sop_name = "Raw Data Storage (Siemens protocol)"
        elif sop.startswith(self.STRUCTURED_REPORT) and self._vendor(manufacturer) == "siemens" \
                and str(ds.get("SeriesDescription", "")) == "PhoenixZIPReport":
            fmt, kind, confidence = Format.SIEMENS_PHOENIX_REPORT, Kind.PROTOCOL, 1.0
            sop_name = "Siemens PhoenixZIPReport (structured report holding the protocol)"
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
            kind = self.bruker_scan_kind(detail.get("method"))
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

    def bruker_scan_kind(self, method: str | None) -> Kind:
        if method is None:
            return Kind.UNKNOWN
        if any(m in method.lower() for m in self.BRUKER_MRS_METHODS):
            return Kind.SPECTROSCOPY
        return Kind.IMAGING

    def _paravision(self, folder: Path) -> bool:
        """A ParaVision scan folder or one of its pdata/<n> folders."""
        folders = (folder, folder.parent.parent) if folder.parent.name == "pdata" else (folder,)
        return any((f / name).is_file() for f in folders for name in self.PARAVISION_FILES)

    @staticmethod
    def _topspin(path: Path, fmt: Format, evidence: str) -> Sniff:
        """A TopSpin experiment (folder of fid or ser with acqus) or processed data set (folder
        pdata/<n> with procs); a file stands for its folder. A folder of experiments is a
        TOPSPIN sniff of the folder with the experiments as companions."""
        folder = path if path.is_dir() else path.parent
        companions = tuple(p for p in sorted(folder.iterdir()) if p.is_file() and p != path)
        return Sniff(
            path,
            fmt,
            Kind.SPECTROSCOPY,
            vendor="bruker",
            confidence=1.0,
            evidence=(evidence,),
            companions=companions,
        )

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
