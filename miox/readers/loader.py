####################################################################################################
#                                            loader.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Loads any supported file: sniffs it and hands it to the reader of its format.           #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import atexit
import dataclasses
import importlib
import shutil
import tempfile
import zipfile
from collections import defaultdict
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

# own
from miox.data import (
    FitResult,
    MRImage,
    MRSData,
    NoiseCovariance,
    Phantom,
    PhysioData,
    PriorSet,
    Protocol,
    PulseSequence,
    RawData,
    RFPulse,
    Surface,
    Tractogram,
    Transform,
)
from miox.readers.base import IncompleteError, Reader, ReaderError
from miox.sniff import Format, Sniff, Sniffer

Dataset = (
    MRSData | MRImage | RawData | PriorSet | FitResult | PulseSequence | NoiseCovariance
    | PhysioData | Protocol | Phantom | Tractogram | Surface | Transform | RFPulse
)


READERS = (  # (module, class, formats it reads): imported when a file of its format is read
    ("miox.readers.bart_cfl", "BartReader", (Format.BART_CFL,)),
    ("miox.readers.basis_folder", "BasisFolderReader", (Format.BASIS_FOLDER,)),
    ("miox.readers.biopac_acq", "AcqKnowledgeReader", (Format.BIOPAC_ACQ,)),
    ("miox.readers.brainvoyager", "BrainVoyagerSurfaceReader", (
        Format.BRAINVOYAGER_SRF, Format.BRAINVOYAGER_SMP, Format.BRAINVOYAGER_MTC,
        Format.BRAINVOYAGER_SSM,
    )),
    ("miox.readers.brainvoyager", "BrainVoyagerVolumeReader", (
        Format.BRAINVOYAGER_VMR, Format.BRAINVOYAGER_VTC, Format.BRAINVOYAGER_GLM,
        Format.BRAINVOYAGER_VMP, Format.BRAINVOYAGER_V16,
    )),
    ("miox.readers.brainvoyager_text", "BrainVoyagerFmrReader", (Format.BRAINVOYAGER_FMR,)),
    ("miox.readers.brainvoyager_text", "BrainVoyagerProtocolReader", (
        Format.BRAINVOYAGER_PRT, Format.BRAINVOYAGER_SDM,
    )),
    ("miox.readers.brainvoyager_text", "BrainVoyagerRegionReader", (
        Format.BRAINVOYAGER_VOI, Format.BRAINVOYAGER_POI,
    )),
    ("miox.readers.brainvoyager_text", "BrainVoyagerTransformReader", (Format.BRAINVOYAGER_TRF,)),
    ("miox.readers.bruker", "BrukerReader", (Format.BRUKER,)),
    ("miox.readers.dicom_image", "DicomImageReader", (Format.DICOM_IMAGE,)),
    ("miox.readers.dicom_mrs", "DicomMrsReader", (Format.DICOM_MRS,)),
    ("miox.readers.dsi_studio", "DsiStudioImageReader", (Format.DSI_STUDIO_SRC,
                                                          Format.DSI_STUDIO_FIB)),
    ("miox.readers.fastmri", "FastMriReader", (Format.FASTMRI_H5,)),
    ("miox.readers.fida_mat", "FidaReader", (Format.FIDA_MAT,)),
    ("miox.readers.gipl", "GiplReader", (Format.GIPL,)),
    ("miox.readers.fsl_mrs_dynamic", "FslMrsDynamicReader", (Format.FSL_MRS_DYNAMIC,)),
    ("miox.readers.fsl_vest", "FslVestReader", (Format.FSL_VEST,)),
    ("miox.readers.gadgetron_array", "GadgetronArrayReader", (Format.GADGETRON_ARRAY,)),
    ("miox.readers.fsl_mrs_basis", "FslMrsBasisReader", (Format.FSL_MRS_BASIS,)),
    ("miox.readers.gava_midas_prior", "GavaMidasPriorReader", (
        Format.GAVA_PRIOR, Format.MIDAS_PRIOR,
    )),
    ("miox.readers.ge_pfile", "GePfileReader", (Format.GE_PFILE,)),
    ("miox.readers.ge_scanarchive", "GeScanArchiveReader", (Format.GE_SCANARCHIVE,)),
    ("miox.readers.ge_signa", "GeSignaReader", (Format.GE_SIGNA,)),
    ("miox.readers.inspector_basis", "InspectorBasisReader", (Format.INSPECTOR_BASIS,)),
    ("miox.readers.ismrmrd", "IsmrmrdReader", (Format.ISMRMRD,)),
    ("miox.readers.jcamp", "JcampReader", (Format.JCAMP_NMR,)),
    ("miox.readers.jeol", "JeolReader", (Format.JEOL_JDF,)),
    ("miox.readers.jmrui", "JmruiReader", (Format.JMRUI_TEXT, Format.JMRUI_BINARY)),
    ("miox.readers.lcmodel_basis", "LcmodelBasisReader", (Format.LCMODEL_BASIS,)),
    ("miox.readers.lcmodel_control", "LcmodelControlReader", (Format.LCMODEL_CONTROL,)),
    ("miox.readers.lcmodel_output", "LcmodelOutputReader", (
        Format.LCMODEL_COORD, Format.LCMODEL_TABLE, Format.LCMODEL_CSV, Format.LCMODEL_PS,
    )),
    ("miox.readers.lcmodel_raw", "LcmodelRawReader", (Format.LCMODEL_RAW,)),
    ("miox.readers.marss_basis", "MarssBasisReader", (Format.MARSS_BASIS,)),
    ("miox.readers.metaimage", "MetaImageReader", (Format.METAIMAGE,)),
    ("miox.readers.jnifti", "JNiftiReader", (Format.JNIFTI, Format.JNIFTI_BINARY)),
    ("miox.readers.vista", "VistaReader", (Format.VISTA,)),
    ("miox.readers.nifti_zarr", "NiftiZarrReader", (Format.NIFTI_ZARR,)),
    ("miox.readers.stimulate", "StimulateReader", (Format.STIMULATE,)),
    ("miox.readers.stimulate", "SivicIdfReader", (Format.SIVIC_IDF,)),
    ("miox.readers.vtk_legacy", "VtkImageReader", (Format.VTK_IMAGE,)),
    ("miox.readers.vtk_xml", "VtkXmlImageReader", (Format.VTK_IMAGE_XML,)),
    ("miox.readers.mrd", "MrdReader", (Format.MRD,)),
    ("miox.readers.mrtrix", "MrtrixReader", (Format.MRTRIX,)),
    ("miox.readers.nibabel_image", "NibabelImageReader", (
        Format.NIFTI, Format.ANALYZE, Format.MGH, Format.MINC1, Format.MINC2, Format.PARREC,
        Format.AFNI,
    )),
    ("miox.readers.nifti_mrs", "NiftiMrsReader", (Format.NIFTI_MRS,)),
    ("miox.readers.nifti_sidecar_mrs", "NiftiSidecarMrsReader", (
        Format.NIFTI_SIDECAR_MRS, Format.FSL_MRS_NIFTI,
    )),
    ("miox.readers.nmrml", "NmrmlReader", (Format.NMRML,)),
    ("miox.readers.nmrpipe", "NmrPipeReader", (Format.NMRPIPE,)),
    ("miox.readers.nrrd", "NrrdReader", (Format.NRRD,)),
    ("miox.readers.osprey_basis", "OspreyBasisReader", (Format.OSPREY_BASIS,)),
    ("miox.readers.osprey_container", "OspreyContainerReader", (Format.OSPREY_MRSCONT,)),
    ("miox.readers.philips_data_list", "PhilipsDataListReader", (Format.PHILIPS_DATA_LIST,)),
    ("miox.readers.philips_examcard", "PhilipsExamCardReader", (Format.PHILIPS_EXAMCARD,)),
    ("miox.readers.philips_dicom_raw", "PhilipsDicomRawReader", (Format.PHILIPS_DICOM_RAW,)),
    ("miox.readers.csdm", "CsdmReader", (Format.CSDM,)),
    ("miox.readers.dmiwizard_map", "DMIWizardMapReader", (Format.DMIWIZARD_MAP,)),
    ("miox.readers.spant_rds", "SpantRdsReader", (Format.SPANT_RDS,)),
    ("miox.readers.siemens_dicom_protocol", "SiemensDicomProtocolReader",
     (Format.SIEMENS_DICOM_RAW, Format.SIEMENS_PHOENIX_REPORT)),
    ("miox.readers.philips_raw_lab", "PhilipsRawLabReader", (Format.PHILIPS_RAW_LAB,)),
    ("miox.readers.philips_spar_sdat", "PhilipsSparSdatReader", (Format.PHILIPS_SPAR_SDAT,)),
    ("miox.readers.physio", "BidsPhysioReader", (Format.BIDS_PHYSIO,)),
    ("miox.readers.physio", "CmrrPhysioReader", (Format.CMRR_PHYSIO,)),
    ("miox.readers.physio", "GePhysioReader", (Format.GE_PHYSIO,)),
    ("miox.readers.physio", "HcpPhysioReader", (Format.HCP_PHYSIO,)),
    ("miox.readers.physio", "PhilipsPhyslogReader", (Format.PHILIPS_PHYSLOG,)),
    ("miox.readers.physio", "SiemensPmuReader", (Format.SIEMENS_PMU,)),
    ("miox.readers.profit_basis", "ProfitBasisReader", (Format.PROFIT_BASIS,)),
    ("miox.readers.mrs_sim_basis", "MrsSimBasisReader", (Format.MRS_SIM_BASIS,)),
    ("miox.readers.pulseq", "PulseqReader", (Format.PULSEQ,)),
    ("miox.readers.pyamares_prior", "PyamaresPriorReader", (Format.PYAMARES_PRIOR,)),
    ("miox.readers.pyamares_result", "PyamaresResultReader", (Format.PYAMARES_RESULT,)),
    ("miox.readers.rf_pulse", "BrukerShapeReader", (Format.BRUKER_SHAPE,)),
    ("miox.readers.rf_pulse", "FslMrsSimReader", (Format.FSL_MRS_SIM,)),
    ("miox.readers.rf_pulse", "SiemensPtaReader", (Format.SIEMENS_PTA,)),
    ("miox.readers.rnmrtk", "RnmrtkReader", (Format.RNMRTK,)),
    ("miox.readers.rs2d", "Rs2dReader", (Format.RS2D,)),
    ("miox.readers.siemens_protocol", "SiemensPrintedProtocolReader", (Format.SIEMENS_PRINTPROT,)),
    ("miox.readers.siemens_protocol", "SiemensProtocolReader", (
        Format.SIEMENS_PRO, Format.SIEMENS_EXAR,
    )),
    ("miox.readers.siemens_rda", "SiemensRdaReader", (Format.SIEMENS_RDA,)),
    ("miox.readers.siemens_twix", "SiemensTwixReader", (Format.SIEMENS_TWIX,)),
    ("miox.readers.simpson", "SimpsonReader", (Format.SIMPSON,)),
    ("miox.readers.simulation", "JemrisMotionReader", (Format.JEMRIS_MOTION,)),
    ("miox.readers.simulation", "JemrisPhantomReader", (Format.JEMRIS_PHANTOM,)),
    ("miox.readers.simulation", "JemrisSignalReader", (Format.JEMRIS_SIGNAL,)),
    ("miox.readers.simulation", "KomaPhantomReader", (Format.KOMA_PHANTOM,)),
    ("miox.readers.simulation", "MrilabPhantomReader", (Format.MRILAB_PHANTOM,)),
    ("miox.readers.sivic_ddf", "SivicDdfReader", (Format.SIVIC_DDF,)),
    ("miox.readers.sparky", "SparkyReader", (Format.SPARKY,)),
    ("miox.readers.spinsolve", "SpinsolveReader", (Format.SPINSOLVE,)),
    ("miox.readers.spinwizard_basis", "SpinwizardBasisReader", (Format.SPINWIZARD_BASIS,)),
    ("miox.readers.spm_mat", "SpmMatReader", (Format.SPM_MAT,)),
    ("miox.readers.surfaces", "FreeSurferReader", (
        Format.FREESURFER_SURFACE, Format.FREESURFER_MORPH, Format.FREESURFER_ANNOT,
        Format.FREESURFER_LABEL,
    )),
    ("miox.readers.surfaces", "GiftiReader", (Format.GIFTI,)),
    ("miox.readers.surfaces", "Mz3Reader", (Format.MZ3,)),
    ("miox.readers.tarquin_dpt", "TarquinDptReader", (Format.TARQUIN_DPT,)),
    ("miox.readers.tarquin_output", "TarquinOutputReader", (
        Format.TARQUIN_RESULTS, Format.TARQUIN_FIT, Format.TARQUIN_CSV,
    )),
    ("miox.readers.tecmag", "TecmagReader", (Format.TECMAG_TNT,)),
    ("miox.readers.text_fid", "TextFidReader", (Format.TEXT_FID,)),
    ("miox.readers.toppe", "ToppeReader", (
        Format.TOPPE_ENTRY, Format.TOPPE_MODULES, Format.TOPPE_LOOP, Format.TOPPE_MOD,
    )),
    ("miox.readers.topspin", "TopSpinReader", (Format.TOPSPIN, Format.TOPSPIN_PROCESSED)),
    ("miox.readers.tractograms", "TrackReader", (Format.TRK, Format.TCK)),
    ("miox.readers.tractograms", "TrxReader", (Format.TRX,)),
    ("miox.readers.tractograms", "DsiStudioTrackReader", (Format.DSI_STUDIO_TT,)),
    ("miox.readers.tractograms", "TsfReader", (Format.TSF,)),
    ("miox.readers.transforms", "AfniReader", (Format.TRANSFORM_AFNI,)),
    ("miox.readers.transforms", "FslReader", (Format.TRANSFORM_FSL,)),
    ("miox.readers.transforms", "ItkReader", (Format.TRANSFORM_ITK,)),
    ("miox.readers.transforms", "LtaReader", (Format.TRANSFORM_LTA,)),
    ("miox.readers.varian", "VarianReader", (Format.VARIAN,)),
    ("miox.readers.varian_fdf", "VarianFdfReader", (Format.VARIAN_FDF,)),
    ("miox.readers.vasf", "VasfReader", (Format.VASF,)),
    ("miox.readers.vespa_prior", "VespaPriorReader", (Format.VESPA_PRIOR,)),
    ("miox.readers.vespa_viff", "VespaViffReader", (Format.VESPA_VIFF,)),
    ("miox.readers.vtk_legacy", "VtkReader", (Format.VTK_POLYDATA,)),
    ("miox.readers.yarra", "YarraReader", (Format.YARRA_MODE, Format.YARRA_MODULE)),
)


#**************************************************************************************************#
#                                             Readers                                              #
#**************************************************************************************************#
#                                                                                                  #
# The reader of each format, as READERS lists them: a reader's module is imported and the reader   #
# made the first time a file of one of its formats is read, so loading a file imports only what    #
# that file needs (the readers' libraries take seconds to import all together).                    #
#                                                                                                  #
#**************************************************************************************************#
class Readers(Mapping):
    def __init__(self, loader: Loader):
        self.loader = loader
        self.table = {fmt: (module, name) for module, name, formats in READERS for fmt in formats}
        self.made: dict[tuple[str, str], Reader] = {}

    def __getitem__(self, fmt: Format) -> Reader:
        module, name = self.table[fmt]
        if (module, name) not in self.made:
            cls = getattr(importlib.import_module(module), name)
            self.made[module, name] = cls(self.loader) if name == "BasisFolderReader" else cls()
        return self.made[module, name]

    def __contains__(self, fmt: object) -> bool:
        return fmt in self.table

    def __iter__(self) -> Iterator[Format]:
        return iter(self.table)

    def __len__(self) -> int:
        return len(self.table)

#**************************************************************************************************#
#                                              Loader                                              #
#**************************************************************************************************#
#                                                                                                  #
# Holds one instance of every reader, keyed by the formats it handles. ``load`` returns what a     #
# file states; ``given`` fills standard keys it lacks and ``complete`` demands the keys NIfTI-MRS  #
# requires. A DICOM folder (one series or a whole study) is split by SOP class first, so its       #
# image series and its spectroscopy series each reach their own reader. A zip archive is unpacked  #
# into a folder named after it in a temporary folder (removed when the interpreter exits, as some  #
# readers map their files) and read like a folder whose format is unknown: every entry of a known  #
# format is loaded, other folders are searched further; provenance paths name the archive and the  #
# member.                                                                                          #
#                                                                                                  #
#**************************************************************************************************#
class Loader:
    DICOM_FORMATS = (Format.DICOM_IMAGE, Format.DICOM_MRS, Format.CMRR_PHYSIO)

    def __init__(self):
        self.sniffer = Sniffer()
        self.readers = Readers(self)

    def load(
        self,
        path: str | Path,
        *,
        given: dict[str, Any] | None = None,
        complete: bool = False,
    ) -> list[Dataset]:
        """Every dataset in ``path``: one MRSData, MRImage, RawData, PriorSet, FitResult,
        PulseSequence, NoiseCovariance, PhysioData, Protocol, Phantom, Tractogram, Surface,
        Transform or RFPulse each.

        ``given`` maps standard keys to values for what the file does not state
        (SpectrometerFrequency, ResonantNucleus, DwellTime or SpectralWidth, ...); a stated
        value is never overridden. With ``complete``, a spectroscopy dataset that still lacks a
        key NIfTI-MRS requires raises IncompleteError instead of being returned as read.
        """
        sniff = self.sniffer.sniff(path)
        datasets = self._read(sniff)
        for d in datasets:
            if not isinstance(d, MRSData):
                continue
            if given:
                d.supply(given)
            if complete and d.missing:
                name = f" ({d.name})" if d.name else ""
                raise IncompleteError(
                    f"{sniff.path.name}{name} states no {', '.join(d.missing)}; pass them in given"
                )
        return datasets

    def _read(self, sniff: Sniff) -> list[Dataset]:
        if sniff.format == Format.ZIP_ARCHIVE:
            return self._archive(sniff)
        if sniff.path.is_dir() and sniff.format in self.DICOM_FORMATS:
            return self._dicom_folder(sniff)
        reader = self.readers.get(sniff.format)
        if reader is None:
            raise ReaderError(f"no reader for {sniff.format}: {sniff.path}")
        return reader.read(sniff)

    def _archive(self, sniff: Sniff) -> list[Dataset]:
        """Every dataset in a zip archive. Members that would land outside the unpacked folder
        are refused; members that fail to read are noted on the other datasets, and the error is
        raised only when nothing could be read."""
        temporary = Path(tempfile.mkdtemp(prefix="miox-zip-")).resolve()
        atexit.register(shutil.rmtree, temporary, ignore_errors=True)
        folder = temporary / sniff.path.stem  # names taken from folders match an unpacked copy
        try:
            with zipfile.ZipFile(sniff.path) as archive:
                for member in archive.namelist():
                    if not (folder / member).resolve().is_relative_to(folder):
                        raise ReaderError(f"{sniff.path}: member {member!r} lies outside the "
                                          "archive")
                archive.extractall(folder)
        except zipfile.BadZipFile as e:
            raise ReaderError(f"{sniff.path}: not a readable zip archive ({e})") from e
        failed: list[str] = []
        datasets = self._entries(folder, folder, failed)
        if not datasets:
            raise ReaderError("; ".join(failed) or f"{sniff.path.name}: nothing readable inside")
        member = lambda p: sniff.path / p.relative_to(folder) if p.is_relative_to(folder) else p  # noqa: E731
        for d in datasets:
            d.provenance = dataclasses.replace(
                d.provenance,
                paths=tuple(member(Path(p)) for p in d.provenance.paths),
                notes=d.provenance.notes + (f"read from the zip archive {sniff.path.name}",)
                + tuple(failed),
            )
        return datasets

    def _entries(self, path: Path, folder: Path, failed: list[str]) -> list[Dataset]:
        """The datasets of ``path`` if its format is known, else of the entries of a folder."""
        sniff = self.sniffer.sniff(path)
        if sniff.format != Format.UNKNOWN:
            if sniff.format not in self.readers and sniff.format != Format.ZIP_ARCHIVE:
                return []
            try:
                return self._read(sniff)
            except ReaderError as e:
                failed.append(f"{path.relative_to(folder)} not read: {e}")
                return []
        if not path.is_dir():
            return []
        datasets = []
        for child in sorted(path.iterdir()):
            if not child.name.startswith((".", "__MACOSX")):
                datasets += self._entries(child, folder, failed)
        return datasets

    def _dicom_folder(self, sniff: Sniff) -> list[MRSData | MRImage]:
        """Every series of a DICOM folder: the files of each format go to its reader together,
        which splits them into series. Files of other SOP classes (presentation states, reports,
        DICOMDIR) are passed over. A format whose reader fails is noted on the datasets of the
        others; the error is raised only when nothing could be read."""
        groups: dict[Format, list[tuple[Path, Sniff]]] = defaultdict(list)
        for path in sniff.companions:
            if path.is_file():
                file = self.sniffer.sniff(path)
                if file.format in self.DICOM_FORMATS:
                    groups[file.format].append((path, file))
        datasets, failed = [], []
        for fmt, files in groups.items():
            part = dataclasses.replace(
                files[0][1],
                path=sniff.path,
                evidence=(f"{fmt} files of {sniff.path.name}",),
                companions=tuple(path for path, _ in files),
            )
            try:
                datasets += self.readers[fmt].read(part)
            except ReaderError as e:
                failed.append(f"{fmt} series not read: {e}")
        if not datasets:
            raise ReaderError("; ".join(failed) or f"{sniff.path.name}: no DICOM data files")
        for d in datasets:
            d.provenance = dataclasses.replace(
                d.provenance, notes=d.provenance.notes + tuple(failed)
            )
        return datasets


#**************************************************************************************************#
#                                               load                                               #
#**************************************************************************************************#
#                                                                                                  #
# Load one path with a fresh Loader.                                                               #
#                                                                                                  #
#**************************************************************************************************#
def load(
    path: str | Path, *, given: dict[str, Any] | None = None, complete: bool = False
) -> list[Dataset]:
    return Loader().load(path, given=given, complete=complete)
