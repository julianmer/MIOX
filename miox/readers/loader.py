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

import dataclasses
from collections import defaultdict
from pathlib import Path
from typing import Any

# own
from miox.data import (
    FitResult,
    MRImage,
    MRSData,
    NoiseCovariance,
    PriorSet,
    PulseSequence,
    RawData,
)
from miox.readers.bart_cfl import BartReader
from miox.readers.base import IncompleteError, Reader, ReaderError
from miox.readers.basis_folder import BasisFolderReader
from miox.readers.bruker import BrukerReader
from miox.readers.dicom_image import DicomImageReader
from miox.readers.dicom_mrs import DicomMrsReader
from miox.readers.fastmri import FastMriReader
from miox.readers.fida_mat import FidaReader
from miox.readers.fsl_mrs_basis import FslMrsBasisReader
from miox.readers.gava_midas_prior import GavaMidasPriorReader
from miox.readers.ge_pfile import GePfileReader
from miox.readers.ge_scanarchive import GeScanArchiveReader
from miox.readers.ge_signa import GeSignaReader
from miox.readers.inspector_basis import InspectorBasisReader
from miox.readers.ismrmrd import IsmrmrdReader
from miox.readers.jmrui import JmruiReader
from miox.readers.lcmodel_basis import LcmodelBasisReader
from miox.readers.lcmodel_output import LcmodelOutputReader
from miox.readers.lcmodel_raw import LcmodelRawReader
from miox.readers.marss_basis import MarssBasisReader
from miox.readers.metaimage import MetaImageReader
from miox.readers.mrd import MrdReader
from miox.readers.mrtrix import MrtrixReader
from miox.readers.nibabel_image import NibabelImageReader
from miox.readers.nifti_mrs import NiftiMrsReader
from miox.readers.nifti_sidecar_mrs import NiftiSidecarMrsReader
from miox.readers.nrrd import NrrdReader
from miox.readers.osprey_basis import OspreyBasisReader
from miox.readers.osprey_container import OspreyContainerReader
from miox.readers.philips_data_list import PhilipsDataListReader
from miox.readers.philips_raw_lab import PhilipsRawLabReader
from miox.readers.philips_spar_sdat import PhilipsSparSdatReader
from miox.readers.profit_basis import ProfitBasisReader
from miox.readers.pulseq import PulseqReader
from miox.readers.pyamares_prior import PyamaresPriorReader
from miox.readers.pyamares_result import PyamaresResultReader
from miox.readers.siemens_rda import SiemensRdaReader
from miox.readers.siemens_twix import SiemensTwixReader
from miox.readers.sivic_ddf import SivicDdfReader
from miox.readers.spinwizard_basis import SpinwizardBasisReader
from miox.readers.tarquin_dpt import TarquinDptReader
from miox.readers.tarquin_output import TarquinOutputReader
from miox.readers.text_fid import TextFidReader
from miox.readers.varian import VarianReader
from miox.readers.varian_fdf import VarianFdfReader
from miox.readers.vasf import VasfReader
from miox.readers.vespa_prior import VespaPriorReader
from miox.readers.vespa_viff import VespaViffReader
from miox.sniff import Format, Sniff, Sniffer

Dataset = MRSData | MRImage | RawData | PriorSet | FitResult | PulseSequence | NoiseCovariance


#**************************************************************************************************#
#                                              Loader                                              #
#**************************************************************************************************#
#                                                                                                  #
# Holds one instance of every reader, keyed by the formats it handles. ``load`` returns what a     #
# file states; ``given`` fills standard keys it lacks and ``complete`` demands the keys NIfTI-MRS  #
# requires. A DICOM folder (one series or a whole study) is split by SOP class first, so its       #
# image series and its spectroscopy series each reach their own reader.                            #
#                                                                                                  #
#**************************************************************************************************#
class Loader:
    DICOM_FORMATS = (Format.DICOM_IMAGE, Format.DICOM_MRS)

    def __init__(self):
        self.sniffer = Sniffer()
        self.readers: dict[Format, Reader] = {}
        for reader in (
            BartReader(),
            BasisFolderReader(self),
            BrukerReader(),
            DicomImageReader(),
            DicomMrsReader(),
            FastMriReader(),
            FidaReader(),
            FslMrsBasisReader(),
            GavaMidasPriorReader(),
            GePfileReader(),
            GeScanArchiveReader(),
            GeSignaReader(),
            InspectorBasisReader(),
            IsmrmrdReader(),
            JmruiReader(),
            LcmodelBasisReader(),
            LcmodelOutputReader(),
            LcmodelRawReader(),
            MarssBasisReader(),
            MetaImageReader(),
            MrdReader(),
            MrtrixReader(),
            NibabelImageReader(),
            NrrdReader(),
            NiftiMrsReader(),
            NiftiSidecarMrsReader(),
            OspreyBasisReader(),
            OspreyContainerReader(),
            PhilipsDataListReader(),
            PhilipsRawLabReader(),
            PhilipsSparSdatReader(),
            ProfitBasisReader(),
            PulseqReader(),
            PyamaresPriorReader(),
            PyamaresResultReader(),
            SiemensRdaReader(),
            SiemensTwixReader(),
            SivicDdfReader(),
            SpinwizardBasisReader(),
            TarquinDptReader(),
            TarquinOutputReader(),
            TextFidReader(),
            VarianReader(),
            VarianFdfReader(),
            VasfReader(),
            VespaPriorReader(),
            VespaViffReader(),
        ):
            for fmt in reader.FORMATS:
                self.readers[fmt] = reader

    def load(
        self,
        path: str | Path,
        *,
        given: dict[str, Any] | None = None,
        complete: bool = False,
    ) -> list[Dataset]:
        """Every dataset in ``path``: one MRSData, MRImage, RawData, PriorSet, FitResult,
        PulseSequence or NoiseCovariance each.

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
        if sniff.path.is_dir() and sniff.format in self.DICOM_FORMATS:
            return self._dicom_folder(sniff)
        reader = self.readers.get(sniff.format)
        if reader is None:
            raise ReaderError(f"no reader for {sniff.format}: {sniff.path}")
        return reader.read(sniff)

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
