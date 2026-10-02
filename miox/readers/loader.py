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

from pathlib import Path

# own
from miox.data import MRSData, PriorSet
from miox.readers.base import Reader, ReaderError
from miox.readers.basis_folder import BasisFolderReader
from miox.readers.bruker import BrukerReader
from miox.readers.dicom_mrs import DicomMrsReader
from miox.readers.fida_mat import FidaReader
from miox.readers.fsl_mrs_basis import FslMrsBasisReader
from miox.readers.gava_midas_prior import GavaMidasPriorReader
from miox.readers.ge_pfile import GePfileReader
from miox.readers.inspector_basis import InspectorBasisReader
from miox.readers.jmrui import JmruiReader
from miox.readers.lcmodel_basis import LcmodelBasisReader
from miox.readers.lcmodel_raw import LcmodelRawReader
from miox.readers.marss_basis import MarssBasisReader
from miox.readers.nifti_mrs import NiftiMrsReader
from miox.readers.osprey_basis import OspreyBasisReader
from miox.readers.philips_data_list import PhilipsDataListReader
from miox.readers.philips_spar_sdat import PhilipsSparSdatReader
from miox.readers.profit_basis import ProfitBasisReader
from miox.readers.pyamares_prior import PyamaresPriorReader
from miox.readers.siemens_rda import SiemensRdaReader
from miox.readers.siemens_twix import SiemensTwixReader
from miox.readers.sivic_ddf import SivicDdfReader
from miox.readers.spinwizard_basis import SpinwizardBasisReader
from miox.readers.tarquin_dpt import TarquinDptReader
from miox.readers.varian import VarianReader
from miox.readers.vasf import VasfReader
from miox.readers.vespa_prior import VespaPriorReader
from miox.readers.vespa_viff import VespaViffReader
from miox.sniff import Format, Sniffer


#**************************************************************************************************#
#                                              Loader                                              #
#**************************************************************************************************#
#                                                                                                  #
# Holds one instance of every reader, keyed by the formats it handles.                             #
#                                                                                                  #
#**************************************************************************************************#
class Loader:
    def __init__(self):
        self.sniffer = Sniffer()
        self.readers: dict[Format, Reader] = {}
        for reader in (
            BasisFolderReader(self),
            BrukerReader(),
            DicomMrsReader(),
            FidaReader(),
            FslMrsBasisReader(),
            GavaMidasPriorReader(),
            GePfileReader(),
            InspectorBasisReader(),
            JmruiReader(),
            LcmodelBasisReader(),
            LcmodelRawReader(),
            MarssBasisReader(),
            NiftiMrsReader(),
            OspreyBasisReader(),
            PhilipsDataListReader(),
            PhilipsSparSdatReader(),
            ProfitBasisReader(),
            PyamaresPriorReader(),
            SiemensRdaReader(),
            SiemensTwixReader(),
            SivicDdfReader(),
            SpinwizardBasisReader(),
            TarquinDptReader(),
            VarianReader(),
            VasfReader(),
            VespaPriorReader(),
            VespaViffReader(),
        ):
            for fmt in reader.FORMATS:
                self.readers[fmt] = reader

    def load(self, path: str | Path) -> list[MRSData | PriorSet]:
        """Every dataset in ``path``: one MRSData (or PriorSet) each."""
        sniff = self.sniffer.sniff(path)
        reader = self.readers.get(sniff.format)
        if reader is None:
            raise ReaderError(f"no reader for {sniff.format}: {sniff.path}")
        return reader.read(sniff)


#**************************************************************************************************#
#                                               load                                               #
#**************************************************************************************************#
#                                                                                                  #
# Load one path with a fresh Loader.                                                               #
#                                                                                                  #
#**************************************************************************************************#
def load(path: str | Path) -> list[MRSData | PriorSet]:
    return Loader().load(path)
