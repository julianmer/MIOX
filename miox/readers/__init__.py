####################################################################################################
#                                           __init__.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Readers: one class per file format, and the loader that picks the right one.            #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#

# own
from miox.readers.base import Reader, ReaderError, Spec2niiReader
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
from miox.readers.lcmodel_output import LcmodelOutputReader
from miox.readers.lcmodel_raw import LcmodelRawReader
from miox.readers.loader import Loader, load
from miox.readers.marss_basis import MarssBasisReader
from miox.readers.nifti_mrs import NiftiMrsReader
from miox.readers.osprey_basis import OspreyBasisReader
from miox.readers.osprey_container import OspreyContainerReader
from miox.readers.philips_data_list import PhilipsDataListReader
from miox.readers.philips_spar_sdat import PhilipsSparSdatReader
from miox.readers.profit_basis import ProfitBasisReader
from miox.readers.pyamares_prior import PyamaresPriorReader
from miox.readers.pyamares_result import PyamaresResultReader
from miox.readers.siemens_rda import SiemensRdaReader
from miox.readers.siemens_twix import SiemensTwixReader
from miox.readers.sivic_ddf import SivicDdfReader
from miox.readers.spinwizard_basis import SpinwizardBasisReader
from miox.readers.tarquin_dpt import TarquinDptReader
from miox.readers.tarquin_output import TarquinOutputReader
from miox.readers.varian import VarianReader
from miox.readers.vasf import VasfReader
from miox.readers.vespa_prior import VespaPriorReader
from miox.readers.vespa_viff import VespaViffReader

__all__ = [
    "BasisFolderReader",
    "BrukerReader",
    "DicomMrsReader",
    "FidaReader",
    "FslMrsBasisReader",
    "GavaMidasPriorReader",
    "GePfileReader",
    "InspectorBasisReader",
    "JmruiReader",
    "LcmodelBasisReader",
    "LcmodelOutputReader",
    "LcmodelRawReader",
    "Loader",
    "MarssBasisReader",
    "NiftiMrsReader",
    "OspreyBasisReader",
    "OspreyContainerReader",
    "PhilipsDataListReader",
    "PhilipsSparSdatReader",
    "ProfitBasisReader",
    "PyamaresPriorReader",
    "PyamaresResultReader",
    "Reader",
    "ReaderError",
    "SiemensRdaReader",
    "SiemensTwixReader",
    "SivicDdfReader",
    "Spec2niiReader",
    "SpinwizardBasisReader",
    "TarquinDptReader",
    "TarquinOutputReader",
    "VarianReader",
    "VasfReader",
    "VespaPriorReader",
    "VespaViffReader",
    "load",
]
