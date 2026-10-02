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
from miox.readers.bruker import BrukerReader
from miox.readers.dicom_mrs import DicomMrsReader
from miox.readers.ge_pfile import GePfileReader
from miox.readers.jmrui import JmruiReader
from miox.readers.lcmodel_raw import LcmodelRawReader
from miox.readers.loader import Loader, load
from miox.readers.nifti_mrs import NiftiMrsReader
from miox.readers.philips_spar_sdat import PhilipsSparSdatReader
from miox.readers.siemens_rda import SiemensRdaReader
from miox.readers.siemens_twix import SiemensTwixReader

__all__ = [
    "BrukerReader",
    "DicomMrsReader",
    "GePfileReader",
    "JmruiReader",
    "LcmodelRawReader",
    "Loader",
    "NiftiMrsReader",
    "PhilipsSparSdatReader",
    "Reader",
    "ReaderError",
    "SiemensRdaReader",
    "SiemensTwixReader",
    "Spec2niiReader",
    "load",
]
