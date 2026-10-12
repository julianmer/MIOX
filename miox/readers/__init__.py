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

import importlib

# own
from miox.readers.base import IncompleteError, Reader, ReaderError, Spec2niiReader
from miox.readers.loader import READERS, Loader, load

__all__ = [
    "BasisFolderReader",
    "BrukerReader",
    "DicomImageReader",
    "DicomMrsReader",
    "FidaReader",
    "FslMrsBasisReader",
    "GavaMidasPriorReader",
    "GePfileReader",
    "GeSignaReader",
    "IncompleteError",
    "InspectorBasisReader",
    "JNiftiReader",
    "JmruiReader",
    "LcmodelBasisReader",
    "LcmodelOutputReader",
    "LcmodelRawReader",
    "Loader",
    "MarssBasisReader",
    "MetaImageReader",
    "MrtrixReader",
    "NibabelImageReader",
    "NrrdReader",
    "NiftiMrsReader",
    "NiftiSidecarMrsReader",
    "NiftiZarrReader",
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
    "SivicIdfReader",
    "Spec2niiReader",
    "SpinwizardBasisReader",
    "StimulateReader",
    "TarquinDptReader",
    "TarquinOutputReader",
    "TextFidReader",
    "VarianReader",
    "VarianFdfReader",
    "VasfReader",
    "VespaPriorReader",
    "VespaViffReader",
    "VistaReader",
    "load",
]


def __getattr__(name: str):
    """Reader classes are imported on first use (their libraries are slow to import)."""
    for module, cls, _ in READERS:
        if cls == name:
            return getattr(importlib.import_module(module), name)
    raise AttributeError(f"module 'miox.readers' has no attribute {name!r}")
