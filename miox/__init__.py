####################################################################################################
#                                           __init__.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 01/10/26                                                                                #
#                                                                                                  #
# Purpose: Package entry: the version and the top-level API.                                       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#

# own
from miox.data import (
    BasisSet,
    FitResult,
    MRImage,
    MRSData,
    NoiseCovariance,
    PhysioData,
    PriorSet,
    Protocol,
    Provenance,
    PulseSequence,
    RawData,
)
from miox.readers import IncompleteError, Loader, ReaderError, load
from miox.sniff import Format, Kind, Sniff, Sniffer, sniff

__version__ = "0.0.0"
__all__ = [
    "BasisSet",
    "FitResult",
    "Format",
    "IncompleteError",
    "Kind",
    "Loader",
    "MRImage",
    "MRSData",
    "NoiseCovariance",
    "PhysioData",
    "PriorSet",
    "Protocol",
    "Provenance",
    "PulseSequence",
    "RawData",
    "ReaderError",
    "Sniff",
    "Sniffer",
    "__version__",
    "load",
    "sniff",
]
