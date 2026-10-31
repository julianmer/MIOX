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
    Phantom,
    PhysioData,
    PriorSet,
    Protocol,
    Provenance,
    PulseSequence,
    RawData,
    RFPulse,
    Surface,
    Tractogram,
    Transform,
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
    "Phantom",
    "PhysioData",
    "PriorSet",
    "Protocol",
    "Provenance",
    "PulseSequence",
    "RawData",
    "ReaderError",
    "RFPulse",
    "Sniff",
    "Sniffer",
    "Surface",
    "Tractogram",
    "Transform",
    "__version__",
    "load",
    "sniff",
    "view",
]


def __getattr__(name: str):
    if name == "view":  # on first use: matplotlib adds about 0.3 s to every import
        from miox.viewer import view

        return view
    raise AttributeError(f"module 'miox' has no attribute {name!r}")
