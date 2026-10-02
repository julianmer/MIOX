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
from miox.sniff import Format, Kind, Sniff, Sniffer, sniff

__version__ = "0.0.0"
__all__ = ["Format", "Kind", "Sniff", "Sniffer", "__version__", "sniff"]
