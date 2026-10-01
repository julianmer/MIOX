####################################################################################################
#                                           __init__.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 01/10/26                                                                                #
#                                                                                                  #
# Purpose: Package entry: the version and the top-level API (sniff for now; load, view, convert    #
#          follow with the readers).                                                               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#

# own
from miox.sniff import Sniff, sniff

__version__ = "0.0.0"
__all__ = ["Sniff", "__version__", "sniff"]
