####################################################################################################
#                                           __init__.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 01/10/26                                                                                #
#                                                                                                  #
# Purpose: Metadata layer: the standard key table. Extraction and derivation rules follow with the #
#          readers.                                                                                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#

# own
from miox.meta.keys import KEYS, Key

__all__ = ["KEYS", "Key"]
