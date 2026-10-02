####################################################################################################
#                                           __init__.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 01/10/26                                                                                #
#                                                                                                  #
# Purpose: Metadata layer: the standard key table and the metadata of one dataset.                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#

# own
from miox.meta.keys import Key, KeyTable
from miox.meta.meta import Meta

__all__ = ["Key", "KeyTable", "Meta"]
