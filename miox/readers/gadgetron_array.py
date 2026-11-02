####################################################################################################
#                                       gadgetron_array.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/11/26                                                                                #
#                                                                                                  #
# Purpose: Reader for Gadgetron hoNDArray dumps (.real, .cplx, .short): the dimensions, then the   #
#          elements with the first dimension fastest.                                              #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import struct

import numpy as np

# own
from miox.data import MRImage
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                      GadgetronArrayReader                                        #
#**************************************************************************************************#
#                                                                                                  #
# Gadgetron writes an array as the number of dimensions (int32), the dimensions (int32 each) and   #
# the elements in column-major order; the extension names the element type (.real float32, .cplx  #
# complex64, .short int16). Nothing says what the array holds, so it becomes an MRImage without    #
# geometry with the dimensions as stored, three at least (as for BART arrays); the values may be   #
# k-space.                                                                                         #
#                                                                                                  #
#**************************************************************************************************#
class GadgetronArrayReader(Reader):
    FORMATS = (Format.GADGETRON_ARRAY,)
    PACKAGES = ("miox",)
    SOURCE = "Gadgetron array header"
    DTYPES = {".real": "<f4", ".cplx": "<c8", ".short": "<i2"}

    def read(self, sniff: Sniff) -> list[MRImage]:
        dtype = self.DTYPES.get(sniff.path.suffix.lower())
        if dtype is None:
            raise ReaderError(f"{sniff.path.name}: not a .real, .cplx or .short array")
        with open(sniff.path, "rb") as fh:
            ndim = struct.unpack("<i", fh.read(4))[0]
            if not 1 <= ndim <= 16:
                raise ReaderError(f"{sniff.path.name}: {ndim} dimensions")
            dims = list(struct.unpack(f"<{ndim}i", fh.read(4 * ndim)))
            data = np.fromfile(fh, dtype=dtype)
        if data.size != int(np.prod(dims)):
            raise ReaderError(f"{sniff.path.name}: {data.size} elements for dimensions {dims}")
        kept = dims + [1] * (3 - len(dims))
        array = data.reshape(kept, order="F")
        meta = Meta({"dimensions": dims, "element_type": str(np.dtype(dtype))})
        meta.set("Modality", "MR", f"{self.SOURCE}: Gadgetron array", "dimensions")
        notes = ("no geometry stated; the affine is a placeholder; the values may be k-space",)
        return [MRImage(array, self.NO_GEOMETRY.copy(), meta, self.provenance(sniff, notes),
                        sniff.path.stem)]
