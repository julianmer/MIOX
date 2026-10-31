####################################################################################################
#                                             gipl.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 31/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for GIPL images (Guy's Image Processing Lab format, .gipl and .gipl.gz) as ITK   #
#          reads and writes them.                                                                  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import gzip
import struct

import numpy as np

# own
from miox.data import MRImage
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                            GiplReader                                            #
#**************************************************************************************************#
#                                                                                                  #
# A 256-byte header, then the voxels with x fastest. The byte order is the one in which the magic  #
# number at byte 252 reads 0xefffe9b0 or 0x2ae389b8. Header: dimensions (4 x uint16 at 0; a fourth #
# above 1 is a time axis), the voxel type (uint16 at 8), voxel sizes in mm (4 x float32 at 10), a  #
# description (80 bytes at 26), minimum and maximum (float64 at 188 and 196) and the origin (4 x   #
# float64 at 204). As in ITK, the axes are those of its LPS patient space with no rotation; the    #
# affine turns that into RAS.                                                                      #
#                                                                                                  #
#**************************************************************************************************#
class GiplReader(Reader):
    FORMATS = (Format.GIPL,)
    PACKAGES = ("miox",)
    HEADER = 256
    MAGIC = (0xEFFFE9B0, 0x2AE389B8)
    TYPES = {1: "u1", 7: "i1", 8: "u1", 15: "i2", 16: "u2", 31: "u4", 32: "i4", 64: "f4",
             65: "f8"}
    LPS_TO_RAS = np.array([-1.0, -1.0, 1.0])

    def read(self, sniff: Sniff) -> list[MRImage]:
        opener = gzip.open if sniff.path.name.lower().endswith(".gz") else open
        with opener(sniff.path, "rb") as fh:
            blob = fh.read()
        order = self.byte_order(blob[:self.HEADER])
        if order is None:
            raise ReaderError(f"{sniff.path.name}: no GIPL magic number at byte 252")
        dims = [int(n) for n in struct.unpack_from(f"{order}4H", blob, 0)]
        code = struct.unpack_from(f"{order}H", blob, 8)[0]
        if code not in self.TYPES:
            raise ReaderError(f"{sniff.path.name}: voxel type {code} not read")
        spacing = struct.unpack_from(f"{order}4f", blob, 10)
        origin = struct.unpack_from(f"{order}4d", blob, 204)
        shape = [max(n, 1) for n in dims[:3]] + ([dims[3]] if dims[3] > 1 else [])
        count = int(np.prod(shape))
        data = np.frombuffer(blob, np.dtype(self.TYPES[code]).newbyteorder(order), count,
                             self.HEADER).reshape(shape[::-1]).transpose()
        data = data.astype(data.dtype.newbyteorder("="))
        affine = np.eye(4)
        affine[:3, :3] = np.diag([s if s > 0 else 1.0 for s in spacing[:3]])
        affine[:3, 3] = origin[:3]
        affine[:3, :] *= self.LPS_TO_RAS[:, np.newaxis]
        description = blob[26:106].split(b"\0")[0].decode("latin-1").strip()
        minimum, maximum = struct.unpack_from(f"{order}2d", blob, 188)
        meta = Meta({"dims": dims, "type": code, "spacing": list(spacing), "origin": list(origin),
                     "description": description, "min": minimum, "max": maximum,
                     "byte_order": "big" if order == ">" else "little"})
        name = sniff.path.name.removesuffix(".gz").removesuffix(".gipl")
        dims_names = ("time",) if len(shape) == 4 else ()
        return [MRImage(np.ascontiguousarray(data), affine, meta, self.provenance(sniff), name,
                        dims_names)]

    @classmethod
    def byte_order(cls, header: bytes) -> str | None:
        """">" or "<", whichever reads the magic number at byte 252; None if neither does."""
        if len(header) < cls.HEADER:
            return None
        for order in (">", "<"):
            if struct.unpack_from(f"{order}I", header, 252)[0] in cls.MAGIC:
                return order
        return None
