####################################################################################################
#                                           bart_cfl.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for BART arrays: a .cfl file of complex single-precision values in column-major  #
#          order with its .hdr text header, or a .ra "rawarray" file carrying the same values with #
#          a binary header.                                                                        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

# own
from miox.data import MRImage
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                            BartReader                                            #
#**************************************************************************************************#
#                                                                                                  #
# The .hdr file holds sections headed by a line starting with # (Dimensions, Command, Files,       #
# Creator, ...), each followed by one line; the Dimensions line lists the 16 sizes of BART's       #
# dimensions (read, phase 1, phase 2, coil, maps, TE, coefficient, coefficient 2, iteration,       #
# circular shift, time, time 2, level, slice, average, batch). The .cfl file holds the complex64   #
# values with the first dimension fastest. A .ra file starts with eight 64-bit words (the magic    #
# "rawarray", flags, element type 4 for complex, element size, data bytes, dimension count), then  #
# the sizes, then the values. The array becomes an MRImage without geometry: BART does not record  #
# whether it holds an image or k-space, and a reconstruction's output is the former. Trailing      #
# dimensions of size one are dropped, as BART's own reader does, three at least kept.              #
#                                                                                                  #
#**************************************************************************************************#
class BartReader(Reader):
    FORMATS = (Format.BART_CFL,)
    SOURCE = "BART header"
    DIMENSIONS = (
        "read", "phase1", "phase2", "coil", "maps", "te", "coeff", "coeff2", "iter", "cshift",
        "time", "time2", "level", "slice", "avg", "batch",
    )
    RA_MAGIC = 0x7961727261776172
    RA_COMPLEX = 4

    def read(self, sniff: Sniff) -> list[MRImage]:
        files = {p.suffix.lower(): p for p in (sniff.path,) + sniff.companions}
        if ".ra" in files:
            dims, data, header = self._ra(files[".ra"])
        elif ".hdr" in files and ".cfl" in files:
            header = self.header(files[".hdr"])
            dims = [int(v) for v in str(header.get("Dimensions", "")).split()]
            if not dims:
                raise ReaderError(f"{files['.hdr'].name}: no Dimensions line")
            data = np.fromfile(files[".cfl"], dtype="<c8")
            if data.size != int(np.prod(dims)):
                raise ReaderError(
                    f"{files['.cfl'].name}: {data.size} values, {int(np.prod(dims))} expected"
                )
        else:
            raise ReaderError(f"{sniff.path.name}: no .cfl with .hdr, nor a .ra file")
        kept = list(dims)
        while len(kept) > 3 and kept[-1] == 1:
            kept.pop()
        while len(kept) < 3:
            kept.append(1)
        array = data.reshape(kept, order="F")
        meta = Meta({"header": header, "dimensions": dims})
        meta.set("Modality", "MR", f"{self.SOURCE}: BART array", "dimensions")
        names = self.DIMENSIONS[3 : len(kept)]
        notes = ("no geometry stated; the affine is a placeholder; the values may be k-space",)
        provenance = self.provenance(sniff, notes)
        name = sniff.path.with_suffix("").name
        return [MRImage(array, self.NO_GEOMETRY.copy(), meta, provenance, name, names)]

    @staticmethod
    def header(path: Path) -> dict[str, str]:
        """The .hdr sections: the line after each "# Name" line, by name."""
        header: dict[str, str] = {}
        current = None
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith("#"):
                current = line.lstrip("#").strip()
                header.setdefault(current, "")
            elif current is not None:
                header[current] = (header[current] + " " + line.strip()).strip()
        return header

    def _ra(self, path: Path) -> tuple[list[int], np.ndarray, dict]:
        with open(path, "rb") as fh:
            magic, flags, element_type, element_size, data_size, ndims = struct.unpack(
                "<6Q", fh.read(48)
            )
            if magic != self.RA_MAGIC or element_type != self.RA_COMPLEX or element_size != 8:
                raise ReaderError(f"{path.name}: not a complex single-precision rawarray")
            if flags & 1:
                raise ReaderError(f"{path.name}: big-endian rawarray not supported")
            dims = [int(v) for v in struct.unpack(f"<{ndims}Q", fh.read(8 * ndims))]
            data = np.fromfile(fh, dtype="<c8", count=data_size // 8)
        header = {"flags": flags, "element_type": element_type, "data_bytes": data_size}
        return dims, data, header
