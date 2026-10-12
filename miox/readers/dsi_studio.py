####################################################################################################
#                                          dsi_studio.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 13/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for DSI Studio's source (.src) and fibre (.fib) files: gzipped MATLAB v4         #
#          variables holding the diffusion images with their b-table, or the maps reconstructed    #
#          from them.                                                                              #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import gzip
import re
import struct
import zlib

import numpy as np

# own
from miox.data import MRImage
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.readers.tractograms import matlab_v4
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                        DsiStudioImageReader                                      #
#**************************************************************************************************#
#                                                                                                  #
# "dimension" and "voxel_size" give the grid; every variable with one value per voxel is a volume, #
# x fastest (MATLAB stores the (x * y, z) matrices column by column). A .src file's image0,        #
# image1, ... are the diffusion-weighted volumes, in the order of b_table's columns (b-value, then #
# the direction); a .fib file's maps (fa0, dti_fa, iso, ...) become one image each, the first      #
# named one first. The geometry is "trans" (or "trans_to_mni") when the file holds it, else the    #
# voxel sizes only; the b-table directions stay in the vendor header as written.                   #
#                                                                                                  #
#**************************************************************************************************#
class DsiStudioImageReader(Reader):
    FORMATS = (Format.DSI_STUDIO_SRC, Format.DSI_STUDIO_FIB)
    PACKAGES = ("miox",)
    IMAGE = re.compile(r"^image(\d+)$")
    FIRST = ("fa0", "dti_fa", "gfa", "qa0")  # the map shown first in a .fib file

    def read(self, sniff: Sniff) -> list[MRImage]:
        raw = sniff.path.read_bytes()
        try:
            variables = matlab_v4(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
            sizes = [int(v) for v in variables["dimension"].ravel()[:3]]
        except (KeyError, OSError, EOFError, struct.error, ValueError, zlib.error) as e:
            raise ReaderError(f"{sniff.path.name}: not a readable DSI Studio file ({e})") from e
        voxel = [float(v) for v in variables.get("voxel_size", np.ones((1, 3))).ravel()[:3]]
        count = int(np.prod(sizes))
        notes: list[str] = []
        affine = self.affine(variables, voxel, notes)
        volumes = {k: v for k, v in variables.items() if v.size == count}

        def volume(values: np.ndarray) -> np.ndarray:
            return values.T.ravel().reshape(sizes[::-1]).transpose()  # column-major, x fastest

        report = variables.get("report")
        vendor = {k: self.plain(v.ravel() if v.size < 4096 else list(v.shape))
                  for k, v in variables.items() if k not in volumes and k != "report"}
        if report is not None:
            vendor["report"] = report.ravel().astype(np.uint8).tobytes().decode("utf-8", "replace")
        name = sniff.path.name.removesuffix(".gz").rsplit(".", 1)[0]
        images = sorted((int(m[1]), k) for k in volumes if (m := self.IMAGE.match(k)))
        if images:  # diffusion-weighted volumes of a .src file
            data = np.stack([volume(volumes[k]) for _, k in images], axis=-1)
            meta = Meta(vendor)
            table = variables.get("b_table")
            if table is not None and table.shape[1] == len(images):
                meta.set("DiffusionBValue", [float(b) for b in table[0]],
                         "DSI Studio b_table, first row", "b_table")
            return [MRImage(data, affine, meta, self.provenance(sniff, tuple(notes)), name,
                            ("diffusion",))]
        if not volumes:
            raise ReaderError(f"{sniff.path.name}: no variable with one value per voxel")
        order = sorted(volumes, key=lambda k: (k not in self.FIRST, self.FIRST.index(k)
                                               if k in self.FIRST else 0))
        return [MRImage(volume(volumes[k]), affine, Meta(vendor),
                        self.provenance(sniff, tuple(notes)), f"{name}_{k}") for k in order]

    @staticmethod
    def affine(variables: dict, voxel: list[float], notes: list[str]) -> np.ndarray:
        for key in ("trans", "trans_to_mni"):
            value = variables.get(key)
            if value is not None and value.size == 16:
                # DSI Studio writes the matrix row by row: a 4 x 4 variable reads transposed
                matrix = value.T if value.shape == (4, 4) else value.reshape(4, 4)
                return matrix.astype(np.float64)
        notes.append("no trans: voxel sizes only (DSI Studio states no world frame)")
        return np.diag(voxel + [1.0])
