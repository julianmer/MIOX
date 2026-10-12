####################################################################################################
#                                             vista.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 13/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for Vista images (.v; Lipsia, Cylarim): a text header of nested attributes, a    #
#          form feed, then the binary blocks the header points to.                                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re

import numpy as np

# own
from miox.data import MRImage
from miox.meta.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                           VistaReader                                            #
#**************************************************************************************************#
#                                                                                                  #
# "V-data 2 {" opens the header: lines "name: value" or "name: type {" ... "}"; "\f\n" ends it.    #
# Every "image" object is one image: nbands x nrows x ncolumns values of type repn, big-endian, at #
# its data offset after the header (bit: packed, most significant first; NIfTI's x, y, z are       #
# ncolumns, nrows, nbands). Lipsia's geoinfo object keeps the NIfTI geometry: the sform as a 4 x 4 #
# big-endian float image (last row stored as zeros) when sform_code is set, else the qform         #
# (quaternion b, c, d and offsets, six floats) with pixdim (the qfac first), both bundles of       #
# little-endian floats; else the voxel attribute gives the sizes.                                  #
#                                                                                                  #
#**************************************************************************************************#
class VistaReader(Reader):
    FORMATS = (Format.VISTA,)
    END = b"\x0c\n"  # form feed: the end of the header
    TYPES = {"ubyte": ">u1", "sbyte": ">i1", "short": ">i2", "long": ">i4", "float": ">f4",
             "double": ">f8"}
    LINE = re.compile(r"^\s*([^:{}]+?)\s*:\s*(.*?)\s*$")

    def read(self, sniff: Sniff) -> list[MRImage]:
        blob = sniff.path.read_bytes()
        end = blob.find(self.END)
        if end < 0:
            raise ReaderError(f"{sniff.path.name}: no end of the Vista header")
        tree = self.parse(blob[:end].decode("latin-1"))
        body = blob[end + len(self.END):]
        images = [(name, node) for name, node in tree.items()
                  if isinstance(node, dict) and node.get("_type") == "image"]
        if not images:
            raise ReaderError(f"{sniff.path.name}: no image in the Vista file")
        geo = tree.get("geoinfo") if isinstance(tree.get("geoinfo"), dict) else {}
        out = []
        for name, node in images:
            notes: list[str] = []
            data = self.block(body, node).transpose(2, 1, 0)  # (columns, rows, bands)
            affine = self.affine(body, geo, node, notes)
            meta = Meta({"header": {k: v for k, v in tree.items()}})
            label = sniff.path.name.rsplit(".", 1)[0] + (f"_{name}" if len(images) > 1 else "")
            out.append(MRImage(data, affine, meta, self.provenance(sniff, tuple(notes)), label))
        return out

    @classmethod
    def parse(cls, text: str) -> dict:
        """The attributes as nested dicts; an object's type is kept under "_type"."""
        lines = text.splitlines()[1:]  # after "V-data 2 {"
        stack = [{}]
        for line in lines:
            if line.strip() == "}":
                if len(stack) > 1:
                    stack.pop()
                continue
            match = cls.LINE.match(line)
            if not match:
                continue
            key, value = match.groups()
            if value.endswith("{"):
                child = {"_type": value[:-1].strip()} if value[:-1].strip() else {}
                stack[-1][key] = child
                stack.append(child)
            else:
                stack[-1][key] = value.strip('"')
        return stack[0]

    def block(self, body: bytes, node: dict) -> np.ndarray:
        repn = node.get("repn", "")
        if repn not in self.TYPES and repn != "bit":
            raise ReaderError(f"Vista repn {repn!r} not supported")
        shape = [int(node.get(k, 1)) for k in ("nbands", "nrows", "ncolumns")]
        start, length = int(node["data"]), int(node["length"])
        raw = body[start:start + length]
        if repn == "bit":  # packed, most significant bit first
            bits = np.unpackbits(np.frombuffer(raw, np.uint8), bitorder="big")
            return bits[:int(np.prod(shape))].reshape(shape)
        values = np.frombuffer(raw, self.TYPES[repn])
        if values.size != int(np.prod(shape)):
            raise ReaderError(f"Vista image of {values.size} values, expected {shape}")
        return values.reshape(shape).astype(values.dtype.newbyteorder("="))

    def affine(self, body: bytes, geo: dict, node: dict, notes: list[str]) -> np.ndarray:
        def floats(name: str, dtype: str) -> np.ndarray | None:
            part = geo.get(name)
            if not isinstance(part, dict) or "data" not in part:
                return None
            start, length = int(part["data"]), int(part["length"])
            return np.frombuffer(body[start:start + length], dtype).astype(float)

        if int(geo.get("sform_code", 0) or 0) > 0 and (sform := floats("sform", ">f4")) is not None:
            affine = sform.reshape(4, 4).copy()
            affine[3] = [0.0, 0.0, 0.0, 1.0]  # stored as zeros
            return affine
        qform, pixdim = floats("qform", "<f4"), floats("pixdim", "<f4")
        if int(geo.get("qform_code", 0) or 0) > 0 and qform is not None and pixdim is not None:
            b, c, d, x, y, z = qform[:6]
            a = np.sqrt(max(0.0, 1.0 - b * b - c * c - d * d))
            rotation = np.array([
                [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - c * c - b * b]])
            qfac = -1.0 if pixdim[0] < 0 else 1.0
            affine = np.eye(4)
            affine[:3, :3] = rotation * np.array([pixdim[1], pixdim[2], pixdim[3] * qfac])
            affine[:3, 3] = [x, y, z]
            return affine
        voxel = node.get("voxel") or geo.get("voxel")
        sizes = [float(v) for v in str(voxel).split()][:3] if voxel else [1.0, 1.0, 1.0]
        notes.append("no sform or qform: voxel sizes only" if voxel else "no geometry")
        return np.diag(sizes + [1.0] * (4 - len(sizes)))
