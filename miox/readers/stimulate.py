####################################################################################################
#                                          stimulate.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 13/10/26                                                                                #
#                                                                                                  #
# Purpose: Readers for two text-header image formats with raw data beside them: Stimulate          #
#          (.spr/.sdt; CMRR, written by Slicer and Mango) and the UCSF image descriptor file of    #
#          SIVIC (.idf with .byt, .int2 or .real).                                                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

from pathlib import Path

import numpy as np

# own
from miox.data import MRImage
from miox.meta.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff

LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0])


def sibling(path: Path, suffixes: tuple[str, ...]) -> Path | None:
    """The file next to ``path`` with one of the suffixes (any case)."""
    for suffix in suffixes:
        for candidate in (path.with_suffix(suffix), path.with_suffix(suffix.upper())):
            if candidate.is_file():
                return candidate
    return None


#**************************************************************************************************#
#                                         StimulateReader                                          #
#**************************************************************************************************#
#                                                                                                  #
# "name: value" lines: dim (x fastest), dataType (BYTE, WORD, LWORD, REAL, COMPLEX), endian        #
# (ieee-be or ieee-le, big when absent), interval (voxel sizes), origin (the first voxel's centre  #
# in LPS). Axial files ("sdtOrient: ax") run from right to left, back to front and foot to head,   #
# as Slicer writes them (checked against the same scan as MetaImage); other orientations are read  #
# with these axes and say so.                                                                      #
#                                                                                                  #
#**************************************************************************************************#
class StimulateReader(Reader):
    FORMATS = (Format.STIMULATE,)
    TYPES = {"BYTE": "u1", "BTYE": "u1", "WORD": "i2", "LWORD": "i4", "REAL": "f4",
             "COMPLEX": "c8"}

    def read(self, sniff: Sniff) -> list[MRImage]:
        header_path = sniff.path if sniff.path.suffix.lower() == ".spr" else sibling(
            sniff.path, (".spr",))
        data_path = sibling(header_path, (".sdt",)) if header_path else None
        if header_path is None or data_path is None:
            raise ReaderError(f"{sniff.path.name}: Stimulate needs both .spr and .sdt")
        header = {}
        for line in header_path.read_text(errors="replace").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                header[key.strip()] = value.strip()
        try:
            sizes = [int(v) for v in header["dim"].split()]
            kind = header.get("dataType", "BYTE").upper()
            big = "le" not in header.get("endian", "ieee-be").lower()
            dtype = np.dtype(self.TYPES[kind]).newbyteorder(">" if big else "<")
        except (KeyError, ValueError) as e:
            raise ReaderError(f"{header_path.name}: {type(e).__name__}: {e}") from e
        values = np.fromfile(data_path, dtype)
        if values.size < int(np.prod(sizes)):
            raise ReaderError(f"{data_path.name}: {values.size} values, expected {sizes}")
        data = values[:int(np.prod(sizes))].reshape(sizes[::-1]).transpose()
        data = data.astype(data.dtype.newbyteorder("="))
        data = data.reshape(data.shape + (1,) * (3 - data.ndim))
        notes = []
        orient = header.get("sdtOrient", "ax").lower()
        if orient != "ax":
            notes.append(f"sdtOrient {orient}: axes taken as for axial files (not checked)")
        interval = [float(v) for v in header.get("interval", "1 1 1").split()][:3]
        interval += [1.0] * (3 - len(interval))
        origin = [float(v) for v in header.get("origin", "0 0 0").split()][:3]
        origin += [0.0] * (3 - len(origin))
        affine = np.eye(4)
        affine[:3, :3] = np.diag([-interval[0], interval[1], interval[2]])
        affine[:3, 3] = LPS_TO_RAS @ np.asarray(origin)
        meta = Meta({"header": header})
        dims = ("volume",) * max(data.ndim - 3, 0)
        return [MRImage(data, affine, meta, self.provenance(sniff, tuple(notes)),
                        header_path.stem, dims)]


#**************************************************************************************************#
#                                         SivicIdfReader                                           #
#**************************************************************************************************#
#                                                                                                  #
# The UCSF image descriptor (SIVIC): "dimension: n" lines each followed by "npix: ... pixelsize    #
# (mm): ..."; LPS geometry from toplc (the first voxel's centre) and dcos1-3 (the directions of    #
# the three axes). The voxels are in the .byt (unsigned bytes), .int2 (unsigned 16-bit) or .real   #
# (float) file beside it, big-endian, x fastest.                                                   #
#                                                                                                  #
#**************************************************************************************************#
class SivicIdfReader(Reader):
    FORMATS = (Format.SIVIC_IDF,)
    DATA = ((".byt", ">u1"), (".int2", ">u2"), (".real", ">f4"))

    def read(self, sniff: Sniff) -> list[MRImage]:
        header_path = sniff.path if sniff.path.suffix.lower() == ".idf" else sibling(
            sniff.path, (".idf",))
        if header_path is None:
            raise ReaderError(f"{sniff.path.name}: no .idf header beside it")
        found = [(p, t) for s, t in self.DATA if (p := sibling(header_path, (s,)))]
        if not found:
            raise ReaderError(f"{header_path.name}: no .byt, .int2 or .real data beside it")
        data_path, dtype = found[0]
        lines = header_path.read_text(errors="replace").splitlines()
        sizes, pixel, vectors = {}, {}, {}
        for n, line in enumerate(lines):
            parts = line.split()
            if parts[:1] == ["dimension:"] and n + 1 < len(lines):
                axis, spec = int(parts[1]), lines[n + 1].split()
                if spec[:1] == ["npix:"] and len(spec) >= 8:
                    sizes[axis], pixel[axis] = int(spec[1]), float(spec[7])
            if parts and parts[0] in ("toplc:", "dcos1:", "dcos2:", "dcos3:") and len(parts) > 3:
                vectors[parts[0][:-1]] = np.array([float(v) for v in parts[1:4]])
        if not sizes:
            raise ReaderError(f"{header_path.name}: no dimensions")
        shape = [sizes[k] for k in sorted(sizes)]
        values = np.fromfile(data_path, dtype)
        if values.size < int(np.prod(shape)):
            raise ReaderError(f"{data_path.name}: {values.size} values, expected {shape}")
        data = values[:int(np.prod(shape))].reshape(shape[::-1]).transpose()
        data = data.astype(data.dtype.newbyteorder("="))
        data = data.reshape(data.shape + (1,) * (3 - data.ndim))
        notes = []
        directions = np.eye(3)
        for i, key in enumerate(("dcos1", "dcos2", "dcos3")):
            if key in vectors:
                directions[:, i] = vectors[key]
            else:
                notes.append(f"no {key}: axis {i + 1} along the scanner axis")
        steps = [pixel.get(k, 1.0) for k in (1, 2, 3)]
        affine = np.eye(4)
        affine[:3, :3] = LPS_TO_RAS @ directions @ np.diag(steps)
        affine[:3, 3] = LPS_TO_RAS @ vectors.get("toplc", np.zeros(3))
        meta = Meta({"header": {"lines": lines}})
        dims = ("volume",) * max(data.ndim - 3, 0)
        return [MRImage(data, affine, meta, self.provenance(sniff, tuple(notes)),
                        header_path.stem, dims)]
