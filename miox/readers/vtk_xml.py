####################################################################################################
#                                           vtk_xml.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 13/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for VTK XML image data (.vti): an ImageData grid whose point data are inline     #
#          text, inline base64 or appended, raw or zlib-compressed.                                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import base64
import re
import zlib

import numpy as np

# own
from miox.data import MRImage
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                        VtkXmlImageReader                                         #
#**************************************************************************************************#
#                                                                                                  #
# <VTKFile type="ImageData"> names the byte order, the header integer type and the compressor;     #
# <ImageData> the WholeExtent, Origin, Spacing and (optionally) Direction. The first DataArray of  #
# the PointData (the active Scalars when named) is the image: format "ascii" (numbers in the       #
# element), "binary" (base64 in the element) or "appended" (at its offset after the "_" of         #
# <AppendedData>, raw or base64). Binary blocks start with a header of the header type: the byte   #
# count, or with zlib the block count, block size, last block size and the compressed sizes - the  #
# header and the blocks base64-encoded apart. Values run x fastest, components last.               #
#                                                                                                  #
#**************************************************************************************************#
class VtkXmlImageReader(Reader):
    FORMATS = (Format.VTK_IMAGE_XML,)
    TYPES = {"Int8": "i1", "UInt8": "u1", "Int16": "i2", "UInt16": "u2", "Int32": "i4",
             "UInt32": "u4", "Int64": "i8", "UInt64": "u8", "Float32": "f4", "Float64": "f8"}
    ATTRIBUTE = re.compile(r'(\w+)="([^"]*)"')
    NOTE = "VTK states no coordinate frame; the grid is kept as stored"

    def read(self, sniff: Sniff) -> list[MRImage]:
        raw = sniff.path.read_bytes()
        cut = raw.find(b"<AppendedData")
        text = raw[: cut if cut >= 0 else len(raw)].decode("utf-8", "replace")
        try:
            data, vendor, affine = self._image(raw, text, cut)
        except (KeyError, ValueError, IndexError, zlib.error) as e:
            raise ReaderError(f"{sniff.path.name}: {type(e).__name__}: {e}") from e
        dims = ("component",) * (data.ndim > 3)
        return [MRImage(data, affine, Meta(vendor), self.provenance(sniff, (self.NOTE,)),
                        sniff.path.stem, dims)]

    def attributes(self, text: str, tag: str) -> dict[str, str]:
        match = re.search(rf"<{tag}\b([^>]*)>", text)
        if match is None:
            raise KeyError(f"no <{tag}>")
        return dict(self.ATTRIBUTE.findall(match.group(1)))

    def _image(self, raw: bytes, text: str, cut: int):
        top = self.attributes(text, "VTKFile")
        if top.get("type") != "ImageData":
            raise ValueError(f"VTKFile type {top.get('type')}, not ImageData")
        order = ">" if top.get("byte_order", "LittleEndian") == "BigEndian" else "<"
        header = np.dtype(self.TYPES[top.get("header_type", "UInt32")]).newbyteorder(order)
        compressed = "ZLib" in top.get("compressor", "")
        grid = self.attributes(text, "ImageData")
        extent = [int(v) for v in grid["WholeExtent"].split()]
        sizes = [extent[1] - extent[0] + 1, extent[3] - extent[2] + 1, extent[5] - extent[4] + 1]
        point = re.search(r"<PointData\b([^>]*)>(.*?)</PointData>", text, re.S)
        if point is None and cut >= 0:
            point = re.search(r"<PointData\b([^>]*)>(.*)", text, re.S)
        if point is None:
            raise KeyError("no <PointData>")
        scalars = dict(self.ATTRIBUTE.findall(point.group(1))).get("Scalars")
        arrays = re.findall(r"<DataArray\b([^>]*?)(/>|>(.*?)</DataArray>)", point.group(2), re.S)
        if not arrays:
            raise KeyError("no DataArray in the PointData")
        chosen = next((a for a in arrays if dict(self.ATTRIBUTE.findall(a[0])).get("Name")
                       == scalars), arrays[0])
        info = dict(self.ATTRIBUTE.findall(chosen[0]))
        dtype = np.dtype(self.TYPES[info["type"]]).newbyteorder(order)
        components = int(info.get("NumberOfComponents", 1))
        count = int(np.prod(sizes)) * components
        kind = info.get("format", "ascii")
        if kind == "ascii":
            values = np.array(chosen[2].split(), dtype=np.float64).astype(dtype)
        elif kind == "binary":
            values = self.decode_base64(chosen[2].strip().encode(), header, dtype, compressed)
        else:  # appended
            values = self.appended(raw, cut, int(info.get("offset", 0)), header, dtype,
                                   compressed)
        values = values[:count].astype(dtype.newbyteorder("="))
        if values.size < count:
            raise ValueError(f"{values.size} values, expected {count}")
        data = values.reshape(sizes[::-1] + ([components] if components > 1 else []))
        data = np.moveaxis(data, (0, 1, 2), (2, 1, 0))
        spacing = [float(v) for v in grid.get("Spacing", "1 1 1").split()]
        origin = [float(v) for v in grid.get("Origin", "0 0 0").split()]
        direction = np.array([float(v) for v in grid.get("Direction", "1 0 0 0 1 0 0 0 1")
                              .split()]).reshape(3, 3)
        affine = np.eye(4)
        affine[:3, :3] = direction @ np.diag(spacing)
        affine[:3, 3] = np.asarray(origin) + affine[:3, :3] @ np.array(extent[0::2], dtype=float)
        vendor = {"VTKFile": top, "ImageData": grid, "DataArray": info,
                  "arrays": [dict(self.ATTRIBUTE.findall(a[0])).get("Name") for a in arrays]}
        return data, vendor, affine

    @staticmethod
    def blocks(head: np.ndarray, payload: bytes, compressed: bool) -> bytes:
        """The data bytes after a block header (already read as integers)."""
        if not compressed:
            return payload[: int(head[0])]
        out, at = [], 0
        for size in head[3:3 + int(head[0])]:
            out.append(zlib.decompress(payload[at:at + int(size)]))
            at += int(size)
        return b"".join(out)

    def decode_base64(self, text: bytes, header: np.dtype, dtype: np.dtype,
                      compressed: bool) -> np.ndarray:
        size = header.itemsize
        first = base64.b64decode(text[: 4 * ((size * (3 if compressed else 1) + 2) // 3)])
        count = int(np.frombuffer(first[:size], header)[0])
        fields = 3 + count if compressed else 1
        length = 4 * ((size * fields + 2) // 3)  # the header is encoded on its own
        head = np.frombuffer(base64.b64decode(text[:length])[: size * fields], header)
        payload = base64.b64decode(text[length:])
        if not compressed and len(payload) < int(head[0]):  # header and data encoded together
            whole = base64.b64decode(text)
            head, payload = np.frombuffer(whole[:size], header), whole[size:]
        return np.frombuffer(self.blocks(head, payload, compressed), dtype)

    def appended(self, raw: bytes, cut: int, offset: int, header: np.dtype, dtype: np.dtype,
                 compressed: bool) -> np.ndarray:
        if cut < 0:
            raise KeyError("appended data array without <AppendedData>")
        tag_end = raw.index(b">", cut)
        encoding = dict(self.ATTRIBUTE.findall(raw[cut:tag_end].decode())).get("encoding", "raw")
        start = raw.index(b"_", tag_end) + 1
        if encoding == "base64":
            end = raw.find(b"</AppendedData>", start)
            text = raw[start:end].strip()[offset:]
            return self.decode_base64(text, header, dtype, compressed)
        at = start + offset
        size = header.itemsize
        count = int(np.frombuffer(raw[at:at + size], header)[0])
        fields = 3 + count if compressed else 1
        head = np.frombuffer(raw[at:at + size * fields], header)
        return np.frombuffer(self.blocks(head, raw[at + size * fields:], compressed), dtype)
