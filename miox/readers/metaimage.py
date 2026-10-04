####################################################################################################
#                                           metaimage.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 04/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for MetaImage files (.mha with attached data, .mhd with a detached data file),   #
#          the ITK/MetaIO image format.                                                            #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np

# own
from miox.data import MRImage
from miox.meta.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                         MetaImageReader                                          #
#**************************************************************************************************#
#                                                                                                  #
# "Key = value" lines up to ElementDataFile, which names the data file (LOCAL: the bytes that      #
# follow; a file name; LIST: one file per slice named on the following lines; a printf pattern     #
# with a range). DimSize gives the sizes, the first axis fastest; ElementType the MetaIO type;     #
# TransformMatrix (direction cosines, row-major) and Offset the geometry in ITK's patient space    #
# (LPS, turned into RAS here); ElementSpacing the voxel size; ElementNumberOfChannels a trailing   #
# channel dimension. CompressedData marks zlib-compressed bytes. Values are stored as written.     #
#                                                                                                  #
#**************************************************************************************************#
class MetaImageReader(Reader):
    FORMATS = (Format.METAIMAGE,)
    SOURCE = "MetaImage header"
    TYPES = {
        "MET_CHAR": "i1", "MET_UCHAR": "u1", "MET_SHORT": "i2", "MET_USHORT": "u2",
        "MET_INT": "i4", "MET_UINT": "u4", "MET_LONG": "i4", "MET_ULONG": "u4",
        "MET_LONG_LONG": "i8", "MET_ULONG_LONG": "u8", "MET_FLOAT": "f4", "MET_DOUBLE": "f8",
    }
    LPS_TO_RAS = np.array([-1.0, -1.0, 1.0])
    KEYS = {  # standard key: header key
        "Modality": "Modality",
        "PatientName": "PatientName",
        "PatientID": "PatientID",
        "StudyDescription": "StudyDescription",
        "SeriesDescription": "SeriesDescription",
    }

    def read(self, sniff: Sniff) -> list[MRImage]:
        header, offset = self._header(sniff.path)
        try:
            data = self._data(sniff.path, header, offset)
        except (ValueError, OSError, KeyError) as e:
            raise ReaderError(f"{sniff.path.name}: {e}") from e
        ndims = int(header["NDims"])
        notes = []
        channels = int(header.get("ElementNumberOfChannels", 1))
        while data.ndim - (channels > 1) < 3:
            data = np.expand_dims(data, data.ndim - (channels > 1))
        affine = self._affine(header, ndims, notes)
        meta = Meta({"header": header})
        for key, field in self.KEYS.items():
            if header.get(field):
                meta.set(key, header[field], f"{self.SOURCE}: {field}", f"header/{field}")
        if "AnatomicalOrientation" in header:
            meta.set("metaimage_AnatomicalOrientation", header["AnatomicalOrientation"],
                     f"{self.SOURCE}: AnatomicalOrientation", "header/AnatomicalOrientation")
        dims = ("volume",) * max(data.ndim - 3 - (channels > 1), 0)
        dims += ("channel",) if channels > 1 else ()
        name = sniff.path.name.rsplit(".", 1)[0]
        return [MRImage(data, affine, meta, self.provenance(sniff, tuple(notes)), name, dims)]

    def _header(self, path: Path) -> tuple[dict[str, str], int]:
        header: dict[str, str] = {}
        with open(path, "rb") as fh:
            while True:
                line = fh.readline()
                if not line:
                    break
                text = line.decode("utf-8", "replace").strip()
                if "=" not in text:
                    continue
                key, value = (part.strip() for part in text.split("=", 1))
                header[key] = value
                if key == "ElementDataFile":
                    break
            offset = fh.tell()
        if "NDims" not in header or "DimSize" not in header or "ElementType" not in header:
            raise ReaderError(f"{path.name}: header without NDims, DimSize or ElementType")
        return header, offset

    def _data(self, path: Path, header: dict[str, str], offset: int) -> np.ndarray:
        sizes = [int(v) for v in header["DimSize"].split()]
        dtype = np.dtype(self.TYPES[header["ElementType"].strip()])
        if dtype.itemsize > 1:
            big = header.get("BinaryDataByteOrderMSB", "False").strip().lower() == "true"
            dtype = dtype.newbyteorder(">" if big else "<")
        channels = int(header.get("ElementNumberOfChannels", 1))
        count = int(np.prod(sizes)) * channels
        compressed = header.get("CompressedData", "False").strip().lower() == "true"
        spec = header["ElementDataFile"].strip()
        skip = int(header.get("HeaderSize", 0) or 0)
        if spec.upper() == "LOCAL":
            blobs = [self._bytes(path, offset, compressed)]
        else:
            blobs = [self._bytes(f, skip, compressed) for f in self._files(path, spec, sizes)]
        size = dtype.itemsize
        raw = np.concatenate([np.frombuffer(b[: (len(b) // size) * size], dtype) for b in blobs])
        if raw.size < count:
            raise ValueError(f"{raw.size} values, expected {count}")
        shape = sizes[::-1] + ([channels] if channels > 1 else [])
        data = raw[:count].reshape(shape)
        order = list(range(len(sizes)))[::-1] + ([len(sizes)] if channels > 1 else [])
        return np.transpose(data, order)

    @staticmethod
    def _bytes(path: Path, skip: int, compressed: bool) -> bytes:
        with open(path, "rb") as fh:
            fh.seek(skip if skip > 0 else 0)
            blob = fh.read()
        if skip == -1:
            return blob  # the caller takes the trailing bytes; sizes are checked afterwards
        return zlib.decompress(blob) if compressed else blob

    @staticmethod
    def _files(path: Path, spec: str, sizes: list[int]) -> list[Path]:
        parts = spec.split()
        if parts[0].upper() == "LIST":
            text = path.read_text(errors="replace")
            after = text.split("ElementDataFile", 1)[1].split("\n", 1)[1]
            return [path.parent / line.strip() for line in after.splitlines() if line.strip()]
        if len(parts) >= 3 and "%" in parts[0]:
            pattern, start, stop = parts[0], int(parts[1]), int(parts[2])
            step = int(parts[3]) if len(parts) > 3 else 1
            return [path.parent / (pattern % n) for n in range(start, stop + 1, step)]
        return [path.parent / spec]

    def _affine(self, header: dict[str, str], ndims: int, notes: list[str]) -> np.ndarray:
        n = min(ndims, 3)
        spacing = [float(v) for v in header.get("ElementSpacing", "1 " * ndims).split()][:n]
        spacing += [1.0] * (3 - len(spacing))
        matrix = np.eye(3)
        if "TransformMatrix" in header:
            values = [float(v) for v in header["TransformMatrix"].split()]
            if len(values) == ndims * ndims:
                matrix[:n, :n] = np.asarray(values).reshape(ndims, ndims)[:n, :n].T
        offset = np.zeros(3)
        if "Offset" in header:
            values = [float(v) for v in header["Offset"].split()]
            offset[: min(len(values), 3)] = values[:3]
        elif "Position" in header:
            values = [float(v) for v in header["Position"].split()]
            offset[: min(len(values), 3)] = values[:3]
        else:
            notes.append("no Offset: origin at zero")
        affine = np.eye(4)
        affine[:3, :3] = matrix * np.asarray(spacing)
        affine[:3, 3] = offset
        affine[:3, :] *= self.LPS_TO_RAS[:, np.newaxis]  # ITK's patient space is LPS
        return affine
