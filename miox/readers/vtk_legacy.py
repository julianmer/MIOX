####################################################################################################
#                                          vtk_legacy.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 12/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for VTK legacy files (.vtk) holding POLYDATA: lines as streamlines, polygons as  #
#          a surface mesh, with their point and cell data.                                         #
#                                                                                                  #
####################################################################################################


#*************#
# imports                                                                                          #
#*************#
from __future__ import annotations

import re

import numpy as np

# own
from miox.data import Surface, Tractogram
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                            VtkCursor                                             #
#**************************************************************************************************#
#                                                                                                  #
# Walks a legacy VTK file: keyword lines, then blocks of values, whitespace-separated text in      #
# ASCII files and big-endian binary in BINARY files (a newline follows each binary block).         #
#                                                                                                  #
#**************************************************************************************************#
class VtkCursor:
    TYPES = {
        "bit": "u1", "unsigned_char": "u1", "char": "i1", "unsigned_short": "u2", "short": "i2",
        "unsigned_int": "u4", "int": "i4", "unsigned_long": "u8", "long": "i8", "float": "f4",
        "double": "f8", "vtktypeint64": "i8", "vtktypeuint64": "u8", "vtkidtype": "i8",
    }
    TOKEN = re.compile(rb"\S+")

    def __init__(self, raw: bytes, binary: bool, position: int):
        self.raw, self.binary, self.position = raw, binary, position

    def line(self) -> list[str] | None:
        """The words of the next non-empty line, or None at the end of the file."""
        while self.position < len(self.raw):
            end = self.raw.find(b"\n", self.position)
            end = len(self.raw) if end < 0 else end
            words = self.raw[self.position:end].decode("ascii", "replace").split()
            self.position = end + 1
            if words:
                return words
        return None

    def values(self, count: int, kind: str) -> np.ndarray:
        """``count`` values of the VTK type ``kind``."""
        dtype = np.dtype(self.TYPES[kind.lower()])
        if self.binary:
            size = count * dtype.itemsize
            if self.position + size > len(self.raw):
                raise ReaderError(f"binary block of {count} {kind} runs past the end of the file")
            values = np.frombuffer(self.raw, dtype.newbyteorder(">"), count, self.position)
            self.position += size
            if self.raw[self.position:self.position + 1] == b"\n":
                self.position += 1
            return values.astype(dtype.newbyteorder("="))
        tokens = []
        for match in self.TOKEN.finditer(self.raw, self.position):
            tokens.append(match.group())
            if len(tokens) == count:
                self.position = match.end()
                break
        if len(tokens) < count:
            raise ReaderError(f"{len(tokens)} of {count} {kind} values found")
        return np.array(tokens, dtype=np.float64).astype(dtype)


#**************************************************************************************************#
#                                            VtkReader                                             #
#**************************************************************************************************#
#                                                                                                  #
# A legacy file starts with "# vtk DataFile Version x.y", a title line, ASCII or BINARY and the    #
# DATASET line. POLYDATA holds POINTS and cells: LINES become the streamlines of a Tractogram (one #
# per line cell, its points taken in cell order), POLYGONS a Surface (triangles; other polygons    #
# are refused). Cells are stored as (count, ids...) rows, or from version 5 as OFFSETS and         #
# CONNECTIVITY arrays. POINT_DATA (SCALARS, VECTORS, NORMALS, TENSORS, COLOR_SCALARS,              #
# TEXTURE_COORDINATES, FIELD arrays) go to point data, CELL_DATA of lines to streamline data       #
# (cells counted as VTK does: vertices, lines, polygons, strips), FIELD arrays of the dataset to   #
# the vendor header. The file states no coordinate frame: points are kept as stored.               #
#                                                                                                  #
#**************************************************************************************************#
class VtkReader(Reader):
    FORMATS = (Format.VTK_POLYDATA,)
    PACKAGES = ("miox",)
    SPACE = "not stated (VTK)"
    CELLS = ("VERTICES", "LINES", "POLYGONS", "TRIANGLE_STRIPS")
    NOTE = ("VTK states no coordinate frame; points kept as stored (dipy and MI-Brain write LPS, "
            "MRtrix RAS)")

    def read(self, sniff: Sniff) -> list[Tractogram | Surface]:
        raw = sniff.path.read_bytes()
        lines = raw.split(b"\n", 4)
        if len(lines) < 5:
            raise ReaderError(f"{sniff.path}: VTK header incomplete")
        version, title, encoding = (lines[i].decode("ascii", "replace").strip() for i in range(3))
        cursor = VtkCursor(raw, encoding.upper() == "BINARY", sum(len(x) + 1 for x in lines[:3]))
        dataset = cursor.line()
        if dataset is None or dataset[:2] != ["DATASET", "POLYDATA"]:
            raise ReaderError(f"{sniff.path}: only POLYDATA is read, not {dataset}")
        points, cells, data = None, {}, {"POINT_DATA": {}, "CELL_DATA": {}}
        field_data = {}  # FIELD arrays of the dataset itself, before point and cell data
        section = None
        while (words := cursor.line()) is not None:
            keyword = words[0].upper()
            if keyword == "POINTS":
                points = cursor.values(3 * int(words[1]), words[2]).reshape(-1, 3)
            elif keyword in self.CELLS:
                cells[keyword] = self._cells(cursor, words, version)
            elif keyword in data:
                section = keyword
            elif keyword == "METADATA":
                self._skip_metadata(cursor)
            elif keyword == "FIELD" and section is None:
                field_data.update(self._attribute(cursor, words, 0))
            elif section is not None:
                count = len(points) if section == "POINT_DATA" else sum(
                    len(c) for c in cells.values())
                data[section].update(self._attribute(cursor, words, count))
            else:
                raise ReaderError(f"{sniff.path}: unexpected line {' '.join(words)!r}")
        if points is None:
            raise ReaderError(f"{sniff.path}: no POINTS")
        vendor = {"version": version, "title": title, "encoding": encoding,
                  "cells": {k: len(v) for k, v in cells.items()},
                  "field_data": self.plain(field_data)}
        datasets = []
        if "LINES" in cells:
            datasets.append(self._tractogram(sniff, points, cells, data, vendor))
        if "POLYGONS" in cells:
            datasets.append(self._surface(sniff, points, cells["POLYGONS"], data, vendor))
        if not datasets:
            raise ReaderError(f"{sniff.path}: neither LINES nor POLYGONS")
        return datasets

    def _tractogram(self, sniff: Sniff, points: np.ndarray, cells: dict, data: dict,
                    vendor: dict) -> Tractogram:
        lines = cells["LINES"]
        order = np.concatenate(lines) if lines else np.zeros(0, dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum([len(c) for c in lines])]).astype(np.int64)
        point_data = {k: v[order] for k, v in data["POINT_DATA"].items()}
        first = len(cells.get("VERTICES", ()))  # line cells follow the vertex cells
        streamline_data = {k: v[first:first + len(lines)] for k, v in data["CELL_DATA"].items()}
        return Tractogram(points[order].astype(np.float64), offsets, self.SPACE, point_data,
                          streamline_data, None, Meta(vendor),
                          self.provenance(sniff, (self.NOTE,)), sniff.path.stem)

    def _surface(self, sniff: Sniff, points: np.ndarray, polygons: list, data: dict,
                 vendor: dict) -> Surface:
        if any(len(p) != 3 for p in polygons):
            raise ReaderError(f"{sniff.path}: polygons other than triangles are not read")
        return Surface(points.astype(np.float64), np.array(polygons, dtype=np.int64).reshape(-1, 3),
                       self.SPACE, dict(data["POINT_DATA"]), {}, Meta(vendor),
                       self.provenance(sniff, (self.NOTE,)), sniff.path.stem)

    @staticmethod
    def _cells(cursor: VtkCursor, words: list[str], version: str) -> list[np.ndarray]:
        """The point indices of each cell."""
        count, size = int(words[1]), int(words[2])
        if version.split()[-1] >= "5":  # OFFSETS and CONNECTIVITY arrays, count = cells + 1
            arrays = {}
            for _ in range(2):
                name, kind = cursor.line()
                arrays[name.upper()] = cursor.values(count if name.upper() == "OFFSETS" else size,
                                                     kind).astype(np.int64)
            offsets, connectivity = arrays["OFFSETS"], arrays["CONNECTIVITY"]
            return [connectivity[offsets[i]:offsets[i + 1]] for i in range(count - 1)]
        values = cursor.values(size, "int").astype(np.int64)
        cells, i = [], 0
        while i < len(values):
            cells.append(values[i + 1:i + 1 + values[i]])
            i += 1 + values[i]
        if len(cells) != count:
            raise ReaderError(f"{len(cells)} cells found, {count} stated")
        return cells

    @staticmethod
    def _skip_metadata(cursor: VtkCursor) -> None:
        """A METADATA block (information keys) runs to the next empty line."""
        while cursor.position < len(cursor.raw):
            end = cursor.raw.find(b"\n", cursor.position)
            end = len(cursor.raw) if end < 0 else end
            empty = not cursor.raw[cursor.position:end].strip()
            cursor.position = end + 1
            if empty:
                return

    @staticmethod
    def _attribute(cursor: VtkCursor, words: list[str], count: int) -> dict[str, np.ndarray]:
        """The arrays of one attribute (SCALARS, VECTORS, FIELD, ...) of ``count`` points or
        cells."""
        keyword = words[0].upper()
        if keyword == "SCALARS":
            columns = int(words[3]) if len(words) > 3 else 1
            table = cursor.line()
            if table is None or table[0].upper() != "LOOKUP_TABLE":
                raise ReaderError("SCALARS without LOOKUP_TABLE")
            values = cursor.values(count * columns, words[2])
            return {words[1]: values.reshape(count, columns) if columns > 1 else values}
        if keyword in ("VECTORS", "NORMALS"):
            return {words[1]: cursor.values(3 * count, words[2]).reshape(count, 3)}
        if keyword == "TENSORS":
            return {words[1]: cursor.values(9 * count, words[2]).reshape(count, 3, 3)}
        if keyword == "COLOR_SCALARS":
            columns = int(words[2])
            kind = "unsigned_char" if cursor.binary else "float"
            values = cursor.values(count * columns, kind).reshape(count, columns)
            return {words[1]: values / 255.0 if cursor.binary else values}
        if keyword == "TEXTURE_COORDINATES":
            columns = int(words[2])
            return {words[1]: cursor.values(count * columns, words[3]).reshape(count, columns)}
        if keyword == "FIELD":
            arrays = {}
            for _ in range(int(words[2])):
                name, columns, tuples, kind = cursor.line()
                values = cursor.values(int(columns) * int(tuples), kind)
                arrays[name] = values.reshape(int(tuples), -1) if int(columns) > 1 else values
            return arrays
        if keyword == "LOOKUP_TABLE":  # a colour table of its own: name, size, RGBA rows
            cursor.values(4 * int(words[2]), "unsigned_char" if cursor.binary else "float")
            return {}
        raise ReaderError(f"unknown attribute {keyword}")
