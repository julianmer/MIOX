####################################################################################################
#                                           surfaces.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 10/10/26                                                                                #
#                                                                                                  #
# Purpose: Readers for surface meshes and per-vertex data: GIFTI (.gii, also gzip/bzip2) and       #
#          FreeSurfer surfaces, morphometry (curv, sulc, thickness, ...), annotations and labels,  #
#          read by nibabel; NiiVue/Surfice MZ3 meshes.                                             #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import gzip
import struct

import nibabel as nib
import numpy as np
from nibabel.gifti.util import gifti_encoding_codes, gifti_endian_codes

# own
from miox.data import Surface
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                           GiftiReader                                            #
#**************************************************************************************************#
#                                                                                                  #
# A GIFTI file holds data arrays by intent: a pointset (vertices in mm, in the coordinate system   #
# its DataSpace names, RAS as in NIfTI) and triangles make the mesh, label arrays use the file's   #
# label table (key: name, RGBA from 0 to 1), every other array (shape, time series, ...) is        #
# per-vertex data under its Name (or intent); arrays of one name are stacked as columns. Each      #
# array's encoding, data type and coordinate system go to the vendor header.                       #
#                                                                                                  #
#**************************************************************************************************#
class GiftiReader(Reader):
    FORMATS = (Format.GIFTI,)
    PACKAGES = ("miox", "nibabel")

    def read(self, sniff: Sniff) -> list[Surface]:
        path = sniff.path
        notes: list[str] = []
        try:
            with self.captured(notes):
                image = nib.load(path)
                arrays = [(a, np.asarray(a.data)) for a in image.darrays]
        except Exception as e:  # nibabel raises several types for malformed XML or encodings
            raise ReaderError(f"{path}: not a readable GIFTI file ({e})") from e
        vertices = faces = None
        space = "not stated"
        data: dict[str, list[np.ndarray]] = {}
        labels = {}
        table = {int(label.key): (label.label, tuple(np.nan if c is None else float(c)
                                                     for c in label.rgba))
                 for label in image.labeltable.labels}
        for i, (array, values) in enumerate(arrays):
            intent = nib.nifti1.intent_codes.label[array.intent]
            name = dict(array.meta).get("Name") or intent
            if intent == "pointset" and vertices is None:
                vertices = values.astype(np.float64)
                if array.coordsys is not None:
                    space = nib.nifti1.xform_codes.label[array.coordsys.dataspace]
            elif intent == "pointset":
                notes.append(f"further pointset {i} kept as vertex data")
                data.setdefault(f"pointset_{i}", []).append(values)
            elif intent == "triangle":
                faces = values.astype(np.int64)
            elif intent == "label":
                labels[name] = {"values": values, "table": table}
            else:
                data.setdefault(name, []).append(values)
        vertex_data = {
            name: values[0] if len(values) == 1 else np.column_stack(values)
            for name, values in data.items()
        }
        vendor = {
            "version": image.version,
            "meta": dict(image.meta),
            "labeltable": {k: {"label": v[0], "rgba": list(v[1])} for k, v in table.items()},
            "arrays": [self._array(a) for a, _ in arrays],
        }
        name = path.name
        for suffix in (".gz", ".bz2", ".gii"):
            name = name.removesuffix(suffix)
        return [Surface(vertices, faces, f"{space} (GIFTI, mm)", vertex_data, labels,
                        Meta(self.plain(vendor)), self.provenance(sniff, tuple(notes)), name)]

    @staticmethod
    def _array(array) -> dict:
        """What a data array states besides its values."""
        coordsys = array.coordsys
        return {
            "intent": nib.nifti1.intent_codes.label[array.intent],
            "datatype": nib.nifti1.data_type_codes.label[array.datatype],
            "encoding": gifti_encoding_codes.specs[array.encoding],
            "endian": gifti_endian_codes.specs[array.endian],
            "external_file": array.ext_fname or None,
            "external_offset": array.ext_offset,
            "shape": list(array.data.shape),
            "meta": dict(array.meta),
            "coordsys": None if coordsys is None else {
                "dataspace": nib.nifti1.xform_codes.label[coordsys.dataspace],
                "xformspace": nib.nifti1.xform_codes.label[coordsys.xformspace],
                "xform": coordsys.xform,
            },
        }


#**************************************************************************************************#
#                                         FreeSurferReader                                         #
#**************************************************************************************************#
#                                                                                                  #
# FreeSurfer files of a subject's surf/ and label/ folders. A surface (triangle or quad format;    #
# nibabel splits each quad into two triangles) gives vertices in tkr RAS mm and faces; scanner RAS #
# = tkr RAS + c_ras, which the volume info at the end of a triangle file states when it is valid.  #
# A morphometry file (curv, sulc, thickness, area, ...) gives one value per vertex, an annotation  #
# per-vertex indices into its colour table (-1 where a vertex has no entry; colours as RGBA from 0 #
# to 1, alpha = 1 - transparency / 255), a label file (text) the vertices it holds with their      #
# coordinates (tkr or scanner RAS, as its first line states), indices and values.                  #
#                                                                                                  #
#**************************************************************************************************#
class FreeSurferReader(Reader):
    FORMATS = (Format.FREESURFER_SURFACE, Format.FREESURFER_MORPH, Format.FREESURFER_ANNOT,
               Format.FREESURFER_LABEL)
    PACKAGES = ("miox", "nibabel")
    TKR = "FreeSurfer tkr RAS (mm)"
    SCANNER = "FreeSurfer scanner RAS (mm)"
    TRIANGLE_MAGIC = b"\xff\xff\xfe"

    def read(self, sniff: Sniff) -> list[Surface]:
        path, name = sniff.path, sniff.path.name
        notes: list[str] = []
        try:
            with self.captured(notes):
                if sniff.format == Format.FREESURFER_SURFACE:
                    return [self._surface(sniff, notes)]
                if sniff.format == Format.FREESURFER_MORPH:
                    values = nib.freesurfer.read_morph_data(path)
                    return [Surface(None, None, self.TKR, {self._measure(name): values}, {},
                                    Meta({}), self.provenance(sniff, tuple(notes)), name)]
                if sniff.format == Format.FREESURFER_ANNOT:
                    return [self._annotation(sniff, notes)]
                return [self._label(sniff, notes)]
        except (ValueError, OSError, IndexError) as e:
            raise ReaderError(f"{path}: not a readable FreeSurfer file ({e})") from e

    def _surface(self, sniff: Sniff, notes: list[str]) -> Surface:
        with open(sniff.path, "rb") as fh:
            triangles = fh.read(3) == self.TRIANGLE_MAGIC
        vertices, faces, info, *created = nib.freesurfer.read_geometry(
            sniff.path, read_metadata=True, read_stamp=triangles)
        vendor = {"format": "triangle" if triangles else "quad",
                  "created": created[0] if created else None,
                  "volume_info": {k: self.plain(v) for k, v in info.items()}}
        meta = Meta(vendor)
        if str(info.get("valid", "")).split()[:1] == ["1"] and "cras" in info:
            meta.set("SurfaceCRAS", [float(v) for v in info["cras"]],
                     "FreeSurfer surface volume info: cras", "volume_info/cras")
        else:
            notes.append("no valid volume info: c_ras not stated")
        if not triangles:
            notes.append("quads split into two triangles each")
        return Surface(np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64),
                       self.TKR, {}, {}, meta, self.provenance(sniff, tuple(notes)),
                       sniff.path.name)

    def _annotation(self, sniff: Sniff, notes: list[str]) -> Surface:
        values, colours, names = nib.freesurfer.read_annot(sniff.path)
        table = {i: (n.decode() if isinstance(n, bytes) else str(n),
                     (*(colours[i, :3] / 255.0), 1.0 - colours[i, 3] / 255.0))
                 for i, n in enumerate(names)}
        labels = {"annotation": {"values": values, "table": table}}
        vendor = {"colortable": {"names": [entry[0] for entry in table.values()],
                                 "rgbt_id": self.plain(colours)}}
        return Surface(None, None, self.TKR, {}, labels, Meta(vendor),
                       self.provenance(sniff, tuple(notes)), sniff.path.name)

    def _label(self, sniff: Sniff, notes: list[str]) -> Surface:
        lines = sniff.path.read_text().splitlines()
        rows = np.array([[float(v) for v in line.split()] for line in lines[2:] if line.strip()])
        if rows.ndim != 2 or rows.shape[1] != 5:
            raise ReaderError(f"{sniff.path}: label rows of 5 numbers expected")
        count = int(lines[1].split()[0])
        if count != rows.shape[0]:
            notes.append(f"header states {count} vertices, {rows.shape[0]} rows found")
        space = self.SCANNER if "vox2ras=scanner" in lines[0].lower() else self.TKR
        vertex_data = {"vertex_index": rows[:, 0].astype(np.int64), "value": rows[:, 4]}
        meta = Meta({"comment": lines[0], "count": count})
        notes.append("vertices of the label only")
        return Surface(rows[:, 1:4], None, space, vertex_data, {}, meta,
                       self.provenance(sniff, tuple(notes)), sniff.path.name)

    @staticmethod
    def _measure(name: str) -> str:
        """The measure a morphometry file holds, from its name (lh.thickness -> thickness)."""
        return name.split(".", 1)[1] if "." in name else name


#**************************************************************************************************#
#                                            Mz3Reader                                             #
#**************************************************************************************************#
#                                                                                                  #
# MZ3 (Surfice, NiiVue), little-endian, often gzipped: magic 0x5A4D, attribute bits (1 faces, 2    #
# vertices, 4 RGBA colours, 8 float scalars, 16 double scalars), the face and vertex counts and a  #
# number of bytes to skip; then the faces (three int32 each), the vertices (three float32 each),   #
# the colours (four bytes each) and the scalars, one layer per vertex count. A file of scalars     #
# only belongs to a mesh read elsewhere. The file states no coordinate frame (NiiVue shows the     #
# vertices as world mm).                                                                           #
#                                                                                                  #
#**************************************************************************************************#
class Mz3Reader(Reader):
    FORMATS = (Format.MZ3,)
    PACKAGES = ("miox",)
    MAGIC = 0x5A4D
    SPACE = "not stated (MZ3; world mm in NiiVue)"

    def read(self, sniff: Sniff) -> list[Surface]:
        raw = sniff.path.read_bytes()
        try:
            raw = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
            magic, bits, n_faces, n_vertices, skip = struct.unpack_from("<HHIII", raw)
        except (OSError, EOFError, struct.error) as e:
            raise ReaderError(f"{sniff.path.name}: {type(e).__name__}: {e}") from e
        if magic != self.MAGIC:
            raise ReaderError(f"{sniff.path.name}: not MZ3 (magic {magic:#x})")
        at = 16 + skip
        faces = vertices = None
        vertex_data = {}
        try:
            if bits & 1:
                faces = np.frombuffer(raw, "<i4", 3 * n_faces, at).reshape(-1, 3).astype(np.int64)
                at += 12 * n_faces
            if bits & 2:
                vertices = np.frombuffer(raw, "<f4", 3 * n_vertices, at).reshape(-1, 3)
                vertices = vertices.astype(np.float64)
                at += 12 * n_vertices
            if bits & 4:
                vertex_data["rgba"] = np.frombuffer(raw, np.uint8, 4 * n_vertices, at).reshape(
                    -1, 4)
                at += 4 * n_vertices
            for bit, dtype in ((8, "<f4"), (16, "<f8")):
                if bits & bit and n_vertices:
                    size = np.dtype(dtype).itemsize * n_vertices
                    for layer in range((len(raw) - at) // size):
                        vertex_data[f"scalar{layer + 1}" if layer else "scalar"] = np.frombuffer(
                            raw, dtype, n_vertices, at).astype(np.float64)
                        at += size
        except ValueError as e:
            raise ReaderError(f"{sniff.path.name}: file shorter than its header states") from e
        vendor = {"attributes": bits, "faces": n_faces, "vertices": n_vertices, "skip": skip}
        return [Surface(vertices, faces, self.SPACE, vertex_data, {}, Meta(vendor),
                        self.provenance(sniff), sniff.path.stem)]
