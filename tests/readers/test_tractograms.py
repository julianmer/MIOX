####################################################################################################
#                                      test_tractograms.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 12/10/26                                                                                #
#                                                                                                  #
# Purpose: Tractogram readers on synthetic files: TRK with scalars and properties, TCK, TRX with   #
#          groups and per-group data (also with the older offsets lacking the closing value),      #
#          VTK legacy polydata in ASCII and binary, version 5 offsets, polygons as a surface.      #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import json
import zipfile

import nibabel as nib
import numpy as np
import pytest

# own
import miox

LINES = [np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32),
         np.array([[0, 1, 0], [0, 2, 0]], dtype=np.float32)]


def test_trk_with_scalars_and_properties(tmp_path):
    path = tmp_path / "a.trk"
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    tractogram = nib.streamlines.Tractogram(
        LINES, data_per_point={"fa": [np.ones((3, 1)), np.zeros((2, 1))]},
        data_per_streamline={"id": np.array([[7.0], [9.0]])}, affine_to_rasmm=np.eye(4))
    header = {"dimensions": (4, 4, 4), "voxel_sizes": (2.0, 2.0, 2.0), "voxel_to_rasmm": affine}
    nib.streamlines.save(tractogram, path, header=header)
    sniff = miox.sniff(path)
    assert (sniff.format, sniff.kind) == (miox.Format.TRK, miox.Kind.TRACTOGRAM)
    [t] = miox.load(path)
    assert np.allclose(t.points, np.concatenate(LINES), atol=1e-5)
    assert np.array_equal(t.offsets, [0, 3, 5])
    assert np.array_equal(t.point_data["fa"].ravel(), [1, 1, 1, 0, 0])
    assert np.array_equal(t.streamline_data["id"].ravel(), [7, 9])
    assert np.allclose(t.reference["voxel_to_rasmm"], affine)


def test_tck(tmp_path):
    path = tmp_path / "a.tck"
    nib.streamlines.save(nib.streamlines.Tractogram(LINES, affine_to_rasmm=np.eye(4)), path)
    assert miox.sniff(path).format == miox.Format.TCK
    [t] = miox.load(path)
    assert np.allclose(t.streamline(1), LINES[1]) and t.reference is None


def write_trx(path, offsets):
    points = np.concatenate(LINES)
    header = {"DIMENSIONS": [4, 4, 4], "VOXEL_TO_RASMM": np.eye(4).tolist(), "NB_VERTICES": 5,
              "NB_STREAMLINES": 2}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("header.json", json.dumps(header))
        archive.writestr("positions.3.float16", points.astype("<f2").tobytes())
        archive.writestr("offsets.uint64", np.array(offsets, "<u8").tobytes())
        archive.writestr("dpv/fa.float32", np.arange(5, dtype="<f4").tobytes())
        archive.writestr("dps/colour.3.uint8", np.arange(6, dtype="u1").tobytes())
        archive.writestr("groups/cst.uint32", np.array([1], "<u4").tobytes())
        archive.writestr("dpg/cst/volume.float32", np.array([3.5], "<f4").tobytes())


@pytest.mark.parametrize("offsets", [[0, 3, 5], [0, 3]])
def test_trx_zip(tmp_path, offsets):
    path = tmp_path / "a.trx"
    write_trx(path, offsets)
    sniff = miox.sniff(path)
    assert (sniff.format, sniff.kind) == (miox.Format.TRX, miox.Kind.TRACTOGRAM)
    [t] = miox.load(path)
    assert np.allclose(t.points, np.concatenate(LINES)) and np.array_equal(t.offsets, [0, 3, 5])
    assert np.array_equal(t.point_data["fa"], np.arange(5))
    assert t.streamline_data["colour"].shape == (2, 3)
    assert np.array_equal(t.streamline_data["group/cst"], [False, True])
    assert t.meta.vendor["dpg"] == {"cst": {"volume": [3.5]}}


def test_trx_folder(tmp_path):
    archive = tmp_path / "a.trx"
    write_trx(archive, [0, 3, 5])
    folder = tmp_path / "b.trx"
    zipfile.ZipFile(archive).extractall(folder)
    assert miox.sniff(folder).format == miox.Format.TRX
    assert miox.load(folder)[0].n_streamlines == 2


VTK_ASCII = """# vtk DataFile Version 3.0
lines
ASCII
DATASET POLYDATA
POINTS 5 float
0 0 0 1 0 0 2 0 0
0 1 0 0 2 0
LINES 2 7
3 0 1 2
2 3 4
POINT_DATA 5
SCALARS fa float 1
LOOKUP_TABLE default
0.1 0.2 0.3 0.4 0.5
CELL_DATA 2
FIELD FieldData 1
length 1 2 double
2.0 1.0
"""


def test_vtk_ascii_lines(tmp_path):
    path = tmp_path / "a.vtk"
    path.write_text(VTK_ASCII)
    sniff = miox.sniff(path)
    assert (sniff.format, sniff.kind) == (miox.Format.VTK_POLYDATA, miox.Kind.TRACTOGRAM)
    [t] = miox.load(path)
    assert np.allclose(t.points, np.concatenate(LINES)) and np.array_equal(t.offsets, [0, 3, 5])
    assert np.allclose(t.point_data["fa"], [0.1, 0.2, 0.3, 0.4, 0.5])
    assert np.array_equal(t.streamline_data["length"], [2.0, 1.0])
    assert t.space == "not stated (VTK)"


def test_vtk_binary_version_5(tmp_path):
    """Version 5 cells as OFFSETS and CONNECTIVITY; lines in reverse point order."""
    points = np.concatenate(LINES)[::-1]
    path = tmp_path / "b.vtk"
    path.write_bytes(
        b"# vtk DataFile Version 5.1\nlines\nBINARY\nDATASET POLYDATA\nPOINTS 5 float\n"
        + points.astype(">f4").tobytes() + b"\nLINES 3 5\nOFFSETS vtktypeint64\n"
        + np.array([0, 3, 5], ">i8").tobytes() + b"\nCONNECTIVITY vtktypeint64\n"
        + np.array([4, 3, 2, 1, 0], ">i8").tobytes() + b"\n")
    [t] = miox.load(path)
    assert np.allclose(t.points, np.concatenate(LINES)) and np.array_equal(t.offsets, [0, 3, 5])


def test_vtk_polygons_are_a_surface(tmp_path):
    path = tmp_path / "mesh.vtk"
    path.write_text("# vtk DataFile Version 3.0\nmesh\nASCII\nDATASET POLYDATA\n"
                    "POINTS 4 float\n0 0 0 1 0 0 0 1 0 0 0 1\n"
                    "POLYGONS 2 8\n3 0 1 2\n3 0 1 3\n")
    sniff = miox.sniff(path)
    assert (sniff.format, sniff.kind) == (miox.Format.VTK_POLYDATA, miox.Kind.SURFACE)
    [s] = miox.load(path)
    assert isinstance(s, miox.Surface) and np.array_equal(s.faces, [[0, 1, 2], [0, 1, 3]])
