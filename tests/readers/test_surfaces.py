####################################################################################################
#                                        test_surfaces.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 12/10/26                                                                                #
#                                                                                                  #
# Purpose: Surface readers on synthetic files: a GIFTI mesh with shape and label arrays (also      #
#          gzipped), FreeSurfer surfaces with and without valid volume info, curv, annotation and  #
#          label files, and files that only look like FreeSurfer files.                            #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import gzip

import nibabel as nib
import numpy as np

# own
import miox

VERTICES = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
FACES = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)


def gifti_tetrahedron() -> nib.gifti.GiftiImage:
    table = nib.gifti.GiftiLabelTable()
    for key, name, rgba in ((0, "outside", (0, 0, 0, 0)), (1, "inside", (1, 0, 0, 1))):
        label = nib.gifti.GiftiLabel(key, *rgba)
        label.label = name
        table.labels.append(label)
    arrays = [
        nib.gifti.GiftiDataArray(VERTICES, "NIFTI_INTENT_POINTSET",
                                 coordsys=nib.gifti.GiftiCoordSystem(3, 3)),
        nib.gifti.GiftiDataArray(FACES, "NIFTI_INTENT_TRIANGLE"),
        nib.gifti.GiftiDataArray(np.arange(4, dtype=np.float32), "NIFTI_INTENT_SHAPE",
                                 meta=nib.gifti.GiftiMetaData(Name="depth")),
        nib.gifti.GiftiDataArray(np.array([0, 1, 1, 0], dtype=np.int32), "NIFTI_INTENT_LABEL",
                                 meta=nib.gifti.GiftiMetaData(Name="parcels")),
    ]
    return nib.gifti.GiftiImage(labeltable=table, darrays=arrays)


def test_gifti_mesh_with_data_and_labels(tmp_path):
    path = tmp_path / "mesh.surf.gii"
    nib.save(gifti_tetrahedron(), path)
    sniff = miox.sniff(path)
    assert (sniff.format, sniff.kind) == (miox.Format.GIFTI, miox.Kind.SURFACE)
    [s] = miox.load(path)
    assert np.array_equal(s.vertices, VERTICES) and np.array_equal(s.faces, FACES)
    assert s.space == "talairach (GIFTI, mm)" and s.name == "mesh.surf"
    assert np.array_equal(s.vertex_data["depth"], np.arange(4))
    assert np.array_equal(s.labels["parcels"]["values"], [0, 1, 1, 0])
    assert s.labels["parcels"]["table"][1] == ("inside", (1.0, 0.0, 0.0, 1.0))
    assert s.meta.vendor["arrays"][0]["encoding"] == "GZipBase64Binary"


def test_gifti_gzipped(tmp_path):
    nib.save(gifti_tetrahedron(), tmp_path / "mesh.gii")
    path = tmp_path / "mesh.gii.gz"
    path.write_bytes(gzip.compress((tmp_path / "mesh.gii").read_bytes()))
    assert miox.sniff(path).format == miox.Format.GIFTI
    [s] = miox.load(path)
    assert s.name == "mesh" and np.array_equal(s.faces, FACES)


def test_freesurfer_surface_with_volume_info(tmp_path):
    path = tmp_path / "lh.white"
    info = {"head": np.array([2, 0, 20], dtype=np.int32), "valid": "1  # volume info valid",
            "filename": "orig.mgz", "volume": np.array([256, 256, 256]),
            "voxelsize": np.array([1.0, 1.0, 1.0]), "xras": np.array([-1.0, 0.0, 0.0]),
            "yras": np.array([0.0, 0.0, -1.0]), "zras": np.array([0.0, 1.0, 0.0]),
            "cras": np.array([1.5, -2.0, 3.0])}
    nib.freesurfer.write_geometry(path, VERTICES, FACES, volume_info=info)
    assert miox.sniff(path).format == miox.Format.FREESURFER_SURFACE
    [s] = miox.load(path)
    assert np.allclose(s.vertices, VERTICES) and np.array_equal(s.faces, FACES)
    assert s.meta["SurfaceCRAS"] == [1.5, -2.0, 3.0]
    assert s.space == "FreeSurfer tkr RAS (mm)"


def test_freesurfer_surface_without_volume_info(tmp_path):
    path = tmp_path / "lh.sphere"
    nib.freesurfer.write_geometry(path, VERTICES, FACES)
    [s] = miox.load(path)
    assert "SurfaceCRAS" not in s.meta
    assert "no valid volume info: c_ras not stated" in s.provenance.notes


def test_freesurfer_curv_and_annotation(tmp_path):
    curv, annot = tmp_path / "lh.thickness", tmp_path / "lh.parc.annot"
    nib.freesurfer.write_morph_data(curv, np.array([1.0, 2.0, 2.5, 3.0]))
    colours = np.array([[25, 5, 25, 0], [220, 20, 10, 255]])
    nib.freesurfer.write_annot(annot, np.array([0, 1, 1, -1]), colours, [b"unknown", b"spot"])
    assert miox.sniff(curv).format == miox.Format.FREESURFER_MORPH
    assert miox.sniff(annot).format == miox.Format.FREESURFER_ANNOT
    assert np.allclose(miox.load(curv)[0].vertex_data["thickness"], [1.0, 2.0, 2.5, 3.0])
    annotation = miox.load(annot)[0].labels["annotation"]
    assert np.array_equal(annotation["values"], [0, 1, 1, -1])
    assert annotation["table"][0] == ("unknown", (25 / 255, 5 / 255, 25 / 255, 1.0))
    assert annotation["table"][1][1][3] == 0.0


def test_freesurfer_label(tmp_path):
    path = tmp_path / "lh.spot.label"
    path.write_text("#!ascii label  , from subject s1 vox2ras=scanner\n2\n"
                    "3  0.000  1.000  0.000 0.500000\n1  1.000  0.000  0.000 0.250000\n")
    assert miox.sniff(path).format == miox.Format.FREESURFER_LABEL
    [s] = miox.load(path)
    assert np.array_equal(s.vertex_data["vertex_index"], [3, 1])
    assert np.allclose(s.vertices, [[0, 1, 0], [1, 0, 0]])
    assert s.space == "FreeSurfer scanner RAS (mm)"


def test_lookalikes_are_not_freesurfer(tmp_path):
    """A triangle magic whose counts do not fit the size, a curv magic with the wrong size, an
    .annot too small for its count."""
    short = tmp_path / "short"
    short.write_bytes(b"\xff\xff\xfecreated by x\n\n" + (100).to_bytes(4, "big") * 2 + bytes(40))
    curv = tmp_path / "data.bin"
    curv.write_bytes(b"\xff\xff\xff" + (1000).to_bytes(4, "big") + bytes(20))
    annot = tmp_path / "x.annot"
    annot.write_bytes((1000).to_bytes(4, "big") + bytes(16))
    for path in (short, curv, annot):
        assert miox.sniff(path).kind != miox.Kind.SURFACE
