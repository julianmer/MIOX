####################################################################################################
#                                        test_surfaces.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 12/10/26                                                                                #
#                                                                                                  #
# Purpose: Surfaces against geometry and each other: fsaverage meshes are closed spheres (163842   #
#          vertices, Euler characteristic 2), its per-vertex files match the mesh, its V1 label    #
#          lies in the occipital parcels of its annotation, GIFTI and FreeSurfer colour tables of  #
#          aparc agree, one GIFTI mesh in ASCII and base64 encodings is equal, an external GIFTI   #
#          array equals the bytes of its .dat file, and compressed copies equal the original;      #
#          NiiVue's MZ3 meshes are closed.                                                         #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import numpy as np
import pytest

# own
from miox import load
from tests.conftest import PUBLIC_DATA

FSAVERAGE = PUBLIC_DATA / "nitest_freesurfer/fsaverage"
GIFTI = PUBLIC_DATA / "nibabel/nibabel/gifti/tests/data"
OCCIPITAL = ("pericalcarine", "lingual", "lateraloccipital", "cuneus")


def needed(*paths):
    for path in paths:
        if not path.is_file():
            pytest.skip(f"{path.name} not fetched")


def euler(surface) -> int:
    """V - E + F of a triangle mesh."""
    faces = surface.faces
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    return surface.n_vertices - len(np.unique(edges, axis=0)) + len(faces)


@pytest.mark.parametrize("name", ["white", "pial", "inflated"])
def test_fsaverage_meshes_are_closed_spheres(name):
    """fsaverage is an icosahedron subdivided 7 times: 10 * 4^7 + 2 vertices, twice as many
    faces less 4, and a closed surface of genus 0 (Euler characteristic 2)."""
    path = FSAVERAGE / f"surf/lh.{name}"
    needed(path)
    [surface] = load(path)
    assert (surface.n_vertices, surface.faces.shape[0]) == (163842, 327680)
    assert euler(surface) == 2


@pytest.mark.parametrize("name", ["surf/lh.curv", "surf/lh.sulc", "surf/lh.thickness",
                                  "label/lh.aparc.annot"])
def test_fsaverage_vertex_files_match_the_mesh(name):
    path = FSAVERAGE / name
    needed(path, FSAVERAGE / "surf/lh.white")
    assert load(path)[0].n_vertices == load(FSAVERAGE / "surf/lh.white")[0].n_vertices


def test_v1_label_lies_in_occipital_parcels():
    """The V1 label's vertex indices fall into the occipital parcels of the aparc annotation
    (97.5 %: pericalcarine, lingual, lateraloccipital, cuneus). Its coordinates are those of the
    subject it was mapped from (two indices share one point), not fsaverage's."""
    label, annot = FSAVERAGE / "label/lh.V1.label", FSAVERAGE / "label/lh.aparc.annot"
    needed(label, annot)
    [v1], [parcels] = load(label), load(annot)
    annotation = parcels.labels["annotation"]
    values = annotation["values"][v1.vertex_data["vertex_index"]]
    names = [annotation["table"][v][0] for v in values]
    assert v1.n_vertices == 4641
    assert np.mean([n in OCCIPITAL for n in names]) > 0.97


def test_aparc_colour_tables_agree():
    """rh.aparc.annot.gii (GIFTI, RGBA 0..1) and fsaverage's lh.aparc.annot (FreeSurfer, 0..255
    with transparency) name the same 36 parcels; 34 have equal colours and GIFTI keys equal
    FreeSurfer's annotation ids (R + 256 G + 65536 B). The GIFTI writer made 'unknown'
    transparent black; 'temporalpole' has another colour in fsaverage's older table."""
    gifti, annot = GIFTI / "rh.aparc.annot.gii", FSAVERAGE / "label/lh.aparc.annot"
    needed(gifti, annot)
    table = load(gifti)[0].labels["node label"]["table"]
    freesurfer = dict(load(annot)[0].labels["annotation"]["table"].values())
    assert {name for name, _ in table.values()} == set(freesurfer)
    differ = {name for name, rgba in table.values()
              if not np.allclose(rgba, freesurfer[name], atol=1e-5)}
    assert differ == {"unknown", "temporalpole"}
    for key, (name, rgba) in table.items():
        if name != "unknown":
            r, g, b = np.round(np.array(rgba[:3]) * 255).astype(int)
            assert key == r + 256 * g + 65536 * b


def test_gifti_ascii_equals_base64():
    """nibabel's ascii_flat_data.gii and base64bin.gii hold one mesh in two encodings."""
    ascii, base64 = GIFTI / "ascii_flat_data.gii", GIFTI / "base64bin.gii"
    needed(ascii, base64)
    [a], [b] = load(ascii), load(base64)
    assert np.allclose(a.vertices, b.vertices) and np.array_equal(a.faces, b.faces)


def test_gifti_external_array_equals_its_bytes():
    """external.gii stores a cube (8 vertices, 12 triangles, Euler characteristic 2) in
    external.dat: float32 vertices at offset 0, int32 faces right after, little endian."""
    path, dat = GIFTI / "external.gii", GIFTI / "external.dat"
    needed(path, dat)
    [cube] = load(path)
    raw = dat.read_bytes()
    assert np.array_equal(cube.vertices, np.frombuffer(raw, "<f4", 24).reshape(8, 3))
    assert np.array_equal(cube.faces, np.frombuffer(raw, "<i4", 36, offset=96).reshape(12, 3))
    assert euler(cube) == 2


@pytest.mark.parametrize("suffix", [".gz", ".bz2"])
def test_gifti_compressed_equals_original(suffix):
    path = GIFTI / "external.gii"
    needed(path, path.with_name(path.name + suffix))
    [plain], [packed] = load(path), load(path.with_name(path.name + suffix))
    assert np.array_equal(plain.vertices, packed.vertices)
    assert np.array_equal(plain.faces, packed.faces)


MZ3 = PUBLIC_DATA / "niivue" / "packages" / "niivue" / "demos" / "images" / "mz3"


@pytest.mark.parametrize("name, closed, scalars", [
    ("3Mesh.mz3", 1, None), ("11ScalarMesh.mz3", 1, [1, 2, 3, 4, 5, 6]),
    ("cortex_5124.mz3", 2, None),
])
def test_mz3_meshes_are_closed(name, closed, scalars):
    """NiiVue's MZ3 demos: an octahedron (with one scalar per vertex) and two cortical
    hemispheres, each a closed surface of genus 0 (Euler characteristic 2 apiece)."""
    if not (MZ3 / name).exists():
        pytest.skip("niivue demos not fetched")
    surface = load(MZ3 / name)[0]
    assert euler(surface) == 2 * closed
    if scalars is not None:
        assert surface.vertex_data["scalar"].tolist() == scalars
