####################################################################################################
#                                      test_brainvoyager.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 12/10/26                                                                                #
#                                                                                                  #
# Purpose: BrainVoyager files of bvbabel's test data against each other and against what they      #
#          state: the VMR axes equal the scanner directions of its PosInfos, the cube mesh         #
#          encloses the cube of its VMR, mesh time courses sample the VTC at the vertices, the     #
#          GLM's betas give the VMP's contrast t maps, the stored inverse X'X inverts the design,  #
#          per-vertex files match their meshes, smoothing lowers curvature spread, VOIs sit on     #
#          their VMR's grid, a protocol written with tabs equals its space-separated twin and the  #
#          rigid transformations are rotations. Turbo-BrainVoyager's sample: a V16 equals the VMR  #
#          made from it, and the first functional volume as STC and as V16 equals the Analyze     #
#          volume it was imported from.                                                            #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

# own
from miox import load
from tests.conftest import PUBLIC_DATA

DATA = PUBLIC_DATA / "bvbabel/test_data"
TBV = PUBLIC_DATA / "brainvoyager_tbv"  # Turbo-BrainVoyager's FacesHouses sample
LPS = np.array([-1.0, -1.0, 1.0])


def needed(*names):
    for name in names:
        if not (DATA / name).is_file():
            pytest.skip(f"{name} not fetched")
    return [load(DATA / name)[0] for name in names]


def voxel_size(vmr) -> np.ndarray:
    return np.array([vmr.meta.vendor[f"VoxelSize{a}"] for a in "XYZ"])


def test_vmr_axes_equal_its_scanner_directions():
    """sub-test01 states verified PosInfos (DICOM LPS): rows run along X, columns along Y and
    slices along Z; the affine's axes point the same ways."""
    [vmr] = needed("sub-test01_fileversion-2.vmr.gz")
    h = vmr.meta.vendor
    row = np.array([h[f"RowDir{a}"] for a in "XYZ"]) * LPS
    column = np.array([h[f"ColDir{a}"] for a in "XYZ"]) * LPS
    first = np.array([h[f"Slice1Center{a}"] for a in "XYZ"]) * LPS
    last = np.array([h[f"SliceNCenter{a}"] for a in "XYZ"]) * LPS
    axes = vmr.affine[:3, :3] / np.linalg.norm(vmr.affine[:3, :3], axis=0)
    assert h["PosInfosVerified"] == 1
    assert np.allclose(axes[:, 0], row) and np.allclose(axes[:, 1], column)
    assert np.allclose(axes[:, 2], (last - first) / np.linalg.norm(last - first))


def test_cube_mesh_encloses_the_cube_of_its_vmr():
    """The cube VMR holds a 12^3 block of 240; its mesh, back in VMR voxels, surrounds that
    block within one voxel on every side and shares its centre."""
    srf, vmr = needed("sub-test03_cube.srf.gz", "sub-test03_cube.vmr.gz")
    block = np.argwhere(vmr.data > 0)
    voxels = (128.0 - srf.vertices[:, [1, 2, 0]]) / voxel_size(vmr)
    assert len(block) == 12 ** 3
    assert np.all(voxels.min(axis=0) > block.min(axis=0) - 1)
    assert np.all(voxels.max(axis=0) < block.max(axis=0) + 1)
    assert np.allclose(voxels.mean(axis=0), block.mean(axis=0), atol=0.1)


def test_mesh_time_courses_sample_the_vtc_at_the_vertices():
    """The MTC was made from the VTC on the cube mesh: its values follow the VTC at the vertices
    (correlation > 0.8) better than one voxel inside or outside along the normals."""
    srf, mtc, vtc, vmr = needed("sub-test03_cube.srf.gz", "sub-test03_cube.mtc.gz",
                                "sub-test03.vtc.gz", "sub-test03_cube.vmr.gz")
    coordinates = 128.0 - srf.vertices[:, [1, 2, 0]]
    normals = -srf.vertex_data["normal"][:, [1, 2, 0]]
    values = mtc.vertex_data["time_course"][:, 0]
    correlation = {}
    for depth in (-1, 0, 1):
        index = np.rint((coordinates + depth * normals) / voxel_size(vmr)).astype(int)
        sampled = vtc.data[index[:, 0], index[:, 1], index[:, 2], 0]
        correlation[depth] = np.corrcoef(sampled, values)[0, 1]
    assert correlation[0] > 0.8
    assert correlation[0] > max(correlation[-1], correlation[1])


def test_glm_gives_the_vmp_contrasts():
    """The VMP holds t maps of five contrasts of the GLM's predictors. The stored inverse X'X is
    the inverse of the stored design matrix; per voxel, the five t values are c'b /
    sqrt(c' (X'X)^-1 c) times one factor (1 / residual SD) to 3 % (median; the GLM's AR(2)
    correction is not redone here), so betas, maps and voxels line up."""
    glm, vmp = needed("sub-test07_partial_coverage.glm.gz", "sub-test07_partial_coverage.vmp.gz")
    design = np.array(glm.meta.vendor["DesignMatrix"])
    inverse = np.array(glm.meta.vendor["InvertedXX"])
    assert np.allclose(inverse, np.linalg.inv(design.T @ design), rtol=1e-4, atol=1e-6)
    contrasts = np.array([[1, 1, 1, 0], [1, 1, -2, 0], [1, -1, 0, 0], [1, 0, -1, 0],
                          [0, 1, -1, 0]], dtype=float)
    assert glm.meta.vendor["Maps"][2:6] == [f"beta {n}" for n in
                                           ("Horizontal", "Vertical", "Baseline", "Constant")]
    betas = glm.data[..., 2:6]
    strong = np.argwhere((np.abs(vmp.data[..., 2]) > 2) & (np.abs(vmp.data[..., 3]) > 2))
    spread = []
    for i in strong[np.random.default_rng(0).choice(len(strong), 300, replace=False)]:
        i = tuple(i)
        unscaled = contrasts @ betas[i] / np.sqrt(np.einsum("ij,jk,ik->i", contrasts, inverse,
                                                            contrasts))
        factor = vmp.data[i] / unscaled
        spread.append(np.std(factor) / abs(np.mean(factor)))
    assert np.median(spread) < 0.03


@pytest.mark.parametrize("data, mesh, key", [
    ("sub-test02_left_hemisphere_4_curvature_maps.smp.gz", "sub-test02_left_hemisphere.srf.gz",
     None),
    ("sub-test09.ssm", "sub-test02_left_hemisphere.srf.gz", None),
    ("sub-test03_cube.mtc.gz", "sub-test03_cube.srf.gz", None),
    ("sub-test03_cube.poi", "sub-test03_cube.srf.gz", "NrOfMeshVertices"),
])
def test_vertex_files_match_their_meshes(data, mesh, key):
    values, surface = needed(data, mesh)
    assert values.n_vertices == surface.n_vertices
    if key:
        assert values.meta.vendor["header"][key] == surface.n_vertices


def test_smoothing_lowers_curvature_spread():
    """The four maps are curvature smoothed 5, 15, 35 and 70 times: their spread falls in that
    order."""
    [smp] = needed("sub-test02_left_hemisphere_4_curvature_maps.smp.gz")
    spreads = [np.std(v) for v in smp.vertex_data.values()]
    assert list(smp.vertex_data) == [f"Curvature, sm{n}" for n in (5, 15, 35, 70)]
    assert spreads == sorted(spreads, reverse=True)


def test_voi_lies_on_its_vmr_grid():
    """The VOI names the VMR's resolution and framing cube: its label grid is the VMR grid moved
    by whole voxels, and its voxels lie inside the VMR."""
    voi, vmr = needed("sub-test03.voi", "sub-test03.vmr.gz")
    shift = np.linalg.solve(vmr.affine, voi.affine)
    assert np.allclose(shift[:3, :3], np.eye(3), atol=1e-6)
    assert np.allclose(shift[:3, 3], np.rint(shift[:3, 3]), atol=1e-6)
    corners = shift[:3, 3] + np.array(voi.data.shape) - 1
    assert np.all(shift[:3, 3] >= 0) and np.all(corners < vmr.data.shape)
    assert [int((voi.data == n).sum()) for n in (1, 2, 3)] == voi.meta.vendor["voxel_counts"]


def test_protocol_with_tabs_equals_its_twin():
    tabs, spaces = needed("sub-test05_v3_tabs.prt", "sub-test05_v3_vols.prt")
    assert tabs.meta.vendor["conditions"] == spaces.meta.vendor["conditions"]
    assert tabs.meta["ProtocolTimeResolution"] == "Volumes"


@pytest.mark.parametrize("name", ["sub-test06_fileversion-5_IA.trf",
                                  "sub-test06_fileversion-7_extravmrtrf_IA.trf",
                                  "sub-test06_fileversion-8_IA.trf",
                                  "sub-test06_fileversion-8_ToMSP.trf",
                                  "sub-test06_fileversion-8_aACPC.trf"])
def test_rigid_transformations_are_rotations(name):
    path = DATA / "trf" / name
    if not path.is_file():
        pytest.skip(f"{name} not fetched")
    for transform in load(path):
        rotation = transform.matrix[:3, :3]
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-3)
        assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3)


def test_v16_equals_the_vmr_made_from_it():
    """NK1.vmr was made from NK1.v16 without rescaling (both hold 0-225): same voxels, same
    axes, and the V16's framing cube is the VMR's 256."""
    if not (TBV / "SubjectNK_PreparedAnatomy/NK1.v16").is_file():
        pytest.skip("brainvoyager_tbv not fetched")
    [v16] = load(TBV / "SubjectNK_PreparedAnatomy/NK1.v16")
    [vmr] = load(TBV / "SubjectNK_PreparedAnatomy/NK1.vmr")
    assert v16.data.dtype == np.uint16 and v16.shape == vmr.shape == (256, 256, 176)
    assert np.array_equal(v16.data, vmr.data)
    assert np.allclose(v16.affine, vmr.affine)


def test_first_functional_volume_as_stc_and_v16_equals_its_analyze_source():
    """FacesHouses_firstvol.fmr names its STC (one volume) and the Analyze volume it was imported
    from; the STC read through the FMR or the STC, and untitled.v16 (the same volume as V16),
    equal the Analyze voxels (x, y, slice) exactly, with the FMR's voxel sizes and TR."""
    if not (TBV / "Ser0004/FacesHouses_firstvol.stc").is_file():
        pytest.skip("brainvoyager_tbv not fetched")
    analyze = np.asanyarray(nib.load(TBV / "Ser0004/Analyze00001.hdr").dataobj)[..., 0]
    [by_fmr] = load(TBV / "Ser0004/FacesHouses_firstvol.fmr")
    [by_stc] = load(TBV / "Ser0004/FacesHouses_firstvol.stc")
    [v16] = load(TBV / "Ser0004/untitled.v16")
    assert by_fmr.meta.vendor["FirstDataSourceFile"] == "Analyze00001.img"
    assert by_fmr.shape == (64, 64, 30, 1) and by_fmr.dims == ("i", "j", "k", "time")
    assert np.array_equal(by_fmr.data[..., 0], analyze)
    assert np.array_equal(by_stc.data, by_fmr.data) and np.array_equal(by_stc.affine, by_fmr.affine)
    assert np.array_equal(v16.data, analyze)
    assert by_fmr.voxel_sizes == pytest.approx((3.5, 3.5, 3.99))
    assert by_fmr.meta["RepetitionTime"] == 3.0
    assert any("not verified" in note for note in by_fmr.provenance.notes)
