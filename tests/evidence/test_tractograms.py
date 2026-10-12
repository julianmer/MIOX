####################################################################################################
#                                      test_tractograms.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 12/10/26                                                                                #
#                                                                                                  #
# Purpose: trx-test-data's gold standard: the same 13 streamlines as .trk, .tck, .trx (zip and     #
#          folder) and .vtk equal the published RAS mm points (the .vtk in LPS), their per-point   #
#          and per-streamline data equal the published JSON, and the stated reference maps the     #
#          published voxel coordinates to the RAS points and equals the reference image's affine.  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pytest

# own
from miox import load
from tests.conftest import PUBLIC_DATA

DATA = PUBLIC_DATA / "trx_test_data"
LPS = np.array([-1.0, -1.0, 1.0])


def needed(*paths):
    for path in paths:
        if not path.exists():
            pytest.skip(f"{path.name} not fetched")


@pytest.mark.parametrize("name", ["gs.trk", "gs.tck", "gs.trx", "gs_fldr.trx", "gs.vtk"])
def test_points_equal_the_gold_standard(name):
    path, gold = DATA / name, DATA / "gs_rasmm_space.txt"
    needed(path, gold)
    [t] = load(path)
    points = t.points * LPS if name.endswith(".vtk") else t.points
    assert t.n_streamlines == 13
    assert np.array_equal(t.offsets, np.arange(0, 105, 8))
    assert np.allclose(points, np.loadtxt(gold), atol=1e-5)


@pytest.mark.parametrize("name", ["gs.trk", "gs.trx", "gs_fldr.trx"])
def test_point_and_streamline_data_equal_the_gold_standard(name):
    path = DATA / name
    needed(path, DATA / "points_data.txt", DATA / "streamlines_data.txt")
    [t] = load(path)
    for key, values in json.loads((DATA / "points_data.txt").read_text()).items():
        assert np.array_equal(t.point_data[key].ravel(), np.concatenate(values).ravel())
    random = json.loads((DATA / "streamlines_data.txt").read_text())["random_coord"]
    assert np.array_equal(t.streamline_data["random_coord"], random)


@pytest.mark.parametrize("name", ["gs.trk", "gs.trx"])
def test_reference_maps_voxels_to_ras(name):
    """The stated voxel-to-RAS affine takes the published voxel coordinates to the published RAS
    points and equals the affine of the reference image gs.nii."""
    path = DATA / name
    needed(path, DATA / "gs_vox_space.txt", DATA / "gs.nii")
    [t] = load(path)
    affine = np.array(t.reference["voxel_to_rasmm"])
    voxels = np.loadtxt(DATA / "gs_vox_space.txt")
    assert np.allclose(voxels @ affine[:3, :3].T + affine[:3, 3], t.points, atol=1e-5)
    assert np.allclose(affine, nib.load(DATA / "gs.nii").affine, atol=1e-6)
    assert list(t.reference["dimensions"]) == [5, 10, 20]
