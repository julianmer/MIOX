####################################################################################################
#                                          test_images.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Images: the same MINC data as MINC 1 and MINC 2 read alike; PAR/REC spacing equals the  #
#          voxel size of the affine; a dcm2niix sidecar's values land in the standard keys;        #
#          ParaVision 6.0.1 reconstructions as brkraw converts them.                               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import json

import numpy as np
import pytest

# own
from miox import MRImage, load
from tests.conftest import PUBLIC_DATA

NIBABEL = "nibabel/nibabel/tests/data/"


def image(relpath: str) -> MRImage:
    path = PUBLIC_DATA / relpath
    if not path.exists():
        pytest.skip("file not fetched")
    [im] = load(path)
    return im


@pytest.mark.data
@pytest.mark.parametrize("name", ["_4d", "_1_scale", "-no-att"])
def test_minc1_and_minc2_agree(name):
    a, b = image(f"{NIBABEL}minc1{name}.mnc"), image(f"{NIBABEL}minc2{name}.mnc")
    assert a.shape == b.shape and a.dims == b.dims
    assert np.allclose(a.data, b.data) and np.allclose(a.affine, b.affine)


@pytest.mark.data
def test_minc_time_axis_last():
    im = image(f"{NIBABEL}minc1_4d.mnc")
    assert im.dims == ("i", "j", "k", "time") and im.shape[3] == 2


@pytest.mark.data
def test_parrec_slice_spacing_is_the_affine_voxel_size():
    im = image(f"{NIBABEL}phantom_EPI_asc_CLEAR_2_1.PAR")
    assert im.voxel_sizes[2] == pytest.approx(im.meta["SpacingBetweenSlices"])
    assert im.meta["PatientPosition"] == "HFS" and im.meta["RepetitionTime"] == 2.0


@pytest.mark.data
def test_dcm2niix_sidecar_values():
    im = image("dcm_qa_nih/Ref/DCM2NIIX_regression_test_4.nii")
    sidecar = PUBLIC_DATA / "dcm_qa_nih/Ref/DCM2NIIX_regression_test_4.json"
    sidecar = json.loads(sidecar.read_text())
    assert im.meta["EchoTime"] == sidecar["EchoTime"]
    assert im.meta["ExcitationFlipAngle"] == sidecar["FlipAngle"]
    assert im.meta["SpectrometerFrequency"] == [sidecar["ImagingFrequency"]]


BRKRAW = PUBLIC_DATA / "brkraw_dataset/PV6.0.1/UNC_PV6.0.1_FLASH_TurboRARE_EPI"
# (scan, slice pack, voxel sum, affine rows) as brkraw 0.5.7 converts pdata/1 ("scanner" space,
# x negated: brkraw's scanner frame has the opposite x to MIOX's RAS)
BRKRAW_SCANS = [
    ("3", 0, 79404.7, [
        [-0.1172, 0, 0, 15],
        [0, -0.1172, 0, 10],
        [0, 0, 4, 0]]),
    ("3", 1, 113432.09, [
        [0, 0, 4, 0],
        [-0.1172, 0, 0, 10],
        [0, -0.1172, 0, 15]]),
    ("3", 2, 103749.79, [
        [-0.1172, 0, 0, 15],
        [0, 0, -4, -5],
        [0, -0.1172, 0, 15]]),
    ("6", 0, 3178187.38, [
        [-0.075, 0, 0, 14.4],
        [0, -0.075, 0, 9.4],
        [0, 0, 1, -5.5]]),
    ("7", 0, 9874585.13, [
        [-0.075, 0, 0, 14.4],
        [0, -0.075, 0, 9.4],
        [0, 0, 1, -5.5]]),
    ("10", 0, 50345.87, [
        [0, 0, 0.9062, -28.5469],
        [-0.9062, 0, 0, 29],
        [0, -0.9062, 0, 29]]),
    ("11", 0, 709514.64, [
        [-0.36, 0, 0, 14.4],
        [0, -0.36, 0, 9.4],
        [0, 0, 1, -5.5]]),
]


@pytest.mark.data
@pytest.mark.parametrize("scan,pack,total,rows", BRKRAW_SCANS)
def test_paravision_reconstruction_as_brkraw_converts_it(scan, pack, total, rows):
    """The same voxels and geometry as brkraw's NIfTI, read from the scan folder (no raw data in
    this pruned study). A single-slice pack has no slice spacing: brkraw gives it 4 mm, MIOX the
    stated VisuCoreFrameThickness (2 mm), so there only the slice direction is compared."""
    folder = BRKRAW / "UNC_PV6.0.1_FLASH_TurboRARE_EPI" / scan
    if not folder.is_dir():
        pytest.skip("brkraw-dataset not fetched")
    image = load(folder)[pack]
    assert float(image.data.astype(float).sum()) == pytest.approx(total, rel=1e-6)
    expected, affine = np.asarray(rows), image.affine[:3]
    columns = [0, 1, 3] if image.shape[2] == 1 else [0, 1, 2, 3]
    assert np.allclose(affine[:, columns], expected[:, columns], atol=1e-3)
    unit = expected[:, 2] / np.linalg.norm(expected[:, 2])
    assert np.allclose(affine[:, 2] / np.linalg.norm(affine[:, 2]), unit, atol=1e-6)
