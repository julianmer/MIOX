####################################################################################################
#                                          test_images.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Images: the same MINC data as MINC 1 and MINC 2 read alike; PAR/REC spacing equals the  #
#          voxel size of the affine; a dcm2niix sidecar's values land in the standard keys.        #
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
