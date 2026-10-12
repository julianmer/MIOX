####################################################################################################
#                                          test_vista.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 13/10/26                                                                                #
#                                                                                                  #
# Purpose: Vista images of lipsia (Apache-2.0) against the LAYNII NIfTI files they were imported   #
#          from (BSD-3): the same voxels and affine, from an sform (ana_Ding) and from a qform     #
#          (rim_M); the bit mask covers the cortical rim it was drawn on.                          #
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
import miox
from tests.conftest import PUBLIC_DATA

LIPSIA = PUBLIC_DATA / "lipsia" / "data"
LAYNII = PUBLIC_DATA / "laynii" / "test_data"


@pytest.mark.parametrize("vista, nifti", [
    ("ana_Ding.v", "Ding2016_occip_T2starweighted_filtered_for_tests.nii.gz"),  # sform
    ("rim_M.v", "rim_M.nii.gz"),  # qform
])
def test_vista_equals_its_nifti_source(vista, nifti):
    if not (LIPSIA / vista).exists() or not (LAYNII / nifti).exists():
        pytest.skip("lipsia or LAYNII data not fetched")
    image = miox.load(LIPSIA / vista)[0]
    source = nib.load(LAYNII / nifti)
    assert np.array_equal(image.data, np.asanyarray(source.dataobj).astype(image.data.dtype))
    assert np.allclose(image.affine, source.affine, atol=1e-4)


def test_bit_mask_covers_the_rim():
    if not (LIPSIA / "mask_Ding.v").exists():
        pytest.skip("lipsia data not fetched")
    mask = miox.load(LIPSIA / "mask_Ding.v")[0].data
    rim = miox.load(LIPSIA / "rim_Ding.v")[0].data
    assert mask.shape == rim.shape and set(np.unique(mask)) == {0, 1}
    assert np.all(mask[rim > 0] == 1)
