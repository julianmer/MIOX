####################################################################################################
#                                     test_i2nii_images.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 13/10/26                                                                                #
#                                                                                                  #
# Purpose: One GE scan converted to many formats (i2nii test data, BSD-2): the Stimulate and       #
#          SIVIC IDF files put every voxel at the world position the MetaImage and MGH files put   #
#          it (their grids differ by whole-voxel flips only).                                      #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import numpy as np
import pytest

# own
import miox
from tests.conftest import PUBLIC_DATA

TEST = PUBLIC_DATA / "i2nii" / "test"


@pytest.mark.parametrize("name", ["spr.spr", "spr.sdt", "sivic.idf", "sivic.int2"])
@pytest.mark.parametrize("reference", ["mha.mha", "mgh.mgz"])
def test_voxels_sit_where_the_reference_puts_them(name, reference):
    if not (TEST / name).exists():
        pytest.skip("i2nii test data not fetched")
    image, ref = miox.load(TEST / name)[0], miox.load(TEST / reference)[0]
    index = np.linalg.inv(ref.affine) @ image.affine  # our voxel index -> reference voxel index
    assert np.allclose(index, np.round(index), atol=1e-3)
    index = np.round(index).astype(int)
    ijk = np.indices(image.data.shape).reshape(3, -1)
    there = index[:3, :3] @ ijk + index[:3, 3:4]
    assert np.array_equal(np.asarray(image.data)[tuple(ijk)].astype(np.int64),
                          np.asarray(ref.data)[tuple(there)].astype(np.int64))
