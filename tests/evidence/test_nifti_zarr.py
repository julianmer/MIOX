####################################################################################################
#                                       test_nifti_zarr.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 13/10/26                                                                                #
#                                                                                                  #
# Purpose: NIfTI-Zarr stores of nifti-zarr-py (MIT) against the NIfTI files they were written      #
#          from (NIfTI-1 and NIfTI-2): the same voxels and affine; files inside a store are not    #
#          taken for files of their own.                                                           #
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

DATA = PUBLIC_DATA / "nifti_zarr_py" / "tests" / "data"


@pytest.mark.parametrize("name", ["example4d", "example_nifti2"])
def test_store_equals_its_nifti_source(name):
    store = DATA / f"{name}.nii.zarr"
    if not store.is_dir():
        pytest.skip("nifti-zarr-py data not fetched")
    assert miox.sniff(store).format == miox.Format.NIFTI_ZARR
    image = miox.load(store)[0]
    source = nib.load(DATA / f"{name}.nii.gz")
    assert np.array_equal(image.data, np.asanyarray(source.dataobj))
    assert np.allclose(image.affine, source.affine)
    assert miox.sniff(store / "nifti" / "0").format == miox.Format.UNKNOWN
