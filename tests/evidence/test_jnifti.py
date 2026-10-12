####################################################################################################
#                                          test_jnifti.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 13/10/26                                                                                #
#                                                                                                  #
# Purpose: JNIfTI samples (NeuroJSON jnifti, Apache-2.0) against the data they were made from:     #
#          every serialisation (JSON or BJData; raw, zlib, gzip, lzma) gives the voxels of the     #
#          MATLAB source or the NIfTI file, and the NIfTI file's affine.                           #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest
import scipy.io

# own
import miox
from tests.conftest import PUBLIC_DATA

SAMPLES = PUBLIC_DATA / "jnifti" / "samples"
CASES = {  # folder: (files, source)
    "basic": (("mousehead.jnii", "mousehead_lzma.jnii", "mousehead.bnii", "mousehead_gzip.bnii"),
              ("mouse_head.mat", "volimage")),
    "colin27": (("colin27_zlib.jnii", "colin27_zlib.bnii"), ("colin27.mat", "colin27")),
    "digimouse": (("digimouse_zlib.jnii", "digimouse_lzma.jnii", "digimouse_zlib.bnii",
                   "digimouse_lzma.bnii"), ("digimouse.mat", "digimouse")),
    "headct": (("headct_lzma.jnii", "headct_lzma.bnii", "headct_zlib.bnii"),
               ("headct.nii.gz", None)),
    "nifti2": (("avg152T1_LR_nifti2_lzma.bnii", "avg152T1_LR_nifti2_zlib.bnii"),
               ("avg152T1_LR_nifti2.nii.gz", None)),
}


def source(folder: str):
    name, key = CASES[folder][1]
    path = SAMPLES / folder / name
    if key is None:
        image = nib.load(path)
        return np.asanyarray(image.dataobj), image.affine
    value = scipy.io.loadmat(path)[key]
    if value.dtype.names:  # a saved JNIfTI structure
        value = value["NIFTIData"][0, 0]
    return value, None


@pytest.mark.parametrize("folder", CASES)
def test_every_serialisation_gives_the_source(folder):
    if not (SAMPLES / folder).is_dir():
        pytest.skip("jnifti samples not fetched")
    voxels, affine = source(folder)
    for name in CASES[folder][0]:
        image = miox.load(SAMPLES / folder / name)[0]
        assert isinstance(image, miox.MRImage)
        assert np.array_equal(np.asarray(image.data), voxels), name
        if affine is not None:
            assert np.allclose(image.affine, affine, atol=1e-4), name
