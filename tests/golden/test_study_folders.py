####################################################################################################
#                                      test_study_folders.py                                       #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 05/10/26                                                                                #
#                                                                                                  #
# Purpose: DICOM study folders that hold image series next to spectroscopy series load as every    #
#          series, each equal to the series loaded from its own folder.                            #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import numpy as np
import pytest

# own
from miox import MRImage, MRSData, load
from tests.conftest import PUBLIC_DATA

STUDIES = (  # a folder of series folders, with the kinds it holds
    "spec2nii_test_data/UIH/mrs_data/dicom",
    "spec2nii_test_data/UIH/mrs_3d/dicom",
    "spec2nii_test_data/Siemens/VBData/DICOM",
    "spec2nii_test_data/Siemens/VEData/DICOM",
    "spec2nii_test_data/philips/DICOM",
)
NOT_COMPARED = ("ConversionTime",)


def _series_folders(study):
    return sorted(c for c in study.iterdir() if c.is_dir() and c.name != "screenshots")


@pytest.mark.parametrize("relpath", STUDIES)
def test_study_loads_every_series(relpath):
    study = PUBLIC_DATA / relpath
    if not study.is_dir():
        pytest.skip(f"{relpath} not fetched")
    whole = sorted(load(study), key=lambda d: d.name)
    parts = sorted((d for f in _series_folders(study) for d in load(f)), key=lambda d: d.name)
    assert [d.name for d in whole] == [d.name for d in parts]
    assert any(isinstance(d, MRImage) for d in whole)
    assert any(isinstance(d, MRSData) for d in whole)
    for ours, theirs in zip(whole, parts, strict=True):
        assert type(ours) is type(theirs)
        assert np.array_equal(ours.data, theirs.data)
        keys = (set(ours.meta) | set(theirs.meta)) - set(NOT_COMPARED)
        assert {k: ours.meta.get(k) for k in keys} == {k: theirs.meta.get(k) for k in keys}
