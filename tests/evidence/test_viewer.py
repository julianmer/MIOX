####################################################################################################
#                                          test_viewer.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 11/10/26                                                                                #
#                                                                                                  #
# Purpose: The spectrum view of measured water-suppressed brain spectra puts NAA at 2.01 ppm, far  #
#          above anything at its mirror position (7.29 ppm): the view's sign convention is right.  #
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
from miox.viewer import SpectrumView
from tests.conftest import PUBLIC_DATA

SPECTRA = [  # averaged in vivo spectra
    "spec2nii_test_data/philips/P3/SV_PRESS_sh_4_2_raw_act.SDAT",
    "spec2nii_test_data/philips/P4/SV_PRESS_sh_11_2_raw_act.SDAT",
    "spec2nii_test_data/Siemens/rda/latin1.rda",
]


@pytest.mark.data
@pytest.mark.parametrize("relpath", SPECTRA)
def test_naa_at_its_chemical_shift(relpath):
    path = PUBLIC_DATA / relpath
    if not path.exists():
        pytest.skip("file not fetched")
    view = SpectrumView(load(path)[0])
    x, y = view.line.get_data()
    naa = (x > 1.8) & (x < 2.2)
    mirror = (x > 7.1) & (x < 7.5)
    assert x[naa][np.argmax(y[naa])] == pytest.approx(2.01, abs=0.05)
    assert y[naa].max() > 3 * y[mirror].max()
