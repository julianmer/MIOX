####################################################################################################
#                                        test_reference.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: ReferencePPM as measured files state it: on that axis a known peak lies at its chemical #
#          shift (NAA at 2.008 ppm).                                                               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import numpy as np
import pytest

# own
from miox import MRSData, load
from tests.conftest import PUBLIC_DATA

VESPA = "vespa_io/docs/user_manuals/files/tutorial_data_analysis/tutorial_analysis/"

# (file, window searched in ppm, expected peak, tolerance)
STATED = [
    ("tarquin/data/phillips/svs_30_ws.dpt", (1.8, 2.2), 2.008, 0.02),  # TARQUIN's own referencing
    (VESPA + "tutorial_03_open-multiple-datasets/press_cp0.xml", (1.8, 2.2), 2.008, 0.02),
    # in vivo at body temperature; the scanner's 4.7 ppm is nominal
    ("spec2nii_test_data/bruker/PV-360.3.5/19/pdata/1/fid_proc.64", (1.8, 2.2), 2.008, 0.05),
]


def peak(d: MRSData, low: float, high: float) -> float:
    """Position of the largest peak between ``low`` and ``high`` on the stated axis."""
    n = d.shape[3]
    fid = np.moveaxis(d.data, 3, 0).reshape(n, -1)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(fid, axis=0), axes=0)).sum(axis=1)
    frequencies = np.fft.fftshift(np.fft.fftfreq(n, d.dwelltime))
    ppm = d.meta["ReferencePPM"] - frequencies / d.spectrometer_frequency
    window = (ppm > low) & (ppm < high)
    return float(ppm[window][np.argmax(spectrum[window])])


@pytest.mark.data
@pytest.mark.parametrize("relpath,window,expected,tolerance", STATED, ids=[s[0] for s in STATED])
def test_known_peak_on_the_stated_axis(relpath, window, expected, tolerance):
    path = PUBLIC_DATA / relpath
    if not path.exists():
        pytest.skip("file not fetched")
    d = load(path)[0]
    assert peak(d, *window) == pytest.approx(expected, abs=tolerance)
