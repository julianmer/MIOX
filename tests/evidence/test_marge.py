####################################################################################################
#                                          test_marge.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 10/11/26                                                                                #
#                                                                                                  #
# Purpose: MaRGE .mat files against what they store beside the readouts: the oversampled ADC       #
#          stream decimated, sorted by sweepOrder and cropped at the echo gives the readouts;     #
#          their mean times the FOV-shift phase ramp (k in 1/m, dfov in metres) gives kSpace3D;   #
#          its centred inverse FFT gives image3D, which the MRImage holds on the scanner's axes.   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import numpy as np
import pytest
from scipy import signal
from scipy.io import loadmat

# own
from miox import load
from tests.conftest import PUBLIC_DATA

FILES = [
    "marge/FENCE_InVivoSNR/FENCE/Spectrum/Localizer.972.mat",
    "marge/FENCE_Phantom_OSIIcoil/NoShield/Imaging/Grounding_RARE.2025.11.07.15.38.47.924.mat",
]
OVERSAMPLING = 6  # the Red Pitaya acquires six samples per stored sample


def _need(path):
    if not path.is_file():
        pytest.skip(f"{path.relative_to(PUBLIC_DATA)} not fetched")


def _rel(a, b) -> float:
    return float(np.abs(a - b).max() / np.abs(b).max())


@pytest.mark.parametrize("relpath", FILES)
def test_readouts_come_from_the_oversampled_stream(relpath):
    """Decimating overData (FIR, zero phase) by the oversampling factor, sorting the lines by
    sweepOrder and cropping each to the readout matrix around the echo (within addRdPoints of
    the middle, as MaRGE searches it) reproduces the readouts exactly."""
    path = PUBLIC_DATA / relpath
    _need(path)
    stored = loadmat(path)
    raw, _ = load(path)
    nrd, nph, nsl = (int(v) for v in stored["nPoints"][0])
    add = int(stored["addRdPoints"][0, 0])
    stream = stored["overData"].ravel()
    assert stream.size == OVERSAMPLING * nsl * nph * (nrd + 2 * add)
    lines = signal.decimate(stream, OVERSAMPLING, ftype="fir", zero_phase=True)
    lines = lines.reshape(nsl, nph, nrd + 2 * add)
    sorted_lines = np.zeros_like(lines)
    sorted_lines[:, stored["sweepOrder"][0], :] = lines
    ours = raw.data[:, 0, :].reshape(nsl, nph, nrd)
    errors = [_rel(sorted_lines[:, :, c - nrd // 2:c + nrd // 2], ours)
              for c in range((nrd + 2 * add) // 2 - add, (nrd + 2 * add) // 2 + add + 1)]
    assert min(errors) < 1e-12


@pytest.mark.parametrize("relpath", FILES)
def test_kspace_and_image_as_stored(relpath):
    """kSpace3D is the readouts times exp(-2 pi i k . dfov) with k from the sampled columns
    (1/m) and dfov in metres; image3D is its centred inverse 3-D FFT; the MRImage holds image3D
    with its axes placed by axesOrientation."""
    path = PUBLIC_DATA / relpath
    _need(path)
    stored = loadmat(path)
    raw, image = load(path)
    nrd, nph, nsl = (int(v) for v in stored["nPoints"][0])
    ours = raw.data[:, 0, :].reshape(nsl, nph, nrd)
    k = stored["sampled"][:, :3].real.reshape(nsl, nph, nrd, 3)
    axes = stored["axesOrientation"][0]
    shift = stored["dfov"][0][axes]  # metres along readout, phase, slice
    ramp = np.exp(-2j * np.pi * (k * shift).sum(axis=-1))
    assert _rel(ours * ramp, stored["kSpace3D"]) < 1e-12
    recon = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(stored["kSpace3D"])))
    assert _rel(recon, stored["image3D"]) < 1e-12
    back = np.transpose(np.transpose(image.data, np.argsort(np.argsort(axes))), (2, 1, 0))
    assert np.array_equal(back, stored["image3D"])
    assert image.voxel_sizes[axes[0]] == pytest.approx(stored["resolution"][0, 0] * 1e3)
