####################################################################################################
#                                       test_processing.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 12/10/26                                                                                #
#                                                                                                  #
# Purpose: The raw pipeline on synthetic single-voxel data with known answers: shifted and         #
#          dephased transients (drift within +-2 Hz) are aligned back, coils with known            #
#          sensitivities combine coherently, a corrupted transient is found unlike, and with a     #
#          water reference creatine ends at 3.027 ppm with a real, positive peak.                  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import numpy as np

# own
from miox.processing import CoilCombination, RawPipeline, Registration, Spectral, Unlike

SW, SF, N = 2000.0, 123.2, 1024
T = np.arange(N) / SW


def peaks(shift_hz=0.0, phase=0.0):
    """NAA, creatine and choline (FSL-MRS convention: ppm = 4.65 + f / SF)."""
    fid = sum(a * np.exp(2j * np.pi * ((ppm - 4.65) * SF + shift_hz) * T - np.pi * 6 * T)
              for ppm, a in ((2.01, 1.0), (3.027, 0.8), (3.2, 0.6)))
    return fid * np.exp(1j * phase)


def test_alignment_recovers_shifts_and_phases():
    rng = np.random.default_rng(0)
    shifts, phases = rng.uniform(-2, 2, 12), rng.uniform(-1, 1, 12)
    fids = np.stack([peaks(s, p) for s, p in zip(shifts, phases, strict=True)])[None]
    fids = fids + 0.002 * (rng.normal(size=fids.shape) + 1j * rng.normal(size=fids.shape))
    phi, eps = Registration.align(fids, SW, SF, (0.2, 4.2))
    aligned = fids * Registration.phasor(phi, eps, N, SW)
    # e^{-2 pi i eps t} removes a shift eps: every transient ends at the target's frequency
    spread = np.linalg.norm(aligned - aligned.mean(axis=1, keepdims=True), axis=-1)
    assert spread.max() < 0.05 * np.linalg.norm(aligned[0, 0])
    assert np.ptp(eps - shifts) < 0.05


def test_coils_combine_coherently():
    rng = np.random.default_rng(1)
    sensitivity = rng.uniform(0.2, 1, 8) * np.exp(1j * rng.uniform(-np.pi, np.pi, 8))
    signal = peaks()
    x = (signal[None, :] * sensitivity[:, None])[None, None, :, None, :]
    x = x + 0.01 * (rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape))
    combined, _, weights = CoilCombination.combine(x, None)
    fid = combined[0, 0, 0, 0]
    assert abs(np.vdot(fid, signal)) / (np.linalg.norm(fid) * np.linalg.norm(signal)) > 0.999
    assert weights.shape == (1, 1, 1, 8)


def test_corrupted_transient_is_unlike():
    rng = np.random.default_rng(2)
    fids = np.stack([peaks() + 0.01 * rng.normal(size=N) for _ in range(16)])[None]
    fids[0, 5] = peaks(40.0, 2.0)
    keep = Unlike.keep(fids)
    assert not keep[0, 5] and keep[0].sum() == 15


def test_pipeline_puts_creatine_at_3027_with_a_real_peak():
    rng = np.random.default_rng(3)
    x = np.stack([peaks(4.0 + rng.normal(0, 1), rng.normal(0, 0.3)) for _ in range(8)])
    x = x[None, None, None] + 0.003 * rng.normal(size=(1, 1, 1, 8, N))
    water = 50 * np.exp(2j * np.pi * 4.0 * T - np.pi * 6 * T)[None, None, None, None]
    result = RawPipeline(SW, SF).run(x, water)
    fid = np.asarray(result.fid).reshape(-1)
    spec = Spectral.fid_to_spec(Spectral.zero_fill(fid[None], 4))[0]
    axis = Spectral.ppm_axis(4 * N, SW, SF)
    window = (axis > 2.9) & (axis < 3.1)
    peak = np.flatnonzero(window)[np.argmax(np.abs(spec[window]))]
    assert abs(axis[peak] - 3.027) < 0.01
    assert spec[peak].real > 0.99 * abs(spec[peak])
