####################################################################################################
#                                     test_varian_kspace.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/10/26                                                                                #
#                                                                                                  #
# Purpose: Varian imaging k-space against the quantitative maps published with the scans (zenodo   #
#          7442381): the multi-echo 2D scan reconstructs slice by slice to the S0 map and its      #
#          echoes fit to the T2 map, the multi-echo 3D scan's echoes fit exactly to the T2* map.   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import least_squares

# own
from miox import load
from miox.matfile import MatFile
from tests.conftest import PUBLIC_DATA

STRAWBERRY = PUBLIC_DATA / "varian_strawberry"


def _need(*paths):
    for path in paths:
        if not path.exists():
            pytest.skip(f"{path.relative_to(PUBLIC_DATA)} not fetched")


def _fit(echoes: np.ndarray, times: np.ndarray) -> np.ndarray:
    """S0 and the decay time of S0·exp(-t/T) per row, by nonlinear least squares."""
    return np.array([
        least_squares(lambda p, y=y: p[0] * np.exp(-times / p[1]) - y, x0=[y[0] * 1.2, 50.0]).x
        for y in echoes
    ])


def _voxels(mask: np.ndarray, count: int) -> np.ndarray:
    return np.random.default_rng(0).choice(np.flatnonzero(mask.ravel()), count, replace=False)


def test_multi_echo_2d_scan_reconstructs_to_the_published_maps():
    """The S0 and T2 maps of the 8-slice, 10-echo spin-echo scan: each slice's first echo,
    reconstructed by an inverse FFT, matches the S0 map of the same slice without flips, and the
    echoes at the TE list fit to the published T2 and S0 within a few percent."""
    _need(STRAWBERRY / "mems.fid", STRAWBERRY / "t2_map.MAT")
    [raw] = load(STRAWBERRY / "mems.fid")
    maps = MatFile.load(STRAWBERRY / "t2_map.MAT")["DATA"]["FTDATA"]  # (read, phase, slice, 3)
    kspace = np.zeros((8, 10, 256, 512), dtype=np.complex64)
    kspace[raw.counters("slice"), raw.counters("contrast"),
           raw.counters("kspace_encode_step_1")] = raw.data[:, 0]
    images = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace, axes=(2, 3))),
                                    axes=(2, 3))) * 256 * 512
    images = np.transpose(images, (3, 2, 0, 1))  # (read, phase, slice, echo)
    for s in range(8):
        r = np.corrcoef(images[:, :, s, 0].ravel(), maps[:, :, s, 0].ravel())[0, 1]
        assert r > 0.93, s
    s0, t2 = maps[..., 0], maps[..., 1]
    voxels = _voxels((s0 > np.percentile(s0, 95)) & (t2 > 0) & (t2 < 1000), 200)
    times = np.array(raw.meta["EchoTime"]) * 1000
    fitted = _fit(images.reshape(-1, 10)[voxels], times)
    assert np.median(fitted[:, 1] / t2.ravel()[voxels]) == pytest.approx(1, abs=0.03)
    assert np.median(fitted[:, 0] / s0.ravel()[voxels]) == pytest.approx(1, abs=0.03)


def test_multi_echo_3d_scan_fits_exactly_to_the_published_t2star_map():
    """The 6-echo 3D gradient-echo scan (fetched with --all): its echoes, reconstructed by a 3D
    inverse FFT and fitted at the TE list, give the published S0 and T2* maps to 0.1 %. The maps
    are mirrored through the centre in all three axes (a forward FFT) and hold (phase, read,
    slab)."""
    _need(STRAWBERRY / "mge3d.fid", STRAWBERRY / "t2star_map.MAT")
    [raw] = load(STRAWBERRY / "mge3d.fid")
    maps = MatFile.load(STRAWBERRY / "t2star_map.MAT")["DATA"]["FTDATA"]
    kspace = np.zeros((6, 192, 192, 384), dtype=np.complex64)
    kspace[raw.counters("contrast"), raw.counters("kspace_encode_step_2"),
           raw.counters("kspace_encode_step_1")] = raw.data[:, 0]
    images = np.abs(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(kspace, axes=(1, 2, 3)),
                                                 axes=(1, 2, 3)), axes=(1, 2, 3))) * kspace[0].size
    images = np.transpose(images, (2, 3, 1, 0))  # (phase, read, slab, echo)
    images = np.roll(np.flip(images, (0, 1, 2)), 1, axis=(0, 1, 2))
    s0, t2star = maps[..., 0], maps[..., 1]
    voxels = _voxels((s0 > np.percentile(s0, 97)) & (t2star > 0) & (t2star < 200), 100)
    fitted = _fit(images.reshape(-1, 6)[voxels], np.array(raw.meta["EchoTime"]) * 1000)
    assert np.allclose(fitted[:, 1], t2star.ravel()[voxels], rtol=1e-3)
    assert np.allclose(fitted[:, 0], s0.ravel()[voxels], rtol=1e-3)
