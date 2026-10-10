####################################################################################################
#                                         test_simpson.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 10/10/26                                                                                #
#                                                                                                  #
# Purpose: SIMPSON reader on synthetic text files: an FID (conjugated on reading), a spectrum      #
#          turned into the transform of the conjugated FID, a 2D FID, and the sniffing.            #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# own
import miox as gu


def simpson(path: Path, values: np.ndarray, kind: str, extra: str = "") -> Path:
    lines = "\n".join(f"{v.real:.9g} {v.imag:.9g}" for v in values.ravel())
    path.write_text(f"SIMP\nNP={values.shape[-1]}\nSW=1000\n{extra}TYPE={kind}\nDATA\n{lines}\nEND\n")
    return path


FID = np.exp(2j * np.pi * 125.0 * np.arange(8) / 1000.0)  # one line at +125 Hz as SIMPSON writes it


def test_fid_is_conjugated(tmp_path):
    path = simpson(tmp_path / "x.fid", FID, "FID")
    sniff = gu.sniff(path)
    assert (sniff.format, sniff.kind) == (gu.Format.SIMPSON, gu.Kind.SPECTROSCOPY)
    [d] = gu.load(path)
    assert isinstance(d, gu.MRSData) and d.shape == (1, 1, 1, 8)
    assert np.allclose(d.data[0, 0, 0], FID.conj())
    assert d.dwelltime == pytest.approx(1e-3)
    assert {"SpectrometerFrequency", "ResonantNucleus"} <= set(d.missing)


def test_spectrum_is_the_transform_of_the_conjugated_fid(tmp_path):
    first_halved = np.r_[FID[0] / 2, FID[1:]]
    spectrum = np.fft.fftshift(np.fft.fft(first_halved))  # what SIMPSON's fft saves
    [image] = gu.load(simpson(tmp_path / "x.spe", spectrum, "SPE"))
    assert isinstance(image, gu.MRImage) and image.dims[3] == "frequency"
    ours = np.fft.fftshift(np.fft.fft(first_halved.conj()))
    assert np.allclose(image.data[0, 0, 0], ours)
    assert image.meta["SpectralWidth"] == 1000.0


def test_two_dimensional_fid(tmp_path):
    rows = np.stack([FID, 2 * FID, 3 * FID])
    [d] = gu.load(simpson(tmp_path / "x2d.fid", rows, "FID", "NI=3\nSW1=500\n"))
    assert d.shape == (1, 1, 1, 8, 3) and d.dim_tags[0] == "DIM_INDIRECT_0"
    assert np.allclose(d.data[0, 0, 0, :, 2], 3 * FID.conj())


def test_other_text_is_not_simpson(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("SIMPLE notes\n")
    assert gu.sniff(path).format != gu.Format.SIMPSON
