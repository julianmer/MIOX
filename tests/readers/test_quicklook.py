####################################################################################################
#                                       test_quicklook.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 12/10/26                                                                                #
#                                                                                                  #
# Purpose: The Quick Look page on synthetic data: five tabs, the processed spectrum with its       #
#          steps, coils with their weights, averages before and after alignment, edited data with  #
#          their difference, and the empty states (one coil, no water, no spectral axis).          #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import numpy as np

# own
import miox
from miox.data import MRSData, Provenance
from miox.meta import Meta
from miox.quicklook import Page

SW, SF, N = 2000.0, 123.2, 1024
T = np.arange(N) / SW


def fid():
    """NAA and creatine as NIfTI-MRS stores them (the conjugate of FSL-MRS's orientation)."""
    return sum(a * np.exp(2j * np.pi * (ppm - 4.65) * SF * T - np.pi * 6 * T)
               for ppm, a in ((2.01, 1.0), (3.027, 0.8))).conj()


def dataset(data, tags=(None, None, None), dwell=1 / SW, name="test"):
    meta = Meta({})
    meta.set("SpectrometerFrequency", [SF], "test")
    meta.set("ResonantNucleus", ["1H"], "test")
    return MRSData(data.reshape(1, 1, 1, *data.shape), dwell, np.eye(4), meta,
                   Provenance((), miox.Format.NIFTI_MRS, "test", {}), tuple(tags), name=name)


def test_page_with_coils_and_averages():
    rng = np.random.default_rng(0)
    data = fid()[:, None, None] * rng.uniform(0.5, 1, (1, 4, 1)) + 0.01 * rng.normal(
        size=(N, 4, 8))
    page = Page([dataset(data, ("DIM_COIL", "DIM_DYN", None))], "a.dat").html()
    for tab in ("Spectrum", "Coils", "Averages", "Water", "Details"):
        assert f">{tab}</label>" in page
    assert "4 coils combined (wSVD)" in page and "8 transients aligned" in page
    assert page.count("<figure>") == 4 and "before alignment" in page
    assert "No water reference in this file." in page


def test_edited_page_shows_the_difference():
    data = np.stack([fid(), 0.5 * fid()], axis=1)[:, :, None] * np.ones((1, 1, 4))
    page = Page([dataset(data, ("DIM_EDIT", "DIM_DYN", None))], "mega.dat").html()
    assert "difference 1 − 2" in page and "each edit condition processed on its own" in page


def test_single_fid_and_missing_axis():
    page = Page([dataset(fid())], "one.rda").html()
    assert "One channel" in page and "One transient" in page
    unknown = Page([dataset(fid(), dwell=None)], "raw.txt").html()
    assert "nothing is processed" in unknown
