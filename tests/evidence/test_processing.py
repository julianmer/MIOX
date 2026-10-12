####################################################################################################
#                                       test_processing.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 12/10/26                                                                                #
#                                                                                                  #
# Purpose: MIOX's raw pipeline against Augmentrum's RawProcessor (torch engine, defaults) on       #
#          public files (references made by scripts/augmentrum_reference.py): the processed data   #
#          and water to rounding, the same phase and frequency per transient and the same kept     #
#          transients; and the orientation: processed brain spectra have NAA at 2.01 ppm and       #
#          creatine at 3.027 ppm on FSL-MRS's axis.                                                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import json

import numpy as np
import pytest

# own
from miox.processing import RawPipeline, Spectral
from tests.conftest import PUBLIC_DATA

REFERENCES = PUBLIC_DATA / "augmentrum_reference"
CASES = sorted(p.name.removesuffix(".reference.npz") for p in REFERENCES.glob("*.reference.npz"))


def layout(array, tags):
    """(B, X, Y, Z, <tags>, T) as (B, V, C, D, T)."""
    b, x, y, z = array.shape[:4]
    present = list(tags)
    for tag in ("DIM_COIL", "DIM_DYN"):
        if tag not in present:
            array = array[..., None, :]
            present.append(tag)
    order = [0, 1, 2, 3] + [4 + present.index(t) for t in ("DIM_COIL", "DIM_DYN")]
    array = np.transpose(array, order + [array.ndim - 1])
    return array.reshape(b, x * y * z, *array.shape[4:])


def relative(a, b) -> float:
    return float(np.linalg.norm(np.ravel(a) - np.ravel(b)) / np.linalg.norm(np.ravel(b)))


@pytest.mark.skipif(not CASES, reason="no Augmentrum references (scripts/augmentrum_reference.py)")
@pytest.mark.parametrize("name", CASES)
def test_pipeline_equals_augmentrum(name):
    case = np.load(REFERENCES / f"{name}.npz")
    info = json.loads((REFERENCES / f"{name}.json").read_text())
    reference = np.load(REFERENCES / f"{name}.reference.npz")
    water = None
    if "wat" in case:
        water = layout(np.moveaxis(case["wat"], 4, -1), info["wtags"])
    result = RawPipeline(float(case["sw"]), float(case["sf"])).run(
        layout(case["met"], info["tags"]), water)
    assert relative(result.fid, reference["met"]) < 1e-9
    if water is not None:
        assert relative(result.water, reference["wat"]) < 1e-9
    if "eps" in reference:
        assert np.allclose(np.ravel(result.eps), np.ravel(reference["eps"]), atol=1e-6)
        assert np.allclose(np.exp(1j * np.ravel(result.phi)),
                           np.exp(1j * np.ravel(reference["phi"])), atol=1e-6)
    if "keep" in reference:
        assert np.array_equal(np.ravel(result.keep), np.ravel(reference["keep"]))


@pytest.mark.parametrize("name", ["siemens_drift", "siemens_xa20", "ge_press", "nifti_7t"])
def test_brain_spectra_have_naa_and_creatine_in_place(name):
    """The cases are conjugated NIfTI-MRS (FSL-MRS's orientation): after processing, the tallest
    peak in 1.9-2.1 ppm is NAA at 2.01 ppm and creatine sits at 3.027 ppm, both positive. Without
    water the eddy current step is left out here: Augmentrum's default corrects with the data
    itself, which distorts the 7T example (NAA at 2.099 ppm) - kept in MIOX by choice, but not
    what this orientation check is about."""
    if not (REFERENCES / f"{name}.npz").exists():
        pytest.skip(f"{name} not exported")
    case = np.load(REFERENCES / f"{name}.npz")
    info = json.loads((REFERENCES / f"{name}.json").read_text())
    water = None
    if "wat" in case:
        water = layout(np.moveaxis(case["wat"], 4, -1), info["wtags"])
    sw, sf = float(case["sw"]), float(case["sf"])
    result = RawPipeline(sw, sf, ecc_without_water=False).run(layout(case["met"], info["tags"]),
                                                              water)
    fid = np.asarray(result.fid).reshape(1, -1)
    spec = Spectral.fid_to_spec(Spectral.zero_fill(fid, 4))[0]
    axis = Spectral.ppm_axis(spec.size, sw, sf)
    for low, high, expected in ((1.9, 2.1, 2.01), (2.95, 3.1, 3.027)):
        window = (axis > low) & (axis < high)
        peak = np.flatnonzero(window)[np.argmax(np.abs(spec[window]))]
        assert abs(axis[peak] - expected) < 0.03
        assert spec[peak].real > 0
