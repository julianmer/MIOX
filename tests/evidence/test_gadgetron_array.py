####################################################################################################
#                                     test_gadgetron_array.py                                      #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/11/26                                                                                #
#                                                                                                  #
# Purpose: Gadgetron's standalone test arrays against what their names say: the noise-free         #
#          Shepp-Logan phantom is piecewise constant (few distinct values), the noisy copies       #
#          differ from it by noise that grows from low to high, and the complex radial data set    #
#          keeps its stored dimensions.                                                            #
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
from tests.conftest import PUBLIC_DATA

DATA = PUBLIC_DATA / "gadgetron_arrays"


def _array(name: str):
    path = DATA / name
    if not path.is_file():
        pytest.skip(f"{name} not fetched")
    [image] = load(path)
    return image


def test_shepp_logan_phantom_and_its_noisy_copies():
    clean = _array("shepp_logan_256_256_no_noise.real")
    assert clean.shape == (256, 256, 1) and clean.data.dtype == np.float32
    assert len(np.unique(clean.data)) <= 12  # the ellipses' intensities
    assert np.isclose(clean.data.max(), 1.0, atol=0.1)
    spreads = [np.std(_array(f"shepp_logan_256_256_{level}_noise.real").data - clean.data)
               for level in ("low", "med", "high")]
    assert 0 < spreads[0] < spreads[1] < spreads[2]
    assert any("may be k-space" in note for note in clean.provenance.notes)


def test_complex_array_keeps_its_dimensions():
    data = _array("fb_data.cplx")
    assert data.shape == (256, 2560, 4) and data.data.dtype == np.complex64
    assert data.meta.vendor["dimensions"] == [256, 2560, 4]
    assert np.isfinite(data.data).all() and np.abs(data.data).max() > 0
