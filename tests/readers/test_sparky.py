####################################################################################################
#                                          test_sparky.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 10/10/26                                                                                #
#                                                                                                  #
# Purpose: Sparky reader on a UCSF file written by nmrglue: a 2D spectrum with the direct axis     #
#          first and both axes from the header, and the sniffing of the identifier.                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import nmrglue as ng
import numpy as np
import pytest

# own
import miox


def test_two_dimensional_spectrum(tmp_path):
    udic = {"ndim": 2}
    for i, (size, label, obs, sw, car) in enumerate(
        ((16, "15N", 50.0, 1000.0, 5900.0), (32, "1H", 500.0, 4000.0, 2350.0))
    ):
        udic[i] = {"size": size, "complex": False, "encoding": "states", "sw": sw, "obs": obs,
                   "car": car, "label": label, "time": False, "freq": True}
    values = np.arange(16 * 32, dtype=np.float32).reshape(16, 32)
    path = tmp_path / "test.ucsf"
    ng.sparky.write(str(path), ng.sparky.create_dic(udic), values, overwrite=True)
    sniff = miox.sniff(path)
    assert (sniff.format, sniff.kind) == (miox.Format.SPARKY, miox.Kind.SPECTROSCOPY)
    [image] = miox.load(path)
    assert image.shape == (1, 1, 1, 32, 16)
    assert image.dims[3:] == ("frequency", "frequency_indirect_0")
    assert np.array_equal(image.data[0, 0, 0], values.T)
    assert image.meta["SpectrometerFrequency"] == pytest.approx([500.0, 50.0])
    assert image.meta["ResonantNucleus"] == ["1H", "15N"]
    # the transmitter (2350 Hz at 500 MHz: 4.7 ppm) lies at point N / 2
    assert image.meta["ReferencePPM"] == pytest.approx(4.7)
    assert image.meta["SpectralWidth"] == pytest.approx(4000.0)


def test_other_files_are_not_sparky(tmp_path):
    path = tmp_path / "other.ucsf"
    path.write_bytes(b"UCSF RMN" + bytes(200))
    assert miox.sniff(path).format != miox.Format.SPARKY
