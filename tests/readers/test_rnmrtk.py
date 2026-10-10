####################################################################################################
#                                          test_rnmrtk.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 10/10/26                                                                                #
#                                                                                                  #
# Purpose: RNMRTK reader on files written by nmrglue: a 1D FID kept as stored with its .par keys,  #
#          a 2D FID with its rows, frequency-domain data refused, and the sniffing.                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

from pathlib import Path

import nmrglue as ng
import numpy as np
import pytest

# own
import miox as gu


def write(path: Path, data: np.ndarray, time: bool = True) -> Path:
    udic = {"ndim": data.ndim}
    for i in range(data.ndim):
        udic[i] = {"size": data.shape[i], "complex": True, "encoding": "states",
                   "sw": 1000.0 * (i + 1), "obs": 500.0 / (i + 1), "car": 2350.0 / (i + 1),
                   "label": "H1", "time": time, "freq": not time}
    dic = ng.rnmrtk.create_dic(udic)
    ng.rnmrtk.write(str(path), dic, data.astype(np.complex64), overwrite=True)
    return path


FID = (np.arange(8) + 1j * np.arange(8, 0, -1)).astype(np.complex64)


def test_one_dimensional_fid(tmp_path):
    path = write(tmp_path / "test.sec", FID)
    for file in (path, path.with_suffix(".par")):
        sniff = gu.sniff(file)
        assert (sniff.format, sniff.kind) == (gu.Format.RNMRTK, gu.Kind.SPECTROSCOPY)
    [d] = gu.load(path)
    assert np.array_equal(d.data[0, 0, 0], FID)  # as stored
    assert d.dwelltime == pytest.approx(1e-3)
    assert d.meta["SpectrometerFrequency"] == pytest.approx([500.0])
    assert d.meta["ReferencePPM"] == pytest.approx(4.7)
    assert "ResonantNucleus" in d.missing


def test_two_dimensional_fid(tmp_path):
    rows = np.stack([FID * (k + 1) for k in range(4)])
    [d] = gu.load(write(tmp_path / "test2d.sec", rows))
    assert d.shape == (1, 1, 1, 8, 4) and d.dim_tags[0] == "DIM_INDIRECT_0"
    assert np.array_equal(d.data[0, 0, 0, :, 3], 4 * FID)


def test_frequency_domain_is_refused(tmp_path):
    with pytest.raises(gu.ReaderError, match="frequency"):
        gu.load(write(tmp_path / "spectrum.sec", FID, time=False))


def test_sec_without_par_is_not_rnmrtk(tmp_path):
    path = tmp_path / "lonely.sec"
    path.write_bytes(bytes(64))
    assert gu.sniff(path).format != gu.Format.RNMRTK
