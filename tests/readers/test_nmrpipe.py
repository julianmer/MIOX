####################################################################################################
#                                         test_nmrpipe.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 10/10/26                                                                                #
#                                                                                                  #
# Purpose: NMRPipe reader on files written by nmrglue: a 1D FID (conjugated on reading), a 2D FID  #
#          with its indirect rows, a 2D spectrum with both axes, transposed time-domain data       #
#          refused, and the sniffing of the header.                                                #
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
import miox
from tests.conftest import PUBLIC_DATA


def udic(sizes: tuple[int, ...], time: bool) -> dict:
    labels = (("C13", 125.0, 2500.0, 12375.0), ("H1", 500.0, 5000.0, 2350.0))[-len(sizes):]
    u = {"ndim": len(sizes)}
    for i, (size, (label, obs, sw, car)) in enumerate(zip(sizes, labels, strict=True)):
        u[i] = {"size": size, "complex": time, "encoding": "states" if i < len(sizes) - 1 else
                "direct", "sw": sw, "obs": obs, "car": car, "label": label, "time": time,
                "freq": not time}
    return u


def write(path: Path, data: np.ndarray, time: bool) -> Path:
    dic = ng.pipe.create_dic(udic(data.shape, time))
    ng.pipe.write(str(path), dic, data.astype(np.complex64 if time else np.float32), overwrite=True)
    return path


FID = (np.arange(8) + 1j * np.arange(8, 0, -1)).astype(np.complex64)


def test_one_dimensional_fid_is_conjugated(tmp_path):
    path = write(tmp_path / "test.fid", FID, time=True)
    sniff = miox.sniff(path)
    assert (sniff.format, sniff.kind) == (miox.Format.NMRPIPE, miox.Kind.SPECTROSCOPY)
    [d] = miox.load(path)
    assert isinstance(d, miox.MRSData) and d.shape == (1, 1, 1, 8)
    assert np.array_equal(d.data[0, 0, 0], FID.conj())
    assert d.dwelltime == pytest.approx(1 / 5000.0)
    assert d.spectrometer_frequency == pytest.approx(500.0)
    assert d.meta["ReferencePPM"] == pytest.approx(4.7) and d.meta["ResonantNucleus"] == ["1H"]


def test_two_dimensional_fid_keeps_its_rows(tmp_path):
    rows = np.stack([FID * (k + 1) for k in range(4)])  # 2 complex increments as pairs of rows
    [d] = miox.load(write(tmp_path / "test2d.fid", rows, time=True))
    assert d.shape == (1, 1, 1, 8, 4) and d.dim_tags[0] == "DIM_INDIRECT_0"
    assert np.array_equal(d.data[0, 0, 0, :, 3], 4 * FID.conj())
    assert d.meta["ResonantNucleus"] == ["1H", "13C"]
    assert d.meta["SpectrometerFrequency"] == pytest.approx([500.0, 125.0])


def test_two_dimensional_spectrum(tmp_path):
    values = np.arange(32.0).reshape(4, 8)
    [image] = miox.load(write(tmp_path / "test.ft2", values, time=False))
    assert isinstance(image, miox.MRImage) and image.shape == (1, 1, 1, 8, 4)
    assert image.dims[3:] == ("frequency", "frequency_indirect_0")
    assert image.meta["SpectrometerFrequency"] == pytest.approx([500.0, 125.0])
    # the carrier (4.7 ppm) lies at point N / 2, the shift falls with the index
    assert image.meta["ReferencePPM"] == pytest.approx(4.7)
    assert image.meta["SpectralWidth"] == pytest.approx(5000.0)
    assert np.array_equal(image.data[0, 0, 0], values.T)


def test_transposed_time_domain_data_are_refused():
    path = PUBLIC_DATA / "nmrglue_repo/nmrglue/fileio/tests/data/nmrpipe_2d_time_tp.fid"
    if not path.is_file():
        pytest.skip("not fetched")
    with pytest.raises(miox.ReaderError, match="transposed"):
        miox.load(path)


def test_other_files_are_not_nmrpipe(tmp_path):
    path = tmp_path / "zeros.fid"
    path.write_bytes(bytes(4096))
    assert miox.sniff(path).format != miox.Format.NMRPIPE
