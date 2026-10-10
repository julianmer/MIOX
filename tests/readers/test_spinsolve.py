####################################################################################################
#                                        test_spinsolve.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 10/10/26                                                                                #
#                                                                                                  #
# Purpose: Spinsolve reader on synthetic Prospa files: an FID with its acqu.par, a spectrum        #
#          written with the shift rising, an experiment folder with both, and the sniffing.        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

# own
import miox as gu

ACQU = """b1Freq = 43.5
bandwidth = 5
dwellTime = 200
lowestFrequency = -2282.5
nrPnts = 8
nrScans = 4
rxChannel = "1H"
Solvent = "D2O"
startTime = "2021-08-06T09:22:37.745"
Protocol = "1D PROTON"
"""


def prospa(path: Path, axis: np.ndarray, values: np.ndarray) -> Path:
    header = b"SORPATAD1V1V" + struct.pack("<5i", 504, values.size, 1, 1, 1)
    pairs = np.empty(2 * values.size, "<f4")
    pairs[0::2], pairs[1::2] = values.real, values.imag
    path.write_bytes(header + axis.astype("<f4").tobytes() + pairs.tobytes())
    return path


FID = (np.arange(8) - 1j * np.arange(8, 0, -1)).astype(np.complex64)


def test_fid_with_acquisition_parameters(tmp_path):
    (tmp_path / "acqu.par").write_text(ACQU)
    path = prospa(tmp_path / "data.1d", np.arange(8) * 2e-4, FID)
    sniff = gu.sniff(path)
    assert (sniff.format, sniff.kind) == (gu.Format.SPINSOLVE, gu.Kind.SPECTROSCOPY)
    [d] = gu.load(path)
    assert isinstance(d, gu.MRSData) and np.array_equal(d.data[0, 0, 0], FID)  # as stored
    assert d.dwelltime == pytest.approx(2e-4)
    assert d.spectrometer_frequency == pytest.approx(43.5)
    assert d.meta["ReferencePPM"] == pytest.approx((-2282.5 + 2500.0) / 43.5)
    assert d.meta["NumberOfAverages"] == 4 and d.meta["Solvent"] == "D2O"
    assert d.meta["AcquisitionDateTime"] == "2021-08-06T09:22:37.745"


def test_spectrum_is_reversed(tmp_path):
    (tmp_path / "acqu.par").write_text(ACQU)
    axis = np.linspace(-1.0, 6.0, 8)  # rising, as Spinsolve writes it
    [image] = gu.load(prospa(tmp_path / "spectrum.1d", axis, FID))
    assert isinstance(image, gu.MRImage) and image.dims[3] == "frequency"
    assert np.array_equal(image.data[0, 0, 0], FID[::-1])
    assert image.meta["ReferencePPM"] == pytest.approx(axis[::-1][4])
    assert image.meta["SpectralWidth"] == pytest.approx(1.0 * 43.5 * 8)


def test_experiment_folder(tmp_path):
    (tmp_path / "acqu.par").write_text(ACQU)
    prospa(tmp_path / "data.1d", np.arange(8) * 2e-4, FID)
    prospa(tmp_path / "spectrum.1d", np.linspace(-1.0, 6.0, 8), FID)
    assert gu.sniff(tmp_path).format == gu.Format.SPINSOLVE
    datasets = gu.load(tmp_path)
    assert [type(d).__name__ for d in datasets] == ["MRSData", "MRImage"]


def test_other_files_are_not_spinsolve(tmp_path):
    path = tmp_path / "plot.pt1"
    path.write_bytes(b"SORPD1LP" + bytes(64))
    assert gu.sniff(path).format != gu.Format.SPINSOLVE
