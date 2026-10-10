####################################################################################################
#                                          test_tecmag.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 10/10/26                                                                                #
#                                                                                                  #
# Purpose: Tecmag TNMR reader on synthetic .tnt files: a 1D FID with its parameters, a 2D FID on   #
#          DIM_INDIRECT_0, frequency-domain data refused, and the sniffing of the magic.           #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from nmrglue.fileio.tecmag import TNTTLV, TNTTMAG, TNTTMG2

# own
import miox as gu


def tnt(path: Path, data: np.ndarray, fft: bool = False) -> Path:
    """A .tnt with TMAG, DATA and TMG2 sections; ``data`` up to 4D, stored in Fortran order."""
    shape = list(data.shape) + [1] * (4 - data.ndim)
    tmag = np.zeros(1, TNTTMAG)
    tmag["npts"] = tmag["actual_npts"] = shape
    tmag["actual_scans"] = 8
    tmag["ob_freq"][0, :2] = (117.2, 75.4)
    tmag["sw"][0, :2] = (10000.0, 500.0)
    tmag["dwell"][0, :2] = (1e-4, 2e-3)
    tmag["actual_temperature"] = 293.0
    tmag["nuclei"][0, 0] = b"Li7\x00222"
    tmag["lock_solvent"] = b"D2O\x00222"
    tmg2 = np.zeros(1, TNTTMG2)
    tmg2["fft_flag"][0, 0] = int(fft)
    raw = np.asarray(data, "<c8").reshape(-1, order="F").tobytes()

    def section(tag: bytes, payload: bytes) -> bytes:
        return np.array([(tag, 1, len(payload))], TNTTLV).tobytes() + payload

    path.write_bytes(b"TNT1.005" + section(b"TMAG", tmag.tobytes()) + section(b"DATA", raw)
                     + section(b"TMG2", tmg2.tobytes()))
    return path


FID = (np.arange(16) - 1j * np.arange(16, 0, -1)).astype(np.complex64)


def test_one_dimensional_fid(tmp_path):
    path = tnt(tmp_path / "fid.tnt", FID)
    sniff = gu.sniff(path)
    assert (sniff.format, sniff.kind) == (gu.Format.TECMAG_TNT, gu.Kind.SPECTROSCOPY)
    [d] = gu.load(path)
    assert d.shape == (1, 1, 1, 16) and np.array_equal(d.data[0, 0, 0], FID)
    assert d.dwelltime == pytest.approx(1e-4)
    assert d.spectrometer_frequency == pytest.approx(117.2)
    assert d.meta["ResonantNucleus"] == ["7Li"] and d.meta["Solvent"] == "D2O"
    assert d.meta["NumberOfAverages"] == 8 and d.meta["SampleTemperature"] == 293.0
    assert "ReferencePPM" not in d.meta
    assert d.meta.vendor["nuclei"][0] == "Li7"


def test_second_dimension_on_dim_indirect_0(tmp_path):
    data = np.stack([FID, 2 * FID, 3 * FID], axis=1)
    [d] = gu.load(tnt(tmp_path / "2d.tnt", data))
    assert d.shape == (1, 1, 1, 16, 3) and d.dim_tags[0] == "DIM_INDIRECT_0"
    assert np.array_equal(d.data[0, 0, 0, :, 2], 3 * FID)


def test_frequency_domain_data_are_refused(tmp_path):
    with pytest.raises(gu.ReaderError):
        gu.load(tnt(tmp_path / "spectrum.tnt", FID, fft=True))


def test_other_files_are_not_tecmag(tmp_path):
    path = tmp_path / "other.tnt"
    path.write_bytes(b"TNT2.000" + bytes(64))
    assert gu.sniff(path).format != gu.Format.TECMAG_TNT
