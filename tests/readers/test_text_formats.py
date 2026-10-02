####################################################################################################
#                                       test_text_formats.py                                       #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader tests on synthetic LCModel .RAW and jMRUI text files: the whole header is kept,  #
#          sources name the header field, and frequencies stored in Hz come out in MHz with a      #
#          note. No test data needed.                                                              #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import numpy as np
import pytest

# own
import miox as gu


def _fid(n: int = 64) -> np.ndarray:
    t = np.arange(n) * 2.5e-4
    return np.exp(2j * np.pi * 100 * t - t / 0.05)


def test_lcmodel_raw(tmp_path):
    f = tmp_path / "metab.RAW"
    lines = [
        " $SEQPAR",
        " echot= 30.00",
        " seq= 'PRESS'",
        " hzpppm= 123.25",
        " dwellTime= 0.00025",
        " $END",
        " $NMID",
        " id='test', fmtdat='(2E15.6)'",
        " volume=8.0",
        " $END",
    ] + [f" {z.real:15.6e}{z.imag:15.6e}" for z in _fid()]
    f.write_text("\n".join(lines) + "\n")
    [d] = gu.load(f)
    assert d.shape == (1, 1, 1, 64) and d.dwelltime == pytest.approx(2.5e-4)
    assert d.spectrometer_frequency == pytest.approx(123.25)
    assert d.meta.vendor["SEQPAR"]["ECHOT"] == "30.00"
    assert d.meta.vendor["NMID"]["ID"] == "'test'"
    assert d.meta.source["SpectrometerFrequency"] == "spec2nii raw: SEQPAR/HZPPPM"
    assert d.meta.source["DwellTime"] == "spec2nii raw: SEQPAR/DWELLTIME"
    notes = " ".join(d.provenance.notes)
    assert "in Hz; divided by 1e6" in notes and "1H assumed" in notes


def test_jmrui_text(tmp_path):
    f = tmp_path / "metab.txt"
    lines = [
        "jMRUI Data Textfile",
        "",
        "Filename: metab.txt",
        "",
        "PointsInDataset: 64",
        "DatasetsInFile: 1",
        "SamplingInterval: 0.25",
        "ZeroOrderPhase: 0E0",
        "BeginTime: 0E0",
        "TransmitterFrequency: 1.2325E8",
        "MagneticField: 2.89",
        "TypeOfNucleus: 1H",
        "NameOfPatient: anonymous",
        "DateOfExperiment: ",
        "Spectrometer: test",
        "AdditionalInfo: none",
        "SignalNames: metab",
        "",
        "",
        "Signal and FFT",
        "sig(real)\tsig(imag)\tfft(real)\tfft(imag)",
        "Signal 1 out of 1 in file",
    ] + [f"{z.real:.6E}\t{z.imag:.6E}\t0\t0" for z in _fid()]
    f.write_text("\n".join(lines) + "\n")
    [d] = gu.load(f)
    assert d.shape == (1, 1, 1, 64) and d.nucleus == "1H"
    assert d.spectrometer_frequency == pytest.approx(123.25)
    assert d.meta.vendor["PointsInDataset"] == 64
    assert d.meta.source["DwellTime"] == "spec2nii jmrui: SamplingInterval"
