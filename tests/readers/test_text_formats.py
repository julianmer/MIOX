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


def _raw(path, header: list[str], fid: np.ndarray, width: int = 15) -> None:
    lines = header + [f"{z.real:{width}.6E}{z.imag:{width}.6E}" for z in fid]
    path.write_text("\n".join(lines) + "\n")


def test_lcmodel_raw(tmp_path):
    f = tmp_path / "metab.RAW"
    header = [
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
    ]
    _raw(f, header, _fid())
    [d] = gu.load(f)
    assert d.shape == (1, 1, 1, 64) and d.dwelltime == pytest.approx(2.5e-4)
    assert d.spectrometer_frequency == pytest.approx(123.25)
    assert d.meta["EchoTime"] == pytest.approx(0.03)
    assert np.allclose(d.data[0, 0, 0], _fid().conj(), atol=1e-6)  # LCModel to NIfTI-MRS
    assert d.meta.vendor["namelists"]["nmid"]["id"] == "test"
    assert d.meta.fields["SpectrometerFrequency"] == "namelists/seqpar/hzpppm"
    assert any("1H assumed" in note for note in d.provenance.notes)


def test_lcmodel_raw_text_tight_format_bruker_and_control(tmp_path):
    f = tmp_path / "test.RAW"
    header = [
        "Free text written by some exporter",
        " $NMID ID='x', FMTDAT='(2E13.5)', BRUKER=T $END",
    ]
    fid = -_fid()  # negative numbers run together at width 13
    _raw(f, header, fid, width=13)
    (tmp_path / "test.CONTROL").write_text(" $LCMODL\n HZPPPM=84.47\n DELTAT=5E-04\n $END\n")
    [d] = gu.load(f)
    assert d.spectrometer_frequency == pytest.approx(84.47) and d.dwelltime == pytest.approx(5e-4)
    assert np.allclose(d.data[0, 0, 0], fid, atol=1e-5)  # BRUKER=T: already conjugated
    assert d.meta.vendor["text"] == ["Free text written by some exporter"]
    assert d.meta.fields["DwellTime"] == "side_namelists/lcmodl/deltat"


def test_lcmodel_raw_without_frequency(tmp_path):
    f = tmp_path / "basis_component.RAW"
    _raw(f, [" $NMID ID='x', FMTDAT='(2E15.6)' $END"], _fid())
    with pytest.raises(gu.ReaderError, match="HZPPPM and dwell time"):
        gu.load(f)


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
        "Name of Study: spaced key",
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
    assert d.meta.vendor["Name of Study"] == "spaced key"
    assert d.meta.source["DwellTime"] == "spec2nii jmrui: SamplingInterval"
