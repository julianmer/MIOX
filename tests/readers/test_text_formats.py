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
    assert np.allclose(d.data[0, 0, 0], _fid(), atol=1e-6)  # same convention as NIfTI-MRS
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
    (tmp_path / "control.file").write_text(" $LCMODL\n HZPPPM=84.47\n DELTAT=5E-04\n $END\n")
    [d] = gu.load(f)
    assert d.spectrometer_frequency == pytest.approx(84.47) and d.dwelltime == pytest.approx(5e-4)
    assert np.allclose(d.data[0, 0, 0], fid.conj(), atol=1e-5)  # BRUKER=T: conjugated
    assert d.meta.vendor["text"] == ["Free text written by some exporter"]
    assert d.meta.fields["DwellTime"] == "side_namelists/lcmodl/deltat"


def test_lcmodel_raw_without_frequency(tmp_path):
    f = tmp_path / "basis_component.RAW"
    _raw(f, [" $NMID ID='x', FMTDAT='(2E15.6)' $END"], _fid())
    [d] = gu.load(f)
    assert d.missing == ("SpectrometerFrequency", "DwellTime") and d.dwelltime is None
    assert d.nucleus == "1H" and d.spectrometer_frequency is None and d.spectral_width is None
    assert np.allclose(d.data[0, 0, 0], _fid(), atol=1e-6)
    assert any("no HZPPPM and dwell time" in note for note in d.provenance.notes)
    with pytest.raises(gu.IncompleteError, match="states no SpectrometerFrequency, DwellTime"):
        gu.load(f, complete=True)
    [d] = gu.load(f, given={"SpectrometerFrequency": 123.25, "DwellTime": 2.5e-4}, complete=True)
    assert d.missing == () and d.dwelltime == 2.5e-4 and d.spectrometer_frequency == 123.25
    assert d.meta["SpectralWidth"] == pytest.approx(4000) and d.meta["FieldStrength"] > 2.8
    assert d.meta.source["SpectrometerFrequency"] == "given"


def test_given_fills_only_what_the_file_lacks(tmp_path):
    f = tmp_path / "metab.RAW"
    _raw(f, [" $SEQPAR", " hzpppm= 123.25", " dwellTime= 0.00025", " $END", " $NMID $END"], _fid())
    [d] = gu.load(f, given={"SpectrometerFrequency": 99.0, "EchoTime": 0.03})
    assert d.spectrometer_frequency == 123.25 and d.meta["EchoTime"] == 0.03
    assert d.meta.source["EchoTime"] == "given"


def test_text_fid_columns(tmp_path):
    f = tmp_path / "signal.txt"
    f.write_text("".join(f"{z.real:15.6E}{z.imag:15.6E}\n" for z in _fid()))
    [d] = gu.load(f)
    assert d.shape == (1, 1, 1, 64) and d.name == "signal" and d.provenance.format == "text_fid"
    assert d.missing == ("SpectrometerFrequency", "ResonantNucleus", "DwellTime")
    assert d.dwelltime is None and d.time_axis is None and d.meta["NumberOfPoints"] == 64
    assert np.allclose(d.data[0, 0, 0], _fid(), atol=1e-6)
    assert "unknown nucleus at unknown frequency, unknown dwell time" in repr(d)
    missing = "states no SpectrometerFrequency, ResonantNucleus, DwellTime"
    with pytest.raises(gu.IncompleteError, match=missing):
        gu.load(f, complete=True)
    given = {"SpectrometerFrequency": 123.25, "ResonantNucleus": "1H", "SpectralWidth": 4000}
    [d] = gu.load(f, given=given, complete=True)
    assert d.missing == () and d.nucleus == "1H" and d.dwelltime == pytest.approx(2.5e-4)
    assert d.meta["SpectrometerFrequency"] == [123.25] and d.meta["ResonantNucleus"] == ["1H"]
    assert d.meta.source["DwellTime"] == "derived: 1 / SpectralWidth"
    assert d.meta["FieldStrength"] == pytest.approx(123.25 / 42.576, rel=1e-3)
    assert "1H at 123.2500 MHz, dwell 0.25 ms" in repr(d)


def test_text_fid_column_pairs_separated_by_commas(tmp_path):
    f = tmp_path / "two.csv"
    a, b = _fid(), 2 * _fid()
    rows = zip(a, b, strict=True)
    f.write_text("".join(f"{x.real},{x.imag},{y.real},{y.imag}\n" for x, y in rows))
    first, second = gu.load(f)
    assert first.name == "two columns 1-2" and second.name == "two columns 3-4"
    assert np.allclose(first.data[0, 0, 0], a) and np.allclose(second.data[0, 0, 0], b)


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
