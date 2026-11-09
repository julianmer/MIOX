####################################################################################################
#                                         test_minispec.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 09/11/26                                                                                #
#                                                                                                  #
# Purpose: Bruker minispec relaxometry: the .dps decay with the fit of the report that names it   #
#          and the CONTIN distribution beside it, a CONTIN export and a report on their own, the   #
#          latest of several reports, and the sniff of synthetic files.                            #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import numpy as np
import pytest

# own
import miox
from miox.cli import CLI
from miox.quicklook import page
from miox.viewer import RelaxationView, view_class

FOLDER = "minispec/2024-10-08 T-21C"
DECAY = "chitosan 5 mg-L_1_20241008093123"
TWICE = "chitosan 1000 mg-L_46_20241008135816"  # in the fit report and the CONTIN report


@pytest.mark.data
def test_decay_with_its_report_and_distribution(public_data):
    folder = public_data / FOLDER
    if not folder.is_dir():
        pytest.skip("minispec samples absent")
    path = folder / f"{DECAY}.dps"
    s = miox.sniff(path)
    assert (s.format, s.kind, s.vendor) == (miox.Format.MINISPEC_DPS, miox.Kind.RELAXOMETRY,
                                            "bruker")
    assert [c.name for c in s.companions] == ["08074045.mdt", f"{DECAY}_CONTIN_1008140714.txt"]
    [d] = miox.load(path)
    assert isinstance(d, miox.RelaxationData) and d.sequence == "CPMG"
    table = np.loadtxt(path, delimiter="\t")
    assert d.n_points == 7500 and np.allclose(d.times, table[:, 1] * 1e-3)
    assert np.array_equal(d.curves[0], table[:, 2])
    assert d.meta["EchoTime"] == pytest.approx(0.002)  # 2 x Pulse Separ. 1.000 ms
    assert d.meta["SampleID"] == "chitosan 5 mg-L_1"
    assert d.fit["amplitude"] == [82.146] and d.fit["time"] == [pytest.approx(2.4761)]
    assert d.fit["offset"] == 0.031 and d.fit["time_error"] == [pytest.approx(0.0004)]
    assert d.meta["RelaxationFitTime"] == d.fit["time"]
    times, amplitude = d.distribution
    assert len(times) == 500 and np.all(np.diff(times) > 0)
    assert times[0] == pytest.approx(0.01e-3, rel=1e-3) and times[-1] == pytest.approx(15.0)
    assert amplitude[times > 1.0].max() > 100  # the water peak near 2.5 s
    assert "7500 points" in CLI()._describe(d)
    assert d.to_dict()["distribution_points"] == 500


@pytest.mark.data
def test_fit_report_and_contin_report_of_one_decay(public_data):
    """08074045.mdt fits the decay, 08161550.mdt holds the CONTIN results of the same file:
    the fit comes from the former, the CONTIN components go to the vendor header."""
    folder = public_data / FOLDER
    if not folder.is_dir():
        pytest.skip("minispec samples absent")
    path = folder / f"{TWICE}.dps"
    assert [c.name for c in miox.sniff(path).companions][:2] == ["08074045.mdt", "08161550.mdt"]
    [d] = miox.load(path)
    assert d.fit["report"] == "08074045.mdt" and d.fit["time"] == [pytest.approx(1.7989)]
    contin = d.meta.vendor["contin_results"]
    assert contin["report"] == "08161550.mdt"
    assert contin["components_T2_s_error_s_amplitude"][0][0] < 1e-3  # the 0.07 ms artefact
    protocols = miox.load(folder / "08161550.mdt")
    one = next(p for p in protocols if p.name == TWICE)
    components = contin["components_T2_s_error_s_amplitude"]
    assert one.meta["RelaxationFitTime"] == [c[0] for c in components]


@pytest.mark.data
def test_contin_export_and_report_on_their_own(public_data):
    folder = public_data / FOLDER
    if not folder.is_dir():
        pytest.skip("minispec samples absent")
    contin = folder / f"{DECAY}_CONTIN_1008140714.txt"
    s = miox.sniff(contin)
    assert (s.format, s.kind) == (miox.Format.MINISPEC_CONTIN, miox.Kind.RELAXOMETRY)
    [d] = miox.load(contin)
    assert d.name == DECAY and d.distribution is not None
    report = folder / "08074045.mdt"
    assert miox.sniff(report).format == miox.Format.MINISPEC_REPORT
    protocols = miox.load(report)
    assert all(isinstance(p, miox.Protocol) for p in protocols)
    # 50 titles, but the last fit block breaks off after its amplitude (no T2): 49 fits; the
    # other protocols are the CONTIN result blocks of the same report
    assert sum("SampleID" in p.meta for p in protocols) == 49
    first = next(p for p in protocols if p.name == DECAY)
    assert first.meta["RelaxationFitTime"] == [pytest.approx(2.4761)]
    assert first.meta["EchoTime"] == pytest.approx(0.002)


@pytest.mark.data
def test_quick_look_page_and_viewer(public_data):
    folder = public_data / FOLDER
    if not folder.is_dir():
        pytest.skip("minispec samples absent")
    [d] = miox.load(folder / f"{DECAY}.dps")
    html = page([d], DECAY)
    for tab in ("Decay", "Log", "T2 distribution", "Details"):
        assert f">{tab}</label>" in html
    assert "T2 2476.1 ± 0.4 ms" in html and "below the first stored time" in html
    assert view_class(d) is RelaxationView
    view = RelaxationView(d)
    assert len(view.axes) == 3 and view.axes[1].get_yscale() == "log"


def test_synthetic_files_sniff(tmp_path):
    rows = "\n".join(f"{i}\t{2.0 + 2.0 * i:.10f}\t{90.0 * np.exp(-i / 50):.10f}" for i in range(20))
    (tmp_path / "a.dps").write_text(rows + "\n", newline="\r\n")
    report = ("------------------------\r\nT2 Curve Fit Results\r\n09.11.2026 10:00\r\n"
              "Pulse Separ.  : 1.000 ms\r\nData Points   : 20\r\n------------------------\r\n"
              "SampleId: a\r\n------------------------\r\nTitle of this Data File:\r\na\r\n"
              "------------------------\r\nAmplitude  :   90.000  %\r\n         +/-    0.100  %\r\n"
              "T2         :    100.0 ms\r\n         +/-      1.0 ms\r\n"
              "Offset     :    0.000  %\r\n         +/-    0.001  %\r\n")
    (tmp_path / "r.mdt").write_bytes(report.encode())
    (tmp_path / "a_CONTIN_1.txt").write_text("1000.0000000000\t0.0000000000\n1.0000000000\t5.0\n")
    s = miox.sniff(tmp_path / "a.dps")
    assert s.format == miox.Format.MINISPEC_DPS
    assert [c.name for c in s.companions] == ["r.mdt", "a_CONTIN_1.txt"]
    [d] = miox.load(tmp_path / "a.dps")
    assert d.fit["time"] == [pytest.approx(0.1)] and d.fit["amplitude"] == [90.0]
    assert d.meta["EchoTime"] == pytest.approx(0.002)
    assert d.distribution[0][0] == pytest.approx(1e-3)
    assert miox.sniff(tmp_path / "r.mdt").format == miox.Format.MINISPEC_REPORT
    assert miox.sniff(tmp_path / "a_CONTIN_1.txt").format == miox.Format.MINISPEC_CONTIN
    (tmp_path / "lone.dps").write_text(rows + "\n")
    [lone] = miox.load(tmp_path / "lone.dps")
    assert lone.fit is None and lone.meta["EchoTime"] == pytest.approx(0.002)
    assert "no fit report names this file" in lone.provenance.notes[0]
    html = page([lone], "lone")
    assert "No fit" in html and "T2 distribution" not in html
