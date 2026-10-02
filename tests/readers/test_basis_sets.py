####################################################################################################
#                                        test_basis_sets.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Basis-set readers on synthetic files: LCModel .BASIS, FSL-MRS JSON, SpinWizard folders, #
#          MATLAB structs (Osprey, INSPECTOR, MARSS, ProFit, FID-A) and folders of one file per    #
#          metabolite. No test data needed.                                                        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import json

import numpy as np
import pytest
import scipy.io

# own
import miox as gu


def _fid(frequency: float = 100, n: int = 64) -> np.ndarray:
    t = np.arange(n) * 2.5e-4
    return np.exp(2j * np.pi * frequency * t - t / 0.05)


def _fsl_json(path, name: str, fid: np.ndarray, dwell: float = 2.5e-4) -> None:
    stored = np.conj(fid)  # FSL-MRS convention
    basis = {"basis_re": stored.real.tolist(), "basis_im": stored.imag.tolist(),
             "basis_dwell": dwell, "basis_centre": 123.2, "basis_width": None, "basis_name": name}
    path.write_text(json.dumps({"basis": basis, "meta": {"SimVersion": "test"}}))


def test_lcmodel_basis(tmp_path):
    f = tmp_path / "x.BASIS"
    lines = [" $SEQPAR", " HZPPPM = 123.2,", " ECHOT = 30.0", " $END", " $BASIS1",
             " IDBASI = 'test',", " FMTBAS = '(2E15.6)',", " BADELT = 0.00025,", " NDATAB = 64",
             " $END"]
    for name, frequency, shift in (("NAA", 100, 0), ("Cr", -50, 5)):
        spectrum = np.roll(np.fft.fft(_fid(frequency)), shift)
        lines += [" $BASIS", f" ID = '{name}',", f" METABO = '{name}',", f" ISHIFT = {shift}",
                  " $END"]
        lines += [f"{z.real:15.6E}{z.imag:15.6E}" for z in spectrum]
    f.write_text("\n".join(lines) + "\n")
    [b] = gu.load(f)
    assert isinstance(b, gu.BasisSet) and b.metabolites == ["NAA", "Cr"]
    assert b.spectrometer_frequency == pytest.approx(123.2) and b.dwelltime == pytest.approx(2.5e-4)
    assert b.meta["EchoTime"] == pytest.approx(0.03)
    assert np.allclose(b.fid("NAA"), _fid(100), atol=1e-5)
    assert np.allclose(b.fid("Cr"), _fid(-50), atol=1e-5)


def test_fsl_mrs_json(tmp_path):
    f = tmp_path / "NAA.json"
    _fsl_json(f, "NAA", _fid())
    [b] = gu.load(f)
    assert isinstance(b, gu.BasisSet) and b.metabolites == ["NAA"]
    assert b.spectrometer_frequency == pytest.approx(123.2) and b.dwelltime == pytest.approx(2.5e-4)
    assert np.allclose(b.fid("NAA"), _fid())
    assert b.meta.vendor["meta"] == {"SimVersion": "test"}


def test_folder_of_fsl_mrs_json(tmp_path):
    for name, frequency in (("NAA", 100), ("Cr", -50), ("GABA", 20)):
        _fsl_json(tmp_path / f"{name}.json", name, _fid(frequency))
    [b] = gu.load(tmp_path)
    assert b.metabolites == ["Cr", "GABA", "NAA"] and b.shape == (1, 1, 1, 64, 3)
    assert np.allclose(b.fid("GABA"), _fid(20))
    assert set(b.meta.vendor) == {"Cr", "GABA", "NAA"}
    assert b.meta.source["SpectrometerFrequency"].startswith("Cr: ")


def test_folder_of_mismatched_files(tmp_path):
    _fsl_json(tmp_path / "NAA.json", "NAA", _fid())
    _fsl_json(tmp_path / "Cr.json", "Cr", _fid(), dwell=5e-4)
    with pytest.raises(gu.ReaderError, match="differs"):
        gu.load(tmp_path)


def test_spinwizard_folder(tmp_path):
    (tmp_path / "BasisSetParameters.txt").write_text("  123.20000\n  4.00000\n")
    (tmp_path / "BasisRFOffset.txt").write_text("  4.6500\n")
    for name, frequency in (("NAA", 100), ("Cr", -50)):
        stored = np.conj(_fid(frequency))
        (tmp_path / name).write_text("".join(f"{z.real}\t\t{z.imag}\n" for z in stored))
    [b] = gu.load(tmp_path)
    assert b.metabolites == ["Cr", "NAA"]
    assert b.spectrometer_frequency == pytest.approx(123.2) and b.dwelltime == pytest.approx(2.5e-4)
    assert np.allclose(b.fid("NAA"), _fid(100))
    assert b.meta.vendor["BasisRFOffset.txt"] == {"1": "4.6500"} and b.meta["ReferencePPM"] == 4.65


def _cell(*items) -> np.ndarray:
    cell = np.empty((1, len(items)), dtype=object)
    for i, item in enumerate(items):
        cell[0, i] = item
    return cell


def test_osprey_basis_with_sub_spectra(tmp_path):
    fids = np.stack([np.stack([_fid(100), _fid(110)], -1), np.stack([_fid(-50), _fid(-40)], -1)], 1)
    basis = {"fids": np.conj(fids), "name": _cell("NAA", "Cr"), "Bo": 2.89, "te": 30.0,
             "dwelltime": 2.5e-4, "centerFreq": 3.0}
    scipy.io.savemat(tmp_path / "basis.mat", {"BASIS": basis})
    [b] = gu.load(tmp_path / "basis.mat")
    assert b.metabolites == ["NAA", "Cr"] and b.shape == (1, 1, 1, 64, 2, 2)
    assert b.spectrometer_frequency == pytest.approx(2.89 * 42.577)
    assert b.meta["EchoTime"] == pytest.approx(0.03) and b.meta["ReferencePPM"] == 3.0
    assert np.allclose(b.fid("Cr")[:, 1], _fid(-40))


def test_inspector_basis(tmp_path):
    data = _cell(*(_cell(name, 1.5, 0.025, np.conj(_fid(f))[:, None], "") for name, f in
                   (("NAA", 100), ("Cr", -50))))
    lcm = {"data": data, "sw_h": 4000, "sf": 123.2, "ppmCalib": 4.65}
    scipy.io.savemat(tmp_path / "inspector.mat", {"lcmBasis": lcm})
    [b] = gu.load(tmp_path / "inspector.mat")
    assert b.metabolites == ["NAA", "Cr"] and b.dwelltime == pytest.approx(2.5e-4)
    assert b.spectrometer_frequency == pytest.approx(123.2) and b.meta["ReferencePPM"] == 4.65
    assert np.allclose(b.fid("NAA"), _fid(100))


def test_folder_of_marss_files(tmp_path):
    for name, frequency in (("NAA", 100), ("Cr", -50), ("GABA", 20)):
        expt = {"sw_h": 4000, "sf": 123.2, "nspecC": 64, "ppmCalib": 4.65,
                "fid": np.conj(_fid(frequency))[:, None]}
        scipy.io.savemat(tmp_path / f"{name}.mat", {"exptDat": expt})
    [b] = gu.load(tmp_path)
    assert b.metabolites == ["Cr", "GABA", "NAA"] and b.provenance.format == "basis_folder"
    assert b.meta["ReferencePPM"] == 4.65 and b.meta.fields["ReferencePPM"] == "Cr/exptDat/ppmCalib"
    assert np.allclose(b.fid("Cr"), _fid(-50))


def test_profit_struct(tmp_path):
    fids = _cell(np.conj(_fid(100))[None, :], np.conj(_fid(-50))[None, :])
    basis = {"fid": fids, "met": _cell("NAA", "Cr"), "bw": 4000, "dt": 2.5e-4, "f0": 123.2e6,
             "pts": 64, "Bo": 2.89}
    scipy.io.savemat(tmp_path / "profit.mat", {"basis": basis})
    [b] = gu.load(tmp_path / "profit.mat")
    assert b.metabolites == ["NAA", "Cr"] and b.spectrometer_frequency == pytest.approx(123.2)
    assert np.allclose(b.fid("Cr"), _fid(-50))


def test_fida_struct_dimensions(tmp_path):
    fids = np.stack([np.stack([_fid(100 + 10 * c + a) for a in range(3)], -1) for c in range(4)], 1)
    out = {"fids": np.conj(fids), "sz": np.array([64, 4, 3]), "txfrq": 123.2e6,
           "dwelltime": 2.5e-4, "te": 30.0, "tr": 2000.0, "nucleus": "1H",
           "dims": {"t": 1, "coils": 2, "averages": 3, "subSpecs": 0, "extras": 0}}
    scipy.io.savemat(tmp_path / "scan.mat", {"out": out})
    [d] = gu.load(tmp_path / "scan.mat")
    assert d.dims == ("x", "y", "z", "t", "DIM_COIL", "DIM_DYN") and d.shape == (1, 1, 1, 64, 4, 3)
    assert d.meta["EchoTime"] == pytest.approx(0.03) and d.meta["RepetitionTime"] == 2.0
    assert np.allclose(d.data[0, 0, 0, :, 2, 1], _fid(121))
