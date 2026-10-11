####################################################################################################
#                                         test_topspin.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 09/10/26                                                                                #
#                                                                                                  #
# Purpose: Bruker TopSpin reader on synthetic experiments: a padded 1D FID with its processed      #
#          spectrum, a 2D ser stopped early, an experiment without data, a folder of experiments,  #
#          and the sniffing of TopSpin files apart from ParaVision scans.                          #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# own
import miox

ACQUS = {
    "TD": 200, "SW_h": 5000.0, "SW": 12.5, "SFO1": 400.1325, "BF1": 400.13, "NUC1": "<1H>",
    "AQ_mod": 3, "DTYPA": 0, "BYTORDA": 0, "DSPFVS": 20, "DECIM": 16, "GRPDLY": 67.98,
    "NS": 16, "PULPROG": "<zg30>", "SOLVENT": "<D2O>", "TE": 298.1, "FnMODE": 0,
    "PROBHD": "<5 mm BBO>", "NC": 0,
}
PROCS = {
    "SI": 8, "XDIM": 8, "SF": 400.13, "OFFSET": 10.0, "SW_p": 4000.0, "NC_proc": 1,
    "BYTORDP": 0, "DTYPP": 0, "AXNUC": "<1H>",
}


def jcamp(path: Path, parameters: dict, title: str = "Parameter file, TopSpin 4.1.4") -> None:
    lines = [f"##TITLE= {title}", "##JCAMPDX= 5.0", "##DATATYPE= Parameter Values",
             "##ORIGIN= Bruker BioSpin GmbH", "##OWNER= tester"]
    lines += [f"##${k}= {v}" for k, v in parameters.items()]
    path.write_text("\n".join(lines + ["##END="]) + "\n")


def interleave(fid: np.ndarray, padded: int) -> np.ndarray:
    """Real and imaginary parts as int32, each row padded to ``padded`` values."""
    rows = np.atleast_2d(fid)
    out = np.zeros((rows.shape[0], padded), dtype="<i4")
    out[:, 0 : 2 * rows.shape[1] : 2] = rows.real
    out[:, 1 : 2 * rows.shape[1] : 2] = rows.imag
    return out


def experiment(folder: Path, fid: np.ndarray, spectrum: np.ndarray | None = None) -> Path:
    folder.mkdir(parents=True)
    jcamp(folder / "acqus", ACQUS)
    interleave(fid, 256).tofile(folder / "fid")
    (folder / "pulseprogram").write_text("; zg30\nze\n")
    if spectrum is not None:
        pdata = folder / "pdata" / "1"
        pdata.mkdir(parents=True)
        jcamp(pdata / "procs", PROCS)
        spectrum.real.astype("<i4").tofile(pdata / "1r")
        spectrum.imag.astype("<i4").tofile(pdata / "1i")
    return folder


def test_fid_conjugated_and_cut_to_td(tmp_path):
    fid = (np.arange(100) * 10 + 1j * (np.arange(100) - 50)).astype(complex)
    spectrum = np.arange(8) * 100 - 300j
    folder = experiment(tmp_path / "study" / "10", fid, spectrum)
    sniff = miox.sniff(folder / "fid")
    assert (sniff.format, sniff.kind, sniff.vendor) == (miox.Format.TOPSPIN,
                                                        miox.Kind.SPECTROSCOPY, "bruker")
    d, image = miox.load(folder)
    assert isinstance(d, miox.MRSData) and d.shape == (1, 1, 1, 100)
    assert np.array_equal(d.data[0, 0, 0], fid.conj())
    assert d.name == "study_10" and d.dwelltime == pytest.approx(1 / 5000)
    assert d.meta["SpectrometerFrequency"] == [400.1325] and d.meta["ResonantNucleus"] == ["1H"]
    assert d.meta["GroupDelay"] == 67.98 and d.meta["SequenceName"] == "zg30"
    assert d.meta["NumberOfAverages"] == 16 and d.meta["Solvent"] == "D2O"
    assert d.meta["SampleTemperature"] == 298.1 and d.meta["RxCoil"] == "5 mm BBO"
    assert d.meta["SoftwareVersions"] == "TopSpin 4.1.4"
    assert d.meta["Manufacturer"] == "Bruker BioSpin GmbH"
    # carrier 400.1325 MHz on the axis of pdata/1, whose 0 ppm lies at SF 400.13 MHz
    assert d.meta["ReferencePPM"] == pytest.approx((400.1325 - 400.13) / 400.13 * 1e6)
    assert d.meta.vendor["pulseprogram"] == ["; zg30", "ze"]
    assert isinstance(image, miox.MRImage) and image.dims[3] == "frequency"
    assert np.array_equal(image.data[0, 0, 0], spectrum * 2)  # scaled by 2^NC_proc
    assert image.meta["ReferencePPM"] == pytest.approx(10.0 - 4000.0 / (2 * 400.13))
    assert image.meta["SpectralWidth"] == 4000.0 and image.name == "study_10_pdata1"
    [alone] = miox.load(folder / "pdata" / "1" / "1r")
    assert np.array_equal(alone.data, image.data)


def test_ser_cut_to_acquired_rows(tmp_path):
    folder = tmp_path / "hsqc" / "3"
    folder.mkdir(parents=True)
    jcamp(folder / "acqus", {**ACQUS, "TD": 4})
    jcamp(folder / "acqu2s", {"TD": 3, "SFO1": 100.62, "NUC1": "<13C>", "SW": 160.0,
                              "FnMODE": 6})
    rows = np.arange(8).reshape(4, 2) + 1j  # four stored rows, three acquired
    interleave(rows, 256).tofile(folder / "ser")
    [d] = miox.load(folder)
    assert d.shape == (1, 1, 1, 2, 3) and d.dim_tags[0] == "DIM_INDIRECT_0"
    assert np.array_equal(d.data[0, 0, 0], rows[:3].conj().T)
    assert d.meta["SpectrometerFrequency"] == [400.1325, 100.62]
    assert d.meta["ResonantNucleus"] == ["1H", "13C"]
    assert any("cut to the acquired ones" in n for n in d.provenance.notes)


def test_parameters_only_and_folder_of_experiments(tmp_path):
    study = tmp_path / "study"
    experiment(study / "1", np.ones(100, dtype=complex))
    empty = study / "2"
    empty.mkdir()
    jcamp(empty / "acqus", ACQUS)
    sniff = miox.sniff(study)
    assert sniff.format == miox.Format.TOPSPIN and len(sniff.companions) == 2
    fid, protocol = miox.load(study)
    assert isinstance(fid, miox.MRSData) and isinstance(protocol, miox.Protocol)
    assert protocol.name == "study_2" and protocol.meta["SequenceName"] == "zg30"


def test_paravision_scan_with_acqus_stays_paravision(tmp_path):
    scan = tmp_path / "12"
    scan.mkdir()
    jcamp(scan / "acqus", ACQUS)
    jcamp(scan / "acqp", {"ACQ_method": "<PRESS>"})
    (scan / "fid").write_bytes(b"\0" * 1024)
    for path in (scan, scan / "acqus", scan / "fid"):
        assert miox.sniff(path).format == miox.Format.BRUKER
