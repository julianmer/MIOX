####################################################################################################
#                                       test_fit_results.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Fit-result readers on synthetic files: LCModel TABLE, COORD and CSV, TARQUIN results,   #
#          fit and CSV, PyAMARES results, Osprey containers, and R data as spant saves it. No test #
#          data needed.                                                                            #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import gzip
import struct

import numpy as np
import scipy.io

# own
import miox
from miox.readers.spant_rds import RSerial

TABLES = """ LCModel (Version 6.3-1B)
 test title

$$CONC  3 lines in following concentration table = NCONC+1
    Conc.  %SD /Cr+PCr  Metabolite
    0.500   4%   1.000 Cr
 1.20E-02 999%   0.024 NAA

$$MISC  3 lines in following misc. output table
  FWHM = 0.050 ppm    S/N =  20
  Data shift =-0.010 ppm
  Ph:  10 deg      -2.0 deg/ppm

$$INPU  6 lines in following table of input changes:
 HZPPPM =1.232000e+02, DELTAT=2.500000D-04, NUNFI
   L=4096
 FILBAS = '/data/basis sets/a very long name for
   a basis.basis'
 FILRAW = '/data/a/very/long/path/to/the/raws/fil
   e.RAW'
"""


def test_lcmodel_table(tmp_path):
    f = tmp_path / "fit.TABLE"
    f.write_text(TABLES)
    [r] = miox.load(f)
    assert isinstance(r, miox.FitResult) and r.metabolites == ["Cr", "NAA"]
    assert r.row("NAA") == {"metabolite": "NAA", "concentration": 0.012, "sd_percent": 999.0,
                            "/Cr+PCr": 0.024}
    assert r.diagnostics == {"FWHM (ppm)": 0.05, "S/N": 20.0, "Data shift (ppm)": -0.01,
                             "Ph (deg)": 10.0, "Ph (deg/ppm)": -2.0}
    assert r.meta["SpectrometerFrequency"] == [123.2] and r.meta["DwellTime"] == 0.00025
    assert r.meta["NumberOfPoints"] == 4096  # a namelist line, cut inside NUNFIL
    inputs = r.meta.vendor["inputs"]
    assert inputs["FILBAS"] == "/data/basis sets/a very long name for a basis.basis"
    assert inputs["FILRAW"] == "/data/a/very/long/path/to/the/raws/file.RAW"


def test_lcmodel_coord(tmp_path):
    f = tmp_path / "fit.COORD"
    f.write_text(
        " LCModel (Version 6.3-1B)\n 2 lines in following concentration table = NCONC+1\n"
        "    Conc.  %SD /Cr+PCr  Metabolite\n    0.500   4%   1.000 Cr\n"
        "   1 lines in following misc. output table\n  FWHM = 0.050 ppm    S/N =  20\n"
        "    4 points on ppm-axis = NY\n  4.0 3.0\n  2.0 1.0\n NY phased data points follow\n"
        "  1.0 2.0 3.0 4.0\n NY points of the fit to the data follow\n  1.5 2.5 3.5 4.5\n"
        " NY background values follow\n  0.5 0.5 0.5 0.5\n Cr       Conc. = 5.00E-01\n"
        "  1.5 2.5 3.5 4.5\n"
    )
    [r] = miox.load(f)
    assert list(r.curves) == ["ppm", "data", "fit", "baseline", "Cr"]
    assert np.allclose(r.curves["ppm"], [4, 3, 2, 1])
    assert np.allclose(r.curves["Cr"], r.curves["fit"])


def test_lcmodel_csv_voxels(tmp_path):
    f = tmp_path / "fit.CSV"
    f.write_text(
        "Row, Col, Cr, Cr %SD, Cr/Cr+PCr, NAA, NAA %SD, NAA/Cr+PCr\n"
        "  1,   1, 0.500,  4, 1.000, 0.300, 5, 0.600\n  1,   2, 0.400,  6, 1.000, 0.200, 9, 0.500\n"
    )
    results = miox.load(f)
    assert [r.voxel for r in results] == [(1, 1), (1, 2)]
    assert results[1].row("NAA") == {"metabolite": "NAA", "concentration": 0.2, "sd_percent": 9.0,
                                      "/Cr+PCr": 0.5}


def test_tarquin_results_and_fit(tmp_path):
    results = tmp_path / "results.txt"
    results.write_text(
        "Signal     Conc (au)       %SD   SD (au)\n----------------------------------------\n"
        "NAA         0.002788     5.361 0.0001495\nTau            0.000       inf 6.493e-05\n\n"
        "Fit quality\n-----------\nQ                : 4.048\n\n"
        "Fitting parameters\n------------------\nRef (ppm)    : 4.668\n\n"
        "Data parameters\n---------------\nFs (Hz) : 4000.\nFt (Hz) : 1.233e+08\nN       : 2048\n"
        "TE (s)  : 0.03000\n\nTARQUIN version 4.3.5\n"
    )
    [r] = miox.load(results)
    assert r.row("Tau")["sd_percent"] == np.inf and r.row("NAA")["SD (au)"] == 0.0001495
    assert r.diagnostics == {"Q": 4.048} and r.meta["ReferencePPM"] == 4.668
    assert r.meta["DwellTime"] == 0.00025 and r.meta["NumberOfPoints"] == 2048
    fit = tmp_path / "fit.txt"
    block = "PPMScale,Data,Fit,Baseline,NAA\n2.0,1.0,0.9,0.1,0.9\n1.9,0.5,0.4,0.1,0.4\n"
    voxels = ("Row : 1, Col : 1, Slice : 1\n", "Row : 1, Col : 2, Slice : 1\n")
    fit.write_text(voxels[0] + block + voxels[1] + block)
    fits = miox.load(fit)
    assert [f.voxel for f in fits] == [(1, 1, 1), (1, 2, 1)]
    assert list(fits[0].curves) == ["ppm", "data", "fit", "baseline", "NAA"]


def test_pyamares_result(tmp_path):
    f = tmp_path / "result.csv"
    f.write_text("name,amplitude,sd,CRLB(%),chem shift(ppm),g (%),SNR\nPCr,4.4,0.1,2.3,0.0,,30\n")
    [r] = miox.load(f)
    assert r.rows == [{"metabolite": "PCr", "concentration": 4.4, "sd": 0.1, "sd_percent": 2.3,
                       "chem shift(ppm)": 0.0, "g (%)": None, "SNR": 30.0}]


def _fid(frequency: float = 100, n: int = 64) -> np.ndarray:
    t = np.arange(n) * 2.5e-4
    return np.exp(2j * np.pi * frequency * t - t / 0.05)


def test_osprey_container(tmp_path):
    def struct(frequency):
        return {"fids": np.conj(_fid(frequency))[:, None], "sz": np.array([64, 1]),
                "txfrq": 123.2e6, "dwelltime": 2.5e-4, "te": 35.0, "tr": 2000.0,
                "dims": {"t": 1, "coils": 0, "averages": 0, "subSpecs": 0, "extras": 0}}
    raw = np.empty((1, 2), dtype=object)
    raw[0, 0], raw[0, 1] = struct(100), struct(110)
    ref = np.empty((1, 1), dtype=object)
    ref[0, 0] = struct(0)
    container = {"raw": raw, "raw_ref": ref, "fit": {"method": "LCModel"}, "ver": {"Osp": "x"}}
    scipy.io.savemat(tmp_path / "job.mat", {"MRSCont": container})
    datasets = miox.load(tmp_path / "job.mat")
    assert [d.name for d in datasets] == ["raw_1", "raw_2", "raw_ref_1"]
    assert np.allclose(datasets[1].data.ravel(), _fid(110))
    assert datasets[0].meta["EchoTime"] == 0.035
    assert any("fit results not read" in note for note in datasets[0].provenance.notes)


def _r(kind: int, *words: int, attributes: bool = False) -> bytes:
    """An item of R's XDR serialization: flags (type, attribute bit), then integers."""
    return struct.pack(">i", kind | (0x200 if attributes else 0)) + struct.pack(
        f">{len(words)}i", *words)


def _text(value: str) -> bytes:
    return _r(9, len(value)) + value.encode()


def _symbol(name: str) -> bytes:
    return _r(1) + _text(name)


def test_r_serialization_version_3_with_compact_sequence_na_and_factor():
    """list(seq = 1:4, na = c(NA, 2.5), f = factor(c("b", "a"))) as saveRDS writes it (version 3:
    the native encoding; 1:4 as an ALTREP compact sequence)."""
    header = b"X\n" + struct.pack(">iii", 3, 0x40400, 0x30500) + struct.pack(">i", 5) + b"UTF-8"
    pairlist = 2 | 0x400  # a node with a tag
    info = (_r(2) + _symbol("compact_intseq") + _r(2) + _symbol("base") + _r(2) + _r(13, 1, 13)
            + _r(254))
    sequence = _r(238) + info + _r(14, 3) + struct.pack(">3d", 4, 1, 1) + _r(254)
    missing = _r(14, 2) + struct.pack(">2d", float("nan"), 2.5)
    labels = _r(16, 2) + _text("a") + _text("b")
    factor = (_r(13, 2, 2, 1, attributes=True) + _r(pairlist) + _symbol("levels") + labels
              + _r(pairlist) + _symbol("class") + _r(16, 1) + _text("factor") + _r(254))
    names = _r(16, 3) + _text("seq") + _text("na") + _text("f")
    blob = (header + _r(19, 3, attributes=True) + sequence + missing + factor + _r(pairlist)
            + _symbol("names") + names + _r(254))
    value = RSerial(gzip.compress(blob)).read()
    assert value.plain() == {"seq": [1, 2, 3, 4], "na": [None, 2.5], "f": ["b", "a"]}


def test_tarquin_csv(tmp_path):
    f = tmp_path / "fit.csv"
    f.write_text(
        "Signal amplitudes\nRow,Col,Slice,NAA,NAAG,TNAA\n1,1,1,2.0,0.5,2.5\n"
        "CRLBs (standard deviation)\nRow,Col,Slice,NAA,NAAG,TNAA\n1,1,1,0.1,0.5,0.1\n"
        "Fit diagnostics\nRow,Col,Slice,Q,SNR\n1,1,1,1.5,40\n"
        "Basis shifts and dampings\nRow,Col,Slice,name,group,shift (PPM),damping (Hz)\n"
        "1,1,1,NAA,1,0.01,2.0\n1,1,1,NAAG,1,-0.02,3.0\n"
        "Geometry parameters\nRows,1\nPixelSpacing,Unknown\n"
        "Data parameters\nFs (Hz),4.0e+03\nFt (Hz),1.232e+08\nN,2048\nTE (s),3.0e-02\n\n"
        "TARQUIN version,4.3.11\n"
    )
    [r] = miox.load(f)
    assert r.provenance.format == "tarquin_csv" and r.voxel == (1, 1, 1)
    assert r.row("NAA") == {"metabolite": "NAA", "concentration": 2.0, "sd": 0.1,
                            "sd_percent": 5.0, "group": 1.0, "shift (PPM)": 0.01,
                            "damping (Hz)": 2.0}
    assert r.row("TNAA")["concentration"] == 2.5 and "shift (PPM)" not in r.row("TNAA")
    assert r.diagnostics == {"Q": 1.5, "SNR": 40.0}
    assert r.meta["SpectrometerFrequency"] == [123.2] and r.meta["DwellTime"] == 0.00025
    assert r.meta["NumberOfPoints"] == 2048 and r.meta["EchoTime"] == 0.03
    assert r.meta.vendor["Geometry parameters"]["PixelSpacing"] == "Unknown"
