####################################################################################################
#                                          test_jcamp.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 09/10/26                                                                                #
#                                                                                                  #
# Purpose: NMR JCAMP-DX reader on synthetic files: a 1D FID as NTUPLES pages, an XYDATA spectrum   #
#          in Hz written from low to high shift, a 2D FID with page times, an XY table spectrum,   #
#          an FID kept as stored under $REVERSE= yes, and the sniffing of JCAMP-DX NMR files apart #
#          from TopSpin parameter files.                                                           #
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

HEADER = """##TITLE= test
##JCAMP-DX= 5.01
##DATA TYPE= {kind}
##DATA CLASS= {cls}
##.OBSERVE FREQUENCY= 100.0
##.OBSERVE NUCLEUS= ^1H
##.SOLVENT NAME= CDCl3
##.AVERAGES= 8
"""


def lines(values, x0=0) -> str:
    """AFFN lines of four values, each led by the index of its first point."""
    return "\n".join(
        f"{x0 + i} " + " ".join(str(v) for v in values[i : i + 4]) for i in range(0, len(values), 4)
    )


def write(path: Path, text: str) -> Path:
    path.write_text(text + "\n##END=\n")
    return path


def fid_text(real, imag, extra: str = "") -> str:
    return HEADER.format(kind="NMR FID", cls="NTUPLES") + extra + f"""\
##.SHIFT REFERENCE= INTERNAL, TMS, 1, 10.0
##NTUPLES= NMR FID
##VAR_NAME= TIME, FID/REAL, FID/IMAG
##SYMBOL= X, R, I
##VAR_DIM= 8, 8, 8
##UNITS= SECONDS, ARBITRARY UNITS, ARBITRARY UNITS
##FACTOR= 0.001, 2, 2
##FIRST= 0, 20, 0
##LAST= 0.007, -2, 14
##PAGE= N=1
##DATA TABLE= (X++(R..R)), XYDATA
{lines(real)}
##PAGE= N=2
##DATA TABLE= (X++(I..I)), XYDATA
{lines(imag)}
##END NTUPLES= NMR FID"""


def test_fid_pages(tmp_path):
    real, imag = [10, 8, 6, 4, 2, 1, 0, -1], [0, 1, 2, 3, 4, 5, 6, 7]
    path = write(tmp_path / "fid.jdx", fid_text(real, imag))
    sniff = miox.sniff(path)
    assert (sniff.format, sniff.kind) == (miox.Format.JCAMP_NMR, miox.Kind.SPECTROSCOPY)
    [d] = miox.load(path)
    assert isinstance(d, miox.MRSData) and d.shape == (1, 1, 1, 8)
    assert np.array_equal(d.data[0, 0, 0], 2 * (np.array(real) - 1j * np.array(imag)))
    assert d.dwelltime == pytest.approx(1e-3)
    # point 1 at 10 ppm, the carrier at point N / 2 + 1 = 5, 4 points of 1000 / 8 Hz further
    assert d.meta["ReferencePPM"] == pytest.approx(10.0 - 4 * 125.0 / 100.0)
    assert d.meta["ResonantNucleus"] == ["1H"] and d.meta["Solvent"] == "CDCl3"
    assert d.meta["NumberOfAverages"] == 8


def test_reversed_fid_is_kept_as_stored(tmp_path):
    real, imag = [10, 8, 6, 4, 2, 1, 0, -1], [0, 1, 2, 3, 4, 5, 6, 7]
    [d] = miox.load(write(tmp_path / "fid.jdx", fid_text(real, imag, "##$REVERSE= yes\n")))
    assert np.array_equal(d.data[0, 0, 0], 2 * (np.array(real) + 1j * np.array(imag)))


def test_xydata_spectrum_in_hz_rising(tmp_path):
    y = [0, 1, 5, 1, 0, 0, 2, 0]
    text = HEADER.format(kind="NMR SPECTRUM", cls="XYDATA") + f"""##XUNITS= HZ
##YUNITS= ARBITRARY UNITS
##YFACTOR= 0.5
##FIRSTX= 0
##LASTX= 700
##DELTAX= 100
##NPOINTS= 8
##XYDATA= (X++(Y..Y))
{lines(y)}"""
    [image] = miox.load(write(tmp_path / "spectrum.dx", text))
    assert isinstance(image, miox.MRImage) and image.dims[3] == "frequency"
    assert np.array_equal(image.data[0, 0, 0], 0.5 * np.array(y[::-1]))  # shift falls
    assert image.meta["SpectralWidth"] == pytest.approx(800.0)
    assert image.meta["ReferencePPM"] == pytest.approx(3.0)  # point N / 2 of 7, 6, ..., 0 ppm


def test_2d_fid_page_times(tmp_path):
    rows = {0.0: ([4, 3, 2, 1], [0, 0, 1, 1]), 0.002: ([1, 2, 3, 4], [1, 1, 0, 0])}
    pages = "\n".join(
        f"##PAGE= T1={t}\n##DATA TABLE= (T2++({p}..{p})), PROFILE\n{lines(v)}"
        for t, (r, i) in rows.items() for p, v in (("R", r), ("I", i))
    )
    text = HEADER.format(kind="nD NMR FID", cls="NTUPLES") + f"""##NTUPLES= nD NMR FID
##VAR_NAME= TIME1, TIME2, FID/REAL, FID/IMAG
##SYMBOL= T1, T2, R, I
##.NUCLEUS= 13C, 1H
##VAR_DIM= 2, 4, 4, 4
##UNITS= SECONDS, SECONDS, ARBITRARY UNITS, ARBITRARY UNITS
##FACTOR= 0.002, 0.0005, 1, 1
##FIRST= 0, 0, 4, 0
##LAST= 0.002, 0.0015, 4, 0
##$SFO1= 100.0
##$SFO1= 25.1
{pages}
##END NTUPLES= nD NMR FID"""
    [d] = miox.load(write(tmp_path / "hsqc.jdx", text))
    assert d.shape == (1, 1, 1, 4, 2) and d.dim_tags[0] == "DIM_INDIRECT_0"
    second = np.array([1, 2, 3, 4]) - 1j * np.array([1, 1, 0, 0])
    assert np.array_equal(d.data[0, 0, 0, :, 1], second)
    assert d.dim_info[5]["header"]["IndirectTime"]["Value"] == [0.0, 0.002]
    assert d.meta["SpectrometerFrequency"] == [100.0, 25.1]
    assert d.meta["ResonantNucleus"] == ["1H", "13C"] and d.dwelltime == pytest.approx(5e-4)


def test_xy_table_spectrum(tmp_path):
    pairs = "\n".join(f"{x}, {y}" for x, y in ((0.0, 1), (0.5, 2), (1.0, 3), (1.5, 4)))
    text = HEADER.format(kind="NMR SPECTRUM", cls="PEAK TABLE") + f"""##XUNITS= PPM
##PEAK TABLE= (XY..XY)
{pairs}"""
    [image] = miox.load(write(tmp_path / "simulated.jdx", text))
    assert image.data[0, 0, 0].tolist() == [4, 3, 2, 1]
    assert image.meta["ReferencePPM"] == pytest.approx(0.5)


def test_topspin_parameter_file_is_not_jcamp_nmr(tmp_path):
    path = write(tmp_path / "acqus", "##TITLE= Parameter file, TopSpin 4.1\n##JCAMPDX= 5.0\n"
                 "##DATATYPE= Parameter Values\n##$TD= 16")
    assert miox.sniff(path).format != miox.Format.JCAMP_NMR
