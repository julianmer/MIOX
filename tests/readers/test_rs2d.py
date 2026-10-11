####################################################################################################
#                                           test_rs2d.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 10/10/26                                                                                #
#                                                                                                  #
# Purpose: RS2D reader on a synthetic experiment folder: the FID (conjugated on reading) with its  #
#          header keys, a processed spectrum in Proc/0, a 2D FID, and the sniffing.                #
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


def header(path: Path, values: dict[str, object]) -> None:
    entries = []
    for key, value in values.items():
        texts = value if isinstance(value, list) else [value]
        inner = "".join(f"<value>{v}</value>" for v in texts)
        entries.append(f"<entry><key>{key}</key><value><name>{key}</name>{inner}</value></entry>")
    body = "".join(entries)
    path.write_text(f'<?xml version="1.0"?>\n<header>\n<params>{body}</params></header>')


BASE = {
    "BASE_FREQ_1": 400.0e6, "OBSERVED_FREQUENCY": 400.0016e6, "DWELL_TIME": 2.5e-4,
    "SPECTRAL_WIDTH": 10.0, "NUCLEUS_1": "1H", "NUCLEUS_2": "13C", "RECEIVER_COUNT": 1,
    "MATRIX_DIMENSION_1D": 8, "MATRIX_DIMENSION_2D": 1, "MATRIX_DIMENSION_3D": 1,
    "MATRIX_DIMENSION_4D": 1, "DATA_REPRESENTATION": ["COMPLEX", "REAL", "REAL", "REAL"],
    "ACQUISITION_MODE": ["COMPLEX", "ECHO_ANTIECHO", "REAL", "REAL"],
    "NUMBER_OF_AVERAGES": 16, "SOLVENT": "CDCl3", "SEQUENCE_NAME": "PulseAcquisition",
    "ACQUISITION_DATE": "2024-08-22T17:19:01+02:00",
}
FID = (np.arange(8) + 1j * np.arange(8, 0, -1)).astype(np.complex64)


def experiment(folder: Path, data: np.ndarray, **values) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    header(folder / "header.xml", {**BASE, **values})
    data.astype(">c8").tofile(folder / "data.dat")
    return folder


def test_fid_and_processed_spectrum(tmp_path):
    folder = experiment(tmp_path / "proton" / "1", FID)
    experiment(folder / "Proc" / "0", FID[::-1], STATE="1", MATRIX_DIMENSION_1D=8)
    assert miox.sniff(folder).format == miox.Format.RS2D
    assert miox.sniff(folder / "data.dat").format == miox.Format.RS2D
    fid, spectrum = miox.load(folder)
    assert isinstance(fid, miox.MRSData) and np.array_equal(fid.data[0, 0, 0], FID.conj())
    assert fid.dwelltime == pytest.approx(2.5e-4)
    assert fid.spectrometer_frequency == pytest.approx(400.0016)
    assert fid.meta["ReferencePPM"] == pytest.approx(4.0)  # (400.0016 - 400) / 400 in ppm
    assert fid.meta["NumberOfAverages"] == 16 and fid.meta["Solvent"] == "CDCl3"
    assert isinstance(spectrum, miox.MRImage) and spectrum.name == "proton_1_proc0"
    assert np.array_equal(spectrum.data[0, 0, 0], FID[::-1])  # as stored
    assert spectrum.meta["SpectralWidth"] == pytest.approx(10.0 * 400.0016)


def test_two_dimensional_fid(tmp_path):
    rows = np.stack([FID * (k + 1) for k in range(3)])  # (2D, 1D) as stored
    folder = experiment(tmp_path / "hsqc" / "17", rows, MATRIX_DIMENSION_2D=3)
    [d] = miox.load(folder)
    assert d.shape == (1, 1, 1, 8, 3) and d.dim_tags[0] == "DIM_INDIRECT_0"
    assert np.array_equal(d.data[0, 0, 0, :, 2], 3 * FID.conj())
    assert d.meta["ResonantNucleus"] == ["1H", "13C"]
    assert "ECHO_ANTIECHO" in d.dim_info[5]["info"]


def test_xml_without_partner_is_not_rs2d(tmp_path):
    path = tmp_path / "header.xml"
    path.write_text("<header><params></params></header>")
    assert miox.sniff(path).format != miox.Format.RS2D
