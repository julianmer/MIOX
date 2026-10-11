####################################################################################################
#                                           test_jeol.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 10/10/26                                                                                #
#                                                                                                  #
# Purpose: JEOL Delta reader on synthetic .jdf files: a 1D FID, a processed 1D spectrum that is    #
#          cut to the points kept, a 2D FID with two complex axes (truncated to the increments     #
#          acquired) and with a real indirect axis, and the sniffing of the file identifier.       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

# own
import miox

BASES = {"second": 28, "ppm": 26, "hertz": 13}
AXIS_TYPES = {None: 0, "real": 1, "complex": 3, "real_complex": 4}
HEADER_SIZE = 1536


def chars(text: str, size: int) -> bytes:
    return text.encode().ljust(size, b"\0")


def parameter(name: str, value) -> bytes:
    if isinstance(value, str):
        field, kind = chars(value, 16), 0
    else:
        field, kind = struct.pack("<d", value).ljust(16, b"\0"), 2
    return b"\0" * 4 + struct.pack("<h", 0) + b"\0" * 10 + field + struct.pack("<i", kind) \
        + chars(name, 28)


def jdf(path: Path, sections, points, kept, axes, types, unit, parameters,
        data_format: int = 1) -> Path:
    """A .jdf with one section per array in ``sections`` (each (indirect, direct) for 2D, in
    one block of 32 x 32 at most), ``points`` per axis, the kept (start, stop) offsets and the
    axis ranges ``axes``."""
    n = len(types)
    pad = lambda values, fill=0: list(values) + [fill] * (8 - len(values))  # noqa: E731
    header = bytearray()
    header += b"JEOL.NMR" + struct.pack(">bBHB", 1, 1, 2, n)
    header += bytes([0b10000000 if n == 1 else 0b11000000, data_format])
    header += struct.pack(">b", 25) + bytes(pad(range(1, 9)))
    header += bytes(pad([AXIS_TYPES[t] for t in types]))
    for i in range(8):
        header += bytes([0]) + struct.pack(">b", BASES[unit] if i < n else 0)
    header += chars("synthetic", 124) + bytes(4)
    header += struct.pack(">8I", *pad(points, 1)) + struct.pack(">8I", *pad([s for s, _ in kept]))
    header += struct.pack(">8I", *pad([e for _, e in kept]))
    header += struct.pack(">8d", *pad([a for a, _ in axes], 0.0))
    header += struct.pack(">8d", *pad([b for _, b in axes], 0.0))
    header += bytes(8) + chars("node", 16) + chars("site", 128) + chars("author", 128)
    header += chars("comment", 128) + b"".join(chars("", 32) for _ in range(8))
    header += struct.pack(">8d", *pad([400.0], 0.0)) + struct.pack(">8d", *pad([], 0.0))
    header += bytes(8) + bytes(3) + bytes([0]) + struct.pack(">II", 0, 0)
    body = b"".join(parameter(k, v) for k, v in parameters.items())
    body = struct.pack("<IIII", 64, 0, len(parameters), len(body)) + body
    data = b"".join(np.asarray(s, "<f8").tobytes() for s in sections)
    start = HEADER_SIZE + len(body)
    tail = struct.pack(">II8I8I", HEADER_SIZE, len(body), *([start] * 16))
    tail += struct.pack(">IIIII", start, 0, len(data), 0, 0) + struct.pack(">IIIII", 0, 0, 0, 0, 0)
    tail += bytes(8) + bytes(24)
    header += tail
    assert len(header) <= HEADER_SIZE, len(header)
    path.write_bytes(bytes(header).ljust(HEADER_SIZE, b"\0") + body + data)
    return path


PARAMETERS = {"x_freq": 400e6, "x_domain": "Proton", "x_offset": 5.0, "x_sweep": 1000.0,
              "scans": 4.0, "temp_get": 25.0, "solvent": "CDCl3", "experiment": "single_pulse"}
FID = np.exp(1j * 2 * np.pi * 100.0 * np.arange(16) / 1000.0 - 0.05 * np.arange(16))


def test_one_dimensional_fid(tmp_path):
    path = jdf(tmp_path / "fid.jdf", [FID.real, FID.imag], [16], [(0, 15)],
               [(0.0, 15e-3)], ["complex"], "second", PARAMETERS)
    sniff = miox.sniff(path)
    assert (sniff.format, sniff.kind) == (miox.Format.JEOL_JDF, miox.Kind.SPECTROSCOPY)
    [d] = miox.load(path)
    assert isinstance(d, miox.MRSData) and d.shape == (1, 1, 1, 16)
    assert np.allclose(d.data[0, 0, 0], FID)  # real + i imaginary, as stored
    assert d.dwelltime == pytest.approx(1e-3)
    assert d.spectrometer_frequency == pytest.approx(400.0)
    assert d.meta["ResonantNucleus"] == ["1H"] and d.meta["ReferencePPM"] == 5.0
    assert d.meta["NumberOfAverages"] == 4 and d.meta["Solvent"] == "CDCl3"
    assert d.meta["SampleTemperature"] == pytest.approx(298.15)


def test_spectrum_is_cut_to_the_points_kept(tmp_path):
    real = np.arange(16.0)
    path = jdf(tmp_path / "spectrum.jdf", [real, np.zeros(16)], [16], [(0, 9)],
               [(9.0, 0.0)], ["complex"], "ppm", PARAMETERS)
    [image] = miox.load(path)
    assert isinstance(image, miox.MRImage) and image.dims[3] == "frequency"
    assert image.shape == (1, 1, 1, 10)
    assert np.array_equal(image.data[0, 0, 0].real, real[:10])
    # 10 points over 9 ppm: 1 ppm per point, the centre point (index 5) at 4 ppm
    assert image.meta["ReferencePPM"] == pytest.approx(4.0)
    assert image.meta["SpectralWidth"] == pytest.approx(10 * 400.0)


def test_rising_spectrum_axis_is_reversed(tmp_path):
    real = np.arange(8.0)
    path = jdf(tmp_path / "rising.jdf", [real], [8], [(0, 7)], [(0.0, 7.0)], ["real"], "ppm",
               PARAMETERS)
    [image] = miox.load(path)
    assert np.array_equal(image.data[0, 0, 0], real[::-1])
    assert image.meta["ReferencePPM"] == pytest.approx(3.0)


def block(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(32, 32))


def test_two_complex_axes_give_two_rows_per_increment(tmp_path):
    sections = [block(i) for i in range(4)]
    parameters = dict(PARAMETERS, y_freq=100e6, y_domain="Carbon13")
    path = jdf(tmp_path / "hsqc.jdf", sections, [32, 32], [(0, 31), (0, 7)],
               [(0.0, 0.031), (0.0, 0.0105)], ["complex", "complex"], "second", parameters,
               data_format=2)
    [d] = miox.load(path)
    assert d.shape == (1, 1, 1, 32, 16)  # 8 increments, 2 rows each
    assert d.dim_tags[0] == "DIM_INDIRECT_0"
    assert np.allclose(d.data[0, 0, 0, :, 0], sections[0][0] + 1j * sections[1][0])
    assert np.allclose(d.data[0, 0, 0, :, 1], sections[2][0] + 1j * sections[3][0])
    assert np.allclose(d.data[0, 0, 0, :, 2], sections[0][1] + 1j * sections[1][1])
    times = d.dim_info[5]["header"]["IndirectTime"]["Value"]
    assert times[:4] == pytest.approx([0.0, 0.0, 0.0015, 0.0015])
    assert d.meta["SpectrometerFrequency"] == [400.0, 100.0]
    assert d.meta["ResonantNucleus"] == ["1H", "13C"]


def test_real_indirect_axis_gives_one_row_per_increment(tmp_path):
    sections = [block(10), block(11)]
    path = jdf(tmp_path / "hmbc.jdf", sections, [32, 32], [(0, 31), (0, 31)],
               [(0.0, 0.031), (0.0, 0.0155)], ["real_complex", "real_complex"], "second",
               dict(PARAMETERS, y_freq=100e6, y_domain="Carbon13"), data_format=2)
    [d] = miox.load(path)
    assert d.shape == (1, 1, 1, 32, 32)
    assert np.allclose(d.data[0, 0, 0, :, 5], sections[0][5] + 1j * sections[1][5])
    assert d.dim_info[5]["header"]["IndirectTime"]["Value"][1] == pytest.approx(0.0005)


def test_other_files_are_not_jeol(tmp_path):
    path = tmp_path / "plain.jdf"
    path.write_bytes(b"not a JEOL file" * 100)
    assert miox.sniff(path).format != miox.Format.JEOL_JDF
