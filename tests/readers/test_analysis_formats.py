####################################################################################################
#                                     test_analysis_formats.py                                     #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Readers of analysis-package formats on synthetic files: TARQUIN .dpt, VASF, Vespa VIFF  #
#          and UCSF/SIVIC DDF. No test data needed.                                                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import base64
import zlib

import numpy as np
import pytest

# own
import miox


def _fid(n: int = 64) -> np.ndarray:
    t = np.arange(n) * 2.5e-4
    return np.exp(2j * np.pi * 100 * t - t / 0.05)


def test_tarquin_dpt(tmp_path):
    f = tmp_path / "x.dpt"
    lines = [
        "Dangerplot_version\t1.0",
        "Number_of_points\t64",
        "Sampling_frequency\t4.0e+03",
        "Transmitter_frequency\t1.232e+08",
        "Phi0\t0.0",
        "Phi1\t0.0",
        "PPM_reference\t4.65",
        "Echo_time\t3.0e-02",
        "Real_FID\tImag_FID\t",
    ] + [f"{z.real:.8e}\t{z.imag:.8e}" for z in _fid()]
    f.write_text("\n".join(lines) + "\n")
    [d] = miox.load(f)
    assert d.spectrometer_frequency == pytest.approx(123.2) and d.dwelltime == pytest.approx(2.5e-4)
    assert d.meta["EchoTime"] == pytest.approx(0.03) and d.meta["ReferencePPM"] == 4.65
    assert np.allclose(d.data[0, 0, 0], _fid())  # written in the NIfTI-MRS convention


def test_vasf(tmp_path):
    rsp, rsd = tmp_path / "press.rsp", tmp_path / "press.rsd"
    rsp.write_text(
        "[FILE INFORMATION]\ndata_format= xdr\ndata_type= complex float\n\n"
        "[MEASUREMENT INFORMATION]\naverages=   16\necho_time_1=  30.0\nflip_angle=90\t;(degrees)\n"
        "frequency=  123200000.\ninversion_time_1=0.00\t;(ms)\nnucleus=1H  \n"
        "repetition_time_1=  2000.0\nsweep_width=  4000.0\nmeasure_size_spectral=  64\n"
    )
    pairs = np.column_stack([_fid().real, _fid().imag]).astype(">f4")
    rsd.write_bytes(pairs.tobytes())
    [d] = miox.load(rsp)
    assert d.meta["EchoTime"] == pytest.approx(0.03) and d.meta["NumberOfAverages"] == 16
    assert "InversionTime" not in d.meta
    assert np.allclose(d.data[0, 0, 0], _fid(), atol=1e-6)


def test_vespa_viff(tmp_path):
    pairs = np.column_stack([_fid().real, _fid().imag]).astype(">f4").tobytes()
    encoded = base64.b64encode(zlib.compress(pairs)).decode()
    f = tmp_path / "press.xml"
    f.write_text(
        '<vespa_export version="1.0.0"><dataset id="a"><blocks><block_raw id="b">'
        "<data_type>complex64</data_type><sweep_width>4000.0</sweep_width>"
        "<frequency>123.2</frequency><nucleus>1H</nucleus><seqte>30.0</seqte>"
        '<data data_type="complex64" encoding="xdr zlib base64" shape="1,1,1,64">'
        f"{encoded}</data></block_raw></blocks></dataset></vespa_export>"
    )
    [d] = miox.load(f)
    assert d.meta["EchoTime"] == pytest.approx(0.03)
    assert np.allclose(d.data[0, 0, 0], _fid(), atol=1e-6)
    assert d.meta.vendor["dataset"]["blocks"]["block_raw"]["data"].startswith("<array")


def test_sivic_ddf_spectra_and_geometry(tmp_path):
    n, (nx, ny, nz) = 64, (3, 2, 1)
    fids = np.stack([_fid(n) * (i + 1) for i in range(nx * ny * nz)]).reshape(nz, ny, nx, n)
    spectra = np.fft.fftshift(np.fft.fft(fids, axis=-1), axes=-1)
    data = np.stack([spectra.real, spectra.imag], axis=-1).astype(">f4")
    (tmp_path / "csi.cmplx").write_bytes(data.tobytes())
    (tmp_path / "csi.ddf").write_text(
        "DATA DESCRIPTOR FILE\nversion: 6.1\nnumber of dimensions: 4\n"
        f"dimension 1: type: frequency npoints: {n} \n"
        f"dimension 2: type: space npoints: {nx} pixel spacing(mm): 10.0\n"
        f"dimension 3: type: space npoints: {ny} pixel spacing(mm): 10.0\n"
        f"dimension 4: type: space npoints: {nz} pixel spacing(mm): 20.0\n"
        "center(lps, mm): 10.0 5.0 0.0\ntoplc(lps, mm): 0.0 0.0 0.0\n"
        "dcos0: 1.0 0.0 0.0\ndcos1: 0.0 1.0 0.0\ndcos2: 0.0 0.0 1.0\n"
        "===================================================\nMR Parameters\n"
        "echo time(ms): 30.0\ntransmitter frequency(MHz): 123.2\nisotope: 1H\n"
        "===================================================\nSpectroscopy Parameters\n"
        "dwelltime(ms): 0.25\n"
    )
    [d] = miox.load(tmp_path / "csi.ddf")
    assert d.shape == (nx, ny, nz, n)
    assert np.allclose(d.data[2, 1, 0], fids[0, 1, 2])  # x = 2, y = 1
    assert np.allclose(d.affine @ [1, 0.5, 0, 1], [-10.0, -5.0, 0.0, 1.0])  # LPS centre in RAS
