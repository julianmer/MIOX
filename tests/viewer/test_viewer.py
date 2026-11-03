####################################################################################################
#                                          test_viewer.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 11/10/26                                                                                #
#                                                                                                  #
# Purpose: Every view and ``miox view`` on synthetic data. No test data needed.                    #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest
from nifti_mrs.create_nmrs import gen_nifti_mrs

# own
import miox
from miox.cli import main
from miox.meta import Meta
from miox.viewer import (
    FitView,
    ImageView,
    NoiseView,
    PhysioView,
    RawView,
    SequenceView,
    SpectrumView,
)

FREQUENCY = 123.2  # MHz
DWELL = 2.5e-4  # s


def _spectroscopy(ppm=2.01, coils=1, frequency=FREQUENCY, dwell=DWELL, cls=miox.MRSData, **keys):
    """One decaying line that lies at ``ppm`` on the axis 4.65 - f / frequency, per coil c scaled
    by c + 1."""
    t = np.arange(1024) * (dwell or DWELL)
    f = (4.65 - ppm) * FREQUENCY  # Hz in the data as stored
    fid = np.exp(2j * np.pi * f * t - t / 0.05)
    data = fid[None, None, None, :, None] * np.arange(1, coils + 1)
    meta = Meta()
    if frequency is not None:
        meta.set("SpectrometerFrequency", [frequency], "test")
    meta.set("ResonantNucleus", ["1H"], "test")
    for key, value in keys.items():
        meta.set(key, value, "test")
    provenance = miox.Provenance((), miox.Format.UNKNOWN, "test", {})
    return cls(data, dwell, np.eye(4), meta, provenance, dim_tags=("DIM_COIL", None, None),
               name="svs")


def _peak(view: SpectrumView) -> float:
    x, y = view.line.get_data()
    return float(x[np.argmax(y)])


def test_spectrum_peak_on_the_ppm_axis():
    view = SpectrumView(_spectroscopy(2.01))
    assert view.unit == "ppm" and view.part == "magnitude"
    assert _peak(view) == pytest.approx(2.01, abs=0.04)  # one point is 0.032 ppm
    assert view.ax.xaxis_inverted()
    view.set_part("real")
    assert _peak(view) == pytest.approx(2.01, abs=0.04)  # absorption: the real part peaks
    view.set_unit("Hz")
    assert _peak(view) == pytest.approx((2.01 - 4.65) * FREQUENCY, abs=4)


def test_spectrum_uses_the_stated_reference():
    view = SpectrumView(_spectroscopy(2.01, ReferencePPM=4.7))
    assert view.reference == 4.7 and _peak(view) == pytest.approx(2.06, abs=0.04)


def test_spectrum_slider_per_further_dimension():
    view = SpectrumView(_spectroscopy(coils=3))
    assert list(view.sliders) == [4]
    first = view.line.get_data()[1].max()
    view.sliders[4].set_val(2)
    assert view.index[4] == 2 and view.line.get_data()[1].max() == pytest.approx(3 * first)


def test_spectrum_without_frequency_or_dwell_time():
    view = SpectrumView(_spectroscopy(frequency=None))
    assert view.unit == "Hz" and not hasattr(view, "units")
    view = SpectrumView(_spectroscopy(dwell=None))
    assert np.array_equal(view.line.get_data()[0], np.arange(1024))
    assert view.ax.get_xlabel() == "point"


def test_basis_set_names_the_metabolite():
    d = _spectroscopy(coils=2, cls=miox.BasisSet)
    d.dim_info = {5: {"header": {"Metabolite": ["NAA", "Cr"]}}}
    view = SpectrumView(d)
    assert view.ax.get_title().endswith("NAA")
    view.sliders[4].set_val(1)
    assert view.ax.get_title().endswith("Cr")


def _image(data, affine, dims=()):
    provenance = miox.Provenance((), miox.Format.UNKNOWN, "test", {})
    return miox.MRImage(data, affine, Meta(), provenance, name="image", dims=dims)


def test_image_turned_to_ras():
    data = np.zeros((4, 5, 6))
    data[0, 1, 2] = 1  # voxel axes run right to left and anterior to posterior
    view = ImageView(_image(data, np.diag([-2.0, -3.0, 4.0, 1.0])))
    assert np.argwhere(view.data == 1).tolist() == [[3, 3, 2]]
    assert np.allclose(view.sizes, (2, 3, 4))
    view.sliders[2].set_val(2)
    axial = view.images[2].get_array()
    assert axial.shape == (5, 4) and np.argwhere(axial == 1).tolist() == [[3, 3]]  # row y, col x
    assert [p[0] for p in view.planes] == ["sagittal", "coronal", "axial"]


def test_single_slice_complex_image_with_further_dimension():
    data = np.ones((8, 8, 1, 3), complex) * np.array([1j, 2, -3])
    view = ImageView(_image(data, np.eye(4), dims=("echo",)))
    assert [p[0] for p in view.planes] == ["axial"]
    assert list(view.sliders) == [3]
    view.sliders[3].set_val(2)
    assert np.all(view.images[2].get_array() == 3)  # magnitude


def _provenance():
    return miox.Provenance((), miox.Format.UNKNOWN, "test", {})


def test_raw_readouts_per_coil():
    data = np.ones((3, 2, 8), complex)
    data[:, 1] *= 10
    data[0, 0, 0] = 0  # floored, not -inf
    view = RawView(miox.RawData(data, np.zeros(3, miox.RawData.HEADER), Meta(), _provenance()))
    assert view.image.get_array().shape == (3, 8) and np.isfinite(view.log_magnitude()).all()
    assert view.log_magnitude()[1, 1] == pytest.approx(0)
    view.slider.set_val(1)
    assert view.coil == 1 and view.log_magnitude()[1, 1] == pytest.approx(1)


def test_physio_traces_and_events():
    data = np.stack([np.arange(100.0), -np.arange(100.0)], axis=1)
    d = miox.PhysioData(data, ("cardiac", "respiratory"), 50.0, 3600.0, "scanner clock",
                        {"trigger": np.array([10, 60, 100])}, Meta(), _provenance())
    view = PhysioView(d)
    assert [ax.get_ylabel() for ax in view.axes] == ["cardiac", "respiratory"]
    x, y = view.axes[1].lines[0].get_data()
    assert x[-1] == pytest.approx(99 / 50) and y[-1] == -99  # seconds from the first sample
    [events] = view.axes[0].collections
    assert len(events.get_segments()) == 2  # the index past the last sample is left out


def test_fit_curves_and_concentrations():
    ppm = np.linspace(4, 0.2, 50)
    curves = {"ppm": ppm, "data": np.sin(ppm), "fit": np.sin(ppm) * 0.9, "baseline": 0 * ppm}
    rows = [{"metabolite": "NAA", "concentration": 10.0, "sd_percent": 5.0},
            {"metabolite": "Cr", "concentration": 8.0, "sd_percent": None}]
    view = FitView(miox.FitResult(rows, {}, curves, Meta(), _provenance(), "fit"))
    spectrum, residual, bars = view.figure.axes
    assert spectrum.get_xlim() == (4, 0.2)
    assert np.allclose(residual.lines[0].get_ydata(), np.sin(ppm) * 0.1)
    assert [label.get_text() for label in bars.get_yticklabels()] == ["NAA", "Cr"]
    assert [bar.get_width() for bar in bars.patches] == [10, 8]
    assert len(FitView(miox.FitResult(rows, {}, {}, Meta(), _provenance())).figure.axes) == 1
    empty = FitView(miox.FitResult([], {}, {}, Meta(), _provenance()))
    assert empty.figure.texts[0].get_text() == "no curves or concentrations"


def _sequence(blocks, layout):
    names = ("id", layout, "rf", "gx", "gy", "gz", "adc", "ext")
    blocks = np.array(blocks, dtype=[(name, "<i8") for name in names])
    events = {
        "rf": {1: {"amp": 500.0, "mag_id": 1, "phase_id": 2, "time_id": 0, "delay": 100}},
        "trap": {2: {"amp": 1000.0, "rise": 10, "flat": 20, "fall": 10, "delay": 5}},
        "gradients": {3: {"amp": 2000.0, "shape_id": 3, "time_id": 4, "delay": 0},
                      4: {"amp": 100.0, "shape_id": 5, "time_id": -1, "delay": 0}},
        "adc": {1: {"num": 4, "dwell": 5000, "delay": 10}},
        "delays": {1: {"delay": 500}},
    }
    shapes = {1: np.array([0.5, 1.0, 0.5]), 2: np.zeros(3), 3: np.array([0.0, 1.0, 0.0]),
              4: np.array([0.0, 2.0, 4.0]), 5: np.array([1.0, 1.0, 1.0])}
    definitions = {"BlockDurationRaster": 1e-5, "GradientRasterTime": 1e-5,
                   "RadiofrequencyRasterTime": 1e-6}
    return miox.PulseSequence("1.4.2", definitions, blocks, events, {}, shapes, {}, Meta(),
                              _provenance(), "seq")


def test_sequence_waveforms():
    view = SequenceView(_sequence([(1, 30, 1, 2, 3, 4, 1, 0), (2, 10, 0, 0, 0, 0, 0, 0)],
                                  "duration"))
    t, magnitude, _, end = view.d.rf(1)
    assert np.allclose(t, 100e-6 + np.array([0.5, 1.5, 2.5]) * 1e-6)  # raster centres
    assert np.allclose(magnitude, [250, 500, 250]) and end == pytest.approx(103e-6)
    t, amplitude, end = view.d.gradient(2)
    assert np.allclose(t, [5e-6, 15e-6, 35e-6, 45e-6]) and np.allclose(amplitude, [0, 1e3, 1e3, 0])
    t, amplitude, end = view.d.gradient(3)  # time shape in raster units
    assert np.allclose(t, [0, 2e-5, 4e-5]) and np.allclose(amplitude, [0, 2e3, 0])
    t, _, end = view.d.gradient(4)  # oversampled by two
    assert np.allclose(t, [0.5e-5, 1e-5, 1.5e-5]) and end == pytest.approx(4e-5)
    assert view.d.adc(1) == pytest.approx((10e-6, 30e-6))
    assert np.allclose(view.starts, [0, 3e-4, 4e-4])
    x, y = view.lines["adc"].get_data()
    assert np.allclose(x[:4], [0.01, 0.01, 0.03, 0.03]) and np.isnan(x[4])  # ms


def test_sequence_block_ends_before_1_4():
    view = SequenceView(_sequence([(1, 1, 1, 0, 0, 0, 0, 0), (2, 0, 0, 2, 0, 0, 1, 0)], "delay"))
    assert np.allclose(view.starts, [0, 500e-6, 545e-6])  # the delay, then the trapezoid's end
    assert view.first_slider.valmax == 1


def test_noise_correlation_and_deviation():
    matrix = np.array([[4, 3], [3, 9]], complex)
    labels = [{"number": 0, "name": "H1"}, {"number": 1, "name": "H2"}]
    view = NoiseView(miox.NoiseCovariance(matrix, labels, 1.0, 1e-6, 256, Meta(), _provenance()))
    assert np.allclose(view.correlation, [[1, 0.5], [0.5, 1]])
    bars = view.figure.axes[1]
    assert [bar.get_height() for bar in bars.patches] == [2, 3]
    assert [label.get_text() for label in bars.get_xticklabels()] == ["H1", "H2"]


def test_import_leaves_matplotlib_until_view_is_used():
    code = ("import sys, miox; assert 'matplotlib' not in sys.modules; miox.view; "
            "assert 'matplotlib' in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_cli_view_saves_figures(tmp_path):
    t = np.arange(512) * DWELL
    fid = np.exp(-t / 0.05 + 0j)[None, None, None, :]
    image = gen_nifti_mrs(fid, DWELL, FREQUENCY, "1H")
    image.save(tmp_path / "svs.nii.gz")
    nib.Nifti1Image(np.ones((4, 4, 4), np.float32), np.eye(4)).to_filename(tmp_path / "t1.nii")
    assert main(["view", str(tmp_path / "svs.nii.gz"), "--save", str(tmp_path / "s.png")]) == 0
    assert (tmp_path / "s.png").stat().st_size > 0
    paths = [str(tmp_path / "svs.nii.gz"), str(tmp_path / "t1.nii")]
    assert main(["view", *paths, "--save", str(tmp_path / "both.png")]) == 0
    assert (tmp_path / "both-1.png").exists() and (tmp_path / "both-2.png").exists()


def test_cli_view_without_a_view(tmp_path, capsys):
    path = tmp_path / "flirt.mat"
    path.write_text("1 0 0 2\n0 1 0 3\n0 0 1 4\n0 0 0 1\n")
    assert main(["view", str(path), "--save", str(tmp_path / "t.png")]) == 1
    assert "no view for Transform yet" in capsys.readouterr().err
    assert not (tmp_path / "t.png").exists()
