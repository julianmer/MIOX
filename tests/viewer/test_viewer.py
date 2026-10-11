####################################################################################################
#                                          test_viewer.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 11/10/26                                                                                #
#                                                                                                  #
# Purpose: Spectrum and image views and ``miox view`` on synthetic data. No test data needed.      #
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
import miox as gu
from miox.cli import main
from miox.meta import Meta
from miox.viewer import ImageView, SpectrumView

FREQUENCY = 123.2  # MHz
DWELL = 2.5e-4  # s


def _spectroscopy(ppm=2.01, coils=1, frequency=FREQUENCY, dwell=DWELL, cls=gu.MRSData, **keys):
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
    provenance = gu.Provenance((), gu.Format.UNKNOWN, "test", {})
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
    d = _spectroscopy(coils=2, cls=gu.BasisSet)
    d.dim_info = {5: {"header": {"Metabolite": ["NAA", "Cr"]}}}
    view = SpectrumView(d)
    assert view.ax.get_title().endswith("NAA")
    view.sliders[4].set_val(1)
    assert view.ax.get_title().endswith("Cr")


def _image(data, affine, dims=()):
    provenance = gu.Provenance((), gu.Format.UNKNOWN, "test", {})
    return gu.MRImage(data, affine, Meta(), provenance, name="image", dims=dims)


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
