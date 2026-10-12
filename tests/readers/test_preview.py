####################################################################################################
#                                         test_preview.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 12/10/26                                                                                #
#                                                                                                  #
# Purpose: The quick-look picture on synthetic spectra with known answers: coils with different    #
#          phases and noise combine coherently, shifted and dephased averages are aligned,         #
#          residual water is removed without touching the metabolites, edit-on is told from        #
#          edit-off, and the command line writes a PDF or declines files that are no spectra.      #
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
from miox.cli import main
from miox.data import MRSData, Provenance
from miox.meta import Meta
from miox.preview import Preview, Reducer, Spectrum

FREQUENCY = 123.2  # MHz, 3 T
DWELL = 2.5e-4  # s: 4000 Hz
POINTS = 2048
T = np.arange(POINTS) * DWELL


def peak(ppm: float, amplitude: float = 1.0, width: float = 4.0) -> np.ndarray:
    """A Lorentzian line at ``ppm`` in the NIfTI-MRS convention (4.65 ppm at 0 Hz)."""
    hz = (4.65 - ppm) * FREQUENCY
    return amplitude * np.exp(2j * np.pi * hz * T - np.pi * width * T)


def dataset(data: np.ndarray, tags=(None, None, None), name: str = "test") -> MRSData:
    meta = Meta({})
    meta.set("SpectrometerFrequency", [FREQUENCY], "test")
    meta.set("ResonantNucleus", ["1H"], "test")
    provenance = Provenance((), miox.Format.NIFTI_MRS, "test", {})
    return MRSData(data.reshape(1, 1, 1, *data.shape), DWELL, np.eye(4), meta, provenance,
                   tuple(tags), name=name)


def tallest(d: MRSData, fid: np.ndarray, low: float, high: float) -> float:
    spectrum = Spectrum(d, fid)
    region = (spectrum.x >= low) & (spectrum.x <= high)
    return float(spectrum.x[region][np.argmax(np.abs(spectrum.values[region]))])


def test_coils_combine_coherently():
    rng = np.random.default_rng(0)
    signal = peak(2.01) + peak(3.03, 0.8)
    phases = np.exp(1j * rng.uniform(-np.pi, np.pi, 8))
    gains = rng.uniform(0.2, 1.0, 8)
    coils = signal[:, None] * gains * phases + 0.01 * (rng.normal(size=(POINTS, 8))
                                                       + 1j * rng.normal(size=(POINTS, 8)))
    d = dataset(coils, ("DIM_COIL", None, None))
    fid = Reducer(d).reduce().fids["spectrum"]
    # coherent: the combined FID keeps the shape of the signal (correlation of magnitudes ~ 1)
    assert abs(np.vdot(fid, signal)) / (np.linalg.norm(fid) * np.linalg.norm(signal)) > 0.99


def test_averages_are_aligned():
    rng = np.random.default_rng(1)
    shifts = rng.uniform(-8, 8, 16)
    phases = rng.uniform(-np.pi, np.pi, 16)
    averages = np.stack([peak(2.01) * np.exp(2j * np.pi * s * T + 1j * p)
                         for s, p in zip(shifts, phases, strict=True)], axis=1)
    d = dataset(averages, ("DIM_DYN", None, None))
    fid = Reducer(d).reduce().fids["spectrum"]
    # aligned and averaged: one line as tall as a single average, at a common frequency
    single = np.abs(Spectrum(d, averages[:, 0]).values).max()
    assert np.abs(Spectrum(d, fid).values).max() > 0.95 * single


def test_residual_water_is_removed_and_metabolites_kept():
    fid = peak(4.70, 30.0, 6.0) + peak(2.01) + peak(3.03, 0.8)
    d = dataset(fid)
    spectrum = Spectrum(d, fid)
    assert "residual water removed for display (HLSVD)" in spectrum.notes
    assert spectrum.range == (0.2, 4.3)
    assert abs(tallest(d, fid, 1.8, 2.2) - 2.01) < 0.01
    clean = Spectrum(d, peak(2.01) + peak(3.03, 0.8))
    assert abs(spectrum.height(1.9, 2.1) - clean.height(1.9, 2.1)) < 0.05 * clean.height(1.9, 2.1)


def test_water_reference_is_kept():
    fid = peak(4.65, 1000.0) + peak(2.01)
    spectrum = Spectrum(dataset(fid, name="x_ref"), fid, role="water")
    assert spectrum.water and spectrum.range[1] > 4.65


def test_edit_on_is_the_lower_naa():
    off = peak(2.01) + peak(3.0, 0.3)
    on = peak(2.01, 0.6) + peak(3.0, 0.45)
    d = dataset(np.stack([on, off], axis=1), ("DIM_EDIT", None, None))
    preview = Preview([d])
    labels = [t.get_text() for t in preview.figure.axes[0].get_legend().get_texts()]
    assert labels == ["edit off (edit 2)", "edit on (edit 1)", "difference (on - off)"]


def test_command_line_writes_pdf_and_declines_other_files(tmp_path, capsysbinary):
    import nibabel as nib

    fid = peak(2.01) + peak(3.03, 0.8)
    path = tmp_path / "svs.nii.gz"
    image = nib.Nifti2Image(fid.astype(np.complex64).reshape(1, 1, 1, -1), np.eye(4))
    image.header.set_intent(0, name="mrs_v0_9")
    image.header["pixdim"][4] = DWELL
    extension = b'{"SpectrometerFrequency": [123.2], "ResonantNucleus": ["1H"]}'
    image.header.extensions.append(nib.nifti1.Nifti1Extension(44, extension))
    image.to_filename(path)
    assert main(["preview", str(path), "--to", "-"]) == 0
    assert capsysbinary.readouterr().out.startswith(b"%PDF")
    other = tmp_path / "notes.dat"
    other.write_bytes(b"\x00\x01" * 100)
    assert main(["preview", str(other), "--to", "-"]) == 2


@pytest.mark.parametrize("tags", [("DIM_COIL", "DIM_DYN", None), (None, None, None)])
def test_preview_draws(tags):
    data = peak(2.01).reshape(-1, 1, 1) * np.ones((1, 2, 3)) if tags[0] else peak(2.01)
    figure = Preview([dataset(data, tags)], "file.dat").figure
    assert figure.texts[0].get_text() == "file.dat"
