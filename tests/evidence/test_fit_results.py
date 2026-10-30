####################################################################################################
#                                       test_fit_results.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Fit results: one LCModel fit read from its TABLE, COORD and CSV gives the same rows;    #
#          curves add up as each tool composes them; combined rows equal the sum of their parts.   #
#          spant's saved fits: their data, with spant's fitted shift and phase, give the spectrum  #
#          spant fitted; its simulated spectrum is its basis weighted by the amplitudes it was     #
#          made with; the basis signals lie at their shifts.                                       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import numpy as np
import pytest

# own
from miox import FitResult, load
from tests.conftest import PUBLIC_DATA

SUSPECT = "suspect/tests/test_data/"
SPANT = "spant/tests/testthat/"
CURVES = ("ppm", "data", "fit", "baseline")


def result(relpath: str) -> FitResult:
    path = PUBLIC_DATA / relpath
    if not path.exists():
        pytest.skip("file not fetched")
    [r] = load(path)
    return r


@pytest.mark.data
def test_lcmodel_table_coord_and_csv_agree():
    table, coord, csv = (result(f"{SUSPECT}lcmodel/svs_97.{x}") for x in ("TABLE", "COORD", "CSV"))
    assert len(table.rows) == len(csv.rows) == 37 and table.rows == coord.rows
    for a, b in zip(table.rows, csv.rows, strict=True):  # the CSV truncates long names
        assert a["metabolite"].startswith(b.pop("metabolite")) and a.items() >= b.items()
    assert table.diagnostics == coord.diagnostics and dict(table.meta) == dict(coord.meta)


@pytest.mark.data
def test_lcmodel_coord_curves_add_up():
    curves = result(SUSPECT + "lcmodel/svs_97.COORD").curves
    signals = [c for c in curves if c not in CURVES]
    total = curves["baseline"] + sum(curves[c] - curves["baseline"] for c in signals)
    assert np.abs(curves["fit"] - total).max() < 1e-3 * np.ptp(curves["fit"])


@pytest.mark.data
def test_tarquin_fit_is_the_sum_of_its_signals():
    [fit] = load(PUBLIC_DATA / SUSPECT / "tarquin/tarquin_megapress_fit.txt")
    signals = [c for c in fit.curves if c not in CURVES]
    total = sum(fit.curves[c] for c in signals)
    assert np.abs(fit.curves["fit"] - total).max() < 1e-4 * np.ptp(fit.curves["fit"])


@pytest.mark.data
@pytest.mark.parametrize(
    "relpath,combined,parts",
    [
        (SUSPECT + "lcmodel/svs_97.TABLE", "NAA+NAAG", ("NAA", "NAAG")),
        (SUSPECT + "lcmodel/svs_97.TABLE", "Glu+Gln", ("Glu", "Gln")),
        (SUSPECT + "tarquin/tarquin_results.txt", "TNAA", ("NAA", "NAAG")),
        (SUSPECT + "tarquin/tarquin_results.txt", "TCr", ("Cr", "PCr")),
    ],
)
def test_combined_rows_are_sums(relpath, combined, parts):
    r = result(relpath)
    total = sum(r.row(p)["concentration"] for p in parts)
    assert r.row(combined)["concentration"] == pytest.approx(total, rel=0.01)


@pytest.mark.data
@pytest.mark.parametrize("name", ["default", "coarse", "fine", "no_optim"])
def test_spant_fit_data_with_its_shift_and_phase_is_the_spectrum_it_fitted(name):
    """abfit's table holds the data as fitted: the FID (zero-filled twice), shifted by -shift ppm
    and phased by phase degrees, on PPMScale. The fit's own basis names its amplitude columns."""
    path = PUBLIC_DATA / f"{SPANT}abfit_res_{name}.rds"
    if not path.exists():
        pytest.skip("spant not fetched")
    fit, data, basis = load(path)
    assert fit.metabolites[: len(basis.metabolites)] == basis.metabolites
    frequency, n = data.spectrometer_frequency, data.n_points
    t = np.arange(n) * data.dwelltime
    shift, phase = fit.diagnostics["shift"], fit.diagnostics["phase"]
    fid = data.data[0, 0, 0] * np.exp(-2j * np.pi * shift * frequency * t + 1j * np.radians(phase))
    spectrum = np.fft.fftshift(np.fft.fft(fid, 2 * n))
    offsets = np.fft.fftshift(np.fft.fftfreq(2 * n, data.dwelltime))
    ppm = data.meta["ReferencePPM"] - offsets / frequency
    stated = np.interp(fit.curves["ppm"], ppm[::-1], spectrum.real[::-1])
    assert np.corrcoef(stated, fit.curves["data"])[0, 1] > 0.9999
    mirrored = np.fft.fftshift(np.fft.fft(np.conj(fid), 2 * n))
    mirrored = np.interp(fit.curves["ppm"], ppm[::-1], mirrored.real[::-1])
    assert abs(np.corrcoef(mirrored, fit.curves["data"])[0, 1]) < 0.5


@pytest.mark.data
def test_spant_simulation_is_its_basis_weighted_by_its_amplitudes():
    path = PUBLIC_DATA / f"{SPANT}abfit_sim_res.rds"
    if not path.exists():
        pytest.skip("spant not fetched")
    data, basis = load(path)
    amplitudes = basis.meta.vendor["list"]["amps"]  # by signal name
    model = sum(basis.fid(signal) * amplitudes[signal] for signal in basis.metabolites)
    assert np.abs(model - data.data[0, 0, 0]).max() < 1e-9 * np.abs(model).max()
    ppm = basis.meta["ReferencePPM"] - np.fft.fftshift(
        np.fft.fftfreq(basis.n_points, basis.dwelltime)) / basis.spectrometer_frequency
    for signal, shift in (("NAA", 2.008), ("Cr", 3.027), ("PCr", 3.029)):
        spectrum = np.fft.fftshift(np.fft.fft(basis.fid(signal)))
        assert ppm[np.argmax(np.abs(spectrum))] == pytest.approx(shift, abs=0.01)
