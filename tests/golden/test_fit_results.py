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
