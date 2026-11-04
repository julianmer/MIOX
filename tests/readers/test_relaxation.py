####################################################################################################
#                                        test_relaxation.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 04/11/26                                                                                #
#                                                                                                  #
# Purpose: The RelaxationData object on synthetic decays: one curve and several, its summary and   #
#          the CLI's description.                                                                  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

from pathlib import Path

import numpy as np

# own
import miox
from miox.cli import CLI
from miox.meta.meta import Meta


def _decay(curves: int = 1) -> miox.RelaxationData:
    times = np.arange(1, 101) * 2e-3  # a CPMG train, 2 ms echoes
    signal = np.exp(-times / 0.1)
    if curves > 1:
        signal = np.stack([signal * (k + 1) for k in range(curves)])
    provenance = miox.Provenance((Path("sample.dps"),), miox.Format.UNKNOWN, "test", {})
    return miox.RelaxationData(times, signal, "CPMG", Meta(), provenance, "sample")


def test_one_decay():
    d = _decay()
    assert d.n_points == 100 and d.n_curves == 1 and d.curves.shape == (1, 100)
    summary = d.to_dict()
    assert summary["sequence"] == "CPMG" and summary["time_range"] == [0.002, 0.2]
    assert not summary["complex"]
    assert repr(d) == "<RelaxationData sample unknown: CPMG, 100 points to 0.2 s>"
    assert CLI()._describe(d).splitlines()[1] == "  sample: CPMG relaxation, 100 points"


def test_several_decays():
    d = _decay(3)
    assert d.n_curves == 3 and d.curves.shape == (3, 100)
    assert np.allclose(d.curves[2], 3 * d.curves[0])
    assert "x 3 curves" in repr(d) and "x 3 curves" in CLI()._describe(d)
