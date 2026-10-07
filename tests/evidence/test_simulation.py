####################################################################################################
#                                        test_simulation.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/10/26                                                                                #
#                                                                                                  #
# Purpose: Simulation files against independent evidence: a JEMRIS phantom grid against the spin   #
#          list KomaMRI made of it, JEMRIS signals against the EPI they were simulated with (as    #
#          KomaMRI's tests define it), KomaMRI phantoms against their own attributes, an MRiLab    #
#          phantom against MRiLab's field grids of the same object.                                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import numpy as np
import pytest
import scipy.io

# own
from miox import load
from tests.conftest import PUBLIC_DATA

KOMA = PUBLIC_DATA / "komamri"
PHANTOMS = KOMA / "examples/2.phantoms"
CORE = KOMA / "KomaMRICore/test/test_files"
MRILAB = KOMA / "KomaMRIFiles/test/test_files/phantom"


def _need(*paths):
    for path in paths:
        if not path.is_file():
            pytest.skip(f"{path.relative_to(PUBLIC_DATA)} not fetched")


def test_jemris_grid_equals_komamri_spin_list():
    """KomaMRI's tests hold the spins it made of the JEMRIS chemical-shift sphere (7825 spins):
    positions, M0, T1, T2 and T2* (from the stored rates) and the off-resonance equal those read
    from the grid."""
    _need(PHANTOMS / "sphere_chemical_shift.h5", CORE / "phantom_sphere.h5")
    [grid] = load(PHANTOMS / "sphere_chemical_shift.h5")
    [spins] = load(CORE / "phantom_sphere.h5")
    assert grid.n_spins == spins.n_spins == 7825
    a = np.lexsort(grid.positions.T[::-1])
    b = np.lexsort(spins.positions.T[::-1])
    assert np.allclose(grid.positions[a], spins.positions[b], atol=1e-12)
    for ours, theirs in (("rho", "rho"), ("T1", "T1"), ("T2", "T2"), ("T2s", "T2s"),
                         ("DB", "dw")):
        assert np.array_equal(grid.properties[ours][a], spins.properties[theirs][b]), ours


@pytest.mark.parametrize(
    "name", ("jemris_signals_epi_sphere_cs", "jemris_signals_epi_brain_motion"))
def test_jemris_signal_readouts_follow_the_epi(name):
    """Both signals were simulated with KomaMRI's EPI of 100 x 100 samples, 10 us ADC dwell and
    TE 100 ms: the gaps in the sample times give 100 readouts of 100 samples at 10 us, and the
    strongest sample (the centre of k-space) lies within one readout of TE."""
    _need(CORE / f"{name}.h5")
    [raw] = load(CORE / f"{name}.h5")
    assert raw.data.shape == (100, 1, 100)
    assert raw.meta["DwellTime"] == 10e-6
    times = raw.waveforms[1]["data"]
    peak = np.unravel_index(np.abs(raw.data[:, 0]).argmax(), times.shape)
    assert times[peak] == pytest.approx(0.1, abs=times[0, -1] - times[0, 0])


@pytest.mark.parametrize("name", ("brain_motion.phantom", "contracting_ring.phantom"))
def test_koma_phantom_matches_its_attributes(name):
    """A .phantom file states its spin count (Ns) and dimensions (Dims): the positions read
    have that many spins and vary along that many axes."""
    _need(PHANTOMS / name)
    [phantom] = load(PHANTOMS / name)
    attributes = phantom.meta.vendor["attributes"]
    assert phantom.n_spins == attributes["Ns"]
    assert int((np.ptp(phantom.positions, axis=0) > 0).sum()) == attributes["Dims"]
    assert all(len(v) == phantom.n_spins for v in phantom.properties.values())


def test_mrilab_positions_equal_its_field_grids():
    """MRiLab stores the position of every voxel in its field object (VMag.Gxgrid, Gygrid,
    Gzgrid, saved beside the brain phantom): the spin positions read from the phantom's grid
    equal them."""
    _need(MRILAB / "brain_mrilab.mat", MRILAB / "FRange.mat")
    [phantom] = load(MRILAB / "brain_mrilab.mat")
    field = scipy.io.loadmat(MRILAB / "FRange.mat", squeeze_me=True,
                             struct_as_record=False)["VMag"]
    at = tuple(phantom.grid["indices"].T)
    grids = np.stack([field.Gxgrid[at], field.Gygrid[at], field.Gzgrid[at]], axis=1)
    assert np.allclose(phantom.positions, grids, atol=1e-12)
    assert phantom.n_spins == int((phantom.properties["rho"] != 0).sum())
