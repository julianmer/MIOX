####################################################################################################
#                                         test_twix_csi.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 13/10/26                                                                                #
#                                                                                                  #
# Purpose: MIOX's reconstruction of twix CSI against the scanner's own (the RDA of the same scan,  #
#          spec2nii test data, VE): same geometry, and per voxel with signal the coil-combined FID #
#          matches the scanner's (|correlation| > 0.99); grids not checked yet (VB) stay empty.    #
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
from tests.conftest import PUBLIC_DATA

SIEMENS = PUBLIC_DATA / "spec2nii_test_data" / "Siemens"
PAIRS = {
    "transverse": ("meas_MID00223_FID62728_csi_se_t_c23_5_s20_3_R10.dat",
                   "csi_se_t_c235_s203_r10.rda"),
    "sagittal": ("meas_MID00225_FID62730_csi_se_s_t23_5_c20_3_R10.dat",
                 "csi_se_s_t235_c203_r10.rda"),
    "coronal": ("meas_MID00227_FID62732_csi_se_c_s23_5_t20_3_R10.dat",
                "csi_se_c_s235_t203_r10.rda"),
    "3D transverse": ("meas_MID00229_FID62734_csi_se_3D_t_c23_5_s20_3_R10.dat",
                      "csi_se_3d_t_c235_s203_r10.rda"),
}


def spectroscopy(path):
    if not path.exists():
        pytest.skip(f"{path.name} not fetched")
    return next(d for d in miox.load(path) if isinstance(d, miox.MRSData))


def combined(data: np.ndarray) -> np.ndarray:
    """Coils (last axis) combined with the conjugate of their first points as weights."""
    first = data[..., :8, :].mean(axis=-2)
    weights = np.conj(first) / (np.linalg.norm(first, axis=-1, keepdims=True) + 1e-30)
    return (data * weights[..., None, :]).sum(axis=-1)


@pytest.mark.parametrize("case", PAIRS)
def test_csi_matches_the_scanner(case):
    twix, rda = PAIRS[case]
    ours = spectroscopy(SIEMENS / "VEData" / "Twix" / twix)
    scanner = spectroscopy(SIEMENS / "VEData" / "rda" / rda)
    assert ours.dim_tags[0] == "DIM_COIL"
    assert np.allclose(ours.affine, scanner.affine, atol=1e-3)  # mm, as the headers round
    # combined as FIDs (the conjugate of NIfTI-MRS as stored), compared as stored
    x = combined(np.asarray(ours.data).conj()).conj()
    y = np.asarray(scanner.data)
    signal = np.abs(y[..., :32]).sum(axis=-1)
    voxels = [i for i in np.ndindex(signal.shape) if signal[i] > 0.3 * signal.max()]
    match = [abs(np.vdot(x[i], y[i])) / (np.linalg.norm(x[i]) * np.linalg.norm(y[i]))
             for i in voxels]
    assert np.median(match) > 0.995 and min(match) > 0.99


def test_unchecked_grids_stay_empty():
    vb = spectroscopy(SIEMENS / "VBData" / "Twix"
                      / "meas_MID133_csi_se_T_C23_5_S20_3_10_FID108723.dat")
    assert not np.any(vb.data)
    assert any("not reconstructed" in note for note in vb.provenance.notes)
