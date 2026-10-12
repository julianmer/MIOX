####################################################################################################
#                                       test_dsi_studio.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 13/10/26                                                                                #
#                                                                                                  #
# Purpose: DSI Studio files against each other: a tensor fitted to NiiVue's demo .src (its         #
#          diffusion images and b-table) gives the FA its .fib holds, voxel by voxel (a wrong      #
#          voxel order drops the agreement to nothing, shuffled gradients to well below); the      #
#          CMU60 template's trans puts its grid on DSI Studio's MNI box.                           #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import numpy as np
import pytest

# own
from miox import load
from tests.conftest import PUBLIC_DATA

NIIVUE = PUBLIC_DATA / "niivue" / "packages" / "niivue" / "demos" / "images"
CMU60 = PUBLIC_DATA / "dsistudio_templates" / "CMU60.2mm.fib.gz"


def fractional_anisotropy(signal: np.ndarray, bvalues: np.ndarray, directions: np.ndarray):
    """Log-linear tensor fit per voxel (rows of signal), FA of the eigenvalues."""
    g = directions
    design = np.c_[np.ones(len(bvalues)), -bvalues[:, None] * np.stack(
        [g[:, 0] ** 2, g[:, 1] ** 2, g[:, 2] ** 2, 2 * g[:, 0] * g[:, 1], 2 * g[:, 0] * g[:, 2],
         2 * g[:, 1] * g[:, 2]], axis=1)]
    coef = np.linalg.lstsq(design, np.log(np.maximum(signal, 1)).T, rcond=None)[0].T
    tensor = np.zeros((len(coef), 3, 3))
    for i, j, k in ((0, 0, 1), (1, 1, 2), (2, 2, 3), (0, 1, 4), (0, 2, 5), (1, 2, 6)):
        tensor[:, i, j] = tensor[:, j, i] = coef[:, k]
    values = np.linalg.eigvalsh(tensor)
    spread = ((values - values.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    return np.sqrt(1.5 * spread / np.maximum((values ** 2).sum(axis=1), 1e-12))


def test_src_tensor_fit_gives_the_fib_fa():
    if not (NIIVUE / "dsistudio.src.gz").exists():
        pytest.skip("niivue demos not fetched")
    source = load(NIIVUE / "dsistudio.src.gz")[0]
    fa = next(d for d in load(NIIVUE / "dsistudio.fib.gz") if d.name.endswith("_dti_fa"))
    table = np.asarray(source.meta.vendor["b_table"], dtype=float).reshape(4, -1)
    signal = np.asarray(source.data, dtype=float).reshape(-1, source.data.shape[-1])
    b0 = signal[:, table[0] == 0].mean(axis=1)
    brain = b0 > np.percentile(b0, 70)
    ours = fractional_anisotropy(signal[brain], table[0], table[1:4].T)
    assert source.meta["DiffusionBValue"][:2] == [0.0, 2000.0]
    assert np.corrcoef(ours, np.asarray(fa.data, dtype=float).ravel()[brain])[0, 1] > 0.85


def test_template_trans_maps_onto_the_mni_box():
    if not CMU60.exists():
        pytest.skip("DSI Studio templates not fetched")
    image = load(CMU60)[0]
    assert image.data.shape == (79, 95, 69) and image.name.endswith("_fa0")
    corners = np.array([[0, 0, 0, 1], [78, 94, 68, 1]], dtype=float) @ image.affine.T
    assert np.allclose(corners[:, :3], [[78, 76, -50], [-78, -112, 86]])
