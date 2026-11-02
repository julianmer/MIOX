####################################################################################################
#                                         test_spm_mat.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/11/26                                                                                #
#                                                                                                  #
# Purpose: SPM.mat files of the NIDM-Results examples against the NIDM-Results documents written   #
#          from them (the project's ground truth): regressor names, contrast name and weights,     #
#          the volume's dimensions, voxel sizes and voxel-to-world mapping (NIDM's is for 0-based  #
#          voxel indices, SPM's for 1-based), the drift cutoff and the error degrees of freedom.   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import json
import re

import numpy as np
import pytest

# own
from miox import load
from tests.conftest import PUBLIC_DATA

DATA = PUBLIC_DATA / "nidmresults"
GROUND_TRUTH = DATA / "_ground_truth/1.3.0"


def _ttl_value(text: str, name: str) -> str:
    match = re.search(rf"nidm_{name}:\s*\"(.*?)\"\^\^", text)
    assert match, name
    return match[1].replace("\\", "")


def _protocol(folder: str):
    path = DATA / folder / "SPM.mat"
    if not path.is_file():
        pytest.skip(f"{folder}/SPM.mat not fetched")
    [protocol] = load(path)
    return protocol


def test_full_example_matches_its_nidm_results_document():
    """spm_full_example001: SPM12's auditory tutorial (96 scans, TR 7 s, one condition)."""
    spm = _protocol("spm_full_example001")
    ttl = (GROUND_TRUTH / "spm_full_example001/example001_spm_results.ttl").read_text()
    v = spm.meta.vendor
    assert v["regressor_names"] == json.loads(_ttl_value(ttl, "regressorNames"))
    [contrast] = v["contrasts"]
    assert contrast["name"] == _ttl_value(ttl, "contrastName") == "passive listening > rest"
    assert contrast["statistic"] == "T" and contrast["weights"] == [1.0, 0.0]
    assert v["dimensions"] == json.loads(_ttl_value(ttl, "dimensionsInVoxels"))
    assert np.array(v["affine"])[:3, :3].diagonal().tolist() == pytest.approx(
        [-3.0, 3.0, 3.0]) and json.loads(_ttl_value(ttl, "voxelSize")) == [3, 3, 3]
    world = np.array(json.loads(_ttl_value(ttl, "voxelToWorldMapping")), dtype=float)
    shifted = np.array(v["affine"]) @ np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1],
                                                [0, 0, 0, 1]])
    assert np.allclose(shifted, world)
    assert v["SPM"]["xX"]["K"]["HParam"] == float(re.search(
        r"spm_SPMsDriftCutoffPeriod:\s*\"(\d+)\"", ttl)[1]) == 128
    assert v["SPM"]["xX"]["erdf"] == pytest.approx(float(_ttl_value(ttl, "errorDegreesOfFreedom")))
    assert spm.meta["RepetitionTime"] == 7.0 and v["scans_per_session"] == [96]
    assert np.array(v["design_matrix"]).shape == (96, 2)
    assert v["images"][0].endswith(".img") and len(v["images"]) == 96


def test_default_example_contrast_and_hrf():
    """spm_default: a t contrast [1 0 0] on the canonical HRF, as the ground-truth fragments its
    config names say."""
    spm = _protocol("spm_default")
    v = spm.meta.vendor
    assert v["SPM"]["xBF"]["name"] == "hrf"
    assert (GROUND_TRUTH / "hrf_spmcanonical/nidm.ttl").read_text().count("SPMsCanonicalHRF") >= 1
    [contrast] = v["contrasts"]
    assert contrast["weights"] == [1.0, 0.0, 0.0] and contrast["statistic"] == "T"
    con_t = (GROUND_TRUTH / "con_t/nidm.ttl").read_text()
    assert "tstatistic" in con_t.lower() or "obo_tstatistic" in con_t
    assert len(v["regressor_names"]) == 3 and v["regressor_names"][-1] == "Sn(1) constant"
    assert spm.meta["RepetitionTime"] == 2.0 and v["scans_per_session"] == [104]
