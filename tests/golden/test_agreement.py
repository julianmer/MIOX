####################################################################################################
#                                        test_agreement.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Cross-format agreement on the public test data: the same acquisition stored in several  #
#          formats must give the same values. Unlike the golden records, this is independent       #
#          evidence that the values are correct.                                                   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest

# own
from miox import Format, MRSData, load
from tests.conftest import PUBLIC_DATA
from tests.golden.records import ReadRecords

PHILIPS = PUBLIC_DATA / "spec2nii_test_data/philips/spar_dcm_orientation_tests"
SIEMENS_KEYS = (  # values twix, DICOM and RDA of one Siemens scan must share
    "SpectrometerFrequency",
    "ResonantNucleus",
    "EchoTime",
    "RepetitionTime",
    "ExcitationFlipAngle",
)
AVERAGED_KEYS = ("DwellTime", "NumberOfPoints")  # twix keeps readout oversampling, so not twix
ABSOLUTE = {"DwellTime": 5e-7}  # RDA writes the dwell time in whole µs


def _same(a: MRSData, b: MRSData, keys: tuple[str, ...]) -> list[str]:
    differences = []
    for key in keys:
        x, y = a.meta.get(key), b.meta.get(key)
        if isinstance(x, (str, list)) and np.ravel(x).dtype.kind not in "fiu":
            if x != y:
                differences.append(f"{key}: {x!r} vs {y!r}")
        elif not np.allclose(
            np.ravel(x).astype(float),
            np.ravel(y).astype(float),
            rtol=1e-6,
            atol=ABSOLUTE.get(key, 0),
        ):
            differences.append(f"{key}: {x!r} vs {y!r}")
    return differences


def _siemens_groups() -> list[list[str]]:
    """Public Siemens files of one acquisition (same protocol, patient and frequency)."""
    groups = defaultdict(list)
    for relpath, record in ReadRecords().expected().items():
        if isinstance(record, dict) or not relpath.startswith("spec2nii_test_data/Siemens"):
            continue
        meta = record[0]["meta"]
        frequency = str(meta["SpectrometerFrequency"])
        key = (meta.get("ProtocolName"), meta.get("PatientName"), frequency)
        groups[key].append(relpath)
    return [sorted(paths) for paths in groups.values() if len(paths) > 1]


@pytest.mark.data
@pytest.mark.parametrize("rotation", range(9))
def test_philips_spar_classic_enhanced_agree(rotation):
    spars = sorted(PHILIPS.glob("dbiex_4*_2_raw_act.spar"))
    classic = sorted((PHILIPS / "classic").iterdir())
    enhanced = sorted((PHILIPS / "enhanced").iterdir())
    if len(spars) < 9 or len(classic) < 9 or len(enhanced) < 9:
        pytest.skip("Philips orientation tests not fetched")
    datasets = [load(p)[0] for p in (spars[rotation], classic[rotation], enhanced[rotation])]
    keys = ("SpectrometerFrequency", "EchoTime", "RepetitionTime") + AVERAGED_KEYS
    for d in datasets[1:]:
        assert _same(datasets[0], d, keys) == []
        assert np.allclose(d.affine, datasets[0].affine, atol=1e-6)
        x, y = (v.data.ravel() / np.linalg.norm(v.data) for v in (datasets[0], d))
        assert abs(np.vdot(x, y)) > 1 - 1e-5


@pytest.mark.data
@pytest.mark.parametrize("relpaths", _siemens_groups(), ids=lambda paths: paths[0].split("/")[-1])
def test_siemens_formats_agree(relpaths):
    datasets = {}
    for relpath in relpaths:
        d = load(PUBLIC_DATA / relpath)[0]
        datasets.setdefault(d.provenance.format, d)
    formats = list(datasets)
    reference = datasets[formats[0]]
    for fmt in formats[1:]:
        d = datasets[fmt]
        keys = SIEMENS_KEYS
        if Format.SIEMENS_TWIX not in (fmt, formats[0]):
            keys += AVERAGED_KEYS
        assert _same(reference, d, keys) == [], f"{formats[0]} vs {fmt}"
        assert d.shape[:3] == reference.shape[:3]
        assert np.allclose(d.affine, reference.affine, atol=0.05), f"{formats[0]} vs {fmt}"
