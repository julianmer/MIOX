####################################################################################################
#                                   test_spec2nii_reference.py                                     #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 16/11/26                                                                                #
#                                                                                                  #
# Purpose: Formats MIOX converts itself, checked against spec2nii's conversion of the same files:  #
#          data, affine, dimension tags, dwell time and keys equal, apart from the differences     #
#          named here (each one a spec2nii defect MIOX corrects).                                  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import contextlib
import io
import warnings
from pathlib import Path

import numpy as np
import pytest

# own
import miox
from miox.readers.philips_spar_sdat import PhilipsSparSdatReader

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from spec2nii.Philips import philips
    from spec2nii.spec2nii import remove_zero_higher_dim_indices

KEYS = ("EchoTime", "RepetitionTime", "SpectrometerFrequency", "ResonantNucleus", "TxOffset",
        "ProtocolName", "SoftwareVersions", "PatientName", "PatientDoB", "Manufacturer",
        "EditPulse", "WaterSuppressed", "kSpace", "OriginalFile")
HYPER_REF_SHORT_TE = "_hyper_ref_short_te"  # spec2nii writes 80 ms on it; the block is 35 ms


def _pairs(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() == ".sdat"
                  and not p.name.startswith("._") and ".git" not in p.parts
                  and any(q.exists() for q in (p.with_suffix(".SPAR"), p.with_suffix(".spar"))))


def _spec2nii(sdat: Path) -> list:
    """spec2nii's conversion as its command line runs it, given MIOX's parse of the SPAR (spec2nii's
    own fails on indented comments and placeholder values)."""
    spar = next(q for q in (sdat.with_suffix(".SPAR"), sdat.with_suffix(".spar")) if q.exists())
    original = philips.read_spar
    philips.read_spar = lambda path: {"spectrum_inversion_time": 0,
                                      **PhilipsSparSdatReader.read_spar(Path(path))}
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            images, _ = philips.read_sdat_spar_pair(sdat, spar, None, [None, None, None])
    finally:
        philips.read_spar = original
    return [remove_zero_higher_dim_indices(image, remove=True)[0] for image in images]


def _compare(sdat: Path) -> None:
    header = PhilipsSparSdatReader.read_spar(next(
        q for q in (sdat.with_suffix(".SPAR"), sdat.with_suffix(".spar")) if q.exists()))
    if header.get("phase_encoding_enable") == "yes":
        pytest.skip("spectroscopic imaging: MIOX puts the rows on their grid, spec2nii does not")
    try:
        expected = _spec2nii(sdat)
    except ValueError as e:
        pytest.skip(f"spec2nii cannot convert it: {e}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = miox.load(sdat)
    assert len(got) == len(expected)
    for image, d in zip(expected, got, strict=True):
        np.testing.assert_array_equal(d.data, np.asarray(image[:]).conj())  # image[:] conjugates
        np.testing.assert_allclose(d.affine, image.getAffine("voxel", "world"), atol=1e-9)
        assert list(d.dim_tags) == list(image.dim_tags)
        assert d.dwelltime == pytest.approx(image.dwelltime, abs=1e-12)
        header_ext = image.hdr_ext.to_dict()
        for key in KEYS:
            if key == "EchoTime" and d.name.endswith(HYPER_REF_SHORT_TE):
                assert d.meta[key] == PhilipsSparSdatReader.HYPER_SHORT_TE
                continue
            value, reference = d.meta.get(key), header_ext.get(key)
            if isinstance(reference, float):
                assert value == pytest.approx(reference, abs=1e-12), key
            else:
                assert value == reference, key


def test_spar_sdat_equals_spec2nii(public_data):
    pairs = _pairs(public_data)
    if not pairs:
        pytest.skip("no SPAR/SDAT fetched")
    assert _compare_all(pairs) > 0


@pytest.mark.data
def test_private_spar_sdat_equals_spec2nii(private_roots):
    assert _compare_all([sdat for root in private_roots for sdat in _pairs(root)]) > 0


def _compare_all(pairs: list[Path]) -> int:
    """The number of pairs compared; those that cannot be (see _compare) are passed over."""
    compared = 0
    for sdat in pairs:
        with contextlib.suppress(pytest.skip.Exception):
            _compare(sdat)
            compared += 1
    return compared
