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
from miox.readers.base import ReaderError
from miox.readers.philips_data_list import PhilipsDataListReader
from miox.readers.philips_list_raw import PhilipsListRaw
from miox.readers.philips_spar_sdat import PhilipsSparSdatReader

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from spec2nii.Philips import philips, philips_data_list
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


def _compare_all(pairs: list[Path], compare=_compare) -> int:
    """The number of pairs compared; those that cannot be (see _compare) are passed over."""
    compared = 0
    for path in pairs:
        with contextlib.suppress(pytest.skip.Exception):
            compare(path)
            compared += 1
    return compared


#**************************************************************************************************#
#                                     data/list (with SPAR or DICOM)                               #
#**************************************************************************************************#
#                                                                                                  #
# Differences named: HYPER short-TE block at 35 ms (spec2nii: the SPAR's 80 ms); WaterSuppressed   #
# on the HYPER metabolite blocks (spec2nii: none); HERCULES conditions B and C (spec2nii's         #
# data/list swaps their pulses); HYPER water references with their echo time per index (spec2nii:  #
# one, 80 ms); echo and repetition time of a classic DICOM in s (spec2nii: ms). Edited (MEGA)      #
# scans: spec2nii keeps extr1 x extr2, MIOX pairs the transients; checked transient by transient,  #
# ON being the transients the .list puts at extr2 1.                                               #
#                                                                                                  #
#**************************************************************************************************#
def _data_lists(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() == ".data"
                  and not p.name.startswith("._") and p.with_suffix(".list").exists())


def _spec2nii_data_list(data: Path, aux: Path) -> dict:
    """spec2nii's conversion by name, as its command line runs it, given MIOX's parse of a SPAR."""
    parse = lambda path: {"spectrum_inversion_time": 0,  # noqa: E731
                          **PhilipsSparSdatReader.read_spar(Path(path))}
    originals = philips.read_spar, philips_data_list.read_spar
    philips.read_spar = philips_data_list.read_spar = parse
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            images, names = philips_data_list.read_data_list_pair(
                data, data.with_suffix(".list"), aux, None)
    finally:
        philips.read_spar, philips_data_list.read_spar = originals
    return {name: remove_zero_higher_dim_indices(image, remove=True)[0]
            for image, name in zip(images, names, strict=True)}


def _compare_data_list(data: Path) -> None:
    reader = PhilipsDataListReader()
    parsed = reader._list(data.with_suffix(".list"))
    rows = PhilipsListRaw.vectors(parsed["list_vectors"])
    if PhilipsListRaw.imaging(parsed["list_info"], rows):
        pytest.skip("imaging k-space, read as RawData")
    try:
        aux = reader._auxiliary(data.parent, data.stem)
    except ReaderError as e:
        pytest.skip(str(e))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            got = miox.load(data)
        except ReaderError as e:
            pytest.skip(str(e))
    expected = _spec2nii_data_list(data, aux)
    assert [d.name for d in got] == [f"{data.stem}_{name}" for name in expected]
    extr2 = {r["extr1"]: r["extr2"] for r in rows if r["typ"] == "STD" and r["mix"] == 0}
    dicom = aux.suffix.lower() != ".spar"
    for d, (name, image) in zip(got, expected.items(), strict=True):
        reference = np.asarray(image[:]).conj()  # image[:] conjugates
        if "DIM_EDIT" in d.dim_tags and not name.startswith("hyper"):
            _compare_edit_pairs(d, reference, extr2)
        else:
            np.testing.assert_array_equal(d.data, reference)
            assert list(d.dim_tags) == list(image.dim_tags)
        np.testing.assert_allclose(d.affine, image.getAffine("voxel", "world"), atol=1e-9)
        assert d.dwelltime == pytest.approx(image.dwelltime, abs=1e-12)
        header_ext = image.hdr_ext.to_dict()
        for key in KEYS:
            value, reference = d.meta.get(key), header_ext.get(key)
            if key == "EchoTime" and name == "hyper_short_te":
                reference = PhilipsSparSdatReader.HYPER_SHORT_TE
            elif key == "EchoTime" and name == "hyper_water_ref":
                assert value is None
                times = d.dim_info[6]["header"]["EchoTime"]
                assert times[0::2] == [PhilipsSparSdatReader.HYPER_EDITED_TE] * len(times[0::2])
                assert times[1::2] == [PhilipsSparSdatReader.HYPER_SHORT_TE] * len(times[1::2])
                continue
            elif key in ("EchoTime", "RepetitionTime") and dicom and reference is not None:
                reference = reference * 1e-3
            elif key == "WaterSuppressed" and name in ("hyper_short_te", "hyper_edited"):
                reference = True
            elif key == "EditPulse" and name == "hyper_edited":
                reference = PhilipsSparSdatReader.HERCULES_EDIT_PULSES
            if isinstance(reference, float):
                assert value == pytest.approx(reference, abs=1e-12), key
            else:
                assert value == reference, key


def _compare_edit_pairs(d, reference: np.ndarray, extr2: dict[int, int]) -> None:
    """MIOX's pairs (ON, OFF) against spec2nii's (x, y, z, points, coils, extr1, extr2 slot): each
    transient is the one of its extr1 whose .list extr2 is 1 (ON) or 0 (OFF), in extr1 order, over
    the pairs spec2nii holds complete in every coil."""
    assert d.dim_tags[:3] == ("DIM_COIL", "DIM_DYN", "DIM_EDIT")
    transients = reference.sum(axis=-1)  # one slot of the two is filled
    present = (np.abs(transients) > 0).any(axis=3).all(axis=(0, 1, 2, 3))  # every coil
    pairs = [j for j in range(transients.shape[-1] // 2) if present[2 * j] and present[2 * j + 1]]
    for condition, value in ((0, 1), (1, 0)):
        indices = [next(i for i in (2 * j, 2 * j + 1) if extr2[i] == value) for j in pairs]
        np.testing.assert_array_equal(np.asarray(d.data)[..., condition],
                                      transients[..., indices])


def test_data_list_equals_spec2nii(public_data):
    files = _data_lists(public_data)
    if not files:
        pytest.skip("no data/list fetched")
    assert _compare_all(files, _compare_data_list) > 0


def test_data_list_with_dicom_equals_spec2nii(public_data, tmp_path):
    """The HYPER data/list with the scan's DICOM as the only auxiliary file."""
    folder = public_data / "spec2nii_test_data" / "philips" / "hyper"
    if not (folder / "converted_dcm.dcm").exists():
        pytest.skip("spec2nii test data not fetched")
    for name in ("raw_226.data", "raw_226.list", "converted_dcm.dcm"):
        (tmp_path / name).symlink_to(folder / name)
    _compare_data_list(tmp_path / "raw_226.data")


@pytest.mark.data
def test_private_data_list_equals_spec2nii(private_roots):
    files = [data for root in private_roots for data in _data_lists(root)]
    assert _compare_all(files, _compare_data_list) > 0
