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
#          named here (each one a spec2nii defect MIOX corrects); in every format an empty text    #
#          spec2nii writes is no value in MIOX.                                                    #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import argparse
import contextlib
import ctypes
import io
import struct
import warnings
from pathlib import Path

import numpy as np
import pydicom
import pytest
from mapvbvd import mapVBVD

# own
import miox
from miox import Format, Sniffer
from miox.readers import ge_rdb_fields
from miox.readers.base import ReaderError
from miox.readers.bruker import BrukerReader
from miox.readers.ge_pfile_mrs import GePfileMrs
from miox.readers.ge_rdb import GeRdbHeader
from miox.readers.philips_data_list import PhilipsDataListReader
from miox.readers.philips_dicom_mrs import PhilipsDicomMrs
from miox.readers.philips_list_raw import PhilipsListRaw
from miox.readers.philips_spar_sdat import PhilipsSparSdatReader
from miox.readers.siemens_dicom_mrs import SiemensDicomMrs
from miox.readers.siemens_twix import SiemensTwixReader, TwixCsi

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from spec2nii.bruker import read_bruker
    from spec2nii.GE import ge_pfile
    from spec2nii.GE.ge_hdr_fields import get_pfile_hdr_fields
    from spec2nii.GE.ge_read_pfile import Pfile
    from spec2nii.jmrui import jmrui_format
    from spec2nii.Philips import philips, philips_data_list, philips_dcm
    from spec2nii.Siemens.dicomfunctions import multi_file_dicom as siemens_multi_file_dicom
    from spec2nii.Siemens.rda import convert_rda
    from spec2nii.Siemens.twixfunctions import process_twix
    from spec2nii.spec2nii import remove_zero_higher_dim_indices
    from spec2nii.uih import multi_file_dicom as uih_multi_file_dicom
    from spec2nii.varian_importer import read_varian

KEYS = ("EchoTime", "RepetitionTime", "SpectrometerFrequency", "ResonantNucleus", "TxOffset",
        "ProtocolName", "SoftwareVersions", "PatientName", "PatientDoB", "Manufacturer",
        "EditPulse", "WaterSuppressed", "kSpace", "OriginalFile")
HYPER_REF_SHORT_TE = "_hyper_ref_short_te"  # spec2nii writes 80 ms on it; the block is 35 ms


def _stated(reference):
    """spec2nii's value as MIOX states it: an empty text is no value."""
    return None if isinstance(reference, str) and not reference.strip() else reference


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
            reference = _stated(reference)
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
# ON being the transients the .list puts at extr2 1. A folder with several SPARs that may be the   #
# scan's is passed over: MIOX keeps what they state alike, spec2nii takes the one it is given.     #
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
    aux, others = reader._auxiliary(data.parent, data.stem, [])
    if aux is None:
        pytest.skip("no SPAR or DICOM of the scan: spec2nii needs one, MIOX reads the data alone")
    if others:
        pytest.skip("several SPARs may be the scan's: MIOX keeps what they state alike")
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
            reference = _stated(reference)
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


#**************************************************************************************************#
#                                          Philips DICOM                                           #
#**************************************************************************************************#
#                                                                                                  #
# Against spec2nii's philips_dicom (multi_file_dicom with its default tag DIM_DYN), one file or    #
# one series of files. Differences named: spec2nii tags a 5th dimension on single spectra, which   #
# have none; classic files: echo time ((2005,1310)) and repetition time in s (spec2nii: ms), the   #
# flip angle from FlipAngle (spec2nii: the repetition time where per-frame groups exist). Files    #
# spec2nii cannot convert (newer software's frame flags, one slab, spectroscopic imaging) are      #
# passed over; the imaging grids are checked against the SPAR/SDAT in test_philips_spectroscopy.   #
#                                                                                                  #
#**************************************************************************************************#
DICOM_KEYS = KEYS + ("ExcitationFlipAngle", "ManufacturersModelName", "DeviceSerialNumber",
                     "InstitutionName", "InstitutionAddress", "SequenceName", "PatientPosition",
                     "PatientWeight", "PatientSex")


def _philips_dicoms(roots: list[Path]) -> list[list[Path]]:
    """Philips spectroscopy DICOM series: [file] each, and the files of a folder that share one
    series as one list."""
    sniffer, series = Sniffer(), {}
    for root in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name.startswith(".") or path.suffix.lower() in (
                    ".spar", ".sdat", ".data", ".list", ".dat", ".nii", ".gz", ".json", ".txt"):
                continue
            with contextlib.suppress(Exception):
                sniff = sniffer.sniff(path)
                if sniff.format == Format.DICOM_MRS and sniff.vendor == "philips":
                    uid = pydicom.dcmread(path, stop_before_pixels=True).SeriesInstanceUID
                    series.setdefault((path.parent, uid), []).append(path)
    singles = [[p] for paths in series.values() for p in paths]
    return singles + [paths for paths in series.values() if len(paths) > 1]


def _compare_philips_dicom(paths: list[Path], tmp_path: Path) -> None:
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            images, names = philips_dcm.multi_file_dicom(paths, None, "DIM_DYN", False)
    except Exception as e:  # noqa: BLE001  any failure of spec2nii's own reading
        pytest.skip(f"spec2nii cannot convert it: {type(e).__name__}")
    expected = [remove_zero_higher_dim_indices(image, remove=True)[0] for image in images]
    if any(np.asarray(image[:]).size == 0 for image in expected):
        pytest.skip("spec2nii's edit split leaves no frames (numeric edit flags)")
    target = paths[0]
    if len(paths) > 1:  # the series alone in a folder
        target = tmp_path / f"series_{abs(hash(tuple(paths)))}"
        target.mkdir()
        for path in paths:
            (target / path.name).symlink_to(path.resolve())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = miox.load(target)
    ds = pydicom.dcmread(paths[0], stop_before_pixels=True)
    classic = ds.file_meta.MediaStorageSOPClassUID != PhilipsDicomMrs.ENHANCED
    datas = [np.asarray(image[:]).conj() for image in expected]  # image[:] conjugates
    items, split = ds.get((0x2005, 0x140F)), False
    if len(expected) == 1 and len(got) == 2 and items is not None and (0x2005, 0x1004) in \
            items.value[0] and (0x2005, 0x1304) not in items.value[0]:
        # newer software flags the water references in (2005,1004); spec2nii keeps them as data
        reference = np.array([int(item[0x2005, 0x1004].value) == 1 for item in items.value])
        datas = [datas[0][..., ~reference], datas[0][..., reference]]
        expected, names, split = expected * 2, [names[0], names[0] + "_ref"], True
    assert [d.name for d in got] == list(names)
    for d, image, data in zip(got, expected, datas, strict=True):
        np.testing.assert_array_equal(d.data, data)
        np.testing.assert_allclose(d.affine, image.getAffine("voxel", "world"), atol=1e-9)
        assert d.dwelltime == pytest.approx(image.dwelltime, abs=1e-12)
        tags = list(image.dim_tags)
        if d.data.ndim == 4:  # spec2nii tags a 5th dimension the data do not have
            tags = [None, None, None]
        assert list(d.dim_tags) == tags, (d.name, d.dim_tags, tags)
        header_ext = image.hdr_ext.to_dict()
        for key in DICOM_KEYS:
            value, reference = d.meta.get(key), header_ext.get(key)
            reference = _stated(reference)
            if classic and key in ("EchoTime", "RepetitionTime") and reference is not None:
                reference = float(reference) * 1e-3
            elif key == "WaterSuppressed" and split and d.name.endswith("_ref"):
                reference = False
            elif classic and key == "ExcitationFlipAngle":
                assert value == float(ds.FlipAngle)
                continue
            if isinstance(reference, float):
                assert value == pytest.approx(reference, abs=1e-12), key
            else:
                assert value == reference, key


def _compare_all_dicoms(series: list[list[Path]], tmp_path: Path) -> int:
    compared = 0
    for paths in series:
        with contextlib.suppress(pytest.skip.Exception):
            _compare_philips_dicom(paths, tmp_path)
            compared += 1
    return compared


def test_philips_dicom_equals_spec2nii(public_data, tmp_path):
    roots = [public_data / name for name in ("spec2nii_test_data", "nema_wg16", "suspect",
                                              "tarquin") if (public_data / name).is_dir()]
    series = _philips_dicoms(roots)
    if not series:
        pytest.skip("no Philips DICOM fetched")
    assert _compare_all_dicoms(series, tmp_path) > 0


@pytest.mark.data
def test_private_philips_dicom_equals_spec2nii(private_roots, tmp_path):
    assert _compare_all_dicoms(_philips_dicoms(private_roots), tmp_path) > 0


#**************************************************************************************************#
#                                            GE P-file                                             #
#**************************************************************************************************#
#                                                                                                  #
# Against spec2nii's own Pfile and processing (svs or mrsi), every spectroscopy P-file. HBCD PRESS #
# is split by the header's water (user19) and metabolite (user4) counts in MIOX, at 16 / nex in    #
# spec2nii: the frames of both splits, references first, are checked equal. Files spec2nii cannot  #
# read (revisions before 12, protocol names it does not map) are passed over. The header layouts   #
# (ge_rdb_fields) are checked against spec2nii's ctypes structures for every revision it knows,    #
# and the header values of every P-file, spectroscopy or not, against its reading. Differences     #
# named: no InversionTime where the header holds 0 and no PatientSex where none was entered        #
# (spec2nii: 0.0 and "O").                                                                         #
#                                                                                                  #
#**************************************************************************************************#
SPEC2NII_REVISIONS = (7, 8, 9, 10, 11, 14.0, 14.1, 14.2, 14.3, 15.000, 15.001, 16.000, 20.001,
                      20.002, 20.003, 20.004, 20.005, 20.006, 20.007, 24.000, 25.001, 25.002,
                      25.003, 25.004, 26.000, 26.001, 26.002, 27.000, 27.001, 28.000, 28.002,
                      28.003, 29.1, 30.0, 30.1, 31.0)  # Pfile._version's known revisions


def test_ge_rdb_layouts_equal_spec2nii():
    for version in SPEC2NII_REVISIONS:
        if version == 10:  # spec2nii gives it the 21/22 layout (a fall-through), MIOX none
            assert ge_rdb_fields.layout(version) is None
            continue
        fields = get_pfile_hdr_fields(version)
        structure = type("Hdr", (ctypes.LittleEndianStructure,), {"_pack_": 1, "_fields_": fields})
        expected = {name: (getattr(structure, name).offset, ctypes.sizeof(kind), kind)
                    for name, kind in fields if not name.startswith("pad")}
        got = {name: (offset, struct.calcsize("<" + fmt), fmt)
               for name, offset, fmt in ge_rdb_fields.layout(version)}
        for name, (offset, size, kind) in expected.items():
            assert got[name][:2] == (offset, size), (version, name)
            if hasattr(kind, "_length_"):
                assert got[name][2] == f"{kind._length_}s", (version, name)
            else:  # the same signedness: a negative value reads back the same
                assert kind(-1).value < 0 if got[name][2].islower() else kind(-1).value > 0


def _all_pfiles(roots: list[Path]) -> list[Path]:
    return [path for root in roots for path in sorted(root.rglob("*.7"))
            if path.is_file() and not path.name.startswith("._")]


def _compare_ge_header(path: Path) -> None:
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pfile = Pfile.__new__(Pfile)
            pfile.file_name, pfile.endian = str(path), "little"
            pfile.read_header()
    except Exception as e:  # noqa: BLE001  any failure of spec2nii's own reading
        pytest.skip(f"spec2nii cannot read the header: {type(e).__name__}")
    header = GeRdbHeader.from_file(path)
    assert header.version == pytest.approx(pfile.version, abs=5e-4)
    assert header.endian == pfile.endian
    for name, _ in pfile.hdr._fields_:
        if not name.startswith("pad"):
            assert getattr(header.hdr, name) == getattr(pfile.hdr, name), name


def test_ge_rdb_header_equals_spec2nii(public_data):
    files = _all_pfiles([public_data])
    if not files:
        pytest.skip("no GE P-files fetched")
    assert _compare_all(files, _compare_ge_header) > 0


@pytest.mark.data
def test_private_ge_rdb_header_equals_spec2nii(private_roots):
    assert _compare_all(_all_pfiles(private_roots), _compare_ge_header) > 0


GE_KEYS = KEYS + ("ExcitationFlipAngle", "InversionTime", "ManufacturersModelName",
                  "DeviceSerialNumber", "InstitutionName", "RxCoil", "SequenceName",
                  "PatientSex")


def _ge_pfiles(roots: list[Path]) -> list[Path]:
    files = []
    for root in roots:
        for path in sorted(root.rglob("*.7")):
            if path.is_file() and not path.name.startswith("._"):
                with contextlib.suppress(Exception):
                    if GePfileMrs.recognises(GeRdbHeader.from_file(path).hdr):
                        files.append(path)
    return files


def _compare_ge(path: Path) -> None:
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pfile = Pfile(str(path))
            process = ge_pfile._process_svs_pfile if pfile.is_svs else ge_pfile._process_mrsi_pfile
            images, suffixes = process(pfile)
    except Exception as e:  # noqa: BLE001  any failure of spec2nii's own reading
        pytest.skip(f"spec2nii cannot convert it: {type(e).__name__}")
    expected = [remove_zero_higher_dim_indices(image, remove=True)[0] for image in images]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = miox.load(path)
    assert [d.name for d in got] == [path.stem + suffix for suffix in suffixes]
    datas = [np.asarray(image[:]).conj() for image in expected]  # image[:] conjugates
    hbcd_press = pfile.hdr.rhi_psdname.lower().startswith(b"hbcd") and b"press" in \
        pfile.hdr.rhs_se_desc.lower()
    if hbcd_press:  # [metabolites, references]: frames in file order, references first
        ours = np.concatenate([np.asarray(got[1].data), np.asarray(got[0].data)], axis=-1)
        np.testing.assert_array_equal(ours, np.concatenate([datas[1], datas[0]], axis=-1))
    for d, image, data in zip(got, expected, datas, strict=True):
        if not hbcd_press:
            np.testing.assert_array_equal(d.data, data)
        np.testing.assert_allclose(d.affine, image.getAffine("voxel", "world"), atol=1e-9)
        assert d.dwelltime == pytest.approx(image.dwelltime, abs=1e-12)
        assert list(d.dim_tags) == list(image.dim_tags), (d.name, d.dim_tags, image.dim_tags)
        header_ext = image.hdr_ext.to_dict()
        for key in GE_KEYS:
            value, reference = d.meta.get(key), header_ext.get(key)
            reference = _stated(reference)
            if (key, reference) in (("InversionTime", 0.0), ("PatientSex", "O")):
                reference = None  # the header states neither: MIOX leaves the key out
            if isinstance(reference, float):
                assert value == pytest.approx(reference, abs=1e-12), key
            else:
                assert value == reference, key


def test_ge_pfile_equals_spec2nii(public_data):
    files = _ge_pfiles([public_data / name for name in ("spec2nii_test_data", "tarquin")
                        if (public_data / name).is_dir()])
    if not files:
        pytest.skip("no GE P-files fetched")
    assert _compare_all(files, _compare_ge) > 0


@pytest.mark.data
def test_private_ge_pfile_equals_spec2nii(private_roots):
    assert _compare_all(_ge_pfiles(private_roots), _compare_ge) > 0


#**************************************************************************************************#
#                                               RDA                                                #
#**************************************************************************************************#
#                                                                                                  #
# Against spec2nii's convert_rda, every RDA file. Differences named: text values without the       #
# padding spaces the scanner writes after them, and fields the header leaves empty not set         #
# (spec2nii: an empty text).                                                                       #
#                                                                                                  #
#**************************************************************************************************#
RDA_KEYS = KEYS + ("InversionTime", "MixingTime", "ExcitationFlipAngle",
                   "ManufacturersModelName", "DeviceSerialNumber", "InstitutionName", "TxCoil",
                   "SequenceName", "PatientPosition", "PatientID", "PatientWeight", "PatientSex")


def _rdas(roots: list[Path]) -> list[Path]:
    return sorted(p for root in roots for p in root.rglob("*") if p.suffix.lower() == ".rda"
                  and not p.name.startswith("._") and p.is_file())


def _compare_rda(path: Path) -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        images, names = convert_rda(path, None, False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = miox.load(path)
    assert [d.name for d in got] == list(names)
    for d, image in zip(got, images, strict=True):
        np.testing.assert_array_equal(d.data, np.asarray(image[:]).conj())  # image[:] conjugates
        np.testing.assert_allclose(d.affine, image.getAffine("voxel", "world"), atol=1e-9)
        assert d.dwelltime == pytest.approx(image.dwelltime, abs=1e-12)
        assert list(d.dim_tags) == list(image.dim_tags)
        header_ext = image.hdr_ext.to_dict()
        for key in RDA_KEYS:
            value, reference = d.meta.get(key), header_ext.get(key)
            reference = _stated(reference)
            if isinstance(reference, str):
                reference = reference.strip() or None
            if isinstance(reference, float):
                assert value == pytest.approx(reference, abs=1e-12), key
            else:
                assert value == reference, key


def test_rda_equals_spec2nii(public_data):
    files = _rdas([public_data])
    if not files:
        pytest.skip("no RDA fetched")
    assert _compare_all(files, _compare_rda) == len(files)


@pytest.mark.data
def test_private_rda_equals_spec2nii(private_roots):
    files = _rdas(private_roots)
    assert _compare_all(files, _compare_rda) == len(files) > 0


#**************************************************************************************************#
#                                              twix                                                #
#**************************************************************************************************#
#                                                                                                  #
# Against spec2nii's process_twix (oversampling kept, no dimension overrides), every stream of     #
# every spectroscopy twix file; a stream spec2nii cannot lay out is one MIOX does not convert      #
# either. Differences named: DeviceSerialNumber and PatientDoB without the '.0' of a float, coil   #
# names without quotes, the inversion time from MeasYaps alTI[0] in s (spec2nii: µs on XA, none or #
# TI_Time unscaled on VB to VE), XA product reference scans not water suppressed (spec2nii: they   #
# are; test_xa_reference_scans_are_unsuppressed); grids MIOX reconstructs (TwixCsi) hold data      #
# where spec2nii's are empty: there only the geometry and keys are compared.                       #
#                                                                                                  #
#**************************************************************************************************#
TWIX_KEYS = KEYS + ("ExcitationFlipAngle", "ManufacturersModelName", "DeviceSerialNumber",
                    "InstitutionName", "InstitutionAddress", "RxCoil", "TxCoil", "SequenceName",
                    "PatientPosition", "PatientWeight", "PatientSex", "PulseSequenceFile",
                    "IceProgramFile")
NO_OVERRIDES = {"dims": (None, None, None), "tags": (None, None, None)}


def _twix_measurement(path: Path):
    twix = mapVBVD(str(path), quiet=True)
    if isinstance(twix, list):
        twix = twix[min(SiemensTwixReader.MULTIRAID, len(twix)) - 1]
    return twix


def _twixes(roots: list[Path]) -> list[Path]:
    """Spectroscopy twix files."""
    sniffer, files = Sniffer(), []
    for root in roots:
        for path in sorted(root.rglob("*.dat")):
            if not path.is_file() or path.name.startswith("._"):
                continue
            with contextlib.suppress(Exception):
                if sniffer.sniff(path).format == Format.SIEMENS_TWIX:
                    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        if SiemensTwixReader.is_spectroscopy(_twix_measurement(path)):
                            files.append(path)
    return files


def _spec2nii_twix(path: Path) -> list[tuple[str, object]]:
    """spec2nii's datasets of every stream it can lay out, named as MIOX names them."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        twix = _twix_measurement(path)
        out = []
        streams = [s for s in twix.keys() if s != "hdr"]
        for stream in ["image"] + [s for s in streams if s != "image"]:
            try:
                images, names = process_twix(twix, None, path.name, stream, NO_OVERRIDES,
                                             quiet=True, remove_os=False)
                images = [remove_zero_higher_dim_indices(i, remove=True)[0] for i in images]
                np.asarray(images[0][:])  # a header spec2nii writes but cannot validate fails here
            except Exception:  # noqa: BLE001  any failure of spec2nii's own layout
                if stream == "image":
                    raise
                continue
            suffix = "" if stream == "image" else f"_{stream}"
            out += [(name + suffix, image) for name, image in zip(names, images, strict=True)]
    return out


def _compare_twix(path: Path) -> None:
    try:
        expected = _spec2nii_twix(path)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"spec2nii cannot convert it: {type(e).__name__}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = miox.load(path)
    reconstructed = any(TwixCsi.NOTE in d.provenance.notes for d in got)
    names = [n.removesuffix("_empty") if reconstructed and n.endswith("_empty") and "_" not in
             n.removeprefix(path.stem).removesuffix("_empty") else n for n, _ in expected]
    assert [d.name for d in got] == names
    alti = _twix_measurement(path)["hdr"]["MeasYaps"].get(("alTI", "0"))
    for d, (name, image) in zip(got, expected, strict=True):
        if not (reconstructed and d.name + "_empty" == name):
            np.testing.assert_array_equal(d.data, np.asarray(image[:]).conj())  # [:] conjugates
            assert list(d.dim_tags) == list(image.dim_tags), (d.name, d.dim_tags)
        np.testing.assert_allclose(d.affine, image.getAffine("voxel", "world"), atol=1e-9)
        assert d.dwelltime == pytest.approx(image.dwelltime, abs=1e-15)
        header_ext = image.hdr_ext.to_dict()
        for key in TWIX_KEYS:
            value, reference = d.meta.get(key), header_ext.get(key)
            reference = _stated(reference)
            if key in ("DeviceSerialNumber", "PatientDoB") and isinstance(reference, str):
                reference = reference.removesuffix(".0")
            elif key in ("RxCoil", "TxCoil") and isinstance(reference, str):
                reference = reference.strip('"')
            elif key == "WaterSuppressed" and name.endswith("_ref"):
                reference = False  # XA product reference scans: unsuppressed water
            if isinstance(reference, float):
                assert value == pytest.approx(reference, abs=1e-12), key
            else:
                assert value == reference, key
        inversion = alti * 1e-6 if isinstance(alti, (int, float)) else None
        assert d.meta.get("InversionTime") == inversion


def test_twix_equals_spec2nii(public_data):
    files = _twixes([public_data])
    if not files:
        pytest.skip("no twix fetched")
    assert _compare_all(files, _compare_twix) == len(files)


@pytest.mark.data
def test_private_twix_equals_spec2nii(private_roots):
    files = _twixes(private_roots)
    assert _compare_all(files, _compare_twix) == len(files) > 0


def test_xa_reference_scans_are_unsuppressed(public_data):
    """The reference transients of a water-suppressed XA svs_se scan carry the full water signal
    (hundreds of times the scan's), those of the same protocol without suppression as much as its
    own transients."""
    folder = public_data / "spec2nii_test_data" / "Siemens" / "XAData" / "XA20" / "twix"
    on, off = (folder / f"meas_MID000{n}_svs_se_30_WS_{s}.dat" for n, s in
               (("27_FID07653", "on"), ("33_FID07659", "off")))
    if not on.exists() or not off.exists():
        pytest.skip("spec2nii test data not fetched")
    ratios = []
    for path in (on, off):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref, scan = miox.load(path)[:2]
        assert ref.name.endswith("_ref") and ref.meta["WaterSuppressed"] is False
        ratios.append(np.abs(ref.data[0, 0, 0, :8]).mean() / np.abs(scan.data[0, 0, 0, :8]).mean())
    assert ratios[0] > 100 and 0.8 < ratios[1] < 1.25


#**************************************************************************************************#
#                                       Siemens and UIH DICOM                                      #
#**************************************************************************************************#
#                                                                                                  #
# Against spec2nii's Siemens (tag DIM_DYN) and UIH multi_file_dicom, one file or one series of     #
# files. Differences named: OriginalFile holds the file names (spec2nii's Siemens converter: the   #
# paths as given); no tag on a 5th dimension the data lack (spec2nii tags one on single spectra);  #
# the CSA inversion time in s (spec2nii: ms), left out where the header states 0 (no inversion;    #
# spec2nii writes the 0); the echo time the CSA header states, alTE[0]                             #
# (spec2nii adds up every alTE entry, stale ones too); the references of svs_slaser_dkd2 (scan     #
# mode 2) by instance numbers counted from 1 (spec2nii counts them from 0 and moves one transient: #
# only the transients as a whole are compared, test_agreement checks the split against the twix    #
# files). Files spec2nii cannot convert (syngo VA, an orientation of fewer than six values) are    #
# passed over.                                                                                     #
#                                                                                                  #
#**************************************************************************************************#
SIEMENS_DICOM_KEYS = tuple(k for k in KEYS if k != "OriginalFile") + (
                             "ExcitationFlipAngle", "ManufacturersModelName",
                             "DeviceSerialNumber", "InstitutionName", "InstitutionAddress",
                             "TxCoil", "RxCoil", "SequenceName", "PatientPosition", "PatientID",
                             "PatientWeight", "PatientSex", "PulseSequenceFile", "VOI")


def _vendor_dicoms(roots: list[Path], vendors: tuple[str, ...]) -> list[list[Path]]:
    """DICOM spectroscopy series of these vendors: [file] each, and the files of a folder that
    share one series as one list."""
    sniffer, series = Sniffer(), {}
    for root in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name.startswith(".") or ".git" in path.parts or \
                    path.suffix.lower() in (".spar", ".sdat", ".data", ".list", ".dat", ".nii",
                                            ".gz", ".json", ".txt", ".rda", ".7"):
                continue
            with contextlib.suppress(Exception):
                sniff = sniffer.sniff(path)
                if sniff.format == Format.DICOM_MRS and sniff.vendor in vendors:
                    uid = pydicom.dcmread(path, stop_before_pixels=True).SeriesInstanceUID
                    series.setdefault((path.parent, uid, sniff.vendor), []).append(path)
    singles = [[p] for paths in series.values() for p in paths]
    return singles + [paths for paths in series.values() if len(paths) > 1]


def _compare_vendor_dicom(paths: list[Path], tmp_path: Path) -> None:
    uih = Sniffer().sniff(paths[0]).vendor == "uih"
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            convert = uih_multi_file_dicom if uih else siemens_multi_file_dicom
            images, names = convert(paths, None, None if uih else "DIM_DYN", False)
    except Exception as e:  # noqa: BLE001  any failure of spec2nii's own reading
        pytest.skip(f"spec2nii cannot convert it: {type(e).__name__}")
    expected = [remove_zero_higher_dim_indices(image, remove=True)[0] for image in images]
    target = paths[0]
    if len(paths) > 1:  # the series alone in a folder
        target = tmp_path / f"series_{abs(hash(tuple(paths)))}"
        target.mkdir()
        for path in paths:
            (target / path.name).symlink_to(path.resolve())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = miox.load(target)
    if any(name.endswith("_vapor_ovs_rfoff") for name in [d.name for d in got] + list(names)):
        mine = {t.tobytes() for d in got for t in d.data.reshape(d.data.shape[3], -1).T}
        theirs = {t.tobytes() for image in expected
                  for t in np.asarray(image[:]).conj().reshape(image.shape[3], -1).T}
        assert mine == theirs
        return
    assert [d.name for d in got] == list(names)
    for d, image in zip(got, expected, strict=True):
        np.testing.assert_array_equal(d.data, np.asarray(image[:]).conj())  # [:] conjugates
        np.testing.assert_allclose(d.affine, image.getAffine("voxel", "world"), atol=1e-9)
        assert d.dwelltime == pytest.approx(image.dwelltime, abs=1e-15)
        tags = list(image.dim_tags)
        if d.data.ndim == 4:  # spec2nii tags a 5th dimension the data do not have
            tags = [None, None, None]
        assert list(d.dim_tags) == tags, (d.name, d.dim_tags, tags)
        header_ext = image.hdr_ext.to_dict()
        assert d.meta["OriginalFile"] == [Path(f).name for f in header_ext["OriginalFile"]]
        for key in SIEMENS_DICOM_KEYS:
            value, reference = d.meta.get(key), header_ext.get(key)
            reference = _stated(reference)
            if key == "PatientID" and reference is None:
                continue  # MIOX adds it from the element (syngo VB to VE: spec2nii does not)
            if key == "EchoTime" and not uih and value != pytest.approx(reference, abs=1e-12):
                csa = SiemensDicomMrs.csa(pydicom.dcmread(paths[0]))  # spec2nii: sum of alTE
                reference = float(csa["EchoTime"][0]) * 1e-3
            if isinstance(reference, float):
                assert value == pytest.approx(reference, abs=1e-12), key
            elif key == "VOI" and reference is not None:
                np.testing.assert_allclose(value, reference, atol=1e-9)
            else:
                assert value == reference, key
        reference = header_ext.get("InversionTime")
        scale = 1 if uih else 1e-3  # the CSA value in ms
        assert d.meta.get("InversionTime") == (None if not reference else  # 0: no inversion
                                               pytest.approx(reference * scale, abs=1e-12))


def _compare_all_vendor_dicoms(series: list[list[Path]], tmp_path: Path) -> int:
    compared = 0
    for paths in series:
        with contextlib.suppress(pytest.skip.Exception):
            _compare_vendor_dicom(paths, tmp_path)
            compared += 1
    return compared


def test_siemens_uih_dicom_equals_spec2nii(public_data, tmp_path):
    series = _vendor_dicoms([public_data], ("siemens", "uih"))
    if not series:
        pytest.skip("no Siemens or UIH DICOM fetched")
    assert _compare_all_vendor_dicoms(series, tmp_path) == len(series) - 4  # 3 VA, 1 orientation


@pytest.mark.data
def test_private_siemens_dicom_equals_spec2nii(private_roots, tmp_path):
    series = _vendor_dicoms(private_roots, ("siemens", "uih"))
    assert _compare_all_vendor_dicoms(series, tmp_path) == len(series) > 0


#**************************************************************************************************#
#                                             Bruker                                               #
#**************************************************************************************************#
#                                                                                                  #
# Against spec2nii's read_bruker (brukerapi with spec2nii's property files), every spectroscopy    #
# fid, fid_proc.64, rawdata.job0 and complex 2dseq; a scan whose method file is renamed is given   #
# to spec2nii as a folder of links with the name it expects. Differences named: OriginalFile from  #
# the scan folder on (spec2nii: the path as given); rawdata transients of averages x repetitions   #
# DIM_DYN (spec2nii: DIM_USER_0); complex 2dseq spectra as FIDs (test_bruker checks them against   #
# the scan's fid), a single voxel's reconstruction in the method's voxel (spec2nii: identity) and  #
# its echo time from the method where visu_pars states 0. Magnitude reconstructions (MRImage) and  #
# files neither can read are passed over.                                                          #
#                                                                                                  #
#**************************************************************************************************#
BRUKER_KEYS = ("EchoTime", "RepetitionTime", "SpectrometerFrequency", "ResonantNucleus",
               "TxOffset", "SoftwareVersions", "PatientName", "Manufacturer", "SequenceName",
               "kSpace")
BRUKER_MODES = {"fid": "FID", "fid_proc.64": "FID", "2dseq": "2DSEQ", "rawdata.job0": "RAWDATA"}


def _bruker_files(roots: list[Path]) -> list[Path]:
    sniffer, files = Sniffer(), []
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.name in BRUKER_MODES and path.is_file() and ".git" not in path.parts:
                with contextlib.suppress(Exception):
                    if sniffer.sniff(path).format == Format.BRUKER:
                        files.append(path)
    return files


def _spec2nii_bruker(path: Path, tmp_path: Path) -> tuple[list, list]:
    reader = BrukerReader()
    scan = path.parents[2] if path.name in reader.PROCESSED else path.parent
    method = reader.method_file(scan)
    given = path
    if method is not None and method.name != "method":  # spec2nii reads the method by its name
        links = tmp_path / f"links_{abs(hash(str(scan)))}" / scan.name
        links.mkdir(parents=True)
        for entry in scan.iterdir():
            (links / entry.name).symlink_to(entry.resolve())
        (links / "method").symlink_to(method.resolve())
        given = links / path.relative_to(scan)
    args = argparse.Namespace(file=str(given), mode=BRUKER_MODES[path.name], query=[],
                              dump_headers=False, zeropad=False, nospectrum=False, fileout=None)
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        images, names = read_bruker(args)
    return [remove_zero_higher_dim_indices(i, remove=True)[0] for i in images], list(names)


def _compare_bruker(path: Path, tmp_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            got = miox.load(path)
        except ReaderError as e:
            pytest.skip(f"MIOX cannot read it: {e}")
    if not all(isinstance(d, miox.MRSData) for d in got):
        pytest.skip("magnitude reconstruction or imaging: not spec2nii's")
    try:
        expected, names = _spec2nii_bruker(path, tmp_path)
    except Exception as e:  # noqa: BLE001  any failure of spec2nii's own reading
        pytest.skip(f"spec2nii cannot convert it: {type(e).__name__}")
    assert [d.name for d in got] == names
    header = BrukerReader()._header(path)
    transients = header["method"].get("PVM_NAverages", 0) * header["acqp"].get("NR", 0)
    reconstruction = path.name == "2dseq"
    for d, image in zip(got, expected, strict=True):
        tags = [("DIM_DYN" if t and t.startswith("DIM_USER") and image.shape[4 + i] == transients
                 else t) for i, t in enumerate(image.dim_tags)]
        assert list(d.dim_tags) == tags, (d.name, d.dim_tags, tags)
        if not reconstruction:
            np.testing.assert_array_equal(d.data, np.asarray(image[:]).conj())  # [:] conjugates
        if not (reconstruction and d.data.shape[:3] == (1, 1, 1)):
            np.testing.assert_allclose(d.affine, image.getAffine("voxel", "world"), atol=1e-9)
        assert d.dwelltime == pytest.approx(image.dwelltime, abs=1e-15)
        header_ext = image.hdr_ext.to_dict()
        assert d.meta["OriginalFile"] == [str(path.relative_to(
            (path.parents[2] if path.name in BrukerReader.PROCESSED else path.parent).parent))]
        for key in BRUKER_KEYS:
            value, reference = d.meta.get(key), _stated(header_ext.get(key))
            if key == "EchoTime" and reconstruction and not reference:
                reference = header["method"]["PVM_EchoTime"] * 1e-3
            if isinstance(reference, float):
                assert value == pytest.approx(reference, abs=1e-12), key
            else:
                assert value == reference, key


def _compare_all_bruker(files: list[Path], tmp_path: Path) -> int:
    compared = 0
    for path in files:
        with contextlib.suppress(pytest.skip.Exception):
            _compare_bruker(path, tmp_path)
            compared += 1
    return compared


def test_bruker_equals_spec2nii(public_data, tmp_path):
    files = _bruker_files([public_data])
    if not files:
        pytest.skip("no Bruker data fetched")
    assert _compare_all_bruker(files, tmp_path) == 21  # every spectroscopy dataset fetched


@pytest.mark.data
def test_private_bruker_equals_spec2nii(private_roots, tmp_path):
    assert _compare_all_bruker(_bruker_files(private_roots), tmp_path) > 0


#**************************************************************************************************#
#                                              jMRUI                                               #
#**************************************************************************************************#
#                                                                                                  #
# Against spec2nii's jmrui_format, every jMRUI text and binary file: data, geometry, tags, dwell   #
# time and keys equal (a .mrui frequency stated in Hz in MHz, as spec2nii's command line leaves    #
# it for MIOX to correct). Basis sets are compared as the data they hold.                          #
#                                                                                                  #
#**************************************************************************************************#
JMRUI_KEYS = ("SpectrometerFrequency", "ResonantNucleus", "ManufacturersModelName", "PatientName",
              "TxOffset", "AdditionalInformation", "SignalNames", "OriginalFile")


def _jmrui_files(roots: list[Path]) -> list[Path]:
    sniffer, files = Sniffer(), []
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() in (".txt", ".mrui") and path.is_file() and \
                    not path.name.startswith(".") and ".git" not in path.parts:
                with contextlib.suppress(Exception):
                    if sniffer.sniff(path).format in (Format.JMRUI_TEXT, Format.JMRUI_BINARY):
                        files.append(path)
    return files


def _compare_jmrui(path: Path) -> None:
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            images, names = jmrui_format(argparse.Namespace(file=path, affine=None, fileout=None))
    except Exception as e:  # noqa: BLE001  any failure of spec2nii's own reading
        pytest.skip(f"spec2nii cannot convert it: {type(e).__name__}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = miox.load(path)
    assert [d.name for d in got] == list(names)
    for d, image in zip(got, images, strict=True):
        np.testing.assert_array_equal(d.data, np.asarray(image[:]).conj())  # [:] conjugates
        np.testing.assert_allclose(d.affine, image.getAffine("voxel", "world"), atol=1e-9)
        assert list(d.dim_tags) == list(image.dim_tags)
        assert d.dwelltime == pytest.approx(image.dwelltime, abs=1e-15)
        header_ext = image.hdr_ext.to_dict()
        for key in JMRUI_KEYS:
            value, reference = d.meta.get(key), _stated(header_ext.get(key))
            if key == "SpectrometerFrequency" and reference[0] > 1e5:
                reference = [reference[0] / 1e6]
            assert value == reference, key


def test_jmrui_equals_spec2nii(public_data):
    files = _jmrui_files([public_data])
    if not files:
        pytest.skip("no jMRUI file fetched")
    assert _compare_all(files, _compare_jmrui) == len(files)


#**************************************************************************************************#
#                                             Varian                                               #
#**************************************************************************************************#
#                                                                                                  #
# Against spec2nii's read_varian (tag DIM_DYN), imaging-console spectroscopy (im1D...): data,      #
# geometry, tags, dwell time and keys equal. Difference named: the echo time te in s, else the     #
# pulse plus alfa from µs in s (spec2nii: the pulse plus alfa, unscaled, over te).                 #
#                                                                                                  #
#**************************************************************************************************#
VARIAN_KEYS = ("SpectrometerFrequency", "ResonantNucleus", "RepetitionTime", "Manufacturer",
               "ProtocolName", "PatientName", "SoftwareVersions", "TxCoil", "RxCoil",
               "InversionTime", "ExcitationFlipAngle", "OriginalFile", "kSpace")


def test_varian_console_equals_spec2nii(public_data):
    folders = sorted(p for p in public_data.rglob("*.fid") if p.is_dir() and
                     (p / "procpar").is_file() and "im1D" in (p / "procpar").read_text(
                         errors="replace").split("apptype", 1)[-1][:60])
    if not folders:
        pytest.skip("no imaging-console Varian spectroscopy fetched")
    for folder in folders:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            images, names = read_varian(argparse.Namespace(file=str(folder), tag6="DIM_DYN",
                                                           dump_headers=False, fileout=None))
            got = miox.load(folder)
        assert [d.name for d in got] == list(names)
        for d, image in zip(got, images, strict=True):
            np.testing.assert_array_equal(d.data, np.asarray(image[:]).conj())  # [:] conjugates
            np.testing.assert_allclose(d.affine, image.getAffine("voxel", "world"), atol=1e-9)
            assert list(d.dim_tags) == list(image.dim_tags)
            assert d.dwelltime == pytest.approx(image.dwelltime, abs=1e-15)
            header_ext = image.hdr_ext.to_dict()
            for key in VARIAN_KEYS:
                value, reference = d.meta.get(key), _stated(header_ext.get(key))
                assert value == reference, key
            procpar = d.meta.vendor["procpar"]
            pulse = "p1" if "p1" in procpar else "pw"
            assert header_ext["EchoTime"] == pytest.approx(
                float(procpar[pulse][0]) + float(procpar["alfa"][0]))
            assert d.meta["EchoTime"] == pytest.approx(
                float(procpar["te"][0]) if "te" in procpar else header_ext["EchoTime"] * 1e-6)
