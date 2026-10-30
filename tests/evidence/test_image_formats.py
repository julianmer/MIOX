####################################################################################################
#                                       test_image_formats.py                                      #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 04/10/26                                                                                #
#                                                                                                  #
# Purpose: Checks the NRRD, MetaImage, MRtrix, Varian fdf, GE Signa, VTK and DMIWizard map readers #
#          against evidence produced by other software: pynrrd's NIfTI twin of its NRRD test       #
#          image, MRtrix's own PNG exports and reductions, ImageJ text exports of fdf images,      #
#          ITK's MetaImage baselines of its GE test images, the value range a .vti file states,    #
#          and the DICOM copies DMIWizard writes of its text maps.                                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re
import struct
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pydicom
import pytest

from miox import MRImage, load
from miox.readers.base import ReaderError
from tests.conftest import PUBLIC_DATA

PYNRRD = "pynrrd/"
MRTRIX = "mrtrix_test_data/"
FDF = "agilent_fdf/"
GE = "itk_ge/"


def image(relpath: str) -> MRImage:
    path = PUBLIC_DATA / relpath
    if not path.exists():
        pytest.skip("file not fetched")
    [im] = load(path)
    return im


def images(relpath: str) -> list[MRImage]:
    path = PUBLIC_DATA / relpath
    if not path.exists():
        pytest.skip("file not fetched")
    return load(path)


def read_png(path: Path) -> np.ndarray:
    """An 8-bit greyscale, non-interlaced PNG as (rows, columns), decoded here so that the check
    does not rest on the image library it tests."""
    blob = path.read_bytes()
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat = 8, b""
    while pos < len(blob):
        n = int.from_bytes(blob[pos : pos + 4], "big")
        kind, data = blob[pos + 4 : pos + 8], blob[pos + 8 : pos + 8 + n]
        pos += 12 + n
        if kind == b"IHDR":
            width, height, depth, colour, _, _, interlace = struct.unpack(">IIBBBBB", data)
            assert (depth, colour, interlace) == (8, 0, 0)
        elif kind == b"IDAT":
            idat += data
    raw = zlib.decompress(idat)
    out = np.zeros((height, width), np.int32)
    prev = np.zeros(width, np.int32)
    for r in range(height):
        start = r * (width + 1)
        kind, line = raw[start], np.frombuffer(raw[start + 1 : start + 1 + width], np.uint8)
        cur = np.zeros(width, np.int32)
        for x in range(width):
            a, b, c = (cur[x - 1] if x else 0), prev[x], (prev[x - 1] if x else 0)
            if kind == 0:
                p = 0
            elif kind == 1:
                p = a
            elif kind == 2:
                p = b
            elif kind == 3:
                p = (a + b) // 2
            else:
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                p = a if pa <= pb and pa <= pc else b if pb <= pc else c
            cur[x] = (int(line[x]) + p) & 0xFF
        out[r], prev = cur, cur
    return out


# --- NRRD -----------------------------------------------------------------------------------------


@pytest.mark.data
@pytest.mark.parametrize(
    "name",
    ["BallBinary30x30x30.nrrd", "BallBinary30x30x30.nhdr", "BallBinary30x30x30_gz.nrrd",
     "BallBinary30x30x30_bz2.nrrd", "BallBinary30x30x30_gz_lineskip.nrrd",
     "BallBinary30x30x30_byteskip_minus_one.nhdr", "BallBinary30x30x30_byteskip_minus_five.nhdr",
     "BallBinary30x30x30_gz_byteskip_minus_one.nrrd",
     "BallBinary30x30x30_byteskip_minus_one_nifti.nhdr"],
)
def test_nrrd_variants_equal_the_nifti_twin(name):
    nrrd, nifti = image(f"{PYNRRD}{name}"), image(f"{PYNRRD}BallBinary30x30x30.nii.gz")
    assert nrrd.shape == nifti.shape == (30, 30, 30)
    assert np.array_equal(nrrd.data, nifti.data)
    assert np.allclose(nrrd.affine, nifti.affine)
    assert nrrd.meta["nrrd_space"] == "left-posterior-superior"


@pytest.mark.data
def test_nrrd_detached_nifti_without_byte_skip_is_refused():
    path = PUBLIC_DATA / f"{PYNRRD}BallBinary30x30x30_nifti.nhdr"
    if not path.exists():
        pytest.skip("file not fetched")
    with pytest.raises(ReaderError, match="values, expected"):
        load(path)  # the NIfTI header bytes are part of the gzip stream; the sizes do not match


@pytest.mark.data
def test_nrrd_ascii_and_4d():
    im = image(f"{PYNRRD}test2d_ascii.nrrd")
    assert im.shape == (3, 9, 1) and im.data.dtype == np.uint16 and im.data.sum() == 378
    assert im.voxel_sizes == pytest.approx((1.0458, 2.0, 1.0))
    im = image(f"{PYNRRD}test_simple4d_raw.nrrd")
    assert im.dims == ("i", "j", "k", "domain") and im.shape == (1, 1, 1, 1)


# --- MRtrix ---------------------------------------------------------------------------------------


@pytest.mark.data
@pytest.mark.parametrize("name,count", [("pngaxial", 5), ("pngmask", 9)])
def test_mrtrix_image_matches_its_png_export(name, count):
    """MRtrix writes PNGs with both in-plane axes reversed (strides -2, -3); pngmask.mif.gz is
    stored with negative strides itself, so this also checks the layout handling."""
    im = image(f"{MRTRIX}mrconvert/{name}.mif.gz")
    assert im.shape[2] == count
    for k in range(count):
        png = read_png(PUBLIC_DATA / f"{MRTRIX}mrconvert/{name}{k}.png")
        assert np.array_equal(png, im.data[::-1, ::-1, k].T)


@pytest.mark.data
def test_mrtrix_mean_over_volumes_matches_mrmath():
    dwi, mean = image(f"{MRTRIX}dwi.mif"), image(f"{MRTRIX}mrmath/out1.mif")
    assert dwi.shape == (6, 8, 9, 68) and dwi.dims[3] == "volume"
    assert np.allclose(dwi.data.mean(axis=3), mean.data, rtol=1e-5, atol=1e-3)
    assert np.allclose(dwi.affine, mean.affine)
    b = np.asarray(dwi.meta["DiffusionBValue"])
    directions = np.asarray(dwi.meta["DiffusionGradientOrientation"])
    assert b.shape == (68,) and directions.shape == (68, 3) and (b < 50).sum() == 8
    assert np.allclose(np.linalg.norm(directions[b > 50], axis=1), 1.0)
    assert np.all(directions[b < 50] == 0)


@pytest.mark.data
def test_mrtrix_selected_volumes_and_axes():
    dwi = image(f"{MRTRIX}dwi.mif")
    picked = image(f"{MRTRIX}mrconvert/dwi_select_axes.mif")  # -coord 3 1:2:end -axes 0:2,-1,3
    assert picked.shape == (6, 8, 9, 1, 34) and picked.dims == ("i", "j", "k", "volume", "volume")
    assert np.array_equal(picked.data[:, :, :, 0, :], dwi.data[:, :, :, 1::2])


# --- Varian fdf -----------------------------------------------------------------------------------


@pytest.mark.data
@pytest.mark.parametrize("name", ["0009", "0030"])
def test_fdf_pixels_equal_the_imagej_text_export(name):
    im = image(f"{FDF}MRI_FDF/{name}.fdf")
    text = PUBLIC_DATA / f"{FDF}MRI_TXTimages/{name}.txt"
    rows = np.loadtxt(text)  # ImageJ writes one row per line, rounded to three decimals
    assert im.shape == (128, 128, 1) and im.data.dtype == np.float32
    assert np.allclose(rows, im.data[:, :, 0].T, atol=6e-4, rtol=0)
    assert im.meta["SpectrometerFrequency"] == [399.508087]
    assert im.meta["ResonantNucleus"] == ["1H"] and im.meta["SequenceName"] == "mems"
    assert im.meta["EchoTime"] == 0.01 and im.meta["RepetitionTime"] == 10.0
    assert im.meta["SliceThickness"] == pytest.approx(1.0)
    assert im.voxel_sizes == pytest.approx((40.13 / 128, 40.0 / 128, 1.0), abs=1e-6)


# --- GE Signa -------------------------------------------------------------------------------------


def half_voxel(im: MRImage) -> np.ndarray:
    return 0.5 * (im.affine[:3, 0] + im.affine[:3, 1])


@pytest.mark.data
def test_genesis_series_against_itk_baseline():
    """ITK puts the origin at the top-left corner GE records; that corner is the edge of the field
    of view (the corners span the FOV exactly), so our origin lies half a voxel inside."""
    echoes = images(f"{GE}test_ge5")
    itk = image(f"{GE}baseline/113766.003.001.mha")
    assert [im.name for im in echoes] == ["exam6451_series3_echo1", "exam6451_series3_echo2"]
    first = echoes[0]
    assert np.array_equal(first.data, itk.data)
    assert np.allclose(first.affine[:3, :3], itk.affine[:3, :3])
    assert np.allclose(first.affine[:3, 3] - itk.affine[:3, 3], half_voxel(first))
    assert first.meta["EchoTime"] == 0.028 and echoes[1].meta["EchoTime"] == 0.084
    assert first.meta["RepetitionTime"] == 3.0 and first.meta["EchoTrainLength"] == 8
    assert first.meta["SpectrometerFrequency"] == [63.889634]  # 1.5 T
    assert first.meta["FieldStrength"] == 1.5 and first.meta["PatientPosition"] == "HFS"
    assert first.meta["PulseSequenceName"] == "fse" and first.meta["RxCoil"] == "HEAD"
    assert first.meta["SpacingBetweenSlices"] == 3.0 and first.meta["ExcitationFlipAngle"] == 90
    assert first.meta["AcquisitionDateTime"] == "1998-04-06T19:54:24"


@pytest.mark.data
def test_genesis_single_file_reads_its_series_and_echo():
    [im] = images(f"{GE}test_ge5/113766.003.002")
    assert im.name == "exam6451_series3_echo2" and im.shape == (4, 4, 3)


@pytest.mark.data
def test_signa4_series_against_itk_baseline():
    """ITK orders Signa 4.x slices by their location and leaves the origin at zero; our slice axis
    completes a right-handed set, so the data agree once the slice order is reversed."""
    [im] = images(f"{GE}test_ge4")
    itk = image(f"{GE}baseline/19771.002.001.mha")
    assert im.shape == itk.shape == (4, 4, 6)
    assert np.array_equal(im.data[:, :, ::-1], itk.data)
    assert np.allclose(im.affine[:3, :2], itk.affine[:3, :2])
    assert np.allclose(im.affine[:3, 2], -itk.affine[:3, 2])
    assert im.meta["RepetitionTime"] == 0.024 and im.meta["EchoTime"] == 0.005
    assert im.meta["ExcitationFlipAngle"] == 40 and im.meta["NumberOfAverages"] == 2.0
    assert im.meta["FieldStrength"] == 1.5 and im.meta["SliceThickness"] == 1.5
    assert im.meta["SpectrometerFrequency"] == pytest.approx([63.891457])
    assert im.meta["ImageOrientationText"] == "CORONAL"
    assert im.meta["AcquisitionDateTime"] == "1993-06-07T21:34:42"


@pytest.mark.data
@pytest.mark.parametrize("name", ["c_vf1122", "c_vf1210"])
def test_genesis_pixel_header_ct_against_itk_baseline(name):
    [im] = images(f"{GE}ct_genesis/{name}.fre")
    itk = image(f"{GE}baseline/{name}.fre.mha")
    assert im.shape == itk.shape == (512, 512, 2)
    assert np.array_equal(im.data, itk.data)
    assert np.allclose(im.affine[:3, :3], itk.affine[:3, :3])
    assert np.allclose(im.affine[:3, 3] - itk.affine[:3, 3], half_voxel(im))
    assert im.meta["Modality"] == "CT" and "RepetitionTime" not in im.meta


def test_vti_values_span_the_range_the_file_states():
    """head.vti (PyVista vtk-data) states RangeMin and RangeMax for its zlib-compressed array."""
    path = PUBLIC_DATA / "pyvista_vtk_data" / "Data" / "head.vti"
    if not path.exists():
        pytest.skip("pyvista vtk-data not fetched")
    stated = re.search(rb'RangeMin="([^"]+)" RangeMax="([^"]+)"', path.read_bytes()[:4096])
    image = load(path)[0]
    assert image.data.shape == (256, 256, 94)
    assert (image.data.min(), image.data.max()) == tuple(float(v) for v in stated.groups())


def test_dmiwizard_text_maps_against_their_dicom_copies(tmp_path):
    """DMIWizard writes each metabolic map as text and as DICOM (clipped at zero, divided by its
    largest magnitude, stored as 0-65535); its ROI masks are 0/1 on the zero-filled grid."""
    archive = PUBLIC_DATA / "dmiwizard" / "DMIWizard_Data.zip"
    if not archive.exists():
        pytest.skip("DMIWizard not fetched")
    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.namelist() if "/maps/" in m and not m.endswith("/")]
        zf.extractall(tmp_path, members)
    maps = sorted(tmp_path.rglob("MetabolicMap_*"))
    texts = [m for m in maps if m.suffix != ".dcm"]
    assert texts
    for text in texts:
        [im] = load(text)
        assert im.provenance.format == "dmiwizard_map" and im.meta.vendor["map"] == "MetabolicMap"
        copy = next(m for m in maps if m.name == f"{text.name}.dcm")
        stored = pydicom.dcmread(copy).pixel_array.astype(float)
        values = im.data[:, :, 0]
        assert values.shape == stored.shape
        expected = np.round(np.clip(values, 0, None) / np.abs(values).max() * 65535)
        assert np.abs(expected - stored).max() <= 1
    for mask in tmp_path.rglob("ROIMap_*"):
        [im] = load(mask)
        rows, columns = (int(n) for n in re.match(r"ROIMap_(\d+)x(\d+)_", mask.name).groups())
        assert im.shape == (rows, columns, 1) and set(np.unique(im.data)) <= {0.0, 1.0}
