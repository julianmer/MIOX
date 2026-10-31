####################################################################################################
#                                          test_images.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Images: the same MINC data as MINC 1 and MINC 2 read alike; PAR/REC spacing equals the  #
#          voxel size of the affine; a dcm2niix sidecar's values land in the standard keys;        #
#          ParaVision 6.0.1 reconstructions as brkraw converts them; legacy MR DICOM and ACR-NEMA  #
#          files against GDCM's checksums of their decoded pixels; Hitachi DICOM as dcm2niix       #
#          converts it; bruker2nifti's banana study as bruker2nifti converts it.                   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pydicom
import pytest

# own
from miox import MRImage, load
from tests.conftest import PUBLIC_DATA

NIBABEL = "nibabel/nibabel/tests/data/"


def image(relpath: str) -> MRImage:
    path = PUBLIC_DATA / relpath
    if not path.exists():
        pytest.skip("file not fetched")
    [im] = load(path)
    return im


@pytest.mark.data
@pytest.mark.parametrize("name", ["_4d", "_1_scale", "-no-att"])
def test_minc1_and_minc2_agree(name):
    a, b = image(f"{NIBABEL}minc1{name}.mnc"), image(f"{NIBABEL}minc2{name}.mnc")
    assert a.shape == b.shape and a.dims == b.dims
    assert np.allclose(a.data, b.data) and np.allclose(a.affine, b.affine)


@pytest.mark.data
def test_minc_time_axis_last():
    im = image(f"{NIBABEL}minc1_4d.mnc")
    assert im.dims == ("i", "j", "k", "time") and im.shape[3] == 2


@pytest.mark.data
def test_parrec_slice_spacing_is_the_affine_voxel_size():
    im = image(f"{NIBABEL}phantom_EPI_asc_CLEAR_2_1.PAR")
    assert im.voxel_sizes[2] == pytest.approx(im.meta["SpacingBetweenSlices"])
    assert im.meta["PatientPosition"] == "HFS" and im.meta["RepetitionTime"] == 2.0


@pytest.mark.data
def test_dcm2niix_sidecar_values():
    im = image("dcm_qa_nih/Ref/DCM2NIIX_regression_test_4.nii")
    sidecar = PUBLIC_DATA / "dcm_qa_nih/Ref/DCM2NIIX_regression_test_4.json"
    sidecar = json.loads(sidecar.read_text())
    assert im.meta["EchoTime"] == sidecar["EchoTime"]
    assert im.meta["ExcitationFlipAngle"] == sidecar["FlipAngle"]
    assert im.meta["SpectrometerFrequency"] == [sidecar["ImagingFrequency"]]


BRKRAW = PUBLIC_DATA / "brkraw_dataset/PV6.0.1/UNC_PV6.0.1_FLASH_TurboRARE_EPI"
# (scan, slice pack, voxel sum, affine rows) as brkraw 0.5.7 converts pdata/1 ("scanner" space,
# x negated: brkraw's scanner frame has the opposite x to MIOX's RAS)
BRKRAW_SCANS = [
    ("3", 0, 79404.7, [
        [-0.1172, 0, 0, 15],
        [0, -0.1172, 0, 10],
        [0, 0, 4, 0]]),
    ("3", 1, 113432.09, [
        [0, 0, 4, 0],
        [-0.1172, 0, 0, 10],
        [0, -0.1172, 0, 15]]),
    ("3", 2, 103749.79, [
        [-0.1172, 0, 0, 15],
        [0, 0, -4, -5],
        [0, -0.1172, 0, 15]]),
    ("6", 0, 3178187.38, [
        [-0.075, 0, 0, 14.4],
        [0, -0.075, 0, 9.4],
        [0, 0, 1, -5.5]]),
    ("7", 0, 9874585.13, [
        [-0.075, 0, 0, 14.4],
        [0, -0.075, 0, 9.4],
        [0, 0, 1, -5.5]]),
    ("10", 0, 50345.87, [
        [0, 0, 0.9062, -28.5469],
        [-0.9062, 0, 0, 29],
        [0, -0.9062, 0, 29]]),
    ("11", 0, 709514.64, [
        [-0.36, 0, 0, 14.4],
        [0, -0.36, 0, 9.4],
        [0, 0, 1, -5.5]]),
]


@pytest.mark.data
@pytest.mark.parametrize("scan,pack,total,rows", BRKRAW_SCANS)
def test_paravision_reconstruction_as_brkraw_converts_it(scan, pack, total, rows):
    """The same voxels and geometry as brkraw's NIfTI, read from the scan folder (no raw data in
    this pruned study). A single-slice pack has no slice spacing: brkraw gives it 4 mm, MIOX the
    stated VisuCoreFrameThickness (2 mm), so there only the slice direction is compared."""
    folder = BRKRAW / "UNC_PV6.0.1_FLASH_TurboRARE_EPI" / scan
    if not folder.is_dir():
        pytest.skip("brkraw-dataset not fetched")
    image = load(folder)[pack]
    assert float(image.data.astype(float).sum()) == pytest.approx(total, rel=1e-6)
    expected, affine = np.asarray(rows), image.affine[:3]
    columns = [0, 1, 3] if image.shape[2] == 1 else [0, 1, 2, 3]
    assert np.allclose(affine[:, columns], expected[:, columns], atol=1e-3)
    unit = expected[:, 2] / np.linalg.norm(expected[:, 2])
    assert np.allclose(affine[:, 2] / np.linalg.norm(affine[:, 2]), unit, atol=1e-6)


GDCMDATA = PUBLIC_DATA / "gdcmdata"


def _gdcm_checksums() -> dict[str, str]:
    path = GDCMDATA / "gdcmMD5DataImages.cxx"
    if not path.exists():
        return {}
    pairs = re.findall(r'^\s*\{\s*"([0-9a-f]{32})"\s*,\s*"([^"]+)"\s*\}', path.read_text(), re.M)
    return {name: checksum for checksum, name in pairs}


@pytest.mark.data
@pytest.mark.parametrize("path", sorted(GDCMDATA.glob("MR*")) + sorted(GDCMDATA.glob("*ACR*"))
                         + sorted(GDCMDATA.glob("gdcm-*")) + sorted(GDCMDATA.glob("PICKER*Pre*")),
                         ids=lambda p: p.name)
def test_legacy_dicom_pixels_as_gdcm_decodes_them(path):
    """GDCM keeps the MD5 of every test image's decoded pixel buffer: rows by columns, frame
    after frame, in the stored integer type. MIOX's voxels, laid out that way, give the same
    checksum (ACR-NEMA 1 and 2, files without preamble, 12-bit packed pixels, a cine)."""
    checksums = _gdcm_checksums()
    if path.name not in checksums or path.name.startswith("MR_Spectroscopy"):
        pytest.skip("no checksum (or not an image)")
    [image] = load(path)
    ds = pydicom.dcmread(path, force=True, stop_before_pixels=True)
    bits = 16 if int(ds.get("BitsAllocated", 16)) == 12 else int(ds.get("BitsAllocated", 16))
    signed = int(ds.get("PixelRepresentation", 0) or 0)
    dtype = np.dtype(f"{'i' if signed else 'u'}{bits // 8}").newbyteorder("<")
    data = image.data.reshape(image.shape[:3] + (-1,))
    frames = [data[:, :, k, t].T for t in range(data.shape[3]) for k in range(data.shape[2])]
    assert hashlib.md5(np.stack(frames).astype(dtype).tobytes()).hexdigest() == checksums[path.name]


# Hitachi series as dcm2niix v1.0.20261029 converts them: its affine (rows 1-3) and the MD5 of its
# voxel array (int16, as nibabel loads it)
HITACHI = {
    "tcia_hitachi_oasis": ((192, 192, 27), "746f8e0e79960e583121b9e5d5847b9e", [
        [-1.2473, -0.0057, 0.3904, 126.1281],
        [0.0125, 1.219, 1.3258, -138.6243],
        [0.0806, -0.2764, 5.8386, -42.3042]]),
    "tcia_hitachi_echelon": ((256, 256, 12), "41e8077b8d2b71c92fdc4e7920da0efe", [
        [0.0005, 0.0107, 4.9998, -18.6614],
        [-1.1719, 0.0, 0.0023, 109.3782],
        [0.0, 1.1718, -0.0455, -148.2396]]),
}


@pytest.mark.data
@pytest.mark.parametrize("folder", list(HITACHI))
def test_hitachi_series_as_dcm2niix_converts_it(folder):
    """dcm2niix orders the voxels its own way; every voxel read here, placed by its world
    position on dcm2niix's grid, rebuilds dcm2niix's array exactly."""
    path = PUBLIC_DATA / folder
    if not path.is_dir():
        pytest.skip("TCIA Hitachi series not fetched")
    shape, checksum, rows = HITACHI[folder]
    [image] = load(path)
    theirs = np.vstack([rows, [0, 0, 0, 1]])
    ijk = np.indices(image.shape[:3]).reshape(3, -1)
    position = np.linalg.inv(theirs) @ np.vstack([image.affine[:3] @ np.vstack([ijk, np.ones(
        ijk.shape[1])]), np.ones(ijk.shape[1])])
    index = np.round(position[:3]).astype(int)
    assert np.abs(position[:3] - index).max() < 0.01  # dcm2niix's affine is rounded here
    rebuilt = np.zeros(shape, "<i2")
    rebuilt[tuple(index)] = image.data.reshape(-1)
    assert hashlib.md5(rebuilt.tobytes()).hexdigest() == checksum


BANANA = PUBLIC_DATA / "bruker2nifti/test_data/bru_banana"


@pytest.mark.data
@pytest.mark.parametrize("scan", ["1", "2", "3"])
def test_banana_as_bruker2nifti_converts_it(scan):
    """bruker2nifti (slope corrected) gives the same voxels for all three scans (one scan with
    its orientation edited): sum 935878540.509, sum of squares 9.882199641335e13. Scans 2 and 3
    state sagittal and coronal orientations but axial slice positions; the slice normal is then
    followed, so the affine stays a proper one."""
    if not BANANA.is_dir():
        pytest.skip("bruker2nifti test data not fetched")
    [image] = load(BANANA / scan)
    data = image.data.astype(float)
    assert data.sum() == pytest.approx(935878540.5088512, rel=1e-9)
    assert (data ** 2).sum() == pytest.approx(98821996413349.98, rel=1e-7)  # float32 voxels
    contradiction = any("slice normal" in note for note in image.provenance.notes)
    assert contradiction == (scan != "1")
    assert abs(np.linalg.det(image.affine[:3, :3])) > 0.5
