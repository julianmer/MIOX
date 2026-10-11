####################################################################################################
#                                       test_image_formats.py                                      #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 04/10/26                                                                                #
#                                                                                                  #
# Purpose: Unit tests of the NRRD, MetaImage, MRtrix, Varian fdf and GE Signa readers on files     #
#          written here: layouts, detached data, world coordinates and the keys derived.           #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import gzip

import numpy as np
import pytest

import miox
from miox.readers.ge_signa import GeHeader


def test_nrrd_detached_lps_with_diffusion_keys(tmp_path):
    data = np.arange(2 * 3 * 4, dtype="<i2").reshape(4, 3, 2)  # written k, j, i: i fastest
    (tmp_path / "vol.raw").write_bytes(data.tobytes())
    header = "\n".join([
        "NRRD0005", "type: short", "dimension: 3", "space: left-posterior-superior",
        "sizes: 2 3 4", "space directions: (2,0,0) (0,3,0) (0,0,4)", "space origin: (10,20,30)",
        "kinds: domain domain domain", "endian: little", "encoding: raw", "data file: vol.raw",
        "DWMRI_b-value:=1000", "DWMRI_gradient_0000:=0 0 0", "DWMRI_gradient_0001:=0.5 0 0",
        "", "",
    ])
    (tmp_path / "vol.nhdr").write_text(header)
    [im] = miox.load(tmp_path / "vol.nhdr")
    assert im.shape == (2, 3, 4) and np.array_equal(im.data, data.T)
    expected = np.diag([-2.0, -3.0, 4.0, 1.0])
    expected[:3, 3] = [-10, -20, 30]
    assert np.allclose(im.affine, expected)
    assert im.meta["DiffusionBValue"] == [0.0, 250.0]  # scaled by the squared gradient length
    assert im.meta["DiffusionGradientOrientation"] == [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]


def test_nrrd_gzip_attached_with_vector_axis(tmp_path):
    values = np.arange(3 * 2 * 2 * 1, dtype="<f4")
    header = "\n".join([
        "NRRD0004", "type: float", "dimension: 4", "sizes: 3 2 2 1",
        "kinds: vector domain domain domain", "spacings: nan 1.5 1.5 2", "encoding: gzip", "", "",
    ]).encode()
    (tmp_path / "vec.nrrd").write_bytes(header + gzip.compress(values.tobytes()))
    [im] = miox.load(tmp_path / "vec.nrrd")
    assert im.shape == (2, 2, 1, 3) and im.dims == ("i", "j", "k", "vector")
    assert im.voxel_sizes == pytest.approx((1.5, 1.5, 2.0))
    assert im.data[1, 0, 0, 2] == values.reshape(1, 2, 2, 3)[0, 0, 1, 2]


def test_metaimage_local_and_listed_files(tmp_path):
    data = np.arange(2 * 3 * 2, dtype="<u2").reshape(2, 3, 2)  # z, y, x
    header = "\n".join([
        "ObjectType = Image", "NDims = 3", "BinaryData = True", "BinaryDataByteOrderMSB = False",
        "TransformMatrix = 0 1 0 -1 0 0 0 0 1", "Offset = 1 2 3", "ElementSpacing = 1 2 3",
        "DimSize = 2 3 2", "ElementType = MET_USHORT", "ElementDataFile = LOCAL", "",
    ]).encode()
    (tmp_path / "im.mha").write_bytes(header + data.tobytes())
    [im] = miox.load(tmp_path / "im.mha")
    assert im.shape == (2, 3, 2) and np.array_equal(im.data, data.T)
    lps = np.array([[0, -2, 0, 1], [1, 0, 0, 2], [0, 0, 3, 3], [0, 0, 0, 1]], float)
    ras = lps * np.array([-1, -1, 1, 1])[:, np.newaxis]
    assert np.allclose(im.affine, ras)
    (tmp_path / "slice0.raw").write_bytes(data[0].tobytes())
    (tmp_path / "slice1.raw").write_bytes(data[1].tobytes())
    listed = header.decode().replace("ElementDataFile = LOCAL", "ElementDataFile = LIST")
    (tmp_path / "im.mhd").write_text(listed + "slice0.raw\nslice1.raw\n")
    [im2] = miox.load(tmp_path / "im.mhd")
    assert np.array_equal(im2.data, im.data)


def test_mrtrix_layout_and_gradient_table(tmp_path):
    shape = (2, 3, 4)
    volume = np.arange(np.prod(shape), dtype="<f4").reshape(shape)  # indexed i, j, k
    # layout -1,+2,+0: axis 2 has the shortest stride, axis 0 the next (traversed in reverse),
    # axis 1 the longest; so the file holds axis 1 slowest, then reversed axis 0, then axis 2
    stored = np.ascontiguousarray(np.transpose(volume[::-1, :, :], (1, 0, 2)))
    header = "\n".join([
        "mrtrix image", "dim: 2,3,4", "vox: 1,2,3", "layout: -1,+2,+0", "datatype: Float32LE",
        "transform: 1,0,0,5", "transform: 0,1,0,6", "transform: 0,0,1,7",
        "dw_scheme: 0,0,0,0", "comments: made for the test", "file: . 1024", "END", "",
    ]).encode()
    (tmp_path / "im.mif").write_bytes(header.ljust(1024, b"\n") + stored.tobytes())
    [im] = miox.load(tmp_path / "im.mif")
    assert im.shape == shape and np.array_equal(im.data, volume)
    expected = np.array([[1, 0, 0, 5], [0, 2, 0, 6], [0, 0, 3, 7], [0, 0, 0, 1]], float)
    assert np.allclose(im.affine, expected)
    assert im.meta["DiffusionBValue"] == [0.0]
    assert im.meta["mrtrix_comments"] == "made for the test"


def fdf_file(path, slice_no: int, z_cm: float) -> None:
    data = np.full((3, 2), slice_no, "<f4")  # phase x read: read fastest
    header = "\n".join([
        "#!/usr/local/fdf/startup", "float  rank = 2;", 'char  *spatial_rank = "2dfov";',
        'char  *storage = "float";', "float  bits = 32;", 'char  *nucleus[] = {"H1"};',
        "float  nucfreq[] = {400.0};", 'char  *sequence = "gems";', "float  matrix[] = {2, 3};",
        "float  span[] = {2.000000, 6.000000};", "float  roi[] = {2.000000,6.000000,0.200000};",
        f"float  location[] = {{0.000000,0.000000,{z_cm:.6f}}};", "float  gap = 0.100000;",
        f"int    slice_no = {slice_no};", "int    slices = 3;", "int    echo_no = 1;",
        "int    echoes = 1;", "int    array_index = 1;",
        "float  orientation[] = {1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0};", "float  te = 0.005;",
        "float  tr = 0.5;", 'char  *position1 = "head first";', 'char  *position2 = "supine";',
        "int bigendian = 0;", "", "\x0c", "",
    ]).encode()
    path.write_bytes(header + b"\0" + data.tobytes())


def test_fdf_folder_assembles_slices(tmp_path):
    folder = tmp_path / "gems_01.img"
    folder.mkdir()
    for n, z in ((1, -0.3), (2, 0.0), (3, 0.3)):
        fdf_file(folder / f"slice00{n}image001echo001.fdf", n, z)
    assert miox.sniff(folder).format == miox.Format.VARIAN_FDF
    [im] = miox.load(folder)
    assert im.shape == (2, 3, 3) and im.dims == ("i", "j", "k")
    assert np.all(im.data[:, :, 0] == 1) and np.all(im.data[:, :, 2] == 3)
    # 10 mm read FOV over 2 voxels, 60 mm phase over 3, slices 3 mm apart at -3, 0, 3 mm
    assert im.voxel_sizes == pytest.approx((10.0, 20.0, 3.0))
    # magnet X maps to -R: the first voxel centre sits at +5 mm X, i.e. R = -5
    assert np.allclose(im.affine[:3, 3], [5.0, -20.0, -3.0])
    assert im.affine[0, 0] == pytest.approx(-10.0) and im.affine[1, 1] == pytest.approx(20.0)
    assert im.meta["PatientPosition"] == "HFS" and im.meta["ResonantNucleus"] == ["1H"]
    assert im.meta["SliceThickness"] == pytest.approx(2.0)
    assert im.meta["SpacingBetweenSlices"] == pytest.approx(3.0)
    [single] = miox.load(folder / "slice002image001echo001.fdf")
    assert single.shape == (2, 3, 1) and np.all(single.data == 2)


@pytest.mark.parametrize(
    "word,value", [(0x41100000, 1.0), (0x41180000, 1.5), (0x43104000, 260.0), (0, 0.0),
                   (0xC1100000, -1.0)]
)
def test_data_general_float(word, value):
    assert GeHeader.data_general_float(word) == value
