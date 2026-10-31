####################################################################################################
#                                       test_image_formats.py                                      #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 04/10/26                                                                                #
#                                                                                                  #
# Purpose: Unit tests of the NRRD, MetaImage, MRtrix, Varian fdf, GE Signa, JNIfTI, VTK and GIPL   #
#          image readers on files written here: layouts, encodings, detached data, world           #
#          coordinates and the keys derived.                                                       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import base64
import gzip
import struct
import zlib

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


def bjdata_key(name: str) -> bytes:
    return b"U" + bytes([len(name)]) + name.encode()


def test_jnifti_binary_typed_array_and_qform(tmp_path):
    """A BJData file written by hand: an object with a NIFTIHeader (qform with a 90-degree turn
    about z) and NIFTIData as an N-D typed int16 array (little-endian, row-major)."""
    voxels = np.arange(2 * 3 * 4, dtype="<i2").reshape(2, 3, 4)
    header = (b"{" + bjdata_key("QForm") + b"S" + bjdata_key("scanner_anat")
              + bjdata_key("VoxelSize") + b"[$d#U\x03" + np.array([2, 2, 3], "<f4").tobytes()
              + bjdata_key("Quatern") + b"{" + bjdata_key("b") + b"d" + np.float32(0).tobytes()
              + bjdata_key("c") + b"d" + np.float32(0).tobytes()
              + bjdata_key("d") + b"d" + np.float32(np.sqrt(0.5)).tobytes() + b"}" + b"}")
    data = (b"{" + bjdata_key("_ArrayType_") + b"S" + bjdata_key("int16")
            + bjdata_key("_ArraySize_") + b"[$U#U\x03\x02\x03\x04"
            + bjdata_key("_ArrayData_") + b"[$I#[$U#U\x03\x02\x03\x04" + voxels.tobytes() + b"}")
    path = tmp_path / "small.bnii"
    path.write_bytes(b"{" + bjdata_key("NIFTIHeader") + header + bjdata_key("NIFTIData") + data
                     + b"}")
    assert miox.sniff(path).format == miox.Format.JNIFTI_BINARY
    image = miox.load(path)[0]
    assert np.array_equal(image.data, voxels)
    assert np.allclose(image.affine[:3, :3], [[0, -2, 0], [2, 0, 0], [0, 0, 3]], atol=1e-6)


VTK_VOXELS = np.arange(2 * 3 * 4, dtype="<u2").reshape(4, 3, 2)  # (z, y, x): x fastest on disk


@pytest.mark.parametrize("encoding", ["ASCII", "BINARY"])
def test_vtk_legacy_structured_points(tmp_path, encoding):
    head = (f"# vtk DataFile Version 3.0\ntest\n{encoding}\nDATASET STRUCTURED_POINTS\n"
            "DIMENSIONS 2 3 4\nORIGIN 1 2 3\nSPACING 0.5 1 2\nPOINT_DATA 24\n"
            "SCALARS v unsigned_short 1\nLOOKUP_TABLE default\n").encode()
    body = (" ".join(map(str, VTK_VOXELS.ravel())).encode() if encoding == "ASCII"
            else VTK_VOXELS.astype(">u2").tobytes())
    path = tmp_path / "grid.vtk"
    path.write_bytes(head + body + b"\n")
    image = miox.load(path)[0]
    assert image.data.shape == (2, 3, 4) and np.array_equal(image.data, VTK_VOXELS.transpose())
    assert np.allclose(image.affine, [[0.5, 0, 0, 1], [0, 1, 0, 2], [0, 0, 2, 3], [0, 0, 0, 1]])


def vti(array: str, appended: bytes = b"") -> bytes:
    return (b'<?xml version="1.0"?>\n<VTKFile type="ImageData" version="1.0" '
            b'byte_order="LittleEndian" header_type="UInt32"'
            + (b' compressor="vtkZLibDataCompressor"' if b"zlib" in array.encode() else b"")
            + b'>\n<ImageData WholeExtent="1 2 0 2 0 3" Origin="0 0 0" Spacing="1 1 1" '
            b'Direction="0 -1 0 1 0 0 0 0 1">\n<Piece Extent="1 2 0 2 0 3">\n'
            b'<PointData Scalars="v">\n' + array.replace("zlib", "").encode()
            + b"\n</PointData>\n</Piece>\n</ImageData>\n" + appended + b"</VTKFile>\n")


@pytest.mark.parametrize("form", ["ascii", "binary", "zlib", "appended"])
def test_vtk_xml_image_data(tmp_path, form):
    raw = VTK_VOXELS.tobytes()
    size = np.uint32(len(raw)).tobytes()
    tag = '<DataArray type="UInt16" Name="v" format="{}"{}>{}</DataArray>'
    appended = b""
    if form == "ascii":
        array = tag.format("ascii", "", " ".join(map(str, VTK_VOXELS.ravel())))
    elif form == "binary":
        array = tag.format("binary", "", base64.b64encode(size).decode()
                           + base64.b64encode(raw).decode())
    elif form == "zlib":
        block = zlib.compress(raw)
        header = np.array([1, len(raw), len(raw), len(block)], "<u4").tobytes()
        array = "zlib" + tag.format("binary", "", base64.b64encode(header).decode()
                                    + base64.b64encode(block).decode())
    else:
        array = tag.format("appended", ' offset="0"', "")
        appended = b'<AppendedData encoding="raw">\n_' + size + raw + b"\n</AppendedData>\n"
    path = tmp_path / "grid.vti"
    path.write_bytes(vti(array, appended))
    assert miox.sniff(path).format == miox.Format.VTK_IMAGE_XML
    image = miox.load(path)[0]
    assert np.array_equal(image.data, VTK_VOXELS.transpose())
    # Direction turns x into y: voxel (0, 0, 0) at the extent start (1, 0, 0) lies at (0, 1, 0)
    assert np.allclose(image.affine[:3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert np.allclose(image.affine[:3, 3], [0, 1, 0])


@pytest.mark.parametrize("order", ["<", ">"])
def test_gipl_little_and_big_endian(tmp_path, order):
    header = bytearray(256)
    struct.pack_into(f"{order}4H", header, 0, 3, 2, 1, 1)
    struct.pack_into(f"{order}H", header, 8, 64)  # float
    struct.pack_into(f"{order}4f", header, 10, 0.5, 2.0, 3.0, 1.0)
    struct.pack_into(f"{order}4d", header, 204, 10.0, 20.0, 30.0, 0.0)
    struct.pack_into(f"{order}I", header, 252, 0xEFFFE9B0)
    values = np.arange(6, dtype=f"{order}f4")
    f = tmp_path / "test.gipl.gz"
    f.write_bytes(gzip.compress(bytes(header) + values.tobytes()))
    [image] = miox.load(f)
    assert image.shape == (3, 2, 1) and image.data[2, 1, 0] == 5 and image.data[1, 0, 0] == 1
    assert np.allclose(np.diag(image.affine)[:3], [-0.5, -2.0, 3.0])  # LPS axes as RAS
    assert np.allclose(image.affine[:3, 3], [-10, -20, 30])
