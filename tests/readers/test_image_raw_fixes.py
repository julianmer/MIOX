####################################################################################################
#                                      test_image_raw_fixes.py                                     #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 19/11/26                                                                                #
#                                                                                                  #
# Purpose: Unit tests of header variants of the MetaImage, NRRD, NIfTI (FSL diffusion tables),     #
#          ISMRMRD, fastMRI, DICOM and GE Signa readers, on files written here.                    #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import EnhancedMRImageStorage, ExplicitVRLittleEndian, generate_uid

# own
import miox
from miox.readers.base import ReaderError
from tests.conftest import PUBLIC_DATA
from tests.readers.test_dicom_images import _series

PHILIPS_CREATOR = "Philips MR Imaging DD 001"


# --- MetaImage ------------------------------------------------------------------------------------

def mhd(path: Path, *lines: str) -> None:
    header = ("ObjectType = Image", "NDims = 3", "DimSize = 2 3 2", *lines)
    path.write_text("\n".join(header) + "\n")


def test_metaimage_header_size_minus_one_takes_the_trailing_values(tmp_path):
    data = np.arange(12, dtype="<i2")
    (tmp_path / "vol.raw").write_bytes(b"\x07" * 10 + data.tobytes())
    mhd(tmp_path / "vol.mhd", "HeaderSize = -1", "ElementType = MET_SHORT",
        "ElementDataFile = vol.raw")
    [im] = miox.load(tmp_path / "vol.mhd")
    assert np.array_equal(im.data, data.reshape(2, 3, 2).T)


def test_metaimage_header_size_minus_one_per_listed_file(tmp_path):
    data = np.arange(12, dtype="<i2").reshape(2, 6)
    for name, values in zip(("a.raw", "b.raw"), data, strict=True):
        (tmp_path / name).write_bytes(b"\x07" * 4 + values.tobytes())
    mhd(tmp_path / "vol.mhd", "HeaderSize = -1", "ElementType = MET_SHORT",
        "ElementDataFile = LIST", "a.raw", "b.raw")
    [im] = miox.load(tmp_path / "vol.mhd")
    assert np.array_equal(im.data, data.reshape(2, 3, 2).T)


def test_metaimage_element_byte_order_msb(tmp_path):
    data = (np.arange(12) * 300).astype(">i2")
    (tmp_path / "vol.raw").write_bytes(data.tobytes())
    mhd(tmp_path / "vol.mhd", "ElementByteOrderMSB = True", "ElementType = MET_SHORT",
        "ElementDataFile = vol.raw")
    [im] = miox.load(tmp_path / "vol.mhd")
    assert np.array_equal(im.data, data.reshape(2, 3, 2).T)


@pytest.mark.parametrize("element, values", [("MET_SHORT", "0 -1 2 3 4 5\n6 7 8 9 10 300"),
                                             ("MET_FLOAT", "0 -1.5 2 3 4 5 6 7 8 9 1e1 300")])
def test_metaimage_ascii_data(tmp_path, element, values):
    expected = np.array([float(v) for v in values.split()])
    (tmp_path / "vol.txt").write_text(values + "\n")
    mhd(tmp_path / "vol.mhd", "BinaryData = False", f"ElementType = {element}",
        "ElementDataFile = vol.txt")
    mhd(tmp_path / "local.mha", "BinaryData = False", f"ElementType = {element}",
        "ElementDataFile = LOCAL", values)
    for name in ("vol.mhd", "local.mha"):
        [im] = miox.load(tmp_path / name)
        assert np.array_equal(im.data, expected.reshape(2, 3, 2).T)
        assert im.data.dtype.kind == ("i" if element == "MET_SHORT" else "f")


def test_metaimage_origin_and_rotation_name_the_offset_and_the_matrix(tmp_path):
    (tmp_path / "vol.raw").write_bytes(bytes(12))
    geometry = {"Offset": "10 20 30", "TransformMatrix": "0 1 0 1 0 0 0 0 1"}
    affines = []
    for name, (offset, matrix) in {"a": ("Offset", "TransformMatrix"), "b": ("Origin", "Rotation"),
                                   "c": ("Position", "Orientation")}.items():
        mhd(tmp_path / f"{name}.mhd", f"{offset} = {geometry['Offset']}",
            f"{matrix} = {geometry['TransformMatrix']}", "ElementSpacing = 2 3 4",
            "ElementType = MET_UCHAR", "ElementDataFile = vol.raw")
        [im] = miox.load(tmp_path / f"{name}.mhd")
        affines.append(im.affine)
    expected = np.array([[0, -3, 0, -10], [-2, 0, 0, -20], [0, 0, 4, 30], [0, 0, 0, 1]])
    for affine in affines:
        assert np.allclose(affine, expected)


# --- NIfTI with FSL diffusion tables --------------------------------------------------------------

def dwi(folder: Path, affine: np.ndarray) -> list[list[float]]:
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2, 2), np.int16), affine), folder / "dwi.nii")
    (folder / "dwi.bval").write_text("0 1000\n")
    (folder / "dwi.bvec").write_text("0 0.6\n0 0.8\n0 0\n")
    [im] = miox.load(folder / "dwi.nii")
    assert im.meta["DiffusionBValue"] == [0.0, 1000.0]
    assert im.meta.vendor["bvec"] == [[0.0, 0.6], [0.0, 0.8], [0.0, 0.0]]  # as the file has them
    return im.meta["DiffusionGradientOrientation"]


def test_fsl_bvec_x_is_negated_for_a_positive_determinant(tmp_path):
    assert dwi(tmp_path, np.eye(4)) == [[0.0, 0.0, 0.0], [-0.6, 0.8, 0.0]]


def test_fsl_bvec_along_the_voxel_axes_for_a_negative_determinant(tmp_path):
    """The radiological grid of the same volume (first axis flipped) gives the same direction."""
    assert dwi(tmp_path, np.diag([-1.0, 1.0, 1.0, 1.0])) == [[0.0, 0.0, 0.0], [-0.6, 0.8, 0.0]]


# --- NRRD -----------------------------------------------------------------------------------------

def test_nrrd_none_space_direction_marks_a_non_spatial_axis(tmp_path):
    values = np.arange(3 * 2 * 2 * 2, dtype="<i2")
    header = "\n".join([
        "NRRD0004", "type: short", "dimension: 4", "space: right-anterior-superior",
        "sizes: 3 2 2 2", "space directions: none (2,0,0) (0,3,0) (0,0,4)",
        "space origin: (1,2,3)", "endian: little", "encoding: raw", "", "",
    ]).encode()
    (tmp_path / "vec.nrrd").write_bytes(header + values.tobytes())
    [im] = miox.load(tmp_path / "vec.nrrd")
    assert im.shape == (2, 2, 2, 3) and im.dims[3:] == ("volume",)
    assert np.array_equal(im.data[1, 0, 0], [3, 4, 5])  # the first file axis is the trailing one
    expected = np.diag([2.0, 3.0, 4.0, 1.0])
    expected[:3, 3] = [1, 2, 3]
    assert np.allclose(im.affine, expected)


@pytest.mark.parametrize("space", ["left-posterior-superior-time", "LPST"])
def test_nrrd_space_with_time_is_the_spatial_space(tmp_path, space):
    values = np.arange(2 * 2 * 2 * 3, dtype="<i2")
    header = "\n".join([
        "NRRD0004", "type: short", "dimension: 4", f"space: {space}", "sizes: 2 2 2 3",
        "space directions: (2,0,0,0) (0,3,0,0) (0,0,4,0) (0,0,0,1.5)",
        "space origin: (10,20,30,0)", "endian: little", "encoding: raw", "", "",
    ]).encode()
    (tmp_path / "time.nrrd").write_bytes(header + values.tobytes())
    [im] = miox.load(tmp_path / "time.nrrd")
    assert im.shape == (2, 2, 2, 3)
    expected = np.diag([-2.0, -3.0, 4.0, 1.0])
    expected[:3, 3] = [-10, -20, 30]
    assert np.allclose(im.affine, expected)
    assert not any("not a patient space" in note for note in im.provenance.notes)


@pytest.mark.parametrize("gap", ["", "\n"])
def test_nrrd_listed_data_files_follow_the_field(tmp_path, gap):
    data = np.arange(2 * 3 * 2, dtype="<i2").reshape(2, 6)
    for name, values in zip(("a.raw", "b.raw"), data, strict=True):
        (tmp_path / name).write_bytes(values.tobytes())
    header = "\n".join([
        "NRRD0005", "type: short", "dimension: 3", "sizes: 2 3 2", "endian: little",
        "encoding: raw", "data file: LIST",
    ])
    (tmp_path / "vol.nhdr").write_text(header + "\n" + gap + "a.raw\nb.raw\n")
    [im] = miox.load(tmp_path / "vol.nhdr")
    assert np.array_equal(im.data, data.reshape(2, 3, 2).T)


# --- ISMRMRD header text and fastMRI --------------------------------------------------------------

XML = """<?xml version="1.0"?>
<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD">
  <subjectInformation><patientName>007</patientName><patientID>1E5</patientID>
    <patientWeight_kg>70</patientWeight_kg></subjectInformation>
  <measurementInformation><measurementID>42</measurementID>
    <initialSeriesNumber>7</initialSeriesNumber></measurementInformation>
  <acquisitionSystemInformation><systemModel>1.50</systemModel>
    <systemFieldStrength_T>1.50</systemFieldStrength_T>
    <deviceSerialNumber>45E12</deviceSerialNumber>
    <coilLabel><coilNumber>0</coilNumber><coilName>1</coilName></coilLabel>
  </acquisitionSystemInformation>
  <sequenceParameters><TR>500</TR></sequenceParameters>
  <userParameters>
    <userParameterLong><name>12</name><value>3</value></userParameterLong>
    <userParameterString><name>inf</name><value>1e3</value></userParameterString>
  </userParameters>
</ismrmrdHeader>
"""


def fastmri(path: Path, kspace: np.ndarray) -> None:
    with h5py.File(path, "w") as fh:
        fh.create_dataset("kspace", data=kspace)
        fh.create_dataset("ismrmrd_header", data=XML.encode())
        fh.create_dataset("reconstruction_esc", data=np.ones((2, 4, 5), dtype=np.float32))
        fh.attrs["acquisition"] = "CORPD_FBK"


def test_ismrmrd_text_elements_stay_as_written(tmp_path):
    fastmri(tmp_path / "file.h5", np.ones((2, 3, 4, 5), dtype=np.complex64))
    raw, _ = miox.load(tmp_path / "file.h5")
    meta = raw.meta
    assert meta["PatientID"] == "1E5" and meta["PatientName"] == "007"
    assert meta["DeviceSerialNumber"] == "45E12" and meta["ManufacturersModelName"] == "1.50"
    assert meta["FieldStrength"] == 1.5 and meta["PatientWeight"] == 70
    assert meta["SeriesNumber"] == 7 and meta["RepetitionTime"] == 0.5
    system = meta.vendor["acquisitionSystemInformation"]
    assert system["systemModel"] == "1.50" and system["deviceSerialNumber"] == "45E12"
    assert system["coilLabel"] == {"coilNumber": 0, "coilName": "1"}
    assert meta.vendor["subjectInformation"]["patientID"] == "1E5"
    assert meta.vendor["measurementInformation"]["measurementID"] == "42"
    parameters = meta.vendor["userParameters"]
    assert parameters["userParameterLong"] == {"name": "12", "value": 3}
    assert parameters["userParameterString"] == {"name": "inf", "value": "1e3"}


def test_fastmri_single_coil(tmp_path):
    kspace = np.ones((2, 4, 5), dtype=np.complex64)  # slices, frequency, phase
    kspace[1, :, 3] = np.arange(4) + 1j
    fastmri(tmp_path / "file.h5", kspace)
    raw, image = miox.load(tmp_path / "file.h5")
    assert isinstance(raw, miox.RawData) and raw.shape == (10, 1, 4)
    assert raw.data[8, 0].tolist() == [0 + 1j, 1 + 1j, 2 + 1j, 3 + 1j]
    assert raw.meta["NumberOfReceiveChannels"] == 1 and raw.meta["EncodedMatrixSize"] == [4, 5, 2]
    assert image.shape == (5, 4, 2) and image.meta["ImageType"] == ["ESC"]


# --- DICOM ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("stated, state", [("ON_RESONANCE", True), ("OFF_RESONANCE", True),
                                           ("NONE", False)])
def test_magnetization_transfer_terms(tmp_path, stated, state):
    _series(tmp_path, MagnetizationTransfer=stated)
    [im] = miox.load(tmp_path)
    assert im.meta["MTState"] is state


def enhanced(path: Path, pixel_measures: bool = True) -> np.ndarray:
    """A Philips enhanced MR file of two frames (2 rows by 3 columns) whose rescale slope is in
    the shared functional groups; returns the stored values (frames, rows, columns)."""
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = EnhancedMRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = EnhancedMRImageStorage, meta.MediaStorageSOPInstanceUID
    ds.Modality, ds.Manufacturer, ds.SeriesInstanceUID = "MR", "Philips", "1.2.3"
    ds.SeriesNumber, ds.InstanceNumber, ds.NumberOfFrames = 3, 1, 2
    ds.ImageType = ["ORIGINAL", "PRIMARY", "M", "FFE"]
    ds.Rows, ds.Columns = 2, 3
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    shared = Dataset()
    orientation, transformation = Dataset(), Dataset()
    orientation.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    transformation.RescaleSlope, transformation.RescaleIntercept = 2.0, 0.0
    shared.PlaneOrientationSequence = Sequence([orientation])
    shared.PixelValueTransformationSequence = Sequence([transformation])
    if pixel_measures:
        measures = Dataset()
        measures.PixelSpacing, measures.SliceThickness = [2.0, 3.0], 4.0
        shared.PixelMeasuresSequence = Sequence([measures])
    ds.SharedFunctionalGroupsSequence = Sequence([shared])
    frames = []
    for z in (5.0, 9.0):
        frame, position = Dataset(), Dataset()
        position.ImagePositionPatient = [10.0, 20.0, z]
        frame.PlanePositionSequence = Sequence([position])
        frames.append(frame)
    ds.PerFrameFunctionalGroupsSequence = Sequence(frames)
    pixels = np.arange(12, dtype=np.uint16).reshape(2, 2, 3) * 8
    ds.PixelData = pixels.tobytes()
    ds.save_as(path, enforce_file_format=True)
    return pixels


@pytest.mark.parametrize("place", ["dataset", "shared"])
def test_philips_enhanced_scale_slope_outside_the_frames(tmp_path, place):
    """MRScaleSlope stated once for all frames scales the pixels, as the keys say."""
    path = tmp_path / "enhanced.dcm"
    pixels = enhanced(path)
    ds = pydicom.dcmread(path)
    holder = ds
    if place == "shared":
        holder = Dataset()
        block = ds.SharedFunctionalGroupsSequence[0].private_block(0x2005, PHILIPS_CREATOR,
                                                                   create=True)
        block.add_new(0x0F, "SQ", Sequence([holder]))
    holder.private_block(0x2005, PHILIPS_CREATOR, create=True).add_new(0x0E, "FL", 4.0)
    ds.save_as(path)
    [im] = miox.load(path)
    assert im.meta["UsePhilipsFloatNotDisplayScaling"] == 1 and im.meta["PhilipsScaleSlope"] == 4.0
    assert np.allclose(im.data[:, :, 1], pixels[1].T / 4.0)  # (v * 2) / (2 * 4)


def test_enhanced_without_pixel_spacing_is_a_reader_error(tmp_path):
    enhanced(tmp_path / "enhanced.dcm", pixel_measures=False)
    with pytest.raises(ReaderError, match=r"enhanced\.dcm.*PixelSpacing"):
        miox.load(tmp_path / "enhanced.dcm")


# --- GE Signa -------------------------------------------------------------------------------------

@pytest.mark.data
@pytest.mark.parametrize("folder", ["test_ge5", "test_ge4"])
def test_ge_signa_reads_only_the_files_of_the_series(tmp_path, monkeypatch, folder):
    source = PUBLIC_DATA / "itk_ge" / folder
    if not source.is_dir():
        pytest.skip("files not fetched")
    names = sorted(f.name for f in source.iterdir())[:3:2]  # Genesis echoes alternate
    for name in names:
        shutil.copy(source / name, tmp_path / name)
    size = (tmp_path / names[0]).stat().st_size
    (tmp_path / "other_size.bin").write_bytes(bytes(3 * size))
    (tmp_path / "same_size.bin").write_bytes(bytes(size))
    whole = []
    read_bytes = Path.read_bytes
    monkeypatch.setattr(
        Path, "read_bytes", lambda self: whole.append(self.name) or read_bytes(self))
    [im] = miox.load(tmp_path / names[0])
    assert im.shape[2] == 2
    assert set(whole) <= set(names)
