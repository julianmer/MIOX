####################################################################################################
#                                       test_dicom_images.py                                       #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: DICOM image reader on synthetic series: slice and volume order, the RAS affine,         #
#          rescaling, Philips floating-point scaling, echoes split into images, study folders      #
#          with and without a DICOMDIR. No test data needed.                                       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.fileset import FileSet
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

# own
import miox as gu


def _slice(folder, number, z, volume, pixels, series="1.2.3", echo=30.0, **extra):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = MRImageStorage, meta.MediaStorageSOPInstanceUID
    ds.Modality, ds.Manufacturer, ds.SeriesInstanceUID = "MR", "ACME", series
    ds.SeriesNumber, ds.ProtocolName, ds.InstanceNumber = 3, "t1", number
    ds.AcquisitionNumber = volume
    ds.ImageType = ["ORIGINAL", "PRIMARY", "M", "ND"]
    ds.ImagePositionPatient = [10.0, 20.0, z]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.PixelSpacing = [2.0, 3.0]  # row spacing, column spacing
    ds.SliceThickness, ds.EchoTime, ds.RepetitionTime, ds.FlipAngle = 4.0, echo, 2000.0, 9.0
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    for key, value in extra.items():
        setattr(ds, key, value)
    ds.PixelData = pixels.astype(np.uint16).tobytes()
    path = folder / f"{number:03d}_{echo:.0f}.dcm"
    ds.save_as(path, enforce_file_format=True)
    return path


def _series(folder, echo=30.0, **extra):
    """Two slices (z = 5, 9 mm, written in reverse) by two volumes, 2 rows by 3 columns."""
    values = {}
    number = 0
    for volume in (1, 2):
        for z in (9.0, 5.0):
            number += 1
            pixels = np.arange(6).reshape(2, 3) + 100 * z + 1000 * volume
            values[(z, volume)] = pixels
            _slice(folder, number, z, volume, pixels, echo=echo, **extra)
    return values


def test_classic_series(tmp_path):
    values = _series(tmp_path)
    [im] = gu.load(tmp_path)
    assert im.shape == (3, 2, 2, 2) and im.dims == ("i", "j", "k", "volume")
    assert np.array_equal(im.data[:, :, 0, 1], values[(5.0, 2)].T)  # slices ascending along z
    expected = np.array([[-3, 0, 0, -10], [0, -2, 0, -20], [0, 0, 4, 5], [0, 0, 0, 1]])
    assert np.allclose(im.affine, expected)  # columns by 3 mm, rows by 2 mm, LPS to RAS
    assert im.meta["EchoTime"] == pytest.approx(0.03) and im.meta["RepetitionTime"] == 2.0
    assert im.meta["ExcitationFlipAngle"] == 9.0 and im.name == "3"


def test_rescaling_and_philips_floating_point(tmp_path):
    (tmp_path / "plain").mkdir()
    (tmp_path / "philips").mkdir()
    values = _series(tmp_path / "plain", RescaleSlope=2.0, RescaleIntercept=-1.0)
    [plain] = gu.load(tmp_path / "plain")
    assert np.allclose(plain.data[:, :, 1, 0], values[(9.0, 1)].T * 2 - 1)
    values = _series(tmp_path / "philips", Manufacturer="Philips", RescaleSlope=2.0)
    for path in (tmp_path / "philips").iterdir():  # private MRScaleSlope (2005,100E)
        ds = pydicom.dcmread(path)
        ds.private_block(0x2005, "Philips MR Imaging DD 001", create=True).add_new(0x0E, "FL", 4.0)
        ds.save_as(path)
    [philips] = gu.load(tmp_path / "philips")
    assert np.allclose(philips.data[:, :, 0, 0], values[(5.0, 1)].T / 4.0)  # (v*2)/(2*4)


def test_echoes_are_separate_images(tmp_path):
    _series(tmp_path, echo=10.0)
    _series(tmp_path, echo=20.0)
    images = sorted(gu.load(tmp_path), key=lambda im: im.name)
    assert [im.name for im in images] == ["3_e1", "3_e2"]
    assert [im.meta["EchoTime"] for im in images] == pytest.approx([0.01, 0.02])


def test_study_folder(tmp_path):
    (tmp_path / "series_a").mkdir()
    (tmp_path / "README.txt").write_text("a study")
    _series(tmp_path / "series_a")
    assert gu.sniff(tmp_path).format == gu.Format.DICOM_IMAGE
    [im] = gu.load(tmp_path)
    assert im.shape == (3, 2, 2, 2)


def test_study_with_dicomdir(tmp_path):
    """A DICOMDIR at the top of the study is not data; the series below it are read."""
    study = {
        "PatientID": "p", "PatientName": "P", "StudyInstanceUID": "1.2", "StudyID": "1",
        "StudyDate": "20200101", "StudyTime": "120000",
    }
    (tmp_path / "flat").mkdir()
    values = _series(tmp_path / "flat", **study)
    file_set = FileSet()
    for path in sorted((tmp_path / "flat").iterdir()):
        file_set.add(pydicom.dcmread(path))
    file_set.write(tmp_path / "cd")
    assert (tmp_path / "cd" / "DICOMDIR").is_file()
    assert gu.sniff(tmp_path / "cd" / "DICOMDIR").format == gu.Format.DICOM_OTHER
    assert gu.sniff(tmp_path / "cd").format == gu.Format.DICOM_IMAGE
    [im] = gu.load(tmp_path / "cd")
    assert im.shape == (3, 2, 2, 2)
    assert np.array_equal(im.data[:, :, 0, 1], values[(5.0, 2)].T)
