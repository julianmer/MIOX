####################################################################################################
#                                          test_images.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Image readers on synthetic files: NIfTI with a BIDS sidecar, MGH units, the written PAR #
#          patient position. No test data needed.                                                  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pytest

# own
import miox as gu
from miox.readers.nibabel_image import NibabelImageReader


def test_nifti_with_bids_sidecar(tmp_path):
    affine = np.diag([2.0, 2.0, 3.0, 1.0])
    data = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    nib.Nifti1Image(data, affine).to_filename(tmp_path / "t1.nii.gz")
    sidecar = {"EchoTime": 0.003, "RepetitionTime": 2.3, "FlipAngle": 9,
               "MagneticFieldStrength": 3, "ImagingFrequency": 123.2, "SliceThickness": 3,
               "NotAStandardField": "kept"}
    (tmp_path / "t1.json").write_text(json.dumps(sidecar))
    [im] = gu.load(tmp_path / "t1.nii.gz")
    assert isinstance(im, gu.MRImage) and im.shape == (2, 3, 4) and im.voxel_sizes == (2, 2, 3)
    assert np.array_equal(im.data, data) and np.allclose(im.affine, affine)
    assert im.meta["ExcitationFlipAngle"] == 9 and im.meta["FieldStrength"] == 3
    assert im.meta["SpectrometerFrequency"] == [123.2] and im.meta["SliceThickness"] == 3
    assert im.meta.source["EchoTime"] == "BIDS sidecar: EchoTime"
    assert "NotAStandardField" not in im.meta
    assert im.meta.vendor["sidecar"]["NotAStandardField"] == "kept"


def test_mgh_units(tmp_path):
    img = nib.MGHImage(np.zeros((2, 2, 2), np.float32), np.eye(4))
    img.header["tr"], img.header["te"], img.header["ti"] = 2300.0, 2.98, 900.0
    img.header["flip_angle"] = np.radians(9.0)
    img.to_filename(tmp_path / "t1.mgz")
    [im] = gu.load(tmp_path / "t1.mgz")
    assert im.meta["RepetitionTime"] == pytest.approx(2.3)
    assert im.meta["EchoTime"] == pytest.approx(0.00298)
    assert im.meta["InversionTime"] == pytest.approx(0.9)
    assert im.meta["ExcitationFlipAngle"] == pytest.approx(9.0)


@pytest.mark.parametrize(
    "text,code",
    [("Head First Supine", "HFS"), ("Feet First Prone", "FFP"),
     ("Head First Decubitus Right", "HFDR"), ("sideways", "")],
)
def test_written_patient_position(text, code):
    assert NibabelImageReader()._position(text) == code
