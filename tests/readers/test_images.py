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
import miox
from miox.readers.nibabel_image import NibabelImageReader


def test_nifti_with_bids_sidecar(tmp_path):
    affine = np.diag([2.0, 2.0, 3.0, 1.0])
    data = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    nib.Nifti1Image(data, affine).to_filename(tmp_path / "t1.nii.gz")
    sidecar = {"EchoTime": 0.003, "RepetitionTime": 2.3, "FlipAngle": 9,
               "MagneticFieldStrength": 3, "ImagingFrequency": 123.2, "SliceThickness": 3,
               "NotAStandardField": "kept"}
    (tmp_path / "t1.json").write_text(json.dumps(sidecar))
    [im] = miox.load(tmp_path / "t1.nii.gz")
    assert isinstance(im, miox.MRImage) and im.shape == (2, 3, 4) and im.voxel_sizes == (2, 2, 3)
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
    [im] = miox.load(tmp_path / "t1.mgz")
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


def test_bruker_2dseq_frames_groups_and_geometry(tmp_path):
    from miox.readers.bruker_2dseq import Bruker2dseq

    visu = {
        "VisuCoreSize": [4, 3],
        "VisuCoreDimDesc": ["spatial", "spatial"],
        "VisuCoreExtent": [8.0, 6.0],
        "VisuCoreFrameCount": 6,
        "VisuCoreFrameType": "MAGNITUDE_IMAGE",
        "VisuCoreWordType": "_16BIT_SGN_INT",
        "VisuCoreByteOrder": "littleEndian",
        "VisuCoreDataSlope": [0.5] * 6,
        "VisuCoreDataOffs": [1.0] * 6,
        "VisuFGOrderDesc": [[3, "<FG_SLICE>", "<>", 0, 2], [2, "<FG_ECHO>", "<>", 2, 1]],
        "VisuCoreOrientation": [[1, 0, 0, 0, 1, 0, 0, 0, 1]] * 6,
        "VisuCorePosition": [[0, 0, float(k)] for k in (0, 2, 4)] * 2,
        "VisuCoreSlicePacksSlices": [[0, 3]],
    }
    path = tmp_path / "2dseq"
    try:
        np.arange(6 * 12, dtype="<i2").tofile(path)
        d = Bruker2dseq(path, visu)
        frames = d.frames()
        assert frames.shape == (6, 3, 4) and frames[0, 0, 1] == 1 * 0.5 + 1.0
        assert d.groups() == [("FG_SLICE", 3), ("FG_ECHO", 2)] and d.spacing() == [2.0, 2.0]
        assert d.slice_packs() == [(0, 3)] and d.spectral_axis() is None
        assert np.allclose(d.position(5), [0, 0, 4]) and not d.transposed(0)
    finally:
        path.unlink(missing_ok=True)
