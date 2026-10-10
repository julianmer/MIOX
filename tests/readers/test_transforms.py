####################################################################################################
#                                       test_transforms.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 10/10/26                                                                                #
#                                                                                                  #
# Purpose: Transform readers on synthetic files: an ITK affine with a centre (text and HDF5), AFNI #
#          rows, an FSL matrix, LTA RAS-to-RAS and voxel-to-voxel with their volume geometry, and  #
#          a numeric text table that is not a transform.                                           #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import h5py
import numpy as np

# own
import miox as gu

LPS = np.diag([-1.0, -1.0, 1.0, 1.0])
LINEAR = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]])


def test_itk_text_with_a_centre(tmp_path):
    path = tmp_path / "a.tfm"
    path.write_text("#Insight Transform File V1.0\n#Transform 0\n"
                    "Transform: AffineTransform_double_3_3\n"
                    f"Parameters: {' '.join(map(str, LINEAR.ravel()))} 1 2 3\n"
                    "FixedParameters: 10 0 0\n")
    sniff = gu.sniff(path)
    assert (sniff.format, sniff.kind) == (gu.Format.TRANSFORM_ITK, gu.Kind.TRANSFORM)
    [t] = gu.load(path)
    centre = np.array([10.0, 0.0, 0.0])
    point = np.array([3.0, 4.0, 5.0])
    expected = LINEAR @ (point - centre) + np.array([1.0, 2.0, 3.0]) + centre
    assert np.allclose(t.matrix[:3, :3] @ point + t.matrix[:3, 3], expected)
    assert np.allclose(t.ras_matrix, LPS @ t.matrix @ LPS)


def test_itk_hdf5(tmp_path):
    path = tmp_path / "composite.h5"
    with h5py.File(path, "w") as fh:
        fh["TransformGroup/0/TransformType"] = [b"CompositeTransform_double_3_3"]
        fh["TransformGroup/1/TransformType"] = [b"AffineTransform_double_3_3"]
        fh["TransformGroup/1/TransformParameters"] = np.r_[LINEAR.ravel(), 1.0, 2.0, 3.0]
        fh["TransformGroup/1/TransformFixedParameters"] = np.zeros(3)
    [t] = gu.load(path)
    assert np.allclose(t.matrix[:3, :3], LINEAR) and np.allclose(t.matrix[:3, 3], [1, 2, 3])


def test_afni_rows(tmp_path):
    path = tmp_path / "motion.aff12.1D"
    row = " ".join(map(str, np.c_[LINEAR, [1, 2, 3]].ravel()))
    path.write_text(f"# 3dvolreg matrices (DICOM-to-DICOM, row-by-row):\n{row}\n{row}\n")
    assert gu.sniff(path).format == gu.Format.TRANSFORM_AFNI
    transforms = gu.load(path)
    assert len(transforms) == 2
    assert np.allclose(transforms[1].ras_matrix, LPS @ transforms[1].matrix @ LPS)


def test_fsl_matrix(tmp_path):
    path = tmp_path / "flirt.mat"
    path.write_text("1 0 0 2\n0 1 0 3\n0 0 1 4\n0 0 0 1.00000012\n")
    assert gu.sniff(path).format == gu.Format.TRANSFORM_FSL
    [t] = gu.load(path)
    assert t.ras_matrix is None and t.matrix[2, 3] == 4.0 and t.provenance.notes


def volume(cras: str) -> str:
    return ("valid = 1\nfilename = img.nii\nvolume = 10 10 10\nvoxelsize = 2 2 2\n"
            "xras = 1 0 0\nyras = 0 1 0\nzras = 0 0 1\n" f"cras = {cras}\n")


def test_lta_voxel_to_voxel_through_its_geometry(tmp_path):
    path = tmp_path / "v2v.lta"
    path.write_text("type = 0 # LINEAR_VOX_TO_VOX\nnxforms = 1\nmean = 0 0 0\nsigma = 1\n"
                    "1 4 4\n1 0 0 1\n0 1 0 0\n0 0 1 0\n0 0 0 1\n"
                    f"src volume info\n{volume('0 0 0')}dst volume info\n{volume('5 0 0')}")
    assert gu.sniff(path).format == gu.Format.TRANSFORM_LTA
    [t] = gu.load(path)
    # one voxel (2 mm) along x plus the 5 mm between the centres
    assert np.allclose(t.ras_matrix[:3, 3], [7.0, 0.0, 0.0])
    assert t.source["filename"] == "img.nii"


def test_numeric_table_is_not_a_transform(tmp_path):
    path = tmp_path / "fid.txt"
    path.write_text("".join(f"{i} {i + 1} {i + 2} {i + 3}\n" for i in range(4)))
    assert gu.sniff(path).kind != gu.Kind.TRANSFORM
