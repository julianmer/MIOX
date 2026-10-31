####################################################################################################
#                                       test_transforms.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 10/10/26                                                                                #
#                                                                                                  #
# Purpose: Linear transforms against each other: one affine written as LTA, ITK and AFNI files     #
#          gives one RAS map; voxel-to-voxel LTAs equal their RAS-to-RAS twins through the volume  #
#          geometry they state; real FreeSurfer, ITK and FSL outputs of one registration agree     #
#          (ITK maps the other way, FSL through the LTA's image geometry); an ANTs registration's  #
#          GenericAffine.mat is its composite's translation and affine stages applied one after    #
#          the other.                                                                              #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import numpy as np
import pytest

# own
from miox import load
from tests.conftest import PUBLIC_DATA

DATA = PUBLIC_DATA / "nitransforms/nitransforms/tests/data"
REGRESSIONS = ["from-fsnative_to-bold", "from-scanner_to-bold", "from-fsnative_to-scanner",
               "from-scanner_to-fsnative"]


def needed(*paths):
    for path in paths:
        if not path.is_file():
            pytest.skip(f"{path.name} not fetched")


@pytest.mark.parametrize("name", ["RAS", "LAS", "LPS", "oblique"])
def test_one_affine_in_three_conventions(name):
    """nitransforms wrote one affine as LTA (RAS to RAS), ITK (.tfm, LPS) and AFNI (3dvolreg
    rows, LPS, two identical rows): all read to the same RAS map to single precision."""
    lta, itk, afni = (DATA / f"affine-{name}.{s}" for s in ("fs.lta", "itk.tfm", "afni-array"))
    needed(lta, itk, afni)
    reference = load(lta)[0].ras_matrix
    for transform in load(itk) + load(afni):
        assert np.allclose(transform.ras_matrix, reference, atol=1e-5)


@pytest.mark.parametrize("stem", ["affine-RAS.fs", "bold-to-t1w"])
def test_voxel_lta_equals_its_ras_twin(stem):
    """A voxel-to-voxel LTA, mapped through the source and destination geometry it states
    (centre voxel at cras), equals the RAS-to-RAS LTA of the same transform."""
    v2v, ras = DATA / f"{stem}.v2v.lta", DATA / f"{stem}.lta"
    needed(v2v, ras)
    assert np.allclose(load(v2v)[0].ras_matrix, load(ras)[0].ras_matrix, atol=1e-4)


def fsl_frame(info: dict) -> tuple[np.ndarray, np.ndarray]:
    """FSL's scaled-voxel matrix of an image (x flipped when its vox2ras has a positive
    determinant) and the image's vox2ras, from an LTA volume block."""
    vox2ras = np.asarray(info["vox2ras"])
    sizes = [float(v) for v in info["voxelsize"].split()]
    scale = np.diag([*sizes, 1.0])
    if np.linalg.det(vox2ras[:3, :3]) > 0:
        flip = np.eye(4)
        flip[0, 0], flip[0, 3] = -1.0, float(info["volume"].split()[0]) - 1
        scale = scale @ flip
    return scale, vox2ras


@pytest.mark.parametrize("name", REGRESSIONS)
def test_freesurfer_itk_and_fsl_outputs_of_one_registration(name):
    """One registration as written by FreeSurfer (LTA, RAS to RAS), ITK (.tfm) and FSL: the ITK
    RAS map is the inverse of the LTA's (ITK maps fixed to moving points), and the FSL matrix
    through the two images' geometry the LTA states equals the LTA to 1e-4."""
    lta, itk, fsl = (DATA / "regressions" / f"{name}_mode-image.{s}" for s in ("lta", "tfm", "fsl"))
    needed(lta, itk, fsl)
    [freesurfer], [ants], [flirt] = load(lta), load(itk), load(fsl)
    assert np.allclose(ants.ras_matrix, np.linalg.inv(freesurfer.ras_matrix), atol=1e-4)
    source_scale, source = fsl_frame(freesurfer.source)
    target_scale, target = fsl_frame(freesurfer.target)
    ras = target @ np.linalg.inv(target_scale) @ flirt.matrix @ source_scale @ np.linalg.inv(source)
    assert np.allclose(ras, freesurfer.ras_matrix, atol=1e-4)


ANTS = PUBLIC_DATA / "simpleitk_ants"


@pytest.mark.data
def test_ants_generic_affine_is_its_composite_linear_stages():
    """ANTs writes the linear stages of a registration twice: collapsed into one affine
    (0GenericAffine.mat) and as the first stages of the composite (.h5: translation, affine, then a
    displacement field). ITK applies a composite's stages last added first, so the .mat equals
    the translation applied after the affine."""
    if not ANTS.is_dir():
        pytest.skip("SimpleITK-Notebooks data not fetched")
    [generic] = load(ANTS / "B1_fixed_B2_moving0GenericAffine.mat")
    translation, linear = load(ANTS / "B1_fixed_B2_movingComposite.h5")
    assert translation.meta.vendor["type"].startswith("TranslationTransform")
    assert linear.meta.vendor["type"].startswith("AffineTransform")
    assert np.allclose(generic.matrix, translation.matrix @ linear.matrix, atol=1e-12)
    assert np.allclose(generic.ras_matrix, translation.ras_matrix @ linear.ras_matrix, atol=1e-12)
