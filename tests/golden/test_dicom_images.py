####################################################################################################
#                                       test_dicom_images.py                                       #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: DICOM images against dcm2niix: every reference NIfTI of the dcm_qa sets must equal the  #
#          image MIOX reads from the same DICOM, voxel by voxel in world space (dcm2niix may    #
#          flip or permute axes), and every key both state must agree.                             #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import functools

import numpy as np
import pytest

# own
from miox import MRImage, ReaderError, load
from tests.conftest import PUBLIC_DATA

SETS = (
    "dcm_qa_nih",
    "dcm_qa_enh",
    "dcm_qa_mosaic",
    "dcm_qa_ts",
    "dcm_qa_12bit",
    "dcm_qa_me",
    "dcm_qa_pdt2",
    "dcm_qa_frag",
)
REFERENCES = sorted(
    p.relative_to(PUBLIC_DATA).as_posix()
    for s in SETS
    for p in (PUBLIC_DATA / s / "Ref").glob("*.nii")
)
SKIPPED_FILES = (".txt", ".md", ".pdf", ".png")


@functools.cache
def inputs(name: str) -> tuple[list[MRImage], list[str]]:
    """Every image MIOX reads from a set's input (series folders or files), and the errors."""
    root = PUBLIC_DATA / name / "In"
    leaves = set()
    for path in root.rglob("*"):
        if path.is_file() and not path.name.startswith(".") and path.suffix not in SKIPPED_FILES:
            leaves.add(path if path.parent == root else path.parent)
    images, errors = [], []
    for leaf in sorted(leaves):
        try:
            loaded = load(leaf)
        except ReaderError as e:
            errors.append(str(e))
            continue
        images += [im for im in loaded if isinstance(im, MRImage)]
        errors += [n for im in loaded for n in im.provenance.notes if "not read" in n]
    return images, errors


def kind(name: str, image_type) -> str:
    types = {str(v).upper() for v in (image_type or [])}
    if "_fieldmap" in name or "FIELDMAPHZ" in types:
        return "fieldmap"
    if "_ph" in name or types & {"P", "PHASE"}:
        return "phase"
    return "magnitude"


def on_reference_grid(image: MRImage, reference: MRImage) -> np.ndarray:
    """``image``'s data indexed as ``reference``'s; the affines may differ by axis flips and
    permutations only."""
    mapping = np.linalg.inv(reference.affine) @ image.affine
    axes = np.round(mapping[:3, :3])
    assert np.allclose(mapping[:3, :3], axes, atol=2e-3), "affines not related by axis flips"
    order = [int(np.flatnonzero(axes[a])[0]) for a in range(3)]
    data = np.transpose(image.data, order + list(range(3, image.data.ndim)))
    offsets = []
    for a in range(3):
        flipped = axes[a, order[a]] < 0
        data = np.flip(data, axis=a) if flipped else data
        offsets.append(data.shape[a] - 1 if flipped else 0)
    assert np.allclose(mapping[:3, 3], offsets, atol=0.05), "grids shifted"
    return data


def same(key: str, ours, theirs) -> bool:
    """dcm2niix shortens Manufacturer and rewrites ImageType; both are compared as prefixes."""
    if key == "Manufacturer":
        return str(ours).lower().startswith(str(theirs).lower())
    if key == "ImageType":
        return "_".join(map(str, theirs)).startswith("_".join(map(str, ours)))
    numbers = (int, float)
    if isinstance(ours, numbers) and isinstance(theirs, numbers):
        return bool(np.isclose(ours, theirs, rtol=1e-4, atol=1e-6))
    if isinstance(ours, list) and isinstance(theirs, list):
        if all(isinstance(v, numbers) for v in ours + theirs):
            return len(ours) == len(theirs) and np.allclose(ours, theirs, rtol=1e-4, atol=1e-6)
    return str(ours).strip() == str(theirs).strip()


@pytest.mark.data
@pytest.mark.parametrize("relpath", REFERENCES)
def test_dicom_image_equals_dcm2niix(relpath):
    [reference] = load(PUBLIC_DATA / relpath)
    images, errors = inputs(relpath.split("/")[0])
    series = reference.meta.get("SeriesNumber")
    protocol = str(reference.meta.get("ProtocolName")).strip()
    wanted = kind(relpath, reference.meta.get("ImageType"))
    candidates = [
        im
        for im in images
        if im.meta.get("SeriesNumber") == series
        and str(im.meta.get("ProtocolName")).strip() == protocol
        and kind(im.name, []) == wanted
    ]
    if len(candidates) > 1 and reference.meta.get("EchoTime") is not None:
        echo = reference.meta["EchoTime"]
        candidates = sorted(candidates, key=lambda im: abs(im.meta.get("EchoTime", 0) - echo))
    if not candidates:
        if any("not decoded" in e for e in errors):
            pytest.skip("pixel data compressed in a syntax no installed decoder reads")
        pytest.fail(f"no image for series {series} {protocol} ({wanted}); errors: {errors}")
    image = candidates[0]
    data = on_reference_grid(image, reference)
    assert data.shape == reference.shape
    scale = float(np.abs(reference.data).max()) or 1.0
    assert np.allclose(data, reference.data, rtol=1e-4, atol=1e-3 * scale)
    differ = [
        key
        for key in set(image.meta) & set(reference.meta)
        if not same(key, image.meta[key], reference.meta[key])
    ]
    assert differ == []
