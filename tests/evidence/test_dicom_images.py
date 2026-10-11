####################################################################################################
#                                       test_dicom_images.py                                       #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: DICOM images against dcm2niix: every reference NIfTI of the dcm_qa sets must equal the  #
#          image Miox reads from the same DICOM, voxel by voxel in world space (dcm2niix may       #
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
    "dcm_qa",
    "dcm_qa_ge",
    "dcm_qa_xa30",
    "dcm_qa_philips",
    "dcm_qa_philips_enh",
    "dcm_qa_philips_asl",
    "dcm_qa_uih",
    "dcm_qa_canon_enh",
    "dcm_qa_toshiba",
    "dcm_qa_fmap",
    "dcm_qa_dti",
    "dcm_qa_asl",
    "dcm_qa_stc",
)
REFERENCES = sorted(
    p.relative_to(PUBLIC_DATA).as_posix()
    for s in SETS
    for p in (PUBLIC_DATA / s / "Ref").glob("*.nii")
)
SKIPPED_FILES = (".txt", ".md", ".pdf", ".png", ".csv", ".xlsx")
KNOWN_DIFFERENCES = {  # (set, key): why Miox's value differs from dcm2niix's on purpose
    ("dcm_qa_philips_asl", "RepetitionTime"):
        "dcm2niix replaces TR by the mean spacing of the label and control volumes",
    ("dcm_qa_philips_asl", "EchoTrainLength"):
        "dcm2niix writes the Philips EPI factor (2001,1013) for classic files, not (0018,0091)",
    ("dcm_qa_philips_enh", "EchoTrainLength"):
        "dcm2niix writes the Philips EPI factor (2001,1013) for classic files, not (0018,0091)",
    ("dcm_qa_philips_asl", "EstimatedEffectiveEchoSpacing"): "follows from the EPI factor above",
    ("dcm_qa_philips_asl", "EstimatedTotalReadoutTime"): "follows from the EPI factor above",
    ("dcm_qa_philips_enh", "EstimatedEffectiveEchoSpacing"): "follows from the EPI factor above",
    ("dcm_qa_philips_enh", "EstimatedTotalReadoutTime"): "follows from the EPI factor above",
    ("dcm_qa_asl", "SliceTiming"):
        "dcm2niix replaces MosaicRefAcqTimes spanning more than TR (two concatenations) by an "
        "ascending estimate",
    ("dcm_qa_enh", "SpoilingType"):
        "dcm2niix writes COMBINED for Spoiling RF (its RF-and-gradient test checks RF twice)",
    ("dcm_qa_philips_enh", "SpoilingType"):
        "dcm2niix writes COMBINED for Spoiling RF (its RF-and-gradient test checks RF twice)",
    ("dcm_qa_philips_enh", "RepetitionTime"):
        "dcm2niix replaces TR by the mean spacing of the volumes when they differ",
}
KNOWN_DATA = {  # reference: (why its voxel values differ from Miox's, what still must hold)
    "dcm_qa_philips_asl/Ref/301_ASL_MultiPhase_classic.nii": (
        "Philips scale factors vary per volume; dcm2niix applies the first file's RescaleSlope on "
        "top of each file's own, Miox the precise scaling of each file",
        "proportional",
    ),
    "dcm_qa_philips_enh/Ref/2201_WIP_DTI_S2MB2_cl.nii": (
        "at slices 68-76 dcm2niix assigns images to other volumes than their gradient tags say",
        "most slices",
    ),
    "dcm_qa_philips_enh/Ref/2301_WIP_DTIphase_S2MB2_cl.nii": (
        "at slices 68-76 dcm2niix assigns images to other volumes than their gradient tags say",
        "most slices",
    ),
}


@functools.cache
def inputs(name: str) -> tuple[list[MRImage], list[str]]:
    """Every image Miox reads from a set's input (series folders or files), and the errors."""
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
    name = name.rsplit("/", 1)[-1]
    types = {str(v).upper() for v in (image_type or [])}
    if "_fieldmap" in name or "FIELDMAPHZ" in types:
        return "fieldmap"
    if "_ph" in name or types & {"P", "PHASE"}:
        return "phase"
    return "magnitude"


def grid_mapping(image: MRImage, reference: MRImage) -> tuple[list[int], list[bool]]:
    """For each reference axis, the image axis it is and whether it runs the other way; the
    affines may differ by axis flips and permutations only."""
    mapping = np.linalg.inv(reference.affine) @ image.affine
    axes = np.round(mapping[:3, :3])
    assert np.allclose(mapping[:3, :3], axes, atol=2e-3), "affines not related by axis flips"
    order = [int(np.flatnonzero(axes[a])[0]) for a in range(3)]
    flipped = [bool(axes[a, order[a]] < 0) for a in range(3)]
    offsets = [image.shape[order[a]] - 1 if flipped[a] else 0 for a in range(3)]
    assert np.allclose(mapping[:3, 3], offsets, atol=0.05), "grids shifted"
    return order, flipped


def on_reference_grid(image: MRImage, reference: MRImage) -> np.ndarray:
    """``image``'s data indexed as ``reference``'s."""
    order, flipped = grid_mapping(image, reference)
    data = np.transpose(image.data, order + list(range(3, image.data.ndim)))
    for a in range(3):
        data = np.flip(data, axis=a) if flipped[a] else data
    return data


def on_reference_axes(key: str, value, order: list[int], flipped: list[bool], is_3d=False):
    """A voxel-axis value of the image expressed on the reference's axes. dcm2niix reorients 3D
    acquisitions but states their phase-encoding axis on the DICOM grid with its rows flipped,
    so that one only gets its j sign inverted."""
    if key in ("PhaseEncodingDirection", "PhaseEncodingAxis") and is_3d:
        if key == "PhaseEncodingDirection" and str(value).startswith("j"):
            return "j" if value.endswith("-") else "j-"
        return value
    if key in ("PhaseEncodingDirection", "PhaseEncodingAxis") and isinstance(value, str):
        axis = "ijk".find(value[0])
        if axis < 0:
            return value
        target = order.index(axis)
        negative = value.endswith("-") != flipped[target]
        return "ijk"[target] + ("-" if negative and key == "PhaseEncodingDirection" else "")
    if key == "SliceTiming" and isinstance(value, list) and flipped[2]:
        return value[::-1]
    return value


def same(key: str, ours, theirs) -> bool:
    """dcm2niix shortens Manufacturer and rewrites ImageType; both are compared as prefixes.
    Times of day are compared to 10 µs (dcm2niix rounds to 5 decimals) and slice times to 20 ms
    (dcm2niix holds times of day in single precision, 4 to 8 ms steps)."""
    if key == "Manufacturer":
        return str(ours).lower().startswith(str(theirs).lower())
    if key == "ImageType":
        return "_".join(map(str, theirs)).startswith("_".join(map(str, ours)))
    if key == "ImageTypeText":  # dcm2niix splits values at underscores (TENSOR_B0)
        return "_".join(map(str, theirs)) == "_".join(map(str, ours))
    if key in ("AcquisitionTime", "AcquisitionDateTime"):
        a, b = str(ours)[-15:], str(theirs)[-15:]
        if a[:6] != b[:6]:
            return False
        try:
            return abs(float(a[6:]) - float(b[6:])) < 1e-5
        except ValueError:
            return a == b
    if key == "SliceTiming" and isinstance(ours, list) and isinstance(theirs, list):
        return len(ours) == len(theirs) and np.allclose(ours, theirs, rtol=0, atol=2e-2)
    numbers = (int, float)
    if isinstance(ours, numbers) and isinstance(theirs, numbers):
        return bool(np.isclose(ours, theirs, rtol=1e-4, atol=1e-6))
    if isinstance(ours, list) and isinstance(theirs, list):
        if all(isinstance(v, numbers) for v in ours + theirs):
            return len(ours) == len(theirs) and np.allclose(ours, theirs, rtol=1e-4, atol=1e-6)
        if key == "DiffusionGradientOrientation":
            a, b = np.asarray(ours, float), np.asarray(theirs, float)
            return a.shape == b.shape and np.allclose(a, b, rtol=0, atol=2e-3)
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
    problems, matches = [], []
    for image in candidates:  # one series number may recur in a set (sessions, classic + enhanced)
        try:
            order, flipped = grid_mapping(image, reference)
            assert image.shape[3:] == reference.shape[3:], "volumes differ"
        except AssertionError as e:
            problems.append(f"{image.name} {image.shape}: {e}")
            continue
        differ = differences(image, reference, order, flipped, relpath)
        matches.append((differ, image, order, flipped))
    if not matches:
        pytest.fail(f"no image on the reference grid: {problems}")
    differ, image, order, flipped = min(matches, key=lambda m: len(m[0]))
    data = on_reference_grid(image, reference)
    assert data.shape == reference.shape
    scale = float(np.abs(reference.data).max()) or 1.0
    if relpath not in KNOWN_DATA:
        assert np.allclose(data, reference.data, rtol=1e-4, atol=1e-3 * scale)
    elif KNOWN_DATA[relpath][1] == "proportional":
        ours, theirs = data.astype(float).ravel(), reference.data.astype(float).ravel()
        assert np.corrcoef(ours, theirs)[0, 1] > 0.9999, KNOWN_DATA[relpath][0]
    else:  # most slices equal
        equal = [
            np.allclose(data[:, :, k], reference.data[:, :, k], rtol=1e-4, atol=1e-3 * scale)
            for k in range(data.shape[2])
        ]
        assert sum(equal) >= 0.8 * len(equal), KNOWN_DATA[relpath][0]
    assert not differ, differ


def differences(image, reference, order, flipped, relpath) -> list[str]:
    """Keys both state with different values, known deviations from dcm2niix left out."""
    differ = []
    is_3d = str(image.meta.get("MRAcquisitionType", "")).upper() == "3D"
    for key in sorted(set(image.meta) & set(reference.meta)):
        if (relpath.split("/")[0], key) in KNOWN_DIFFERENCES:
            continue
        ours = on_reference_axes(key, image.meta[key], order, flipped, is_3d)
        theirs = reference.meta[key]
        if key == "DiffusionGradientOrientation":  # dcm2niix keeps the stated vector of b=0 volumes
            weighted = [b > 0 for b in reference.meta.get("DiffusionBValue", [])]
            if len(weighted) == len(ours) == len(theirs):
                ours = [v for v, w in zip(ours, weighted, strict=True) if w]
                theirs = [v for v, w in zip(theirs, weighted, strict=True) if w]
        if not same(key, ours, theirs):
            differ.append(f"{key}: {ours!r} vs {theirs!r}")
    return differ
