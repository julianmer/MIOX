####################################################################################################
#                                        test_agreement.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Cross-format agreement on the public test data: the same acquisition stored in several  #
#          formats must give the same values. Unlike the records, this is independent              #
#          evidence that the values are correct.                                                   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest

# own
from miox import Format, MRSData, load
from tests.conftest import PUBLIC_DATA
from tests.records.records import ReadRecords

PHILIPS = PUBLIC_DATA / "spec2nii_test_data/philips/spar_dcm_orientation_tests"
SIEMENS_KEYS = (  # values twix, DICOM and RDA of one Siemens scan must share
    "SpectrometerFrequency",
    "ResonantNucleus",
    "EchoTime",
    "RepetitionTime",
    "ExcitationFlipAngle",
)
AVERAGED_KEYS = ("DwellTime", "NumberOfPoints")  # twix keeps readout oversampling, so not twix
ABSOLUTE = {"DwellTime": 5e-7}  # RDA writes the dwell time in whole µs


def _same(a: MRSData, b: MRSData, keys: tuple[str, ...]) -> list[str]:
    differences = []
    for key in keys:
        x, y = a.meta.get(key), b.meta.get(key)
        if isinstance(x, (str, list)) and np.ravel(x).dtype.kind not in "fiu":
            if x != y:
                differences.append(f"{key}: {x!r} vs {y!r}")
        elif not np.allclose(
            np.ravel(x).astype(float),
            np.ravel(y).astype(float),
            rtol=1e-6,
            atol=ABSOLUTE.get(key, 0),
        ):
            differences.append(f"{key}: {x!r} vs {y!r}")
    return differences


def _siemens_groups() -> list[list[str]]:
    """Public Siemens files of one acquisition (same protocol, patient and frequency)."""
    groups = defaultdict(list)
    for relpath, record in ReadRecords().expected().items():
        if isinstance(record, dict) or not relpath.startswith("spec2nii_test_data/Siemens"):
            continue
        meta = record[0]["meta"]
        if "SpectrometerFrequency" not in meta:  # images next to the spectra
            continue
        frequency = str(meta["SpectrometerFrequency"])
        key = (meta.get("ProtocolName"), meta.get("PatientName"), frequency)
        groups[key].append(relpath)
    return [sorted(paths) for paths in groups.values() if len(paths) > 1]


@pytest.mark.data
@pytest.mark.parametrize("rotation", range(9))
def test_philips_spar_classic_enhanced_agree(rotation):
    spars = sorted(PHILIPS.glob("dbiex_4*_2_raw_act.spar"))
    classic = sorted((PHILIPS / "classic").iterdir())
    enhanced = sorted((PHILIPS / "enhanced").iterdir())
    if len(spars) < 9 or len(classic) < 9 or len(enhanced) < 9:
        pytest.skip("Philips orientation tests not fetched")
    datasets = [load(p)[0] for p in (spars[rotation], classic[rotation], enhanced[rotation])]
    keys = ("SpectrometerFrequency", "EchoTime", "RepetitionTime") + AVERAGED_KEYS
    for d in datasets[1:]:
        assert _same(datasets[0], d, keys) == []
        assert np.allclose(d.affine, datasets[0].affine, atol=1e-6)
        x, y = (v.data.ravel() / np.linalg.norm(v.data) for v in (datasets[0], d))
        assert abs(np.vdot(x, y)) > 1 - 1e-5


@pytest.mark.data
@pytest.mark.parametrize("relpaths", _siemens_groups(), ids=lambda paths: paths[0].split("/")[-1])
def test_siemens_formats_agree(relpaths):
    datasets = {}
    for relpath in relpaths:
        d = load(PUBLIC_DATA / relpath)[0]
        datasets.setdefault(d.provenance.format, d)
    formats = list(datasets)
    reference = datasets[formats[0]]
    for fmt in formats[1:]:
        d = datasets[fmt]
        keys = SIEMENS_KEYS
        if Format.SIEMENS_TWIX not in (fmt, formats[0]):
            keys += AVERAGED_KEYS
        assert _same(reference, d, keys) == [], f"{formats[0]} vs {fmt}"
        assert d.shape[:3] == reference.shape[:3]
        assert np.allclose(d.affine, reference.affine, atol=0.05), f"{formats[0]} vs {fmt}"


@pytest.mark.data
def test_siemens_va_dicom_and_rda_of_one_protocol_agree():
    """Two acquisitions of one protocol on one VA25A scanner: geometry, timing and sign."""
    dicom = PUBLIC_DATA / "tarquin/src/unit_tests/data/siemens_ima_WS.ima"
    rda = PUBLIC_DATA / "tarquin/data/siemens/example.rda"
    if not dicom.exists() or not rda.exists():
        pytest.skip("file not fetched")
    a, b = load(dicom)[0], load(rda)[0]
    assert _same(a, b, ("EchoTime", "RepetitionTime", "ExcitationFlipAngle")) == []
    assert a.dwelltime == b.dwelltime and np.allclose(a.affine, b.affine)
    x, y = (d.data.ravel() / np.linalg.norm(d.data) for d in (a, b))
    assert abs(np.vdot(x, y)) > 0.9 > abs(np.vdot(x, np.conj(y)))


@pytest.mark.data
def test_philips_csi_signal_lies_in_the_press_box():
    """The voxels with strong water signal lie inside the localised volume (one voxel margin)."""
    path = PUBLIC_DATA / "nema_wg16/Philips/Philips MR Spectroscopy/DICOM/XX_0020"
    if not path.exists():
        pytest.skip("file not fetched")
    reference = next(d for d in load(path) if d.name.endswith("_ref"))
    power = np.linalg.norm(reference.data[:, :, 0], axis=-1)
    slabs = reference.meta.vendor["dicom"]["VolumeLocalizationSequence"]
    margin = abs(reference.affine[0, 0])
    strong = np.argwhere(power > 0.5 * power.max())
    assert len(strong) > 20
    for i, j in strong:
        centre = np.diag([-1, -1, 1]) @ (reference.affine @ [i, j, 0, 1])[:3]  # RAS to LPS
        for slab in slabs:
            offset = centre - np.asarray(slab["MidSlabPosition"])
            assert abs(offset @ slab["SlabOrientation"]) <= slab["SlabThickness"] / 2 + margin


@pytest.mark.data
@pytest.mark.parametrize("series", ["30", "31"])
def test_dcm2niix_spectroscopy_nifti_equals_the_dicom(series):
    """dcm2niix's complex NIfTI with an MRS sidecar holds what MIOX reads from the DICOM."""
    reference = PUBLIC_DATA / f"dcm_qa_spec/Ref/svs_se_{series}.nii"
    folder = PUBLIC_DATA / f"dcm_qa_spec/In/{series}_svs_se"
    if not reference.exists() or not folder.exists():
        pytest.skip("dcm_qa_spec not fetched")
    [nifti], [dicom] = load(reference), load(folder)
    assert nifti.shape == dicom.shape and np.allclose(nifti.data, dicom.data)
    assert np.allclose(nifti.affine, dicom.affine, atol=1e-4)
    assert _same(nifti, dicom, ("SpectrometerFrequency", "EchoTime")) == []


CHALLENGE = PUBLIC_DATA / "mrs_fitting_challenge"
CHALLENGE_GIVEN = {"SpectrometerFrequency": 123.22, "ResonantNucleus": "1H", "DwellTime": 2.5e-4}


@pytest.mark.data
def test_fitting_challenge_text_lcmodel_and_jmrui_agree():
    """The headerless text columns hold the same values as the jMRUI and LCModel exports; the
    text and LCModel files state no frequency or dwell time and take jMRUI's."""
    if not CHALLENGE.is_dir():
        pytest.skip("mrs_fitting_challenge not fetched")
    suppressed, water = load(CHALLENGE / "datasets_text/dataset1.txt", given=CHALLENGE_GIVEN)
    [raw] = load(CHALLENGE / "datasets_LCModel/dataset1.RAW", given=CHALLENGE_GIVEN, complete=True)
    [h2o] = load(CHALLENGE / "datasets_LCModel/dataset1.H2O", given=CHALLENGE_GIVEN, complete=True)
    [jmrui_ws] = load(CHALLENGE / "datasets_JMRUI/dataset1_WS.txt")
    [jmrui_water] = load(CHALLENGE / "datasets_JMRUI/dataset1_nWS.txt")
    assert suppressed.name == "dataset1 columns 1-2" and water.name == "dataset1 columns 3-4"
    for a, b, c in ((suppressed, raw, jmrui_ws), (water, h2o, jmrui_water)):
        assert _same(a, c, ("SpectrometerFrequency", "ResonantNucleus") + AVERAGED_KEYS) == []
        assert np.allclose(a.data, c.data, rtol=1e-6, atol=1e-3)
        assert np.allclose(b.data, c.data, rtol=1e-5, atol=1e-3)  # .RAW: 6 significant digits


@pytest.mark.data
def test_fitting_challenge_text_basis_equals_lcmodel_basis():
    if not CHALLENGE.is_dir():
        pytest.skip("mrs_fitting_challenge not fetched")
    [lcmodel] = load(CHALLENGE / "basisset_LCModel")
    given = CHALLENGE_GIVEN | {"SpectrometerFrequency": lcmodel.meta["SpectrometerFrequency"]}
    [text] = load(CHALLENGE / "basisset_text", given=given, complete=True)
    assert text.provenance.format == Format.BASIS_FOLDER and text.metabolites == lcmodel.metabolites
    assert _same(text, lcmodel, ("SpectrometerFrequency", "ResonantNucleus") + AVERAGED_KEYS) == []
    for metabolite in text.metabolites:
        assert np.allclose(text.fid(metabolite), lcmodel.fid(metabolite), rtol=1e-5, atol=1e-5)
