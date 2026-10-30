####################################################################################################
#                                        test_protocols.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/10/26                                                                                #
#                                                                                                  #
# Purpose: Protocol files against independent evidence: one Siemens exam exported as an exam tree  #
#          (.exar1) and printed by the scanner (XML), every protocol's keys from the ASCCONV block #
#          against the printed values; Philips ExamCards against gt-examcard-parser's expectations #
#          and their computed TR against the set TR; an LCModel control file against the inputs    #
#          LCModel printed in the report of its run; Philips Raw Data Storage against the SPAR of  #
#          the same scan, and an exam log with its ExamCard; Siemens PhoenixZIPReports and XA Raw  #
#          Data Storage against images of the same protocol.                                       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re
from collections import defaultdict

import pytest

# own
from miox import load
from tests.conftest import PUBLIC_DATA

COMA = PUBLIC_DATA / "mri_protocol/SiemensVidaProtocol"
LCMODEL = PUBLIC_DATA / "lcmodel/test_lcm"
KEYS = (
    "RepetitionTime", "EchoTime", "InversionTime", "ExcitationFlipAngle", "SliceThickness",
    "BaseResolution", "NumberOfAverages", "ParallelReductionFactorInPlane",
    "ParallelReductionFactorOutOfPlane", "RefLinesPE",
)


def _need(*paths):
    for path in paths:
        if not path.is_file():
            pytest.skip(f"{path.relative_to(PUBLIC_DATA)} not fetched")


def _same(a, b) -> bool:
    a, b = (a if isinstance(a, list) else [a]), (b if isinstance(b, list) else [b])
    return len(a) == len(b) and all(abs(x - y) <= 5e-4 * max(abs(x), abs(y)) for x, y in zip(
        a, b, strict=True))


def test_exam_tree_protocols_equal_the_printed_protocols():
    """The 40 protocols of the Coma exam: the exam tree and the printout name the same protocols
    in the same directories, and every key both state agrees (TR, TE, inversion times, flip angle,
    slice thickness also of 3D protocols, resolution, averages, acceleration, reference lines)."""
    _need(COMA / "Coma Science Group.exar1", COMA / "Coma Science Group.xml")
    tree = load(COMA / "Coma Science Group.exar1")
    printed = defaultdict(list)
    for protocol in load(COMA / "Coma Science Group.xml"):
        printed[protocol.name].append(protocol)
    assert len(tree) == 40 and sorted(p.name for p in tree) == sorted(
        p.name for ps in printed.values() for p in ps)
    compared = 0
    for name in printed:
        ours = [p for p in tree if p.name == name]
        assert len(ours) == len(printed[name]), name
        for exported, shown in zip(ours, printed[name], strict=True):  # same order for duplicates
            assert exported.location[-3:] == shown.location[-3:]  # Head / Coma / program
            for key in KEYS:
                a, b = exported.meta.get(key), shown.meta.get(key)
                if a is not None and b is not None:
                    assert _same(a, b), (name, key, a, b)
                    compared += 1
    assert compared > 300


def test_examcards_match_gt_examcard_parser_and_their_computed_values():
    """gt-examcard-parser's test expects two scans in DUAL_ECHO_EPI, the first with 215
    parameters; and in every ExamCard the TR the scanner computed for a scan agrees with the value
    its user set (where set, not SHORTEST)."""
    card = PUBLIC_DATA / "dicompare/DUAL_ECHO_EPI.ExamCard"
    abcd = PUBLIC_DATA / "abcd_protocols/ABCD  V_EMS_20OCT2016_RawData.ExamCard"
    _need(card, abcd)
    scans = load(card)
    assert [s.name for s in scans] == ["GE-SE EPI 2Sh 1Sl", "SE EPI 2Sh 1Sl"]
    assert len(scans[0].meta.vendor["parameters"]) == 215
    assert scans[0].meta.vendor["parameters"]["EX_ACQ_scan_mode"] == "MS"
    checked = 0
    for scan in load(abcd) + load(PUBLIC_DATA / "dicompare/SSdense_July2024.ExamCard"):
        parameters = scan.meta.vendor["parameters"]
        for name in ("EX_ACQ_ffe_rep_time", "EX_ACQ_se_rep_time"):
            if parameters.get(f"{name}_enum") == "USER_DEF" and "IF_act" in scan.meta.source.get(
                    "RepetitionTime", ""):
                assert scan.meta["RepetitionTime"] == pytest.approx(parameters[name] / 1000)
                checked += 1
    assert checked > 20


def test_lcmodel_control_equals_the_inputs_of_its_run():
    """LCModel ran test_lcm's control file and printed the inputs it read in its report: every
    printed input equals the namelist value (the licence key is not printed), and HZPPPM,
    DELTAT and NUNFIL give the standard keys."""
    _need(LCMODEL / "control.file", LCMODEL / "out_ref_build.ps")
    [protocol] = load(LCMODEL / "control.file")
    report = (LCMODEL / "out_ref_build.ps").read_text()
    printed = dict(re.findall(r"\{\((\w+)=(.*?)\)s\}lst", report))
    values = protocol.meta.vendor["namelists"]["lcmodl"]
    assert set(printed) == set(values) - {"key"}
    for name, text in printed.items():
        expected = text.strip("'") if text.startswith("'") else float(text)
        assert values[name] == expected, name
    assert protocol.meta["SpectrometerFrequency"] == [float(printed["hzpppm"])]
    assert protocol.meta["DwellTime"] == float(printed["deltat"])
    assert protocol.meta["NumberOfPoints"] == int(printed["nunfil"])


ORIENTATION = PUBLIC_DATA / "spec2nii_test_data/philips/spar_dcm_orientation_tests"


@pytest.mark.data
@pytest.mark.parametrize("scan", range(40, 49))
def test_philips_raw_data_storage_equals_the_spar_of_the_scan(scan):
    """The scan parameters in Raw Data Storage: TR and TE as the scanner displays them (ms,
    rounded) and the flip angle against the SPAR the scanner exported for the same scan."""
    from spec2nii.Philips.philips import read_spar

    raw = sorted(ORIENTATION.glob(f"{scan}02-*/RAW*.dcm"))[0]  # the folder holds two copies
    [protocol] = load(raw)
    spar = read_spar(ORIENTATION / f"dbiex_{scan}_2_raw_act.spar")
    assert protocol.meta["RepetitionTime"] * 1000 == pytest.approx(spar["repetition_time"], abs=1)
    assert protocol.meta["EchoTime"] * 1000 == pytest.approx(spar["echo_time"], abs=1)
    assert protocol.meta["ExcitationFlipAngle"] == pytest.approx(90.0)
    assert "SliceThickness" not in protocol.meta
    assert "MultibandAccelerationFactor" not in protocol.meta


@pytest.mark.data
def test_philips_exam_log_holds_its_examcard():
    """An exam log in Raw Data Storage: the scans of its ExamCard, each with its log entry; the
    multiband factor only where multiband is on."""
    [raw] = (PUBLIC_DATA / "dcm_qa_philips_enh/In/enh").glob("*ExamCard_Raw.dcm*")
    protocols = load(raw)
    assert len(protocols) == 31
    assert all(p.meta.vendor["exam_log"]["Name"] == p.name for p in protocols
               if "exam_log" in p.meta.vendor)
    dti = next(p for p in protocols if p.name == "DTI_S2MB2")
    assert dti.meta["MultibandAccelerationFactor"] == 2
    assert all(p.meta.vendor["parameters"].get("EX_MB_multiband") == "YES"
               for p in protocols if "MultibandAccelerationFactor" in p.meta)


def _header(ds) -> tuple:
    """TR, TE (ms) and flip angle of an image header, classic or enhanced."""
    def get(keyword):
        if keyword in ds:
            return ds.get(keyword)
        for groups in ("SharedFunctionalGroupsSequence", "PerFrameFunctionalGroupsSequence"):
            for item in ds.get(groups, [{}])[0] if groups in ds else []:
                if item.VR == "SQ" and len(item.value) and keyword in item.value[0]:
                    return item.value[0].get(keyword)
        return None
    return (get("RepetitionTime"), get("EchoTime") or get("EffectiveEchoTime"), get("FlipAngle"))


@pytest.mark.data
def test_siemens_dicom_protocols_equal_their_images():
    """The protocol in a PhoenixZIPReport and in XA Raw Data Storage against an image of the same
    protocol in the same set: TR, TE and flip angle."""
    import pydicom

    from miox.sniff import open_dicom

    files = (sorted(PUBLIC_DATA.glob("dcm_qa*/In/**/*.SR"))
             + sorted(PUBLIC_DATA.glob("dcm_qa_xa30/In/**/*_Raw.dcm"))
             + sorted(PUBLIC_DATA.glob("dcm_qa_xa30/In/99_Phoenix_Document/*.dcm")))
    compared = 0
    for path in files:
        [protocol] = load(path)
        for image in sorted(path.parent.parent.rglob("*")):
            if (not image.is_file() or image.suffix in (".SR", ".json", ".nii")
                    or "_Raw" in image.name):
                continue
            try:
                with open_dicom(image) as fh:
                    ds = pydicom.dcmread(fh, stop_before_pixels=True)
            except pydicom.errors.InvalidDicomError:
                continue
            if str(ds.get("ProtocolName")) != protocol.name or "Image" not in ds.SOPClassUID.name:
                continue
            tr, te, flip = (float(v) for v in _header(ds))
            assert protocol.meta["RepetitionTime"] * 1000 == pytest.approx(tr, abs=0.5)
            echoes = protocol.meta["EchoTime"]
            echoes = echoes if isinstance(echoes, list) else [echoes]  # one image: one echo
            assert any(abs(e * 1000 - te) <= 0.05 for e in echoes)
            assert protocol.meta["ExcitationFlipAngle"] == pytest.approx(flip)
            compared += 1
            break
    assert compared >= 25
