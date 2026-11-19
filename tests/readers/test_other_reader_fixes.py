####################################################################################################
#                                   test_other_reader_fixes.py                                     #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 19/11/26                                                                                #
#                                                                                                  #
# Purpose: Edge cases of the smaller readers on synthetic files: R's special items in spant data,  #
#          Siemens protocols with CR LF line ends, PMU logs with an unopened text section, physio  #
#          tables that are no numbers, JCAMP-DX FIDs without imaginary page, ExamCards with an     #
#          unresolved reference, LCModel overflow cells, ITK rigid transforms, TopSpin qseq FIDs   #
#          and reports printed with a decimal comma.                                               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import base64
import struct
from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import FileMetaDataset
from scipy.spatial.transform import Rotation

# own
import miox
from miox.meta import Meta
from miox.readers.base import ReaderError
from miox.readers.jcamp import JcampReader
from miox.readers.lcmodel_output import LcmodelOutputReader
from miox.readers.minispec import Report
from miox.readers.philips_examcard import PhilipsExamCardReader
from miox.readers.physio import BidsPhysioReader, GePhysioReader, HcpPhysioReader
from miox.readers.siemens_dicom_protocol import SiemensDicomProtocolReader
from miox.readers.siemens_protocol import SiemensPrintedProtocolReader, SiemensProtocolReader
from miox.readers.spant_rds import RSerial
from miox.sniff import Format, Kind, Sniff

PMU = Path(__file__).parents[1] / "data/bidsphysio/bidsphysio.pmu2bids/tests/data"


#**************************************************************************************************#
#                                            spant .rds                                            #
#**************************************************************************************************#
def r_stream(*items: bytes) -> bytes:
    """An XDR stream of serialization version 2 holding a list of ``items``."""
    head = b"X\n" + struct.pack(">3i", 2, 0x040601, 0x020300)
    return head + struct.pack(">2i", 19, len(items)) + b"".join(items)


def r_strings(*texts: str) -> bytes:
    """A string vector as the special items carry it: a zero, the count, the strings."""
    strings = b"".join(struct.pack(">2i", 0x40009, len(t)) + t.encode() for t in texts)
    return struct.pack(">2i", 0, len(texts)) + strings


INTEGER_5 = struct.pack(">3i", 13, 1, 5)


def test_r_base_namespace_has_no_payload():
    """list(.BaseNamespaceEnv, 5L) as R 4 writes it: item 250 stands alone."""
    namespace, five = RSerial(r_stream(struct.pack(">i", 250), INTEGER_5)).read().value
    assert namespace.kind == "environment" and five.value.tolist() == [5]


@pytest.mark.parametrize("code, texts", [(249, ("stats", "4.6.1")), (248, ("package:stats",)),
                                         (247, ("name",))])
def test_r_namespace_package_and_persistent_name_carry_strings(code, texts):
    """Items 247 to 249 are followed by a string vector and enter the reference table: the
    reference after them (index 1 packed into its flags) is the same item."""
    stream = r_stream(struct.pack(">i", code) + r_strings(*texts), struct.pack(">i", 0x1FF),
                      INTEGER_5)
    item, again, five = RSerial(stream).read().value
    assert item.value == list(texts) and again is item and five.value.tolist() == [5]


def test_r_environments_without_payload():
    codes = (241, 242, 253, 250)  # base, empty, global environment, base namespace
    items = RSerial(r_stream(*(struct.pack(">i", c) for c in codes), INTEGER_5)).read().value
    assert [i.kind for i in items[:4]] == ["environment"] * 4 and items[4].value.tolist() == [5]


#**************************************************************************************************#
#                                        Siemens protocols                                         #
#**************************************************************************************************#
ASCCONV = ("<XProtocol>\n{\n}\n### ASCCONV BEGIN object=MrProtDataImpl ###  \n"
           'tProtocolName\t = \t"gre"\nalTR[0]\t = \t2300000\n### ASCCONV END ###\n')


def test_ascconv_with_cr_lf_line_ends():
    for text in (ASCCONV, ASCCONV.replace("\n", "\r\n"), ASCCONV.replace("\n", "\r")):
        assert SiemensProtocolReader.ascconv(text) == {"tProtocolName": "gre", "alTR[0]": 2300000}


def test_dicom_protocol_with_cr_lf_line_ends(tmp_path):
    ds = pydicom.Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.RawDataStorage
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    ds.add_new((0x0021, 0x0010), "LO", "SIEMENS MR MRS 05")
    ds.add_new((0x0021, 0x1019), "OB", ASCCONV.replace("\n", "\r\n").encode("latin-1"))
    path = tmp_path / "raw.dcm"
    ds.save_as(path, enforce_file_format=True)
    reader = SiemensDicomProtocolReader()
    [protocol] = reader.read(Sniff(path, Format.SIEMENS_DICOM_RAW, Kind.PROTOCOL))
    assert protocol.name == "gre" and protocol.meta["RepetitionTime"] == 2.3
    assert protocol.provenance.notes == ()
    ds[0x0021, 0x1019].value = b"### ASCCONV BEGIN ###\r\n### ASCCONV END ###\r\n"
    ds.save_as(path, enforce_file_format=True)
    [empty] = reader.read(Sniff(path, Format.SIEMENS_DICOM_RAW, Kind.PROTOCOL))
    assert empty.provenance.notes == ("ASCCONV block without readable values",)


def test_printed_protocol_values_with_decimal_comma_and_exponent():
    cards = {"Routine": [("TR", "427,5 ms"), ("TE", "4.92e0 ms"), ("Flip angle", "6,0E1 deg"),
                         ("Averages", "2"), ("Coil Elements", "HE1-4;NE1,2"),
                         ("Slice thickness", "1,2,3 mm")]}
    meta = Meta({})
    SiemensPrintedProtocolReader()._keys(meta, cards)
    assert meta["RepetitionTime"] == pytest.approx(0.4275)
    assert meta["EchoTime"] == pytest.approx(0.00492)
    assert meta["ExcitationFlipAngle"] == 60 and meta["NumberOfAverages"] == 2
    assert "SliceThickness" not in meta  # a list, not one number


#**************************************************************************************************#
#                                               physio                                             #
#**************************************************************************************************#
def test_pmu_section_end_without_start(tmp_path):
    """The samples end before the words of a section that 6002 closes but 5002 never opened;
    15002 and 6002 inside longer numbers are samples."""
    path = tmp_path / "log.puls"
    path.write_text("1 2 40 280 5002 PULS_SAMPLES_PER_SECOND = 50 6002 15002 26002 5000 30 \n"
                    "ACQ FINISHED\n 6002 40 50 5000 60 5003\nLogStartMDHTime: 1000\n6003\n")
    [d] = miox.load(path)
    assert d.data[:, 0].tolist() == [15002, 26002, 30] and d.events["trigger"].tolist() == [2]
    assert d.meta.vendor["text_sections"] == ["PULS_SAMPLES_PER_SECOND = 50", "ACQ FINISHED"]
    assert d.meta.vendor["values_after_unopened_section"] == [40, 50, 5000, 60]
    assert any("6002 without its start 5002" in n for n in d.provenance.notes)
    assert not any("words outside" in n for n in d.provenance.notes)


@pytest.mark.data
def test_pmu_sample_holds_the_samples_of_its_reference():
    if not (PMU / "sample_VBX.puls").is_file():
        pytest.skip("bidsphysio samples absent")
    [d] = miox.load(PMU / "sample_VBX.puls")
    assert d.data[:, 0].tolist() == np.loadtxt(PMU / "pmu_VBX_cardiac.tsv").tolist()


def test_physio_tables_that_are_no_numbers(tmp_path):
    tsv = tmp_path / "sub-1_physio.tsv"
    (tmp_path / "sub-1_physio.json").write_text('{"Columns": ["cardiac", "trigger"]}')
    tsv.write_text("cardiac\ttrigger\n1.5\t0\n2.5\t1\n")
    sniff = Sniff(tsv, Format.BIDS_PHYSIO, Kind.PHYSIO)
    with pytest.raises(ReaderError, match=r"sub-1_physio\.tsv: starts with a header line"):
        BidsPhysioReader().read(sniff)
    tsv.write_text("1.5\t0\n2.5\tx\n")
    with pytest.raises(ReaderError, match=r"sub-1_physio\.tsv: not a table of numbers"):
        BidsPhysioReader().read(sniff)
    tsv.write_text("")
    with pytest.raises(ReaderError, match=r"sub-1_physio\.tsv: no samples"):
        BidsPhysioReader().read(sniff)
    ge = tmp_path / "PPGData_epiRT_0000000000_00_00_000"
    ge.write_text("1\ntwo\n3\n")
    with pytest.raises(ReaderError, match="PPGData_epiRT"):
        GePhysioReader().read(Sniff(ge, Format.GE_PHYSIO, Kind.PHYSIO))
    hcp = tmp_path / "rest_Physio_log.txt"
    hcp.write_text("1\t1900\t1750\n0\tx\t1749\n")
    with pytest.raises(ReaderError, match="rest_Physio_log.txt"):
        HcpPhysioReader().read(Sniff(hcp, Format.HCP_PHYSIO, Kind.PHYSIO))


#**************************************************************************************************#
#                                             JCAMP-DX                                             #
#**************************************************************************************************#
JCAMP = """##TITLE= test
##JCAMP-DX= 5.01
##DATA TYPE= {kind}
##DATA CLASS= {cls}
##.OBSERVE FREQUENCY= 100.0
##.OBSERVE NUCLEUS= ^1H
"""


def test_jcamp_fid_without_imaginary_page(tmp_path):
    path = tmp_path / "fid.jdx"
    path.write_text(JCAMP.format(kind="NMR FID", cls="NTUPLES") + """##NTUPLES= NMR FID
##VAR_NAME= TIME, FID/REAL
##SYMBOL= X, R
##VAR_DIM= 4, 4
##UNITS= SECONDS, ARBITRARY UNITS
##FACTOR= 0.001, 2
##FIRST= 0, 20
##LAST= 0.003, 2
##PAGE= N=1
##DATA TABLE= (X++(R..R)), XYDATA
0 10 8 6 4
##END NTUPLES= NMR FID
##END=
""")
    [d] = JcampReader().read(Sniff(path, Format.JCAMP_NMR, Kind.SPECTROSCOPY))
    assert np.array_equal(d.data[0, 0, 0], [20, 16, 12, 8]) and np.iscomplexobj(d.data)
    assert d.dwelltime == pytest.approx(1e-3)
    assert "FID without imaginary page: real samples, stored as the real part" in (
        d.provenance.notes)


def test_jcamp_xydata_without_firstx(tmp_path):
    path = tmp_path / "spectrum.dx"
    path.write_text(JCAMP.format(kind="NMR SPECTRUM", cls="XYDATA") + """##XUNITS= HZ
##LASTX= 300
##NPOINTS= 4
##XYDATA= (X++(Y..Y))
0 1 5 1 0
##END=
""")
    with pytest.raises(ReaderError, match="XYDATA without FIRSTX$"):
        JcampReader().read(Sniff(path, Format.JCAMP_NMR, Kind.SPECTROSCOPY))


#**************************************************************************************************#
#                                        Philips ExamCard                                          #
#**************************************************************************************************#
def test_examcard_unresolved_reference_is_skipped(tmp_path):
    table = struct.pack("<8I", 0, 0, 0, 0, 0, 0, 8, 0)
    acquisition = "http://schemas.microsoft.com/clr/nsassem/Philips.PmsMR.Acquisition/x"
    path = tmp_path / "card.ExamCard"
    path.write_text(f"""<SOAP-ENV:Envelope
 xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/"
 xmlns:SOAP-ENC="http://schemas.xmlsoap.org/soap/encoding/"><SOAP-ENV:Body>
<a1:ScanProcedure id="ref-3" xmlns:a1="{acquisition}">
<name id="ref-4">survey</name><parameterData href="#ref-5"/></a1:ScanProcedure>
<a1:ScanProcedure id="ref-6" xmlns:a1="{acquisition}">
<name id="ref-7">lost</name><parameterData href="#ref-9"/></a1:ScanProcedure>
<SOAP-ENC:Array id="ref-5">{base64.b64encode(table).decode()}</SOAP-ENC:Array>
</SOAP-ENV:Body></SOAP-ENV:Envelope>""")
    sniff = Sniff(path, Format.PHILIPS_EXAMCARD, Kind.PROTOCOL)
    [protocol] = PhilipsExamCardReader().read(sniff)
    assert protocol.name == "survey"
    assert protocol.provenance.notes == (
        "scans whose parameter table is referred to but not in the file, not read: "
        "lost (#ref-9)",)


#**************************************************************************************************#
#                                             LCModel                                              #
#**************************************************************************************************#
def test_lcmodel_overflow_and_touching_cells(tmp_path):
    path = tmp_path / "fit.TABLE"
    path.write_text("""
$$CONC  5 lines in following concentration table = NCONC+1
    Conc.  %SD /Cr+PCr  Metabolite
    0.500   4%   1.000 Cr
********* 999%******** Lac
 1.20E+04   3%2.40E+04 NAA
    0.100  12% 1.2.3.4 Glu
""")
    [r] = LcmodelOutputReader().read(Sniff(path, Format.LCMODEL_TABLE, Kind.FIT_RESULT))
    assert r.row("Cr") == {"metabolite": "Cr", "concentration": 0.5, "sd_percent": 4.0,
                           "/Cr+PCr": 1.0}
    assert r.row("Lac") == {"metabolite": "Lac", "concentration": None, "sd_percent": 999.0,
                            "/Cr+PCr": None}
    assert r.row("NAA")["/Cr+PCr"] == 24000 and r.row("NAA")["sd_percent"] == 3
    assert r.row("Glu")["/Cr+PCr"] is None
    [note] = [n for n in r.provenance.notes if "no number" in n]
    assert note.endswith("Lac Conc. (*********), Lac /Cr+PCr (********), Glu /Cr+PCr (1.2.3.4)")


#**************************************************************************************************#
#                                          ITK transforms                                          #
#**************************************************************************************************#
def tfm(path: Path, *transforms: tuple[str, str, str]) -> Path:
    text = "#Insight Transform File V1.0\n"
    for i, (kind, parameters, fixed) in enumerate(transforms):
        text += (f"#Transform {i}\nTransform: {kind}\nParameters: {parameters}\n"
                 f"FixedParameters: {fixed}\n")
    path.write_text(text)
    return path


def test_itk_skipped_transforms_are_named_on_every_dataset(tmp_path):
    identity = "1 0 0 0 1 0 0 0 1 0 0 0"
    path = tfm(tmp_path / "composite.tfm", ("AffineTransform_double_3_3", identity, "0 0 0"),
               ("BSplineTransform_double_3_3", "0 0 0", "0 0 0"),
               ("AffineTransform_double_3_3", identity, "0 0 0"),
               ("DisplacementFieldTransform_double_3_3", "0 0 0", "0 0 0"))
    first, second = miox.load(path)
    note = ("not affine, not read: BSplineTransform_double_3_3, "
            "DisplacementFieldTransform_double_3_3")
    assert first.provenance.notes == second.provenance.notes == (note,)


def test_itk_euler_angles(tmp_path):
    """A quarter turn about z around the centre (1, 0, 0), then each order of three angles
    against the rotations multiplied out: Rz Rx Ry, and Rz Ry Rx with the ComputeZYX flag."""
    half = np.pi / 2
    [t] = miox.load(tfm(tmp_path / "z.tfm", ("Euler3DTransform_double_3_3",
                                              f"0 0 {half!r} 1 2 3", "1 0 0")))
    assert np.allclose(t.matrix, [[0, -1, 0, 2], [1, 0, 0, 1], [0, 0, 1, 3], [0, 0, 0, 1]])
    assert t.meta.vendor["parameters"] == [0, 0, half, 1, 2, 3]
    [x] = miox.load(tfm(tmp_path / "x.tfm", ("Euler3DTransform_double_3_3",
                                              f"{half!r} 0 0 0 0 0", "0 0 0")))
    assert np.allclose(x.matrix[:3, :3], [[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    [y] = miox.load(tfm(tmp_path / "y.tfm", ("Euler3DTransform_double_3_3",
                                              f"0 {half!r} 0 0 0 0", "0 0 0")))
    assert np.allclose(y.matrix[:3, :3], [[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    ax, ay, az = 0.1, 0.2, 0.3
    for fixed, order, angles in (("0 0 0", "ZXY", (az, ax, ay)), ("0 0 0 0", "ZXY", (az, ax, ay)),
                                 ("0 0 0 1", "ZYX", (az, ay, ax))):
        [t] = miox.load(tfm(tmp_path / "e.tfm", ("Euler3DTransform_double_3_3",
                                                  f"{ax} {ay} {az} 0 0 0", fixed)))
        assert np.allclose(t.matrix[:3, :3], Rotation.from_euler(order, angles).as_matrix())


def test_itk_versor_and_similarity(tmp_path):
    """The versor (0, 0, sin 45 deg) is a quarter turn about z; a similarity scales it."""
    s = float(np.sin(np.pi / 4))
    turn = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    [t] = miox.load(tfm(tmp_path / "v.tfm", ("VersorRigid3DTransform_double_3_3",
                                              f"0 0 {s!r} 1 2 3", "1 0 0")))
    assert np.allclose(t.matrix, [[0, -1, 0, 2], [1, 0, 0, 1], [0, 0, 1, 3], [0, 0, 0, 1]])
    assert np.allclose(t.ras_matrix, [[0, -1, 0, -2], [1, 0, 0, -1], [0, 0, 1, 3], [0, 0, 0, 1]])
    [t] = miox.load(tfm(tmp_path / "s.tfm", ("Similarity3DTransform_double_3_3",
                                              f"0 0 {s!r} 0 0 0 2", "0 0 0")))
    assert np.allclose(t.matrix[:3, :3], 2 * turn)
    [t] = miox.load(tfm(tmp_path / "q.tfm", ("VersorRigid3DTransform_double_3_3",
                                              "0.1 0.2 0.3 0 0 0", "0 0 0")))
    quaternion = [0.1, 0.2, 0.3, np.sqrt(1 - 0.14)]
    assert np.allclose(t.matrix[:3, :3], Rotation.from_quat(quaternion).as_matrix())
    [t] = miox.load(tfm(tmp_path / "r.tfm", ("Rigid3DTransform_double_3_3",
                                              "0 -1 0 1 0 0 0 0 1 1 2 3", "1 0 0")))
    assert np.allclose(t.matrix, [[0, -1, 0, 2], [1, 0, 0, 1], [0, 0, 1, 3], [0, 0, 0, 1]])


#**************************************************************************************************#
#                                             TopSpin                                              #
#**************************************************************************************************#
@pytest.mark.parametrize("mode, name", [(0, "qf"), (2, "qseq")])
def test_topspin_real_acquisitions_stay_real(tmp_path, mode, name):
    """qf and qseq FIDs hold TD real samples: each is one point, its imaginary part zero."""
    folder = tmp_path / "study" / "10"
    folder.mkdir(parents=True)
    acqus = {"TD": 8, "SW_h": 5000.0, "SFO1": 400.13, "BF1": 400.13, "NUC1": "<1H>", "AQ_mod": mode,
             "DTYPA": 0, "BYTORDA": 0, "NC": 0}
    lines = ["##TITLE= Parameter file, TopSpin 4.1.4", "##JCAMPDX= 5.0"]
    lines += [f"##${k}= {v}" for k, v in acqus.items()]
    (folder / "acqus").write_text("\n".join(lines + ["##END="]) + "\n")
    np.arange(1, 9, dtype="<i4").tofile(folder / "fid")
    [d] = miox.load(folder / "fid")
    assert np.array_equal(d.data[0, 0, 0], np.arange(1, 9))
    assert any(n.startswith(f"{name} acquisition: real samples, stored as the real part")
               for n in d.provenance.notes)


#**************************************************************************************************#
#                                             minispec                                             #
#**************************************************************************************************#
REPORT = """------------------------

T2 Curve Fit Results
06.11.2024 12:56
------------------------

------------------------
Pulse Separ.  : 1,000 ms
Data Points   : 5000
------------------------

------------------------
SampleId: IBU 7,5 ppm_1
------------------------
Title of this Data File:
IBU 7,5 ppm_1_20241106125903
------------------------
Amplitude  :   74,576  %
         +/-    0,007  %
T2         :   2.5223e3 ms
         +/-      6E-1 ms
Offset     :    -0,079  %
         +/-    0.004  %
------------------------

CONTIN Results:
'C:\\data\\IBU 7,5 ppm_1_20241106125903.dps'
T2[1]      :   2511,2 +/- 12,5 ms
A [1]      :   1,5e2
Integral   :   150,25
Ampl[1]    :   10,5 +/- 0,5 %
"""


def test_minispec_report_with_decimal_commas_and_exponents(tmp_path):
    path = tmp_path / "06125112.mdt"
    path.write_text(REPORT)
    fit, contin = Report(path).blocks
    assert fit["sample"] == "IBU 7,5 ppm_1" and fit["title"] == "IBU 7,5 ppm_1_20241106125903"
    assert fit["values"] == {"Amplitude": (74.576, "%", 0.007), "T2": (2522.3, "ms", 0.6),
                             "Offset": (-0.079, "%", 0.004)}
    assert Report.pulse_separation(fit["settings"]) == pytest.approx(1e-3)
    assert contin["components"] == [(pytest.approx(2.5112), pytest.approx(0.0125), 150.0)]
    assert contin["integral"] == 150.25 and contin["gauss"] == {"ampl[1]": (10.5, 0.5, "%")}
