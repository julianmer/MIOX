####################################################################################################
#                                        test_protocols.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/10/26                                                                                #
#                                                                                                  #
# Purpose: Protocol files on synthetic files: a Siemens single protocol (XProtocol with ASCCONV),  #
#          exam tree database (.exar1) and printed protocols as XML and as text, and a Philips     #
#          ExamCard; LCModel control and MakeBasis input files.                                    #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import base64
import json
import sqlite3
import struct
import uuid
import zlib

import pytest

# own
import miox

ASCCONV = """<XProtocol>
{
  <ParamLong."Dummy">  { 1  }
}
### ASCCONV BEGIN object=MrProtDataImpl@MrProtocolData version=66110001 ###
tProtocolName	 = 	"t1 = mprage"
tSequenceFileName	 = 	"%SiemensSeq%\\tfl"
alTR[0]	 = 	2300000
alTE[0]	 = 	2980
alTI[0]	 = 	900000
alTI[1]	 = 	1000000
lInvContrasts	 = 	1
sPrepPulses.ucInversion	 = 	0x2
adFlipAngleDegree[0]	 = 	9.0
sKSpace.lBaseResolution	 = 	256
sKSpace.ucDimension	 = 	0x4
sKSpace.lImagesPerSlab	 = 	176
sSliceArray.asSlice[0].dThickness	 = 	176.0
lAverages	 = 	1
sProtConsistencyInfo.flNominalB0	 = 	2.89362  # comment
### ASCCONV END ###
"""


def _protocol(**changes) -> str:
    text = ASCCONV
    for key, value in changes.items():
        text = "\n".join(f"{key}\t = \t{value}" if line.startswith(key + "\t") else line
                         for line in text.splitlines())
    return text


def test_siemens_pro_file(tmp_path):
    (tmp_path / "mprage.pro").write_text(ASCCONV)
    sniff = miox.sniff(tmp_path / "mprage.pro")
    assert (sniff.format, sniff.kind) == (miox.Format.SIEMENS_PRO, miox.Kind.PROTOCOL)
    [protocol] = miox.load(tmp_path / "mprage.pro")
    assert isinstance(protocol, miox.Protocol) and protocol.name == "t1 = mprage"
    m = protocol.meta
    assert m["RepetitionTime"] == 2.3 and m["EchoTime"] == pytest.approx(0.00298)
    assert m["InversionTime"] == 0.9 and m["ExcitationFlipAngle"] == 9.0
    assert m["SliceThickness"] == 1.0  # 3D: the slab per image
    assert m["FieldStrength"] == 2.89362 and m["PulseSequenceName"] == "%SiemensSeq%\\tfl"
    assert m.vendor["ascconv"]["sPrepPulses.ucInversion"] == 2
    assert m.vendor["xprotocol"].startswith("<XProtocol>")
    # inversion off: no inversion time; two inversion contrasts: both times
    (tmp_path / "gre.pro").write_text(_protocol(**{"sPrepPulses.ucInversion": "0x4"}))
    assert "InversionTime" not in miox.load(tmp_path / "gre.pro")[0].meta
    (tmp_path / "mp2rage.pro").write_text(_protocol(lInvContrasts="2"))
    assert miox.load(tmp_path / "mp2rage.pro")[0].meta["InversionTime"] == [0.9, 1.0]


def _content(kind: str, body: dict) -> bytes:
    text = f"EDF V1: ContentType=syngo.MR.ExamDataFoundation.Data.{kind};\n{json.dumps(body)}"
    compressor = zlib.compressobj(wbits=-15)
    return compressor.compress(text.encode()) + compressor.flush()


def test_siemens_exam_tree(tmp_path):
    path = tmp_path / "exam.exar1"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE Content (Hash TEXT, Data BLOB, Format TEXT)")
    db.execute("CREATE TABLE ChangeSet (Id TEXT, Author TEXT, DateTime TEXT)")
    db.execute("CREATE TABLE InstanceChangeSet (InstanceId TEXT, ChangeSetId TEXT, "
               "ElementId TEXT, State INTEGER)")
    db.execute("CREATE TABLE Instance (Id TEXT, Element_id TEXT, ContentHash TEXT, Children BLOB, "
               "Tags TEXT, ParentElementId TEXT, LabelElement_id TEXT, DescriptionElement_id "
               "TEXT, InstanceType TEXT, ObjectId TEXT)")
    db.execute("INSERT INTO ChangeSet VALUES ('c1', 'me', '2025-01-01 10:00'), "
               "('c2', 'me', '2025-01-02 10:00')")
    counter = iter(range(1000))

    def add(kind, content=None, children=(), label=None, element=None, obj=None, change="c2"):
        element = element or str(uuid.uuid4())
        instance, digest = str(uuid.uuid4()), f"h{next(counter)}"
        if content is not None:
            db.execute("INSERT INTO Content VALUES (?, ?, 'DS')", (digest, _content(kind, content)))
        blob = b"".join(uuid.UUID(c).bytes_le for c in children)
        db.execute("INSERT INTO Instance VALUES (?, ?, ?, ?, '', NULL, ?, NULL, ?, ?)",
                   (instance, element, digest, blob, label, f"Edf{kind}", obj or str(uuid.uuid4())))
        db.execute("INSERT INTO InstanceChangeSet VALUES (?, ?, ?, 0)", (instance, change, element))
        return element

    def text(value):
        return add("String", {"Texts": {"en": value}})

    protocol = add("Protocol", {"Data": ASCCONV})
    old = add("Protocol", {"Data": _protocol(**{"alTR[0]": "1000"})}, change="c1")
    step = add("MeasurementStep", {"BodyPartExamined": ""}, [protocol], text("MPRAGE"))
    pause = add("PauseStep", {}, [], text("Give contrast"))
    second = add("MeasurementStep", {}, [old], text("Old"))
    program = add("Program", {}, [step, pause, second], text("Brain"))
    folder_object = str(uuid.uuid4())
    add("Directory", {}, [], text("Neuro"), obj=folder_object)
    add("Structure", {"ParentDirectoryId": {"$id": "1", program: folder_object}})
    db.commit()
    assert miox.sniff(path).format == miox.Format.SIEMENS_EXAR
    first, older = miox.load(path)
    assert (first.name, first.location) == ("MPRAGE", ("Neuro", "Brain"))
    assert first.meta["RepetitionTime"] == 2.3 and older.meta["RepetitionTime"] == 0.001
    steps = first.meta.vendor["program"]["steps"]
    assert steps == [{"type": "MeasurementStep", "label": "MPRAGE"},
                     {"type": "PauseStep", "label": "Give contrast"},
                     {"type": "MeasurementStep", "label": "Old"}]


PRINTED_XML = """<?xml version="1.0"?>
<PrintOut><PrintProtocol><Protocol><HeaderTitle>SIEMENS MAGNETOM</HeaderTitle><SubStep>
<ProtHeaderInfo><HeaderProtPath>\\\\User\\Head\\Exam\\Brain\\mp2rage *</HeaderProtPath>
<HeaderProperty>TA: 5:00</HeaderProperty></ProtHeaderInfo>
<Card name="Routine">
<ProtParameter><Label>TR</Label><ValueAndUnit>5000.0 ms</ValueAndUnit></ProtParameter>
<ProtParameter><Label>TE</Label><ValueAndUnit>2.98 ms</ValueAndUnit></ProtParameter>
<ProtParameter><Label>Slice Thickness</Label><ValueAndUnit>1.0 mm</ValueAndUnit></ProtParameter>
</Card><Card name="Contrast - Common">
<ProtParameter><Label>TR</Label><ValueAndUnit>5000.0 ms</ValueAndUnit></ProtParameter>
<ProtParameter><Label>TI 1</Label><ValueAndUnit>700 ms</ValueAndUnit></ProtParameter>
<ProtParameter><Label>TI 2</Label><ValueAndUnit>2500 ms</ValueAndUnit></ProtParameter>
<ProtParameter><Label>Flip Angle</Label><ValueAndUnit>4 deg</ValueAndUnit></ProtParameter>
<ProtParameter><Label>Base Resolution</Label><ValueAndUnit>256  </ValueAndUnit></ProtParameter>
</Card></SubStep></Protocol></PrintProtocol></PrintOut>
"""

PRINTED_TEXT = (
    "\r\r\r\n                   SIEMENS MAGNETOM Skyra syngo MR E11\r\r\r\n"
    + "-" * 60 + "\r\r\r\n              \\\\USER\\site\\exam\\gre_fm\r\r\r\n"
    "          TA:0:57  PAT:Off\r\r\r\n" + "-" * 60 + "\r\r\r\nRoutine\r\r\r\n"
    "     TR                                      427.0 ms\r\r\r\n"
    "     TE 1                                    4.92 ms\r\r\r\n"
    "     TE 2                                    7.38 ms\r\r\r\n"
    "     Averages                                1 \r\r\r\n"
    "     Before measurement                       \r\r\r\n"
    "Contrast\r\r\r\n     Flip angle                              60 deg\r\r\r\n"
)


def test_siemens_printed_protocols(tmp_path):
    (tmp_path / "exam.xml").write_text(PRINTED_XML)
    assert miox.sniff(tmp_path / "exam.xml").format == miox.Format.SIEMENS_PRINTPROT
    [protocol] = miox.load(tmp_path / "exam.xml")
    assert (protocol.name, protocol.location) == ("mp2rage", ("User", "Head", "Exam", "Brain"))
    m = protocol.meta
    assert m["RepetitionTime"] == 5.0 and m["EchoTime"] == 0.00298
    assert m["InversionTime"] == [0.7, 2.5] and m["ExcitationFlipAngle"] == 4.0
    assert m["BaseResolution"] == 256 and m["SliceThickness"] == 1.0
    assert m.vendor["path"].endswith("mp2rage *") and m.vendor["header"] == "TA: 5:00"
    (tmp_path / "exam.txt").write_bytes(PRINTED_TEXT.encode())
    assert miox.sniff(tmp_path / "exam.txt").format == miox.Format.SIEMENS_PRINTPROT
    [text] = miox.load(tmp_path / "exam.txt")
    assert (text.name, text.location) == ("gre_fm", ("USER", "site", "exam"))
    assert text.meta["EchoTime"] == [0.00492, 0.00738] and text.meta["NumberOfAverages"] == 1
    assert text.meta["ExcitationFlipAngle"] == 60.0
    assert ("Before measurement", "") in text.meta.vendor["cards"]["Routine"]


def _parameter_table(parameters: list[tuple]) -> bytes:
    """An ExamCard parameter table of (name, kind, values[, choices]) entries."""
    entries, data = [], b""
    start = 32 + 50 * len(parameters)
    for name, kind, values, *rest in parameters:
        base = 32 + 50 * len(entries)
        choices_at = start + len(data)
        if kind == 4:
            data += ",".join(rest[0]).encode() + b"\0"
        values_at = start + len(data)
        if kind == 0:
            data += struct.pack(f"<{len(values)}f", *values)
        elif kind == 1 or kind == 4:
            data += struct.pack(f"<{len(values)}i", *values)
        else:
            data += b"".join(v.encode().ljust(81, b"\0") for v in values)
        entry = name.encode().ljust(34, b"\0") + struct.pack(
            "<4I", kind, len(values), choices_at - (base + 42), values_at - (base + 46))
        entries.append(entry)
    header = struct.pack("<8I", 0, 0, 0, 0, 0, 0, 8, len(parameters))
    return header + b"".join(entries) + data


def test_philips_examcard(tmp_path):
    table = _parameter_table([
        ("EX_ACQ_nucleus", 4, [0], ["H1", "P31"]),
        ("EX_ACQ_flip_angle", 0, [52.0]),
        ("EX_ACQ_ffe_rep_time", 0, [800.0]),
        ("EX_ACQ_ffe_rep_time_enum", 4, [1], ["SHORTEST", "USER_DEF"]),
        ("EX_ACQ_first_echo_time", 0, [30.0]),
        ("EX_GEO_slice_thickness", 0, [2.4]),
        ("EX_GEO_sense_enable", 4, [0], ["YES", "NO"]),
        ("EX_GEO_sense_p_red_factor", 0, [1.5]),
        ("EX_MB_multiband", 4, [1], ["NO", "YES"]),
        ("EX_MB_multiband_factor", 1, [6]),
        ("IF_act_rep_time_echo_time", 2, ["801 / 31"]),
    ])
    xml = f"""<SOAP-ENV:Envelope xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/"
 xmlns:SOAP-ENC="http://schemas.xmlsoap.org/soap/encoding/"><SOAP-ENV:Body>
<a1:ExamCard id="ref-1" xmlns:a1="http://schemas.microsoft.com/clr/nsassem/Philips.PmsMR.ExamCards.ECModel/x">
<name id="ref-2">Brain</name><patientOrientation>PatientPos_HFS</patientOrientation></a1:ExamCard>
<a1:ScanProcedure id="ref-3" xmlns:a1="http://schemas.microsoft.com/clr/nsassem/Philips.PmsMR.Acquisition/x">
<name id="ref-4">rsfMRI</name><parameterData href="#ref-5"/></a1:ScanProcedure>
<SOAP-ENC:Array id="ref-5">{base64.b64encode(table).decode()}</SOAP-ENC:Array>
</SOAP-ENV:Body></SOAP-ENV:Envelope>"""
    (tmp_path / "brain.ExamCard").write_bytes(xml.encode() + b"\0" * 64)  # NUL padding
    assert miox.sniff(tmp_path / "brain.ExamCard").format == miox.Format.PHILIPS_EXAMCARD
    [scan] = miox.load(tmp_path / "brain.ExamCard")
    assert (scan.name, scan.location) == ("rsfMRI", ("Brain",))
    m = scan.meta
    assert m["RepetitionTime"] == 0.801 and m["EchoTime"] == 0.031  # the computed values
    assert m["ExcitationFlipAngle"] == 52.0 and m["SliceThickness"] == 2.4
    assert m["ParallelReductionFactorInPlane"] == 1.5 and m["MultibandAccelerationFactor"] == 6
    assert m["ResonantNucleus"] == "1H" and m["PatientPosition"] == "HFS"
    assert m.vendor["parameters"]["EX_ACQ_ffe_rep_time_enum"] == "USER_DEF"
    assert m.vendor["choices"]["EX_ACQ_nucleus"] == ["H1", "P31"]


def test_lcmodel_control(tmp_path):
    path = tmp_path / "fit.control"
    path.write_text(
        " &LCMODL\n title='PRESS TE 30', hzpppm=1.2325D+02, deltat=2.5e-04, nunfil=2048,\n"
        " echot=30., chomit(1)='Lip13a', chomit(2)='MM09', shifmx= 0.3 0.3, dows=T\n /\n"
    )
    [protocol] = miox.load(path)
    assert protocol.name == "fit"
    values = protocol.meta.vendor["namelists"]["lcmodl"]
    assert values["chomit"] == ["Lip13a", "MM09"]
    assert values["shifmx"] == [0.3, 0.3]
    assert values["dows"] is True
    assert protocol.meta["SpectrometerFrequency"] == [123.25]
    assert protocol.meta["DwellTime"] == 2.5e-4
    assert protocol.meta["EchoTime"] == pytest.approx(0.03)
    assert protocol.meta["NumberOfPoints"] == 2048


def test_makebasis_input(tmp_path):
    path = tmp_path / "makebasis.in"
    path.write_text(
        " $SEQPAR\n hzpppm=297.2, echot=8.5, seq='SPECIAL'\n $END\n"
        " $NMALL\n badelt=1.25e-4, ndatab=8192, filbas='press.basis'\n $END\n"
        " $NMEACH\n filraw='naa.raw', metabo='NAA'\n $END\n"
        " $NMEACH\n filraw='cr.raw', metabo='Cr'\n $END\n"
    )
    [protocol] = miox.load(path)
    namelists = protocol.meta.vendor["namelists"]
    assert [m["metabo"] for m in namelists["nmeach"]] == ["NAA", "Cr"]
    assert namelists["nmall"]["filbas"] == "press.basis"
    assert protocol.meta["SpectrometerFrequency"] == [297.2]
    assert protocol.meta["EchoTime"] == pytest.approx(0.0085)
    assert "DwellTime" not in protocol.meta
