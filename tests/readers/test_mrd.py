####################################################################################################
#                                           test_mrd.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/10/26                                                                                #
#                                                                                                  #
# Purpose: Yardl decoding (the binary and NDJSON examples of the yardl documentation, varint       #
#          arrays, unions, enums and flags) and MRD streams (acquisitions with waveforms, images,  #
#          Pulseq items, NDJSON) on synthetic files written by a small encoder.                    #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import json
import struct

import numpy as np
import pytest

# own
import miox as gu
from miox.yardl import YardlError, YardlFile, YardlSchema


#**************************************************************************************************#
#                                           YardlWriter                                            #
#**************************************************************************************************#
#                                                                                                  #
# The binary encoding written for tests: the inverse of the decoder, values as the decoder         #
# returns them (records as dicts, tagged unions as (tag, value), enums as symbols, arrays as numpy #
# arrays); a stream is written as one block.                                                       #
#                                                                                                  #
#**************************************************************************************************#
class YardlWriter:
    FIXED = {"bool": "?", "int8": "b", "uint8": "B", "float32": "<f", "float64": "<d"}

    def __init__(self, schema: dict):
        self.schema = YardlSchema(schema)
        self.text = json.dumps(schema).encode()

    def write(self, path, steps: dict) -> None:
        out = bytearray(b"yardl" + struct.pack("<I", 1))
        out += self.uvarint(len(self.text)) + self.text
        for name, node in self.schema.steps:
            out += self.value(self.schema.concrete(node), steps[name])
        path.write_bytes(bytes(out))

    @staticmethod
    def uvarint(value: int) -> bytes:
        out = bytearray()
        while True:
            byte, value = value & 0x7F, value >> 7
            out.append(byte | (0x80 if value else 0))
            if not value:
                return bytes(out)

    def value(self, node, value) -> bytes:
        if isinstance(node, str):
            return self.primitive(node, value)
        if isinstance(node, list):
            if len(node) == 2 and node[0] is None and node[1]["tag"] is None:
                return b"\x00" if value is None else b"\x01" + self.value(node[1]["type"], value)
            if value is None:
                return self.uvarint(node.index(None))
            tags = [None if c is None else c["tag"] for c in node]
            index = tags.index(value[0])
            return self.uvarint(index) + self.value(node[index]["type"], value[1])
        if "record" in node:
            return b"".join(self.value(t, value[name]) for name, t in node["fields"])
        if "enum" in node:
            numbers = {s: v for v, s in node["values"].items()}
            return self.primitive(node["base"], numbers.get(value, value))
        if "vector" in node:
            items, length = node["vector"]["items"], node["vector"].get("length")
            head = b"" if length is not None else self.uvarint(len(value))
            return head + b"".join(self.value(items, v) for v in value)
        if "array" in node:
            dims = node["array"].get("dimensions")
            array = np.asarray(value)
            head = b""
            if dims is None:
                head += self.uvarint(array.ndim)
            if not (isinstance(dims, list) and all("length" in d for d in dims)):
                head += b"".join(self.uvarint(n) for n in array.shape)
            items = node["array"]["items"]
            return head + b"".join(self.value(items, v) for v in array.ravel().tolist())
        if "map" in node:
            keys, values = node["map"]["keys"], node["map"]["values"]
            return self.uvarint(len(value)) + b"".join(
                self.value(keys, k) + self.value(values, v) for k, v in value.items())
        if "stream" in node:
            items = node["stream"]["items"]
            body = b"".join(self.value(items, v) for v in value)
            return (self.uvarint(len(value)) + body if value else b"") + b"\x00"
        raise ValueError(node)

    def primitive(self, name: str, value) -> bytes:
        if name in self.FIXED:
            return struct.pack(self.FIXED[name], value)
        if name.startswith("complex"):
            code = "<2f" if name == "complexfloat32" else "<2d"
            return struct.pack(code, value.real, value.imag)
        if name == "string":
            raw = value.encode()
            return self.uvarint(len(raw)) + raw
        if name == "date":
            value = int(value.astype("datetime64[D]").astype(np.int64))
        elif name in ("time", "datetime"):
            value = int(value.astype(np.int64))
        if name.startswith("uint") or name == "size":
            return self.uvarint(int(value))
        value = int(value)
        return self.uvarint(2 * value if value >= 0 else -2 * value - 1)


#**************************************************************************************************#
#                                        a small MRD schema                                        #
#**************************************************************************************************#
def _record(name, *fields, params=None):
    entry = {"name": name, "fields": [{"name": n, "type": t} for n, t in fields]}
    if params:
        entry["typeParameters"] = params
    return entry


def _opt(t):
    return [None, t]


def _vec(t, length=None):
    return {"vector": {"items": t, **({"length": length} if length else {})}}


def _arr(t, *names, fixed=None):
    if fixed is not None:
        return {"array": {"items": t, "dimensions": [{"length": n} for n in fixed]}}
    return {"array": {"items": t, "dimensions": [{"name": n} for n in names]}}


F3 = _arr("float32", fixed=(3,))
LIMIT = "Mrd.LimitType"
MRD_SCHEMA = {
    "protocol": {"name": "Mrd", "sequence": [
        {"name": "header", "type": _opt("Mrd.Header")},
        {"name": "data", "type": {"stream": {"items": "Mrd.StreamItem"}}},
    ]},
    "types": [
        {"name": "AcquisitionFlags", "base": "uint64",
         "values": [{"symbol": "firstInSlice", "value": 64},
                    {"symbol": "isNoiseMeasurement", "value": 262144}]},
        {"name": "PatientPosition", "values": [{"symbol": "hFP", "value": 0},
                                               {"symbol": "hFS", "value": 1}]},
        {"name": "PatientGender", "values": [{"symbol": "m", "value": 0},
                                             {"symbol": "f", "value": 1}]},
        {"name": "Trajectory", "values": [{"symbol": "cartesian", "value": 0},
                                          {"symbol": "spiral", "value": 4}]},
        {"name": "ImageType", "values": [{"symbol": "magnitude", "value": 1},
                                         {"symbol": "complex", "value": 5}]},
        {"name": "RFPulseUse", "values": [{"symbol": "undefined", "value": 0},
                                          {"symbol": "excitation", "value": 1}]},
        _record("SubjectInformationType", ("patientName", _opt("string")),
                ("patientWeightKg", _opt("float32")), ("patientHeightM", _opt("float32")),
                ("patientID", _opt("string")), ("patientBirthdate", _opt("date")),
                ("patientGender", _opt("Mrd.PatientGender"))),
        _record("StudyInformationType", ("studyDate", _opt("date")), ("studyTime", _opt("time")),
                ("studyID", _opt("string")), ("accessionNumber", _opt("int64"))),
        _record("MeasurementInformationType", ("seriesDate", _opt("date")),
                ("seriesTime", _opt("time")), ("patientPosition", "Mrd.PatientPosition"),
                ("protocolName", _opt("string"))),
        _record("ExperimentalConditionsType", ("h1resonanceFrequencyHz", "int64")),
        _record("MatrixSizeType", ("x", "uint32"), ("y", "uint32"), ("z", "uint32")),
        _record("FieldOfViewMm", ("x", "float32"), ("y", "float32"), ("z", "float32")),
        _record("EncodingSpaceType", ("matrixSize", "Mrd.MatrixSizeType"),
                ("fieldOfViewMm", "Mrd.FieldOfViewMm")),
        _record("LimitType", ("minimum", "uint32"), ("maximum", "uint32"), ("center", "uint32")),
        _record("EncodingLimitsType", ("kspaceEncodingStep0", _opt(LIMIT)),
                ("kspaceEncodingStep1", _opt(LIMIT)), ("slice", _opt(LIMIT))),
        _record("EncodingType", ("encodedSpace", "Mrd.EncodingSpaceType"),
                ("reconSpace", "Mrd.EncodingSpaceType"),
                ("encodingLimits", "Mrd.EncodingLimitsType"), ("trajectory", "Mrd.Trajectory"),
                ("echoTrainLength", _opt("int64"))),
        _record("SequenceParametersType", ("tR", _vec("float32")), ("tE", _vec("float32")),
                ("flipAngleDeg", _vec("float32"))),
        _record("Header", ("version", _opt("int64")),
                ("subjectInformation", _opt("Mrd.SubjectInformationType")),
                ("studyInformation", _opt("Mrd.StudyInformationType")),
                ("measurementInformation", _opt("Mrd.MeasurementInformationType")),
                ("experimentalConditions", "Mrd.ExperimentalConditionsType"),
                ("encoding", _vec("Mrd.EncodingType")),
                ("sequenceParameters", _opt("Mrd.SequenceParametersType"))),
        _record("EncodingCounters", *[(n, _opt("uint32")) for n in (
            "kspaceEncodeStep1", "kspaceEncodeStep2", "average", "slice", "contrast", "phase",
            "repetition", "set", "segment")], ("user", _vec("uint32"))),
        _record("AcquisitionHeader", ("flags", "Mrd.AcquisitionFlags"),
                ("idx", "Mrd.EncodingCounters"), ("measurementUid", "uint32"),
                ("scanCounter", _opt("uint32")), ("acquisitionCenterFrequency", _opt("uint64")),
                ("acquisitionTimeStampNs", _opt("uint64")),
                ("physiologyTimeStampNs", _vec("uint64")), ("channelOrder", _vec("uint32")),
                ("discardPre", _opt("uint32")), ("discardPost", _opt("uint32")),
                ("centerSample", _opt("uint32")), ("encodingSpaceRef", _opt("uint32")),
                ("sampleTimeNs", _opt("uint64")), ("position", F3), ("readDir", F3),
                ("phaseDir", F3), ("sliceDir", F3), ("patientTablePosition", F3),
                ("userInt", _vec("int32")), ("userFloat", _vec("float32"))),
        _record("Acquisition", ("head", "Mrd.AcquisitionHeader"),
                ("data", _arr("complexfloat32", "coils", "samples")),
                ("phase", _opt(_arr("float32", "samples"))),
                ("trajectory", _arr("float32", "basis", "samples"))),
        _record("Waveform", ("flags", "uint64"), ("measurementUid", "uint32"),
                ("scanCounter", "uint32"), ("timeStampNs", "uint64"), ("sampleTimeNs", "uint64"),
                ("waveformId", "uint32"), ("data", _arr("T", "channels", "samples")),
                params=["T"]),
        {"name": "WaveformUint32", "type": {"name": "Mrd.Waveform", "typeArguments": ["uint32"]}},
        _record("ImageHeader", ("flags", "uint64"), ("measurementUid", "uint32"),
                ("fieldOfView", F3), ("position", F3), ("colDir", F3), ("lineDir", F3),
                ("sliceDir", F3), ("patientTablePosition", F3), ("slice", _opt("uint32")),
                ("imageType", "Mrd.ImageType"), ("imageIndex", _opt("uint32")),
                ("imageSeriesIndex", _opt("uint32")), ("userInt", _vec("int32"))),
        {"name": "ImageMetaValue", "type": [{"tag": "string", "type": "string"},
                                            {"tag": "int64", "type": "int64"},
                                            {"tag": "float64", "type": "float64"}]},
        {"name": "ImageMeta", "type": {"map": {"keys": "string",
                                               "values": _vec("Mrd.ImageMetaValue")}}},
        _record("Image", ("head", "Mrd.ImageHeader"),
                ("data", _arr("T", "channel", "z", "y", "x")), ("meta", "Mrd.ImageMeta"),
                params=["T"]),
        {"name": "ImageUint16", "type": {"name": "Mrd.Image", "typeArguments": ["uint16"]}},
        {"name": "ImageComplexFloat",
         "type": {"name": "Mrd.Image", "typeArguments": ["complexfloat32"]}},
        {"name": "ArrayComplexFloat", "type": {"array": {"items": "complexfloat32"}}},
        _record("ThreeDimensionalFloat", ("x", "float32"), ("y", "float32"), ("z", "float32")),
        _record("PulseqDefinitions", ("gradientRasterTime", "float64"),
                ("radiofrequencyRasterTime", "float64"), ("adcRasterTime", "float64"),
                ("blockDurationRaster", "float64"), ("name", _opt("string")),
                ("fov", _opt("Mrd.ThreeDimensionalFloat")), ("totalDuration", _opt("float64")),
                ("custom", {"map": {"keys": "string", "values": "string"}})),
        _record("PulseqBlock", ("id", "int32"), ("duration", "uint64"), ("rf", "int32"),
                ("gx", "int32"), ("gy", "int32"), ("gz", "int32"), ("adc", "int32"),
                ("ext", "int32")),
        _record("PulseqRFEvent", ("id", "int32"), ("amp", "float64"), ("magId", "int32"),
                ("phaseId", "int32"), ("timeId", "int32"), ("center", "float64"),
                ("delay", "uint64"), ("freqPPM", "float64"), ("phasePPM", "float64"),
                ("freqOffset", "float64"), ("phaseOffset", "float64"),
                ("use", "Mrd.RFPulseUse")),
        _record("PulseqTrapezoidalGradient", ("id", "int32"), ("amp", "float64"),
                ("rise", "uint64"), ("flat", "uint64"), ("fall", "uint64"), ("delay", "uint64")),
        _record("PulseqShape", ("id", "int32"), ("numSamples", "uint64"),
                ("data", {"array": {"items": "float64", "dimensions": 1}})),
        {"name": "StreamItem", "type": [
            {"tag": tag, "explicitTag": True, "type": f"Mrd.{name}"} for tag, name in (
                ("acquisition", "Acquisition"), ("waveformUint32", "WaveformUint32"),
                ("imageUint16", "ImageUint16"), ("imageComplexFloat", "ImageComplexFloat"),
                ("arrayComplexFloat", "ArrayComplexFloat"),
                ("pulseqDefinitions", "PulseqDefinitions"))
        ] + [{"tag": "pulseqBlocks", "explicitTag": True, "type": _vec("Mrd.PulseqBlock")}] + [
            {"tag": tag, "explicitTag": True, "type": f"Mrd.{name}"} for tag, name in (
                ("pulseqRfEvent", "PulseqRFEvent"),
                ("pulseqTrapezoidalGradient", "PulseqTrapezoidalGradient"),
                ("pulseqShape", "PulseqShape"))
        ]},
    ],
}


def _header():
    space = {"matrixSize": {"x": 8, "y": 4, "z": 1},
             "fieldOfViewMm": {"x": 256.0, "y": 128.0, "z": 5.0}}
    return {
        "version": 2,
        "subjectInformation": {
            "patientName": "Doe^Jane", "patientWeightKg": 61.5, "patientHeightM": 1.7,
            "patientID": "007", "patientBirthdate": np.datetime64("1980-02-29"),
            "patientGender": "f",
        },
        "studyInformation": {
            "studyDate": np.datetime64("2024-05-06"),
            "studyTime": np.timedelta64(((9 * 60 + 8) * 60 + 7) * 10**9 + 5 * 10**8, "ns"),
            "studyID": "0042", "accessionNumber": 123456,
        },
        "measurementInformation": {
            "seriesDate": np.datetime64("2024-05-06"),
            "seriesTime": np.timedelta64(((9 * 60 + 10) * 60) * 10**9, "ns"),
            "patientPosition": "hFS", "protocolName": "gre",
        },
        "experimentalConditions": {"h1resonanceFrequencyHz": 123_200_000},
        "encoding": [{
            "encodedSpace": space, "reconSpace": space,
            "encodingLimits": {"kspaceEncodingStep0": None,
                               "kspaceEncodingStep1": {"minimum": 0, "maximum": 3, "center": 2},
                               "slice": None},
            "trajectory": "cartesian", "echoTrainLength": None,
        }],
        "sequenceParameters": {"tR": [5.86], "tE": [2.96], "flipAngleDeg": []},
    }


def _acquisition(line, noise=False, samples=8, coils=2, trajectory=None, users=8, phase=None):
    data = (np.arange(coils * samples) + 1j * line).astype(np.complex64).reshape(coils, samples)
    if trajectory is None:
        trajectory = np.zeros((0, 0), np.float32)
    head = {
        "flags": 262144 if noise else 64,
        "idx": {"kspaceEncodeStep1": None if noise else line, "kspaceEncodeStep2": None,
                "average": 0, "slice": None if noise else 0, "contrast": 0, "phase": 0,
                "repetition": 0, "set": 0, "segment": 0, "user": list(range(users))},
        "measurementUid": 7, "scanCounter": line, "acquisitionCenterFrequency": 123_200_000,
        "acquisitionTimeStampNs": 1_000_000 * (100 + line) + 400_000,
        "physiologyTimeStampNs": [2_500_000 * line], "channelOrder": [3, 1][:coils],
        "discardPre": 0, "discardPost": None, "centerSample": samples // 2,
        "encodingSpaceRef": 0, "sampleTimeNs": 2000 if noise else 7800,
        "position": [1.0, 2.0, 3.0], "readDir": [1.0, 0.0, 0.0], "phaseDir": [0.0, 1.0, 0.0],
        "sliceDir": [0.0, 0.0, 1.0], "patientTablePosition": [0.0, 0.0, -10.0],
        "userInt": list(range(-1, users - 1)), "userFloat": [0.5] * users,
    }
    return ("acquisition", {"head": head, "data": data, "phase": phase,
                            "trajectory": trajectory})


def _image(tag, index, series, kind="magnitude", values=None, position=(0.0, 0.0, 0.0)):
    if values is None:
        values = np.arange(12, dtype=np.uint16).reshape(1, 1, 3, 4) + index
    head = {
        "flags": 0, "measurementUid": 7, "fieldOfView": [40.0, 30.0, 5.0],
        "position": list(position), "colDir": [0.0, 1.0, 0.0], "lineDir": [-1.0, 0.0, 0.0],
        "sliceDir": [0.0, 0.0, 1.0], "patientTablePosition": [0.0, 0.0, 0.0], "slice": 0,
        "imageType": kind, "imageIndex": index, "imageSeriesIndex": series, "userInt": [],
    }
    meta = {"ImageComments": [("string", "test")], "Windows": [("int64", 1), ("int64", 2)]}
    return (tag, {"head": head, "data": values, "meta": meta})


#**************************************************************************************************#
#                                              tests                                               #
#**************************************************************************************************#
def test_yardl_binary_documentation_example(tmp_path):
    schema = (
        '{"protocol":{"name":"MyProtocol","sequence":[{"name":"floatArray","type":{"array":'
        '{"items":"float32","dimensions":[{"length":2},{"length":2}]}}},{"name":"points","type"'
        ':{"stream":{"items":"Sandbox.Point"}}}]},"types":[{"name":"Point","fields":[{"name":'
        '"x","type":"uint64"},{"name":"y","type":"int32"}]}]}'
    )
    body = bytes.fromhex(  # the documentation's bytes after the schema
        "9a99993f 9a995940 3333b340 9a99f940 03 0104 0308 050c 02 bc05c00c 80ea30bfee6d 00"
    )
    assert len(schema) == 304  # written as the varint b0 02
    raw = b"yardl\x01\x00\x00\x00" + bytes.fromhex("b002") + schema.encode() + body
    path = tmp_path / "example.bin"
    path.write_bytes(raw)
    stream = YardlFile(path)
    assert (stream.encoding, stream.version, stream.schema.name) == ("binary", 1, "MyProtocol")
    assert np.allclose(stream.steps["floatArray"], [[1.2, 3.4], [5.6, 7.8]])
    assert stream.steps["floatArray"].dtype == np.float32
    points = [(p["x"], p["y"]) for p in stream.steps["points"]]
    assert points == [(1, 2), (3, 4), (5, 6), (700, 800), (800000, -900000)]
    path.write_bytes(raw + b"\x00")
    with pytest.raises(YardlError, match="after the last"):
        YardlFile(path)
    path.write_bytes(raw[:-3])
    with pytest.raises(YardlError, match="past the end"):
        YardlFile(path)


def test_yardl_ndjson_documentation_example(tmp_path):
    schema = {
        "protocol": {"name": "HelloNDJson", "sequence": [
            {"name": n, "type": t} for n, t in (
                ("anIntStream", {"stream": {"items": "int32"}}), ("aBoolean", "bool"),
                ("aComplex", "complexfloat64"), ("aDate", "date"), ("aTime", "time"),
                ("aDateTime", "datetime"), ("anEnum", "Sandbox.MyEnum"),
                ("someFlags", "Sandbox.MyFlags"), ("anOptionalIntThatIsNotSet", _opt("int32")),
                ("aRecordWithOptionalNotSet", "Sandbox.MyRecord"),
                ("aDynamicArray", {"array": {"items": "int32"}}),
                ("aFixedArray", _arr("int32", fixed=(2, 3))),
                ("aMapWithAnIntKey", {"map": {"keys": "int32", "values": "int32"}}),
                ("aUnionWithSimpleRepresentation", [{"label": "int32", "type": "int32"},
                                                    {"label": "bool", "type": "bool"}]),
                ("aUnionRequiringTag", [{"label": "string", "type": "string"},
                                        {"label": "MyEnum", "type": "Sandbox.MyEnum"}]))]},
        "types": [
            {"name": "MyEnum", "values": [{"symbol": s, "value": i} for i, s in enumerate("abc")]},
            {"name": "MyFlags", "base": "int32",
             "values": [{"symbol": s, "value": 1 << i} for i, s in enumerate("abc")]},
            _record("MyRecord", ("x", "int32"), ("y", "int32"), ("z", _opt("int32"))),
        ],
    }
    lines = [
        json.dumps({"yardl": {"version": 1, "schema": schema}}),
        '{"anIntStream":1}', '{"anIntStream":2}', '{"anIntStream":3}', '{"aBoolean":true}',
        '{"aComplex":[1.0,2.0]}', '{"aDate":"2020-01-17"}', '{"aTime":"10:50:25.777888999"}',
        '{"aDateTime":"2023-05-30T18:36:56.708792349Z"}', '{"anEnum":"a"}',
        '{"someFlags":["a","b"]}', '{"anOptionalIntThatIsNotSet":null}',
        '{"aRecordWithOptionalNotSet":{"x":1,"y":2}}',
        '{"aDynamicArray":{"shape":[2,3],"data":[1,2,3,4,5,6]}}', '{"aFixedArray":[1,2,3,4,5,6]}',
        '{"aMapWithAnIntKey":[[2,2],[1,1]]}', '{"aUnionWithSimpleRepresentation":22}',
        '{"aUnionRequiringTag":{"string":"a"}}',
    ]
    path = tmp_path / "hello.ndjson"
    path.write_text("\n".join(lines) + "\n")
    steps = YardlFile(path).steps
    assert steps["anIntStream"] == [1, 2, 3] and steps["aBoolean"] is True
    assert steps["aComplex"] == 1 + 2j and steps["aDate"] == np.datetime64("2020-01-17")
    assert steps["aTime"] == np.timedelta64(39025777888999, "ns")
    assert steps["aDateTime"] == np.datetime64("2023-05-30T18:36:56.708792349")
    assert steps["anEnum"] == "a" and steps["someFlags"] == 3
    assert steps["anOptionalIntThatIsNotSet"] is None
    assert steps["aRecordWithOptionalNotSet"] == {"x": 1, "y": 2, "z": None}
    assert steps["aDynamicArray"].tolist() == [[1, 2, 3], [4, 5, 6]]
    assert steps["aFixedArray"].shape == (2, 3) and steps["aMapWithAnIntKey"] == {2: 2, 1: 1}
    assert steps["aUnionWithSimpleRepresentation"] == ("int32", 22)
    assert steps["aUnionRequiringTag"] == ("string", "a")


def test_yardl_varint_arrays_raw_bytes_and_enums(tmp_path):
    schema = {
        "protocol": {"name": "P", "sequence": [
            {"name": "big", "type": _arr("uint64", "n")},
            {"name": "signed", "type": {"array": {"items": "int16", "dimensions": 2}}},
            {"name": "bytes", "type": _vec("int8", 3)},
            {"name": "short", "type": _vec("float32")},
            {"name": "kinds", "type": _vec("Ns.Kind")},
            {"name": "flags", "type": "Ns.Bits"},
            {"name": "when", "type": "time"},
        ]},
        "types": [
            {"name": "Kind", "values": [{"symbol": "minus", "value": -1},
                                        {"symbol": "two", "value": 2}]},
            {"name": "Bits", "base": "uint8", "values": [{"symbol": "a", "value": 1},
                                                         {"symbol": "b", "value": 2}]},
        ],
    }
    big = np.array([0, 1, 127, 128, 300, 2**35, 2**63 + 5], dtype=np.uint64)
    signed = np.array([[0, -1, 1], [-300, 32767, -32768]], dtype=np.int16)
    steps = {"big": big, "signed": signed, "bytes": [-128, 0, 127],
             "short": [1.17, 2.96, 0.1, 5.86, 3.0], "kinds": ["minus", "two", 7],
             "flags": 3, "when": np.timedelta64(3_600_000_000_001, "ns")}
    path = tmp_path / "p.bin"
    YardlWriter(schema).write(path, steps)
    stream = YardlFile(path)
    assert np.array_equal(stream.steps["big"], big) and stream.steps["big"].dtype == np.uint64
    assert np.array_equal(stream.steps["signed"], signed)
    assert stream.steps["signed"].dtype == np.int16
    assert stream.steps["bytes"] == [-128, 0, 127]
    assert stream.steps["short"] == [1.17, 2.96, 0.1, 5.86, 3.0]  # float32 as shortest decimals
    assert stream.steps["kinds"] == ["minus", "two", 7]  # unknown enum value kept as integer
    assert stream.steps["flags"] == 3
    assert stream.steps["when"] == np.timedelta64(3_600_000_000_001, "ns")
    assert YardlFile.protocol_name(path.read_bytes()[:200]) == "P"
    assert YardlFile.protocol_name(b"\x89HDF") is None


def test_mrd_acquisitions_waveforms_and_keys(tmp_path):
    trajectory = np.stack([np.linspace(-0.5, 0.5, 8), np.full(8, 0.25)]).astype(np.float32)
    items = [
        _acquisition(0, noise=True, coils=2, samples=8),
        _acquisition(1, trajectory=trajectory),
        _acquisition(2, trajectory=trajectory.T.copy(), users=10),  # (samples, basis)
        _acquisition(3, trajectory=trajectory, samples=8, phase=np.ones(8, np.float32)),
        ("waveformUint32", {"flags": 0, "measurementUid": 7, "scanCounter": 2,
                            "timeStampNs": 101_000_000, "sampleTimeNs": 2_500_000,
                            "waveformId": 1, "data": np.arange(6, dtype=np.uint32).reshape(2, 3)}),
        ("arrayComplexFloat", np.ones((2, 2), np.complex64)),
    ]
    path = tmp_path / "scan.mrd"
    YardlWriter(MRD_SCHEMA).write(path, {"header": _header(), "data": items})
    sniff = gu.sniff(path)
    assert (sniff.format, sniff.kind, sniff.evidence) == (
        gu.Format.MRD, gu.Kind.RAW, ("yardl binary", "protocol Mrd"))
    [raw] = gu.load(path)
    assert isinstance(raw, gu.RawData) and raw.shape == (4, 2, 8) and raw.name == "data"
    assert raw.data[1, 1, 3] == 11 + 1j
    heads = raw.headers
    assert heads["acquisition_time_stamp"].tolist() == [100, 101, 102, 103]  # ns // 10^6
    assert heads["sample_time_us"].tolist() == pytest.approx([2.0, 7.8, 7.8, 7.8])
    assert heads["physiology_time_stamp"][2].tolist() == [5, 0, 0]
    assert heads["idx"]["kspace_encode_step_1"].tolist() == [0, 1, 2, 3]
    assert heads["idx"]["user"][2].tolist() == list(range(8))
    assert heads["user_int"][1].tolist() == list(range(-1, 7))
    assert heads["discard_post"].tolist() == [0] * 4 and heads["center_sample"][1] == 4
    assert heads["active_channels"][0] == 2 and heads["version"][0] == 1
    assert heads["patient_table_position"][0].tolist() == [0.0, 0.0, -10.0]
    assert raw.flagged("IS_NOISE_MEASUREMENT").tolist() == [True, False, False, False]
    assert raw.flags_of(1) == ["FIRST_IN_SLICE"]
    assert raw.trajectory.shape == (4, 8, 2) and heads["trajectory_dimensions"].tolist() == [
        0, 2, 2, 2]
    assert np.array_equal(raw.trajectory[1], trajectory.T)
    assert np.array_equal(raw.trajectory[2], trajectory.T)
    vendor = raw.meta.vendor["acquisitions"]
    assert vendor["acquisition_time_stamp_ns"][1] == 101_400_000
    assert vendor["channel_order"][1] == [3, 1] and vendor["phase"][3].tolist() == [1.0] * 8
    assert vendor["user_int"][2] == list(range(-1, 9))
    notes = " | ".join(raw.provenance.notes)
    assert "1 trajectories stored (samples, basis), transposed" in notes
    assert "discardPost (4)" in notes and "idx.kspaceEncodeStep2 (4)" in notes
    assert "1 arrayComplexFloat" in notes and "beyond eight" in notes
    [waveform] = raw.waveforms
    assert waveform["time_stamp"] == 101 and waveform["sample_time_us"] == 2500
    assert waveform["data"].shape == (2, 3) and waveform["channels"] == 2
    meta = raw.meta
    assert meta["PatientName"] == "Doe^Jane" and meta["PatientID"] == "007"
    assert meta["PatientWeight"] == 61.5 and meta["PatientHeight"] == 1.7
    assert meta["PatientSex"] == "F" and meta["PatientPosition"] == "HFS"
    assert meta["PatientDoB"] == "1980-02-29" and meta["StudyID"] == "0042"
    assert meta["AccessionNumber"] == 123456
    assert meta["StudyDateTime"] == "2024-05-06T09:08:07.5"
    assert meta["AcquisitionDateTime"] == "2024-05-06T09:10:00"
    assert meta["RepetitionTime"] == pytest.approx(0.00586)
    assert meta["EchoTime"] == pytest.approx(0.00296)
    assert "ExcitationFlipAngle" not in meta  # empty vector
    assert meta["H1ResonanceFrequency"] == 123.2
    assert meta["FieldStrength"] == pytest.approx(123.2 / 42.577478518)
    assert meta["EncodedMatrixSize"] == [8, 4, 1] and meta["EncodedFieldOfView"] == [256, 128, 5]
    assert meta["EncodingLimits"] == {
        "kspace_encoding_step_1": {"minimum": 0, "maximum": 3, "center": 2}}
    assert meta["KSpaceTrajectory"] == "cartesian"
    assert meta["DwellTime"] == pytest.approx(7.8e-6)  # the first readout that is not noise
    assert meta["NumberOfPoints"] == 8
    assert meta.vendor["measurementInformation"]["seriesTime"] == "09:10:00"


def test_mrd_images_grouped_by_series_and_set(tmp_path):
    complex_values = (np.arange(12) * (1 + 1j)).astype(np.complex64).reshape(1, 1, 3, 4)
    items = [
        _image("imageUint16", 1, 1), _image("imageUint16", 2, 1, position=(0.0, 0.0, 5.0)),
        _image("imageUint16", 1, 2),
        _image("imageComplexFloat", 1, 1, kind="complex", values=complex_values),
        _image("imageUint16", 0, 1),  # a decreasing index: a new set of series 1
    ]
    path = tmp_path / "images.mrd"
    YardlWriter(MRD_SCHEMA).write(path, {"header": None, "data": items})
    images = gu.load(path)
    assert [i.name for i in images] == [
        "imageUint16/series_1/1", "imageUint16/series_2", "imageComplexFloat/series_1/complex",
        "imageUint16/series_1/2"]
    first = images[0]
    assert first.shape == (4, 3, 1, 2) and first.dims == ("i", "j", "k", "image")
    assert first.data[2, 1, 0, 1] == 1 * 4 + 2 + 2  # (x, y) of image index 2
    assert first.meta["ImageType"] == ["MAGNITUDE"] and first.meta["SeriesNumber"] == 1
    # columns along LPS +y (RAS -y), lines along LPS -x (RAS +x), centre at the position
    assert np.allclose(first.affine[:3, 0], [0, -10, 0]) and np.allclose(
        first.affine[:3, 1], [10, 0, 0])
    centre = first.affine @ np.array([1.5, 1.0, 0.0, 1.0])
    assert np.allclose(centre[:3], [0, 0, 0])
    assert "differing positions" in " ".join(first.provenance.notes)
    assert first.meta.vendor["image_meta"][0] == {"ImageComments": "test", "Windows": [1, 2]}
    assert first.meta.vendor["image_headers"]["imageIndex"] == [1, 2]
    assert images[2].data.dtype == np.complex64 and images[2].meta["ImageType"] == ["COMPLEX"]
    assert any("no MRD header" in n for n in first.provenance.notes)


def test_mrd_pulseq_items(tmp_path):
    items = [
        ("pulseqDefinitions", {
            "gradientRasterTime": 1e-5, "radiofrequencyRasterTime": 1e-6, "adcRasterTime": 1e-7,
            "blockDurationRaster": 1e-5, "name": "gre", "fov": {"x": 0.25, "y": 0.25, "z": 0.005},
            "totalDuration": 0.012, "custom": {"TE": "0.004", "Matrix": "64 64 1"}}),
        ("pulseqBlocks", [
            {"id": 1, "duration": 100, "rf": 1, "gx": 0, "gy": 0, "gz": 1, "adc": 0, "ext": 0},
            {"id": 2, "duration": 1100, "rf": 0, "gx": 1, "gy": 0, "gz": 0, "adc": 0, "ext": 0}]),
        ("pulseqRfEvent", {"id": 1, "amp": 250.0, "magId": 1, "phaseId": 2, "timeId": 0,
                           "center": 500.0, "delay": 100, "freqPPM": 0.0, "phasePPM": 0.0,
                           "freqOffset": 0.0, "phaseOffset": 0.0, "use": "excitation"}),
        ("pulseqTrapezoidalGradient", {"id": 1, "amp": 1000.0, "rise": 100, "flat": 800,
                                       "fall": 100, "delay": 0}),
        ("pulseqShape", {"id": 1, "numSamples": 5, "data": np.array([0.0, 0.25, 0.25, 2.0])}),
        ("pulseqShape", {"id": 2, "numSamples": 2, "data": np.array([0.0, 1.0])}),
    ]
    path = tmp_path / "seq.mrd"
    YardlWriter(MRD_SCHEMA).write(path, {"header": None, "data": items})
    [sequence] = gu.load(path)
    assert isinstance(sequence, gu.PulseSequence) and sequence.n_blocks == 2
    assert sequence.duration == pytest.approx(0.012) and sequence.version == ""
    assert sequence.definitions["TE"] == 0.004 and sequence.definitions["Matrix"] == [64, 64, 1]
    assert sequence.definitions["FOV"] == [0.25, 0.25, 0.005]
    assert sequence.events["rf"][1]["use"] == "e" and sequence.events["rf"][1]["center"] == 500
    assert sequence.events["trap"][1]["flat"] == 800
    assert sequence.shapes[1].tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]  # decompressed
    assert sequence.shapes[2].tolist() == [0.0, 1.0]
    assert sequence.meta["SequenceName"] == "gre" and sequence.meta["ScanDuration"] == 0.012
    assert sequence.meta["EncodedFieldOfView"] == pytest.approx([250, 250, 5])
    assert any("Pulseq revision" in n for n in sequence.provenance.notes)


def test_mrd_ndjson_equals_binary(tmp_path):
    binary = tmp_path / "one.mrd"
    YardlWriter(MRD_SCHEMA).write(binary, {"header": None, "data": [_acquisition(1)]})
    head = _acquisition(1)[1]["head"]
    head = {k: v for k, v in head.items() if v is not None}
    head["idx"] = {k: v for k, v in head["idx"].items() if v is not None}
    head["flags"] = ["firstInSlice"]
    data = _acquisition(1)[1]["data"]
    pairs = [[float(v.real), float(v.imag)] for v in data.ravel()]
    lines = [
        json.dumps({"yardl": {"version": 1, "schema": MRD_SCHEMA}}),
        json.dumps({"header": None}),
        json.dumps({"data": {"acquisition": {
            "head": head, "data": {"shape": list(data.shape), "data": pairs},
            "trajectory": {"shape": [0, 0], "data": []}}}}),
    ]
    text = tmp_path / "one.ndjson"
    text.write_text("\n".join(lines) + "\n")
    assert gu.sniff(text).evidence == ("yardl NDJSON", "protocol Mrd")
    [a], [b] = gu.load(binary), gu.load(text)
    assert np.array_equal(a.data, b.data) and a.headers.tobytes() == b.headers.tobytes()
    assert a.meta.vendor["acquisitions"] == b.meta.vendor["acquisitions"]
    assert "ndjson encoding" in b.provenance.notes[0]


def test_other_yardl_protocols_are_not_mrd(tmp_path):
    schema = {"protocol": {"name": "MrdNoiseCovariance", "sequence": [
        {"name": "value", "type": "int32"}]}, "types": []}
    path = tmp_path / "noise.bin"
    YardlWriter(schema).write(path, {"value": 3})
    assert gu.sniff(path).format != gu.Format.MRD
    with pytest.raises(gu.ReaderError, match="not Mrd"):
        gu.Loader().readers[gu.Format.MRD].read(gu.Sniff(path, gu.Format.MRD))
