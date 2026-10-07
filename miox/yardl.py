####################################################################################################
#                                             yardl.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/10/26                                                                                #
#                                                                                                  #
# Purpose: Yardl protocol files (compact binary and NDJSON encodings) decoded by the protocol      #
#          schema embedded in the file, without the generated code of the protocol.                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import json
import re
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


#**************************************************************************************************#
#                                            YardlError                                            #
#**************************************************************************************************#
class YardlError(ValueError):
    """The file is not a yardl protocol file or does not follow its schema."""


#**************************************************************************************************#
#                                           YardlSchema                                            #
#**************************************************************************************************#
#                                                                                                  #
# The protocol schema: the protocol's steps and the named types (records with ``fields``, enums    #
# and flags with ``values``, aliases with ``type``; generic ones with ``typeParameters``). Type    #
# references carry the namespace ("Mrd.Header"), the type entries do not. ``concrete`` replaces    #
# type parameters by their arguments and aliases by what they name, so that every decoder sees     #
# primitives, records, enums, unions, vectors, arrays, maps and streams only. The schema does not  #
# say whether a type is an enum or a flags type; a type with an explicit base whose values are all #
# single bits is taken as flags (kept as its integer value), every other one as an enum (decoded   #
# to its symbol, an unknown value kept as integer).                                                #
#                                                                                                  #
#**************************************************************************************************#
class YardlSchema:
    PRIMITIVES = {
        "bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "size",
        "float32", "float64", "complexfloat32", "complexfloat64", "string", "date", "time",
        "datetime",
    }

    def __init__(self, schema: dict):
        if "protocol" not in schema or "types" not in schema:
            raise YardlError("schema without protocol or types")
        self.schema = schema
        self.protocol = schema["protocol"]
        self.name = self.protocol.get("name", "")
        self.types = {t["name"]: t for t in schema["types"]}

    @property
    def steps(self) -> list[tuple[str, Any]]:
        return [(step["name"], step["type"]) for step in self.protocol["sequence"]]

    def named(self, reference: str) -> dict:
        name = reference.rsplit(".", 1)[-1]
        if name not in self.types:
            raise YardlError(f"type {reference} is not defined in the schema")
        return self.types[name]

    def concrete(self, node: Any, bindings: dict[str, Any] | None = None) -> Any:
        """``node`` with type parameters bound and aliases resolved, down to named records and
        enums, which become {"record"|"enum": entry, "arguments": [...]}."""
        bindings = bindings or {}
        if node is None:
            return None
        if isinstance(node, str):
            if node in bindings:
                return bindings[node]
            if node in self.PRIMITIVES:
                return node
            return self._named_node(self.named(node), [])
        if isinstance(node, list):
            return [self._case(case, bindings) for case in node]
        if "name" in node and "typeArguments" in node:
            arguments = [self.concrete(a, bindings) for a in node["typeArguments"]]
            return self._named_node(self.named(node["name"]), arguments)
        for kind in ("vector", "array", "stream"):
            if kind in node:
                inner = dict(node[kind])
                inner["items"] = self.concrete(inner["items"], bindings)
                return {kind: inner}
        if "map" in node:
            return {"map": {
                "keys": self.concrete(node["map"]["keys"], bindings),
                "values": self.concrete(node["map"]["values"], bindings),
            }}
        raise YardlError(f"unknown type node {node!r}")

    def _case(self, case: Any, bindings: dict[str, Any]) -> Any:
        if case is None:
            return None
        if isinstance(case, dict) and "type" in case:
            tag = case.get("tag", case.get("label"))
            return {"tag": tag, "type": self.concrete(case["type"], bindings)}
        return {"tag": None, "type": self.concrete(case, bindings)}  # optional: [null, T]

    def _named_node(self, entry: dict, arguments: list[Any]) -> Any:
        bindings = dict(zip(entry.get("typeParameters", []), arguments, strict=True))
        if "fields" in entry:
            fields = [(f["name"], self.concrete(f["type"], bindings)) for f in entry["fields"]]
            return {"record": entry["name"], "fields": fields}
        if "values" in entry:
            values = {v["value"]: v["symbol"] for v in entry["values"]}
            base = entry.get("base", "int32")
            flags = "base" in entry and all(v > 0 and v & (v - 1) == 0 for v in values)
            return {"enum": entry["name"], "base": base, "values": values, "flags": flags}
        if "type" in entry:
            return self.concrete(entry["type"], bindings)
        raise YardlError(f"type {entry.get('name')} is neither record, enum nor alias")


#**************************************************************************************************#
#                                            YardlFile                                             #
#**************************************************************************************************#
#                                                                                                  #
# A yardl protocol file read whole: ``schema`` (YardlSchema), ``encoding`` ("binary" or "ndjson")  #
# and ``steps``, the value of every protocol step by name (a stream as a list of its items).       #
# Values: records as dicts in field order (an absent optional as None), vectors as lists, arrays   #
# as numpy arrays (records inside as object arrays of dicts), maps as dicts, tagged unions as      #
# (tag, value), optionals as the value or None, enums as their symbol, flags as integers, complex  #
# numbers as complex, dates as datetime64[D], times as timedelta64[ns] since midnight and          #
# datetimes as datetime64[ns]. A float32 outside an array is the shortest decimal that rounds to   #
# the stored value (1.17, not 1.1699999570846558); arrays keep their dtype.                        #
#                                                                                                  #
#**************************************************************************************************#
class YardlFile:
    MAGIC = b"yardl"
    NDJSON = re.compile(rb'\s*\{\s*"yardl"\s*:')
    PROTOCOL = re.compile(rb'"protocol"\s*:\s*\{\s*"name"\s*:\s*"([^"]*)"')

    def __init__(self, path: str | Path):
        self.path = Path(path)
        raw = self.path.read_bytes()
        if raw.startswith(self.MAGIC):
            self.encoding = "binary"
            self.version, schema, self.steps = YardlBinary(raw).read()
        elif self.NDJSON.match(raw):
            self.encoding = "ndjson"
            self.version, schema, self.steps = YardlNdjson(raw).read()
        else:
            raise YardlError(f"{self.path.name}: neither yardl binary nor yardl NDJSON")
        self.schema = schema

    @classmethod
    def protocol_name(cls, head: bytes) -> str | None:
        """The protocol's name from the first bytes of a file; None when it is no yardl file."""
        if not (head.startswith(cls.MAGIC) or cls.NDJSON.match(head)):
            return None
        match = cls.PROTOCOL.search(head)
        return match.group(1).decode("utf-8", "replace") if match else ""


#**************************************************************************************************#
#                                           YardlBinary                                            #
#**************************************************************************************************#
#                                                                                                  #
# The compact binary encoding: "yardl", a little-endian u32 encoding version, the schema as a      #
# string, then the step values. Unsigned integers from 16 bits up and sizes are varints (7 bits    #
# per byte, low first), signed ones zig-zag varints; int8, uint8 and bool are single bytes (the    #
# reference implementation does so, its documentation says varints); floats and complex numbers    #
# little-endian IEEE. Strings and vectors without a fixed length are led by their length, arrays   #
# by their dimension count unless the schema gives it and by their dimension lengths unless the    #
# schema fixes them (values row-major), unions by the case index, optionals by 0 or 1, maps by     #
# their entry count; streams are blocks led by their item count and closed by an empty block.      #
#                                                                                                  #
#**************************************************************************************************#
class YardlBinary:
    FIXED = {  # primitive: numpy dtype of the bytes as written
        "bool": np.dtype("?"), "int8": np.dtype("i1"), "uint8": np.dtype("u1"),
        "float32": np.dtype("<f4"), "float64": np.dtype("<f8"),
        "complexfloat32": np.dtype("<c8"), "complexfloat64": np.dtype("<c16"),
    }
    UNSIGNED = {"uint16": np.uint16, "uint32": np.uint32, "uint64": np.uint64, "size": np.uint64}
    SIGNED = {"int16": np.int16, "int32": np.int32, "int64": np.int64}

    def __init__(self, raw: bytes):
        self.raw = raw
        self.cache: dict[str, Callable[[int], tuple[Any, int]]] = {}

    def read(self) -> tuple[int, YardlSchema, dict[str, Any]]:
        if not self.raw.startswith(YardlFile.MAGIC) or len(self.raw) < 9:
            raise YardlError("no yardl binary header")
        version = struct.unpack_from("<I", self.raw, 5)[0]
        size, pos = self.uvarint(9)
        try:
            schema = YardlSchema(json.loads(self.raw[pos : pos + size]))
        except json.JSONDecodeError as error:
            raise YardlError(f"schema is not JSON: {error}") from error
        pos += size
        steps: dict[str, Any] = {}
        for name, node in schema.steps:
            decode = self.decoder(schema.concrete(node))
            try:
                steps[name], pos = decode(pos)
            except (IndexError, struct.error) as error:
                raise YardlError(f"step {name} ends past the end of the file") from error
        if pos != len(self.raw):
            raise YardlError(f"{len(self.raw) - pos} bytes after the last protocol step")
        return version, schema, steps

    # --- varints ---------------------------------------------------------------------------------

    def uvarint(self, pos: int) -> tuple[int, int]:
        raw = self.raw
        byte = raw[pos]
        if byte < 0x80:
            return byte, pos + 1
        value, shift = byte & 0x7F, 7
        while True:
            pos += 1
            byte = raw[pos]
            value |= (byte & 0x7F) << shift
            if byte < 0x80:
                return value, pos + 1
            shift += 7

    def svarint(self, pos: int) -> tuple[int, int]:
        value, pos = self.uvarint(pos)
        return (value >> 1) ^ -(value & 1), pos

    def uvarints(self, pos: int, count: int) -> tuple[np.ndarray, int]:
        """``count`` varints from ``pos`` at once, as uint64."""
        if count == 0:
            return np.zeros(0, np.uint64), pos
        window = np.frombuffer(self.raw, np.uint8, min(len(self.raw) - pos, 10 * count), pos)
        ends = np.flatnonzero(window < 0x80)[:count]
        if len(ends) < count:
            raise YardlError("array of varints ends past the end of the file")
        stop = int(ends[-1]) + 1
        starts = np.concatenate(([0], ends[:-1] + 1))
        group = np.repeat(np.arange(count), ends - starts + 1)
        shifts = (np.arange(stop) - starts[group]).astype(np.uint64) * np.uint64(7)
        parts = (window[:stop] & 0x7F).astype(np.uint64) << shifts
        return np.add.reduceat(parts, starts), pos + stop

    # --- decoders --------------------------------------------------------------------------------

    def decoder(self, node: Any) -> Callable[[int], tuple[Any, int]]:
        key = json.dumps(node, sort_keys=True, default=str)
        if key not in self.cache:
            self.cache[key] = self._compile(node)
        return self.cache[key]

    def _compile(self, node: Any) -> Callable[[int], tuple[Any, int]]:
        if isinstance(node, str):
            return self._primitive(node)
        if isinstance(node, list):
            return self._union(node)
        if "record" in node:
            return self._record(node)
        if "enum" in node:
            return self._enum(node)
        if "vector" in node:
            return self._vector(node["vector"])
        if "array" in node:
            return self._array(node["array"])
        if "map" in node:
            return self._map(node["map"])
        if "stream" in node:
            return self._stream(node["stream"])
        raise YardlError(f"no decoder for {node!r}")

    def _primitive(self, name: str) -> Callable[[int], tuple[Any, int]]:
        raw = self.raw
        if name in self.UNSIGNED:
            return self.uvarint
        if name in self.SIGNED:
            return self.svarint
        if name in self.FIXED:
            dtype = self.FIXED[name]
            code = {"bool": "?", "int8": "b", "uint8": "B", "float32": "<f", "float64": "<d",
                    "complexfloat32": "<2f", "complexfloat64": "<2d"}[name]
            unpack, size = struct.Struct(code).unpack_from, dtype.itemsize
            if name.startswith("complex"):
                return lambda pos: (complex(*unpack(raw, pos)), pos + size)
            if name == "float32":
                return lambda pos: (float(str(np.float32(unpack(raw, pos)[0]))), pos + size)
            return lambda pos: (unpack(raw, pos)[0], pos + size)
        if name == "string":
            def string(pos: int) -> tuple[str, int]:
                size, pos = self.uvarint(pos)
                return raw[pos : pos + size].decode("utf-8"), pos + size
            return string
        if name == "date":
            return lambda pos: self._wrap(self.svarint(pos), "D")
        if name == "time":
            return lambda pos: self._wrap(self.svarint(pos), "ns", delta=True)
        if name == "datetime":
            return lambda pos: self._wrap(self.svarint(pos), "ns")
        raise YardlError(f"unknown primitive {name}")

    @staticmethod
    def _wrap(decoded: tuple[int, int], unit: str, delta: bool = False) -> tuple[Any, int]:
        value, pos = decoded
        return (np.timedelta64(value, unit) if delta else np.datetime64(value, unit)), pos

    def _union(self, cases: list[Any]) -> Callable[[int], tuple[Any, int]]:
        decoders = [None if c is None else self.decoder(c["type"]) for c in cases]
        tags = [None if c is None else c["tag"] for c in cases]
        optional = len(cases) == 2 and cases[0] is None and tags[1] is None

        def union(pos: int) -> tuple[Any, int]:
            index, pos = self.uvarint(pos)
            if index >= len(decoders):
                raise YardlError(f"union case {index} of {len(decoders)}")
            if decoders[index] is None:
                return None, pos
            value, pos = decoders[index](pos)
            return (value if optional else (tags[index], value)), pos
        return union

    def _record(self, node: dict) -> Callable[[int], tuple[Any, int]]:
        fields = [(name, self.decoder(t)) for name, t in node["fields"]]

        def record(pos: int) -> tuple[dict, int]:
            values = {}
            for name, decode in fields:
                values[name], pos = decode(pos)
            return values, pos
        return record

    def _enum(self, node: dict) -> Callable[[int], tuple[Any, int]]:
        read = self._primitive(node["base"])
        if node["flags"]:
            return read
        values = node["values"]

        def enum(pos: int) -> tuple[Any, int]:
            value, pos = read(pos)
            return values.get(value, value), pos
        return enum

    def _numbers(self, items: Any, count: int, pos: int) -> tuple[np.ndarray, int] | None:
        """``count`` primitive numbers at once, or None when ``items`` is no number type."""
        if not isinstance(items, str):
            return None
        if items in self.FIXED:
            dtype = self.FIXED[items]
            values = np.frombuffer(self.raw, dtype, count, pos).copy()
            return values, pos + count * dtype.itemsize
        if items in self.UNSIGNED:
            values, pos = self.uvarints(pos, count)
            return values.astype(self.UNSIGNED[items]), pos
        if items in self.SIGNED:
            values, pos = self.uvarints(pos, count)
            signed = (values >> np.uint64(1)).astype(np.int64) ^ -(values & np.uint64(1)).astype(
                np.int64
            )
            return signed.astype(self.SIGNED[items]), pos
        return None

    def _vector(self, node: dict) -> Callable[[int], tuple[Any, int]]:
        items, length = node["items"], node.get("length")
        decode = self.decoder(items)

        def vector(pos: int) -> tuple[list, int]:
            count = length
            if count is None:
                count, pos = self.uvarint(pos)
            numbers = self._numbers(items, count, pos) if count > 4 else None
            if numbers is not None:
                if items == "float32":
                    return [float(str(v)) for v in numbers[0]], numbers[1]
                return numbers[0].tolist(), numbers[1]
            values = []
            for _ in range(count):
                value, pos = decode(pos)
                values.append(value)
            return values, pos
        return vector

    def _array(self, node: dict) -> Callable[[int], tuple[Any, int]]:
        items, dimensions = node["items"], node.get("dimensions")
        decode = self.decoder(items)
        fixed = None
        if isinstance(dimensions, list) and all("length" in d for d in dimensions):
            fixed = tuple(int(d["length"]) for d in dimensions)
        ndim = len(dimensions) if isinstance(dimensions, list) else dimensions

        def array(pos: int) -> tuple[np.ndarray, int]:
            if fixed is not None:
                shape = fixed
            else:
                count = ndim
                if count is None:
                    count, pos = self.uvarint(pos)
                shape = []
                for _ in range(count):
                    size, pos = self.uvarint(pos)
                    shape.append(size)
                shape = tuple(shape)
            total = int(np.prod(shape, dtype=np.int64))
            numbers = self._numbers(items, total, pos)
            if numbers is not None:
                return numbers[0].reshape(shape), numbers[1]
            values = np.empty(total, dtype=object)
            for i in range(total):
                values[i], pos = decode(pos)
            return values.reshape(shape), pos
        return array

    def _map(self, node: dict) -> Callable[[int], tuple[Any, int]]:
        keys, values = self.decoder(node["keys"]), self.decoder(node["values"])

        def mapping(pos: int) -> tuple[dict, int]:
            count, pos = self.uvarint(pos)
            result = {}
            for _ in range(count):
                key, pos = keys(pos)
                result[key], pos = values(pos)
            return result, pos
        return mapping

    def _stream(self, node: dict) -> Callable[[int], tuple[Any, int]]:
        decode = self.decoder(node["items"])

        def stream(pos: int) -> tuple[list, int]:
            items = []
            while True:
                count, pos = self.uvarint(pos)
                if count == 0:
                    return items, pos
                for _ in range(count):
                    value, pos = decode(pos)
                    items.append(value)
        return stream


#**************************************************************************************************#
#                                           YardlNdjson                                            #
#**************************************************************************************************#
#                                                                                                  #
# The NDJSON encoding: a first line {"yardl": {"version", "schema"}}, then one line per value      #
# {step: value} (a stream's items on consecutive lines). Complex numbers are [re, im], enums their #
# symbol, flags a list of symbols (or the integer), dates/times/datetimes ISO strings, fixed       #
# arrays flat lists, other arrays {"shape", "data"}, maps with string keys objects and others      #
# lists of [key, value]; absent optional record fields are left out. A union is written as its     #
# value when its cases differ in JSON kind, else as {tag: value}. Values come out as YardlFile     #
# describes them.                                                                                  #
#                                                                                                  #
#**************************************************************************************************#
class YardlNdjson:
    DTYPES = {
        "bool": np.bool_, "int8": np.int8, "uint8": np.uint8, "int16": np.int16,
        "uint16": np.uint16, "int32": np.int32, "uint32": np.uint32, "int64": np.int64,
        "uint64": np.uint64, "size": np.uint64, "float32": np.float32, "float64": np.float64,
        "complexfloat32": np.complex64, "complexfloat64": np.complex128,
    }

    def __init__(self, raw: bytes):
        self.lines = [line for line in raw.decode("utf-8").splitlines() if line.strip()]

    def read(self) -> tuple[int, YardlSchema, dict[str, Any]]:
        try:
            first = json.loads(self.lines[0])["yardl"]
            schema = YardlSchema(first["schema"])
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise YardlError(f"no yardl NDJSON header: {error}") from error
        nodes = {name: schema.concrete(node) for name, node in schema.steps}
        steps: dict[str, Any] = {}
        for number, line in enumerate(self.lines[1:], start=2):
            entry = json.loads(line)
            if not isinstance(entry, dict) or len(entry) != 1:
                raise YardlError(f"line {number}: not a single protocol step")
            [(name, value)] = entry.items()
            if name not in nodes:
                raise YardlError(f"line {number}: unknown protocol step {name}")
            node = nodes[name]
            if isinstance(node, dict) and "stream" in node:
                steps.setdefault(name, []).append(self.value(node["stream"]["items"], value))
            else:
                steps[name] = self.value(node, value)
        for name, node in nodes.items():
            if name not in steps:
                steps[name] = [] if isinstance(node, dict) and "stream" in node else None
        return int(first.get("version", 0)), schema, steps

    def value(self, node: Any, value: Any) -> Any:
        if isinstance(node, str):
            return self._primitive(node, value)
        if isinstance(node, list):
            return self._union(node, value)
        if "record" in node:
            value = value or {}
            return {name: self.value(t, value.get(name)) for name, t in node["fields"]}
        if "enum" in node:
            if node["flags"] and isinstance(value, list):
                symbols = {s: v for v, s in node["values"].items()}
                return sum(symbols[s] for s in value)
            return value
        if "vector" in node:
            return [self.value(node["vector"]["items"], v) for v in value]
        if "array" in node:
            return self._array(node["array"], value)
        if "map" in node:
            keys, values = node["map"]["keys"], node["map"]["values"]
            pairs = value.items() if isinstance(value, dict) else value
            return {self.value(keys, k): self.value(values, v) for k, v in pairs}
        raise YardlError(f"no NDJSON decoding for {node!r}")

    @staticmethod
    def _primitive(name: str, value: Any) -> Any:
        if value is None:
            return None
        if name.startswith("complex"):
            return complex(value[0], value[1])
        if name == "date":
            return np.datetime64(value, "D")
        if name == "time":
            hours, minutes, seconds = value.split(":")
            whole, _, fraction = seconds.partition(".")
            ns = ((int(hours) * 60 + int(minutes)) * 60 + int(whole)) * 10**9
            return np.timedelta64(ns + int((fraction + "000000000")[:9]), "ns")
        if name == "datetime":
            return np.datetime64(value.rstrip("Z"), "ns")
        if name == "float32":
            return float(str(np.float32(value)))
        if name == "float64":
            return float(value)
        return value

    def _array(self, node: dict, value: Any) -> np.ndarray:
        items, dimensions = node["items"], node.get("dimensions")
        if isinstance(value, dict):
            shape, data = tuple(value["shape"]), value["data"]
        else:
            shape, data = tuple(int(d["length"]) for d in dimensions), value
        if isinstance(items, str) and items in self.DTYPES:
            if items.startswith("complex"):
                pairs = np.asarray(data, dtype=np.float64).reshape(-1, 2)
                values = (pairs[:, 0] + 1j * pairs[:, 1]).astype(self.DTYPES[items])
            else:
                values = np.asarray(data, dtype=self.DTYPES[items])
            return values.reshape(shape)
        values = np.empty(len(data), dtype=object)
        for i, v in enumerate(data):
            values[i] = self.value(items, v)
        return values.reshape(shape)

    def _union(self, cases: list[Any], value: Any) -> Any:
        if value is None:
            return None
        real = [c for c in cases if c is not None]
        if len(real) == 1 and real[0]["tag"] is None:
            return self.value(real[0]["type"], value)
        kinds = [self._kind(c["type"]) for c in real]
        if len(set(kinds)) == len(kinds):
            kind = self._kind_of(value)
            for case, case_kind in zip(real, kinds, strict=True):
                if case_kind == kind:
                    return case["tag"], self.value(case["type"], value)
            raise YardlError(f"no union case for a JSON {kind}")
        [(tag, inner)] = value.items()
        case = next(c for c in real if c["tag"] == tag)
        return tag, self.value(case["type"], inner)

    @classmethod
    def _kind(cls, node: Any) -> str:
        """The JSON kind a value of ``node`` is written as."""
        if isinstance(node, str):
            if node == "bool":
                return "boolean"
            if node in ("string", "date", "time", "datetime"):
                return "string"
            return "array" if node.startswith("complex") else "number"
        if isinstance(node, list):
            return "union"
        if "enum" in node:
            return "array" if node["flags"] else "string"
        if "vector" in node:
            return "array"
        if "array" in node:
            dims = node["array"].get("dimensions")
            fixed = isinstance(dims, list) and all("length" in d for d in dims)
            return "array" if fixed else "object"
        if "map" in node:
            return "object" if node["map"]["keys"] == "string" else "array"
        return "object"

    @staticmethod
    def _kind_of(value: Any) -> str:
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, (int, float)):
            return "number"
        if isinstance(value, str):
            return "string"
        return "array" if isinstance(value, list) else "object"
