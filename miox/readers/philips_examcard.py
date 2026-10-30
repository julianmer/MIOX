####################################################################################################
#                                       philips_examcard.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for Philips ExamCards: every scan of the exam card with its parameters and       #
#          standard keys.                                                                          #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import base64
import re
import struct
import xml.etree.ElementTree as ET
from typing import Any

# own
from miox.data import Protocol
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                     PhilipsExamCardReader                                        #
#**************************************************************************************************#
#                                                                                                  #
# An ExamCard is a .NET SOAP serialisation (trailing NUL padding is passed over): objects with ids #
# referring to each other by href. Every ScanProcedure is one scan, named by its name, its         #
# parameterData a base64 table: a 32-byte header whose last word is the parameter count, then one  #
# 50-byte entry per parameter (name up to 33 bytes, then at byte 34 the type - 0 float, 1 integer, #
# 2 text of 81-byte fields, 4 enumeration - the count and two offsets relative to their own        #
# position: the enumeration's comma-separated choices and the values), as gt-examcard-parser reads #
# it. Enumerations are kept as their choice (as their index where no choices are written). The     #
# keys: TR and TE from the actual values the scanner computed (IF_act_rep_time_echo_time, "TR /    #
# TE" or TR alone, in ms), else from the settings (EX_ACQ_ffe_rep_time, EX_ACQ_se_rep_time,        #
# EX_ACQ_first_echo_time) when set by the user; flip angle, slice thickness (not of single-voxel   #
# scans), SENSE factors (when SENSE is on), the multiband factor (when multiband is on), nucleus   #
# and the exam card's patient position. Switches are YES or, without choices, index 1 (the         #
# ExamCards' NO, YES order).                                                                       #
#                                                                                                  #
#**************************************************************************************************#
class PhilipsExamCardReader(Reader):
    FORMATS = (Format.PHILIPS_EXAMCARD,)
    PACKAGES = ("miox",)
    SOURCE = "ExamCard"
    KEYS = {  # standard key: parameter, divisor to the key's unit
        "ExcitationFlipAngle": ("EX_ACQ_flip_angle", None),
        "SliceThickness": ("EX_GEO_slice_thickness", None),
        "MultibandAccelerationFactor": ("EX_MB_multiband_factor", None),
    }
    SENSE = {"ParallelReductionFactorInPlane": "EX_GEO_sense_p_red_factor",
             "ParallelReductionFactorOutOfPlane": "EX_GEO_sense_s_red_factor"}

    def read(self, sniff: Sniff) -> list[Protocol]:
        return self.protocols(sniff.path.read_bytes(), sniff.path.stem, sniff)

    def protocols(self, text: bytes, stem: str, sniff: Sniff) -> list[Protocol]:
        """The scans of an ExamCard's XML; ``stem`` names a card without a name."""
        try:
            root = ET.fromstring(text.rstrip(b"\0"))
        except ET.ParseError as error:
            raise ReaderError(f"{sniff.path.name}: not an ExamCard: {error}") from error
        ids = {e.get("id"): e for e in root.iter() if e.get("id")}
        card = next((e for e in root.iter() if e.tag.endswith("}ExamCard")), None)
        card_name = card.findtext("name") if card is not None else stem
        general = self._fields(card, ids) if card is not None else {}
        protocols = []
        for procedure in (e for e in root.iter() if e.tag.endswith("}ScanProcedure")):
            reference = procedure.find("parameterData")
            if reference is None or not reference.get("href"):
                continue
            data = ids.get(reference.get("href")[1:])
            parameters, choices = self.parameters(base64.b64decode(data.text or ""))
            meta = Meta({"parameters": parameters, "choices": choices,
                         "scan_procedure": self._fields(procedure, ids), "exam_card": general})
            self._keys(meta, parameters, general)
            name = procedure.findtext("name") or stem
            protocols.append(Protocol(name, (card_name,), meta, self.provenance(sniff)))
        if not protocols:
            raise ReaderError(f"{sniff.path.name}: no scans in the ExamCard")
        return protocols

    @staticmethod
    def _fields(node: ET.Element, ids: dict) -> dict:
        """The text fields of an object, a referred object's text fields one level deep."""
        fields: dict[str, Any] = {}
        for child in node:
            tag = child.tag.rsplit("}", 1)[-1]
            if child.text and child.text.strip():
                fields[tag] = child.text.strip()
            elif child.get("href") and (target := ids.get(child.get("href")[1:])) is not None:
                texts = {c.tag.rsplit("}", 1)[-1]: c.text.strip() for c in target
                         if c.text and c.text.strip()}
                if texts:
                    fields[tag] = texts
                elif target.text and target.text.strip() and len(target.text) < 1000:
                    fields[tag] = target.text.strip()
        return fields

    @staticmethod
    def parameters(table: bytes) -> tuple[dict[str, Any], dict[str, list[str]]]:
        if len(table) < 32:
            return {}, {}
        count = struct.unpack_from("<I", table, 28)[0]
        values: dict[str, Any] = {}
        choices: dict[str, list[str]] = {}
        for i in range(count):
            base = 32 + 50 * i
            if base + 50 > len(table):
                break
            name = table[base : base + 33].split(b"\0")[0].decode("latin-1")
            kind, n, offset1, offset2 = struct.unpack_from("<4I", table, base + 34)
            first, second = base + 42 + offset1, base + 46 + offset2
            if kind == 0:
                value: list = list(struct.unpack_from(f"<{n}f", table, second))
                value = [float(f"{v:.7g}") for v in value]  # float32 as written
            elif kind == 1:
                value = list(struct.unpack_from(f"<{n}i", table, second))
            elif kind == 2:
                value = [table[second + 81 * k : second + 81 * (k + 1)].split(b"\0")[0].decode(
                    "latin-1") for k in range(n)]
            elif kind == 4:
                end = table.find(b"\0", first)
                options = table[first:end].decode("latin-1").split(",")
                indices = list(struct.unpack_from(f"<{n}i", table, second))
                if options == [""]:  # no choices written (DICOM Raw Data Storage): the indices
                    value = indices
                else:
                    choices[name] = options
                    value = [options[k] if 0 <= k < len(options) else k for k in indices]
            else:
                value = [f"type {kind} not read"]
            values[name] = value[0] if len(value) == 1 else value
        return values, choices

    @staticmethod
    def _on(parameters: dict, name: str) -> bool:
        """A NO/YES switch set to YES (index 1 where the table writes no choices)."""
        return parameters.get(name) in ("YES", 1)

    def _keys(self, meta: Meta, parameters: dict, general: dict) -> None:
        actual = parameters.get("IF_act_rep_time_echo_time")
        numbers = re.findall(r"\d+(?:\.\d+)?", actual) if isinstance(actual, str) else []
        field = "parameters/IF_act_rep_time_echo_time"
        if numbers:
            meta.set("RepetitionTime", float(numbers[0]) / 1000,
                     f"{self.SOURCE}: IF_act_rep_time_echo_time (ms)", field)
        if len(numbers) > 1:
            meta.set("EchoTime", float(numbers[1]) / 1000,
                     f"{self.SOURCE}: IF_act_rep_time_echo_time (ms)", field)
        for key, names in (("RepetitionTime", ("EX_ACQ_ffe_rep_time", "EX_ACQ_se_rep_time")),
                           ("EchoTime", ("EX_ACQ_first_echo_time",))):
            for name in names:
                value = parameters.get(name)
                if key in meta or not isinstance(value, (int, float)):
                    continue
                if parameters.get(f"{name}_enum", "USER_DEF") != "USER_DEF":
                    continue
                meta.set(key, value / 1000, f"{self.SOURCE}: {name} (ms)", f"parameters/{name}")
        single_voxel = parameters.get("EX_ACQ_scan_mode") in ("SV", 4)
        for key, (name, divisor) in self.KEYS.items():
            value = parameters.get(name)
            if not isinstance(value, (int, float)):
                continue
            if key == "MultibandAccelerationFactor" and not self._on(parameters, "EX_MB_multiband"):
                continue
            if key == "SliceThickness" and single_voxel:
                continue
            meta.set(key, value / divisor if divisor else value, f"{self.SOURCE}: {name}",
                     f"parameters/{name}")
        if self._on(parameters, "EX_GEO_sense_enable"):
            for key, name in self.SENSE.items():
                if isinstance(parameters.get(name), (int, float)):
                    meta.set(key, parameters[name], f"{self.SOURCE}: {name} (SENSE on)",
                             f"parameters/{name}")
        nucleus = parameters.get("EX_ACQ_nucleus")
        if isinstance(nucleus, str) and (match := re.fullmatch(r"([A-Za-z]+)(\d+)", nucleus)):
            meta.set("ResonantNucleus", match.group(2) + match.group(1),
                     f"{self.SOURCE}: EX_ACQ_nucleus (element and mass swapped)",
                     "parameters/EX_ACQ_nucleus")
        position = general.get("patientOrientation", "")
        if position.startswith("PatientPos_"):
            meta.set("PatientPosition", position.removeprefix("PatientPos_"),
                     f"{self.SOURCE}: patientOrientation", "exam_card/patientOrientation")
