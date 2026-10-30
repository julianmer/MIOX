####################################################################################################
#                                       siemens_protocol.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/10/26                                                                                #
#                                                                                                  #
# Purpose: Readers for Siemens protocol files: single protocols (.pro), exported exam trees        #
#          (.exar1) and printed protocols (XML or text), every measurement protocol with its       #
#          parameters and standard keys.                                                           #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import json
import re
import sqlite3
import uuid
import xml.etree.ElementTree as ET
import zlib
from typing import Any

# own
from miox.data import Protocol
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.readers.siemens_twix_raw import SiemensTwixRaw
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                     SiemensProtocolReader                                        #
#**************************************************************************************************#
#                                                                                                  #
# A protocol is XProtocol text (kept whole) ending with the ASCCONV block, "name = value" lines in #
# Siemens notation (sKSpace.lBaseResolution, alTE[0]) between "### ASCCONV BEGIN" and "END ###";   #
# the block is read into a flat dict (quoted values as text, XA's doubled quotes too, hexadecimal  #
# and decimal numbers as numbers) and gives the standard keys as for twix files (times in µs,      #
# frequencies in Hz; no inversion time when sPrepPulses.ucInversion is 4, the inversion off; all   #
# alTI and alTE of a protocol with several (alTI: lInvContrasts, or two for a double inversion);   #
# the slice thickness of 3D protocols is the slab over the images per slab). An .exar1 file is an  #
# SQLite database of the exam tree: instances (programs, steps, protocols, texts, directories)     #
# with their content raw-deflated as "EDF V1: ContentType=...;" plus JSON, in change sets; the     #
# latest instance of every element counts. A program lists its steps in order as GUIDs; each       #
# measurement step holds one protocol, whose JSON carries the XProtocol text. Labels are texts by  #
# language (English, else the unnamed one, else any). Every measurement step becomes a Protocol    #
# named by its label, located by the directories and the program; the program's steps              #
# (measurement, pause, decision, ...) and the step's own content go to the vendor header.          #
#                                                                                                  #
#**************************************************************************************************#
class SiemensProtocolReader(Reader):
    FORMATS = (Format.SIEMENS_PRO, Format.SIEMENS_EXAR)
    PACKAGES = ("miox",)
    SOURCE = "ASCCONV"
    ASCCONV = re.compile(r"### ASCCONV BEGIN[^\n]*###\n(.*?)### ASCCONV END ###", re.S)
    KEYS = {  # standard key: (ASCCONV field, divisor to the key's unit)
        **{key: (field, divisor) for key, (buffer, field, divisor) in SiemensTwixRaw.KEYS.items()
           if buffer == "Phoenix"},
        "ProtocolName": ("tProtocolName", None),
        "FieldStrength": ("sProtConsistencyInfo.flNominalB0", None),
        "NumberOfAverages": ("lAverages", None),
    }
    MEASUREMENT = "EdfMeasurementStep"
    DOUBLE_INVERSION = 16  # sPrepPulses.ucInversion of a double inversion recovery: two TIs

    def read(self, sniff: Sniff) -> list[Protocol]:
        if sniff.format == Format.SIEMENS_EXAR:
            return self._exam(sniff)
        text = sniff.path.read_text(errors="replace")
        meta = self.meta(text)
        name = meta.get("ProtocolName") or sniff.path.stem
        return [Protocol(name, (), meta, self.provenance(sniff))]

    # --- one protocol ----------------------------------------------------------------------------

    @classmethod
    def ascconv(cls, text: str) -> dict[str, Any]:
        match = cls.ASCCONV.search(text)
        if match is None:
            return {}
        values: dict[str, Any] = {}
        for line in match.group(1).splitlines():
            key, sep, value = line.partition("=")
            if not sep or not key.strip():
                continue
            values[key.strip()] = cls._value(value.strip())
        return values

    @staticmethod
    def _value(text: str) -> Any:
        if text.startswith('""') and text.endswith('""') and len(text) >= 4:
            return text[2:-2]  # XA writes strings in doubled quotes
        if text.startswith('"'):
            end = text.rfind('"')
            return text[1:end] if end > 0 else text[1:]
        text = text.split("#", 1)[0].strip()
        try:
            return int(text, 16) if text.lower().startswith("0x") else int(text)
        except ValueError:
            try:
                return float(text)
            except ValueError:
                return text

    def meta(self, text: str) -> Meta:
        values = self.ascconv(text)
        xprotocol = self.ASCCONV.sub("", text).strip()
        meta = Meta({"ascconv": values, "xprotocol": xprotocol})
        for key, (field, divisor) in self.KEYS.items():
            value = values.get(field)
            if value is None or value == "":
                continue
            if key == "InversionTime" and values.get(SiemensTwixRaw.INVERSION) == (
                SiemensTwixRaw.INVERSION_OFF
            ):
                continue
            if divisor is not None:
                if not isinstance(value, (int, float)):
                    continue
                value = value / divisor
            meta.set(key, value, f"{self.SOURCE}: {field}", f"ascconv/{field}")
        inversion = values.get(SiemensTwixRaw.INVERSION)
        count = 2 if inversion == self.DOUBLE_INVERSION else values.get("lInvContrasts", 1)
        times = [values.get(f"alTI[{i}]") for i in range(count if isinstance(count, int) else 1)]
        if len(times) > 1 and all(isinstance(t, (int, float)) for t in times) and (
            inversion != SiemensTwixRaw.INVERSION_OFF
        ):
            meta.set("InversionTime", [t / 1e6 for t in times],
                     f"{self.SOURCE}: alTI[0..{len(times) - 1}] (lInvContrasts or double "
                     "inversion)", "ascconv/alTI[0]")
        slab, images = values.get("sSliceArray.asSlice[0].dThickness"), values.get(
            "sKSpace.lImagesPerSlab")
        if values.get("sKSpace.ucDimension") == 4 and isinstance(slab, (int, float)) and images:
            meta.set("SliceThickness", slab / images, f"{self.SOURCE}: sSliceArray.asSlice[0]."
                     "dThickness / sKSpace.lImagesPerSlab (3D)", "ascconv/sKSpace.lImagesPerSlab")
        contrasts = values.get("lContrasts")
        if isinstance(contrasts, int) and contrasts > 1:
            echoes = [values.get(f"alTE[{i}]") for i in range(contrasts)]
            if all(isinstance(e, (int, float)) for e in echoes):
                meta.set("EchoTime", [e / 1e6 for e in echoes],
                         f"{self.SOURCE}: alTE[0..{contrasts - 1}] (lContrasts)", "ascconv/alTE[0]")
        return meta

    # --- exam trees ------------------------------------------------------------------------------

    def _exam(self, sniff: Sniff) -> list[Protocol]:
        try:
            db = sqlite3.connect(f"file:{sniff.path}?mode=ro", uri=True)
            records = db.execute("SELECT Hash, Data FROM Content")
            contents = {h: self._content(d) for h, d in records}
            changes = dict(db.execute("SELECT Id, DateTime FROM ChangeSet"))
            versions = {}
            for instance, change, element in db.execute(
                "SELECT InstanceId, ChangeSetId, ElementId FROM InstanceChangeSet"
            ):
                when = changes.get(change, "")
                if element not in versions or when > versions[element][0]:
                    versions[element] = (when, instance)
            columns = ("Id", "Element_id", "ContentHash", "Children", "Tags", "LabelElement_id",
                       "InstanceType", "ObjectId")
            rows = db.execute(f"SELECT {', '.join(columns)} FROM Instance").fetchall()
        except sqlite3.DatabaseError as error:
            raise ReaderError(f"{sniff.path.name}: not an exam tree database: {error}") from error
        current = {versions[r[1]][1] for r in rows if r[1] in versions}
        elements = {r[1]: dict(zip(columns, r, strict=True)) for r in rows if r[0] in current}
        directories = self._directories(elements, contents)
        protocols = []
        for program in (e for e in elements.values() if e["InstanceType"] == "EdfProgram"):
            program_name = self._label(program, elements, contents)
            location = (*directories.get(program["Element_id"], ()), program_name)
            steps = [elements[s] for s in self._children(program) if s in elements]
            summary = [{"type": s["InstanceType"].removeprefix("Edf"),
                        "label": self._label(s, elements, contents)} for s in steps]
            for step in steps:
                if step["InstanceType"] != self.MEASUREMENT:
                    continue
                protocol = next((elements[c] for c in self._children(step) if c in elements), None)
                text = (contents.get(protocol["ContentHash"], ("", {}))[1].get("Data", "")
                        if protocol else "")
                meta = self.meta(text) if text else Meta({})
                meta.vendor["step"] = contents.get(step["ContentHash"], ("", {}))[1]
                meta.vendor["step_tags"] = step["Tags"]
                meta.vendor["program"] = {"name": program_name, "steps": summary}
                notes = () if text else ("measurement step without protocol",)
                name = self._label(step, elements, contents) or meta.get("ProtocolName", "")
                protocols.append(Protocol(name, location, meta, self.provenance(sniff, notes)))
        if not protocols:
            raise ReaderError(f"{sniff.path.name}: no measurement protocols in the exam tree")
        return protocols

    @staticmethod
    def _content(data: bytes) -> tuple[str, dict]:
        """(content type, JSON) of a raw-deflated "EDF V1: ContentType=...;" record."""
        text = zlib.decompress(data, -15).decode("utf-8", "replace")
        head, _, body = text.partition(";")
        kind = head.split("ContentType=", 1)[-1].rsplit(".", 1)[-1]
        try:
            return kind, json.loads(body)
        except json.JSONDecodeError:
            return kind, {"text": body}

    @staticmethod
    def _children(instance: dict) -> list[str]:
        raw = bytes(instance["Children"] or b"")
        return [str(uuid.UUID(bytes_le=raw[i : i + 16])) for i in range(0, len(raw) - 15, 16)]

    @staticmethod
    def _label(instance: dict, elements: dict, contents: dict) -> str:
        text = elements.get(instance.get("LabelElement_id"))
        if text is None:
            return ""
        texts = {k: v for k, v in contents.get(text["ContentHash"], ("", {}))[1].get(
            "Texts", {}).items() if k != "$id"}
        return texts.get("en") or texts.get("") or next(iter(texts.values()), "")

    def _directories(self, elements: dict, contents: dict) -> dict[str, tuple[str, ...]]:
        """The directory path of every program, from the exam tree's structure."""
        structures = [contents.get(e["ContentHash"], ("", {}))[1] for e in elements.values()
                      if e["InstanceType"] == "EdfStructure"]
        parents = {}
        for structure in structures:
            parents.update({k: v for k, v in structure.get("ParentDirectoryId", {}).items()
                            if k != "$id"})
        folders = {e["ObjectId"]: e for e in elements.values()
                   if e["InstanceType"] == "EdfDirectory"}
        paths = {}
        for element in parents:  # programs by element id, directories by object id
            path, node, seen = [], parents.get(element), set()
            while node in folders and node not in seen:
                seen.add(node)
                path.append(self._label(folders[node], elements, contents))
                node = parents.get(node)
            paths[element] = tuple(reversed(path))
        return paths


#**************************************************************************************************#
#                                  SiemensPrintedProtocolReader                                    #
#**************************************************************************************************#
#                                                                                                  #
# A printed protocol (PrintOut XML, or the same as text) lists every protocol with its path in the #
# exam tree (\\User\region\exam\program\name) and its cards of parameters, each a label and a      #
# value with its unit ("TR 427.0 ms"). The cards are kept whole; labels name the standard keys     #
# without regard to case (the first occurrence counts; "TE 1", "TE 2", ... give a list), values    #
# with ms are converted to s. A trailing " *" of a name (a marked protocol) is kept only in the    #
# vendor header.                                                                                   #
#                                                                                                  #
#**************************************************************************************************#
class SiemensPrintedProtocolReader(Reader):
    FORMATS = (Format.SIEMENS_PRINTPROT,)
    PACKAGES = ("miox",)
    SOURCE = "printed protocol"
    KEYS = {  # lower-case label: (standard key, integer)
        "tr": ("RepetitionTime", False), "te": ("EchoTime", False),
        "ti": ("InversionTime", False), "flip angle": ("ExcitationFlipAngle", False),
        "slice thickness": ("SliceThickness", False), "averages": ("NumberOfAverages", True),
        "base resolution": ("BaseResolution", True),
        "echo spacing": ("VendorReportedEchoSpacing", False),
        "turbo factor": ("EchoTrainLength", True), "bandwidth": ("PixelBandwidth", False),
        "accel. factor pe": ("ParallelReductionFactorInPlane", True),
        "acceleration factor pe": ("ParallelReductionFactorInPlane", True),
        "accel. factor 3d": ("ParallelReductionFactorOutOfPlane", True),
        "acceleration factor 3d": ("ParallelReductionFactorOutOfPlane", True),
        "ref. lines pe": ("RefLinesPE", True), "reference lines pe": ("RefLinesPE", True),
    }
    VALUE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*([A-Za-z/%]*)\s*$")
    TXT_PARAMETER = re.compile(r"^\s{2,}(\S.*?)(?:\s{2,}(.*?))?\s*$")

    def read(self, sniff: Sniff) -> list[Protocol]:
        text = sniff.path.read_bytes().decode("utf-8-sig", errors="replace")
        entries = self._xml(text) if text.lstrip().startswith("<") else self._text(text)
        if not entries:
            raise ReaderError(f"{sniff.path.name}: no protocols in the printed protocol")
        protocols = []
        for path, header, cards in entries:
            parts = [p for p in path.split("\\") if p]
            name = parts[-1].removesuffix(" *") if parts else sniff.path.stem
            meta = Meta({"path": path, "header": header, "cards": cards})
            self._keys(meta, cards)
            protocols.append(Protocol(name, tuple(parts[:-1]), meta, self.provenance(sniff)))
        return protocols

    def _xml(self, text: str) -> list[tuple[str, str, dict]]:
        root = ET.fromstring(text)
        entries = []
        for node in root.iter():
            info = node.find("ProtHeaderInfo")
            if info is None or node.find("Card") is None:
                continue
            cards: dict[str, list] = {}
            for card in node.findall("Card"):
                rows = [(p.findtext("Label") or "", (p.findtext("ValueAndUnit") or "").strip())
                        for p in card.iter("ProtParameter")]
                cards.setdefault(card.get("name", ""), []).extend(rows)
            entries.append((info.findtext("HeaderProtPath") or "",
                            (info.findtext("HeaderProperty") or "").strip(), cards))
        return entries

    def _text(self, text: str) -> list[tuple[str, str, dict]]:
        lines = re.sub(r"\r+\n|\r", "\n", text).split("\n")  # some files end lines with \r\r\r\n
        entries = []
        i = 0
        while i < len(lines):
            if not lines[i].strip().startswith("SIEMENS MAGNETOM"):
                i += 1
                continue
            path, header = lines[i + 2].strip(), lines[i + 3].strip()
            cards: dict[str, list] = {}
            card, i = "", i + 5
            while i < len(lines) and not lines[i].strip().startswith("SIEMENS MAGNETOM"):
                line = lines[i]
                if line.strip() and not line.startswith(" "):
                    card = line.strip()
                    cards.setdefault(card, [])
                elif (match := self.TXT_PARAMETER.match(line)) and card:
                    cards[card].append((match.group(1), (match.group(2) or "").strip()))
                i += 1
            entries.append((path, header, cards))
        return entries

    def _keys(self, meta: Meta, cards: dict[str, list]) -> None:
        echoes, inversions = [], []
        for card, rows in cards.items():
            for label, value in rows:
                lower = label.strip().lower()
                match = self.VALUE.match(value)
                if match is None:
                    continue
                number = float(match.group(1))
                number = round(number / 1000, 9) if match.group(2) == "ms" else number
                for prefix, found in (("te", echoes), ("ti", inversions)):
                    if re.fullmatch(rf"{prefix} \d+", lower) and (label, number) not in found:
                        found.append((label, number))
                if re.fullmatch(r"t[ei] \d+", lower):
                    continue
                if lower not in self.KEYS:
                    continue
                key, integer = self.KEYS[lower]
                if key not in meta:
                    value = int(number) if integer else number
                    meta.set(key, value, f"{self.SOURCE}: {card}/{label}", f"cards/{card}")
        for key, found in (("EchoTime", echoes), ("InversionTime", inversions)):
            if found and key not in meta:
                values = [number for _, number in found]
                meta.set(key, values[0] if len(values) == 1 else values,
                         f"{self.SOURCE}: {', '.join(label for label, _ in found)}", "cards")
