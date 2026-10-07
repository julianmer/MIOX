####################################################################################################
#                                            pulseq.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for Pulseq sequence files (.seq, format revisions 1.2 to 1.5): the version, the  #
#          definitions, the block table, every event table, the extensions with their              #
#          specifications, the decompressed shapes and the signature.                              #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

# own
from miox.data import PulseSequence
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                           PulseqReader                                           #
#**************************************************************************************************#
#                                                                                                  #
# A .seq file is text in sections headed by [NAME]; a line starting with # is a comment. The       #
# columns of the event tables depend on the format revision (RF: 7 columns up to 1.3, 8 in 1.4, 12 #
# from 1.5; gradients: 4, 5 and 7; ADC: 6 up to 1.4, 9 from 1.5; blocks: a delay event id up to    #
# 1.3, the duration in units of BlockDurationRaster from 1.4); the columns are named as the        #
# specification names them. Shapes are stored compressed (the derivative, run-length encoded: two  #
# equal values followed by the number of further repeats); from 1.4 a shape whose stored sample    #
# count equals the declared one is stored as it is, up to 1.3 every shape is compressed.           #
# Extensions are a linked list of (id, type, ref, next) entries and one specification per          #
# extension name with the rows kept as given. The definitions give the keys: Name as the sequence  #
# name, TE, TR and TotalDuration in seconds, FOV in metres.                                        #
#                                                                                                  #
#**************************************************************************************************#
class PulseqReader(Reader):
    FORMATS = (Format.PULSEQ,)
    SOURCE = "seq definitions"
    SECTION = re.compile(r"^\[(\w+)\]\s*$")
    EXTENSION = re.compile(r"^extension\s+(\S+)\s+(\d+)\s*$")
    BLOCK_COLUMNS = {
        "delay": ("id", "delay", "rf", "gx", "gy", "gz", "adc", "ext"),
        "duration": ("id", "duration", "rf", "gx", "gy", "gz", "adc", "ext"),
    }
    COLUMNS = {  # section: columns by count
        "RF": {
            7: ("id", "amp", "mag_id", "phase_id", "delay", "freq", "phase"),
            8: ("id", "amp", "mag_id", "phase_id", "time_id", "delay", "freq", "phase"),
            12: ("id", "amp", "mag_id", "phase_id", "time_id", "center", "delay", "freq_ppm",
                 "phase_ppm", "freq", "phase", "use"),
        },
        "GRADIENTS": {
            4: ("id", "amp", "shape_id", "delay"),
            5: ("id", "amp", "shape_id", "time_id", "delay"),
            7: ("id", "amp", "first", "last", "shape_id", "time_id", "delay"),
        },
        "TRAP": {
            5: ("id", "amp", "rise", "flat", "fall"),
            6: ("id", "amp", "rise", "flat", "fall", "delay"),
        },
        "ADC": {
            6: ("id", "num", "dwell", "delay", "freq", "phase"),
            9: ("id", "num", "dwell", "delay", "freq_ppm", "phase_ppm", "freq", "phase",
                "phase_shape_id"),
        },
        "DELAYS": {2: ("id", "delay")},
        "EXTENSIONS": {4: ("id", "type", "ref", "next")},
    }
    KEYS = {  # standard key: definition, divisor to the key's unit
        "SequenceName": ("Name", None), "EchoTime": ("TE", None), "RepetitionTime": ("TR", None),
        "InversionTime": ("TI", None), "ScanDuration": ("TotalDuration", None),
    }

    def read(self, sniff: Sniff) -> list[PulseSequence]:
        sections = self.sections(sniff.path)
        if "BLOCKS" not in sections:
            raise ReaderError(f"{sniff.path.name}: no [BLOCKS] section")
        version = self._version(sections.get("VERSION", []))
        definitions = self._definitions(sections.get("DEFINITIONS", []))
        notes: list[str] = []
        blocks = self._blocks(sections["BLOCKS"], version)
        events = {
            name.lower(): self._table(name, sections[name], notes)
            for name in ("RF", "GRADIENTS", "TRAP", "ADC", "DELAYS")
            if name in sections
        }
        extensions = self._extensions(sections.get("EXTENSIONS", []), notes)
        shapes = self._shapes(sections.get("SHAPES", []), self._at_least(version, (1, 4)), notes)
        signature = self._definitions(sections.get("SIGNATURE", []))
        meta = self._meta(definitions, version, signature)
        provenance = self.provenance(sniff, tuple(notes))
        return [
            PulseSequence(
                version, definitions, blocks, events, extensions, shapes, signature, meta,
                provenance, sniff.path.stem,
            )
        ]

    # --- text ------------------------------------------------------------------------------------

    @classmethod
    def sections(cls, path: Path) -> dict[str, list[str]]:
        """The non-comment, non-empty lines of every section by its name."""
        sections: dict[str, list[str]] = {}
        current = None
        for raw in path.read_text(errors="replace").splitlines():
            line = raw.strip()
            match = cls.SECTION.match(line)
            if match:
                current = match.group(1).upper()
                sections.setdefault(current, [])
                continue
            if not line or line.startswith("#") or current is None:
                continue
            sections[current].append(line)
        return sections

    @staticmethod
    def _scalar(text: str) -> Any:
        try:
            return int(text)
        except ValueError:
            pass
        try:
            return float(text)
        except ValueError:
            return text

    def _version(self, lines: list[str]) -> str:
        parts = {}
        for line in lines:
            key, _, value = line.partition(" ")
            parts[key.strip()] = value.strip()
        return ".".join(parts.get(k, "0") for k in ("major", "minor", "revision"))

    def _definitions(self, lines: list[str]) -> dict[str, Any]:
        definitions: dict[str, Any] = {}
        for line in lines:
            key, _, rest = line.partition(" ")
            values = [self._scalar(v) for v in rest.split()]
            definitions[key] = values[0] if len(values) == 1 else values
        return definitions

    def _blocks(self, lines: list[str], version: str) -> np.ndarray:
        layout = "duration" if self._at_least(version, (1, 4)) else "delay"
        columns = self.BLOCK_COLUMNS[layout]
        table = np.zeros(len(lines), dtype=[(c, "<i8") for c in columns])
        for i, line in enumerate(lines):
            values = [int(v) for v in line.split()]
            values += [0] * (len(columns) - len(values))  # no extension column before 1.3
            table[i] = tuple(values[: len(columns)])
        return table

    @staticmethod
    def _at_least(version: str, minimum: tuple[int, int]) -> bool:
        parts = tuple(int(p) for p in version.split(".")[:2])
        return parts >= minimum

    def _table(self, name: str, lines: list[str], notes: list[str]) -> dict[int, dict]:
        """The events of a section by id, the columns named by their count."""
        events: dict[int, dict] = {}
        for line in lines:
            fields = line.split()
            columns = self.COLUMNS[name].get(len(fields))
            if columns is None:
                notes.append(f"[{name}] line with {len(fields)} columns not understood: {line}")
                continue
            values = [self._scalar(v) for v in fields]
            events[int(values[0])] = dict(zip(columns[1:], values[1:], strict=True))
        return events

    def _extensions(self, lines: list[str], notes: list[str]) -> dict[str, Any]:
        """The extension list and, per extension name, its type id and rows."""
        entries: dict[int, dict] = {}
        specs: dict[str, dict] = {}
        current = None
        for line in lines:
            match = self.EXTENSION.match(line)
            if match:
                current = match.group(1)
                specs[current] = {"type": int(match.group(2)), "rows": []}
                continue
            values = [self._scalar(v) for v in line.split()]
            if current is None:
                if len(values) == 4:
                    columns = ("type", "ref", "next")
                    entries[int(values[0])] = dict(zip(columns, values[1:], strict=True))
                else:
                    notes.append(f"[EXTENSIONS] list line not understood: {line}")
            else:
                specs[current]["rows"].append(values)
        return {"list": entries, "specifications": specs}

    def _shapes(self, lines: list[str], raw_allowed: bool, notes: list[str]
                ) -> dict[int, np.ndarray]:
        shapes: dict[int, np.ndarray] = {}
        shape_id = None
        count = 0
        samples: list[float] = []

        def finish():
            if shape_id is None:
                return
            values = np.array(samples, dtype=np.float64)
            if len(values) != count or not raw_allowed:
                values = self.decompress(values)
                if len(values) != count:
                    notes.append(
                        f"shape {shape_id}: {len(values)} samples decompressed, {count} declared"
                    )
            shapes[shape_id] = values

        for line in lines:
            key, _, rest = line.partition(" ")
            if key == "shape_id":
                finish()
                shape_id, count, samples = int(rest), 0, []
            elif key == "num_samples":
                count = int(rest)
            else:
                samples.append(float(key))
        finish()
        return shapes

    @staticmethod
    def decompress(packed: np.ndarray) -> np.ndarray:
        """Run-length decoding of the derivative: two equal values are followed by the number
        of further repeats; the samples are the running sum."""
        derivative: list[float] = []
        i = 0
        while i < len(packed):
            value = packed[i]
            if i + 2 < len(packed) and packed[i + 1] == value:
                derivative.extend([value] * (2 + int(packed[i + 2])))
                i += 3
            else:
                derivative.append(value)
                i += 1
        return np.cumsum(derivative)

    # --- keys ------------------------------------------------------------------------------------

    def _meta(self, definitions: dict, version: str, signature: dict) -> Meta:
        meta = Meta({"definitions": definitions, "version": version, "signature": signature})
        for key, (name, divisor) in self.KEYS.items():
            value = definitions.get(name)
            if value is None or isinstance(value, list):
                continue
            if isinstance(value, (int, float)) and divisor:
                value = value / divisor
            meta.set(key, value, f"{self.SOURCE}: {name}", f"definitions/{name}")
        fov = definitions.get("FOV")
        numeric = isinstance(fov, list) and all(isinstance(v, (int, float)) for v in fov)
        if numeric and len(fov) == 3:
            fov_mm = [v * 1000 for v in fov]
            field = "definitions/FOV"
            meta.set("EncodedFieldOfView", fov_mm, f"{self.SOURCE}: FOV in metres", field)
        return meta
