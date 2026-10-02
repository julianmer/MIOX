####################################################################################################
#                                             audit.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 01/10/26                                                                                #
#                                                                                                  #
# Purpose: Coverage audit: every field of a file's vendor header, and whether a standard key, some #
#          other key, or nothing is read from it.                                                  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

# own
from miox.meta.keys import KeyTable
from miox.readers import Loader
from miox.sniff import Format


#**************************************************************************************************#
#                                              Status                                              #
#**************************************************************************************************#
#                                                                                                  #
# What is read from a header field.                                                                #
#                                                                                                  #
#**************************************************************************************************#
class Status(StrEnum):
    STANDARD = "standard"  # a key of the key table
    OTHER = "other"  # a key outside the key table, e.g. a user-defined NIfTI-MRS key
    UNUSED = "unused"  # nothing


#**************************************************************************************************#
#                                              Field                                               #
#**************************************************************************************************#
#                                                                                                  #
# One field of an audited vendor header.                                                           #
#                                                                                                  #
#**************************************************************************************************#
@dataclass(frozen=True)
class Field:
    path: str
    status: Status
    key: str
    """The key read from this field, empty if none."""
    value: str
    """Short preview of the value."""


#**************************************************************************************************#
#                                             Coverage                                             #
#**************************************************************************************************#
#                                                                                                  #
# The audit of one file: all fields of its vendor header, with counts and a Markdown table.        #
#                                                                                                  #
#**************************************************************************************************#
@dataclass(frozen=True)
class Coverage:
    path: Path
    format: Format
    fields: tuple[Field, ...]

    def summary(self) -> dict[Status, int]:
        counts = dict.fromkeys(Status, 0)
        for f in self.fields:
            counts[f.status] += 1
        return counts

    def to_markdown(self) -> str:
        s = self.summary()
        lines = [
            f"# Coverage: `{self.path.name}` ({self.format})",
            "",
            f"{len(self.fields)} header fields: {s[Status.STANDARD]} read into standard keys, "
            f"{s[Status.OTHER]} into other keys, {s[Status.UNUSED]} unused.",
            "",
            "| field | status | key | value |",
            "|---|---|---|---|",
        ]
        for f in self.fields:
            lines.append(f"| `{f.path}` | {f.status} | {f.key} | {f.value} |")
        return "\n".join(lines) + "\n"


#**************************************************************************************************#
#                                             Auditor                                              #
#**************************************************************************************************#
#                                                                                                  #
# Loads a file and matches its vendor header against the fields its keys were read from. A key     #
# read from a nested block covers every field inside it.                                           #
#                                                                                                  #
#**************************************************************************************************#
class Auditor:
    PREVIEW_LIMIT = 60  # characters of a value shown in the table

    def __init__(self):
        self.loader = Loader()
        self.keys = KeyTable()

    def audit(self, path: str | Path) -> Coverage:
        """Audit the metadata coverage of one file (all datasets in it)."""
        path = Path(path)
        datasets = self.loader.load(path)
        read_from = {}  # vendor field -> key
        for d in datasets:
            read_from.update({field: key for key, field in d.meta.fields.items()})
        fields = []
        for field, value in datasets[0].meta.vendor_fields().items():
            key = self._key(field, read_from)
            if not key:
                status = Status.UNUSED
            elif key in self.keys:
                status = Status.STANDARD
            else:
                status = Status.OTHER
            fields.append(Field(field, status, key, self._preview(value)))
        return Coverage(path, datasets[0].provenance.format, tuple(fields))

    @staticmethod
    def _key(field: str, read_from: dict[str, str]) -> str:
        parts = field.split("/")
        for n in range(len(parts), 0, -1):
            key = read_from.get("/".join(parts[:n]))
            if key:
                return key
        return ""

    def _preview(self, value) -> str:
        text = json.dumps(value, default=str)
        if len(text) > self.PREVIEW_LIMIT:
            text = text[: self.PREVIEW_LIMIT - 1] + "…"
        return text.replace("|", "\\|")
