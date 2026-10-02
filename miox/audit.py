####################################################################################################
#                                             audit.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 01/10/26                                                                                #
#                                                                                                  #
# Purpose: Coverage audit: every metadata field a file carries, and whether the key table knows    #
#          it. Supports NIfTI-MRS files, where the header extension is the whole metadata record;  #
#          each key is classified as standard, user-defined or a nested vendor block.              #
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

import nibabel as nib

# own
from miox.meta.keys import KeyTable
from miox.sniff import Format, Sniffer


#**************************************************************************************************#
#                                              Status                                              #
#**************************************************************************************************#
#                                                                                                  #
# How a header field relates to the standard.                                                      #
#                                                                                                  #
#**************************************************************************************************#
class Status(StrEnum):
    STANDARD = "standard"
    USER_DEFINED = "user-defined"
    VENDOR_BLOCK = "vendor-block"


#**************************************************************************************************#
#                                              Field                                               #
#**************************************************************************************************#
#                                                                                                  #
# One header field of an audited file.                                                             #
#                                                                                                  #
#**************************************************************************************************#
@dataclass(frozen=True)
class Field:
    path: str
    status: Status
    in_key_table: bool
    value: str
    """Short preview of the value."""


#**************************************************************************************************#
#                                             Coverage                                             #
#**************************************************************************************************#
#                                                                                                  #
# The audit of one file: all its header fields, with counts and a Markdown table.                  #
#                                                                                                  #
#**************************************************************************************************#
@dataclass(frozen=True)
class Coverage:
    path: Path
    format: Format
    fields: tuple[Field, ...]

    def summary(self) -> dict[str, int]:
        counts = {status: 0 for status in Status} | {"in_key_table": 0}
        for f in self.fields:
            counts[f.status] += 1
            counts["in_key_table"] += f.in_key_table
        return counts

    def to_markdown(self) -> str:
        s = self.summary()
        lines = [
            f"# Coverage: `{self.path.name}` ({self.format})",
            "",
            f"{len(self.fields)} header fields: {s[Status.STANDARD]} standard, "
            f"{s[Status.USER_DEFINED]} user-defined, {s[Status.VENDOR_BLOCK]} vendor blocks; "
            f"{s['in_key_table']} in the MIOX key table.",
            "",
            "| field | status | key table | value |",
            "|---|---|---|---|",
        ]
        for f in self.fields:
            mark = "yes" if f.in_key_table else ""
            lines.append(f"| `{f.path}` | {f.status} | {mark} | {f.value} |")
        return "\n".join(lines) + "\n"


#**************************************************************************************************#
#                                             Auditor                                              #
#**************************************************************************************************#
#                                                                                                  #
# Audits the metadata coverage of files against the key table.                                     #
#                                                                                                  #
#**************************************************************************************************#
class Auditor:
    PREVIEW_LIMIT = 60  # characters of a value shown in the table

    def __init__(self):
        self.sniffer = Sniffer()
        self.keys = KeyTable()

    def audit(self, path: str | Path) -> Coverage:
        """Audit the metadata coverage of one file."""
        path = Path(path)
        fmt = self.sniffer.sniff(path).format
        if fmt != Format.NIFTI_MRS:
            raise NotImplementedError(f"only NIfTI-MRS files can be audited, not {fmt}")
        return self.nifti_mrs(path)

    def nifti_mrs(self, path: Path) -> Coverage:
        """Classify every key of the NIfTI-MRS header extension."""
        image = nib.load(path)
        extension = next(e for e in image.header.extensions if e.get_code() == 44)
        header = json.loads(extension.get_content())
        fields = []
        for name, value in header.items():
            if name in self.keys:
                status = Status.STANDARD
            elif isinstance(value, dict):
                status = Status.VENDOR_BLOCK
            else:
                status = Status.USER_DEFINED
            fields.append(Field(name, status, name in self.keys, self._preview(value)))
        return Coverage(path, Format.NIFTI_MRS, tuple(fields))

    def _preview(self, value) -> str:
        text = json.dumps(value, default=str)
        if len(text) > self.PREVIEW_LIMIT:
            text = text[: self.PREVIEW_LIMIT - 1] + "…"
        return text.replace("|", "\\|")
