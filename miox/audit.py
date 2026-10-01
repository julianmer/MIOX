####################################################################################################
#                                             audit.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 01/10/26                                                                                #
#                                                                                                  #
# Purpose: Coverage audit: what a file carries versus what the ecosystem keeps. Milestone 0 covers #
#          NIfTI-MRS files only, where the header extension is the whole metadata record; each key #
#          is classified as standard, user-defined or a nested vendor block and marked whether the #
#          MIOX key table knows it. Other formats follow with their readers (milestone 1).      #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# own
from miox.meta.keys import KEYS
from miox.sniff import NIFTI_MRS, sniff

STANDARD = "standard"
USER_DEFINED = "user-defined"
VENDOR_BLOCK = "vendor-block"


@dataclass(frozen=True)
class Field:
    path: str
    status: str
    in_key_table: bool
    value: str
    """Short preview of the value."""


@dataclass(frozen=True)
class Coverage:
    path: Path
    format: str
    fields: tuple[Field, ...]

    def summary(self) -> dict[str, int]:
        counts = {STANDARD: 0, USER_DEFINED: 0, VENDOR_BLOCK: 0, "in_key_table": 0}
        for f in self.fields:
            counts[f.status] += 1
            counts["in_key_table"] += f.in_key_table
        return counts

    def to_markdown(self) -> str:
        s = self.summary()
        lines = [
            f"# Coverage: `{self.path.name}` ({self.format})",
            "",
            f"{len(self.fields)} header fields: {s[STANDARD]} standard, {s[USER_DEFINED]} "
            f"user-defined, {s[VENDOR_BLOCK]} vendor blocks; {s['in_key_table']} in the MIOX "
            "key table.",
            "",
            "| field | status | key table | value |",
            "|---|---|---|---|",
        ]
        for f in self.fields:
            mark = "yes" if f.in_key_table else ""
            lines.append(f"| `{f.path}` | {f.status} | {mark} | {f.value} |")
        return "\n".join(lines) + "\n"


def audit(path: str | Path) -> Coverage:
    """Audit the metadata coverage of one file."""
    path = Path(path)
    s = sniff(path)
    if s.format != NIFTI_MRS:
        raise NotImplementedError(
            f"audit of {s.format} files needs its reader (milestone 1); only NIfTI-MRS for now"
        )
    return _audit_nifti_mrs(path)


def _audit_nifti_mrs(path: Path) -> Coverage:
    import nibabel as nib

    img = nib.load(path)
    ext = next(e for e in img.header.extensions if e.get_code() == 44)
    hdr = json.loads(ext.get_content())
    fields = []
    for name, value in hdr.items():
        if name in KEYS:
            status = STANDARD
        elif isinstance(value, dict) and not _is_standard_object(name):
            status = VENDOR_BLOCK
        else:
            status = USER_DEFINED
        fields.append(Field(name, status, name in KEYS, _preview(value)))
    return Coverage(path, NIFTI_MRS, tuple(fields))


def _is_standard_object(name: str) -> bool:
    return name in KEYS


def _preview(value, limit: int = 60) -> str:
    text = json.dumps(value, default=str)
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text.replace("|", "\\|")
