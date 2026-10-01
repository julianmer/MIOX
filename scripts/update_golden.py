####################################################################################################
#                                         update_golden.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 01/10/26                                                                                #
#                                                                                                  #
# Purpose: Regenerates tests/golden/sniff.json from the public data under tests/data. Every file   #
#          is recorded except obvious non-data; records of files no longer present are kept so a   #
#          subset fetch never erases the full set.                                                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# own
from miox import sniff  # noqa: E402
from tests.conftest import GOLDEN, PUBLIC_DATA  # noqa: E402

IGNORE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".pdf",
    ".md",
    ".py",
    ".mat",
    ".json",
    ".log",
    ".csv",
    ".tex",
    ".gitattributes",
    ".gitignore",
}
IGNORE_NAMES = {".DS_Store"}


def main() -> int:
    out = GOLDEN / "sniff.json"
    records = json.loads(out.read_text()) if out.is_file() else {}
    n = 0
    for path in sorted(q for top in PUBLIC_DATA.iterdir() for q in top.rglob("*")):
        if (
            not path.is_file()
            or path.name in IGNORE_NAMES
            or path.suffix.lower() in IGNORE_SUFFIXES
        ):
            continue
        if ".git" in path.parts or _is_lfs_pointer(path):
            continue
        s = sniff(path)
        records[path.relative_to(PUBLIC_DATA).as_posix()] = {
            "format": s.format,
            "vendor": s.vendor,
            "kind": s.kind,
        }
        n += 1
    out.write_text(json.dumps(dict(sorted(records.items())), indent=1) + "\n")
    print(f"{n} files sniffed, {len(records)} records in {out}")
    return 0


def _is_lfs_pointer(path: Path) -> bool:
    with open(path, "rb") as fh:
        return fh.read(30).startswith(b"version https://git-lfs")


if __name__ == "__main__":
    sys.exit(main())
