####################################################################################################
#                                         update_golden.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 01/10/26                                                                                #
#                                                                                                  #
# Purpose: Regenerates the golden records from the public data under tests/data: sniff.json for    #
#          every file except obvious non-data, read.json for every file with a reader. Records of  #
#          files no longer present are kept so a subset fetch never erases the full set.           #
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
from miox import Sniffer  # noqa: E402
from tests.conftest import GOLDEN, PUBLIC_DATA  # noqa: E402
from tests.golden.records import ReadRecords  # noqa: E402


#**************************************************************************************************#
#                                          GoldenUpdater                                           #
#**************************************************************************************************#
#                                                                                                  #
# Sniffs every public data file and reads every one that has a reader, merging the results into    #
# the golden records.                                                                              #
#                                                                                                  #
#**************************************************************************************************#
class GoldenUpdater:
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

    def __init__(self):
        self.sniffer = Sniffer()
        self.reads = ReadRecords()
        self.out = GOLDEN / "sniff.json"

    def run(self) -> int:
        self.sniff()
        self.read()
        return 0

    def sniff(self) -> None:
        records = json.loads(self.out.read_text()) if self.out.is_file() else {}
        n = 0
        for path in self._data_files():
            s = self.sniffer.sniff(path)
            records[path.relative_to(PUBLIC_DATA).as_posix()] = {
                "format": s.format,
                "vendor": s.vendor,
                "kind": s.kind,
            }
            n += 1
        self.out.write_text(json.dumps(dict(sorted(records.items())), indent=1) + "\n")
        print(f"{n} files sniffed, {len(records)} records in {self.out}")

    def read(self) -> None:
        records = self.reads.expected()
        n = 0
        for path in self.reads.targets():
            records[path.relative_to(PUBLIC_DATA).as_posix()] = self.reads.record(path)
            n += 1
        lines = (f"{json.dumps(k)}: {json.dumps(v)}" for k, v in sorted(records.items()))
        self.reads.PATH.write_text("{\n" + ",\n".join(lines) + "\n}\n")  # one line per file
        print(f"{n} files read, {len(records)} records in {self.reads.PATH}")

    def _data_files(self):
        for path in sorted(q for top in PUBLIC_DATA.iterdir() for q in top.rglob("*")):
            if (
                not path.is_file()
                or path.name in self.IGNORE_NAMES
                or path.suffix.lower() in self.IGNORE_SUFFIXES
            ):
                continue
            if ".git" in path.parts or self.reads.is_lfs_pointer(path):
                continue
            yield path


if __name__ == "__main__":
    sys.exit(GoldenUpdater().run())
