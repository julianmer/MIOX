####################################################################################################
#                                        update_records.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 01/10/26                                                                                #
#                                                                                                  #
# Purpose: Regenerates the records from the public data under tests/data: sniff.json for every     #
#          file except obvious non-data, read.json for every file with a reader. Records of files  #
#          no longer present are kept so a subset fetch never erases the full set.                 #
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
from miox import Format, Sniffer  # noqa: E402
from tests.conftest import PUBLIC_DATA, RECORDS  # noqa: E402
from tests.records.records import ReadRecords  # noqa: E402


#**************************************************************************************************#
#                                          RecordUpdater                                           #
#**************************************************************************************************#
#                                                                                                  #
# Sniffs every public data file and reads every one that has a reader, merging the results into    #
# the records.                                                                                     #
#                                                                                                  #
#**************************************************************************************************#
class RecordUpdater:
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
        ".zip",
        ".part",  # a download in progress
    }
    IGNORE_NAMES = {".DS_Store"}

    def __init__(self):
        self.sniffer = Sniffer()
        self.reads = ReadRecords()
        self.out = RECORDS / "sniff.json"

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
        records = self._one_slice_per_series(records)
        self.out.write_text(json.dumps(dict(sorted(records.items())), indent=1) + "\n")
        print(f"{n} files sniffed, {len(records)} records in {self.out}")

    @staticmethod
    def _one_slice_per_series(records: dict) -> dict:
        """DICOM image files are recorded one per folder: the series are checked as wholes."""
        kept, folders = {}, set()
        for key, record in sorted(records.items()):
            folder = key.rsplit("/", 1)[0]
            if record["format"] == str(Format.DICOM_IMAGE):
                if folder in folders:
                    continue
                folders.add(folder)
            kept[key] = record
        return kept

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
    sys.exit(RecordUpdater().run())
