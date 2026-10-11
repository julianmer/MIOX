####################################################################################################
#                                         test_archive.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 10/10/26                                                                                #
#                                                                                                  #
# Purpose: Zip archives on synthetic files: the datasets inside a zip (also in a zip within a zip  #
#          and in folders of unknown format) with provenance paths into the archive, members that  #
#          would land outside the unpacked folder, archives with nothing readable, and Office      #
#          documents (zips too) left to other rules.                                               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

# own
import miox

FID = """##TITLE= test
##JCAMP-DX= 5.01
##DATA TYPE= NMR FID
##DATA CLASS= XYDATA
##.OBSERVE FREQUENCY= 100.0
##.OBSERVE NUCLEUS= ^1H
##.SHIFT REFERENCE= INTERNAL, TMS, 1, 10.0
##NTUPLES= NMR FID
##VAR_NAME= TIME, FID/REAL, FID/IMAG
##SYMBOL= X, R, I
##VAR_DIM= 4, 4, 4
##UNITS= SECONDS, ARBITRARY UNITS, ARBITRARY UNITS
##FIRST= 0, 1, 0
##LAST= 0.003, 4, 0
##PAGE= N=1
##DATA TABLE= (X++(R..R)), XYDATA
0 1 2 3 4
##PAGE= N=2
##DATA TABLE= (X++(I..I)), XYDATA
0 0 0 0 0
##END NTUPLES= NMR FID
##END=
"""


def archive(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return path


def nested(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return buffer.getvalue()


def test_datasets_inside_a_zip(tmp_path):
    path = archive(tmp_path / "export.zip", {
        "study/notes.txt": b"nothing to read here\n",
        "study/sample/fid.jdx": FID.encode(),
        "inner.zip": nested({"second.jdx": FID.encode()}),
    })
    sniff = miox.sniff(path)
    assert (sniff.format, sniff.kind) == (miox.Format.ZIP_ARCHIVE, miox.Kind.ARCHIVE)
    datasets = miox.load(path)
    assert len(datasets) == 2
    paths = sorted(str(d.provenance.paths[0]) for d in datasets)
    assert paths == [str(path / "inner.zip" / "second.jdx"),
                     str(path / "study" / "sample" / "fid.jdx")]
    assert np.allclose(datasets[0].data.real.ravel(), [1, 2, 3, 4])
    assert all(any("zip archive" in note for note in d.provenance.notes) for d in datasets)


def test_member_outside_the_folder_is_refused(tmp_path):
    path = archive(tmp_path / "evil.zip", {"../escape.jdx": FID.encode()})
    with pytest.raises(miox.ReaderError, match="outside"):
        miox.load(path)


def test_nothing_readable(tmp_path):
    path = archive(tmp_path / "text.zip", {"readme.txt": b"hello\n"})
    with pytest.raises(miox.ReaderError):
        miox.load(path)


def test_office_documents_are_not_archives(tmp_path):
    path = archive(tmp_path / "table.xlsx", {"[Content_Types].xml": b"<Types/>",
                                             "xl/workbook.xml": b"<workbook/>"})
    assert miox.sniff(path).format != miox.Format.ZIP_ARCHIVE
