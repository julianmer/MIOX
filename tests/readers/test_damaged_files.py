####################################################################################################
#                                      test_damaged_files.py                                       #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 19/11/26                                                                                #
#                                                                                                  #
# Purpose: Files a reader or a sniff rule trips over: a damaged file is sniffed as unknown with the#
#          failing rule named, whatever a reader raises reaches the caller as a ReaderError, the   #
#          command line reports a missing path, and a page that cannot be drawn falls back to the  #
#          details, with file-derived text escaped.                                                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import pytest

# own
import miox
from miox import quicklook
from miox.cli import CLI
from miox.readers.nifti_mrs import NiftiMrsReader
from tests.conftest import PUBLIC_DATA

NIFTI = PUBLIC_DATA / "nifti_mrs_examples/example_05.nii.gz"


@pytest.mark.data
def test_truncated_file_is_sniffed_as_unknown_with_the_failing_rule_named(tmp_path):
    if not NIFTI.exists():
        pytest.skip("file not fetched")
    cut = tmp_path / "cut.nii.gz"
    cut.write_bytes(NIFTI.read_bytes()[:400])
    sniff = miox.sniff(cut)
    assert sniff.format == miox.Format.UNKNOWN
    assert any("rule:" in line for line in sniff.evidence)


@pytest.mark.data
def test_whatever_a_reader_raises_becomes_a_reader_error(monkeypatch):
    if not NIFTI.exists():
        pytest.skip("file not fetched")

    def read(self, sniff):
        raise KeyError("missing")

    monkeypatch.setattr(NiftiMrsReader, "read", read)
    with pytest.raises(miox.ReaderError, match="example_05.nii.gz: not readable as .*KeyError"):
        miox.load(NIFTI)


def test_command_line_reports_a_missing_path(tmp_path, capsys):
    assert CLI().run(["info", str(tmp_path / "nope.dat")]) == 1
    assert "no such file or folder" in capsys.readouterr().err


@pytest.mark.data
def test_page_that_cannot_be_drawn_shows_the_details(monkeypatch):
    if not NIFTI.exists():
        pytest.skip("file not fetched")

    def html(self):
        raise ZeroDivisionError("division by zero")

    datasets = miox.load(NIFTI)
    monkeypatch.setattr(quicklook.Page, "html", html)
    page = quicklook.page(datasets, "example")
    assert "could not be drawn (ZeroDivisionError" in page and "SpectrometerFrequency" in page


@pytest.mark.data
def test_file_derived_text_is_escaped_in_the_page():
    if not NIFTI.exists():
        pytest.skip("file not fetched")
    page = quicklook.page(miox.load(NIFTI), "example", notes=("<img src=x onerror=alert(1)>",))
    assert "<img" not in page and "&lt;img src=x" in page
