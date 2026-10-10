####################################################################################################
#                                          test_yarra.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 10/10/26                                                                                #
#                                                                                                  #
# Purpose: Yarra reader on a synthetic mode and module manifest (sections, case and % placeholders #
#          kept) and on the fetched example modes (every option of the file in the vendor header). #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re

import pytest

# own
import miox as gu
from tests.conftest import PUBLIC_DATA

MODE = """;##  YarraServer -- Reconstruction Mode Definition

[ClientConfig]
Name=Test Recon (to PACS)
Tag=_TST
RequiresACC=FALSE

[Reconstruction]
Bin="%bu/Test/entrypoint.sh"
Args=--input-file %rid/%rif --output-dir %rod

[PACSTransfer]
IP=203.0.113.0
"""


def test_mode_file(tmp_path):
    path = tmp_path / "Test.mode"
    path.write_text(MODE)
    sniff = gu.sniff(path)
    assert (sniff.format, sniff.kind) == (gu.Format.YARRA_MODE, gu.Kind.PROTOCOL)
    [protocol] = gu.load(path)
    assert protocol.name == "Test Recon (to PACS)"
    assert protocol.meta["ProtocolName"] == "Test Recon (to PACS)"
    vendor = protocol.meta.vendor
    assert vendor["Reconstruction"]["Args"] == "--input-file %rid/%rif --output-dir %rod"
    assert vendor["ClientConfig"]["RequiresACC"] == "FALSE"
    assert vendor["PACSTransfer"]["IP"] == "203.0.113.0"


def test_module_manifest(tmp_path):
    path = tmp_path / "Test.ymf"
    path.write_text("[YarraModule]\nName=Test Module\nVersion=2\nAuthor=Someone\n")
    assert gu.sniff(path).format == gu.Format.YARRA_MODULE
    [protocol] = gu.load(path)
    assert protocol.meta["SoftwareVersions"] == "2"
    assert protocol.meta.vendor["YarraModule"]["Author"] == "Someone"


def test_example_modes_keep_every_option():
    folder = PUBLIC_DATA / "yarra"
    paths = sorted(folder.rglob("*.mode")) + sorted(folder.rglob("*.ymf"))
    if not paths:
        pytest.skip("not fetched")
    for path in paths:
        [protocol] = gu.load(path)
        written = sum(1 for line in path.read_text().splitlines()
                      if re.match(r"^[A-Za-z0-9_]+\s*=", line))
        assert sum(len(options) for options in protocol.meta.vendor.values()) == written
