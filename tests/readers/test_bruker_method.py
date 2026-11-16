####################################################################################################
#                                      test_bruker_method.py                                       #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 16/11/26                                                                                #
#                                                                                                  #
# Purpose: A Bruker scan folder whose method file is renamed <name>_method reads as with the file  #
#          named method; two such files are not guessed between.                                   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import shutil

import numpy as np
import pytest

# own
import miox

SCAN = "spec2nii_test_data/bruker/PV-360.1.1/15"
PER_READ = ("ConversionTime", "OriginalFile")  # differ between any two reads


@pytest.fixture
def scans(public_data, tmp_path):
    source = public_data / SCAN
    if not source.is_dir():
        pytest.skip("file not fetched")
    named, renamed = tmp_path / "named" / "15", tmp_path / "renamed" / "15"
    shutil.copytree(source, named)
    shutil.copytree(source, renamed)
    (renamed / "method").rename(renamed / "scan_15_method")
    return named, renamed


def test_renamed_method_file_reads_as_method(scans):
    named, renamed = scans
    [expected] = miox.load(named / "fid")
    [got] = miox.load(renamed / "fid")
    np.testing.assert_array_equal(got.data, expected.data)
    assert {k: v for k, v in got.meta.items() if k not in PER_READ} == {
        k: v for k, v in expected.meta.items() if k not in PER_READ}
    assert "method file read from scan_15_method" in got.provenance.notes


def test_two_renamed_method_files_are_not_guessed(scans):
    _, renamed = scans
    shutil.copy(renamed / "scan_15_method", renamed / "other_method")
    with pytest.raises(miox.ReaderError, match="no method file"):
        miox.load(renamed / "fid")
