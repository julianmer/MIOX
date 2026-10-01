####################################################################################################
#                                           conftest.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 01/10/26                                                                                #
#                                                                                                  #
# Purpose: Pytest fixtures: test data roots. Public data lives under tests/data (fetched by        #
#          scripts/fetch_test_data.py, never committed). Private sets are given as a path-         #
#          separated list of directories in MIOX_PRIVATE_DATA; their paths never appear in      #
#          committed records.                                                                      #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import os
from pathlib import Path

import pytest

PUBLIC_DATA = Path(__file__).parent / "data"
GOLDEN = Path(__file__).parent / "golden"


@pytest.fixture(scope="session")
def public_data() -> Path:
    if not PUBLIC_DATA.is_dir():
        pytest.skip("no test data; run scripts/fetch_test_data.py")
    return PUBLIC_DATA


@pytest.fixture(scope="session")
def private_roots() -> list[Path]:
    value = os.environ.get("MIOX_PRIVATE_DATA", "")
    roots = [Path(p) for p in value.split(os.pathsep) if p]
    if not roots:
        pytest.skip("MIOX_PRIVATE_DATA not set")
    return roots
