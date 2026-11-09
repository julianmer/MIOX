####################################################################################################
#                                        test_provenance.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 09/11/26                                                                                #
#                                                                                                  #
# Purpose: The package versions a reader records: MIOX's own comes from the module, not from the   #
#          installed distribution (which is named pymiox).                                         #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest

# own
import miox
from miox.readers import base
from miox.sniff import Format, Kind, Sniff


def test_miox_version_comes_from_the_module_not_the_distribution(monkeypatch: pytest.MonkeyPatch):
    def no_miox_distribution(name: str) -> str:
        if name == "miox":
            raise PackageNotFoundError(name)
        return "1.0"

    monkeypatch.setattr(base, "version", no_miox_distribution)
    reader = base.Reader()
    sniff = Sniff(path=Path("x.sdat"), format=Format.PHILIPS_SPAR_SDAT, kind=Kind.SPECTROSCOPY)
    versions = reader.provenance(sniff).versions
    assert versions["miox"] == miox.__version__
    assert all(v == "1.0" for name, v in versions.items() if name != "miox")
