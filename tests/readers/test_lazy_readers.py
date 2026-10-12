####################################################################################################
#                                      test_lazy_readers.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 12/10/26                                                                                #
#                                                                                                  #
# Purpose: The loader's table of readers matches what every reader class says it reads, each       #
#          format has one reader, and importing MIOX imports no reader module (so a quick look     #
#          pays only for the reader its file needs).                                               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import importlib
import subprocess
import sys

# own
import miox
from miox.readers.loader import READERS


def test_table_matches_the_reader_classes():
    for module, name, formats in READERS:
        cls = getattr(importlib.import_module(module), name)
        assert tuple(cls.FORMATS) == tuple(formats), name


def test_every_format_has_one_reader():
    formats = [fmt for _, _, listed in READERS for fmt in listed]
    assert len(formats) == len(set(formats))


def test_readers_are_made_on_first_use():
    loader = miox.Loader()
    assert miox.Format.NIFTI_MRS in loader.readers
    assert loader.readers[miox.Format.NIFTI_MRS] is loader.readers[miox.Format.NIFTI_MRS]
    assert type(loader.readers[miox.Format.BASIS_FOLDER]).__name__ == "BasisFolderReader"


def test_import_loads_no_reader_module():
    code = ("import sys, miox; print(sorted(m for m in sys.modules if m.startswith("
            "'miox.readers.') and m not in ('miox.readers.base', 'miox.readers.loader')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
    assert miox.readers.SiemensTwixReader.__name__ == "SiemensTwixReader"
