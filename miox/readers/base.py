####################################################################################################
#                                             base.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Base class of all readers and the error a reader raises when a file of its format       #
#          cannot be read.                                                                         #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import contextlib
import io
import warnings
from collections.abc import Iterator
from importlib.metadata import version
from typing import Any

import numpy as np

# own
from miox.data import MRSData, Provenance
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                           ReaderError                                            #
#**************************************************************************************************#
#                                                                                                  #
# A file of a supported format that cannot be read.                                                #
#                                                                                                  #
#**************************************************************************************************#
class ReaderError(Exception):
    pass


#**************************************************************************************************#
#                                              Reader                                              #
#**************************************************************************************************#
#                                                                                                  #
# Defines the reader structure. A reader handles the formats in ``FORMATS``, takes the sniff of a  #
# path and returns one MRSData per dataset in it (a file may hold several, e.g. a water            #
# reference next to the metabolite data).                                                          #
#                                                                                                  #
#**************************************************************************************************#
class Reader:
    FORMATS: tuple[Format, ...] = ()
    PACKAGES = ("miox", "nifti-mrs")  # packages whose versions go into the provenance

    def read(self, sniff: Sniff) -> list[MRSData]:
        raise NotImplementedError

    def provenance(self, sniff: Sniff, notes: tuple[str, ...] = ()) -> Provenance:
        return Provenance(
            paths=(sniff.path,) + sniff.companions,
            format=sniff.format,
            reader=type(self).__name__,
            versions={name: version(name) for name in self.PACKAGES},
            notes=notes,
        )

    @contextlib.contextmanager
    def captured(self, notes: list[str]) -> Iterator[None]:
        """Collect what wrapped libraries print or warn into ``notes`` instead of the terminal."""
        out = io.StringIO()
        try:
            with (
                contextlib.redirect_stdout(out),
                contextlib.redirect_stderr(out),
                warnings.catch_warnings(record=True) as caught,
            ):
                warnings.simplefilter("always")
                yield
        finally:
            notes.extend(line.strip() for line in out.getvalue().splitlines() if line.strip())
            notes.extend(str(w.message) for w in caught)

    @classmethod
    def plain(cls, value: Any) -> Any:
        """Header value as plain Python (dict, list, str, int, float, bool, None)."""
        if isinstance(value, dict):
            return {str(k): cls.plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls.plain(v) for v in value]
        if isinstance(value, np.ndarray):
            return cls.plain(value.tolist())
        if isinstance(value, np.generic):
            return cls.plain(value.item())
        if isinstance(value, bytes):
            return value.decode("latin-1").rstrip("\0")
        return value
