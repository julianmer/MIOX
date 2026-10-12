####################################################################################################
#                                            matfile.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: MATLAB .mat files (v5 and v7.3, including class objects, through mat-io) as plain       #
#          Python values, and their variable names without loading the data.                       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


#**************************************************************************************************#
#                                             MatFile                                              #
#**************************************************************************************************#
#                                                                                                  #
# Structs become dicts, class objects dicts of their properties plus "class", cell arrays lists,   #
# char arrays str and numeric arrays numpy arrays in MATLAB axis order with singleton axes dropped #
# (scalars as numpy scalars). A struct, cell or char array of one element becomes that element.    #
#                                                                                                  #
#**************************************************************************************************#
class MatFile:
    HEADERS = (b"MATLAB 5.0 MAT-file", b"MATLAB 7.3 MAT-file")
    LARGE = 16  # arrays with more elements are summarised in headers

    @classmethod
    def is_mat(cls, head: bytes) -> bool:
        return head.startswith(cls.HEADERS)

    @staticmethod
    def variables(path: Path) -> dict[str, str]:
        """Variable names and their MATLAB classes."""
        import matio  # mat-io imports pandas: only when a MATLAB file is read

        return {name: kind for name, (_, kind) in matio.whosmat(path).items()}

    @classmethod
    def load(cls, path: Path) -> dict[str, Any]:
        """Every variable, as plain Python values."""
        import matio

        return {name: cls._plain(value) for name, value in matio.load_from_mat(path).items()}

    @classmethod
    def header(cls, value: Any) -> Any:
        """``value`` for a vendor header: large arrays become a shape summary."""
        if isinstance(value, dict):
            return {str(k): cls.header(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls.header(v) for v in value]
        if isinstance(value, np.ndarray):
            if value.size > cls.LARGE:
                return f"<array {value.dtype} {value.shape}>"
            return cls.header(value.tolist())
        if isinstance(value, np.generic):
            return cls.header(value.item())
        if isinstance(value, complex):
            return [value.real, value.imag]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return f"<{type(value).__name__}>"

    @classmethod
    def _plain(cls, value: Any) -> Any:
        from matio.utils.matclass import MatlabOpaque

        if isinstance(value, MatlabOpaque):
            properties = {k: cls._plain(v) for k, v in (value.properties or {}).items()}
            return {"class": value.classname, **properties}
        if not isinstance(value, np.ndarray):
            return value
        if value.dtype.names:  # struct array
            items = [{n: cls._plain(e[n]) for n in value.dtype.names} for e in value.ravel("F")]
        elif value.dtype == object:  # cell array
            items = [cls._plain(e) for e in value.ravel("F")]
        elif value.dtype.kind == "U":  # char array, one string per row
            items = [str(s) for s in value.ravel()] or [""]
        else:
            value = value.squeeze()
            return value[()] if value.ndim == 0 else value
        return items[0] if len(items) == 1 else items
