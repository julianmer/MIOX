####################################################################################################
#                                            matfile.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: MATLAB .mat files (v5 through scipy, v7.3 through h5py) as plain Python values, and     #
#          their variable names without loading the data.                                          #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.io


#**************************************************************************************************#
#                                             MatFile                                              #
#**************************************************************************************************#
#                                                                                                  #
# Structs become dicts, cell arrays lists, char arrays str and numeric arrays numpy arrays in      #
# MATLAB axis order with singleton axes dropped (scalars as numpy scalars), the same for both      #
# versions. MATLAB class objects (classdef) are not decoded: v5 gives them as opaque values.       #
#                                                                                                  #
#**************************************************************************************************#
class MatFile:
    HEADERS = (b"MATLAB 5.0 MAT-file", b"MATLAB 7.3 MAT-file")
    V73 = b"MATLAB 7.3 MAT-file"
    LARGE = 16  # arrays with more elements are summarised in headers

    @classmethod
    def is_mat(cls, head: bytes) -> bool:
        return head.startswith(cls.HEADERS)

    @classmethod
    def variables(cls, path: Path) -> dict[str, str]:
        """Variable names and their MATLAB classes."""
        if cls._is_v73(path):
            with h5py.File(path, "r") as f:
                return {
                    name: f[name].attrs.get("MATLAB_class", b"").decode()
                    for name in f
                    if not name.startswith("#")
                }
        try:
            return {name: kind for name, _, kind in scipy.io.whosmat(path)}
        except TypeError:  # whosmat cannot size class objects
            values = scipy.io.loadmat(path)
            return {k: type(v).__name__ for k, v in values.items() if not k.startswith("__")}

    @classmethod
    def load(cls, path: Path) -> dict[str, Any]:
        """Every variable, as plain Python values."""
        if cls._is_v73(path):
            with h5py.File(path, "r") as f:
                return {name: cls._h5(f, f[name]) for name in f if not name.startswith("#")}
        values = scipy.io.loadmat(path, simplify_cells=True)
        return {name: cls._v5(v) for name, v in values.items() if not name.startswith("__")}

    @classmethod
    def header(cls, value: Any) -> Any:
        """``value`` for a vendor header: large arrays become a shape summary."""
        if isinstance(value, dict):
            return {str(k): cls.header(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls.header(v) for v in value]
        if isinstance(value, np.ndarray):
            if value.dtype == object:
                return [cls.header(v) for v in value.ravel()]
            if value.size > cls.LARGE:
                return f"<array {value.dtype} {value.shape}>"
            return cls.header(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, complex):
            return [value.real, value.imag]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return f"<{type(value).__name__}>"

    @classmethod
    def _is_v73(cls, path: Path) -> bool:
        with open(path, "rb") as f:
            return f.read(len(cls.V73)) == cls.V73

    @classmethod
    def _v5(cls, value: Any) -> Any:
        """Cell arrays (object arrays) as lists."""
        if isinstance(value, dict):
            return {k: cls._v5(v) for k, v in value.items()}
        if isinstance(value, np.ndarray) and value.dtype == object:
            return [cls._v5(v) for v in value.ravel()]
        if isinstance(value, list):
            return [cls._v5(v) for v in value]
        return value

    @classmethod
    def _h5(cls, f: h5py.File, node: h5py.Group | h5py.Dataset) -> Any:
        kind = node.attrs.get("MATLAB_class", b"").decode()
        if isinstance(node, h5py.Group):
            return {name: cls._h5(f, node[name]) for name in node}
        if node.attrs.get("MATLAB_empty", 0):
            return "" if kind == "char" else np.zeros(0)
        value = node[()]
        if isinstance(value, np.ndarray) and value.dtype == object:  # cell: references
            items = [cls._h5(f, f[ref]) for ref in value.ravel()]
            return items[0] if len(items) == 1 else items  # as v5 squeezes a 1x1 cell
        if isinstance(value, np.ndarray) and value.dtype.names == ("real", "imag"):
            value = value["real"] + 1j * value["imag"]
        value = np.asarray(value).T  # h5py gives MATLAB's axes reversed
        if kind == "char":
            rows = ["".join(map(chr, row)) for row in np.atleast_2d(value)]
            return rows[0] if len(rows) == 1 else rows
        if kind == "logical":
            value = value.astype(bool)
        value = value.squeeze()
        return value[()] if value.ndim == 0 else value
