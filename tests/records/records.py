####################################################################################################
#                                            records.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Records of what the readers return for the public test files: names, shape, dimensions, #
#          dwell time, standard keys and a fingerprint of the data. Files that cannot be read are  #
#          recorded with their error, so a change either way shows up.                             #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

# own
from miox import FitResult, Format, Loader, MRImage, MRSData, PriorSet, RawData, Sniffer
from tests.conftest import PUBLIC_DATA, RECORDS


#**************************************************************************************************#
#                                           ReadRecords                                            #
#**************************************************************************************************#
#                                                                                                  #
# Builds, stores and compares the read records. Floats compare with a relative tolerance; every    #
# other value must be equal.                                                                       #
#                                                                                                  #
#**************************************************************************************************#
class ReadRecords:
    PATH = RECORDS / "read.json"
    VOLATILE = {"ConversionTime", "ConversionMethod"}  # change with every run or version
    RTOL = 1e-6
    BRUKER_DATA = {"fid", "fid_proc.64", "2dseq", "rawdata.job0"}  # Bruker files that hold data

    def __init__(self):
        self.loader = Loader()
        self.sniffer = Sniffer()

    def expected(self) -> dict[str, Any]:
        return json.loads(self.PATH.read_text()) if self.PATH.is_file() else {}

    def record(self, path: Path) -> list[dict] | dict:
        """What the loader returns for ``path``, or the error it raises."""
        try:
            datasets = self.loader.load(path)
        except Exception as e:
            message = str(e).replace(str(PUBLIC_DATA) + "/", "")
            return {"error": f"{type(e).__name__}: {message}"}
        return [self._dataset(d) for d in datasets]

    def targets(self) -> Iterator[Path]:
        """Public files whose format has a reader."""
        for path in sorted(q for top in PUBLIC_DATA.iterdir() for q in top.rglob("*")):
            if not path.is_file() or ".git" in path.parts or self.is_lfs_pointer(path):
                continue
            if path.suffix == ".part":  # a download in progress
                continue
            fmt = self.sniffer.sniff(path).format
            if fmt not in self.loader.readers:
                continue
            if fmt == Format.BRUKER and path.name not in self.BRUKER_DATA:
                continue
            if fmt == Format.DICOM_IMAGE:  # single slices; series are checked against dcm2niix
                continue
            yield path

    def differences(self, expected: Any, got: Any, where: str = "") -> list[str]:
        """Paths at which ``got`` differs from ``expected``."""
        if isinstance(expected, dict) and isinstance(got, dict):
            keys = expected.keys() | got.keys()
            return [d for k in sorted(keys) for d in self._child(expected, got, k, where)]
        if isinstance(expected, list) and isinstance(got, list) and len(expected) == len(got):
            return [
                d
                for i, (e, g) in enumerate(zip(expected, got, strict=True))
                for d in self.differences(e, g, f"{where}[{i}]")
            ]
        if isinstance(expected, float) or isinstance(got, float):
            if self._close(expected, got):
                return []
        elif expected == got:
            return []
        return [f"{where or '/'}: expected {expected!r}, got {got!r}"]

    @staticmethod
    def is_lfs_pointer(path: Path) -> bool:
        with open(path, "rb") as fh:
            return fh.read(30).startswith(b"version https://git-lfs")

    def _child(self, expected: dict, got: dict, key: str, where: str) -> list[str]:
        if key not in expected:
            return [f"{where}/{key}: unexpected"]
        if key not in got:
            return [f"{where}/{key}: missing"]
        return self.differences(expected[key], got[key], f"{where}/{key}")

    def _close(self, a: Any, b: Any) -> bool:
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (a, b)):
            return False
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return math.isclose(a, b, rel_tol=self.RTOL, abs_tol=1e-12)

    def _dataset(self, d: MRSData | MRImage | RawData | PriorSet | FitResult) -> dict:
        if isinstance(d, RawData):
            record = {
                "name": d.name,
                "shape": list(d.shape),
                "dtype": str(d.data.dtype),
                "trajectory": None if d.trajectory is None else list(d.trajectory.shape),
                "waveforms": len(d.waveforms),
                "meta": {k: v for k, v in d.meta.items() if k not in self.VOLATILE},
                "data_norm": float(np.linalg.norm(d.data)),
                "first_readout": [[float(z.real), float(z.imag)] for z in d.data[0, 0, :3]],
                "counters_head": {
                    n: d.counters(n)[:4].tolist()
                    for n in ("kspace_encode_step_1", "kspace_encode_step_2", "slice")
                },
            }
            return json.loads(json.dumps(record, default=str))
        if isinstance(d, MRImage):
            values = np.abs(d.data) if np.iscomplexobj(d.data) else d.data
            record = {
                "name": d.name,
                "shape": list(d.shape),
                "dims": list(d.dims),
                "dtype": str(d.data.dtype),
                "affine": np.round(d.affine, 6).tolist(),
                "meta": {k: v for k, v in d.meta.items() if k not in self.VOLATILE},
                "data_sum": float(np.sum(values, dtype=np.float64)),
                "data_norm": float(np.linalg.norm(np.asarray(values, dtype=np.float64))),
            }
            return json.loads(json.dumps(record, default=str))
        if isinstance(d, FitResult):
            record = {
                "name": d.name,
                "voxel": d.voxel,
                "rows": d.rows,
                "diagnostics": d.diagnostics,
                "meta": dict(d.meta),
                "curves": {k: [len(v), float(np.sum(v))] for k, v in d.curves.items()},
            }
            return json.loads(json.dumps(record, default=str))
        if isinstance(d, PriorSet):
            record = {
                "name": d.name,
                "metabolites": d.metabolites,
                "n_lines": len(d.lines),
                "meta": dict(d.meta),
                "lines_head": d.lines[:3],
            }
            return json.loads(json.dumps(record, default=str))
        first_fid = d.data.reshape(d.shape[:4] + (-1,))[0, 0, 0, :, 0]
        record = {
            "name": d.name,
            "shape": list(d.shape),
            "dims": list(d.dims),
            "dwelltime": d.dwelltime,
            "meta": {k: v for k, v in d.meta.items() if k not in self.VOLATILE},
            "fid_head": [[float(z.real), float(z.imag)] for z in first_fid[:4]],
            "data_norm": float(np.linalg.norm(d.data)),
        }
        text = json.dumps(record, default=str).replace(str(PUBLIC_DATA) + "/", "")  # no local paths
        return json.loads(text)
