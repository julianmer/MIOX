####################################################################################################
#                                             vasf.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for VASF spectroscopy pairs (.rsp text header, .rsd binary data), the format of  #
#          the Duke tools that preceded Vespa. The vendor header keeps every section and key of    #
#          the .rsp file.                                                                          #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

from pathlib import Path

import numpy as np

# own
from miox.data import MRSData
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                            VasfReader                                            #
#**************************************************************************************************#
#                                                                                                  #
# Reads one .rsp/.rsd pair. Times in the header are in ms, the frequency in Hz. The voxel geometry #
# stays in the vendor header (not converted to an affine).                                         #
#                                                                                                  #
#**************************************************************************************************#
class VasfReader(Reader):
    FORMATS = (Format.VASF,)
    SOURCE = "VASF header"
    MEASUREMENT = "MEASUREMENT INFORMATION"
    TYPES = {"complex float": "f4", "complex double": "f8"}
    ORDER = {"xdr": ">", "ieee": "<"}  # data_format: byte order
    GRID = ("measure_size_column", "measure_size_line", "measure_size_partition")

    def read(self, sniff: Sniff) -> list[MRSData]:
        rsp, rsd = self._pair(sniff)
        header = self._header(rsp)
        info, m = header["FILE INFORMATION"], header[self.MEASUREMENT]
        dtype = self.ORDER.get(info["data_format"].strip().lower(), ">")
        dtype += self.TYPES.get(info["data_type"].strip().lower(), "f4")
        values = np.fromfile(rsd, dtype=dtype)
        data = self._shape(values[0::2] + 1j * values[1::2], m)
        field = f"{self.MEASUREMENT}/"
        frequency = float(m["frequency"]) / 1e6
        keys = {
            "SpectrometerFrequency": ([frequency], field + "frequency"),
            "ResonantNucleus": ([m["nucleus"].strip()], field + "nucleus"),
            "EchoTime": (float(m["echo_time_1"]) * 1e-3, field + "echo_time_1"),
            "RepetitionTime": (float(m["repetition_time_1"]) * 1e-3, field + "repetition_time_1"),
            "ExcitationFlipAngle": (float(m["flip_angle"]), field + "flip_angle"),
            "NumberOfAverages": (int(float(m["averages"])), field + "averages"),
        }
        if float(m.get("inversion_time_1", 0) or 0) > 0:
            inversion = float(m["inversion_time_1"]) * 1e-3
            keys["InversionTime"] = (inversion, field + "inversion_time_1")
        notes = ["voxel geometry kept in the vendor header, not converted"]
        dwelltime = 1 / float(m["sweep_width"])
        tags = ("DIM_USER_0",) if data.ndim == 5 else ()
        d = self.dataset(
            sniff, data, dwelltime, keys, header, rsp.stem, notes, self.SOURCE, dim_tags=tags
        )
        d.meta.set("DwellTime", dwelltime, f"{self.SOURCE}: 1 / sweep_width", field + "sweep_width")
        return [d]

    def _pair(self, sniff: Sniff) -> tuple[Path, Path]:
        files = (sniff.path,) + sniff.companions
        rsp = next((f for f in files if f.suffix.lower() == ".rsp"), None)
        rsd = next((f for f in files if f.suffix.lower() == ".rsd"), None)
        if rsp is None or rsd is None:
            raise ReaderError(f"incomplete .rsp/.rsd pair: {sniff.path.name}")
        return rsp, rsd

    @staticmethod
    def _header(path: Path) -> dict[str, dict[str, str]]:
        """``[SECTION]`` lines and ``key=value ;comment`` lines, values as written."""
        header, section = {}, ""
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
                header.setdefault(section, {})
            elif "=" in line:
                key, _, value = line.partition("=")
                header.setdefault(section, {})[key.strip()] = value.split(";")[0].strip()
        return header

    def _shape(self, fid: np.ndarray, m: dict[str, str]) -> np.ndarray:
        points = int(float(m["measure_size_spectral"]))
        grid = tuple(int(float(m.get(key, 1) or 1)) for key in self.GRID)
        if np.prod(grid) * points == fid.size:
            return fid.reshape(grid[::-1] + (points,)).transpose(2, 1, 0, 3)
        return fid.reshape(-1, points).T.reshape(1, 1, 1, points, -1)
