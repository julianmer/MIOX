####################################################################################################
#                                           fsl_vest.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 30/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for FSL's VEST text matrices: design matrices (.mat), contrasts (.con) and       #
#          F-tests (.fts), as FEAT, FLAME and FSL-MRS write them.                                  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import numpy as np

# own
from miox.data import Protocol
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                          FslVestReader                                           #
#**************************************************************************************************#
#                                                                                                  #
# "/Name value" header lines (/NumWaves the columns, /NumPoints or /NumContrasts the rows;         #
# /ContrastName<n>, /PPheights, ... as written), then /Matrix and one row per line. One Protocol:  #
# the header by name, the matrix as rows, and the contrast names in their order.                   #
#                                                                                                  #
#**************************************************************************************************#
class FslVestReader(Reader):
    FORMATS = (Format.FSL_VEST,)
    PACKAGES = ("miox",)

    def read(self, sniff: Sniff) -> list[Protocol]:
        lines = [line.strip() for line in sniff.path.read_text(errors="replace").splitlines()]
        header, rows, matrix = {}, [], False
        for line in lines:
            if not line:
                continue
            if matrix:
                rows.append([float(v) for v in line.split()])
            elif line == "/Matrix":
                matrix = True
            elif line.startswith("/"):
                name, *rest = line[1:].split(None, 1)
                text = rest[0].strip() if rest else ""
                if name.startswith("ContrastName"):  # a name, spaces and all
                    header[name] = text
                    continue
                numbers = [self._number(v) for v in text.split()]
                header[name] = numbers[0] if len(numbers) == 1 else (numbers or None)
        waves, points = header.get("NumWaves"), header.get("NumPoints", header.get("NumContrasts"))
        shape = (len(rows), len(rows[0]) if rows else 0)
        if not matrix or shape != (points, waves):
            raise ReaderError(f"{sniff.path.name}: a {shape} matrix for /NumPoints {points} and "
                              f"/NumWaves {waves}")
        contrasts = [header[f"ContrastName{i}"] for i in range(1, points + 1)
                     if f"ContrastName{i}" in header]
        meta = Meta({"header": header, "matrix": np.asarray(rows).tolist(),
                     "contrast_names": contrasts})
        return [Protocol(sniff.path.name, (), meta, self.provenance(sniff))]

    @staticmethod
    def _number(text: str) -> int | float | str:
        for kind in (int, float):
            try:
                return kind(text)
            except ValueError:
                continue
        return text
