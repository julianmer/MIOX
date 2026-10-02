####################################################################################################
#                                       gava_midas_prior.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for the GAVA text and MIDAS XML priors Vespa-Simulation exports: per line the    #
#          metabolite, three loop indices, line index, ppm, area and phase (degrees); GAVA as      #
#          tab-separated text after ';' comments, MIDAS as '++'-joined values of <param> elements. #
#          The vendor header keeps the comments (and MIDAS's creation attributes).                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

# own
from miox.data import PriorSet
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                       GavaMidasPriorReader                                       #
#**************************************************************************************************#
#                                                                                                  #
# Both formats carry the same eight values per line. The comments are free text; a comment line    #
# "TE: <value> ms" gives EchoTime.                                                                 #
#                                                                                                  #
#**************************************************************************************************#
class GavaMidasPriorReader(Reader):
    FORMATS = (Format.GAVA_PRIOR, Format.MIDAS_PRIOR)
    SOURCE = {Format.GAVA_PRIOR: "GAVA prior", Format.MIDAS_PRIOR: "MIDAS prior"}
    TE = re.compile(r"^\s*TE:\s*([-+\d.eE]+)\s*ms\s*$")

    def read(self, sniff: Sniff) -> list[PriorSet]:
        if sniff.format == Format.GAVA_PRIOR:
            vendor, rows = self._gava(sniff.path)
        else:
            vendor, rows = self._midas(sniff.path)
        if not rows:
            raise ReaderError(f"{sniff.path.name} holds no prior lines")
        lines = [self._line(row, sniff.path) for row in rows]
        values = {}
        for i, comment in enumerate(vendor["comments"]):
            if match := self.TE.match(comment):
                values["EchoTime"] = (float(match[1]) * 1e-3, f"comments/{i}")
                break
        source = self.SOURCE[sniff.format]
        return [self.prior_set(sniff, lines, values, vendor, sniff.path.stem, [], source)]

    @staticmethod
    def _gava(path: Path) -> tuple[dict, list[list[str]]]:
        comments, rows = [], []
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith(";"):
                comments.append(line[1:])
            elif line.strip():
                rows.append(line.split("\t"))
        return {"comments": comments}, rows

    @staticmethod
    def _midas(path: Path) -> tuple[dict, list[list[str]]]:
        root = ET.parse(path).getroot()
        comments = [c.get("value", "") for c in root.iter("comment")]
        rows = [p.get("value", "").split("++") for p in root.iter("param")]
        return {"comments": comments, "attributes": dict(root.attrib)}, rows

    @staticmethod
    def _line(row: list[str], path: Path) -> dict:
        if len(row) != 8:
            raise ReaderError(f"{path.name}: a prior line has {len(row)} values, not 8: {row}")
        name, d1, d2, d3, index, ppm, area, phase = (value.strip() for value in row)
        return {
            "metabolite": name,
            "line": int(index),
            "ppm": float(ppm),
            "area": float(area),
            "phase": float(phase),
            "dims": [int(d1), int(d2), int(d3)],
        }
