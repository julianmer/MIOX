####################################################################################################
#                                        pyamares_prior.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for PyAMARES prior knowledge (.csv): an Index row naming the peaks, then         #
#          sections ("Initial Values", "Bounds") of parameter rows (amplitude, chemicalshift in    #
#          ppm, linewidth in Hz, phase in degrees, g). Rows starting with '#' are comments; the    #
#          vendor header keeps them.                                                               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import csv
import re

# own
from miox.data import PriorSet
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                       PyamaresPriorReader                                        #
#**************************************************************************************************#
#                                                                                                  #
# One line per peak. Initial values become the line's values (chemicalshift as "ppm"), any other   #
# section a dict under its lower-case name ("bounds": [low, high], None for an open side or an     #
# empty cell). A row without values that names no parameter starts a section. Values that are     #
# expressions linking peaks (e.g. "BATP/2", "AATP-16Hz") stay strings.                             #
#                                                                                                  #
#**************************************************************************************************#
class PyamaresPriorReader(Reader):
    FORMATS = (Format.PYAMARES_PRIOR,)
    SOURCE = "PyAMARES prior"
    INITIAL = "Initial Values"
    PARAMETERS = ("amplitude", "chemicalshift", "linewidth", "phase", "g")
    NAMES = {"chemicalshift": "ppm"}
    RANGE = re.compile(r"^\(\s*([^,]*?)\s*,\s*([^)]*?)\s*\)?$")

    def read(self, sniff: Sniff) -> list[PriorSet]:
        with open(sniff.path, newline="", errors="replace") as fh:
            rows = [row for row in csv.reader(fh)]
        comments = [row[0] for row in rows if row and row[0].strip().startswith("#")]
        table = [row for row in rows if row and row[0].strip() and row[0] not in comments]
        index = next((row for row in table if row[0].strip() == "Index"), None)
        if index is None:
            raise ReaderError(f"{sniff.path.name} has no Index row")
        peaks = [(i, name.strip()) for i, name in enumerate(index) if i and name.strip()]
        lines = [{"metabolite": name} for _, name in peaks]
        section = None
        for row in table[table.index(index) + 1 :]:
            label = row[0].strip()
            if label not in self.PARAMETERS and not any(cell.strip() for cell in row[1:]):
                section = label
                continue
            parameter = self.NAMES.get(label, label)
            for line, (i, _) in zip(lines, peaks, strict=True):
                cell = row[i].strip() if i < len(row) else ""
                if section == self.INITIAL:
                    line[parameter] = self._value(cell)
                else:
                    line.setdefault(str(section).lower(), {})[parameter] = self._range(cell)
        vendor = {"comments": comments}
        return [self.prior_set(sniff, lines, {}, vendor, sniff.path.stem, [], self.SOURCE)]

    @staticmethod
    def _value(text: str) -> float | str | None:
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return text

    def _range(self, text: str) -> list | str | None:
        match = self.RANGE.match(text)
        if not match:
            return self._value(text)
        return [self._value(match[1]), self._value(match[2])]
