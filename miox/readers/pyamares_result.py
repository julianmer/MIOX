####################################################################################################
#                                        pyamares_result.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for PyAMARES fit results (.csv): one row per peak with amplitude, sd, CRLB(%),   #
#          chem shift(ppm), LW(Hz), phase(deg), g, their SDs and CRLBs, and SNR. The vendor header #
#          keeps the column names.                                                                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import csv

# own
from miox.data import FitResult
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                       PyamaresResultReader                                       #
#**************************************************************************************************#
#                                                                                                  #
# amplitude and CRLB(%) become concentration and sd_percent; every other column keeps its name.    #
# Empty cells are None.                                                                            #
#                                                                                                  #
#**************************************************************************************************#
class PyamaresResultReader(Reader):
    FORMATS = (Format.PYAMARES_RESULT,)
    SOURCE = "PyAMARES result"
    COLUMNS = {"amplitude": "concentration", "CRLB(%)": "sd_percent"}

    def read(self, sniff: Sniff) -> list[FitResult]:
        with open(sniff.path, newline="", errors="replace") as fh:
            table = [[cell.strip() for cell in row] for row in csv.reader(fh) if row]
        header = table[0]
        if header[0] != "name":
            raise ReaderError(f"{sniff.path.name} does not start with a name column")
        rows = []
        for cells in table[1:]:
            row = {"metabolite": cells[0]}
            for column, text in zip(header[1:], cells[1:], strict=False):
                row[self.COLUMNS.get(column, column)] = float(text) if text else None
            rows.append(row)
        vendor = {"columns": header}
        return [
            self.fit_result(sniff, rows, {}, {}, {}, vendor, sniff.path.stem, [], self.SOURCE)
        ]
