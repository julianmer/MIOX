####################################################################################################
#                                           text_fid.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 04/10/26                                                                                #
#                                                                                                  #
# Purpose: Reads headerless text files of numeric columns: pairs of real and imaginary parts, one  #
#          signal per pair.                                                                        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import io

import numpy as np

# own
from miox.data import MRSData
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                          TextFidReader                                           #
#**************************************************************************************************#
#                                                                                                  #
# Reads one text file of numbers separated by whitespace or commas. Every pair of columns is taken #
# as the real and imaginary parts of one time-domain signal, as written: the file states neither   #
# its sign convention (exports exist in the NIfTI-MRS convention and in its conjugate) nor the     #
# frequency, nucleus and dwell time, which stay missing until given.                               #
#                                                                                                  #
#**************************************************************************************************#
class TextFidReader(Reader):
    FORMATS = (Format.TEXT_FID,)
    SOURCE = "text columns"
    NOTES = (
        "headerless columns: no frequency, nucleus or dwell time stated",
        "column pairs read as written as the real and imaginary parts of a time-domain signal; "
        "the sign convention is not stated",
    )

    def read(self, sniff: Sniff) -> list[MRSData]:
        text = sniff.path.read_text(errors="replace").replace(",", " ")
        try:
            rows = np.loadtxt(io.StringIO(text), ndmin=2)
        except ValueError as e:
            raise ReaderError(f"{sniff.path.name}: {e}") from e
        columns = rows.shape[1]
        if columns < 2 or columns % 2:
            raise ReaderError(f"{sniff.path.name}: {columns} columns are not real/imaginary pairs")
        datasets = []
        for pair in range(columns // 2):
            fid = (rows[:, 2 * pair] + 1j * rows[:, 2 * pair + 1]).reshape(1, 1, 1, -1)
            name = sniff.path.stem
            if columns > 2:
                name = f"{name} columns {2 * pair + 1}-{2 * pair + 2}"
            d = self.dataset(sniff, fid, None, {}, {}, name, list(self.NOTES), self.SOURCE)
            datasets.append(d)
        return datasets
