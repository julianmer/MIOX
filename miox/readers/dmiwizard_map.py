####################################################################################################
#                                         dmiwizard_map.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 30/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for the maps DMIWizard writes as text: metabolic maps of one MRSI slice and the  #
#          ROI masks drawn on them.                                                                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import numpy as np

# own
from miox.data import MRImage
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff, Sniffer


#**************************************************************************************************#
#                                        DMIWizardMapReader                                        #
#**************************************************************************************************#
#                                                                                                  #
# One number per line, the map's points in MATLAB order (first axis fastest); the file name gives  #
# the size (rows x columns), the plane (X, Y or Z slice, its MATLAB index from 1) and the map's ID #
# (or, for an ROI mask on the zero-filled grid, the MRI slice it was drawn on). Rows and columns   #
# are the MRSI grid's first and second axes of that plane (x, y for Z; x, z for Y; y, z for X);    #
# the file holds no geometry. A metabolic map's DICOM copy beside it is the same map, clipped at   #
# zero and scaled to 0-65535.                                                                      #
#                                                                                                  #
#**************************************************************************************************#
class DMIWizardMapReader(Reader):
    FORMATS = (Format.DMIWIZARD_MAP,)
    PACKAGES = ("miox",)
    AXES = {"Z": ("x", "y"), "Y": ("x", "z"), "X": ("y", "z")}  # the grid axes of rows, columns

    def read(self, sniff: Sniff) -> list[MRImage]:
        match = Sniffer.DMIWIZARD_MAP.match(sniff.path.name)
        kind, rows, columns, plane, index, label = match.groups()
        rows, columns = int(rows), int(columns)
        try:
            values = np.loadtxt(sniff.path, ndmin=1)
        except ValueError as e:
            raise ReaderError(f"{sniff.path.name}: {e}") from e
        if values.size != rows * columns:
            raise ReaderError(f"{sniff.path.name}: {values.size} values for {rows} x {columns}")
        data = values.reshape(rows, columns, 1, order="F")
        meta = Meta({"map": kind, "plane": f"{plane} slice", "slice": int(index),
                     "axes": list(self.AXES[plane])})
        if kind == "ROIMap":
            meta.vendor["mri_slice"] = int(label.removeprefix("MRIslice"))
            name = f"ROI {plane} slice {index}"
        else:
            meta.vendor["map_id"] = label
            name = f"{label} {plane} slice {index}"
        notes = ("no geometry stored: the map's grid position and voxel size are unknown",)
        return [MRImage(data, self.NO_GEOMETRY, meta, self.provenance(sniff, notes), name)]
