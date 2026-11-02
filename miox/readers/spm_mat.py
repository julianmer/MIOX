####################################################################################################
#                                            spm_mat.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/11/26                                                                                #
#                                                                                                  #
# Purpose: Reader for SPM.mat files: the design, estimation and contrasts of an SPM first-level    #
#          analysis as one Protocol.                                                               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

from typing import Any

import numpy as np

# own
from miox.data import Protocol
from miox.matfile import MatFile
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                           SpmMatReader                                           #
#**************************************************************************************************#
#                                                                                                  #
# The SPM struct holds the images analysed (xY: TR and the file list), the basis functions (xBF),  #
# the sessions with their conditions (Sess), the design (xX: the matrix X with one column per      #
# regressor named in xX.name, the high-pass filter K, the whitening and the estimation results),   #
# the error model (xVi), the analysis volume (xVol: dimensions and the voxel-to-world matrix M,    #
# for 1-based voxel indices), the images written (Vbeta, VResMS, VM) and the contrasts (xCon:      #
# name, statistic, weights, images). The Protocol keeps the whole struct in the vendor header      #
# (large arrays summarised), with the design matrix, regressor names, contrasts, image list and    #
# volume geometry in full; a results struct saved beside it (xSPM, TabDat) is kept as well.        #
#                                                                                                  #
#**************************************************************************************************#
class SpmMatReader(Reader):
    FORMATS = (Format.SPM_MAT,)
    PACKAGES = ("miox",)
    SOURCE = "SPM.mat"

    def read(self, sniff: Sniff) -> list[Protocol]:
        variables = MatFile.load(sniff.path)
        spm = variables.get("SPM")
        if not isinstance(spm, dict) or "xX" not in spm:
            raise ReaderError(f"{sniff.path.name}: no SPM struct with a design (xX)")
        try:
            design = self._design(spm)
        except (KeyError, TypeError, ValueError) as e:
            raise ReaderError(f"{sniff.path.name}: not a readable SPM.mat ({e})") from e
        vendor = {name: MatFile.header(value) for name, value in variables.items()}
        meta = Meta({**vendor, **design})
        rt = spm.get("xY", {}).get("RT") if isinstance(spm.get("xY"), dict) else None
        if rt is not None:
            meta.set("RepetitionTime", float(rt), f"{self.SOURCE}: SPM.xY.RT (s)", "SPM/xY/RT")
        notes = ()
        if "xVol" not in spm:
            notes = ("design not estimated: no analysis volume (xVol)",)
        name = f"{sniff.path.parent.name}/{sniff.path.stem}"
        return [Protocol(name, (), meta, self.provenance(sniff, notes))]

    @classmethod
    def _design(cls, spm: dict) -> dict:
        """The design matrix, regressor names, contrasts, images and volume geometry in full."""
        x = np.atleast_2d(np.asarray(spm["xX"]["X"], dtype=float))
        names = cls._list(spm["xX"].get("name", []))
        if x.shape[1] != len(names):
            raise ValueError(f"a {x.shape} design matrix for {len(names)} regressor names")
        design: dict[str, Any] = {
            "design_matrix": x.tolist(), "regressor_names": [str(n) for n in names],
            "contrasts": [cls._contrast(c) for c in cls._list(spm.get("xCon", []))],
            "images": [str(p) for p in cls._list(spm.get("xY", {}).get("P", []))],
        }
        if "nscan" in spm:
            scans = np.atleast_1d(np.asarray(spm["nscan"])).astype(int)
            design["scans_per_session"] = scans.tolist()
        volume = spm.get("xVol")
        if isinstance(volume, dict) and "M" in volume:
            design["affine"] = np.asarray(volume["M"], dtype=float).reshape(4, 4).tolist()
            design["dimensions"] = np.asarray(volume["DIM"]).astype(int).ravel().tolist()
        return design

    @classmethod
    def _contrast(cls, c: dict) -> dict:
        contrast = {"name": str(c.get("name", "")), "statistic": str(c.get("STAT", "")),
                    "weights": np.asarray(c.get("c", []), dtype=float).tolist()}
        for key in ("Vcon", "Vspm"):
            if isinstance(c.get(key), dict) and "fname" in c[key]:
                contrast[key] = str(c[key]["fname"])
        return contrast

    @staticmethod
    def _list(value: Any) -> list:
        """A MATLAB cell or struct array as a list (one element comes back bare)."""
        if isinstance(value, list):
            return value
        if isinstance(value, np.ndarray):
            return value.ravel().tolist()
        if value is None or (isinstance(value, dict) and not value):
            return []
        return [value]
