####################################################################################################
#                                       fsl_mrs_dynamic.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 30/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for the result folders of FSL-MRS dynamic (fMRS) fitting: the fit at every time  #
#          point and the model behind them.                                                        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import csv
import json
from pathlib import Path

# own
from miox.data import FitResult
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                     FslMrsDynamicReader                                          #
#**************************************************************************************************#
#                                                                                                  #
# mapped_parameters.csv holds the model's parameters at every time point (first header row: the    #
# time index of each column, second: mean or std): one FitResult per time point, named with its    #
# index, which is also its "dynamic" diagnostic (the voxel stays unset). conc_<name> rows become   #
# rows of that metabolite (mean as concentration, std as sd, sd_percent from them); the other      #
# parameters (line widths, shifts, phases, baseline) are diagnostics, "<name> sd" their std. The   #
# independent fit of each time point (init_results.csv) is in its vendor header; the model shared  #
# by all is in each: free_parameters (mean, sd), dyn_results, dyn_cov (with its names) and the run #
# options (options.txt: the JSON line and the command line).                                       #
#                                                                                                  #
#**************************************************************************************************#
class FslMrsDynamicReader(Reader):
    FORMATS = (Format.FSL_MRS_DYNAMIC,)
    PACKAGES = ("miox",)
    SOURCE = "FSL-MRS dynamic fit"
    CONCENTRATION = "conc_"

    def read(self, sniff: Sniff) -> list[FitResult]:
        folder = sniff.path
        mapped = self._rows(folder / "mapped_parameters.csv")
        if len(mapped) < 3:
            raise ReaderError(f"{folder.name}: mapped_parameters.csv holds no parameters")
        times, kinds = mapped[0][1:], mapped[1][1:]
        model = self._model(folder)
        init = self._rows(folder / "init_results.csv") if (folder / "init_results.csv").is_file() \
            else []
        results = []
        for time in dict.fromkeys(times):
            columns = {kind: j for j, (t, kind) in enumerate(zip(times, kinds, strict=True))
                       if t == time}
            rows, diagnostics = [], {"dynamic": int(time)}
            for line in mapped[2:]:
                mean, std = (float(line[1 + columns[k]]) for k in ("mean", "std"))
                if line[0].startswith(self.CONCENTRATION):
                    rows.append({"metabolite": line[0][len(self.CONCENTRATION):],
                                 "concentration": mean, "sd": std,
                                 "sd_percent": 100 * std / mean if mean else None})
                else:
                    diagnostics |= {line[0]: mean, f"{line[0]} sd": std}
            vendor = dict(model)
            fit = next((r for r in init[1:] if r[0] == time), None)
            if fit is not None:
                vendor["init_results"] = dict(zip(init[0][1:], map(float, fit[1:]), strict=True))
            results.append(self.fit_result(sniff, rows, diagnostics, {}, {}, vendor,
                                           f"{folder.name} dynamic {time}", [], self.SOURCE))
        return results

    def _model(self, folder: Path) -> dict:
        """What every time point shares: the free parameters, their estimate and covariance, and
        the options of the run."""
        model = {}
        if (path := folder / "free_parameters.csv").is_file():
            model["free_parameters"] = {r[0]: {"mean": float(r[1]), "sd": float(r[2])}
                                        for r in self._rows(path)[1:]}
        if (path := folder / "dyn_results.csv").is_file():
            header, *values = self._rows(path)
            model["dyn_results"] = [dict(zip(header[1:], map(float, v[1:]), strict=True))
                                    for v in values]
        if (path := folder / "dyn_cov.csv").is_file():
            header, *values = self._rows(path)
            model["dyn_cov"] = {"names": header[1:],
                                "matrix": [[float(x) for x in v[1:]] for v in values]}
        if (path := folder / "options.txt").is_file():
            first, _, rest = path.read_text(errors="replace").partition("\n")
            try:
                model["options"] = json.loads(first)
            except ValueError:
                model["options"] = first
            model["command_line"] = rest.strip("-\n ")
        return model

    @staticmethod
    def _rows(path: Path) -> list[list[str]]:
        with open(path, newline="", errors="replace") as fh:
            return [row for row in csv.reader(fh) if row]
