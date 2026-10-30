####################################################################################################
#                                        tarquin_output.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for TARQUIN fit results: the results text (concentration table, then sections of #
#          "key : value" lines), the fit text (per voxel a "Row : r, Col : c, Slice : s" line and  #
#          CSV columns PPMScale, Data, Fit, Baseline and one per signal) and the CSV results       #
#          (amplitudes, CRLBs, diagnostics, basis shifts and dampings by voxel). The vendor header #
#          keeps every section of the results by key.                                              #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import csv
import re

import numpy as np

# own
from miox.data import FitResult
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                       TarquinOutputReader                                        #
#**************************************************************************************************#
#                                                                                                  #
# Results: rows by signal ("Conc (..)" and "%SD" become concentration and sd_percent), the "Fit    #
# quality" section as diagnostics, and from "Data parameters" and "Fitting parameters":            #
# SpectrometerFrequency (Ft, Hz), DwellTime (1 / Fs), NumberOfPoints (N), EchoTime (TE, s) and     #
# ReferencePPM (Ref, the ppm at 0 Hz as in .dpt files). Fit: one FitResult of curves per voxel;    #
# "Fit" is the sum of the signal curves, the baseline is a curve of its own. CSV (--output_csv):   #
# titled sections, tables by voxel (Row, Col, Slice first) and name,value lists; per voxel one     #
# FitResult of the signal amplitudes as concentration, the CRLBs as sd (amplitude units) with      #
# sd_percent from it, each basis signal's group, shift and damping, the fit diagnostics, and the   #
# same keys from "Data parameters".                                                                #
#                                                                                                  #
#**************************************************************************************************#
class TarquinOutputReader(Reader):
    FORMATS = (Format.TARQUIN_RESULTS, Format.TARQUIN_FIT, Format.TARQUIN_CSV)
    SOURCE = "TARQUIN results"
    VOXEL = re.compile(r"Row\s*:\s*(\d+),\s*Col\s*:\s*(\d+),\s*Slice\s*:\s*(\d+)")
    CURVES = {"PPMScale": "ppm", "Data": "data", "Fit": "fit", "Baseline": "baseline"}
    PARAMETERS = {  # standard key: (section, key, convert)
        "SpectrometerFrequency": ("Data parameters", "Ft (Hz)", lambda v: [v / 1e6]),
        "DwellTime": ("Data parameters", "Fs (Hz)", lambda v: 1 / v),
        "NumberOfPoints": ("Data parameters", "N", int),
        "EchoTime": ("Data parameters", "TE (s)", float),
        "ReferencePPM": ("Fitting parameters", "Ref (ppm)", float),
    }

    def read(self, sniff: Sniff) -> list[FitResult]:
        lines = sniff.path.read_text(errors="replace").splitlines()
        if sniff.format == Format.TARQUIN_FIT:
            return self._fits(sniff, lines)
        if sniff.format == Format.TARQUIN_CSV:
            return self._csv(sniff, lines)
        return [self._results(sniff, lines)]

    def _csv(self, sniff: Sniff, lines: list[str]) -> list[FitResult]:
        """Titled sections: tables by voxel (Row, Col, Slice first) and "name,value" lists."""
        tables, pairs, title = {}, {}, "other"
        for cells in csv.reader(lines):
            if len(cells) == 1 and cells[0].strip():
                title = cells[0].strip()
            elif len(cells) > 1 and cells[:3] == ["Row", "Col", "Slice"]:
                tables[title] = [cells]
            elif len(cells) > 1 and title in tables:
                tables[title].append(cells)
            elif len(cells) == 2:
                pairs.setdefault(title, {})[cells[0]] = self._value(cells[1])
        amplitudes = tables.get("Signal amplitudes")
        if not amplitudes or len(amplitudes) < 2:
            raise ReaderError(f"{sniff.path.name}: no signal amplitudes")
        values = {}
        for key, (section, name, convert) in self.PARAMETERS.items():
            value = pairs.get(section, {}).get(name)
            if isinstance(value, float):
                values[key] = (convert(value), f"{section}/{name}")
        by_voxel = {title: self._by_voxel(table, title == "Basis shifts and dampings")
                    for title, table in tables.items()}
        results = []
        for voxel, amplitude in by_voxel["Signal amplitudes"].items():
            sds = by_voxel.get("CRLBs (standard deviation)", {}).get(voxel, {})
            signals = by_voxel.get("Basis shifts and dampings", {}).get(voxel, [])
            basis = {s["name"]: {k: v for k, v in s.items() if k != "name"} for s in signals}
            rows = []
            for signal, value in amplitude.items():
                sd = sds.get(signal)
                rows.append({"metabolite": signal, "concentration": value, "sd": sd,
                             "sd_percent": 100 * sd / value if sd is not None and value else None,
                             **basis.get(signal, {})})
            diagnostics = by_voxel.get("Fit diagnostics", {}).get(voxel, {})
            name = sniff.path.stem
            if len(by_voxel["Signal amplitudes"]) > 1:
                name += "_" + "_".join(map(str, voxel))
            results.append(self.fit_result(sniff, rows, diagnostics, {}, values, pairs, name, [],
                                           "TARQUIN CSV results", voxel))
        return results

    def _by_voxel(self, table: list[list[str]], several: bool) -> dict:
        """{voxel: {column: value}}, or {voxel: [rows]} for a table of several rows per voxel."""
        header, voxels = table[0], {}
        for cells in table[1:]:
            voxel = tuple(int(float(c)) for c in cells[:3])
            row = {h: self._value(c) for h, c in zip(header[3:], cells[3:], strict=False)}
            voxels.setdefault(voxel, []).append(row)
        return voxels if several else {voxel: rows[-1] for voxel, rows in voxels.items()}

    def _results(self, sniff: Sniff, lines: list[str]) -> FitResult:
        header = re.split(r"\s{2,}", lines[0].strip())
        rows, i = [], 2  # after the header and its underline
        while i < len(lines) and lines[i].strip():
            tokens = lines[i].split()
            row = {"metabolite": tokens[0]}
            for column, text in zip(header[1:], tokens[1:], strict=False):
                key = "concentration" if column.startswith("Conc") else column
                row["sd_percent" if column == "%SD" else key] = self._value(text)
            rows.append(row)
            i += 1
        sections = self._sections(lines[i:])
        values = {}
        for key, (section, name, convert) in self.PARAMETERS.items():
            value = sections.get(section, {}).get(name)
            if isinstance(value, float):
                values[key] = (convert(value), f"{section}/{name}")
        vendor = {"columns": header, **sections}
        diagnostics = sections.get("Fit quality", {})
        return self.fit_result(
            sniff, rows, diagnostics, {}, values, vendor, sniff.path.stem, [], self.SOURCE
        )

    def _sections(self, lines: list[str]) -> dict:
        """Underlined section titles, each followed by "key : value" lines."""
        sections, title = {}, None
        for i, line in enumerate(lines):
            if i + 1 < len(lines) and set(lines[i + 1].strip()) == {"-"}:
                title = line.strip()
                sections[title] = {}
            elif title and ":" in line:
                key, _, value = line.partition(":")
                sections[title][key.strip()] = self._value(value.strip())
            elif line.strip() and not set(line.strip()) <= {"-"}:
                sections.setdefault("other", []).append(line.strip())
        return sections

    def _fits(self, sniff: Sniff, lines: list[str]) -> list[FitResult]:
        starts = [i for i, line in enumerate(lines) if self.VOXEL.match(line.strip())]
        if not starts:
            raise ReaderError(f"{sniff.path.name} has no 'Row : r, Col : c, Slice : s' line")
        results = []
        for start, end in zip(starts, starts[1:] + [len(lines)], strict=True):
            voxel = tuple(int(v) for v in self.VOXEL.match(lines[start].strip()).groups())
            columns = [c.strip() for c in lines[start + 1].split(",")]
            table = np.array(
                [[float(v) for v in line.split(",")] for line in lines[start + 2 : end] if line],
            )
            curves = {self.CURVES.get(c, c): table[:, j] for j, c in enumerate(columns)}
            name = sniff.path.stem
            if len(starts) > 1:
                name += "_" + "_".join(map(str, voxel))
            results.append(self.fit_result(
                sniff, [], {}, curves, {}, {"columns": columns}, name, [], "TARQUIN fit", voxel
            ))
        return results

    @staticmethod
    def _value(text: str) -> float | str:
        try:
            return float(text)
        except ValueError:
            return text
