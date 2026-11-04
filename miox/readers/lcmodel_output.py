####################################################################################################
#                                        lcmodel_output.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for LCModel fit results: .COORD (tables and the plotted curves), .TABLE (the     #
#          tables), .CSV (one row per voxel) and the .ps page (its printed tables; the plotted     #
#          curves stay undecoded). The vendor header keeps the title lines, the misc and           #
#          diagnostic tables as written and the table of input changes by key.                     #
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
#                                       LcmodelOutputReader                                        #
#**************************************************************************************************#
#                                                                                                  #
# COORD and TABLE share their tables, each announced by "<n> lines in following <name>". Rows keep #
# the ratio column under its own name (e.g. "/Cr+PCr"); "Conc." and "%SD" become concentration and #
# sd_percent. COORD curves are as LCModel plots them: each metabolite curve includes the baseline. #
# The input changes give SpectrometerFrequency (HZPPPM), DwellTime (DELTAT), EchoTime (ECHOT, ms)  #
# and NumberOfPoints (NUNFIL). Input lines are cut at a fixed width (even inside a name) and       #
# padded, the rest on lines indented by three spaces; the width is taken from the longest cut      #
# line. A line may hold several comma-separated KEY=value pairs (namelist style). The PostScript   #
# page shows the same tables as strings, one per line, under the headings DIAGNOSTICS,             #
# MISCELLANEOUS OUTPUT and INPUT CHANGES; a second page repeats them with the names and inputs     #
# the plot page had to cut, so the rows come from the page with the longest names and the inputs   #
# of every page are merged.                                                                        #
#                                                                                                  #
#**************************************************************************************************#
class LcmodelOutputReader(Reader):
    FORMATS = (Format.LCMODEL_COORD, Format.LCMODEL_TABLE, Format.LCMODEL_CSV, Format.LCMODEL_PS)
    SOURCE = "LCModel table of input changes"
    SECTION = re.compile(r"^\s*(?:\$\$\w+\s+)?(\d+)\s+lines in following (\w+)")
    CURVE = re.compile(
        r"^\s*(?:(\d+) points on ppm-axis|NY (phased data points|points of the fit to the data|"
        r"background values) follow|(\S+)\s+Conc\. =)"
    )
    NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?")
    CURVES = {"phased data points": "data", "points of the fit to the data": "fit",
              "background values": "baseline"}
    COLUMNS = {"Conc.": "concentration", "%SD": "sd_percent"}
    MISC = {  # diagnostic name -> pattern on the misc table
        "FWHM (ppm)": re.compile(r"FWHM\s*=\s*([-+.\dE]+)\s*ppm"),
        "S/N": re.compile(r"S/N\s*=\s*([-+.\dE]+)"),
        "Data shift (ppm)": re.compile(r"Data shift\s*=\s*([-+.\dE]+)\s*ppm"),
        "Ph (deg)": re.compile(r"Ph:\s*([-+.\dE]+)\s*deg\b"),
        "Ph (deg/ppm)": re.compile(r"([-+.\dE]+)\s*deg/ppm"),
    }
    PAIR = re.compile(r"([A-Za-z][\w()]*)\s*=\s*('(?:[^']|'')*'|[^,]*)")  # KEY = value
    EXPONENT = re.compile(r"(?<=\d)[dD](?=[+-]?\d)")  # Fortran double exponent: 1.5D+02
    INPUTS = {  # standard key: (input, factor)
        "SpectrometerFrequency": ("hzpppm", 1.0),
        "DwellTime": ("deltat", 1.0),
        "EchoTime": ("echot", 1e-3),
        "NumberOfPoints": ("nunfil", 1),
    }

    STRING = re.compile(r"\{\(((?:[^()\\]|\\.)*)\)s\}[lcr]st")  # a shown string: {(text)s}lst
    ESCAPE = re.compile(r"\\([()\\])")
    HEADINGS = {"DIAGNOSTICS": "diagnostic", "MISCELLANEOUS OUTPUT": "misc",
                "INPUT CHANGES": "table"}

    def read(self, sniff: Sniff) -> list[FitResult]:
        if sniff.format == Format.LCMODEL_CSV:
            return self._csv(sniff)
        lines = sniff.path.read_text(errors="replace").splitlines()
        notes = []
        if sniff.format == Format.LCMODEL_PS:
            tables, curves, title = self._pages(lines), {}, []
            notes.append("the plotted curves are not decoded from the PostScript")
        else:
            tables, curves, title = self._sections(
                lines, sniff.format == Format.LCMODEL_COORD, notes
            )
        if "concentration" not in tables:
            raise ReaderError(f"{sniff.path.name} has no concentration table")
        rows = self._rows(tables["concentration"])
        diagnostics = self._misc(tables.get("misc", []))
        inputs = self._inputs(tables.get("table", []))
        vendor = {
            "title": tables.pop("title", title),
            "misc": tables.get("misc", []),
            "diagnostic": tables.get("diagnostic", []),
            "inputs": inputs,
        }
        values = self._values(inputs)
        name = sniff.path.stem
        return [
            self.fit_result(
                sniff, rows, diagnostics, curves, values, vendor, name, notes, self.SOURCE
            )
        ]

    def _sections(
        self, lines: list[str], coord: bool, notes: list[str]
    ) -> tuple[dict, dict, list[str]]:
        """Tables by the first word of their name, curves (COORD) and the title lines. A table
        ends after its announced lines or at the next section, whichever comes first."""
        tables, curves, title = {}, {}, []
        i, points = 0, 0
        while i < len(lines):
            line = lines[i]
            if match := self.SECTION.match(line):
                count, table = int(match[1]), []
                i += 1
                while len(table) < count and i < len(lines) and not self._starts(lines[i]):
                    table.append(lines[i].rstrip())
                    i += 1
                if len(table) < count:
                    notes.append(f"{match[2]} table announces {count} lines, holds {len(table)}")
                tables[match[2]] = table
                continue
            if coord and (match := self.CURVE.match(line)):
                points = int(match[1]) if match[1] else points
                name = "ppm" if match[1] else self.CURVES.get(match[2], match[3])
                values, i = self._numbers(lines, i + 1, points)
                curves[name] = values
                continue
            if not tables and line.strip():
                title.append(line.strip())
            i += 1
        return tables, curves, title

    def _pages(self, lines: list[str]) -> dict:
        """The tables printed on the PostScript pages: rows from the page with the longest
        metabolite names, the input changes of all pages merged."""
        pages = [self._strings(page) for page in "\n".join(lines).split("%%Page:")[1:]]
        pages = [p for p in pages if "concentration" in p]
        if not pages:
            return {}
        best = max(pages, key=lambda p: len("".join(p["concentration"])))
        tables = {**best, "table": [line for p in pages for line in p.get("table", [])]}
        tables["title"] = best["title"]
        return tables

    def _strings(self, page: str) -> dict:
        tables, title, current = {}, [], None
        for match in self.STRING.finditer(page):
            text = self.ESCAPE.sub(r"\1", match[1])
            word = text.strip()
            if word.startswith("Conc."):
                current = "concentration"
                tables[current] = [text]
            elif word == "NO DIAGNOSTICS":
                current = None
            elif word in self.HEADINGS:
                current = self.HEADINGS[word]
                tables[current] = []
            elif current:
                tables[current].append(text)
            elif "concentration" not in tables and word:
                title.append(word)
        return {**tables, "title": title}

    def _starts(self, line: str) -> bool:
        return bool(self.SECTION.match(line) or self.CURVE.match(line))

    def _numbers(self, lines: list[str], start: int, count: int) -> tuple[np.ndarray, int]:
        values, i = [], start
        while len(values) < count and i < len(lines):
            values += [float(v.replace("D", "E")) for v in self.NUMBER.findall(lines[i])]
            i += 1
        return np.asarray(values[:count]), i

    def _rows(self, table: list[str]) -> list[dict]:
        header = table[0].split()[:-1]  # the last column is the metabolite
        rows = []
        for line in table[1:]:
            tokens = line.split()
            if len(tokens) < 2:
                continue
            row = {"metabolite": tokens[-1]}
            for column, text in zip(header, tokens[:-1], strict=False):
                row[self.COLUMNS.get(column, column)] = float(text.rstrip("%"))
            rows.append(row)
        return rows

    def _misc(self, misc: list[str]) -> dict:
        text = "\n".join(misc)
        return {name: float(m[1]) for name, p in self.MISC.items() if (m := p.search(text))}

    @staticmethod
    def _inputs(table: list[str]) -> dict:
        """``KEY = value`` lines, several on one line separated by commas; a line indented by
        three spaces continues the previous one, cut at a fixed width (even inside a name)."""
        entries = []
        for line in table:
            if line.startswith("   ") and entries:
                entries[-1].append(line.strip())
            elif "=" in line:
                entries.append([line.rstrip()])
        width = max((len(parts[0]) for parts in entries if len(parts) > 1), default=0)
        inputs = {}
        for parts in entries:
            line = parts[0].ljust(width) + "".join(parts[1:]) if len(parts) > 1 else parts[0]
            for key, value in LcmodelOutputReader.PAIR.findall(line):  # namelist: a, b, ...
                inputs[key] = value.strip().strip("'")
        return inputs

    def _values(self, inputs: dict) -> dict:
        lower = {key.lower(): (key, value) for key, value in inputs.items()}
        values = {}
        for key, (name, factor) in self.INPUTS.items():
            if name in lower:
                field, text = lower[name]
                number = float(self.EXPONENT.sub("E", text)) * factor
                number = [number] if key == "SpectrometerFrequency" else number
                values[key] = (int(number) if factor == 1 and name == "nunfil" else number,
                               f"inputs/{field}")
        return values

    def _csv(self, sniff: Sniff) -> list[FitResult]:
        with open(sniff.path, newline="", errors="replace") as fh:
            table = [[cell.strip() for cell in row] for row in csv.reader(fh) if row]
        header = table[0]
        sds = {c[: -len(" %SD")] for c in header if c.endswith(" %SD")}
        metabolites = [c for c in header if c in sds]
        results = []
        for cells in table[1:]:
            record = dict(zip(header, cells, strict=False))
            rows = []
            for name in metabolites:
                row = {"metabolite": name, "concentration": float(record[name]),
                       "sd_percent": float(record[f"{name} %SD"])}
                for column in header:
                    if column.startswith(name + "/") and column[len(name):] not in row:
                        row[column[len(name):]] = float(record[column])
                rows.append(row)
            used = {"Row", "Col"} | {c for c in header for n in metabolites
                                     if c == n or c == f"{n} %SD" or c.startswith(n + "/")}
            diagnostics = {c: record[c] for c in header if c and c not in used}
            voxel = (int(record["Row"]), int(record["Col"]))
            name = f"{sniff.path.stem}_{voxel[0]}_{voxel[1]}" if len(table) > 2 else sniff.path.stem
            results.append(self.fit_result(
                sniff, rows, diagnostics, {}, {}, {"columns": header}, name, [], self.SOURCE, voxel
            ))
        return results
