####################################################################################################
#                                            minispec.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 09/11/26                                                                                #
#                                                                                                  #
# Purpose: Readers for Bruker minispec relaxometry: the decay a .dps file stores (CPMG echo        #
#          amplitudes against time), the vendor's fit report that names it (.mdt), the CONTIN      #
#          relaxation-time distribution exported beside it, and the report on its own.             #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np

# own
from miox.data import Protocol, RelaxationData
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                              Report                                              #
#**************************************************************************************************#
#                                                                                                  #
# A minispec "Curve Fit Results" report: one or more headers (date, pulse separation, data points, #
# dummy echoes, exponential order) each followed by blocks, which take the header before them:     #
# curve-fit blocks (SampleId, the title of the data file, amplitude, T2 and offset with their +/-  #
# errors) or CONTIN blocks (the decay's path in quotes, the components found with T2 and A, the    #
# integral and peak areas, a Gauss fit). Amplitudes are in % of the receiver range, times in ms.   #
#                                                                                                  #
#**************************************************************************************************#
class Report:
    HEADER = re.compile(r"Curve Fit Results\s*\n(\d\d\.\d\d\.\d{4} \d\d:\d\d)")
    SETTING = re.compile(r"^([A-Za-z. ]+?)\s*:\s*(.+?)\s*$", re.M)
    SAMPLE = re.compile(r"SampleId:\s*(.*?)\s*\n")
    TITLE = re.compile(r"Title of this Data File:\s*\n(\S.*?)\s*\n")
    VALUE = re.compile(r"^(Amplitude|T2|T1|Offset)\s*:\s*(-?[\d.]+)\s*(\S*)\s*\n\s*\+/-\s*([\d.]+)",
                       re.M)
    CONTIN_PATH = re.compile(r"CONTIN Results:\s*\n'([^']*)'")
    COMPONENT = re.compile(r"^T[12]\[(\d+)\]\s*:\s*(-?[\d.]+)\s*\+/-\s*([\d.]+)\s*(\S+)\s*\n"
                           r"A \[\1\]\s*:\s*(-?[\d.]+)", re.M)
    GAUSS = re.compile(r"^(Ampl|Center|Width)\[(\d+)\]\s*:\s*(-?[\d.]+)\s*\+/-\s*([\d.]+)\s*(\S*)",
                       re.M)
    SCALE = {"ms": 1e-3, "us": 1e-6, "s": 1.0}

    def __init__(self, path: Path):
        self.path = path
        self.text = path.read_text(encoding="latin-1", errors="replace").replace("\r\n", "\n")
        self.blocks = self.parse()

    def parse(self) -> list[dict]:
        """Every sample block with the settings and date of the header before it."""
        blocks, settings, date = [], {}, None
        chunks = re.split(r"(?=Curve Fit Results\n|SampleId:|CONTIN Results:)", self.text)
        for chunk in chunks:
            if chunk.startswith("Curve Fit Results"):
                m = self.HEADER.match(chunk)
                date = datetime.strptime(m.group(1), "%d.%m.%Y %H:%M") if m else None
                settings = {k.strip(): v for k, v in self.SETTING.findall(chunk)
                            if k.strip() not in ("SampleId", "CONTIN Results")}
            elif chunk.startswith("SampleId:"):
                title = self.TITLE.search(chunk)
                values = {k: (float(v), u, float(e)) for k, v, u, e in self.VALUE.findall(chunk)}
                if title and ("T2" in values or "T1" in values):
                    blocks.append(dict(kind="fit", sample=self.SAMPLE.match(chunk).group(1),
                                       title=title.group(1), values=values, settings=settings,
                                       date=date, report=self.path.name))
            elif chunk.startswith("CONTIN Results:"):
                m = self.CONTIN_PATH.match(chunk)
                components = [(float(t) * self.SCALE.get(u, 1e-3),
                               float(e) * self.SCALE.get(u, 1e-3), float(a))
                              for _, t, e, u, a in self.COMPONENT.findall(chunk)]
                if m and components:
                    stem = re.split(r"[\\/]", m.group(1))[-1].removesuffix(".dps")
                    integral = re.search(r"^Integral\s*:\s*(-?[\d.]+)", chunk, re.M)
                    gauss = {f"{k.lower()}[{i}]": (float(v), float(e), u)
                             for k, i, v, e, u in self.GAUSS.findall(chunk)}
                    blocks.append(dict(kind="contin", sample=stem, title=stem,
                                       components=components,
                                       integral=float(integral.group(1)) if integral else None,
                                       gauss=gauss, settings=settings, date=date,
                                       report=self.path.name))
        return blocks

    def named(self, stem: str) -> list[dict]:
        return [b for b in self.blocks if b["title"] == stem]

    @staticmethod
    def pulse_separation(settings: dict) -> float | None:
        """The CPMG pulse separation tau in seconds, from "Pulse Separ.  : 1.000 ms"."""
        value = settings.get("Pulse Separ.")
        m = re.match(r"(-?[\d.]+)\s*(ms|us|s)", value or "")
        return float(m.group(1)) * {"ms": 1e-3, "us": 1e-6, "s": 1.0}[m.group(2)] if m else None


#**************************************************************************************************#
#                                          MinispecReader                                          #
#**************************************************************************************************#
#                                                                                                  #
# A .dps decay as RelaxationData: times in seconds, the echo amplitudes in % as stored, sequence   #
# CPMG. The fit report of the folder that names the file gives the fit (the latest report when     #
# several do), the pulse separation (EchoTime = 2 tau) and the sample id; the CONTIN export        #
# beside the file gives the T2 distribution. A CONTIN export on its own is read through its decay. #
#                                                                                                  #
#**************************************************************************************************#
class MinispecReader(Reader):
    FORMATS = (Format.MINISPEC_DPS, Format.MINISPEC_CONTIN)
    PACKAGES = ("miox",)

    def read(self, sniff: Sniff) -> list[RelaxationData]:
        if sniff.format == Format.MINISPEC_CONTIN:
            decays = [c for c in sniff.companions if c.suffix.lower() == ".dps"]
            if not decays:
                raise ReaderError(f"{sniff.path.name}: a CONTIN distribution without its decay "
                                  "(.dps of the same name before _CONTIN_)")
            from miox.sniff import Sniffer

            return self.read(Sniffer().sniff(decays[0]))
        notes = []
        table = np.loadtxt(sniff.path, delimiter="\t", ndmin=2)
        if table.shape[1] != 3:
            raise ReaderError(f"{sniff.path.name}: {table.shape[1]} columns, 3 expected")
        times, signal = table[:, 1] * 1e-3, table[:, 2]
        spacing = float(np.mean(np.diff(times))) if len(times) > 1 else None
        meta = Meta()
        blocks = []
        for companion in sniff.companions:
            if companion.suffix.lower() == ".mdt":
                blocks += Report(companion).named(sniff.path.stem)
        contins = [b for b in blocks if b["kind"] == "contin"]
        blocks = [b for b in blocks if b["kind"] == "fit"]
        fit = None
        if contins:
            latest = sorted(contins, key=lambda b: b["date"] or datetime.min)[-1]
            meta.vendor["contin_results"] = {
                "report": latest["report"], "components_T2_s_error_s_amplitude":
                latest["components"], "integral": latest["integral"], "gauss_fit": latest["gauss"]}
        if blocks:
            blocks.sort(key=lambda b: b["date"] or datetime.min)
            block = blocks[-1]
            if len(blocks) > 1:
                others = ", ".join(f"{b['report']} ({b['date']:%d.%m.%Y %H:%M})" if b["date"]
                                   else b["report"] for b in blocks[:-1])
                notes.append(f"{len(blocks)} fit reports name this file; the latest "
                             f"({block['report']}) is used, not {others}")
            fit = self.fit(block)
            meta.set("SampleID", block["sample"], f"report {block['report']}: SampleId")
            meta.set("RelaxationFitAmplitude", fit["amplitude"],
                     f"report {block['report']}: Amplitude")
            meta.set("RelaxationFitTime", fit["time"], f"report {block['report']}: T2")
            meta.set("RelaxationFitOffset", fit["offset"], f"report {block['report']}: Offset")
            tau = Report.pulse_separation(block["settings"])
            if tau is not None:
                meta.set("EchoTime", 2 * tau, f"report {block['report']}: 2 x Pulse Separ.")
            meta.vendor.update({"report": block["report"], "settings": block["settings"],
                                "date": block["date"].isoformat() if block["date"] else None})
        if "EchoTime" not in meta and spacing:
            meta.set("EchoTime", spacing, "mean spacing of the stored echoes")
            notes.append("no fit report names this file: echo time from the data")
        distribution = None
        contin = [c for c in sniff.companions if "_CONTIN_" in c.name]
        if contin:
            d = np.loadtxt(contin[0], delimiter="\t", ndmin=2)
            order = np.argsort(d[:, 0])
            distribution = (d[order, 0] * 1e-3, d[order, 1])
            meta.vendor["contin"] = contin[0].name
            if len(contin) > 1:
                notes.append(f"{len(contin)} CONTIN exports beside the file; {contin[0].name} "
                             "is used")
        return [RelaxationData(times, signal, "CPMG", meta, self.provenance(sniff, tuple(notes)),
                               sniff.path.stem, fit, distribution)]

    @staticmethod
    def fit(block: dict) -> dict:
        """The report's numbers in the object's units: amplitudes and offset in %, times in s."""
        v = block["values"]
        time_key = "T2" if "T2" in v else "T1"
        scale = {"ms": 1e-3, "us": 1e-6, "s": 1.0}.get(v[time_key][1], 1e-3)
        return {
            "amplitude": [v["Amplitude"][0]] if "Amplitude" in v else [],
            "amplitude_error": [v["Amplitude"][2]] if "Amplitude" in v else [],
            "time": [v[time_key][0] * scale],
            "time_error": [v[time_key][2] * scale],
            "offset": v["Offset"][0] if "Offset" in v else 0.0,
            "offset_error": v["Offset"][2] if "Offset" in v else 0.0,
            "unit": "%",
            "report": block["report"],
            "sample": block["sample"],
        }


#**************************************************************************************************#
#                                       MinispecReportReader                                       #
#**************************************************************************************************#
#                                                                                                  #
# A report on its own: one Protocol per block, named after the data file it reports on, with the  #
# fit numbers as the decay would carry them (CONTIN components as times and amplitudes) and the    #
# header settings in the vendor header.                                                            #
#                                                                                                  #
#**************************************************************************************************#
class MinispecReportReader(Reader):
    FORMATS = (Format.MINISPEC_REPORT,)
    PACKAGES = ("miox",)

    def read(self, sniff: Sniff) -> list[Protocol]:
        report = Report(sniff.path)
        if not report.blocks:
            raise ReaderError(f"{sniff.path.name}: no sample block in the report")
        protocols = []
        for block in report.blocks:
            meta = Meta()
            if block["kind"] == "contin":
                meta.set("RelaxationFitTime", [c[0] for c in block["components"]],
                         "CONTIN Results: T2[k]")
                meta.set("RelaxationFitAmplitude", [c[2] for c in block["components"]],
                         "CONTIN Results: A[k]")
                meta.vendor.update({"settings": block["settings"], "integral": block["integral"],
                                    "gauss_fit": block["gauss"],
                                    "time_errors": [c[1] for c in block["components"]],
                                    "date": block["date"].isoformat() if block["date"] else None})
                protocols.append(Protocol(block["title"], (), meta, self.provenance(sniff)))
                continue
            fit = MinispecReader.fit(block)
            meta.set("SampleID", block["sample"], "SampleId")
            meta.set("RelaxationFitAmplitude", fit["amplitude"], "Amplitude")
            meta.set("RelaxationFitTime", fit["time"], "T2")
            meta.set("RelaxationFitOffset", fit["offset"], "Offset")
            tau = Report.pulse_separation(block["settings"])
            if tau is not None:
                meta.set("EchoTime", 2 * tau, "2 x Pulse Separ.")
            meta.vendor.update({"settings": block["settings"], "errors": {
                "amplitude": fit["amplitude_error"], "time": fit["time_error"],
                "offset": fit["offset_error"]},
                "date": block["date"].isoformat() if block["date"] else None})
            protocols.append(Protocol(block["title"], (), meta, self.provenance(sniff)))
        return protocols
