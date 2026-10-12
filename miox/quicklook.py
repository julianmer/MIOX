####################################################################################################
#                                           quicklook.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 12/10/26                                                                                #
#                                                                                                  #
# Purpose: The Quick Look page of a spectroscopy file: a header line, then tabs for the processed  #
# spectrum (miox.processing, Augmentrum's raw pipeline), every coil with its weight, every average #
# before and after alignment, the water reference and all header values; plain HTML with inline    #
# SVG, light and dark.                                                                             #
#                                                                                                  #
####################################################################################################


#*************#
# imports                                                                                          #
#*************#
from __future__ import annotations

import base64
import html
import zlib
from pathlib import Path

import numpy as np

# own
from miox.data import MRSData
from miox.processing import PROTON_REFERENCE, RawPipeline, Spectral
from miox.sniff import Format

NMR_FORMATS = {Format.TOPSPIN, Format.TOPSPIN_PROCESSED, Format.JCAMP_NMR, Format.NMRML,
               Format.JEOL_JDF, Format.TECMAG_TNT, Format.NMRPIPE, Format.SPARKY, Format.SIMPSON,
               Format.SPINSOLVE, Format.RS2D, Format.RNMRTK}
METABOLITES = (4.2, 0.2)  # ppm shown for 1H metabolites (Augmentrum's alignment window)
WATER = (8.0, 0.0)


#**************************************************************************************************#
#                                             Prepared                                             #
#**************************************************************************************************#
#                                                                                                  #
# One dataset in the pipeline's layout (1, 1, coils, transients, points), one array per edit       #
# condition, in FSL-MRS's orientation (conjugated): the voxel with the strongest signal of a grid, #
# the first increment of indirect NMR dimensions, other dimensions summed (named in the notes).    #
#                                                                                                  #
#**************************************************************************************************#
class Prepared:
    def __init__(self, d: MRSData):
        self.d, self.notes = d, []
        data = np.asarray(d.data, dtype=np.complex128)
        spatial = data.shape[:3]
        if np.prod(spatial) > 1:
            head = np.abs(data[:, :, :, :16]).reshape(*spatial, -1).sum(axis=-1)
            index = np.unravel_index(int(np.argmax(head)), spatial)
            data = data[index]
            self.notes.append(f"voxel {tuple(int(i) for i in index)} of "
                              f"{' x '.join(map(str, spatial))} (strongest signal)")
        else:
            data = data[0, 0, 0]
        tags = [str(t) for t in d.dim_tags[: data.ndim - 1]]
        for tag in list(tags):  # first increment / sum of dimensions the pipeline has no use for
            if tag in ("DIM_COIL", "DIM_DYN", "DIM_EDIT"):
                continue
            axis = 1 + tags.index(tag)
            if tag.startswith("DIM_INDIRECT"):
                data = data.take(0, axis=axis)
                self.notes.append(f"first of {d.shape[4 + d.dim_tags.index(tag)]} increments")
            else:
                data = data.sum(axis=axis)
                self.notes.append(f"{tag.removeprefix('DIM_').lower()} summed")
            tags.remove(tag)
        for tag in ("DIM_COIL", "DIM_DYN"):
            if tag not in tags:
                data, tags = data[..., None], tags + [tag]
        if "DIM_EDIT" not in tags:
            data, tags = data[..., None], tags + ["DIM_EDIT"]
        order = [1 + tags.index(t) for t in ("DIM_EDIT", "DIM_COIL", "DIM_DYN")] + [0]
        # FSL-MRS's orientation: the conjugate of NIfTI-MRS as stored (nifti_mrs conjugates at
        # use), which the pipeline and FSL-MRS's spectrum and axis expect
        self.conditions = np.transpose(data, order)[:, None, None].conj()  # (E, 1, 1, C, D, T)

    @property
    def edited(self) -> bool:
        return self.conditions.shape[0] > 1


#**************************************************************************************************#
#                                               Plot                                               #
#**************************************************************************************************#
#                                                                                                  #
# Hand-made SVG: lines in the ink colour of the page (light or dark), one hairline axis with       #
# ticks, no frame, no grid, no y-axis (spectra are in arbitrary units).                            #
#                                                                                                  #
#**************************************************************************************************#
class Plot:
    @staticmethod
    def path(x: np.ndarray, y: np.ndarray, box: tuple, xlim: tuple, ylim: tuple) -> str:
        left, top, width, height = box
        px = left + (xlim[0] - x) / (xlim[0] - xlim[1]) * width  # ppm falls to the right
        py = top + (ylim[1] - y) / ((ylim[1] - ylim[0]) or 1.0) * height
        step = max(1, len(x) // 1600)
        points = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(px[::step], py[::step], strict=True))
        return f'<polyline points="{points}"/>'

    @classmethod
    def spectrum(cls, lines: list[tuple[np.ndarray, np.ndarray, str]], xlim: tuple, unit: str,
                 width: int = 900, height: int = 300, labels: bool = True) -> str:
        """``lines``: (x, y, css class); stacked when more than one."""
        pad, axis = 12, 26 if labels else 4
        box_height = (height - axis - pad) / len(lines)
        parts = []
        for i, (x, y, kind) in enumerate(lines):
            mask = (x >= min(xlim)) & (x <= max(xlim))
            ylim = (float(y[mask].min()), float(y[mask].max())) if mask.any() else (0.0, 1.0)
            box = (0, pad + i * box_height, width, box_height * 0.92)
            parts.append(f'<g class="{kind}">{cls.path(x[mask], y[mask], box, xlim, ylim)}</g>')
        if labels:
            parts.append(cls.axis(xlim, unit, width, height - axis))
        return f'<svg viewBox="0 0 {width} {height}" class="plot">{"".join(parts)}</svg>'

    @staticmethod
    def axis(xlim: tuple, unit: str, width: int, y: float) -> str:
        span = abs(xlim[0] - xlim[1])
        step = next(s for s in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 5000)
                    if span / s <= 10)
        ticks = np.arange(np.ceil(min(xlim) / step) * step, max(xlim) + 1e-9, step)
        out = [f'<line class="axis" x1="0" x2="{width}" y1="{y}" y2="{y}"/>']
        for tick in ticks:
            px = (xlim[0] - tick) / (xlim[0] - xlim[1]) * width
            out.append(f'<line class="axis" x1="{px:.1f}" x2="{px:.1f}" y1="{y}" y2="{y + 4}"/>')
            out.append(f'<text x="{px:.1f}" y="{y + 16}" text-anchor="middle">{tick:g}</text>')
        out.append(f'<text x="{width}" y="{y + 16}" text-anchor="end" class="unit">{unit}</text>')
        return "".join(out)

    @staticmethod
    def series(rows: list[tuple[np.ndarray, str]], width: int = 900, height: int = 60) -> str:
        """Small line charts, one per (values, label), with their range written at the right."""
        out = []
        for values, label in rows:
            n = len(values)
            low, high = float(values.min()), float(values.max())
            span = (high - low) or 1.0
            xs = np.linspace(0, width - 160, n)
            ys = 6 + (high - values) / span * (height - 16)
            points = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, ys, strict=True))
            out.append(f'<svg viewBox="0 0 {width} {height}" class="plot">'
                       f'<g class="line"><polyline points="{points}"/></g>'
                       f'<text x="{width}" y="18" text-anchor="end">{label}</text>'
                       f'<text x="{width}" y="36" text-anchor="end" class="unit">'
                       f"{low:.2f} … {high:.2f}</text></svg>")
        return "".join(out)

    @staticmethod
    def image(values: np.ndarray) -> str:
        """Grey levels as a PNG whose ink is transparency (inverted for dark mode by CSS)."""
        scaled = values / (np.percentile(values, 99.5) or 1.0)
        alpha = (np.clip(scaled, 0, 1) * 255).astype(np.uint8)
        height, width = alpha.shape
        rows = b"".join(b"\x00" + bytes(np.stack([np.zeros_like(r), r], -1).ravel())
                        for r in alpha)  # grey-alpha pixels, black ink
        def chunk(kind: bytes, data: bytes) -> bytes:
            return (len(data).to_bytes(4, "big") + kind + data
                    + zlib.crc32(kind + data).to_bytes(4, "big"))
        png = (b"\x89PNG\r\n\x1a\n"
               + chunk(b"IHDR", width.to_bytes(4, "big") + height.to_bytes(4, "big")
                       + bytes([8, 4, 0, 0, 0]))
               + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))
        return "data:image/png;base64," + base64.b64encode(png).decode()


#**************************************************************************************************#
#                                               Page                                               #
#**************************************************************************************************#
#                                                                                                  #
# The page: header, tabs (radio buttons and CSS, no script), panes. Spectra are FSL-MRS's (first   #
# point halved, orthonormal FFT, zero filled twice) on its ppm axis; 1H metabolites show 0.2-4.2   #
# ppm, water 0-8 ppm, other nuclei the whole range. Edit conditions are processed one by one and   #
# shown with their difference (first minus second, as stored).                                     #
#                                                                                                  #
#**************************************************************************************************#
class Page:
    def __init__(self, datasets: list, title: str):
        mrs = [d for d in datasets if isinstance(d, MRSData)]
        if not mrs:
            raise ValueError("no spectroscopy data")
        self.title = title
        self.main = next((d for d in mrs if not d.name.lower().endswith("_ref")), mrs[0])
        self.water = next((d for d in mrs if d is not self.main), None)
        self.prepared = Prepared(self.main)
        d = self.main
        self.processable = d.dwelltime is not None and d.spectrometer_frequency is not None
        nmr = d.provenance.format in NMR_FORMATS or "Solvent" in d.meta
        self.proton = d.nucleus == "1H" and not nmr
        self.reference = float(d.meta.get("ReferencePPM", PROTON_REFERENCE if self.proton
                                          else 0.0))
        self.results = self.process()

    def process(self) -> list:
        if not self.processable:
            return []
        d, water = self.main, None
        if self.water is not None:
            water = Prepared(self.water).conditions.mean(axis=0)  # (1, 1, C, D, T)
        pipeline = RawPipeline(1.0 / d.dwelltime, d.spectrometer_frequency, self.proton,
                               self.reference)
        return [pipeline.run(condition, water) for condition in self.prepared.conditions]

    def axis(self, n: int) -> np.ndarray:
        return Spectral.ppm_axis(n, 1.0 / self.main.dwelltime, self.main.spectrometer_frequency,
                                 self.reference)

    def spectrum(self, fid: np.ndarray):
        fid = np.asarray(fid).reshape(-1, np.shape(fid)[-1])
        spec = Spectral.fid_to_spec(Spectral.zero_fill(fid, 2))
        return self.axis(spec.shape[-1]), spec

    def xlim(self, x: np.ndarray, water: bool = False) -> tuple:
        if self.proton:
            high, low = WATER if water else METABOLITES
            return (min(high, float(x.max())), max(low, float(x.min())))
        return (float(x.max()), float(x.min()))

    def spectrum_pane(self) -> str:
        if not self.results:
            return ("<div class=empty>The file states no dwell time or spectrometer frequency: "
                    "nothing is processed.</div>")
        lines = []
        for r in self.results:
            x, spec = self.spectrum(r.fid)
            lines.append((x, spec[0].real, "line"))
        legend = ""
        if self.prepared.edited and len(self.results) == 2:
            x, difference = self.spectrum(np.asarray(self.results[0].fid)
                                          - np.asarray(self.results[1].fid))
            lines.append((x, difference[0].real, "line accent"))
            legend = "<div class=legend>edit 1 · edit 2 · <b>difference 1 − 2</b></div>"
        else:
            lines = [(lines[0][0], lines[0][1], "line accent")]
        steps = list(dict.fromkeys(self.prepared.notes + self.results[0].steps))
        if self.prepared.edited:
            steps.append("each edit condition processed on its own")
        x = lines[0][0]
        return (Plot.spectrum(lines, self.xlim(x), "ppm", height=320 if len(lines) == 1 else 440)
                + legend + f"<div class=note>{' · '.join(steps)}.</div>")

    def coils_pane(self) -> str:
        conditions = self.prepared.conditions
        coils = conditions.shape[3]
        if coils == 1:
            return "<div class=empty>One channel: the file holds combined data.</div>"
        x, spec = self.spectrum(conditions[0, 0, 0].mean(axis=1))  # (C, T) mean over transients
        values = np.abs(spec)
        xlim = self.xlim(x)
        mask = (x >= min(xlim)) & (x <= max(xlim))
        top = float(values[:, mask].max()) or 1.0
        weights = None
        if self.results and self.results[0].weights is not None:
            w = np.abs(np.asarray(self.results[0].weights)).reshape(-1, coils).mean(axis=0)
            weights = w / (w.sum() or 1.0)
        cells = []
        for c in range(coils):
            path = Plot.path(x[mask], values[c, mask], (0, 4, 240, 62), xlim, (0.0, top))
            weight = f"<span>{weights[c] * 100:.1f} %</span>" if weights is not None else ""
            cells.append(f'<figure><svg viewBox="0 0 240 70" class="plot" '
                         f'preserveAspectRatio="none"><g class="line">{path}</g></svg>'
                         f"<figcaption>{c + 1}{weight}</figcaption></figure>")
        return ("<div class=note>Every coil, mean of its transients, magnitude, one scale for all. "
                "Right: its share of the wSVD weights.</div>"
                f"<div class=grid>{''.join(cells)}</div>")

    def averages_pane(self) -> str:
        r = self.results[0] if self.results else None
        if r is None or r.aligned is None:
            return "<div class=empty>One transient: the file holds averaged data.</div>"
        parts = []
        for label, data in (("before alignment", r.combined), ("after alignment", r.aligned)):
            x, spec = self.spectrum(np.asarray(data)[0, 0, 0])
            xlim = self.xlim(x)
            mask = (x >= min(xlim)) & (x <= max(xlim))
            image = np.abs(spec[:, mask])[:, ::-1]  # ppm falling to the right
            step = max(1, image.shape[1] // 900)
            rows = image.shape[0]
            parts.append(f"<div class=note>{label}</div><img class=heat "
                         f'src="{Plot.image(image[:, ::step])}" '
                         f'style="height:{min(300, max(90, rows * 2))}px">')
        axis = Plot.axis(xlim, "ppm", 900, 2)
        eps = np.asarray(r.eps).reshape(-1)
        phi = np.degrees(np.asarray(r.phi).reshape(-1))
        keep = np.asarray(r.keep).reshape(-1) if r.keep is not None else np.ones(len(eps), bool)
        removed = [str(i + 1) for i in np.flatnonzero(~keep)]
        drift = Plot.series([(eps, "frequency (Hz)"), (phi, "phase (degrees)")])
        text = (f"Removed as unlike: {', '.join(removed)}." if removed
                else "No transient removed as unlike.")
        return (f"<div class=note>All {len(eps)} transients after coil combination, magnitude, "
                f"top to bottom in acquisition order.</div>{''.join(parts)}"
                f'<svg viewBox="0 0 900 24" class="plot">{axis}</svg>'
                f"<div class=note>Correction found for every transient.</div>{drift}"
                f"<div class=note>{text}</div>")

    def water_pane(self) -> str:
        r = self.results[0] if self.results else None
        if self.water is None or r is None or r.water is None:
            return "<div class=empty>No water reference in this file.</div>"
        x, spec = self.spectrum(r.water)
        return Plot.spectrum([(x, spec[0].real, "line")], self.xlim(x, water=True), "ppm",
                             height=260)

    def details_pane(self) -> str:
        rows = []
        for key, value in self.main.meta.items():
            if isinstance(value, (list, tuple)) and len(value) > 6:
                value = f"{len(value)} values"
            rows.append(f"<tr><td>{html.escape(str(key))}</td>"
                        f"<td>{html.escape(str(value))}</td></tr>")
        return f"<table>{''.join(rows)}</table>"

    def facts(self) -> str:
        d, meta, c = self.main, self.main.meta, self.prepared.conditions
        facts = [d.nucleus or ""]
        if d.spectrometer_frequency:
            facts.append(f"{d.spectrometer_frequency:.2f} MHz")
        if "EchoTime" in meta:
            facts.append(f"TE {float(meta['EchoTime']) * 1e3:g} ms")
        if "RepetitionTime" in meta:
            facts.append(f"TR {float(meta['RepetitionTime']) * 1e3:g} ms")
        facts += [f"{c.shape[3]} coils" if c.shape[3] > 1 else "",
                  f"{c.shape[4]} transients" if c.shape[4] > 1 else "",
                  f"{c.shape[0]} edit conditions" if c.shape[0] > 1 else "",
                  f"{d.n_points} points", str(d.provenance.format).replace("_", " ")]
        return " · ".join(f for f in facts if f)

    def html(self) -> str:
        tabs = [("Spectrum", self.spectrum_pane()), ("Coils", self.coils_pane()),
                ("Averages", self.averages_pane()), ("Water", self.water_pane()),
                ("Details", self.details_pane())]
        radios = "".join(f'<input type=radio name=tab id=t{i}{" checked" if not i else ""}>'
                         for i in range(len(tabs)))
        labels = "".join(f"<label for=t{i}>{name}</label>" for i, (name, _) in enumerate(tabs))
        panes = "".join(f"<section id=p{i}>{body}</section>" for i, (_, body) in enumerate(tabs))
        show = ",".join(f"#t{i}:checked~main #p{i}" for i in range(len(tabs)))
        mark = ",".join(f"#t{i}:checked~nav label[for=t{i}]" for i in range(len(tabs)))
        return PAGE.format(title=html.escape(self.title), facts=html.escape(self.facts()),
                           radios=radios, labels=labels, panes=panes, show=show, mark=mark)


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="color-scheme" content="light dark"><style>
:root {{ --bg:#ffffff; --ink:#1d1d1f; --soft:#6e6e73; --faint:#aeaeb2; --rule:#e8e8ed;
         --accent:#0071e3; --pill:#f2f2f7; --heat:none; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#1c1c1e; --ink:#f5f5f7; --soft:#a1a1a6; --faint:#636366; --rule:#38383a;
           --accent:#2997ff; --pill:#2c2c2e; --heat:invert(1); }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:22px 26px; background:var(--bg); color:var(--ink);
        font:13px/1.45 -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
        -webkit-font-smoothing:antialiased; }}
h1 {{ font-size:15px; font-weight:600; margin:0; letter-spacing:-.01em; }}
.facts {{ color:var(--soft); font-size:12px; margin-top:2px; font-variant-numeric:tabular-nums; }}
input {{ display:none; }}
nav {{ margin:16px 0 14px; display:flex; gap:2px; }}
nav label {{ padding:3px 11px; border-radius:7px; font-size:12px; color:var(--soft);
             cursor:default; }}
{mark} {{ background:var(--pill); color:var(--ink); }}
section {{ display:none; }} {show} {{ display:block; }}
.plot {{ width:100%; height:auto; display:block; overflow:visible; }}
.plot polyline {{ fill:none; stroke:var(--ink); stroke-width:1.1; stroke-linejoin:round;
                  vector-effect:non-scaling-stroke; }}
.plot .accent polyline {{ stroke:var(--accent); stroke-width:1.4; }}
.plot .axis {{ stroke:var(--rule); stroke-width:1; vector-effect:non-scaling-stroke; }}
.plot text {{ fill:var(--soft); font-size:11px; font-variant-numeric:tabular-nums; }}
.plot .unit {{ fill:var(--faint); }}
.note, .legend {{ color:var(--faint); font-size:11px; margin-top:8px; }}
.legend b {{ color:var(--accent); font-weight:500; }}
.empty {{ color:var(--soft); padding:40px 0; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(150px, 1fr));
         gap:14px 18px; margin-top:12px; }}
figure {{ margin:0; }}
figcaption {{ display:flex; justify-content:space-between; font-size:11px; color:var(--soft);
              border-top:1px solid var(--rule); padding-top:3px; margin-top:2px;
              font-variant-numeric:tabular-nums; }}
figcaption span {{ color:var(--faint); }}
.heat {{ width:100%; display:block; image-rendering:auto; filter:var(--heat);
         margin-top:10px; }}
table {{ border-collapse:collapse; width:100%; font-size:12px; }}
td {{ padding:4px 0; border-bottom:1px solid var(--rule); vertical-align:top; }}
td:first-child {{ color:var(--soft); width:38%; padding-right:16px; }}
</style></head><body>
<h1>{title}</h1><div class=facts>{facts}</div>
{radios}<nav>{labels}</nav><main>{panes}</main>
</body></html>"""


def page(datasets: list, title: str) -> str:
    return Page(datasets, title).html()


def write(datasets: list, title: str, path: str | Path) -> None:
    Path(path).write_text(page(datasets, title), encoding="utf-8")
