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
# before and after alignment, the water reference and all header values; of an image file, three   #
# planes, axial slices and all header values. Plain HTML with inline SVG, light and dark.          #
#                                                                                                  #
####################################################################################################


#*************#
# imports                                                                                          #
#*************#
from __future__ import annotations

import base64
import html
import re
import zlib
from pathlib import Path

import numpy as np

# own
from miox.data import BasisSet, FitResult, MRImage, MRSData
from miox.processing import PROTON_REFERENCE, RawPipeline, Spectral
from miox.sniff import Format, Sniffer

NMR_FORMATS = {Format.TOPSPIN, Format.TOPSPIN_PROCESSED, Format.JCAMP_NMR, Format.NMRML,
               Format.JEOL_JDF, Format.TECMAG_TNT, Format.NMRPIPE, Format.SPARKY, Format.SIMPSON,
               Format.SPINSOLVE, Format.RS2D, Format.RNMRTK}
PROTON_RANGE = (6.0, 0.0)  # ppm shown for 1H, water included
METABOLITE_SCALE = (4.2, 0.2)  # ppm whose values set the vertical range of the Metabolites view
GRID = 32  # voxels a side shown on the Grid tab
READER_NOTES = ("reconstruct", "untested", "assumed", "not converted", "caution", "turned into")
KSPACE = "k-space, not reconstructed"  # how a reader marks data that are k-space, not voxels


#**************************************************************************************************#
#                                             Prepared                                             #
#**************************************************************************************************#
#                                                                                                  #
# One dataset in the pipeline's layout (1, voxels, coils, transients, points), one array per edit  #
# condition, in FSL-MRS's orientation (conjugated): every voxel of a grid (the centre one is the   #
# focus), the first increment of indirect NMR dimensions, other dimensions summed (named in the    #
# notes).                                                                                          #
#                                                                                                  #
#**************************************************************************************************#
class Prepared:
    def __init__(self, d: MRSData):
        self.d, self.notes = d, []
        data = np.asarray(d.data, dtype=np.complex128)
        self.spatial = data.shape[:3]
        data = data.reshape(-1, *data.shape[3:])  # (V, T, dims)
        tags = [str(t) for t in d.dim_tags[: data.ndim - 2]]
        for tag in ("DIM_COIL", "DIM_DYN", "DIM_EDIT"):  # a tag given twice: one dimension
            while tags.count(tag) > 1:
                first, second = [2 + i for i, t in enumerate(tags) if t == tag][:2]
                data = np.moveaxis(data, second, first + 1)
                data = data.reshape(*data.shape[:first], -1, *data.shape[first + 2:])
                tags.pop(second - 2)
                self.notes.append(f"two {tag.removeprefix('DIM_').lower()} dimensions as one")
        for tag in list(tags):  # first increment / sum of dimensions the pipeline has no use for
            if tag in ("DIM_COIL", "DIM_DYN", "DIM_EDIT"):
                continue
            axis = 2 + tags.index(tag)
            if tag.startswith("DIM_INDIRECT"):
                data = data.take(0, axis=axis)
                self.notes.append(f"first of {d.shape[4 + d.dim_tags.index(tag)]} increments")
            else:
                data = data.sum(axis=axis)
                self.notes.append(f"{tag.removeprefix('DIM_').lower()} summed")
            tags.remove(tag)
        for tag in ("DIM_COIL", "DIM_DYN", "DIM_EDIT"):
            if tag not in tags:
                data, tags = data[..., None], tags + [tag]
        order = [2 + tags.index(t) for t in ("DIM_EDIT",)] + [0] + [
            2 + tags.index(t) for t in ("DIM_COIL", "DIM_DYN")] + [1]
        # FSL-MRS's orientation: the conjugate of NIfTI-MRS as stored (nifti_mrs conjugates at
        # use), which the pipeline and FSL-MRS's spectrum and axis expect
        self.conditions = np.transpose(data, order)[:, None].conj()  # (E, 1, V, C, D, T)
        centre = tuple(n // 2 for n in self.spatial)
        self.focus = int(np.ravel_multi_index(centre, self.spatial))  # the centre voxel

    @property
    def grid(self) -> bool:
        return int(np.prod(self.spatial)) > 1

    @property
    def focus_index(self) -> tuple:
        return tuple(int(i) for i in np.unravel_index(self.focus, self.spatial))

    @property
    def edited(self) -> bool:
        return self.conditions.shape[0] > 1


#**************************************************************************************************#
#                                              Water                                               #
#**************************************************************************************************#
#                                                                                                  #
# Water scans and water references. A dataset is a water scan when its name ends in ref, h2o or    #
# water, when its header says it is not water suppressed, or when water dominates it (1H; peak in  #
# 4.2-5.1 ppm over 300 times the tallest of NAA, creatine and choline above their surroundings, on #
# the unprocessed spectrum). A separate water reference is a file next to the data named like it   #
# with _H2O, _ref, _water or _wref added, _act or _svs swapped for _ref or _mrsref.                #
#                                                                                                  #
#**************************************************************************************************#
class Water:
    NAME = re.compile(r"(^|[_\-. ])(ref|wref|mrsref|h2o|water)$", re.IGNORECASE)
    ADDED = ("_H2O", "_ref", "_water", "_wref")
    SWAPPED = (("_act", "_ref"), ("_svs", "_mrsref"))
    DOMINANCE = 10 ** 2.5
    METABOLITES = (2.01, 3.03, 3.2)  # ppm of NAA, creatine, choline

    @classmethod
    def named(cls, d: MRSData) -> bool:
        return bool(cls.NAME.search(d.name or "")) or d.meta.get("WaterSuppressed") is False

    @classmethod
    def dominates(cls, prepared: Prepared) -> bool:
        d = prepared.d
        if d.nucleus != "1H" or d.dwelltime is None or d.spectrometer_frequency is None:
            return False
        fids = prepared.conditions[0, 0, prepared.focus].mean(axis=1)  # (C, T)
        spec = np.abs(Spectral.fid_to_spec(Spectral.zero_fill(fids, 2))).sum(axis=0)
        x = Spectral.ppm_axis(spec.size, 1.0 / d.dwelltime, d.spectrometer_frequency)
        water, prominence = (x > 4.2) & (x < 5.1), []
        for ppm in cls.METABOLITES:
            peak, around = np.abs(x - ppm) < 0.06, np.abs(x - ppm) < 0.3
            if not peak.any() or not water.any():
                return False
            prominence.append(spec[peak].max() - np.median(spec[around]))
        return bool(spec[water].max() > cls.DOMINANCE * max(max(prominence), 0.0))

    @classmethod
    def partner(cls, path: Path) -> Path | None:
        """The water reference file next to path, if there is one."""
        name = path.name
        stem, suffix = (name[: -len(".nii.gz")], ".nii.gz") if name.lower().endswith(
            ".nii.gz") else (path.stem, path.suffix)
        stems = [stem + added for added in cls.ADDED] + [
            stem[: -len(old)] + new for old, new in cls.SWAPPED if stem.lower().endswith(old)]
        try:
            present = {p.name.lower(): p for p in path.parent.iterdir() if p.is_file()}
        except OSError:
            return None
        return next((present[c] for s in stems if (c := (s + suffix).lower()) in present), None)


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
    def path(x: np.ndarray, y: np.ndarray, box: tuple, xlim: tuple, ylim: tuple,
             points: int = 1600) -> str:
        """Polylines through (x, y), thinned to about ``points`` by keeping the lowest and the
        highest value of every bin (peaks survive), broken where y leaves ylim (the line runs
        to the edge of the box)."""
        if len(x) > points:
            k = -(-len(x) // (points // 2))
            rows = np.arange(len(x) // k) * k
            bins = y[: len(rows) * k].reshape(-1, k)
            low, high = bins.argmin(axis=1), bins.argmax(axis=1)
            index = np.stack([rows + np.minimum(low, high), rows + np.maximum(low, high)], 1)
            x, y = x[index.ravel()], y[index.ravel()]
        left, top, width, height = box
        px = left + (xlim[0] - x) / (xlim[0] - xlim[1]) * width  # ppm falls to the right
        py = top + (ylim[1] - y) / ((ylim[1] - ylim[0]) or 1.0) * height
        inside = (y >= ylim[0]) & (y <= ylim[1])
        py = np.clip(py, top, top + height)
        runs = np.flatnonzero(np.diff(np.concatenate([[0], inside.astype(int), [0]])))
        out = []
        for start, stop in zip(runs[::2], runs[1::2], strict=True):
            start, stop = max(start - 1, 0), min(stop + 1, len(x))  # on to the edge
            points = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(px[start:stop], py[start:stop],
                                                                  strict=True))
            out.append(f'<polyline points="{points}"/>')
        return "".join(out)

    @staticmethod
    def ylim(x: np.ndarray, ys: list[np.ndarray], window: tuple, pad: float = 0.0) -> tuple:
        """The vertical range of ys over the ppm window, widened by ``pad`` of it."""
        mask = (x >= min(window)) & (x <= max(window))
        if not mask.any():
            return (0.0, 1.0)
        low = min(float(y[mask].min()) for y in ys)
        high = max(float(y[mask].max()) for y in ys)
        return (low - pad * (high - low), high + pad * (high - low))

    @classmethod
    def spectrum(cls, lines: list[tuple[np.ndarray, np.ndarray, str]], xlim: tuple, unit: str,
                 width: int = 900, height: int = 300, labels: bool = True,
                 window: tuple | None = None) -> str:
        """``lines``: (x, y, css class); stacked when more than one. Each on its own vertical
        range: over xlim, or over ``window`` (lines leaving it are cut at the edge)."""
        pad, axis = 12, 26 if labels else 4
        box_height = (height - axis - pad) / len(lines)
        parts = []
        for i, (x, y, kind) in enumerate(lines):
            mask = (x >= min(xlim)) & (x <= max(xlim))
            ylim = cls.ylim(x, [y], window or xlim, 0.06 if window else 0.0)
            box = (0, pad + i * box_height, width, box_height * 0.92)
            parts.append(f'<g class="{kind}">{cls.path(x[mask], y[mask], box, xlim, ylim)}</g>')
        if labels:
            parts.append(cls.axis(xlim, unit, width, height - axis))
        return f'<svg viewBox="0 0 {width} {height}" class="plot">{"".join(parts)}</svg>'

    @classmethod
    def overlay(cls, x: np.ndarray, thin: list[tuple[np.ndarray, str, str]], bold: np.ndarray,
                xlim: tuple, unit: str, window: tuple | None = None, width: int = 900,
                height: int = 280, ylim: tuple | None = None) -> str:
        """Many thin lines (y, css class, hover title) under one bold line, all on one vertical
        range (over xlim, or over ``window``, or given), FSL-MRS's plot_spectra."""
        mask = (x >= min(xlim)) & (x <= max(xlim))
        ylim = ylim or cls.ylim(x, [y for y, _, _ in thin] + [bold], window or xlim,
                                0.06 if window else 0.0)
        box = (0, 12, width, height - 44)
        parts = [f'<g class="{kind}"><title>{html.escape(title)}</title>'
                 f"{cls.path(x[mask], y[mask], box, xlim, ylim, points=700)}</g>"
                 for y, kind, title in thin]
        parts.append(f'<g class="bold">{cls.path(x[mask], bold[mask], box, xlim, ylim)}</g>')
        parts.append(cls.axis(xlim, unit, width, height - 26))
        return f'<svg viewBox="0 0 {width} {height}" class="plot">{"".join(parts)}</svg>'

    @staticmethod
    def axis(xlim: tuple, unit: str, width: int, y: float) -> str:
        span = abs(xlim[0] - xlim[1])
        step = next(s for s in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 5000)
                    if span / s <= width // 90)  # about one tick per 90 units
        ticks = np.arange(np.ceil(min(xlim) / step) * step, max(xlim) + 1e-9, step)
        out = [f'<line class="axis" x1="0" x2="{width}" y1="{y}" y2="{y}"/>']
        unit = f'<tspan class="unit"> {unit}</tspan>'
        for tick in ticks:
            px = (xlim[0] - tick) / (xlim[0] - xlim[1]) * width
            out.append(f'<line class="axis" x1="{px:.1f}" x2="{px:.1f}" y1="{y}" y2="{y + 4}"/>')
            if px > width - 40:  # the last tick carries the unit
                out.append(f'<text x="{width}" y="{y + 16}" text-anchor="end">'
                           f"{tick:g}{unit}</text>")
                unit = ""
            else:
                out.append(f'<text x="{px:.1f}" y="{y + 16}" text-anchor="middle">{tick:g}</text>')
        if unit:
            out.append(f'<text x="{width}" y="{y + 16}" text-anchor="end">{unit}</text>')
        return "".join(out)

    @staticmethod
    def series(values: np.ndarray, label: str, removed: np.ndarray, width: int = 900,
               height: int = 104) -> str:
        """One value per transient: dots joined by a hairline, transients removed as unlike
        hollow; the label above, the range at the left, transient numbers below."""
        n, gutter = len(values), 56
        low, high = float(values.min()), float(values.max())
        span = (high - low) or 1.0
        xs = gutter + np.arange(n) / max(n - 1, 1) * (width - gutter - 6)
        ys = 26 + (high - values) / span * (height - 52)
        points = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, ys, strict=True))
        dots = "".join(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="2.4" '
                       f'class="{"drop" if r else "dot"}"><title>transient {i + 1}: {v:.2f}'
                       "</title></circle>"
                       for i, (a, b, v, r) in enumerate(zip(xs, ys, values, removed, strict=True)))
        step = next(s for s in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if n / s <= 12)
        ticks = "".join(f'<text x="{xs[k - 1]:.1f}" y="{height - 4}" text-anchor="middle">{k}'
                        "</text>" for k in sorted({1, n} | set(range(step, n - step // 2, step))))
        return (f'<svg viewBox="0 0 {width} {height}" class="plot series">'
                f'<text x="0" y="11" class="label">{label}</text>'
                f'<text x="{gutter - 10}" y="{ys.min() + 4:.1f}" text-anchor="end" class="unit">'
                f"{high:.2f}</text>"
                f'<text x="{gutter - 10}" y="{ys.max() + 4:.1f}" text-anchor="end" class="unit">'
                f"{low:.2f}</text>"
                f'<g class="hair"><polyline points="{points}"/></g>{dots}{ticks}</svg>')

    @staticmethod
    def png(pixels: np.ndarray) -> str:
        """8-bit grey pixels (rows, columns) as a PNG data URL."""
        height, width = pixels.shape
        rows = b"".join(b"\x00" + row.tobytes() for row in pixels.astype(np.uint8))
        def chunk(kind: bytes, data: bytes) -> bytes:
            return (len(data).to_bytes(4, "big") + kind + data
                    + zlib.crc32(kind + data).to_bytes(4, "big"))
        png = (b"\x89PNG\r\n\x1a\n"
               + chunk(b"IHDR", width.to_bytes(4, "big") + height.to_bytes(4, "big")
                       + bytes([8, 0, 0, 0, 0]))
               + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))
        return "data:image/png;base64," + base64.b64encode(png).decode()

    @classmethod
    def grey(cls, values: np.ndarray, window: tuple) -> str:
        """An MR image as opaque grey levels, ``window`` (low, high) from black to white."""
        scaled = (values - window[0]) / ((window[1] - window[0]) or 1.0)
        return cls.png(np.round(np.clip(np.nan_to_num(scaled), 0, 1) * 255))


#**************************************************************************************************#
#                                               Page                                               #
#**************************************************************************************************#
#                                                                                                  #
# The page: header, tabs (radio buttons and CSS, no script), panes. Spectra are FSL-MRS's (first   #
# point halved, orthonormal FFT, zero filled twice) on its ppm axis; 1H shows 0-6 ppm, other       #
# nuclei the whole range. Water scans take the pipeline's water path; other data use the water     #
# reference in the file or, given the file's path, next to it. Edit conditions are processed one   #
# by one and shown with their difference (first minus second, as stored).                          #
#                                                                                                  #
#**************************************************************************************************#
class Page:
    def __init__(self, datasets: list, title: str, source: Path | None = None,
                 notes: tuple[str, ...] = ()):
        mrs = [d for d in datasets if isinstance(d, MRSData)]
        if not mrs:
            raise ValueError("no spectroscopy data")
        self.title, self.source, self.notes = title, source, list(notes)
        self.main = next((d for d in mrs if not Water.named(d)), mrs[0])
        self.water = next((d for d in mrs if d is not self.main), None)
        self.prepared = Prepared(self.main)
        self.water_scan = self.water is None and (Water.named(self.main)
                                                  or Water.dominates(self.prepared))
        self.water_notes = []
        if self.water is None and not self.water_scan and source is not None:
            self.water = self.partner(Path(source))
        d = self.main
        self.blank = not np.any(self.prepared.conditions)  # e.g. MRSI a converter left empty
        self.kspace = any(n.startswith(KSPACE) for n in d.provenance.notes)
        self.processable = (d.dwelltime is not None and d.spectrometer_frequency is not None
                            and not self.blank and not self.kspace)
        nmr = d.provenance.format in NMR_FORMATS or "Solvent" in d.meta
        self.proton = d.nucleus == "1H" and not nmr
        self.reference = float(d.meta.get("ReferencePPM", PROTON_REFERENCE if self.proton
                                          else 0.0))
        self.results = self.process()

    def partner(self, source: Path) -> MRSData | None:
        """The water reference next to the file, when it matches the data point for point."""
        path = Water.partner(source)
        if path is None:
            return None
        from miox.readers import Loader, ReaderError

        try:
            water = next((w for w in Loader().load(path) if isinstance(w, MRSData)), None)
        except ReaderError:
            water = None
        d = self.main
        if (water is None or water.n_points != d.n_points or water.dwelltime is None
                or d.dwelltime is None or not np.isclose(water.dwelltime, d.dwelltime)
                or water.data.shape[:3] != d.data.shape[:3]):
            self.water_notes.append(f"{path.name} next to the file does not match the data: "
                                    "not used as water reference")
            return None
        self.water_notes.append(f"water reference {path.name} (next to the file)")
        return water

    def process(self) -> list:
        if not self.processable:
            return []
        d, water = self.main, None
        pipeline = RawPipeline(1.0 / d.dwelltime, d.spectrometer_frequency, self.proton,
                               self.reference)
        if self.water_scan:
            return [pipeline.run_water(condition) for condition in self.prepared.conditions]
        if self.water is not None:
            water = Prepared(self.water).conditions.mean(axis=0)  # (1, V, C, D, T)
        return [pipeline.run(condition, water) for condition in self.prepared.conditions]

    def axis(self, n: int) -> np.ndarray:
        return Spectral.ppm_axis(n, 1.0 / self.main.dwelltime, self.main.spectrometer_frequency,
                                 self.reference)

    def spectrum(self, fid: np.ndarray):
        fid = np.asarray(fid).reshape(-1, np.shape(fid)[-1])
        spec = Spectral.fid_to_spec(Spectral.zero_fill(fid, 2))
        return self.axis(spec.shape[-1]), spec

    def xlim(self, x: np.ndarray) -> tuple:
        if self.proton:
            high, low = PROTON_RANGE
            return (min(high, float(x.max())), max(low, float(x.min())))
        return (float(x.max()), float(x.min()))

    def focus(self, array) -> np.ndarray:
        """The focus voxel of a (1, V, ...) result array, as (..., T)."""
        return np.asarray(array)[0, self.prepared.focus]

    @property
    def switch(self) -> bool:
        """1H data other than water scans: the Full | Metabolites switch."""
        return self.proton and not self.water_scan and bool(self.results)

    def scaled(self, render) -> str:
        """render(window) on the full scale and, with the switch, on the metabolites' scale."""
        if not self.switch:
            return render(None)
        return (f"<div class=full>{render(None)}</div>"
                f"<div class=met>{render(METABOLITE_SCALE)}</div>")

    def spectrum_pane(self) -> str:
        if self.blank:
            notes = "".join(f"<br>{html.escape(n)}." for n in self.main.provenance.notes)
            return f"<div class=empty>The data are all zero: nothing to show.{notes}</div>"
        if self.kspace:
            notes = "".join(f"<br>{html.escape(n)}." for n in self.main.provenance.notes
                            if n.startswith(KSPACE))
            return f"<div class=empty>The data are k-space: nothing to show.{notes}</div>"
        if not self.results:
            return ("<div class=empty>The file states no dwell time or spectrometer frequency: "
                    "nothing is processed.</div>")
        lines = []
        for r in self.results:
            x, spec = self.spectrum(self.focus(r.fid))
            lines.append((x, spec[0].real, "line"))
        legend = ""
        if self.prepared.edited and len(self.results) == 2:
            x, difference = self.spectrum(self.focus(self.results[0].fid)
                                          - self.focus(self.results[1].fid))
            lines.append((x, difference[0].real, "line accent"))
            legend = "<div class=legend>edit 1 · edit 2 · <b>difference 1 − 2</b></div>"
        else:
            lines = [(lines[0][0], lines[0][1], "line accent")]
        where = []
        if self.prepared.grid:
            where = [f"voxel {self.prepared.focus_index} of "
                     f"{' x '.join(map(str, self.prepared.spatial))} (centre)"]
        if self.water_scan:
            where.append("water scan")
        steps = list(dict.fromkeys(where + self.notes + self.prepared.notes + self.water_notes
                                   + self.results[0].steps))
        if self.prepared.edited:
            steps.append("each edit condition processed on its own")
        x, height = lines[0][0], 320 if len(lines) == 1 else 440
        return (self.scaled(lambda window: Plot.spectrum(lines, self.xlim(x), "ppm",
                                                         height=height, window=window))
                + legend + f"<div class=note>{' · '.join(steps)}.</div>" + self.reader_notes())

    def reader_notes(self) -> str:
        """What the reader warned about the data (e.g. not reconstructed, untested orientation);
        console hints and header syntax warnings are left out (miox info lists them)."""
        notes = [html.escape(n.strip().rstrip(".")) for n in self.main.provenance.notes
                 if any(word in n.lower() for word in READER_NOTES)]
        return "".join(f"<div class=note>Reader: {n}.</div>" for n in notes)

    def coils_pane(self) -> str:
        """Every coil as it enters the combination (times its wSVD weight, so in phase; mean of
        its transients), thin and scaled by the number of coils, under the combined spectrum."""
        raw = self.prepared.conditions[0, 0, self.prepared.focus]  # (C, D, T)
        coils = raw.shape[0]
        if coils == 1:
            return "<div class=empty>One channel: the file holds combined data.</div>"
        r = self.results[0] if self.results else None
        if r is None or r.weights is None:
            return "<div class=empty>Not processed: no coil weights.</div>"
        weights = self.focus(r.weights)  # (C,) from water, (D, C) per transient
        if weights.ndim == 1:
            contributions = raw.mean(axis=1) * weights[:, None]
            share = np.abs(weights)
        else:
            contributions = np.einsum("cdt,dc->ct", raw, weights) / raw.shape[1]
            share = np.abs(weights).mean(axis=0)
        share = share / (share.sum() or 1.0)
        x, spec = self.spectrum(contributions * coils)
        _, combined = self.spectrum(contributions.sum(axis=0))
        thin = [(spec[c].real, "thin", f"coil {c + 1} · weight {share[c]:.1%}")
                for c in range(coils)]
        xlim = self.xlim(x)
        return (self.scaled(lambda window: Plot.overlay(x, thin, combined[0].real, xlim, "ppm",
                                                        window, height=340))
                + f"<div class=note>{coils} coils, each times its wSVD weight (in phase), mean of "
                "its transients, real part; scaled so that their mean is the combined spectrum "
                "(bold). Point at a line for its coil and weight.</div>")

    def averages_pane(self) -> str:
        """Every transient before and after alignment (thin; removed as unlike: dashed red),
        their mean (bold; after alignment of the kept ones), then the frequency and phase
        found for each transient, FSL-MRS's alignment report."""
        r = self.results[0] if self.results else None
        if r is None or r.aligned is None:
            return "<div class=empty>One transient: the file holds averaged data.</div>"
        x, before = self.spectrum(self.focus(r.combined)[0])  # (D, T)
        _, after = self.spectrum(self.focus(r.aligned)[0])
        transients = len(before)
        keep = (np.asarray(r.keep).reshape(-1) if r.keep is not None
                else np.ones(transients, bool))
        thin = lambda spec: [(spec[d].real, "thin" if keep[d] else "drop",  # noqa: E731
                              f"transient {d + 1}" + ("" if keep[d] else " · removed as unlike"))
                             for d in range(transients)]
        xlim = self.xlim(x)
        if self.proton and not self.water_scan:
            xlim = METABOLITE_SCALE  # the window the transients are aligned on
        ylim = Plot.ylim(x, [*before.real, *after.real], xlim)  # one range for both
        plots = []
        for label, spec, mean in (("Before alignment", before, before.real.mean(axis=0)),
                                  ("After alignment", after, after.real[keep].mean(axis=0))):
            plots.append(f"<h2>{label}</h2>"
                         + Plot.overlay(x, thin(spec), mean, xlim, "ppm", height=250, ylim=ylim))
        voxels = len(self.prepared.conditions[0, 0])
        eps = np.asarray(r.eps).reshape(voxels, -1)[self.prepared.focus]
        phi = np.degrees(np.asarray(r.phi).reshape(voxels, -1)[self.prepared.focus])
        removed = ~keep
        edit = " (first edit condition)" if self.prepared.edited else ""
        numbers = ", ".join(str(i + 1) for i in np.flatnonzero(removed))
        text = (f"{transients} transients after coil combination{edit}, real part, unphased; "
                "bold: their mean (after alignment: of the kept ones). "
                + (f"Removed as unlike (dashed, hollow): {numbers}." if removed.any()
                   else "No transient removed as unlike."))
        return ("".join(plots) + f"<div class=note>{text}</div>"
                + "<h2>Correction per transient</h2>"
                + Plot.series(eps, "frequency (Hz)", removed)
                + Plot.series(phi, "phase (degrees)", removed))

    def grid_pane(self) -> str:
        """The slice through the focus voxel: every voxel's spectrum (real part, one range for
        all, cut at the 0.1 and 99.9 percentiles), its cell shaded by its signal, beside the focus
        voxel drawn large; grids larger than GRID voxels a side show the GRID around the focus."""
        r, (nx, ny, nz) = self.results[0], self.prepared.spatial
        fi, fj, k = self.prepared.focus_index
        x, spec = self.spectrum(np.asarray(r.fid)[0, :, 0, 0])  # (V, points)
        spec = spec.reshape(nx, ny, nz, -1)[:, :, k]
        xlim = self.xlim(x)
        i0, j0 = (max(0, min(f - GRID // 2, n - GRID)) for f, n in ((fi, nx), (fj, ny)))
        columns, rows = range(i0, min(nx, i0 + GRID)), range(j0, min(ny, j0 + GRID))

        def render(window):
            mask = (x >= min(window or xlim)) & (x <= max(window or xlim))
            values = spec[columns[0]:columns[-1] + 1, rows[0]:rows[-1] + 1][..., mask].real
            ylim = tuple(float(v) for v in np.percentile(values, (0.1, 99.9)))
            signal = np.abs(values).sum(axis=-1)
            signal = signal / (signal.max() or 1.0)
            view = (x >= min(xlim)) & (x <= max(xlim))
            cells = []
            for j in reversed(rows):  # the top row first
                for i in columns:
                    path = Plot.path(x[view], spec[i, j, view].real, (2, 3, 96, 54), xlim, ylim,
                                     points=200)
                    mark = ' class="focus"' if (i, j) == (fi, fj) else ""
                    shade = 0.04 + 0.3 * signal[i - columns[0], j - rows[0]]
                    cells.append(f'<svg viewBox="0 0 100 60" preserveAspectRatio="none"{mark} '
                                 f'style="background:rgba(127,127,127,{shade:.2f})">'
                                 f"<title>voxel ({i}, {j}, {k})</title>{path}</svg>")
            focus = Plot.spectrum([(x, spec[fi, fj].real, "line accent")], xlim, "ppm",
                                  width=400, height=260, window=window)
            return (f"<div class=mrsi><div class=cells style=\"grid-template-columns:"
                    f"repeat({len(columns)}, 1fr)\">{''.join(cells)}</div>"
                    f"<div><h2>Voxel ({fi}, {fj}, {k})</h2>{focus}</div></div>")

        part = (f"voxels {columns[0]}-{columns[-1]} x {rows[0]}-{rows[-1]} of {nx} x {ny}, "
                if len(columns) < nx or len(rows) < ny else "")
        return (self.scaled(render)
                + f"<div class=note>Slice {k + 1} of {nz}, {part}every voxel's spectrum on one "
                "scale, its cell shaded by its signal; the marked voxel (the centre) is drawn "
                "large. Point at a cell for its voxel.</div>")

    def water_pane(self) -> str:
        r = self.results[0] if self.results else None
        if self.water_scan:
            return "<div class=empty>This file is a water scan: the Spectrum tab shows it.</div>"
        if self.water is None or r is None or r.water is None:
            return ("<div class=empty>No water reference in this file"
                    + (" or next to it." if self.source is not None else ".")
                    + "".join(f"<br>{html.escape(n)}." for n in self.water_notes) + "</div>")
        x, spec = self.spectrum(self.focus(r.water))
        source = "".join(f"<div class=note>{html.escape(n)}.</div>" for n in self.water_notes)
        return Plot.spectrum([(x, spec[0].real, "line")], self.xlim(x), "ppm", height=260) + source

    def details_pane(self) -> str:
        return details(self.main.meta)

    def facts(self) -> str:
        d, meta, c = self.main, self.main.meta, self.prepared.conditions
        facts = [d.nucleus or ""]
        if d.spectrometer_frequency:
            facts.append(f"{d.spectrometer_frequency:.2f} MHz")
        if "EchoTime" in meta:
            facts.append(f"TE {float(meta['EchoTime']) * 1e3:g} ms")
        if "RepetitionTime" in meta:
            facts.append(f"TR {float(meta['RepetitionTime']) * 1e3:g} ms")
        if self.prepared.grid:
            facts.append(" x ".join(map(str, self.prepared.spatial)) + " voxels")
        facts += [f"{c.shape[3]} coils" if c.shape[3] > 1 else "",
                  f"{c.shape[4]} transients" if c.shape[4] > 1 else "",
                  f"{c.shape[0]} edit conditions" if c.shape[0] > 1 else "",
                  f"{d.n_points} points", str(d.provenance.format).replace("_", " ")]
        return " · ".join(f for f in facts if f)

    def html(self) -> str:
        tabs = [("Spectrum", self.spectrum_pane())]
        if self.prepared.grid and self.results:
            tabs.append(("Grid", self.grid_pane()))
        tabs += [("Coils", self.coils_pane()), ("Averages", self.averages_pane()),
                 ("Water", self.water_pane()), ("Details", self.details_pane())]
        return document(self.title, self.facts(), tabs, self.switch)


#**************************************************************************************************#
#                                            ImagePage                                             #
#**************************************************************************************************#
#                                                                                                  #
# The page of an image file: three planes through the centre of the volume (axial, coronal and     #
# sagittal, neurological - the left on the left - as FSLeyes shows them), axial slices from 15 to  #
# 85 % of the volume, and every header value. The volume is turned to its nearest RAS axes; of     #
# more volumes the first is shown, of complex values the magnitude, in a 0.5-99.5 % window.        #
#                                                                                                  #
#**************************************************************************************************#
class ImagePage:
    SLICES = 18  # axial slices on the Slices tab
    PIXELS = 512  # longest side of a picture, larger images are thinned

    def __init__(self, image: MRImage, title: str, others: int = 0):
        self.image, self.title, self.notes = image, title, []
        data = np.asarray(image.data)
        if data.ndim > 3:
            self.notes.append(f"first of {int(np.prod(data.shape[3:]))} volumes")
            data = data.reshape(*data.shape[:3], -1)[..., 0]
        data = np.abs(data) if np.iscomplexobj(data) else data.astype(np.float64)
        data = data.reshape(data.shape + (1,) * (3 - data.ndim))
        self.volume, self.sizes = self.ras(data, np.asarray(image.affine, dtype=np.float64))
        finite = self.volume[np.isfinite(self.volume)]
        self.window = (tuple(float(v) for v in np.percentile(finite, (0.5, 99.5)))
                       if finite.size else (0.0, 1.0))
        if others:
            self.notes.append(f"{others} more image{'s' if others > 1 else ''} in the file")

    @staticmethod
    def ras(data: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The volume along its nearest RAS axes (i left to right, j back to front, k foot to
        head) and the voxel sizes in that order; as stored when the axes are ambiguous."""
        rotation = affine[:3, :3]
        sizes = np.linalg.norm(rotation, axis=0)
        axes = np.argmax(np.abs(rotation), axis=0)  # the RAS axis each voxel axis runs along
        if len(set(axes.tolist())) < 3:
            return data, sizes
        order = np.argsort(axes)
        data, sizes = np.transpose(data, order), sizes[order]
        for n, axis in enumerate(order):
            if rotation[axes[axis], axis] < 0:
                data = np.flip(data, n)
        return data, sizes

    def picture(self, plane: np.ndarray, across: float, down: float) -> str:
        """plane (rows top to bottom, columns left to right) with its pixel sizes in mm."""
        step = max(1, -(-max(plane.shape) // self.PIXELS))
        ratio = plane.shape[1] * across / (plane.shape[0] * down)
        return (f'<img src="{Plot.grey(plane[::step, ::step], self.window)}" '
                f'style="aspect-ratio:{ratio:.4f}">')

    def planes_pane(self) -> str:
        v, (si, sj, sk) = self.volume, self.sizes
        i, j, k = (n // 2 for n in v.shape)
        planes = [("axial", v[:, :, k].T[::-1], si, sj, ("L", "R"), f"z {k + 1}/{v.shape[2]}"),
                  ("coronal", v[:, j, :].T[::-1], si, sk, ("L", "R"), f"y {j + 1}/{v.shape[1]}"),
                  ("sagittal", v[i, :, :].T[::-1], sj, sk, ("P", "A"), f"x {i + 1}/{v.shape[0]}")]
        figures = [f'<figure>{self.picture(plane, across, down)}<span class="side left">{a}</span>'
                   f'<span class="side right">{b}</span><figcaption><span>{name}</span>'
                   f"<span>{where}</span></figcaption></figure>"
                   for name, plane, across, down, (a, b), where in planes if min(plane.shape) > 1]
        if not figures:  # a single line of voxels
            figures = [f"<figure>{self.picture(v.reshape(1, -1), 1.0, 1.0)}</figure>"]
        return f"<div class=planes>{''.join(figures)}</div>"

    def slices_pane(self) -> str:
        v, (si, sj, _) = self.volume, self.sizes
        ks = np.unique(np.linspace(v.shape[2] * 0.15, v.shape[2] * 0.85, self.SLICES).astype(int))
        cells = "".join(f'<figure>{self.picture(v[:, :, k].T[::-1], si, sj)}'
                        f"<figcaption><span>{k + 1}</span></figcaption></figure>" for k in ks)
        return (f"<div class=slices>{cells}</div><div class=note>{len(ks)} of {v.shape[2]} axial "
                "slices, foot to head.</div>")

    def facts(self) -> str:
        image, meta = self.image, self.image.meta
        facts = [" x ".join(map(str, image.shape)),
                 " x ".join(f"{s:.3g}" for s in image.voxel_sizes) + " mm"]
        if "RepetitionTime" in meta:
            facts.append(f"TR {float(meta['RepetitionTime']) * 1e3:g} ms")
        if "EchoTime" in meta:
            facts.append(f"TE {float(meta['EchoTime']) * 1e3:g} ms")
        return " · ".join(facts + self.notes + [str(image.provenance.format).replace("_", " ")])

    def html(self) -> str:
        tabs = [("Planes", self.planes_pane())]
        if self.volume.shape[2] > 1:
            tabs.append(("Slices", self.slices_pane()))
        tabs.append(("Details", details(self.image.meta)))
        return document(self.title, self.facts(), tabs)


#**************************************************************************************************#
#                                             FitPage                                              #
#**************************************************************************************************#
#                                                                                                  #
# The page of a fit result, as LCModel and FSL-MRS draw one: the residual on top, then data (ink), #
# fit (red) and baseline (grey) on one scale over the fit's ppm window, each metabolite's          #
# component staggered below when the file holds them (Osprey); the table of concentrations and     #
# their SD; every header value.                                                                    #
#                                                                                                  #
#**************************************************************************************************#
class FitPage:
    CURVES = ("ppm", "data", "fit", "baseline")

    def __init__(self, result: FitResult, title: str, others: int = 0):
        self.result, self.title, self.others = result, title, others

    def fit_pane(self) -> str:
        curves, width = self.result.curves, 900
        if not all(k in curves for k in self.CURVES):
            return "<div class=empty>The file holds no fitted curves: see the table.</div>"
        x, data, fit, baseline = (np.asarray(curves[k], dtype=np.float64) for k in self.CURVES)
        xlim = (float(x.max()), float(x.min()))
        ylim = Plot.ylim(x, [data, fit, baseline], xlim)
        residual = data - fit
        box = (0, 60, width, 260)
        wiggle = Plot.path(x, residual, (0, 18, width, 30), xlim, Plot.ylim(x, [residual], xlim))
        main = (f'<svg viewBox="0 0 {width} 360" class="plot">'
                f'<text x="0" y="12" class="unit">residual</text><g class="hair">{wiggle}'
                f'</g><g class="base">{Plot.path(x, baseline, box, xlim, ylim)}</g>'
                f'<g class="data">{Plot.path(x, data, box, xlim, ylim)}</g>'
                f'<g class="fitted">{Plot.path(x, fit, box, xlim, ylim)}</g>'
                f"{Plot.axis(xlim, 'ppm', width, 334)}</svg>"
                "<div class=note>Residual (data − fit) on top; data, <b class=fit>fit</b> and "
                "baseline on one scale.</div>")
        components = [k for k in curves if k not in self.CURVES]
        if not components:
            return main
        row, label = 30, 90
        ylim = Plot.ylim(x, [np.asarray(curves[k], dtype=np.float64) for k in components], xlim)
        rows = []
        for n, name in enumerate(components):
            box = (0, n * row + 4, width - label, row - 6)
            path = Plot.path(x, np.asarray(curves[name], dtype=np.float64), box, xlim, ylim)
            rows.append(f'<g class="line">{path}</g><text x="{width}" y="{n * row + row - 8}" '
                        f'text-anchor="end">{html.escape(name)}</text>')
        rows = "".join(rows)
        height = len(components) * row + 30
        return (main + "<h2>Components</h2>"
                f'<svg viewBox="0 0 {width} {height}" class="plot">{rows}'
                f"{Plot.axis(xlim, 'ppm', width - label, height - 26)}</svg>"
                "<div class=note>Each metabolite's part of the fit, on one scale.</div>")

    def table_pane(self) -> str:
        rows = "".join(f"<tr><td>{html.escape(str(r['metabolite']))}</td>"
                       f"<td class=n>{html.escape(str(r.get('concentration', '')))}</td>"
                       f"<td class=n>{html.escape(str(r.get('sd_percent', '')))}</td></tr>"
                       for r in self.result.rows)
        return ("<table><tr><td>metabolite</td><td class=n>concentration</td>"
                f"<td class=n>SD %</td></tr>{rows}</table>")

    def facts(self) -> str:
        r = self.result
        facts = [f"{len(r.rows)} metabolites"]
        if r.voxel is not None:
            facts.append(f"voxel {tuple(r.voxel)}")
        if self.others:
            facts.append(f"{self.others} more fit{'s' if self.others > 1 else ''} in the file")
        return " · ".join(facts + [str(r.provenance.format).replace("_", " ")])

    def html(self) -> str:
        tabs = [("Fit", self.fit_pane()), ("Table", self.table_pane()),
                ("Details", details(self.result.meta))]
        if not self.result.curves:
            tabs = tabs[1:]
        return document(self.title, self.facts(), tabs)


#**************************************************************************************************#
#                                            BasisPage                                             #
#**************************************************************************************************#
#                                                                                                  #
# The page of a basis set: one row per metabolite, real part as FSL-MRS shows a spectrum (nothing  #
# processed; of sub-spectra, such as edit conditions, the first), each on its own scale and        #
# labelled (Osprey's stagger); 1H over 6-0 ppm as spectra (FSL-MRS' plot_basis: 4.5-0, which       #
# leaves water out), other nuclei over the whole spectrum, at the set's reference; every header    #
# value.                                                                                           #
#                                                                                                  #
#**************************************************************************************************#
class BasisPage:
    def __init__(self, basis: BasisSet, title: str):
        self.basis, self.title = basis, title

    def basis_pane(self) -> str:
        b, width, row, label = self.basis, 900, 30, 110
        if b.dwelltime is None or b.spectrometer_frequency is None:
            return "<div class=empty>The file states no dwell time or frequency.</div>"
        proton = b.nucleus in (None, "1H")
        reference = float(b.meta.get("ReferencePPM", PROTON_REFERENCE if proton else 0.0))
        rows, more = [], 1
        for n, name in enumerate(b.metabolites):
            fid = np.asarray(b.fid(name), dtype=np.complex128)
            more = int(np.prod(fid.shape[1:]))  # sub-spectra (e.g. edit conditions)
            fid = fid.reshape(fid.shape[0], -1)[:, 0].conj().reshape(1, -1)
            spec = Spectral.fid_to_spec(Spectral.zero_fill(fid, 2))[0].real
            x = Spectral.ppm_axis(spec.size, 1.0 / b.dwelltime, b.spectrometer_frequency,
                                  reference)
            xlim = PROTON_RANGE if proton else (float(x.max()), float(x.min()))
            mask = (x >= min(xlim)) & (x <= max(xlim))
            own = (float(np.abs(spec[mask]).max()) if mask.any() else 0.0) or 1.0
            path = Plot.path(x[mask], spec[mask], (0, n * row + 4, width - label, row - 4),
                             xlim, (-0.2 * own, own))
            rows.append(f'<g class="line">{path}</g><text x="{width}" y="{n * row + row - 8}" '
                        f'text-anchor="end">{html.escape(name)}</text>')
        height = len(rows) * row + 30
        return (f'<svg viewBox="0 0 {width} {height}" class="plot">{"".join(rows)}'
                f"{Plot.axis(xlim, 'ppm', width - label, height - 26)}</svg>"
                "<div class=note>Every metabolite, real part, each on its own scale"
                + (f"; the first of {more} sub-spectra each" if more > 1 else "") + ".</div>")

    def facts(self) -> str:
        b = self.basis
        facts = [f"{len(b.metabolites)} metabolites", b.nucleus or ""]
        if b.spectrometer_frequency:
            facts.append(f"{b.spectrometer_frequency:.2f} MHz")
        facts += [f"{b.n_points} points", str(b.provenance.format).replace("_", " ")]
        return " · ".join(f for f in facts if f)

    def html(self) -> str:
        return document(self.title, self.facts(),
                        [("Basis", self.basis_pane()), ("Details", details(self.basis.meta))])


def details(meta) -> str:
    """Every header value as a table (long lists counted)."""
    rows = []
    for key, value in meta.items():
        if isinstance(value, (list, tuple)) and len(value) > 6:
            value = f"{len(value)} values"
        rows.append(f"<tr><td>{html.escape(str(key))}</td>"
                    f"<td>{html.escape(str(value))}</td></tr>")
    return f"<table>{''.join(rows)}</table>"


def document(title: str, facts: str, tabs: list[tuple[str, str]], switch: bool = False) -> str:
    """The page around the tabs (name, body): radio buttons and CSS, no script; with the switch,
    Full | Metabolites, hidden on the tabs it has no effect on."""
    radios = "".join(f'<input type=radio name=tab id=t{i}{" checked" if not i else ""}>'
                     for i in range(len(tabs)))
    labels = "".join(f"<label for=t{i}>{name}</label>" for i, (name, _) in enumerate(tabs))
    panes = "".join(f"<section id=p{i}>{body}</section>" for i, (_, body) in enumerate(tabs))
    show = ",".join(f"#t{i}:checked~main #p{i}" for i in range(len(tabs)))
    mark = ",".join(f"#t{i}:checked~nav label[for=t{i}]" for i in range(len(tabs)))
    rules = ""
    if switch:
        radios += "<input type=radio name=scale id=sf checked><input type=radio name=scale id=sm>"
        labels += ("<span class=seg><label for=sf>Full</label>"
                   "<label for=sm>Metabolites</label></span>")
        quiet = ",".join(f"#t{i}:checked~nav .seg" for i, (_, body) in enumerate(tabs)
                         if "class=met" not in body)
        rules = f"{quiet} {{ visibility:hidden; }}" if quiet else ""
    return PAGE.format(title=html.escape(title), facts=html.escape(facts), radios=radios,
                       labels=labels, panes=panes, show=show, mark=mark, rules=rules)


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="color-scheme" content="light dark"><style>
:root {{ --bg:#ffffff; --ink:#1d1d1f; --soft:#6e6e73; --faint:#aeaeb2; --rule:#e8e8ed;
         --accent:#0071e3; --pill:#f2f2f7; --drop:#d70015;
         --fit:#cc0000; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#1c1c1e; --ink:#f5f5f7; --soft:#a1a1a6; --faint:#636366; --rule:#38383a;
           --accent:#2997ff; --pill:#2c2c2e; --drop:#ff6961;
           --fit:#ff6961; }} }}
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
.plot .thin polyline {{ stroke:var(--soft); stroke-width:.6; opacity:.45; }}
.plot .drop polyline {{ stroke:var(--drop); stroke-width:.7; stroke-dasharray:3 2; }}
.plot .bold polyline {{ stroke:var(--accent); stroke-width:1.6; }}
.plot .data polyline {{ stroke:var(--ink); stroke-width:.8; }}
.plot .fitted polyline {{ stroke:var(--fit); stroke-width:1.4; }} b.fit {{ color:var(--fit); }}
.plot .base polyline {{ stroke:var(--soft); stroke-width:.8; }}
td.n {{ text-align:right; font-variant-numeric:tabular-nums; }}
.plot .hair polyline {{ stroke:var(--faint); stroke-width:.6; }}
.plot circle.dot {{ fill:var(--ink); }}
.plot circle.drop {{ fill:var(--bg); stroke:var(--drop); stroke-width:1.2; }}
.plot .label {{ fill:var(--ink); }}
h2 {{ font-size:12px; font-weight:600; margin:16px 0 4px; }}
section > h2:first-child {{ margin-top:0; }}
.seg {{ margin-left:auto; display:flex; background:var(--pill); border-radius:7px; padding:2px; }}
.seg label {{ padding:1px 9px; border-radius:5px; }}
#sf:checked~nav label[for=sf], #sm:checked~nav label[for=sm] {{ background:var(--bg);
  color:var(--ink); }}
.met {{ display:none; }} #sm:checked~main .met {{ display:block; }}
#sm:checked~main .full {{ display:none; }}
{rules}
.plot text {{ fill:var(--soft); font-size:11px; font-variant-numeric:tabular-nums; }}
.plot .unit {{ fill:var(--faint); }}
.note, .legend {{ color:var(--faint); font-size:11px; margin-top:8px; }}
.legend b {{ color:var(--accent); font-weight:500; }}
.empty {{ color:var(--soft); padding:40px 0; }}
.mrsi {{ display:grid; grid-template-columns:1fr 280px; gap:18px; align-items:start; }}
.cells {{ display:grid; border-left:1px solid var(--rule); border-top:1px solid var(--rule); }}
.cells svg {{ display:block; width:100%; height:auto; border-right:1px solid var(--rule);
              border-bottom:1px solid var(--rule); }}
.cells svg.focus {{ outline:1.5px solid var(--accent); outline-offset:-1.5px; }}
.cells polyline {{ fill:none; stroke:var(--ink); stroke-width:.8;
                   vector-effect:non-scaling-stroke; }}
.planes {{ display:grid; grid-template-columns:repeat(3, 1fr); gap:10px; align-items:end; }}
.slices {{ display:grid; grid-template-columns:repeat(6, 1fr); gap:3px; }}
figure {{ margin:0; position:relative; }}
figure img {{ width:100%; display:block; background:#000; }}
figcaption {{ display:flex; justify-content:space-between; font-size:11px; color:var(--soft);
              margin-top:4px; font-variant-numeric:tabular-nums; }}
.side {{ position:absolute; top:45%; font-size:10px; color:#8e8e93; }}
.side.left {{ left:5px; }} .side.right {{ right:5px; }}
table {{ border-collapse:collapse; width:100%; font-size:12px; }}
td {{ padding:4px 0; border-bottom:1px solid var(--rule); vertical-align:top; }}
td:first-child {{ color:var(--soft); width:38%; padding-right:16px; }}
</style></head><body>
<h1>{title}</h1><div class=facts>{facts}</div>
{radios}<nav>{labels}</nav><main>{panes}</main>
</body></html>"""


def load(path: Path) -> tuple[list, tuple[str, ...]]:
    """What the page shows of a file, with a note when that is not the file's own data: the
    k-space of a Bruker CSI scan is shown as the scanner's reconstruction of complex spectra."""
    from miox.readers import Loader

    if Sniffer().sniff(path).format == Format.BRUKER:
        from miox.readers.bruker import BrukerReader

        for reconstruction in BrukerReader().reconstructions(path):
            datasets = Loader().load(reconstruction)
            if any(isinstance(d, MRSData) for d in datasets):
                shown = reconstruction.relative_to(path.parent)
                return datasets, (f"{path.name} holds k-space: shown is the scanner's "
                                  f"reconstruction {shown}",)
    return Loader().load(path), ()


def page(datasets: list, title: str, source: Path | None = None,
         notes: tuple[str, ...] = ()) -> str:
    """The page of what the file holds: a basis set, spectroscopy data, a fit, else an image."""
    basis = next((d for d in datasets if isinstance(d, BasisSet)), None)
    if basis is not None:
        return BasisPage(basis, title).html()
    if any(isinstance(d, MRSData) for d in datasets):
        return Page(datasets, title, source, notes).html()
    fits = [d for d in datasets if isinstance(d, FitResult)]
    if fits:
        return FitPage(fits[0], title, len(fits) - 1).html()
    images = [d for d in datasets if isinstance(d, MRImage)]
    if not images:
        raise ValueError("no spectroscopy data or image")
    return ImagePage(images[0], title, len(images) - 1).html()


def write(datasets: list, title: str, path: str | Path, source: Path | None = None,
          notes: tuple[str, ...] = ()) -> None:
    Path(path).write_text(page(datasets, title, source, notes), encoding="utf-8")
