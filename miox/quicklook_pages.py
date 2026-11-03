####################################################################################################
#                                        quicklook_pages.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/11/26                                                                                #
#                                                                                                  #
# Purpose: The Quick Look pages of the kinds that are neither spectroscopy, image, basis set nor   #
# fit: raw k-space, pulse sequences, physiological recordings, tractograms, surfaces, processed    #
# NMR spectra (1-D and 2-D), raw 2-D NMR, RF pulses, transforms, phantoms, noise covariances and   #
# protocols. Each page draws what the file holds on the conventions of the reference viewer of    #
# its kind; the shared pieces (stacked rows, pictures, numbers) are the Panel class.               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import html
from pathlib import Path

import numpy as np

# own
from miox.data import (
    MRImage,
    MRSData,
    NoiseCovariance,
    Phantom,
    PhysioData,
    Protocol,
    PulseSequence,
    RawData,
    RFPulse,
    Surface,
    Tractogram,
    Transform,
)
from miox.processing import Spectral
from miox.quicklook import ImagePage, Plot, details, document
from miox.sniff import Format

WIDTH = 900  # drawing width of every plot


#**************************************************************************************************#
#                                              Panel                                               #
#**************************************************************************************************#
#                                                                                                  #
# Pieces the pages share: rows of traces on one time axis (each on its own scale, broken at NaN),  #
# a picture with its caption, grey pictures windowed by percentiles, short numbers, and the        #
# format name of a dataset.                                                                        #
#                                                                                                  #
#**************************************************************************************************#
class Panel:
    LABEL = 120  # width of the row labels
    ROW = 64  # height of a row

    @staticmethod
    def number(value: float) -> str:
        """A scale label: 1090, -0.25, 3.14, 2.4e+05."""
        if value == 0:
            return "0"
        if 1000 <= abs(value) < 1e5:
            return f"{value:.0f}"
        return f"{value:.3g}"

    @classmethod
    def rows(cls, rows: list[tuple], unit: str, xlim: tuple, extra: str = "") -> str:
        """``rows``: (label, t, y, css class), each row on its own vertical scale, the values
        at NaN breaking the line; one time axis below; ``extra`` SVG drawn over the rows."""
        parts, top, left = [], 8, cls.LABEL
        for label, t, y, kind in rows:
            box = (left, top + 6, WIDTH - left, cls.ROW - 12)
            finite = np.isfinite(y)
            if finite.any():
                low, high = float(np.nanmin(y)), float(np.nanmax(y))
                if low == high:
                    low, high = low - 1, high + 1
                ylim = (low - 0.05 * (high - low), high + 0.05 * (high - low))
                runs = np.flatnonzero(np.diff(np.concatenate([[0], finite.astype(int), [0]])))
                path = "".join(Plot.path(t[a:b], y[a:b], box, xlim, ylim)
                               for a, b in zip(runs[::2], runs[1::2], strict=True))
                parts.append(f'<g class="{kind}">{path}</g>')
                parts.append(f'<text x="{left - 10}" y="{top + 14}" class="left unit">'
                             f"{cls.number(high)}</text>"
                             f'<text x="{left - 10}" y="{top + cls.ROW - 10}" class="left unit">'
                             f"{cls.number(low)}</text>")
            parts.append(f'<text x="0" y="{top + 14}" class="label">{html.escape(label)}</text>'
                         f'<line class="axis" x1="{left}" x2="{WIDTH}" y1="{top + cls.ROW - 2}" '
                         f'y2="{top + cls.ROW - 2}"/>')
            top += cls.ROW
        axis = (f'<g transform="translate({left},0)">'
                f"{Plot.axis(xlim, unit, WIDTH - left, top + 2)}</g>")
        return (f'<svg viewBox="0 0 {WIDTH} {top + 30}" class="plot rows">'
                f'{"".join(parts)}{extra}{axis}</svg>')

    @classmethod
    def x(cls, value: float, xlim: tuple) -> float:
        """Drawing x of a time within the rows."""
        return cls.LABEL + (value - xlim[0]) / (xlim[1] - xlim[0]) * (WIDTH - cls.LABEL)

    @staticmethod
    def picture(src: str, caption: str = "", right: str = "", ratio: float | None = None,
                sides: tuple[str, str] | None = None, cls: str = "") -> str:
        style = f' style="aspect-ratio:{ratio:.4f}"' if ratio else ""
        marks = (f'<span class="side left">{sides[0]}</span>'
                 f'<span class="side right">{sides[1]}</span>' if sides else "")
        return (f'<figure><img class="{cls}" src="{src}"{style}>{marks}<figcaption>'
                f"<span>{html.escape(caption)}</span><span>{html.escape(right)}</span>"
                "</figcaption></figure>")

    @staticmethod
    def grey(values: np.ndarray, log: bool = False) -> str:
        """A grey picture windowed at 0.5-99.5 % of its finite values (of log(1 + v))."""
        v = np.log1p(values) if log else values
        finite = v[np.isfinite(v)]
        window = ((float(np.percentile(finite, 0.5)), float(np.percentile(finite, 99.5)))
                  if finite.size else (0.0, 1.0))
        return Plot.grey(v, window)

    @staticmethod
    def format(d) -> str:
        return str(d.provenance.format).replace("_", " ")


#**************************************************************************************************#
#                                             RawPage                                              #
#**************************************************************************************************#
#                                                                                                  #
# Raw k-space: log|k| summed over coils (root sum of squares) of one slice (the first average,     #
# slice, contrast, phase, repetition, set and segment, the best-filled partition), phase-encode    #
# lines as rows and samples as columns as fastMRI and ismrmrdviewer draw it, lines not acquired    #
# black; beside it, for formats whose plain reconstruction has been checked against a reference,  #
# the 2-D FFT of that slice (the echo centred by center_sample, the readout cropped to its matrix, #
# root sum of squares over coils, no parallel-imaging reconstruction), or the reconstruction the   #
# file itself holds (fastMRI, M4Raw); every coil's log|k|; six readouts; the physiological         #
# waveforms stored between the readouts; every header value.                                      #
#                                                                                                  #
#**************************************************************************************************#
class RawPage:
    VERIFIED = {  # formats whose plain reconstruction equals a reference (tests/evidence)
        Format.SIEMENS_TWIX: "checked against Gadgetron's reconstruction",
        Format.ISMRMRD: "checked against Gadgetron's reconstruction",
        Format.MRD: "checked against Gadgetron's reconstruction",
        Format.FASTMRI_H5: "equals the root-sum-of-squares image the file stores",
    }
    INDICES = ("average", "slice", "contrast", "phase", "repetition", "set", "segment")
    READOUTS = 6

    def __init__(self, raw: RawData, title: str, others: int = 0, image: MRImage | None = None):
        self.raw, self.title, self.others, self.image = raw, title, others, image
        self.k, self.acquired, self.chosen = self.select(raw)
        self.magnitude = np.abs(self.k)
        self.rss = np.sqrt((self.magnitude ** 2).sum(axis=1))

    @classmethod
    def select(cls, raw: RawData) -> tuple[np.ndarray, np.ndarray, dict]:
        """The k-space (lines, coils, samples) of one slice, which lines it holds, and which
        index of every counter with several values was taken."""
        idx = raw.headers["idx"]
        selected = np.ones(len(idx), dtype=bool)
        chosen = {}
        for name in cls.INDICES:
            first = int(idx[name][selected][0])
            selected &= idx[name] == first
            if len(np.unique(idx[name])) > 1:
                chosen[name] = (first, len(np.unique(idx[name])))
        partitions, counts = np.unique(idx["kspace_encode_step_2"][selected], return_counts=True)
        partition = int(partitions[np.argmax(counts)])
        if len(partitions) > 1:
            chosen["partition"] = (partition, len(partitions))
        selected &= idx["kspace_encode_step_2"] == partition
        lines = idx["kspace_encode_step_1"][selected]
        n = int(max(lines.max() + 1, cls.encoded_lines(raw)))
        k = np.zeros((n, raw.n_coils, raw.data.shape[-1]), dtype=np.complex64)
        k[lines] = raw.data[selected]
        acquired = np.zeros(n, dtype=bool)
        acquired[lines] = True
        return k, acquired, chosen

    @staticmethod
    def encoded_lines(raw: RawData) -> int:
        """Phase-encode lines of the encoded space: twix (dims/size) or ISMRMRD (limits)."""
        enc = raw.encoding
        if isinstance(enc, dict) and "dims" in enc:
            return int(dict(zip(enc["dims"], enc["size"], strict=True)).get("Lin", 0))
        if isinstance(enc, list) and enc and isinstance(enc[0], dict):
            limit = enc[0].get("encodingLimits", {}).get("kspace_encoding_step_1", {})
            return int(limit.get("maximum", -1)) + 1
        return 0

    @staticmethod
    def readout_matrix(raw: RawData) -> int:
        """The readout matrix without oversampling: ISMRMRD reconSpace.x, twix BaseResolution;
        0 when the file states none."""
        enc = raw.encoding
        if isinstance(enc, list) and enc and isinstance(enc[0], dict):
            x = enc[0].get("reconSpace", {}).get("matrixSize", {}).get("x")
            return int(x) if x else 0
        base = raw.meta.get("BaseResolution")
        return int(base) if isinstance(base, (int, float)) and base else 0

    @classmethod
    def plain_image(cls, raw: RawData) -> tuple[np.ndarray, str]:
        """The 2-D FFT of the selected slice (rows = phase-encode lines) with the echo centred
        by center_sample in a readout of twice the matrix and the image cropped to the matrix,
        root sum of squares over coils; and what was cropped."""
        k, acquired, _ = cls.select(raw)
        lines, coils, samples = k.shape
        base = cls.readout_matrix(raw)
        full = max(samples, 2 * base) if base and base < samples else samples
        centres = raw.headers["center_sample"]
        centre = int(np.bincount(centres[centres > 0]).argmax()) if centres.any() else samples // 2
        padded = np.zeros((lines, coils, full), dtype=np.complex64)
        start = max(0, min(full - samples, full // 2 - centre))
        padded[:, :, start:start + samples] = k
        image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(padded, axes=(0, 2)), axes=(0, 2)),
                                axes=(0, 2))
        rss = np.sqrt((np.abs(image) ** 2).sum(axis=1))
        if base and base < full:
            first = full // 2 - base // 2
            return rss[:, first:first + base], f"readout cropped to {base}"
        return rss, "readout as stored"

    def kspace_pane(self) -> str:
        lines, _, samples = self.k.shape
        ratio = samples / lines
        pictures = Panel.picture(Panel.grey(self.rss, log=True),
                                 "k-space, all coils (root sum of squares), log scale",
                                 f"{lines} lines × {samples} samples", ratio)
        verified = self.VERIFIED.get(self.raw.provenance.format)
        if self.image is not None:
            stored = ImagePage(self.image, "").volume
            k = stored.shape[2] // 2
            plane = stored[:, :, k].T[::-1]
            pictures += Panel.picture(Panel.grey(plane), f"{self.image.name.split('/')[-1]}, "
                                      "stored in the file", f"slice {k + 1} of {stored.shape[2]}",
                                      plane.shape[1] / plane.shape[0])
        elif verified:
            image, cropped = self.plain_image(self.raw)
            pictures += Panel.picture(Panel.grey(image), "plain 2-D FFT, root sum of squares "
                                      "over coils", cropped, image.shape[1] / image.shape[0])
        shown = ", ".join(f"{name} {value + 1} of {count}"
                          for name, (value, count) in self.chosen.items()) or "the only slice"
        missing = lines - int(self.acquired.sum())
        note = f"Shown: {shown}; {missing} of {lines} phase-encode lines not in the file"
        note += (" (parallel imaging or partial Fourier: a plain FFT aliases)." if missing
                 else ".")
        if self.image is not None:
            note += " The image is the reconstruction the file holds, as the image page shows it."
        elif verified:
            note += (f" The image is a plain FFT of what is stored ({verified}): no "
                     "parallel-imaging reconstruction, no coil combination beyond the root sum "
                     "of squares, no gridding.")
        else:
            note += " No image: this format's plain reconstruction has not been checked."
        return f"<div class=two>{pictures}</div><div class=note>{note}</div>"

    def coils_pane(self) -> str:
        lines, coils, samples = self.k.shape
        cells = "".join(Panel.picture(Panel.grey(self.magnitude[:, c], log=True), f"coil {c + 1}",
                                      "", samples / lines) for c in range(coils))
        return (f"<div class=coils>{cells}</div><div class=note>log |k| per coil, each in its "
                "own window.</div>")

    def readouts_pane(self) -> str:
        lines = np.flatnonzero(self.acquired)
        picks = lines[np.unique(np.linspace(0, len(lines) - 1, self.READOUTS).astype(int))]
        samples = np.arange(self.k.shape[-1], dtype=np.float64)
        rows = [(f"line {int(line) + 1}", samples, self.rss[line], "") for line in picks]
        return (Panel.rows(rows, "sample", (0.0, float(samples[-1])))
                + "<div class=note>|readout| summed over coils, six lines spread over the "
                "acquisition, each on its own scale.</div>")

    def waveforms_pane(self) -> str:
        """Every waveform name as one row (its first channel), on the scanner's clock from
        the first acquisition packet; learning-phase packets left out."""
        series = {}
        for w in self.raw.waveforms:
            if str(w.get("phase", "")).lower() == "learning":
                continue
            series.setdefault(w["name"], []).append(w)
        rows, end = [], 0.0
        for name, packets in series.items():
            packets.sort(key=lambda w: w["time_stamp"])
            t0 = packets[0]["time_stamp"] * 2.5e-3
            ts, values = [], []
            for w in packets:
                data = np.asarray(w["data"])
                first = data[0] if data.ndim > 1 else data
                start = w["time_stamp"] * 2.5e-3 - t0
                ts.append(start + np.arange(first.size) * w["sample_time_us"] * 1e-6)
                values.append(first.astype(np.float64))
            t, y = np.concatenate(ts), np.concatenate(values)
            rows.append((name, t, y, ""))
            end = max(end, float(t[-1]))
        return (Panel.rows(rows, "s", (0.0, end or 1.0))
                + "<div class=note>Physiological waveforms recorded between the readouts (first "
                "channel of each), on the scanner's clock from the first packet; learning-phase "
                "packets left out.</div>")

    def facts(self) -> str:
        raw, h = self.raw, self.raw.headers
        facts = [f"{raw.n_readouts} readouts × {raw.n_coils} coils × {raw.data.shape[-1]} samples",
                 self.encoding_facts(raw.encoding)]
        dwell = np.unique(h["sample_time_us"][h["sample_time_us"] > 0])
        if dwell.size == 1:
            facts.append(f"dwell {float(dwell[0]):g} µs")
        if raw.waveforms:
            facts.append(f"{len(raw.waveforms)} waveform packets")
        if self.others:
            facts.append(f"{self.others} more dataset{'s' if self.others > 1 else ''} in the file")
        return " · ".join(f for f in facts + [Panel.format(raw)] if f)

    @staticmethod
    def encoding_facts(enc) -> str:
        if isinstance(enc, dict) and "dims" in enc:
            return " × ".join(f"{d} {s}" for d, s in zip(enc["dims"], enc["size"], strict=True)
                              if s > 1)
        if isinstance(enc, list) and enc and isinstance(enc[0], dict):
            e = enc[0]
            m = e.get("encodedSpace", {}).get("matrixSize", {})
            r = e.get("reconSpace", {}).get("matrixSize", {})
            limits = ", ".join(f"{k.replace('kspace_encoding_step_', 'step ')} {v['maximum'] + 1}"
                               for k, v in e.get("encodingLimits", {}).items()
                               if v.get("maximum", 0) > 0)
            return (f"{e.get('trajectory', '')} encoded {m.get('x')} × {m.get('y')} × "
                    f"{m.get('z')}, recon {r.get('x')} × {r.get('y')} × {r.get('z')}; {limits}")
        return ""

    def html(self) -> str:
        tabs = [("k-space", self.kspace_pane()), ("Coils", self.coils_pane()),
                ("Readouts", self.readouts_pane())]
        if self.raw.waveforms:
            tabs.append(("Waveforms", self.waveforms_pane()))
        tabs.append(("Details", details(self.raw.meta)))
        return document(self.title, self.facts(), tabs)


#**************************************************************************************************#
#                                           SequencePage                                           #
#**************************************************************************************************#
#                                                                                                  #
# A Pulseq sequence as its own plot draws it: RF magnitude (Hz), RF phase (rad) with the ADC       #
# windows shaded, and the three gradients (kHz/m) against time, for the whole sequence and for    #
# its start (up to past the first readout); every definition. Sequences of other formats (TOPPE)  #
# get their values only.                                                                           #
#                                                                                                  #
#**************************************************************************************************#
class SequencePage:
    GRADIENTS = ("gx", "gy", "gz")

    def __init__(self, seq: PulseSequence, title: str):
        self.seq, self.title = seq, title
        self.timed = seq.provenance.format == Format.PULSEQ and "rf" in seq.blocks.dtype.names
        if self.timed:
            self.traces, self.adcs, self.total = self.expand(seq)

    @classmethod
    def expand(cls, seq: PulseSequence) -> tuple[dict, list, float]:
        """Polylines (t, y) of the RF magnitude, RF phase (NaN between pulses) and the three
        gradients, the ADC windows (start, end), and the total duration, all in seconds."""
        starts = seq.starts
        rf_t, rf_a, rf_p = [0.0], [0.0], [np.nan]
        grads = {g: ([0.0], [0.0]) for g in cls.GRADIENTS}
        adcs = []
        for block, start in zip(seq.blocks, starts[:-1], strict=True):
            if block["rf"]:
                t, magnitude, phase, end = seq.rf(int(block["rf"]))
                rf_t += [start + t[0], *(start + t), start + end]
                rf_a += [0.0, *magnitude, 0.0]
                rf_p += [np.nan, *(phase % (2 * np.pi)), np.nan]
            for g in cls.GRADIENTS:
                if block[g]:
                    t, amplitude, end = seq.gradient(int(block[g]))
                    grads[g][0].extend([start + t[0], *(start + t), start + end])
                    grads[g][1].extend([0.0, *amplitude, 0.0])
            if block["adc"]:
                on, off = seq.adc(int(block["adc"]))
                adcs.append((start + on, start + off))
        total = float(starts[-1])
        rf_t.append(total)
        rf_a.append(0.0)
        rf_p.append(np.nan)
        traces = {"rf": (np.array(rf_t), np.array(rf_a)), "phase": (np.array(rf_t), np.array(rf_p))}
        for g, (t, y) in grads.items():
            traces[g] = (np.array(t + [total]), np.array(y + [0.0]) / 1e3)
        return traces, adcs, total

    def pane(self, xlim: tuple) -> str:
        unit, scale = ("ms", 1e3) if xlim[1] - xlim[0] < 2 else ("s", 1.0)
        rows = []
        labels = {"rf": "RF  Hz", "phase": "RF phase  rad", "gx": "Gx  kHz/m", "gy": "Gy  kHz/m",
                  "gz": "Gz  kHz/m"}
        for key, label in labels.items():
            t, y = self.traces[key]
            inside = np.flatnonzero((t >= xlim[0]) & (t <= xlim[1]))
            if inside.size:
                a, b = max(inside[0] - 1, 0), min(inside[-1] + 2, t.size)
                rows.append((label, t[a:b] * scale, y[a:b], ""))
            else:
                rows.append((label, t[:2] * scale, np.zeros(2), ""))
        drawn = (xlim[0] * scale, xlim[1] * scale)
        windows = [(max(a, xlim[0]), min(b, xlim[1])) for a, b in self.adcs
                   if b >= xlim[0] and a <= xlim[1]][:3000]
        top = 8 + Panel.ROW + 6
        rects = ""
        for a, b in windows:
            left, right = Panel.x(a * scale, drawn), Panel.x(b * scale, drawn)
            rects += (f'<rect x="{left:.1f}" width="{max(0.8, right - left):.1f}" y="{top}" '
                      f'height="{Panel.ROW - 12}"/>')
        return Panel.rows(rows, unit, drawn, extra=f'<g class="adc">{rects}</g>')

    def facts(self) -> str:
        seq = self.seq
        facts = [f"v{seq.version}", f"{seq.n_blocks} blocks"]
        if self.timed:
            facts.append(f"{self.total:.3g} s")
        facts.append(", ".join(f"{len(v)} {k}" for k, v in seq.events.items()))
        return " · ".join(facts + [Panel.format(seq)])

    def html(self) -> str:
        if not self.timed:
            return document(self.title, self.facts(), [("Details", details(self.seq.meta))])
        first_adc = self.adcs[0][1] if self.adcs else None
        span = min(self.total, max(first_adc * 1.3 if first_adc else self.total / 50,
                                   self.total / 200, 1e-3))
        tabs = [("Sequence", self.pane((0.0, self.total))
                 + "<div class=note>Whole sequence: RF magnitude and phase, ADC windows shaded "
                 "in the phase row, the three gradients in kHz/m, as Pulseq's plot.</div>"),
                ("Start", self.pane((0.0, span))
                 + f"<div class=note>The first {span * 1e3:.3g} ms, up to and past the first "
                 "readout.</div>"),
                ("Details", details(self.seq.meta))]
        return document(self.title, self.facts(), tabs)


#**************************************************************************************************#
#                                            PhysioPage                                            #
#**************************************************************************************************#
#                                                                                                  #
# A physiological recording as TAPAS PhysIO draws one: every channel against seconds, each on its  #
# own scale, the events (triggers) as marks below the rows; the first 30 s on their own when the   #
# recording is longer than a minute; every header value.                                          #
#                                                                                                  #
#**************************************************************************************************#
class PhysioPage:
    ZOOM = 30.0  # seconds of the second tab
    LONG = 60.0  # recordings longer than this get it

    def __init__(self, physio: PhysioData, title: str):
        self.physio, self.title = physio, title
        self.rate = physio.sampling_frequency or 1.0
        self.t = np.arange(physio.n_samples) / self.rate
        self.marks = [(name, np.asarray(ev)[np.asarray(ev) < physio.n_samples] / self.rate)
                      for name, ev in physio.events.items()
                      if np.asarray(ev).dtype.kind in "iu" and len(ev) > 1]

    def pane(self, xlim: tuple) -> str:
        p = self.physio
        mask = (self.t >= xlim[0]) & (self.t <= xlim[1])
        rows = [(name, self.t[mask], p.data[mask, i].astype(np.float64), "")
                for i, name in enumerate(p.channels)]
        top = 8 + len(rows) * Panel.ROW
        marks = ""
        for _, times in self.marks:
            inside = times[(times >= xlim[0]) & (times <= xlim[1])]
            if inside.size > 400:
                inside = inside[::int(np.ceil(inside.size / 400))]
            marks += "".join(f'<line x1="{Panel.x(v, xlim):.1f}" x2="{Panel.x(v, xlim):.1f}" '
                             f'y1="{top - 2}" y2="{top + 10}"/>' for v in inside)
        unit = "s" if p.sampling_frequency else "sample"
        return Panel.rows(rows, unit, xlim, extra=f'<g class="stem">{marks}</g>')

    def legend(self) -> str:
        marks = ", ".join(f"{name} ({times.size})" for name, times in self.marks)
        return ("<div class=note>Every channel on its own scale"
                + (f"; {marks} as marks below the rows." if marks else ".") + "</div>")

    def facts(self) -> str:
        p = self.physio
        facts = [f"{p.n_samples} samples", f"{p.sampling_frequency:g} Hz" if p.sampling_frequency
                 else "no sampling rate", f"{float(self.t[-1]):.1f} s" if p.sampling_frequency
                 else "", f"{len(p.channels)} channels"]
        if p.start_time is not None:
            facts.append(f"start {p.start_time:g} {p.time_reference}")
        return " · ".join(f for f in facts + [Panel.format(p)] if f)

    def html(self) -> str:
        whole = (0.0, float(self.t[-1]) or 1.0)
        tabs = [("Recording", self.pane(whole) + self.legend())]
        if whole[1] > self.LONG:
            tabs.append((f"First {self.ZOOM:g} s", self.pane((0.0, self.ZOOM)) + self.legend()))
        tabs.append(("Details", details(self.physio.meta)))
        return document(self.title, self.facts(), tabs)


#**************************************************************************************************#
#                                          TractogramPage                                          #
#**************************************************************************************************#
#                                                                                                  #
# A tractogram projected on the axial, coronal and sagittal planes, every streamline (20000 at    #
# most, evenly picked), coloured by its local direction as mrview and DSI Studio colour one (red   #
# left-right, green front-back, blue foot-head), brighter where more streamlines pass, the left    #
# on the left; every header value.                                                                #
#                                                                                                  #
#**************************************************************************************************#
class TractogramPage:
    MOST = 20000  # streamlines drawn
    SIZE = 380  # pixels of the longest extent

    def __init__(self, tract: Tractogram, title: str):
        self.tract, self.title = tract, title

    def segments(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Start, direction and RGB colour of every segment of the drawn streamlines."""
        points, offsets = np.asarray(self.tract.points, dtype=np.float64), self.tract.offsets
        n = self.tract.n_streamlines
        keep = np.arange(n) if n <= self.MOST else np.linspace(0, n - 1, self.MOST).astype(int)
        first = [np.arange(offsets[i], offsets[i + 1] - 1) for i in keep
                 if offsets[i + 1] - offsets[i] > 1]
        a = np.concatenate(first) if first else np.zeros(0, dtype=int)
        start, direction = points[a], points[a + 1] - points[a]
        length = np.linalg.norm(direction, axis=1)
        colour = np.abs(direction) / np.maximum(length, 1e-9)[:, None]
        return start, direction, colour

    def view(self, across: int, up: int) -> str:
        start, direction, colour = self.segments()
        points = np.asarray(self.tract.points, dtype=np.float64)
        low, size = points.min(axis=0), np.ptp(points, axis=0)
        scale = self.SIZE / max(float(size.max()), 1e-9)
        w, h = int(np.ceil(size[across] * scale)) + 2, int(np.ceil(size[up] * scale)) + 2
        length = np.linalg.norm(direction, axis=1)
        steps = np.maximum(1, np.ceil(length * scale).astype(int))
        which = np.repeat(np.arange(len(start)), steps)
        along = np.arange(which.size) - np.repeat(np.cumsum(steps) - steps, steps)
        fraction = along / np.repeat(steps, steps)
        q = start[which] + direction[which] * fraction[:, None]
        x = ((q[:, across] - low[across]) * scale).astype(int)
        y = h - 1 - ((q[:, up] - low[up]) * scale).astype(int)
        image, count = np.zeros((h, w, 3)), np.zeros((h, w))
        np.add.at(image, (y, x), colour[which])
        np.add.at(count, (y, x), 1)
        mean = image / np.maximum(count, 1)[..., None]
        busy = float(np.percentile(count[count > 0], 75)) if (count > 0).any() else 1.0
        bright = np.sqrt(np.minimum(1.0, count / max(busy, 1.0)))
        return Plot.png(np.round(mean * bright[..., None] * 255))

    def views_pane(self) -> str:
        views = (("axial", 0, 1, ("L", "R"), "from above"), ("coronal", 0, 2, ("L", "R"),
                 "from the front"), ("sagittal", 1, 2, ("P", "A"), "from the left"))
        figures = "".join(Panel.picture(self.view(a, u), name, how, sides=sides)
                          for name, a, u, sides, how in views)
        n = self.tract.n_streamlines
        drawn = "" if n <= self.MOST else f" ({self.MOST} of {n})"
        return (f"<div class=views>{figures}</div><div class=note>Every streamline{drawn} "
                "projected, coloured by local direction (red left-right, green front-back, blue "
                "foot-head), brighter where more pass; left on the left.</div>")

    def facts(self) -> str:
        t = self.tract
        facts = [f"{t.n_streamlines} streamlines", f"{len(t.points)} points", t.space]
        if t.point_data:
            facts.append("per point: " + ", ".join(k.strip() for k in t.point_data))
        if t.streamline_data:
            facts.append("per streamline: " + ", ".join(t.streamline_data))
        return " · ".join(facts + [Panel.format(t)])

    def html(self) -> str:
        if self.tract.points is None or not len(self.tract.points):
            tabs = [("Details", details(self.tract.meta))]
        else:
            tabs = [("Views", self.views_pane()), ("Details", details(self.tract.meta))]
        return document(self.title, self.facts(), tabs)


#**************************************************************************************************#
#                                           SurfacePage                                            #
#**************************************************************************************************#
#                                                                                                  #
# A surface mesh from the side (lateral), the inside (medial) and above, lit from the viewer, as   #
# FreeSurfer and nilearn show one; shaded by the curvature file beside it (binarised: sulci dark)  #
# when there is one. A file of values only (curv, sulc, thickness) is drawn on the inflated,       #
# white or pial surface beside it. Faces are rasterised through a depth buffer, the shade          #
# interpolated between their vertices (Gouraud). Details lists every header value.                 #
#                                                                                                  #
#**************************************************************************************************#
class SurfacePage:
    SIZE = 460  # pixels a view
    MESHES = ("inflated", "white", "pial")  # where values are shown, in order of preference

    def __init__(self, surface: Surface, title: str, source: Path | None = None):
        self.surface, self.title, self.notes = surface, title, []
        self.mesh, self.overlay = surface, None
        hemisphere = self.hemisphere(source)
        if surface.vertices is None:
            mesh = self.companion(source, hemisphere, self.MESHES)
            if mesh is not None and surface.vertex_data:
                self.mesh = mesh
                self.overlay = next(iter(surface.vertex_data.values()))
                self.notes.append(f"{surface.name} holds values only: shown on {mesh.name} "
                                  "beside it, binarised like FreeSurfer (sulci dark).")
        else:
            curv = self.companion(source, hemisphere, ("curv",))
            if curv is not None and curv.vertex_data:
                self.overlay = next(iter(curv.vertex_data.values()))
                self.notes.append(f"Shaded by {curv.name} beside it, binarised like FreeSurfer "
                                  "(sulci dark).")
        self.left = self.mesh.name.startswith("lh") or (
            self.mesh.vertices is not None and float(np.mean(self.mesh.vertices[:, 0])) < 0)

    @staticmethod
    def hemisphere(source: Path | None) -> str | None:
        """FreeSurfer's lh / rh prefix of the file, or None."""
        if source is None:
            return None
        prefix = source.name.split(".")[0]
        return prefix if prefix in ("lh", "rh") else None

    @staticmethod
    def companion(source: Path | None, hemisphere: str | None, names: tuple) -> Surface | None:
        """The first of ``names`` (lh.curv, lh.inflated, ...) beside the file, loaded."""
        if source is None or hemisphere is None:
            return None
        from miox.readers import Loader

        for name in names:
            path = source.parent / f"{hemisphere}.{name}"
            if path.is_file() and path != source:
                try:
                    found = [d for d in Loader().load(path) if isinstance(d, Surface)]
                except Exception:  # a companion that does not read is simply not used
                    return None
                if found:
                    return found[0]
        return None

    def render(self, view: tuple) -> np.ndarray:
        """Grey levels (NaN = background) of the mesh seen along ``view`` = ((sign, axis) of
        the screen's x, of its y, and of its depth), lit from the viewer with a little from
        above. Every face is rasterised at the pixel centres it covers, its depth and shade
        interpolated from the vertices; the nearest face wins the pixel."""
        v, f = np.asarray(self.mesh.vertices, dtype=np.float64), np.asarray(self.mesh.faces)
        (sx, ax), (sy, ay), (sz, az) = view
        screen_x, screen_y, depth = sx * v[:, ax], sy * v[:, ay], sz * v[:, az]
        corners = v[f]
        face_normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        normals = np.zeros_like(v)
        for corner in range(3):
            np.add.at(normals, f[:, corner], face_normals)
        normals /= np.maximum(np.linalg.norm(normals, axis=1), 1e-12)[:, None]
        light = np.clip(0.75 * np.maximum(sz * normals[:, az], 0)
                        + 0.25 * np.maximum(sy * normals[:, ay], 0) + 0.15, 0, 1)
        base = np.full(len(v), 0.72) if self.overlay is None else np.where(self.overlay > 0,
                                                                            0.42, 0.72)
        shade = light * base
        low = np.array([screen_x.min(), screen_y.min()])
        span = max(screen_x.max() - low[0], screen_y.max() - low[1]) or 1.0
        scale = (self.SIZE - 4) / span
        px = (screen_x - low[0]) * scale + 2
        py = (self.SIZE - 2) - (screen_y - low[1]) * scale
        x, y = px[f], py[f]
        area = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (y[:, 1] - y[:, 0]) * (x[:, 2] - x[:, 0])
        keep = area != 0  # faces seen edge-on cover no pixel
        x, y, area, z, s = x[keep], y[keep], area[keep], depth[f][keep], shade[f][keep]
        x0, y0 = np.floor(x.min(axis=1)).astype(int), np.floor(y.min(axis=1)).astype(int)
        width = np.ceil(x.max(axis=1)).astype(int) - x0 + 1
        height = np.ceil(y.max(axis=1)).astype(int) - y0 + 1
        columns, rows, depths, shades = [], [], [], []
        for dy in range(min(int(height.max()), self.SIZE)):
            for dx in range(min(int(width.max()), self.SIZE)):
                sel = np.nonzero((dx < width) & (dy < height))[0]
                cx, cy = x0[sel] + dx + 0.5, y0[sel] + dy + 0.5
                xs, ys = x[sel], y[sel]
                w0 = (xs[:, 1] - cx) * (ys[:, 2] - cy) - (ys[:, 1] - cy) * (xs[:, 2] - cx)
                w1 = (xs[:, 2] - cx) * (ys[:, 0] - cy) - (ys[:, 2] - cy) * (xs[:, 0] - cx)
                weights = np.stack([w0, w1, area[sel] - w0 - w1], axis=1) / area[sel][:, None]
                inside = np.all(weights >= -1e-9, axis=1)
                w, i = weights[inside], sel[inside]
                columns.append(cx[inside].astype(int))
                rows.append(cy[inside].astype(int))
                depths.append(np.sum(w * z[i], axis=1))
                shades.append(np.sum(w * s[i], axis=1))
        columns, rows, depths, shades = map(np.concatenate, (columns, rows, depths, shades))
        order = np.argsort(depths)  # far first: the near face wins the pixel
        columns, rows, shades = columns[order], rows[order], shades[order]
        image = np.full((self.SIZE, self.SIZE), np.nan)
        on = (rows >= 0) & (rows < self.SIZE) & (columns >= 0) & (columns < self.SIZE)
        image[rows[on], columns[on]] = shades[on]
        return image

    def views_pane(self) -> str:
        outward = -1 if self.left else 1  # the lateral side of a left hemisphere faces -x
        views = (("lateral", ((outward, 1), (1, 2), (outward, 0))),
                 ("medial", ((-outward, 1), (1, 2), (-outward, 0))),
                 ("from above", ((1, 0), (1, 1), (1, 2))))
        figures = []
        for name, view in views:
            image = self.render(view)
            picture = Plot.grey(np.where(np.isnan(image), 1.0, image), (0.0, 1.0))
            figures.append(Panel.picture(picture, name, cls="mesh"))
        note = " ".join(self.notes + ["Lit from the viewer; nothing smoothed."])
        return f"<div class=views>{''.join(figures)}</div><div class=note>{note}</div>"

    def facts(self) -> str:
        s, mesh = self.surface, self.mesh
        facts = []
        if mesh.vertices is not None:
            facts += [f"{len(mesh.vertices)} vertices", f"{len(mesh.faces)} faces"]
        facts += [mesh.space, "left hemisphere" if self.left else "right hemisphere"]
        if s.vertex_data:
            facts.append("values: " + ", ".join(s.vertex_data))
        if s.labels:
            facts.append("labels: " + ", ".join(s.labels))
        return " · ".join(f for f in facts + [Panel.format(s)] if f)

    def html(self) -> str:
        tabs = []
        if self.mesh.vertices is not None and self.mesh.faces is not None:
            tabs.append(("Views", self.views_pane()))
        tabs.append(("Details", details(self.surface.meta)))
        return document(self.title, self.facts(), tabs)


#**************************************************************************************************#
#                                           SpectrumPage                                           #
#**************************************************************************************************#
#                                                                                                  #
# A processed spectrum stored as such (TopSpin 1r/2rr, Sparky, NMRPipe ft): the first point is    #
# the highest shift, as those programs store it. 1-D: the real part as stored on its ppm axis,    #
# falling to the right. 2-D: contours as Sparky and TopSpin draw them, 20 levels from six times   #
# the noise (robust SD) rising by 1.2, negative levels red, both ppm axes falling to the right     #
# and the top. Every header value.                                                                 #
#                                                                                                  #
#**************************************************************************************************#
class SpectrumPage:
    LEVELS, FIRST, STEP = 20, 6.0, 1.2
    HEIGHT = 520

    def __init__(self, image: MRImage, title: str, others: int = 0):
        self.image, self.title, self.others = image, title, others
        self.data = np.asarray(image.data)[0, 0, 0]  # (direct, indirect, ...)
        self.two_d = "frequency_indirect_0" in image.dims

    def axes(self) -> tuple[np.ndarray, np.ndarray | None]:
        m, n = self.image.meta, self.data.shape[0]
        f0 = m["SpectrometerFrequency"]
        f0 = list(f0) if isinstance(f0, (list, tuple)) else [f0]
        direct = Spectral.ppm_axis(n, 1.0 / m["DwellTime"], f0[0], m.get("ReferencePPM", 0.0))
        indirect = None
        if self.two_d:
            width = m.get("IndirectSpectralWidth", [None])[0]
            reference = m.get("IndirectReferencePPM", [0.0])[0]
            if width and len(f0) > 1:
                indirect = Spectral.ppm_axis(self.data.shape[1], width, f0[1], reference)[::-1]
            else:
                indirect = np.arange(self.data.shape[1], dtype=np.float64)
        return direct[::-1], indirect  # stored downfield first

    def one_d_pane(self) -> str:
        x, _ = self.axes()
        y = np.real(self.data).reshape(x.size, -1)[:, 0]
        xlim = (float(x.max()), float(x.min()))
        return (Plot.spectrum([(x, y, "line accent")], xlim, "ppm")
                + "<div class=note>The spectrum as stored: real part, nothing processed; the "
                "shift falls to the right.</div>")

    def two_d_pane(self) -> str:
        import contourpy

        x, y = self.axes()
        z = np.real(self.data).reshape(x.size, y.size).T  # (indirect rows, direct columns)
        noise = 1.4826 * float(np.median(np.abs(z - np.median(z)))) or 1.0
        levels = self.FIRST * noise * self.STEP ** np.arange(self.LEVELS)
        generator = contourpy.contour_generator(x=x, y=y, z=z)
        xlim, ylim = (float(x.max()), float(x.min())), (float(y.max()), float(y.min()))
        height = self.HEIGHT

        def paths(values, kind):
            lines = []
            for level in values:
                for line in generator.lines(level):
                    px = (xlim[0] - line[:, 0]) / (xlim[0] - xlim[1]) * WIDTH
                    py = (line[:, 1] - ylim[1]) / (ylim[0] - ylim[1]) * height
                    lines.append('<polyline points="' + " ".join(
                        f"{a:.1f},{b:.1f}" for a, b in zip(px, py, strict=True)) + '"/>')
            return f'<g class="{kind}">{"".join(lines)}</g>'

        span = ylim[0] - ylim[1]
        step = next(s for s in (0.5, 1, 2, 5, 10, 20, 50, 100) if span / s <= 8)
        ticks = "".join(f'<text x="{WIDTH}" y="{(t - ylim[1]) / span * height + 4:.1f}" '
                        f'text-anchor="end">{t:g}</text>'
                        for t in np.arange(np.ceil(ylim[1] / step) * step, ylim[0] + 1e-9, step))
        nuclei = self.image.meta.get("ResonantNucleus", ["", ""])
        nuclei = list(nuclei) if isinstance(nuclei, (list, tuple)) else [nuclei, ""]
        return (f'<svg viewBox="0 0 {WIDTH} {height + 30}" class="plot">'
                f'{paths(levels, "data")}{paths(-levels, "neg")}{ticks}'
                f'{Plot.axis(xlim, "ppm", WIDTH, height + 4)}</svg>'
                f"<div class=note>{self.LEVELS} contour levels from {self.FIRST:g} × noise "
                f"(robust SD {noise:.3g}) rising by {self.STEP}, negative in red; "
                f"{nuclei[0]} across, {nuclei[1] if len(nuclei) > 1 else ''} up, both ppm axes "
                "falling to the right and the top as Sparky and TopSpin draw them.</div>")

    def facts(self) -> str:
        m = self.image.meta
        shape = " × ".join(str(n) for n in self.data.shape) + " points"
        nuclei = m.get("ResonantNucleus", "")
        nuclei = " / ".join(nuclei) if isinstance(nuclei, (list, tuple)) else str(nuclei)
        f0 = m.get("SpectrometerFrequency")
        f0 = f0[0] if isinstance(f0, (list, tuple)) else f0
        facts = [shape, nuclei, f"{f0:.2f} MHz" if f0 else ""]
        if self.others:
            facts.append(f"{self.others} more spectr{'a' if self.others > 1 else 'um'} in the file")
        return " · ".join(f for f in facts + [Panel.format(self.image)] if f)

    def html(self) -> str:
        pane = self.two_d_pane() if self.two_d else self.one_d_pane()
        return document(self.title, self.facts(),
                        [("Spectrum", pane), ("Details", details(self.image.meta))])


#**************************************************************************************************#
#                                             RowsPage                                             #
#**************************************************************************************************#
#                                                                                                  #
# A raw multi-dimensional NMR acquisition (a TopSpin ser): every increment of the first indirect   #
# dimension as a row, its FID Fourier transformed (magnitude, the stored conjugate as the          #
# spectroscopy page takes it), the shift falling to the right; the indirect dimension as acquired, #
# since its quadrature recombination (States, TPPI, echo-antiecho) is not done. Every header       #
# value.                                                                                           #
#                                                                                                  #
#**************************************************************************************************#
class RowsPage:
    RATIO = 1.6  # width over height of the raster

    def __init__(self, d: MRSData, title: str):
        self.d, self.title = d, title

    def rows_pane(self) -> str:
        d = self.d
        data = np.asarray(d.data)[0, 0, 0]  # (t, increments, ...)
        fids = np.conj(data.reshape(data.shape[0], -1).T)  # (rows, t), FSL-MRS's orientation
        spectra = np.abs(Spectral.fid_to_spec(fids))
        x = Spectral.ppm_axis(spectra.shape[1], 1.0 / d.dwelltime, d.spectrometer_frequency,
                              float(d.meta.get("ReferencePPM", 0.0)))
        raster = Panel.grey(spectra[:, ::-1])  # high shift at the left
        xlim = (float(x.max()), float(x.min()))
        return (Panel.picture(raster, "every increment: magnitude spectrum of its FID, high "
                              "shift left", f"{spectra.shape[0]} rows", self.RATIO)
                + f'<svg viewBox="0 0 {WIDTH} 30" class="plot">{Plot.axis(xlim, "ppm", WIDTH, 4)}'
                "</svg><div class=note>The direct dimension Fourier transformed row by row "
                "(magnitude, nothing else); the indirect dimension as acquired, one row per "
                "increment, top to bottom. No quadrature recombination: this is the file, not "
                "the 2-D spectrum.</div>")

    def facts(self) -> str:
        d = self.d
        points = " × ".join(str(n) for n in d.shape[3:5]) + " points × increments"
        nuclei = d.meta.get("ResonantNucleus", d.nucleus or "")
        nuclei = " / ".join(nuclei) if isinstance(nuclei, (list, tuple)) else str(nuclei)
        return " · ".join(f for f in [points, nuclei, f"{d.spectrometer_frequency:.2f} MHz",
                                      Panel.format(d)] if f)

    def html(self) -> str:
        return document(self.title, self.facts(),
                        [("Rows", self.rows_pane()), ("Details", details(self.d.meta))])


#**************************************************************************************************#
#                                            Minor pages                                           #
#**************************************************************************************************#
#                                                                                                  #
# RF pulse: amplitude and phase per sample. Transform: the matrix as stored and in RAS. Phantom:   #
# every property on the phantom's grid. Noise covariance: its magnitude and the correlation.       #
# Protocol: the values of every protocol in the file, one tab each.                                #
#                                                                                                  #
#**************************************************************************************************#
class RFPulsePage:
    def __init__(self, pulse: RFPulse, title: str):
        self.pulse, self.title = pulse, title

    def html(self) -> str:
        p = self.pulse
        n = p.n_samples
        if p.duration:
            t, unit = np.arange(n) * (p.duration / n * 1e6), "µs"
        else:
            t, unit = np.arange(n, dtype=np.float64), "sample"
        rows = [(f"amplitude  {p.amplitude_unit}", t, np.asarray(p.amplitude, dtype=np.float64),
                 ""), ("phase  °", t, np.asarray(p.phase, dtype=np.float64), "")]
        pane = (Panel.rows(rows, unit, (0.0, float(t[-1]) or 1.0))
                + "<div class=note>The pulse as stored: amplitude and phase per sample"
                + ("." if p.duration else "; the file states no duration.") + "</div>")
        facts = [f"{n} samples", f"{p.duration * 1e6:g} µs" if p.duration else "no duration",
                 Panel.format(p)]
        return document(self.title, " · ".join(facts),
                        [("Pulse", pane), ("Details", details(p.meta))])


class TransformPage:
    def __init__(self, transform: Transform, title: str, others: int = 0):
        self.transform, self.title, self.others = transform, title, others

    @staticmethod
    def table(matrix) -> str:
        rows = "".join("<tr>" + "".join(f"<td class=n>{v:.6g}</td>" for v in row) + "</tr>"
                       for row in np.asarray(matrix, dtype=np.float64))
        return f"<table class=matrix>{rows}</table>"

    def html(self) -> str:
        t = self.transform
        pane = f"<h2>{html.escape(t.kind)} · {html.escape(t.convention)}</h2>"
        if t.matrix is not None:
            pane += self.table(t.matrix)
        if t.ras_matrix is not None:
            pane += "<h2>In RAS (mm)</h2>" + self.table(t.ras_matrix)
        if t.field is not None:
            pane += f"<div class=note>Displacement field {np.asarray(t.field).shape}.</div>"
        pane += ("<div class=note>The matrix as stored in the file's own convention, and the "
                 "same mapping in RAS.</div>")
        facts = [t.kind, t.convention]
        if self.others:
            plural = "s" if self.others > 1 else ""
            facts.append(f"{self.others} more transform{plural} in the file")
        return document(self.title, " · ".join(facts + [Panel.format(t)]),
                        [("Matrix", pane), ("Details", details(t.meta))])


class PhantomPage:
    def __init__(self, phantom: Phantom, title: str):
        self.phantom, self.title = phantom, title

    def maps_pane(self) -> str:
        ph = self.phantom
        grid = ph.grid or {}
        if grid.get("indices") is None:
            return ("<div class=empty>The phantom states no grid: its spins are listed in "
                    "Details.</div>")
        shape, index = grid["shape"], np.asarray(grid["indices"])
        slices = np.bincount(index[:, 2])
        k = int(np.argmax(slices))
        inside = index[:, 2] == k
        figures = []
        for name, values in ph.properties.items():
            picture = np.full((shape[1], shape[0]), np.nan)
            picture[index[inside, 1], index[inside, 0]] = np.asarray(values)[inside]
            finite = picture[np.isfinite(picture)]
            window = ((float(np.percentile(finite, 0.5)), float(np.percentile(finite, 99.5)))
                      if finite.size else (0.0, 1.0))
            filled = np.where(np.isnan(picture), window[0], picture)[::-1]
            unit = ph.units.get(name, "")
            figures.append(Panel.picture(Plot.grey(filled, window), f"{name} {unit}".strip(),
                                         f"{window[0]:.3g} – {window[1]:.3g}",
                                         shape[0] / shape[1]))
        return (f"<div class=maps>{''.join(figures)}</div><div class=note>Each property on the "
                f"phantom's grid (slice {k + 1} of {shape[2]}), windowed 0.5–99.5 %.</div>")

    def html(self) -> str:
        ph = self.phantom
        facts = [f"{ph.n_spins} spins", ", ".join(ph.properties), Panel.format(ph)]
        return document(self.title, " · ".join(f for f in facts if f),
                        [("Maps", self.maps_pane()), ("Details", details(ph.meta))])


class NoisePage:
    def __init__(self, noise: NoiseCovariance, title: str):
        self.noise, self.title = noise, title

    def html(self) -> str:
        nc = self.noise
        magnitude = np.abs(np.asarray(nc.matrix))
        sigma = np.sqrt(np.abs(np.real(np.diag(np.asarray(nc.matrix)))))
        with np.errstate(divide="ignore", invalid="ignore"):
            correlation = np.nan_to_num(magnitude / np.outer(sigma, sigma))
        n = magnitude.shape[0]
        pane = ("<div class=two>"
                + Panel.picture(Plot.grey(magnitude, (0.0, float(magnitude.max()) or 1.0)),
                                "|covariance|", f"{n} coils", 1.0)
                + Panel.picture(Plot.grey(correlation, (0.0, 1.0)), "correlation", "diagonal 1",
                                1.0)
                + "</div><div class=note>Coil noise covariance as stored, and the correlation "
                "derived from it.</div>")
        facts = [f"{n} coils", f"{nc.sample_count} samples" if nc.sample_count else "",
                 Panel.format(nc)]
        return document(self.title, " · ".join(f for f in facts if f),
                        [("Covariance", pane), ("Details", details(nc.meta))])


class ProtocolPage:
    MOST = 8  # protocols given a tab

    def __init__(self, protocols: list[Protocol], title: str):
        self.protocols, self.title = protocols, title

    def html(self) -> str:
        n = len(self.protocols)
        facts = [f"{n} protocol{'s' if n > 1 else ''} in the file", Panel.format(self.protocols[0])]
        tabs = [(html.escape(p.name)[:28] or f"protocol {i + 1}", details(p.meta))
                for i, p in enumerate(self.protocols[:self.MOST])]
        return document(self.title, " · ".join(facts), tabs)
