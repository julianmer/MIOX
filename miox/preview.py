####################################################################################################
#                                            preview.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 12/10/26                                                                                #
#                                                                                                  #
# Purpose: One still picture of a spectroscopy file, as a quick look shows it: every MRS dataset   #
#          of the file reduced to the spectra a reader expects (coils combined, averages aligned   #
#          and averaged, edited subspectra and their difference), phased and labelled.             #
#                                                                                                  #
####################################################################################################


#*************#
# imports                                                                                          #
#*************#
from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

# own
from miox.data import BasisSet, MRSData
from miox.sniff import Format

PPM_SHIFT = {"1H": 4.65, "2H": 4.65}  # NIfTI-MRS default ppm at the spectrometer frequency
METABOLITES_1H = (1.8, 3.4)  # NAA, creatine, choline: where a suppressed 1H spectrum is phased
WATER_1H = (4.3, 5.1)
DISPLAY_1H = (0.2, 4.3)  # the usual view of a water-suppressed 1H spectrum
WATER_DOMINANT = 200.0  # water peak this many times the metabolite peak: a water spectrum
WATER_BESIDE = 20.0  # from this ratio on, the full spectrum with its water is shown beside
WATER_REMOVED = (4.4, 5.0)  # ppm of the components HLSVD removes from a 1H metabolite spectrum
NAA_1H = (1.9, 2.1)  # lowered by a 1.9 ppm editing pulse: tells edit-on from edit-off
HLSVD_POINTS = 1024  # first points of the FID modelled
HLSVD_RANK = 24  # damped sinusoids modelled
LINE_BROADENING = 2.0  # Hz of exponential apodisation for display
SHIFTS = np.arange(-20.0, 20.01, 0.25)  # Hz searched when aligning averages
NMR_FORMATS = {Format.TOPSPIN, Format.TOPSPIN_PROCESSED, Format.JCAMP_NMR, Format.NMRML,
               Format.JEOL_JDF, Format.TECMAG_TNT, Format.NMRPIPE, Format.SPARKY, Format.SIMPSON,
               Format.SPINSOLVE, Format.RS2D, Format.RNMRTK}
COLOURS = ("#1f4e79", "#c0392b", "#2e7d32", "#6a1b9a")


#**************************************************************************************************#
#                                              Reduced                                             #
#**************************************************************************************************#
#                                                                                                  #
# What is left of one dataset for display: labelled one-dimensional FIDs (one, or edited           #
# subspectra and their difference) and the steps that led there.                                   #
#                                                                                                  #
#**************************************************************************************************#
@dataclass
class Reduced:
    fids: dict[str, np.ndarray]
    steps: list[str] = field(default_factory=list)


#**************************************************************************************************#
#                                             Reducer                                              #
#**************************************************************************************************#
#                                                                                                  #
# Takes an MRSData of any shape to one FID per spectrum to show. Spatial dimensions: the voxel     #
# with the strongest signal. Coils (DIM_COIL): weighted by their first points over their noise     #
# variance (complex weights, so the phases line up). Averages (DIM_DYN, DIM_MEAS): each aligned    #
# to their mean in frequency (within +-20 Hz) and zero-order phase by maximising the magnitude of  #
# their inner product with the mean over the first 0.2 s, then averaged. DIM_EDIT of size two:     #
# both subspectra and their difference. Indirect NMR dimensions: the first increment. Any other    #
# dimension is summed.                                                                             #
#                                                                                                  #
#**************************************************************************************************#
class Reducer:
    AVERAGES = ("DIM_DYN", "DIM_MEAS")
    FIRST = ("DIM_INDIRECT_0", "DIM_INDIRECT_1", "DIM_INDIRECT_2")

    def __init__(self, d: MRSData):
        self.d = d

    def reduce(self) -> Reduced:
        data = np.asarray(self.d.data)
        steps: list[str] = []
        data = self._voxel(data, steps)  # now (t, dims 5-7 ...)
        tags = [str(t) for t in self.d.dim_tags[: data.ndim - 1]]
        if "DIM_COIL" in tags:
            data = self._combine(data, 1 + tags.index("DIM_COIL"), steps)
            tags.remove("DIM_COIL")
        for tag in list(tags):
            axis = 1 + tags.index(tag)
            if tag in self.FIRST:
                data = data.take(0, axis=axis)
                steps.append(f"first of {self.d.shape[4 + self.d.dim_tags.index(tag)]} "
                             "increments")
                tags.remove(tag)
            elif tag in self.AVERAGES:
                data = self._average(data, axis, steps)
                tags.remove(tag)
        if "DIM_EDIT" in tags and data.shape[1 + tags.index("DIM_EDIT")] == 2:
            axis = 1 + tags.index("DIM_EDIT")
            rest = tuple(a for a in range(1, data.ndim) if a != axis)
            edit = np.moveaxis(data, axis, -1).sum(axis=tuple(a - (a > axis) for a in rest))
            steps.append("edited: both subspectra and their difference")
            return Reduced({"edit 1": edit[:, 0], "edit 2": edit[:, 1],
                            "difference": edit[:, 0] - edit[:, 1]}, steps)
        if data.ndim > 1:
            steps.append(f"{', '.join(tags)} summed")
            data = data.reshape(data.shape[0], -1).sum(axis=1)
        return Reduced({"spectrum": data}, steps)

    def _voxel(self, data: np.ndarray, steps: list[str]) -> np.ndarray:
        spatial = data.shape[:3]
        if np.prod(spatial) == 1:
            return data[0, 0, 0]
        head = np.abs(data[:, :, :, : min(16, data.shape[3])]).reshape(*spatial, -1).sum(axis=-1)
        index = np.unravel_index(int(np.argmax(head)), spatial)
        steps.append(f"voxel {tuple(int(i) for i in index)} of {' x '.join(map(str, spatial))} "
                     "(strongest signal)")
        return data[index]

    def _combine(self, data: np.ndarray, axis: int, steps: list[str]) -> np.ndarray:
        coils = np.moveaxis(data, axis, 1)  # (t, coils, ...)
        n = coils.shape[0]
        mean = coils.reshape(n, coils.shape[1], -1).mean(axis=2)
        noise = np.var(mean[-max(n // 4, 1):], axis=0) + 1e-30
        weights = np.conj(mean[: max(n // 64, 1)].mean(axis=0)) / noise
        weights /= np.abs(weights).sum() or 1.0
        combined = np.tensordot(weights, coils, axes=([0], [1]))
        if coils.shape[1] > 1:
            steps.append(f"{coils.shape[1]} coils combined (signal / noise weighted)")
        return combined

    def _average(self, data: np.ndarray, axis: int, steps: list[str]) -> np.ndarray:
        count = data.shape[axis]
        if count == 1:
            return data.take(0, axis=axis)
        moved = np.moveaxis(data, axis, 1)  # (t, averages, ...)
        if self.d.dwelltime is None:
            steps.append(f"{count} averages averaged")
            return moved.mean(axis=1)
        flat = moved.reshape(moved.shape[0], count, -1)
        t = np.arange(flat.shape[0]) * self.d.dwelltime
        window = t < 0.2
        shifts = np.exp(-2j * np.pi * SHIFTS[:, None] * t[None, window])  # (shifts, t)
        aligned = np.empty_like(flat)
        for k in range(flat.shape[2]):
            reference = flat[:, :, k].mean(axis=1)
            products = shifts @ (np.conj(reference[window, None]) * flat[window, :, k])
            best = np.argmax(np.abs(products), axis=0)  # (averages,)
            phase = np.angle(products[best, np.arange(count)])
            correction = np.exp(-2j * np.pi * SHIFTS[best][None, :] * t[:, None] - 1j * phase)
            aligned[:, :, k] = flat[:, :, k] * correction
        steps.append(f"{count} averages aligned and averaged")
        return aligned.mean(axis=1).reshape(moved.shape[:1] + moved.shape[2:])


def hlsvd_water(fid: np.ndarray, dwell: float, keep, rank: int = HLSVD_RANK) -> np.ndarray:
    """The FID's components whose frequency (Hz) ``keep`` accepts, modelled by HLSVD: the
    signal poles from the SVD of the Hankel matrix of the first points, the amplitudes by least
    squares over the same points; returned over the whole FID."""
    n = min(len(fid), HLSVD_POINTS)
    rows = n // 2
    hankel = np.lib.stride_tricks.sliding_window_view(fid[:n], n - rows + 1)
    u = np.linalg.svd(hankel, full_matrices=False)[0][:, :rank]
    poles = np.linalg.eigvals(np.linalg.lstsq(u[:-1], u[1:], rcond=None)[0])
    poles = poles[np.abs(poles) < 1.0]  # decaying signals only
    if not len(poles):
        return np.zeros_like(fid)
    basis = poles[None, :] ** np.arange(len(fid))[:, None]
    amplitudes = np.linalg.lstsq(basis[:n], fid[:n], rcond=None)[0]
    selected = keep(np.angle(poles) / (2 * np.pi * dwell))
    return basis[:, selected] @ amplitudes[selected]


#**************************************************************************************************#
#                                            Spectrum                                              #
#**************************************************************************************************#
#                                                                                                  #
# One FID as a displayed spectrum (NMR samples: the whole range, nothing removed; the rules below  #
# are for in-vivo 1H): 2 Hz exponential line broadening, zero filling to at least 4096 points, the #
# Fourier transform in the NIfTI-MRS convention (conjugated, ppm rising to the left), the axis     #
# from ReferencePPM (else the nucleus default) and the spectrometer frequency. A 1H spectrum is a  #
# water spectrum when it is a reference (role "water") or water exceeds 200 times the tallest of   #
# NAA, creatine and choline; otherwise its residual water (4.4-5.0 ppm) is removed by HLSVD for    #
# display and 0.2-4.3 ppm is shown. A zero-order phase makes the tallest peak real and positive    #
# (the metabolites in a 1H metabolite spectrum), unless a phase is given; when the phased real     #
# part still dips below -40 % of its maximum, the magnitude is shown instead (not for signed       #
# spectra such as edited differences).                                                             #
#                                                                                                  #
#**************************************************************************************************#
class Spectrum:
    def __init__(self, d: MRSData, fid: np.ndarray, role: str = "auto",
                 phase: float | None = None, signed: bool = False):
        self.d = d
        self.notes: list[str] = []
        fid = np.asarray(fid, dtype=np.complex128)
        size = max(4096, 2 * len(fid))
        self.x, self.unit = self._axis(size)
        nmr = d.provenance.format in NMR_FORMATS or "Solvent" in d.meta
        proton = self.unit == "ppm" and d.nucleus == "1H" and not nmr
        self.ratio = self._ratio(fid, size) if proton else 0.0
        self.water = role == "water" or self.ratio > WATER_DOMINANT
        if proton and not self.water and self._region(*WATER_REMOVED).any():
            fid = fid - self._water(fid)
            self.notes.append("residual water removed for display (HLSVD)")
        self.values = self._transform(fid, size)
        self.range = (float(self.x.min()), float(self.x.max()))
        if proton:
            shown = (max(self.x.min(), 0.0), min(self.x.max(), 9.0)) if self.water else DISPLAY_1H
            if self._region(*shown).sum() > 0.1 * len(self.x):
                self.range = shown
        self.proton = proton
        self.part = "real"
        self.phase = self._phase() if phase is None else phase
        self.values = self.values * np.exp(-1j * self.phase)
        shown = self.values[self._region(*self.range)].real
        if not signed and len(shown) and shown.min() < -0.4 * shown.max():
            self.part = "magnitude"
            self.notes.append("magnitude (zero-order phasing left negative peaks)")
        elif phase is None:
            self.notes.append("zero-order phase on the tallest peak")

    def _transform(self, fid: np.ndarray, size: int) -> np.ndarray:
        if self.d.dwelltime is not None:
            fid = fid * np.exp(-np.pi * LINE_BROADENING * np.arange(len(fid)) * self.d.dwelltime)
        return np.fft.fftshift(np.fft.fft(fid, size)).conj()

    def _axis(self, size: int) -> tuple[np.ndarray, str]:
        if self.d.dwelltime is None:
            return np.arange(size, dtype=float), "point"
        hz = -np.fft.fftshift(np.fft.fftfreq(size, self.d.dwelltime))
        if self.d.spectrometer_frequency is None:
            return hz, "Hz"
        return self.reference + hz / self.d.spectrometer_frequency, "ppm"

    @property
    def reference(self) -> float:
        if "ReferencePPM" in self.d.meta:
            return float(self.d.meta["ReferencePPM"])
        return PPM_SHIFT.get(self.d.nucleus, 0.0)

    def _region(self, low: float, high: float) -> np.ndarray:
        return (self.x >= low) & (self.x <= high)

    def _ratio(self, fid: np.ndarray, size: int) -> float:
        """Tallest water point over the tallest metabolite point."""
        values = np.abs(self._transform(fid, size))
        water, metabolites = self._region(*WATER_1H), self._region(*METABOLITES_1H)
        if not water.any() or not metabolites.any():
            return 0.0
        return float(values[water].max() / (values[metabolites].max() or 1e-30))

    def _water(self, fid: np.ndarray) -> np.ndarray:
        """The FID's water components: in the NIfTI-MRS convention a component of frequency f
        (Hz) appears at ReferencePPM - f / SpectrometerFrequency."""
        frequency = self.d.spectrometer_frequency
        low, high = WATER_REMOVED

        def keep(hz: np.ndarray) -> np.ndarray:
            ppm = self.reference - hz / frequency
            return (ppm >= low) & (ppm <= high)
        try:
            return hlsvd_water(fid, self.d.dwelltime, keep)
        except np.linalg.LinAlgError:
            return np.zeros_like(fid)

    def _phase(self) -> float:
        region = (self._region(*METABOLITES_1H) if self.proton and not self.water
                  else self._region(*self.range))
        if not region.any():
            region = np.ones_like(self.x, dtype=bool)
        peak = np.flatnonzero(region)[np.argmax(np.abs(self.values[region]))]
        return float(np.angle(self.values[peak]))

    def height(self, low: float, high: float) -> float:
        region = self._region(low, high)
        return float(np.abs(self.values[region]).max()) if region.any() else 0.0

    def shown(self) -> tuple[np.ndarray, np.ndarray]:
        mask = self._region(*self.range)
        values = np.abs(self.values) if self.part == "magnitude" else self.values.real
        return self.x[mask], values[mask]


#**************************************************************************************************#
#                                             Preview                                              #
#**************************************************************************************************#
#                                                                                                  #
# The picture of a file: a title with the file name, a line of acquisition facts, one panel per    #
# MRS dataset (up to four; water references, named *_ref or water dominated, beside the metabolite #
# spectrum), edited data as subspectra over their difference, basis sets as stacked metabolite     #
# spectra, and a footer naming what was done to the data.                                          #
#                                                                                                  #
#**************************************************************************************************#
class Preview:
    SIZE = (12.0, 6.75)
    PANELS = 4
    BASIS_LINES = 24

    def __init__(self, datasets: list, title: str = "", figure: Figure | None = None):
        self.datasets = [d for d in datasets if isinstance(d, MRSData)]
        if not self.datasets:
            raise ValueError("no spectroscopy data to preview")
        self.title = title or self.datasets[0].name
        self.figure = figure if figure is not None else Figure(figsize=self.SIZE)
        self.figure.set_facecolor("white")
        self.steps: list[str] = []
        self._draw()

    def _draw(self) -> None:
        shown = self.datasets[: self.PANELS]
        if isinstance(shown[0], BasisSet):
            axes = [self.figure.add_axes((0.06, 0.12, 0.9, 0.72))]
            self._basis(axes[0], shown[0])
        else:
            main = [d for d in shown if not d.name.lower().endswith("_ref")] or shown
            others = [d for d in shown if d not in main[:1]]
            reduced = Reducer(main[0]).reduce()
            beside = None
            if not others and len(reduced.fids) == 1:
                probe = Spectrum(main[0], reduced.fids["spectrum"])
                if probe.ratio > WATER_BESIDE and not probe.water:
                    beside = probe.ratio
            width = 0.62 if others or beside else 0.9
            axes = [self.figure.add_axes((0.06, 0.12, width, 0.72))]
            self._panel(axes[0], main[0], reduced=reduced)
            if beside:
                ax = self.figure.add_axes((0.73, 0.40, 0.24, 0.40))
                full = Spectrum(main[0], reduced.fids["spectrum"], role="water")
                self._style(ax, full, small=True)
                ax.set_title(f"with its water ({beside:.0f} x the metabolites)", fontsize=8,
                             loc="left")
            for i, d in enumerate(others):
                height = 0.72 / len(others)
                ax = self.figure.add_axes((0.73, 0.12 + i * height + 0.06, 0.24, height - 0.08))
                self._panel(ax, d, small=True)
        self.figure.text(0.06, 0.95, self.title, fontsize=15, weight="bold", va="top")
        self.figure.text(0.06, 0.895, self._facts(shown[0]), fontsize=10, color="#444444",
                         va="top")
        footer = "\n".join(textwrap.wrap("; ".join(dict.fromkeys(self.steps)), 150))
        self.figure.text(0.06, 0.012, footer, fontsize=7.5, color="#777777", va="bottom")
        self.figure.text(0.97, 0.95, "MIOX", fontsize=9, color="#999999", ha="right", va="top")

    def _panel(self, ax, d: MRSData, small: bool = False, reduced: Reduced | None = None) -> None:
        reduced = reduced or Reducer(d).reduce()
        self.steps += reduced.steps
        role = "water" if d.name.lower().endswith("_ref") else "auto"
        if "difference" in reduced.fids:
            spectra = self._edited(d, reduced.fids, role)
        else:
            spectra = {"": Spectrum(d, reduced.fids["spectrum"], role)}
        top = 0.0
        for i, (label, spectrum) in enumerate(spectra.items()):
            x, y = spectrum.shown()
            offset = top - (y.min() if i else 0.0)
            ax.plot(x, y + offset, lw=0.6 if small else 0.9, color=COLOURS[i % len(COLOURS)],
                    label=label or None)
            top = offset + y.max() + 0.15 * (np.ptp(y) or 1.0)
            self.steps += spectrum.notes
        self._style(ax, spectrum, small, plot=False)
        if small or len(self.datasets) > 1:
            ax.set_title(d.name, fontsize=8 if small else 10, loc="left")
        if len(spectra) > 1:
            ax.legend(frameon=False, fontsize=8, loc="upper left", reverse=True)


    @staticmethod
    def _style(ax, spectrum: Spectrum, small: bool, plot: bool = True) -> None:
        """Axis range, label and frame of a spectrum panel (and the spectrum, when ``plot``)."""
        x, y = spectrum.shown()
        if plot:
            ax.plot(x, y, lw=0.6, color="#777777")
        ax.set_xlim(x.max(), x.min())
        ax.set_xlabel(spectrum.unit if spectrum.unit != "ppm" else "chemical shift (ppm)",
                      fontsize=8 if small else 10)
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=7 if small else 9)

    def _edited(self, d: MRSData, fids: dict, role: str) -> dict[str, Spectrum]:
        """Edit-off, edit-on and their difference, drawn bottom to top as difference, on, off;
        all with the phase of the edit-off spectrum. In 1H data the subspectrum with the taller
        NAA peak is taken as edit-off (a 1.9 ppm editing pulse lowers NAA)."""
        first, second = Spectrum(d, fids["edit 1"], role), Spectrum(d, fids["edit 2"], role)
        off_first = first.unit != "ppm" or first.height(*NAA_1H) >= second.height(*NAA_1H)
        off, on = (("edit 1", "edit 2") if off_first else ("edit 2", "edit 1"))
        phase = (first if off_first else second).phase
        self.steps.append("edit-off = the subspectrum with the taller NAA peak")
        return {"difference (on - off)": Spectrum(d, fids[on] - fids[off], role, phase, True),
                f"edit on ({on})": Spectrum(d, fids[on], role, phase),
                f"edit off ({off})": Spectrum(d, fids[off], role, phase)}

    def _basis(self, ax, d: BasisSet) -> None:
        fids = np.asarray(d.data[0, 0, 0]).reshape(d.n_points, -1)  # (t, metabolites, ...)
        total = d.shape[4] if d.data.ndim > 4 else 1
        count = min(total, self.BASIS_LINES)
        offset = 0.0
        for m in range(count):
            spectrum = Spectrum(d, fids[:, m])
            x, y = spectrum.shown()
            y = y / (np.abs(y).max() or 1.0)
            ax.plot(x, y + offset, lw=0.7, color=COLOURS[m % len(COLOURS)])
            name = d.metabolites[m] if m < len(d.metabolites) else str(m)
            ax.text(x.min(), offset + 0.2, f" {name}", fontsize=7, va="bottom")
            offset += 1.2
        ax.set_xlim(x.max(), x.min())
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("chemical shift (ppm)" if spectrum.unit == "ppm" else spectrum.unit)
        self.steps.append(f"basis set: {count} of {total} spectra, each scaled to its peak")

    @staticmethod
    def _facts(d: MRSData) -> str:
        meta, facts = d.meta, []
        if d.nucleus:
            facts.append(d.nucleus)
        if d.spectrometer_frequency:
            facts.append(f"{d.spectrometer_frequency:.2f} MHz")
        if "EchoTime" in meta:
            facts.append(f"TE {float(meta['EchoTime']) * 1e3:g} ms")
        if "RepetitionTime" in meta:
            facts.append(f"TR {float(meta['RepetitionTime']) * 1e3:g} ms")
        for key in ("SequenceName", "ProtocolName"):
            if key in meta and str(meta[key]).strip():
                facts.append(str(meta[key]).strip())
                break
        sizes = [f"{tag.removeprefix('DIM_').lower()} {n}" for tag, n in
                 zip(d.dim_tags, d.shape[4:], strict=False) if tag and n > 1]
        facts += sizes
        facts.append(f"{d.n_points} points")
        if d.provenance.format:
            facts.append(str(d.provenance.format))
        return "  ·  ".join(facts)


def preview(path_or_datasets, save: str | Path | None = None, dpi: int = 100,
            title: str = "", figure: Figure | None = None) -> Figure:
    """The quick-look picture of a spectroscopy file (or of loaded datasets); saved when
    ``save`` names a file (PNG, PDF, SVG by its suffix)."""
    if isinstance(path_or_datasets, (str, Path)):
        from miox.readers.loader import load

        path = Path(path_or_datasets)
        shown = Preview(load(path), title or path.name, figure)
    else:
        shown = Preview(list(path_or_datasets), title, figure)
    if save is not None:
        shown.figure.savefig(save, dpi=dpi)
    return shown.figure
