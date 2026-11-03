####################################################################################################
#                                            viewer.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 11/10/26                                                                                #
#                                                                                                  #
# Purpose: Quick look at what a file holds, as matplotlib figures: spectra (real, imaginary or     #
#          magnitude, on a ppm or Hz axis, a slider per further dimension), images (three planes   #
#          in RAS+ orientation, a slider per slice and further dimension), raw k-space readouts,   #
#          physiological recordings, fit results, pulse sequence diagrams and noise covariances.   #
#          Figures render in a window, inline in notebooks, or headless to a file.                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.widgets import RadioButtons, Slider
from mrs_tools.constants import PPM_SHIFT
from nibabel.orientations import apply_orientation, inv_ornt_aff, io_orientation

# own
from miox.data import (
    BasisSet,
    FitResult,
    MRImage,
    MRSData,
    NoiseCovariance,
    PhysioData,
    PulseSequence,
    RawData,
)


#**************************************************************************************************#
#                                           SpectrumView                                           #
#**************************************************************************************************#
#                                                                                                  #
# One spectroscopy dataset (or basis set) as a spectrum. The spectrum is the Fourier transform of  #
# the FID in the NIfTI-MRS convention (conjugated, as FSL-MRS shows it), on the axis ReferencePPM  #
# - f / SpectrometerFrequency; without a stated reference the nucleus' NIfTI-MRS default is used   #
# (4.65 ppm for 1H). Without a frequency the axis is in Hz, without a dwell time in points.        #
#                                                                                                  #
#**************************************************************************************************#
class SpectrumView:
    PARTS = ("magnitude", "real", "imaginary")
    UNITS = ("ppm", "Hz")
    SIZE = (10, 6)

    def __init__(self, d: MRSData, figure: Figure | None = None):
        self.d = d
        self.figure = figure if figure is not None else Figure(figsize=self.SIZE)
        self.part, self.unit = self.PARTS[0], self.UNITS[0]
        if d.spectrometer_frequency is None or d.dwelltime is None:
            self.unit = "Hz"
        self.index = [0] * d.data.ndim
        self.sliders = {}
        axes = [a for a in range(d.data.ndim) if a != 3 and d.shape[a] > 1]

        bottom = 0.08 + 0.05 * len(axes)
        self.ax = self.figure.add_axes((0.08, bottom + 0.06, 0.72, 0.84 - bottom))
        [self.line] = self.ax.plot([], [], lw=0.8)
        self.ax.invert_xaxis()
        part_ax = self.figure.add_axes((0.83, 0.70, 0.15, 0.18))
        self.parts = RadioButtons(part_ax, self.PARTS)
        self.parts.on_clicked(self.set_part)
        if self.unit == "ppm":
            unit_ax = self.figure.add_axes((0.83, 0.50, 0.15, 0.12))
            self.units = RadioButtons(unit_ax, self.UNITS)
            self.units.on_clicked(self.set_unit)
        for n, a in enumerate(axes):
            slider_ax = self.figure.add_axes((0.18, 0.04 + 0.05 * n, 0.55, 0.03))
            slider = Slider(slider_ax, d.dims[a], 0, d.shape[a] - 1, valinit=0, valstep=1,
                            valfmt="%d")
            slider.on_changed(lambda value, a=a: self.set_index(a, int(value)))
            self.sliders[a] = slider
        self.update()

    @property
    def reference(self) -> float:
        """ppm at 0 Hz: the file's ReferencePPM, else the nucleus' NIfTI-MRS default."""
        if "ReferencePPM" in self.d.meta:
            return float(self.d.meta["ReferencePPM"])
        return PPM_SHIFT.get(self.d.nucleus, 0.0)

    def axis(self) -> np.ndarray:
        """Positions of the spectrum's points in the current unit."""
        n = self.d.n_points
        if self.d.dwelltime is None:
            return np.arange(n)
        offset = -np.fft.fftshift(np.fft.fftfreq(n, self.d.dwelltime))  # Hz, rising with ppm
        if self.unit == "Hz":
            return offset
        return self.reference + offset / self.d.spectrometer_frequency

    def spectrum(self) -> np.ndarray:
        """Complex spectrum of the FID at the current index."""
        index = tuple(slice(None) if a == 3 else i for a, i in enumerate(self.index))
        return np.fft.fftshift(np.fft.fft(self.d.data[index])).conj()

    def set_part(self, part: str) -> None:
        self.part = part
        self.update()

    def set_unit(self, unit: str) -> None:
        self.unit = unit
        self.update()

    def set_index(self, axis: int, value: int) -> None:
        self.index[axis] = value
        self.update()

    def update(self) -> None:
        spectrum = self.spectrum()
        values = {"magnitude": np.abs, "real": np.real, "imaginary": np.imag}[self.part](spectrum)
        x = self.axis()
        self.line.set_data(x, values)
        self.ax.set_xlim(x.max(), x.min())
        self.ax.relim()
        self.ax.autoscale_view(scalex=False)
        if self.d.dwelltime is None:
            self.ax.set_xlabel("point")
        elif self.unit == "Hz":
            self.ax.set_xlabel("Hz")
        else:
            self.ax.set_xlabel(f"ppm ({self.reference:g} ppm at the spectrometer frequency)")
        self.ax.set_title(self.title(), fontsize=10)
        self.figure.canvas.draw_idle()

    def title(self) -> str:
        title = f"{self.d.name}: {self.d.acquisition()}"
        if isinstance(self.d, BasisSet) and len(self.d.metabolites) == self.d.shape[4]:
            title += f"\n{self.d.metabolites[self.index[4]]}"
        return title


#**************************************************************************************************#
#                                            ImageView                                             #
#**************************************************************************************************#
#                                                                                                  #
# One image as three planes (sagittal, coronal, axial) after turning the voxel axes to the closest #
# RAS+ orientation, so that patient left is on the left (neurological view) and every plane has    #
# its voxel aspect. Planes with a single row or column are left out. Complex images show their     #
# magnitude; the grey scale spans the 0.5th to 99.5th percentile of the volume shown.              #
#                                                                                                  #
#**************************************************************************************************#
class ImageView:
    PLANES = (  # name, the axis fixed, the letters (left, right, bottom, top)
        ("sagittal", 0, ("P", "A", "I", "S")),
        ("coronal", 1, ("L", "R", "I", "S")),
        ("axial", 2, ("L", "R", "P", "A")),
    )
    SIZE = (12, 5)

    def __init__(self, d: MRImage, figure: Figure | None = None):
        self.d = d
        self.figure = figure if figure is not None else Figure(figsize=self.SIZE)
        data = d.data.reshape(d.shape + (1,) * (3 - d.data.ndim))
        orientation = io_orientation(d.affine)
        self.data = apply_orientation(data, orientation)
        affine = d.affine @ inv_ornt_aff(orientation, data.shape[:3])
        self.sizes = np.linalg.norm(affine[:3, :3], axis=0)
        shape = self.data.shape
        self.index = [n // 2 for n in shape[:3]] + [0] * (self.data.ndim - 3)
        self.planes = [  # planes with more than one row and column
            p for p in self.PLANES if all(shape[a] > 1 for a in range(3) if a != p[1])
        ]
        self.sliders = {}
        names = ("x (L-R)", "y (P-A)", "z (I-S)") + tuple(d.dims[3:])
        axes = [a for a in range(self.data.ndim) if shape[a] > 1 and (
            a >= 3 or any(p[1] == a for p in self.planes))]

        bottom = 0.06 + 0.05 * len(axes)
        width = 0.9 / max(len(self.planes), 1)
        self.images = {}
        for n, (name, fixed, letters) in enumerate(self.planes):
            box = (0.05 + n * width, bottom + 0.02, width - 0.02, 0.86 - bottom)
            ax = self.figure.add_axes(box)
            ax.set_axis_off()
            ax.set_title(name, fontsize=10)
            for x, y, text in zip((0, 1, 0.5, 0.5), (0.5, 0.5, 0, 1), letters, strict=True):
                ax.text(x, y, text, transform=ax.transAxes, color="yellow", ha="center",
                        va="center", fontsize=9)
            rest = [a for a in range(3) if a != fixed]
            aspect = self.sizes[rest[1]] / self.sizes[rest[0]]
            self.images[fixed] = ax.imshow(self.plane(fixed), cmap="gray", origin="lower",
                                           aspect=aspect, interpolation="nearest")
        for n, a in enumerate(axes):
            slider_ax = self.figure.add_axes((0.18, 0.03 + 0.05 * n, 0.55, 0.03))
            slider = Slider(slider_ax, names[a], 0, shape[a] - 1, valinit=self.index[a],
                            valstep=1, valfmt="%d")
            slider.on_changed(lambda value, a=a: self.set_index(a, int(value)))
            self.sliders[a] = slider
        sizes = " x ".join(f"{v:.3g}" for v in d.voxel_sizes)
        self.figure.suptitle(f"{d.name}: {d.shape}, {sizes} mm", fontsize=10)
        self.update()

    def volume(self) -> np.ndarray:
        """The 3D volume at the current index of the further dimensions."""
        volume = self.data[(slice(None),) * 3 + tuple(self.index[3:])]
        return np.abs(volume) if np.iscomplexobj(volume) else volume

    def plane(self, fixed: int) -> np.ndarray:
        """The plane through the current index of axis ``fixed``, transposed for ``imshow``."""
        volume = self.volume()
        return np.take(volume, self.index[fixed], axis=fixed).T

    def set_index(self, axis: int, value: int) -> None:
        self.index[axis] = value
        self.update()

    def update(self) -> None:
        volume = self.volume()
        finite = volume[np.isfinite(volume)]
        low, high = np.percentile(finite, (0.5, 99.5)) if finite.size else (0, 1)
        for fixed, image in self.images.items():
            image.set_data(self.plane(fixed))
            image.set_clim(low, high if high > low else low + 1)
        self.figure.canvas.draw_idle()


#**************************************************************************************************#
#                                             RawView                                              #
#**************************************************************************************************#
#                                                                                                  #
# Raw readouts as acquired: the log magnitude of every sample of every readout of one coil, a      #
# slider for the coil. Nothing is sorted or reconstructed.                                         #
#                                                                                                  #
#**************************************************************************************************#
class RawView:
    SIZE = (10, 7)

    def __init__(self, d: RawData, figure: Figure | None = None):
        self.d = d
        self.figure = figure if figure is not None else Figure(figsize=self.SIZE)
        self.coil = 0
        bottom = 0.14 if d.n_coils > 1 else 0.08
        self.ax = self.figure.add_axes((0.1, bottom, 0.85, 0.84 - bottom))
        self.image = self.ax.imshow(self.log_magnitude(), cmap="gray", aspect="auto",
                                    interpolation="nearest")
        self.ax.set_xlabel("sample")
        self.ax.set_ylabel("readout (as acquired)")
        self.slider = None
        if d.n_coils > 1:
            slider_ax = self.figure.add_axes((0.18, 0.03, 0.55, 0.03))
            self.slider = Slider(slider_ax, "coil", 0, d.n_coils - 1, valinit=0, valstep=1,
                                 valfmt="%d")
            self.slider.on_changed(lambda value: self.set_coil(int(value)))
        self.update()

    def log_magnitude(self) -> np.ndarray:
        """log10 of the magnitude of the current coil's readouts, floored at 1e-6 of its maximum."""
        magnitude = np.abs(self.d.data[:, self.coil, :])
        floor = magnitude.max() * 1e-6 if magnitude.max() > 0 else 1.0
        return np.log10(np.maximum(magnitude, floor))

    def set_coil(self, coil: int) -> None:
        self.coil = coil
        self.update()

    def update(self) -> None:
        image = self.log_magnitude()
        self.image.set_data(image)
        self.image.set_clim(*np.percentile(image, (1, 99.9)))
        self.ax.set_title(f"{self.d.name}: {self.d.n_readouts} readouts x {self.d.n_coils} coils "
                          f"x {self.d.data.shape[-1]} samples, log magnitude", fontsize=10)
        self.figure.canvas.draw_idle()


#**************************************************************************************************#
#                                            PhysioView                                            #
#**************************************************************************************************#
#                                                                                                  #
# A physiological recording: one trace per channel against seconds from the first sample (sample   #
# numbers without a sampling frequency), events as vertical lines.                                 #
#                                                                                                  #
#**************************************************************************************************#
class PhysioView:
    SIZE = (12, 6)

    def __init__(self, d: PhysioData, figure: Figure | None = None):
        self.d = d
        self.figure = figure if figure is not None else Figure(figsize=self.SIZE)
        n = d.n_samples
        rate = d.sampling_frequency
        x = np.arange(n) / rate if rate else np.arange(n)
        self.axes = self.figure.subplots(d.data.shape[1], 1, sharex=True, squeeze=False)[:, 0]
        for k, ax in enumerate(self.axes):
            ax.plot(x, d.data[:, k], lw=0.6)
            ax.set_ylabel(d.channels[k] if k < len(d.channels) else f"channel {k}")
            for m, (name, indices) in enumerate(d.events.items()):
                indices = np.asarray(indices, dtype=int)
                indices = indices[indices < n]
                if indices.size:
                    ax.vlines(x[indices], 0, 1, transform=ax.get_xaxis_transform(), lw=0.5,
                              color=f"C{m + 1}", alpha=0.6, label=name)
        if any(len(indices) for indices in d.events.values()):
            self.axes[0].legend(loc="upper right", fontsize=8)
        self.axes[-1].set_xlabel("s from the first sample" if rate else "sample")
        rate = f" at {rate:g} Hz" if rate else ""
        self.figure.suptitle(f"{d.name}: {n} samples{rate}", fontsize=10)


#**************************************************************************************************#
#                                             FitView                                              #
#**************************************************************************************************#
#                                                                                                  #
# A fit result: the data, fit and baseline on the ppm axis with the residual above, when the       #
# format holds the curves, and the concentrations with their error (from sd_percent) as bars, when #
# it holds them.                                                                                   #
#                                                                                                  #
#**************************************************************************************************#
class FitView:
    SIZE = (12, 6)

    def __init__(self, d: FitResult, figure: Figure | None = None):
        self.d = d
        self.figure = figure if figure is not None else Figure(figsize=self.SIZE)
        curves = "ppm" in d.curves and "data" in d.curves
        rows = [r for r in d.rows if isinstance(r.get("concentration"), (int, float))]
        width = 0.58 if curves and rows else 0.88
        if curves:
            self._curves(width)
        if rows:
            self._concentrations(rows, 0.72 if curves else 0.1, 0.25 if curves else 0.85)
        if not curves and not rows:
            self.figure.text(0.5, 0.5, "no curves or concentrations", ha="center")
        voxel = f", voxel {d.voxel}" if d.voxel else ""
        self.figure.suptitle(f"{d.name}{voxel}", fontsize=10)

    def _curves(self, width: float) -> None:
        curves = self.d.curves
        ppm = curves["ppm"]
        ax = self.figure.add_axes((0.06, 0.1, width, 0.62))
        top = self.figure.add_axes((0.06, 0.76, width, 0.14), sharex=ax)
        ax.plot(ppm, curves["data"], color="k", lw=0.7, label="data")
        for name, color in (("fit", "C3"), ("baseline", "0.5")):
            if name in curves:
                ax.plot(ppm, curves[name], color=color, lw=0.9, label=name)
        if "fit" in curves:
            top.plot(ppm, curves["data"] - curves["fit"], color="k", lw=0.6)
        top.set_ylabel("residual", fontsize=8)
        top.tick_params(labelbottom=False)
        ax.set_xlim(ppm.max(), ppm.min())
        ax.set_xlabel("ppm")
        ax.legend(loc="upper left", fontsize=8)

    def _concentrations(self, rows: list[dict], left: float, width: float) -> None:
        ax = self.figure.add_axes((left, 0.1, width, 0.8))
        values = np.array([r["concentration"] for r in rows], dtype=float)
        sd = [r.get("sd_percent") for r in rows]
        errors = np.array([v * s / 100 if isinstance(s, (int, float)) else 0
                           for v, s in zip(values, sd, strict=True)])
        positions = np.arange(len(rows))
        ax.barh(positions, values, xerr=errors, color="C0", error_kw={"lw": 0.6})
        ax.set_yticks(positions, [r["metabolite"] for r in rows], fontsize=7)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_xlim(0, (values + errors).max() * 1.05 if values.size and values.max() > 0 else 1)
        ax.set_xlabel("concentration", fontsize=8)


#**************************************************************************************************#
#                                           SequenceView                                           #
#**************************************************************************************************#
#                                                                                                  #
# A pulse sequence diagram over a run of blocks: ADC windows, RF magnitude (Hz) and the gradients  #
# (kHz/m) against ms from the sequence start, with sliders for the first block and the number of   #
# blocks. Waveforms follow the Pulseq specification: times in the tables in us (ADC dwell in ns),  #
# shapes sampled at the centres of the raster intervals unless a time shape (in raster units)      #
# gives their times, gradient time id -1 marking a shape oversampled by two; a block lasts its     #
# duration (from 1.4) or until its last event ends.                                                #
#                                                                                                  #
#**************************************************************************************************#
class SequenceView:
    SIZE = (12, 7)
    SHOWN = 50  # blocks at first
    MOST = 1000  # blocks at most
    ROWS = ("adc", "rf", "gx", "gy", "gz")
    LABELS = ("ADC", "RF (Hz)", "Gx (kHz/m)", "Gy (kHz/m)", "Gz (kHz/m)")

    def __init__(self, d: PulseSequence, figure: Figure | None = None):
        self.d = d
        self.figure = figure if figure is not None else Figure(figsize=self.SIZE)
        self.starts = d.starts
        self.first, self.count = 0, min(d.n_blocks, self.SHOWN)

        self.figure.subplots_adjust(left=0.1, right=0.97, top=0.93, bottom=0.2, hspace=0.1)
        self.axes = self.figure.subplots(len(self.ROWS), 1, sharex=True)
        self.lines = {}
        for row, label, ax in zip(self.ROWS, self.LABELS, self.axes, strict=True):
            [self.lines[row]] = ax.plot([], [], lw=0.7, color="C3" if row == "rf" else "C0")
            ax.set_ylabel(label, fontsize=8)
        self.axes[-1].set_xlabel("ms")
        if d.n_blocks > 1:
            first_ax = self.figure.add_axes((0.18, 0.07, 0.55, 0.025))
            self.first_slider = Slider(first_ax, "first block", 0, d.n_blocks - 1, valinit=0,
                                       valstep=1, valfmt="%d")
            self.first_slider.on_changed(lambda value: self.show(int(value), self.count))
            count_ax = self.figure.add_axes((0.18, 0.025, 0.55, 0.025))
            most = min(d.n_blocks, self.MOST)
            self.count_slider = Slider(count_ax, "blocks", 1, most, valinit=self.count,
                                       valstep=1, valfmt="%d")
            self.count_slider.on_changed(lambda value: self.show(self.first, int(value)))
        self.show(self.first, self.count)

    def show(self, first: int, count: int) -> None:
        self.first, self.count = first, count
        traces = {row: ([], []) for row in self.ROWS}
        last = min(first + count, self.d.n_blocks)
        for n in range(first, last):
            block, start = self.d.blocks[n], self.starts[n]
            events = {}
            if block["rf"]:
                events["rf"] = self.d.rf(int(block["rf"]))[:2]
            for g in ("gx", "gy", "gz"):
                if block[g]:
                    t, amplitude, _ = self.d.gradient(int(block[g]))
                    events[g] = (t, amplitude / 1e3)
            if block["adc"]:
                on, off = self.d.adc(int(block["adc"]))
                events["adc"] = (np.array([on, on, off, off]), np.array([0.0, 1, 1, 0]))
            for row, (t, y) in events.items():
                traces[row][0].extend([(start + t) * 1e3, [np.nan]])
                traces[row][1].extend([y, [np.nan]])
        for row, (t, y) in traces.items():
            self.lines[row].set_data(np.concatenate(t) if t else [], np.concatenate(y) if y else [])
        for ax in self.axes:
            ax.relim()
            ax.autoscale_view()
        if self.starts[last] > self.starts[first]:
            self.axes[0].set_xlim(self.starts[first] * 1e3, self.starts[last] * 1e3)
        self.axes[0].set_title(f"{self.d.name}: Pulseq {self.d.version}, blocks {first + 1} to "
                               f"{last} of {self.d.n_blocks}", fontsize=10)
        self.figure.canvas.draw_idle()


#**************************************************************************************************#
#                                            NoiseView                                             #
#**************************************************************************************************#
#                                                                                                  #
# A noise covariance: the magnitude of the noise correlation between every pair of channels and    #
# the noise standard deviation of each channel (the square root of the diagonal).                  #
#                                                                                                  #
#**************************************************************************************************#
class NoiseView:
    SIZE = (11, 5)

    def __init__(self, d: NoiseCovariance, figure: Figure | None = None):
        self.d = d
        self.figure = figure if figure is not None else Figure(figsize=self.SIZE)
        sigma = np.sqrt(np.abs(np.real(np.diag(d.matrix))))
        with np.errstate(divide="ignore", invalid="ignore"):
            self.correlation = np.abs(d.matrix) / np.outer(sigma, sigma)
        left, right = self.figure.subplots(1, 2, width_ratios=(1, 1.3))
        image = left.imshow(self.correlation, vmin=0, vmax=1, cmap="viridis",
                            interpolation="nearest")
        self.figure.colorbar(image, ax=left, fraction=0.046)
        left.set_title("noise correlation (magnitude)", fontsize=10)
        left.set_xlabel("channel")
        left.set_ylabel("channel")
        right.bar(np.arange(d.n_coils), sigma)
        right.set_title("noise standard deviation", fontsize=10)
        names = [label.get("name") for label in d.coil_labels]
        if len(names) == d.n_coils and all(names):
            right.set_xticks(np.arange(d.n_coils), names, rotation=90, fontsize=6)
        else:
            right.set_xlabel("channel")
        samples = f", {d.sample_count} samples" if d.sample_count else ""
        self.figure.suptitle(f"{d.name}: {d.n_coils} channels{samples}", fontsize=10)


#**************************************************************************************************#
#                                               view                                               #
#**************************************************************************************************#
#                                                                                                  #
# The view of a dataset: in a matplotlib window (inline in notebooks), or saved to ``save``        #
# without a window. TypeError for objects that have no view yet.                                   #
#                                                                                                  #
#**************************************************************************************************#
VIEWS = {
    MRSData: SpectrumView,
    MRImage: ImageView,
    RawData: RawView,
    PhysioData: PhysioView,
    FitResult: FitView,
    PulseSequence: SequenceView,
    NoiseCovariance: NoiseView,
}


def view_class(d) -> type:
    for kind, cls in VIEWS.items():
        if isinstance(d, kind):
            return cls
    raise TypeError(f"no view for {type(d).__name__} yet")


def show(datasets: list) -> list:
    """One window per dataset, all open until closed."""
    import matplotlib.pyplot as plt  # only for windows: saving needs no GUI backend

    views = [view_class(d)(d, plt.figure(figsize=view_class(d).SIZE)) for d in datasets]
    plt.show()
    return views


def view(d, save: str | Path | None = None):
    if save is None:
        return show([d])[0]
    shown = view_class(d)(d)
    shown.figure.savefig(save, dpi=150)
    return shown
