####################################################################################################
#                                            viewer.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 11/10/26                                                                                #
#                                                                                                  #
# Purpose: Quick look at what a file holds, as matplotlib figures: spectra (raw: real, imaginary   #
#          or magnitude on a ppm or Hz axis, a slider per further dimension; processed: the Quick  #
#          Look pipeline's result; 2-D NMR as contours), images (three planes in RAS+ orientation, #
#          a slider per slice and further dimension), processed NMR spectra, raw k-space (sorted,  #
#          the plain image where checked, the readouts as acquired), physiological recordings,     #
#          fit results, pulse sequence diagrams, noise covariances, tractograms, surfaces,         #
#          transforms, phantoms, RF pulses, protocols and prior knowledge. The pictures are the    #
#          Quick Look pages' (quicklook_pages computes them). Figures render in a window, inline   #
#          in notebooks, or headless to a file.                                                    #
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
    Phantom,
    PhysioData,
    PriorSet,
    Protocol,
    PulseSequence,
    RawData,
    RFPulse,
    Surface,
    Tractogram,
    Transform,
)
from miox.quicklook_pages import (
    PhantomPage,
    PriorPage,
    RawPage,
    RowsPage,
    SpectrumPage,
    SurfacePage,
    TractogramPage,
)


#**************************************************************************************************#
#                                           SpectrumView                                           #
#**************************************************************************************************#
#                                                                                                  #
# One spectroscopy dataset (or basis set) as a spectrum. Raw: the Fourier transform of the FID at  #
# the sliders' index in the NIfTI-MRS convention (conjugated, as FSL-MRS shows it), on the axis    #
# ReferencePPM - f / SpectrometerFrequency; without a stated reference the nucleus' NIfTI-MRS      #
# default is used (4.65 ppm for 1H). Without a frequency the axis is in Hz, without a dwell time   #
# in points. Processed (a radio, when the Quick Look can process the data): the Quick Look         #
# pipeline's result, a slider per edit condition and over the voxels of a grid. A TopSpin 2-D      #
# experiment the Quick Look transforms (QF, echo-antiecho) shows its 2-D magnitude spectrum below  #
# the row, as contours on the Quick Look's levels.                                                 #
#                                                                                                  #
#**************************************************************************************************#
class SpectrumView:
    PARTS = ("magnitude", "real", "imaginary")
    UNITS = ("ppm", "Hz")
    MODES = ("raw", "processed")
    SIZE = (10, 6)

    def __init__(self, d: MRSData, figure: Figure | None = None):
        self.d = d
        self.figure = figure if figure is not None else Figure(figsize=self.SIZE)
        self.part, self.unit, self.mode = self.PARTS[0], self.UNITS[0], self.MODES[0]
        if d.spectrometer_frequency is None or d.dwelltime is None:
            self.unit = "Hz"
        self.index = [0] * d.data.ndim
        self.sliders = {}
        self.page = self.processing()
        self.condition, self.voxel = 0, self.page.prepared.focus if self.page else 0
        self.rows = RowsPage(d, d.name) if "DIM_INDIRECT_0" in d.dim_tags else None
        self.two_d = self.rows is not None and self.rows.why_not() is None
        axes = [a for a in range(d.data.ndim) if a != 3 and d.shape[a] > 1]
        extra = []  # (label, count, initial, setter) of the processed mode's sliders
        if self.page is not None and len(self.page.results) > 1:
            extra.append(("condition", len(self.page.results), 0, self.set_condition))
        if self.page is not None and self.page.prepared.grid:
            extra.append(("voxel", int(np.prod(self.page.prepared.spatial)), self.voxel,
                          self.set_voxel))

        bottom = 0.08 + 0.05 * (len(axes) + len(extra))
        top = 0.84 - bottom
        if self.two_d:
            self.ax = self.figure.add_axes((0.08, bottom + 0.06 + 0.72 * top, 0.72, 0.28 * top))
            self.ax.tick_params(labelbottom=False)
            self.ax_2d = self.figure.add_axes((0.08, bottom + 0.06, 0.72, 0.64 * top),
                                              sharex=self.ax)
        else:
            self.ax = self.figure.add_axes((0.08, bottom + 0.06, 0.72, top))
        [self.line] = self.ax.plot([], [], lw=0.8)
        self.ax.invert_xaxis()
        part_ax = self.figure.add_axes((0.83, 0.70, 0.15, 0.18))
        self.parts = RadioButtons(part_ax, self.PARTS)
        self.parts.on_clicked(self.set_part)
        if self.unit == "ppm":
            unit_ax = self.figure.add_axes((0.83, 0.50, 0.15, 0.12))
            self.units = RadioButtons(unit_ax, self.UNITS)
            self.units.on_clicked(self.set_unit)
        if self.page is not None:
            mode_ax = self.figure.add_axes((0.83, 0.30, 0.15, 0.12))
            self.modes = RadioButtons(mode_ax, self.MODES)
            self.modes.on_clicked(self.set_mode)
        for n, a in enumerate(axes):
            slider_ax = self.figure.add_axes((0.18, 0.04 + 0.05 * n, 0.55, 0.03))
            slider = Slider(slider_ax, d.dims[a], 0, d.shape[a] - 1, valinit=0, valstep=1,
                            valfmt="%d")
            slider.on_changed(lambda value, a=a: self.set_index(a, int(value)))
            self.sliders[a] = slider
        for n, (label, count, initial, setter) in enumerate(extra, start=len(axes)):
            slider_ax = self.figure.add_axes((0.18, 0.04 + 0.05 * n, 0.55, 0.03))
            slider = Slider(slider_ax, label, 0, count - 1, valinit=initial, valstep=1,
                            valfmt="%d")
            slider.on_changed(lambda value, setter=setter: setter(int(value)))
            self.sliders[label] = slider
        if self.two_d:
            self.contours()
        self.update()

    def processing(self):
        """The Quick Look page of the dataset, which holds the pipeline's results; None when
        the data cannot be processed (no dwell time or frequency, k-space, empty)."""
        from miox.quicklook import Page

        paths = self.d.provenance.paths
        try:
            page = Page([self.d], self.d.name, Path(paths[0]) if paths else None)
        except ValueError:
            return None
        return page if page.results else None

    def contours(self) -> None:
        """The 2-D magnitude spectrum of a TopSpin experiment, as the Quick Look draws it."""
        x, y, z = self.rows.spectrum_2d()
        z = z.T  # (f1 rows, f2 columns)
        levels = SpectrumPage.levels(SpectrumPage.noise(z))
        self.ax_2d.contour(x, y, z, levels=levels, colors="k", linewidths=0.4)
        self.ax_2d.set_ylim(y.max(), y.min())
        nuclei = self.d.meta.get("ResonantNucleus", ["", ""])
        self.ax_2d.set_ylabel(f"{nuclei[1] if len(nuclei) > 1 else ''} ppm (F1)")
        self.ax_2d.set_xlabel("ppm")
        mode = RowsPage.MODES[int(self.d.meta.vendor["acqu2s"].get("FnMODE") or 0)]
        self.ax_2d.set_title(f"2-D: magnitude of the plain transform of the {mode} data, "
                             f"{SpectrumPage.LEVELS} levels from {SpectrumPage.FIRST:g} x noise",
                             fontsize=9)

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

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.update()

    def set_condition(self, condition: int) -> None:
        self.condition = condition
        self.update()

    def set_voxel(self, voxel: int) -> None:
        self.voxel = voxel
        self.update()

    def processed(self) -> tuple[np.ndarray, np.ndarray]:
        """The pipeline's spectrum of the current condition and voxel, on the current axis."""
        result = self.page.results[self.condition]
        x, spec = self.page.spectrum(np.asarray(result.fid)[0, self.voxel])
        if self.unit == "Hz":
            x = (x - self.page.reference) * self.d.spectrometer_frequency
        return x, spec[0]

    def update(self) -> None:
        if self.mode == "processed":
            x, spectrum = self.processed()
        else:
            x, spectrum = self.axis(), self.spectrum()
        values = {"magnitude": np.abs, "real": np.real, "imaginary": np.imag}[self.part](spectrum)
        self.line.set_data(x, values)
        self.ax.set_xlim(x.max(), x.min())
        self.ax.relim()
        self.ax.autoscale_view(scalex=False)
        if self.two_d:
            pass  # the 2-D panel below carries the axis label
        elif self.d.dwelltime is None:
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
        if self.mode == "processed":
            steps = ", ".join(self.page.results[self.condition].steps)
            title += f"\nprocessed ({steps})" if steps else "\nprocessed"
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
# Raw k-space: the sorted k-space of one slice (log magnitude, root sum of squares over coils, the #
# lines at their phase-encode steps; sliders over the counters with several values), the plain     #
# reconstruction for formats whose plain FFT the Quick Look has checked against a reference, and   #
# the readouts as acquired (the log magnitude of every sample of every readout of one coil, a      #
# slider for the coil).                                                                            #
#                                                                                                  #
#**************************************************************************************************#
class RawView:
    SIZE = (14, 5)

    def __init__(self, d: RawData, figure: Figure | None = None):
        self.d = d
        self.figure = figure if figure is not None else Figure(figsize=self.SIZE)
        self.coil, self.pick = 0, {}
        self.verified = d.provenance.format in RawPage.VERIFIED
        _, _, chosen = RawPage.select(d)
        counters = [(name, count) for name, (_, count) in chosen.items()]
        bottom = 0.08 + 0.05 * (len(counters) + (d.n_coils > 1))
        height = 0.82 - bottom
        panes = 3 if self.verified else 2
        width = 0.9 / panes
        self.ax_k = self.figure.add_axes((0.07, bottom + 0.06, width - 0.06, height))
        self.sorted = self.ax_k.imshow(np.zeros((1, 1)), cmap="gray", aspect="auto",
                                       interpolation="nearest")
        self.ax_k.set_xlabel("sample")
        self.ax_k.set_ylabel("phase-encode line")
        self.ax_k.set_title("k-space of the slice shown, log |k|, all coils", fontsize=9)
        self.ax_image = None
        if self.verified:
            self.ax_image = self.figure.add_axes((0.07 + width, bottom + 0.06, width - 0.06,
                                                  height))
            self.ax_image.set_axis_off()
            self.picture = self.ax_image.imshow(np.zeros((1, 1)), cmap="gray",
                                                interpolation="nearest")
        self.ax = self.figure.add_axes((0.07 + (panes - 1) * width, bottom + 0.06, width - 0.06,
                                        height))
        self.image = self.ax.imshow(self.log_magnitude(), cmap="gray", aspect="auto",
                                    interpolation="nearest")
        self.ax.set_xlabel("sample")
        self.ax.set_ylabel("readout (as acquired)")
        self.sliders, self.slider = {}, None
        rows = 0
        if d.n_coils > 1:
            slider_ax = self.figure.add_axes((0.18, 0.03, 0.55, 0.03))
            self.slider = Slider(slider_ax, "coil", 0, d.n_coils - 1, valinit=0, valstep=1,
                                 valfmt="%d")
            self.slider.on_changed(lambda value: self.set_coil(int(value)))
            rows = 1
        for n, (name, count) in enumerate(counters, start=rows):
            slider_ax = self.figure.add_axes((0.18, 0.03 + 0.05 * n, 0.55, 0.03))
            slider = Slider(slider_ax, name, 0, count - 1, valinit=chosen[name][0], valstep=1,
                            valfmt="%d")
            slider.on_changed(lambda value, name=name: self.set_counter(name, int(value)))
            self.sliders[name] = slider
        self.update()

    def log_magnitude(self) -> np.ndarray:
        """log10 of the magnitude of the current coil's readouts, floored at 1e-6 of its maximum."""
        magnitude = np.abs(self.d.data[:, self.coil, :])
        floor = magnitude.max() * 1e-6 if magnitude.max() > 0 else 1.0
        return np.log10(np.maximum(magnitude, floor))

    def set_coil(self, coil: int) -> None:
        self.coil = coil
        self.update()

    def set_counter(self, name: str, value: int) -> None:
        self.pick[name] = value
        self.update()

    def update(self) -> None:
        d = self.d
        k, _, chosen = RawPage.select(d, self.pick)
        rss = np.log1p(np.sqrt((np.abs(k) ** 2).sum(axis=1)))
        self.sorted.set_data(rss)
        self.sorted.set_extent((-0.5, rss.shape[1] - 0.5, rss.shape[0] - 0.5, -0.5))
        self.sorted.set_clim(*np.percentile(rss, (0.5, 99.5)))
        if self.verified:
            picture, note = RawPage.plain_image(d, self.pick)
            self.picture.set_data(picture)
            self.picture.set_extent((-0.5, picture.shape[1] - 0.5, picture.shape[0] - 0.5, -0.5))
            self.picture.set_clim(*np.percentile(picture, (0.5, 99.5)))
            self.ax_image.set_title(f"plain FFT, root sum of squares ({note})", fontsize=9)
        image = self.log_magnitude()
        self.image.set_data(image)
        self.image.set_clim(*np.percentile(image, (1, 99.9)))
        self.ax.set_title(f"readouts as acquired, coil {self.coil + 1}, log magnitude",
                          fontsize=9)
        shown = ", ".join(f"{name} {value + 1} of {count}" for name, (value, count)
                          in chosen.items())
        self.figure.suptitle(f"{d.name}: {d.n_readouts} readouts x {d.n_coils} coils x "
                             f"{d.data.shape[-1]} samples" + (f"; shown: {shown}" if shown else ""),
                             fontsize=10)
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
#                                     Views of the other kinds                                     #
#**************************************************************************************************#
#                                                                                                  #
# The pictures of the Quick Look pages as figures: a tractogram's three projections, a surface's   #
# lateral, medial and top views, a transform's matrices, a phantom's property maps on its grid     #
# (a slider over the slices), an RF pulse's amplitude and phase, a protocol's values (a slider to  #
# scroll), a prior set's lines as sticks and per metabolite, and a processed NMR spectrum as       #
# stored (1-D: a radio for the part; 2-D: contours, a slider for the lowest level).                #
#                                                                                                  #
#**************************************************************************************************#
class TractogramView:
    SIZE = (12, 4.6)

    def __init__(self, d: Tractogram, figure: Figure | None = None):
        self.d = d
        self.figure = figure if figure is not None else Figure(figsize=self.SIZE)
        page = TractogramPage(d, d.name)
        self.axes = []
        if d.points is None or not len(d.points):
            self.figure.text(0.5, 0.5, "no streamline points in the file", ha="center")
        else:
            self.axes = list(self.figure.subplots(1, 3))
            for ax, (name, across, up, sides, how) in zip(self.axes, page.VIEWS, strict=True):
                ax.imshow(page.picture(across, up), interpolation="nearest")
                ax.set_axis_off()
                ax.set_title(f"{name} ({how})", fontsize=10)
                for x, text in zip((0, 1), sides, strict=True):
                    ax.text(x, 0.5, text, transform=ax.transAxes, color="yellow", ha="center",
                            va="center", fontsize=9)
        n = d.n_streamlines
        drawn = "" if n <= page.MOST else f" ({page.MOST} drawn)"
        self.figure.suptitle(f"{d.name}: {n} streamlines{drawn}, {d.space}; direction colours "
                             "(red left-right, green front-back, blue foot-head)", fontsize=10)


class SurfaceView:
    SIZE = (12, 4.6)

    def __init__(self, d: Surface, figure: Figure | None = None):
        self.d = d
        self.figure = figure if figure is not None else Figure(figsize=self.SIZE)
        paths = d.provenance.paths
        self.page = SurfacePage(d, d.name, Path(paths[0]) if paths else None)
        self.axes = []
        if self.page.drawable():
            self.axes = list(self.figure.subplots(1, 3))
            for ax, (name, view) in zip(self.axes, self.page.views(), strict=True):
                image = self.page.render(view)
                ax.imshow(np.where(np.isnan(image), 1.0, image), cmap="gray", vmin=0, vmax=1,
                          interpolation="nearest")
                ax.set_axis_off()
                ax.set_title(name, fontsize=10)
        else:
            self.figure.text(0.5, 0.5, "no mesh to draw (values only, or faces beyond the "
                             "vertices)", ha="center")
        notes = " ".join(self.page.notes)
        self.figure.suptitle(f"{d.name}: {self.page.facts()}" + (f"\n{notes}" if notes else ""),
                             fontsize=9)


class TransformView:
    SIZE = (9, 3.6)

    def __init__(self, d: Transform, figure: Figure | None = None):
        self.d = d
        self.figure = figure if figure is not None else Figure(figsize=self.SIZE)
        tables = [(f"as stored ({d.convention})", d.matrix)]
        if d.ras_matrix is not None:
            tables.append(("RAS", d.ras_matrix))
        if d.matrix is None:
            shape = None if d.field is None else np.asarray(d.field).shape
            self.figure.text(0.5, 0.5, f"a deformation field of shape {shape}", ha="center")
        else:
            axes = self.figure.subplots(1, len(tables), squeeze=False)[0]
            for ax, (name, matrix) in zip(axes, tables, strict=True):
                ax.set_axis_off()
                ax.set_title(name, fontsize=10)
                cells = [[f"{v:.6g}" for v in row] for row in np.asarray(matrix, dtype=float)]
                ax.table(cellText=cells, loc="center", cellLoc="right").scale(1, 1.6)
        self.figure.suptitle(f"{d.name}: {d.kind}", fontsize=10)


class PhantomView:
    SIZE = (12, 4)

    def __init__(self, d: Phantom, figure: Figure | None = None):
        self.d = d
        self.figure = figure if figure is not None else Figure(figsize=self.SIZE)
        self.page = PhantomPage(d, d.name)
        self.images, self.slider = {}, None
        if not self.page.gridded:
            self.figure.text(0.5, 0.5, "the phantom states no grid", ha="center")
            self.figure.suptitle(f"{d.name}: {d.n_spins} spins", fontsize=10)
            return
        self.k, maps = self.page.maps()
        slices = d.grid["shape"][2]
        bottom = 0.14 if slices > 1 else 0.04
        width = 0.96 / max(len(maps), 1)
        for n, (name, unit, picture, window) in enumerate(maps):
            ax = self.figure.add_axes((0.02 + n * width, bottom, width - 0.02, 0.78 - bottom))
            ax.set_axis_off()
            ax.set_title(f"{name} {unit}".strip(), fontsize=9)
            self.images[name] = ax.imshow(picture, cmap="gray", origin="lower", vmin=window[0],
                                          vmax=window[1], interpolation="nearest")
        if slices > 1:
            slider_ax = self.figure.add_axes((0.18, 0.03, 0.55, 0.03))
            self.slider = Slider(slider_ax, "slice", 0, slices - 1, valinit=self.k, valstep=1,
                                 valfmt="%d")
            self.slider.on_changed(lambda value: self.set_slice(int(value)))
        self.update()

    def set_slice(self, k: int) -> None:
        self.k = k
        self.update()

    def update(self) -> None:
        d, shape = self.d, self.d.grid["shape"]
        _, maps = self.page.maps(self.k)
        for name, _, picture, window in maps:
            self.images[name].set_data(picture)
            self.images[name].set_clim(window[0], window[1] if window[1] > window[0]
                                       else window[0] + 1)
        self.figure.suptitle(f"{d.name}: {d.n_spins} spins on a {shape[0]} x {shape[1]} x "
                             f"{shape[2]} grid, slice {self.k + 1}, windowed 0.5-99.5 %",
                             fontsize=10)
        self.figure.canvas.draw_idle()


class RFPulseView:
    SIZE = (9, 4.6)

    def __init__(self, d: RFPulse, figure: Figure | None = None):
        self.d = d
        self.figure = figure if figure is not None else Figure(figsize=self.SIZE)
        n = d.n_samples
        t = np.arange(n) * (d.duration / n * 1e6) if d.duration else np.arange(n)
        self.axes = self.figure.subplots(2, 1, sharex=True)
        self.axes[0].plot(t, np.asarray(d.amplitude, dtype=np.float64), lw=0.9)
        self.axes[0].set_ylabel(f"amplitude {d.amplitude_unit}".strip(), fontsize=9)
        self.axes[1].plot(t, np.asarray(d.phase, dtype=np.float64), lw=0.9, color="C3")
        self.axes[1].set_ylabel("phase (degrees)", fontsize=9)
        self.axes[1].set_xlabel("us" if d.duration else "sample")
        duration = f", {d.duration * 1e6:g} us" if d.duration else ", no duration stated"
        self.figure.suptitle(f"{d.name}: {n} samples{duration}", fontsize=10)


class ProtocolView:
    SIZE = (9, 7)
    ROWS = 30  # values shown at once

    def __init__(self, d: Protocol, figure: Figure | None = None):
        self.d = d
        self.figure = figure if figure is not None else Figure(figsize=self.SIZE)
        self.items = [(k, str(v)) for k, v in d.meta.items()]
        self.first = 0
        bottom = 0.1 if len(self.items) > self.ROWS else 0.03
        self.ax = self.figure.add_axes((0.02, bottom, 0.96, 0.9 - bottom))
        self.ax.set_axis_off()
        self.slider = None
        if len(self.items) > self.ROWS:
            slider_ax = self.figure.add_axes((0.18, 0.03, 0.55, 0.03))
            self.slider = Slider(slider_ax, "first value", 0, len(self.items) - self.ROWS,
                                 valinit=0, valstep=1, valfmt="%d")
            self.slider.on_changed(lambda value: self.set_first(int(value)))
        self.update()

    def set_first(self, first: int) -> None:
        self.first = first
        self.update()

    def update(self) -> None:
        for table in list(self.ax.tables):
            table.remove()
        shown = self.items[self.first:self.first + self.ROWS]
        if shown:
            table = self.ax.table(cellText=[[k, v[:70]] for k, v in shown],
                                  colLabels=["key", "value"], loc="upper left", cellLoc="left",
                                  colWidths=[0.35, 0.65])
            table.auto_set_font_size(False)
            table.set_fontsize(7)
        last = min(self.first + self.ROWS, len(self.items))
        self.figure.suptitle(f"{self.d.name}: values {self.first + 1} to {last} of "
                             f"{len(self.items)}", fontsize=10)
        self.figure.canvas.draw_idle()


class PriorView:
    SIZE = (11, 8)

    def __init__(self, d: PriorSet, figure: Figure | None = None):
        self.d = d
        self.figure = figure if figure is not None else Figure(figsize=self.SIZE)
        page = PriorPage(d, d.name)
        lines, ppm, size = page.values()
        if not lines:
            self.figure.text(0.5, 0.5, "no lines with a shift", ha="center")
            return
        tallest = max(float(size.max()), 1e-12)
        grid = self.figure.add_gridspec(2, 1, height_ratios=(1, 2.2), hspace=0.25)
        self.top = self.figure.add_subplot(grid[0])
        self.top.vlines(ppm, 0, size / tallest, lw=1)
        pad = 0.05 * (np.ptp(ppm) or 1.0)
        self.top.set_xlim(float(ppm.max()) + pad, float(ppm.min()) - pad)
        self.top.set_yticks([])
        self.top.set_ylabel(page.key)
        named = {}
        for i, line in enumerate(lines):
            name = line["metabolite"]
            if name not in named or size[i] > size[named[name]]:
                named[name] = i
        for n, (name, i) in enumerate(sorted(named.items(), key=lambda kv: -ppm[kv[1]])):
            self.top.text(ppm[i], 1.02 + 0.09 * (n % 4), name, fontsize=7, ha="center")
        self.top.set_ylim(0, 1.45)
        self.bottom = self.figure.add_subplot(grid[1], sharex=self.top)
        names = d.metabolites
        for r, name in enumerate(names):
            own = [i for i, line in enumerate(lines) if line["metabolite"] == name]
            tall = max(float(size[own].max()), 1e-12) if own else 1.0
            self.bottom.vlines(ppm[own], r, r + 0.85 * size[own] / tall, lw=1)
            self.bottom.axhline(r, lw=0.3, color="0.7")
        self.bottom.set_yticks(np.arange(len(names)), names, fontsize=7)
        self.bottom.set_ylim(len(names), -0.5)
        self.bottom.set_xlabel("ppm")
        self.figure.suptitle(f"{d.name}: {len(names)} metabolites, {len(d.lines)} lines; every "
                             f"line as a stick as tall as its {page.key} (top), one row per "
                             "metabolite on its own scale (bottom)", fontsize=10)


class SpectrumImageView:
    """A processed spectrum stored as such (TopSpin 1r/2rr, Sparky, NMRPipe ft): an MRImage
    whose dimensions are frequencies."""
    PARTS = ("real", "imaginary", "magnitude")
    FACTORS = (1, 2, 4, 8, 16, 32, 64)  # the lowest contour level, in units of the default
    SIZE = (11, 4)
    SIZE_2D = (9, 7)

    def __init__(self, d: MRImage, figure: Figure | None = None):
        self.d = d
        self.page = SpectrumPage(d, d.name)
        size = self.SIZE_2D if self.page.two_d else self.SIZE
        self.figure = figure if figure is not None else Figure(figsize=size)
        self.part, self.factor = self.PARTS[0], 0
        if self.page.two_d:
            self.ax = self.figure.add_axes((0.08, 0.16, 0.88, 0.76))
            x, y = self.page.axes()
            self.z = np.real(self.page.data).reshape(x.size, y.size).T
            self.x, self.y = x, y
            self.noise = SpectrumPage.noise(self.z)
            slider_ax = self.figure.add_axes((0.18, 0.04, 0.55, 0.03))
            self.slider = Slider(slider_ax, "lowest level x", 0, len(self.FACTORS) - 1,
                                 valinit=0, valstep=1, valfmt="%d")
            self.slider.on_changed(lambda value: self.set_factor(int(value)))
        else:
            self.ax = self.figure.add_axes((0.06, 0.14, 0.72, 0.76))
            [self.line] = self.ax.plot([], [], lw=0.7)
            part_ax = self.figure.add_axes((0.83, 0.60, 0.15, 0.2))
            self.parts = RadioButtons(part_ax, self.PARTS)
            self.parts.on_clicked(self.set_part)
        self.update()

    def set_part(self, part: str) -> None:
        self.part = part
        self.update()

    def set_factor(self, factor: int) -> None:
        self.factor = factor
        self.update()

    def update(self) -> None:
        d, page = self.d, self.page
        nuclei = d.meta.get("ResonantNucleus", ["", ""])
        nuclei = list(nuclei) if isinstance(nuclei, (list, tuple)) else [nuclei, ""]
        if page.two_d:
            self.ax.cla()
            first = self.FACTORS[self.factor] * SpectrumPage.FIRST
            levels = SpectrumPage.levels(self.noise, first)
            self.ax.contour(self.x, self.y, self.z, levels=levels, colors="k", linewidths=0.4)
            self.ax.contour(self.x, self.y, self.z, levels=-levels[::-1], colors="C3",
                            linewidths=0.4)
            self.ax.set_xlim(float(self.x.max()), float(self.x.min()))
            self.ax.set_ylim(float(self.y.max()), float(self.y.min()))
            self.ax.set_xlabel(f"{nuclei[0]} {page.unit}")
            self.ax.set_ylabel(f"{nuclei[1] if len(nuclei) > 1 else ''} {page.unit}")
            self.ax.set_title(f"{d.name}: {SpectrumPage.LEVELS} contour levels from {first:g} x "
                              f"noise rising by {SpectrumPage.STEP}, negative in red", fontsize=10)
        else:
            x, _ = page.axes()
            values = np.asarray(page.data).reshape(x.size, -1)[:, 0]
            values = {"real": np.real, "imaginary": np.imag, "magnitude": np.abs}[self.part](values)
            self.line.set_data(x, values)
            self.ax.set_xlim(float(x.max()), float(x.min()))
            self.ax.relim()
            self.ax.autoscale_view(scalex=False)
            self.ax.set_xlabel(f"{nuclei[0]} {page.unit}")
            self.ax.set_yticks([])
            self.ax.set_title(f"{d.name}: processed spectrum as stored, {self.part}", fontsize=10)
        self.figure.canvas.draw_idle()


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
    Tractogram: TractogramView,
    Surface: SurfaceView,
    Transform: TransformView,
    Phantom: PhantomView,
    RFPulse: RFPulseView,
    Protocol: ProtocolView,
    PriorSet: PriorView,
}


def view_class(d) -> type:
    if isinstance(d, MRImage) and "frequency" in d.dims:
        return SpectrumImageView
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
