####################################################################################################
#                                            viewer.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 11/10/26                                                                                #
#                                                                                                  #
# Purpose: Quick look at what a file holds, as matplotlib figures: spectra (real, imaginary or     #
#          magnitude, on a ppm or Hz axis, a slider per further dimension) and images (three       #
#          planes in RAS+ orientation, a slider per slice and further dimension). Figures render   #
#          in a window, inline in notebooks, or headless to a file.                                #
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
from miox.data import BasisSet, MRImage, MRSData


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
#                                               view                                               #
#**************************************************************************************************#
#                                                                                                  #
# The view of a dataset: in a matplotlib window (inline in notebooks), or saved to ``save``        #
# without a window. TypeError for objects that have no view yet.                                   #
#                                                                                                  #
#**************************************************************************************************#
VIEWS = {MRSData: SpectrumView, MRImage: ImageView}


def view_class(d) -> type[SpectrumView] | type[ImageView]:
    for kind, cls in VIEWS.items():
        if isinstance(d, kind):
            return cls
    raise TypeError(f"no view for {type(d).__name__} yet")


def show(datasets: list) -> list[SpectrumView | ImageView]:
    """One window per dataset, all open until closed."""
    import matplotlib.pyplot as plt  # only for windows: saving needs no GUI backend

    views = [view_class(d)(d, plt.figure(figsize=view_class(d).SIZE)) for d in datasets]
    plt.show()
    return views


def view(d, save: str | Path | None = None) -> SpectrumView | ImageView:
    if save is None:
        return show([d])[0]
    shown = view_class(d)(d)
    shown.figure.savefig(save, dpi=150)
    return shown
