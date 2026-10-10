####################################################################################################
#                                            sparky.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 10/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for Sparky (UCSF) spectra of 2 or 3 dimensions as MRImage, with the axes of the  #
#          header as standard keys. The binary layout is read by nmrglue.                          #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import nmrglue as ng
import numpy as np

# own
from miox.data import MRImage
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                           SparkyReader                                           #
#**************************************************************************************************#
#                                                                                                  #
# A UCSF file holds a processed spectrum, axes w1, w2 (, w3) from the slowest to the fastest. The  #
# file does not say which axis was acquired; the fastest (last) one is taken as the direct axis,   #
# as NMRPipe's pipe2ucsf writes it, and comes first, the others follow as indirect axes. Each axis #
# is placed by its header (spectrometer frequency, spectral width, transmitter frequency in ppm at #
# the centre, as nmrglue's unit conversion reads it), with the chemical shift falling with the     #
# index.                                                                                           #
#                                                                                                  #
#**************************************************************************************************#
class SparkyReader(Reader):
    FORMATS = (Format.SPARKY,)
    PACKAGES = Reader.PACKAGES + ("nmrglue",)
    NUCLEUS = re.compile(r"(\d+)([A-Za-z]+)")  # 1H, 15N

    def read(self, sniff: Sniff) -> list[MRImage]:
        path = Path(sniff.path)
        try:
            dic, data = ng.sparky.read(str(path))
        except (ValueError, OSError, KeyError, IndexError) as e:
            raise ReaderError(f"{path}: not a readable UCSF file ({e})") from e
        data = np.asarray(data)
        order = list(range(data.ndim))[::-1]  # fastest (direct) axis first
        values = np.transpose(data, order).astype(np.float64)
        axes = [ng.sparky.make_uc(dic, data, axis).ppm_scale() for axis in order]
        for i, ppm in enumerate(axes):
            if ppm[0] < ppm[-1]:  # the shift falls with the index
                values, axes[i] = np.flip(values, axis=i), ppm[::-1]
        headers = [dic[f"w{axis + 1}"] for axis in order]
        meta = Meta(self._vendor(dic))
        source = "UCSF header"
        meta.set("SpectrometerFrequency", [float(h["spectrometer_freq"]) for h in headers],
                 f"{source}: spectrometer_freq", "spectrometer_freq")
        nuclei = [self._nucleus(h["nucleus"]) for h in headers]
        if None not in nuclei:
            meta.set("ResonantNucleus", nuclei, f"{source}: nucleus", "nucleus")
        n, ppm = values.shape[0], axes[0]
        frequency = float(headers[0]["spectrometer_freq"])
        step = float(abs(ppm[1] - ppm[0]))
        meta.set("NumberOfPoints", int(n), "data shape")
        meta.set("SpectralWidth", step * frequency * n, f"{source}: axis spacing * points",
                 "spectral_width")
        meta.set("DwellTime", 1 / (step * frequency * n), "derived: 1 / SpectralWidth")
        meta.set("ReferencePPM", float(ppm[n // 2]), f"{source}: shift of point N / 2",
                 "xmtr_freq")
        notes = ("the last axis is taken as the acquired one (not stated in the file)",
                 "indirect axes in the vendor header (w1, w2: spectrometer_freq, "
                 "spectral_width, xmtr_freq)")
        names = ("frequency",) + tuple(f"frequency_indirect_{i}" for i in range(values.ndim - 1))
        return [MRImage(values.reshape((1, 1, 1) + values.shape), self.NO_GEOMETRY, meta,
                        self.provenance(sniff, notes), path.stem, names)]

    def _nucleus(self, text: Any) -> str | None:
        match = self.NUCLEUS.fullmatch(str(text).strip())
        return match.group(1) + match.group(2) if match else None

    def _vendor(self, dic: dict) -> dict[str, Any]:
        vendor = {}
        for key, value in dic.items():
            if isinstance(value, dict):
                vendor[key] = {k: v for k, v in self.plain(value).items() if k != "extended"}
            else:
                vendor[key] = self.plain(value)
        return vendor
