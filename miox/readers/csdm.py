####################################################################################################
#                                             csdm.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 30/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for the Core Scientific Dataset Model (.csdf JSON), as mrsimulator and RMN       #
#          write NMR spectra: frequency-domain dependent variables as MRImage.                     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import base64
import json
import re

import numpy as np

# own
from miox.data import MRImage
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                            CsdmReader                                            #
#**************************************************************************************************#
#                                                                                                  #
# The file's dimensions are listed fastest first; every dependent variable is one MRImage with its #
# dimensions in that order (the first, fastest, is taken as the direct axis). Linear dimensions    #
# only: coordinates offset + increment x index, from -N/2 when complex_fft is set; a frequency     #
# dimension with an origin_offset (the reference frequency) gives the spectrometer frequency.      #
# Components are internal, base64 or plain lists, in the stated numeric type, little endian; a     #
# variable of several components (vector, pixel) has them on the last axis. Time-domain data,      #
# external components and non-linear dimensions are refused (no sample to read them against).      #
#                                                                                                  #
#**************************************************************************************************#
class CsdmReader(Reader):
    FORMATS = (Format.CSDM,)
    PACKAGES = ("miox",)
    TYPES = {"uint8": "<u1", "uint16": "<u2", "uint32": "<u4", "uint64": "<u8", "int8": "<i1",
             "int16": "<i2", "int32": "<i4", "int64": "<i8", "float32": "<f4",
             "float64": "<f8", "complex64": "<c8", "complex128": "<c16"}
    QUANTITY = re.compile(r"^\s*([-+0-9.eE]+)\s*(\S*)\s*$")
    HERTZ = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}

    def read(self, sniff: Sniff) -> list[MRImage]:
        try:
            csdm = json.loads(sniff.path.read_text(encoding="utf-8"))["csdm"]
        except (ValueError, KeyError) as error:
            raise ReaderError(f"{sniff.path.name}: not a CSDM file ({error})") from error
        dimensions = csdm.get("dimensions", [])
        axes = [self._axis(d, sniff) for d in dimensions]
        counts = [len(a) for a in axes]
        images = []
        for n, variable in enumerate(csdm.get("dependent_variables", [])):
            data = self._components(variable, counts, sniff)  # (fastest, ..., slowest, comps)
            if data.shape[-1] == 1:
                data = data[..., 0]
            meta = Meta({key: value for key, value in csdm.items() if key != "dependent_variables"}
                        | {"dependent_variable": {k: v for k, v in variable.items()
                                                  if k != "components"}})
            self._keys(meta, dimensions, axes)
            name = variable.get("name") or (sniff.path.stem if n == 0 else f"{sniff.path.stem}_{n}")
            names = ("frequency",) + tuple(f"frequency_indirect_{i}" for i in range(len(axes) - 1))
            names += ("component",) if data.ndim > len(axes) else ()
            images.append(MRImage(data.reshape((1, 1, 1) + data.shape), self.NO_GEOMETRY, meta,
                                  self.provenance(sniff), name, names))
        if not images:
            raise ReaderError(f"{sniff.path.name}: no dependent variables")
        return images

    def _axis(self, dimension: dict, sniff: Sniff) -> np.ndarray:
        if dimension.get("type") != "linear":
            raise ReaderError(f"{sniff.path.name}: {dimension.get('type')} dimensions not read")
        if dimension.get("quantity_name") != "frequency":
            raise ReaderError(f"{sniff.path.name}: {dimension.get('quantity_name')} dimensions "
                              "not read (frequency only)")
        count = int(dimension["count"])
        increment = self._hertz(dimension["increment"], sniff)
        offset = self._hertz(dimension.get("coordinates_offset", "0 Hz"), sniff)
        index = np.arange(count) - (count // 2 if dimension.get("complex_fft") else 0)
        return offset + increment * index

    def _hertz(self, quantity: str, sniff: Sniff) -> float:
        match = self.QUANTITY.match(str(quantity))
        if match is None or match.group(2) not in self.HERTZ:
            raise ReaderError(f"{sniff.path.name}: quantity {quantity!r} not in Hz")
        return float(match.group(1)) * self.HERTZ[match.group(2)]

    def _components(self, variable: dict, counts: list[int], sniff: Sniff) -> np.ndarray:
        if variable.get("type", "internal") != "internal":
            raise ReaderError(f"{sniff.path.name}: external components not read")
        dtype = self.TYPES.get(variable.get("numeric_type", ""))
        if dtype is None:
            raise ReaderError(f"{sniff.path.name}: numeric type {variable.get('numeric_type')}")
        arrays = []
        for component in variable.get("components", []):
            if variable.get("encoding") == "base64":
                values = np.frombuffer(base64.b64decode(component), dtype=dtype)
            else:  # "none": the values as a list (complex as real, imaginary pairs)
                values = np.asarray(component, dtype=float)
                if np.dtype(dtype).kind == "c":
                    values = values[0::2] + 1j * values[1::2]
            if values.size != int(np.prod(counts)):
                raise ReaderError(f"{sniff.path.name}: {values.size} values for {counts} points")
            arrays.append(values.reshape(counts[::-1]).transpose())  # fastest dimension first
        if not arrays:
            raise ReaderError(f"{sniff.path.name}: a dependent variable without components")
        return np.stack(arrays, axis=-1)

    def _keys(self, meta: Meta, dimensions: list[dict], axes: list[np.ndarray]) -> None:
        increments = [float(a[1] - a[0]) if len(a) > 1 else 0.0 for a in axes]
        source = "CSDM dimensions"
        meta.set("NumberOfPoints", len(axes[0]), f"{source}: count", "dimensions/0/count")
        meta.set("SpectralWidth", increments[0] * len(axes[0]), f"{source}: increment x count",
                 "dimensions/0/increment")
        if len(axes) > 1:
            meta.set("IndirectSpectralWidth", [i * len(a) for i, a in zip(increments[1:],
                                                                          axes[1:], strict=True)],
                     f"{source}: increment x count of each indirect axis", "dimensions/1/increment")
        origins = [d.get("origin_offset") for d in dimensions]
        if all(origins):
            meta.set("SpectrometerFrequency",
                     [float(self.QUANTITY.match(o).group(1)) * self.HERTZ.get(
                         self.QUANTITY.match(o).group(2), 1.0) / 1e6 for o in origins],
                     f"{source}: origin_offset", "dimensions/0/origin_offset")
