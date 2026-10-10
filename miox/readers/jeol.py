####################################################################################################
#                                             jeol.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 10/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for JEOL Delta data files (.jdf): FIDs (1D and 2D) and processed 1D spectra,     #
#          with the acquisition parameters as standard keys. The binary layout is read by          #
#          nmrglue.                                                                                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from nmrglue.fileio import jeol as ng_jeol

# own
from miox.data import MRImage, MRSData
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                            JeolReader                                            #
#**************************************************************************************************#
#                                                                                                  #
# The data sections of a .jdf are the real and imaginary parts: a complex axis stores two          #
# sections (real, imaginary), two complex axes four. nmrglue combines them as real - i imaginary;  #
# the samples are used here as real + i imaginary, which puts the signals of the FIDs at their     #
# shifts in the NIfTI-MRS convention (checked against spectra processed by other software) and     #
# gives the processed spectra the rotation sense of a Fourier transform of such a FID. A 2D FID    #
# becomes rows on DIM_INDIRECT_0 as acquired: with two complex axes each increment gives an even   #
# row (section 0 + i section 1) and an odd row (section 2 + i section 3), each a direct-dimension  #
# FID in the same convention (the other pairing mirrors the direct axis); how the pair combines    #
# in the indirect dimension (States, echo-antiecho) is left to the experiment. With a real        #
# indirect axis each increment is one row. The time of each row comes from the header's axis       #
# range, which spans the points kept (data_offset_start to data_offset_stop). A processed          #
# spectrum is placed by the header's axis range in ppm in the same way.                            #
#                                                                                                  #
#**************************************************************************************************#
class JeolReader(Reader):
    FORMATS = (Format.JEOL_JDF,)
    PACKAGES = Reader.PACKAGES + ("nmrglue",)
    PREFIXES = {"none": 1.0, "milli": 1e-3, "micro": 1e-6, "nano": 1e-9, "kilo": 1e3}
    NUCLEI = {  # names JEOL gives the observed nucleus
        "proton": "1H", "deuterium": "2H", "tritium": "3H", "lithium7": "7Li", "boron10": "10B",
        "boron11": "11B", "carbon13": "13C", "nitrogen14": "14N", "nitrogen15": "15N",
        "oxygen17": "17O", "fluorine19": "19F", "sodium23": "23Na", "aluminum27": "27Al",
        "silicon29": "29Si", "phosphorus31": "31P", "chlorine35": "35Cl", "potassium39": "39K",
        "selenium77": "77Se", "tin119": "119Sn", "xenon129": "129Xe", "platinum195": "195Pt",
        "mercury199": "199Hg", "cadmium113": "113Cd", "silver109": "109Ag", "rhodium103": "103Rh",
    }
    NUCLEUS = re.compile(r"(\d+)([A-Za-z]+)")  # 1H, 13C

    def read(self, sniff: Sniff) -> list[MRSData | MRImage]:
        path = Path(sniff.path)
        buffer = path.read_bytes()
        try:
            dic = ng_jeol.parse_jeol(buffer)
            sections = self._sections(dic, buffer)
        except (ValueError, KeyError, IndexError, NotImplementedError) as e:
            raise ReaderError(f"{path}: not a readable JEOL data file ({e})") from e
        header = dic["header"]
        dimensions = ng_jeol.ndims(dic)
        if dimensions > 2:
            raise ReaderError(f"{path}: {dimensions} dimensions (no sample to check the layout)")
        notes = []
        domain = header["units"][0][2]
        if domain == "second":
            return [self._fid(sniff, dic, sections, notes)]
        if domain in ("ppm", "hertz") and dimensions == 1:
            return [self._spectrum(sniff, dic, sections, notes)]
        raise ReaderError(f"{path}: data in {domain!r} with {dimensions} dimension(s) is not "
                          "supported (no sample)")

    # --- binary layout ---------------------------------------------------------------------------

    def _sections(self, dic: dict, buffer: bytes) -> list[np.ndarray]:
        """The real and imaginary sections, each in (indirect, direct) order, cut to the points
        the header marks as kept."""
        header = dic["header"]
        raw = ng_jeol.read_bin_data(dic, buffer)
        shape = header["data_points"][:2][::-1]
        edge = ng_jeol.submatrix_shape(dic)
        parts = [ng_jeol.reorder_submatrix(data=s, shape=shape, submatrix_shape=edge)
                 for s in ng_jeol.split_sections(dic, raw)[1]]
        cut = tuple(slice(header["data_offset_start"][i], header["data_offset_stop"][i] + 1)
                    for i in range(ng_jeol.ndims(dic)))[::-1]
        return [np.asarray(part)[cut] for part in parts]

    # --- FID -------------------------------------------------------------------------------------

    def _fid(self, sniff: Sniff, dic: dict, sections: list[np.ndarray],
             notes: list[str]) -> MRSData:
        header, parameters = dic["header"], dic["parameters"]
        kinds = [k for k in header["data_axis_type"] if k]
        if len(kinds) == 1:
            data = self._complex(sections, kinds[0], notes)
            tags = ()
        else:
            data, tags = self._rows(sections, kinds, header, notes)
        dwelltime = self._step(header, 0, data.shape[-1])
        data = data.T.reshape((1, 1, 1) + data.T.shape)
        values = self._keys(header, parameters, len(kinds))
        if parameters.get("digital_filter") == "TRUE":
            notes.append("digital filter: the first points hold the filter's ramp; the file "
                         "states no group delay")
        name = Path(sniff.path).stem
        d = self.dataset(sniff, data, dwelltime, values, self._vendor(dic), name, notes,
                         "JEOL parameters", dim_tags=tags)
        if dwelltime:
            d.meta.set("DwellTime", dwelltime, "JEOL header: data axis range of the direct axis",
                       "data_axis_stop")
        return d

    def _complex(self, sections: list[np.ndarray], kind: str, notes: list[str]) -> np.ndarray:
        """Samples of one axis: real + i imaginary for a complex axis, the real values otherwise."""
        if kind == "complex" and len(sections) == 2:
            return sections[0] + 1j * sections[1]
        if kind == "real" and len(sections) == 1:
            notes.append("real samples only: stored as the real part")
            return sections[0].astype(np.complex128)
        raise ReaderError(f"axis type {kind!r} with {len(sections)} sections is not supported")

    def _rows(self, sections: list[np.ndarray], kinds: list[str], header: dict,
              notes: list[str]) -> tuple[np.ndarray, tuple]:
        """The rows of a 2D FID (indirect, direct) and the indirect dimension's tag."""
        n = header["data_offset_stop"][1] - header["data_offset_start"][1] + 1
        times = np.linspace(header["data_axis_start"][1], header["data_axis_stop"][1], n)
        times = times * self._scale(header, 1)
        if kinds == ["complex", "complex"] and len(sections) == 4:
            rows = np.empty((2 * n, sections[0].shape[1]), np.complex128)
            rows[0::2] = sections[0] + 1j * sections[1]
            rows[1::2] = sections[2] + 1j * sections[3]
            times = np.repeat(times, 2)
            info = ("two rows per increment (file sections 0 + i 1, then 2 + i 3), the "
                    "indirect combination depends on the experiment; time per row")
        elif kinds == ["real_complex", "real_complex"] and len(sections) == 2:
            rows = sections[0] + 1j * sections[1]
            info = "real indirect axis (magnitude mode): one row per increment; time per row"
        else:
            raise ReaderError(f"axis types {kinds} with {len(sections)} sections are not "
                              "supported")
        head = {"IndirectTime": {"Value": times.tolist(), "Description": "evolution time in s"}}
        return rows, (("DIM_INDIRECT_0", info, head),)

    # --- spectrum --------------------------------------------------------------------------------

    def _spectrum(self, sniff: Sniff, dic: dict, sections: list[np.ndarray],
                  notes: list[str]) -> MRImage:
        header, parameters = dic["header"], dic["parameters"]
        kind = header["data_axis_type"][0]
        data = self._complex(sections, kind, notes)
        if kind == "real":
            data = data.real
        n = data.size
        unit = header["units"][0][2]
        frequency = float(parameters["x_freq"]) / 1e6
        start, stop = header["data_axis_start"][0], header["data_axis_stop"][0]
        scale = self._scale(header, 0) / (frequency if unit == "hertz" else 1.0)
        ppm = np.linspace(start, stop, n) * scale
        if ppm[0] < ppm[-1]:  # the shift falls with the index
            ppm, data = ppm[::-1], data[::-1]
        step = float(abs(ppm[1] - ppm[0]))
        meta = Meta(self._vendor(dic))
        source = "JEOL parameters"
        for key, (value, field) in self._keys(header, parameters, 1).items():
            meta.set(key, value, f"{source}: {field}", field)
        meta.set("NumberOfPoints", int(n), "data shape")
        meta.set("SpectralWidth", step * frequency * n, "JEOL header: data axis range * points",
                 "data_axis_stop")
        meta.set("DwellTime", 1 / (step * frequency * n), "derived: 1 / SpectralWidth")
        meta.set("ReferencePPM", float(ppm[n // 2]), "JEOL header: shift of point N / 2",
                 "data_axis_start")
        return MRImage(data.reshape(1, 1, 1, -1), self.NO_GEOMETRY, meta,
                       self.provenance(sniff, tuple(notes)), Path(sniff.path).stem,
                       ("frequency",))

    # --- header ----------------------------------------------------------------------------------

    def _scale(self, header: dict, axis: int) -> float:
        prefix = header["units"][axis][0]
        if prefix not in self.PREFIXES:
            raise ReaderError(f"axis unit prefix {prefix!r} is not supported")
        return self.PREFIXES[prefix] ** header["units"][axis][1]

    def _step(self, header: dict, axis: int, n: int) -> float | None:
        """Seconds between the points of a time axis, from the header's axis range."""
        if n < 2:
            return None
        span = header["data_axis_stop"][axis] - header["data_axis_start"][axis]
        return span * self._scale(header, axis) / (n - 1)

    def _keys(self, header: dict, parameters: dict, dimensions: int) -> dict[str, tuple]:
        """Standard keys; frequencies and nuclei of the first ``dimensions`` axes (x, y)."""
        values: dict[str, tuple] = {"Manufacturer": ("JEOL", "file format")}
        axes = "xy"[:dimensions]
        nuclei = [self._nucleus(parameters.get(f"{a}_domain")) for a in axes]
        if all(f"{a}_freq" in parameters for a in axes):
            values["SpectrometerFrequency"] = (
                [float(parameters[f"{a}_freq"]) / 1e6 for a in axes], "x_freq")
        if None not in nuclei:
            values["ResonantNucleus"] = (nuclei, "x_domain")
        if "x_offset" in parameters:
            values["ReferencePPM"] = (float(parameters["x_offset"]), "x_offset")
        for key, field, convert in (
            ("NumberOfAverages", "scans", int),
            ("SampleTemperature", "temp_get", lambda c: float(c) + 273.15),
            ("Solvent", "solvent", str),
            ("SequenceName", "experiment", str),
            ("FieldStrength", "field_strength", float),
            ("ManufacturersModelName", "inst_model_number", str),
            ("DeviceSerialNumber", "inst_serial_number", str),
        ):
            if parameters.get(field) not in (None, ""):
                values[key] = (convert(parameters[field]), field)
        return values

    def _nucleus(self, name: Any) -> str | None:
        """A nucleus as NIfTI-MRS spells it (1H, 13C) from JEOL's Proton, Carbon13 or 13C."""
        if not name:
            return None
        text = str(name)
        if match := self.NUCLEUS.fullmatch(text):
            return match.group(1) + match.group(2)
        return self.NUCLEI.get(text.lower())

    def _vendor(self, dic: dict) -> dict[str, Any]:
        parameters = {k: v for k, v in dic["parameters"].items() if k != "info"}
        return {"header": self.plain(dic["header"]), "parameters": self.plain(parameters)}
