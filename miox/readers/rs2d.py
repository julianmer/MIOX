####################################################################################################
#                                             rs2d.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 10/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for RS2D (Cameleon / Spinlab) NMR data: an experiment folder (header.xml and     #
#          data.dat) gives its FID and every processed spectrum in Proc/<n>.                       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports                                                                                        #
#*************#
from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np

# own
from miox.data import MRImage, MRSData
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                            Rs2dReader                                            #
#**************************************************************************************************#
#                                                                                                  #
# header.xml lists parameters (key, value or values, numberEnum); data.dat holds big-endian        #
# complex64 values (float32 when the first DATA_REPRESENTATION is REAL) of shape (receivers,       #
# MATRIX_DIMENSION_4D, 3D, 2D, 1D). FIDs are stored with lines of positive shift at positive       #
# frequencies of a transform: they are conjugated into the NIfTI-MRS convention (the solvent       #
# lines of the samples then lie at their shifts and on the vendor's processed spectrum). A         #
# processed spectrum (a header with STATE, in Proc/<n>) runs with the shift falling, its centre    #
# point at the carrier, and is kept as stored. The carrier lies at (OBSERVED_FREQUENCY -           #
# BASE_FREQ_1) / BASE_FREQ_1, OFFSET_FREQ_1 in ppm; SPECTRAL_WIDTH is in ppm of the observed       #
# frequency. DIGITAL_FILTER_SHIFT is kept in the vendor header only: the samples' filter ramp does #
# not peak where a group delay of that many points would put it.                                   #
#                                                                                                  #
#**************************************************************************************************#
class Rs2dReader(Reader):
    FORMATS = (Format.RS2D,)
    NUCLEUS = re.compile(r"(\d+)([A-Za-z]+)")
    INDIRECT = ("DIM_INDIRECT_0", "DIM_INDIRECT_1", "DIM_INDIRECT_2")
    DIMENSIONS = ("MATRIX_DIMENSION_4D", "MATRIX_DIMENSION_3D", "MATRIX_DIMENSION_2D",
                  "MATRIX_DIMENSION_1D")

    def read(self, sniff: Sniff) -> list[MRSData | MRImage]:
        folder = sniff.path if sniff.path.is_dir() else sniff.path.parent
        header = self.header(folder / "header.xml")
        if "STATE" in header:
            return [self._spectrum(sniff, folder, header)]
        datasets: list[MRSData | MRImage] = [self._fid(sniff, folder, header)]
        processed = folder / "Proc"
        if processed.is_dir():
            for sub in sorted(processed.iterdir(), key=lambda p: p.name):
                if (sub / "header.xml").is_file() and (sub / "data.dat").is_file():
                    datasets.append(self._spectrum(sniff, sub, self.header(sub / "header.xml")))
        return datasets

    @staticmethod
    def header(path: Path) -> dict[str, dict[str, Any]]:
        """header.xml as {key: {"value": text or list of texts, "numberEnum": ..., ...}}."""
        try:
            root = ElementTree.parse(path).getroot()
        except (ElementTree.ParseError, OSError) as e:
            raise ReaderError(f"{path}: not a readable RS2D header ({e})") from e
        out = {}
        for entry in root.iter("entry"):
            key, value = entry.find("key"), entry.find("value")
            if key is None or value is None or key.text is None:
                continue
            fields = {}
            values = [v.text for v in value.findall("value")]
            fields["value"] = values[0] if len(values) == 1 else values
            for child in value:
                if child.tag not in ("value", "uuid") and child.text and child.text.strip():
                    fields[child.tag] = child.text.strip()
            out[key.text] = fields
        return out

    def _data(self, folder: Path, header: dict) -> np.ndarray:
        shape = [int(self._first(header, "RECEIVER_COUNT", 1))] + [
            int(self._first(header, k, 1)) for k in self.DIMENSIONS]
        real = self._first(header, "DATA_REPRESENTATION", "COMPLEX") == "REAL"
        data = np.fromfile(folder / "data.dat", dtype=">f4" if real else ">c8")
        if data.size != int(np.prod(shape)):
            raise ReaderError(f"{folder}: {data.size} values, header states {shape}")
        return data.reshape(shape).astype(np.float64 if real else np.complex128)

    # --- FID -------------------------------------------------------------------------------------

    def _fid(self, sniff: Sniff, folder: Path, header: dict) -> MRSData:
        data = np.transpose(np.conj(self._data(folder, header)), (4, 3, 2, 1, 0))
        sizes = data.shape  # (1D, 2D, 3D, 4D, receivers)
        samples = data.reshape([s for i, s in enumerate(sizes) if i == 0 or s > 1])
        mode = self._list(header, "ACQUISITION_MODE")
        tags: list = []
        for n in (2, 3, 4):
            if sizes[n - 1] > 1:
                info = (f"{n}D as stored: {self._first(header, f'NUCLEUS_{n}', '?')}, mode "
                        f"{mode[n - 1] if len(mode) >= n else '?'}")
                tags.append((self.INDIRECT[len(tags)], info, None))
        indirect = len(tags)
        if sizes[4] > 1:
            tags.append(("DIM_COIL", None, None))
        if len(tags) > 3:
            raise ReaderError(f"{folder}: {len(tags)} dimensions beyond the first")
        values = self._keys(header, indirect)
        dwelltime = float(self._first(header, "DWELL_TIME"))
        name = f"{folder.parent.name}_{folder.name}"
        d = self.dataset(sniff, samples.reshape((1, 1, 1) + samples.shape), dwelltime, values,
                         self._vendor(header), name, [], "RS2D header", dim_tags=tuple(tags))
        d.meta.set("DwellTime", dwelltime, "RS2D header: DWELL_TIME", "DWELL_TIME")
        return d

    # --- spectra ---------------------------------------------------------------------------------

    def _spectrum(self, sniff: Sniff, folder: Path, header: dict) -> MRImage:
        data = self._data(folder, header)
        if data.shape[0] > 1:
            raise ReaderError(f"{folder}: processed data of several receivers (no sample)")
        values = np.transpose(data[0], (3, 2, 1, 0))  # (1D, 2D, 3D, 4D)
        values = values.reshape([s for i, s in enumerate(values.shape) if i == 0 or s > 1])
        meta = Meta(self._vendor(header))
        for key, (value, field) in self._keys(header, values.ndim - 1).items():
            meta.set(key, value, f"RS2D header: {field}", field)
        n = values.shape[0]
        frequency = float(self._first(header, "OBSERVED_FREQUENCY")) / 1e6
        width = float(self._first(header, "SPECTRAL_WIDTH")) * frequency
        meta.set("NumberOfPoints", int(n), "data shape")
        meta.set("SpectralWidth", width, "RS2D header: SPECTRAL_WIDTH (ppm) * observed frequency",
                 "SPECTRAL_WIDTH")
        meta.set("DwellTime", 1 / width, "derived: 1 / SpectralWidth")
        names = ("frequency",) + tuple(f"frequency_indirect_{i}" for i in range(values.ndim - 1))
        notes = ("indirect axes in the vendor header",) if values.ndim > 1 else ()
        experiment = folder.parent.parent
        name = f"{experiment.parent.name}_{experiment.name}_proc{folder.name}"
        return MRImage(values.reshape((1, 1, 1) + values.shape), self.NO_GEOMETRY, meta,
                       self.provenance(sniff, notes), name, names)

    # --- header ----------------------------------------------------------------------------------

    def _keys(self, header: dict, indirect: int) -> dict[str, tuple]:
        observed = float(self._first(header, "OBSERVED_FREQUENCY")) / 1e6
        base = float(self._first(header, "BASE_FREQ_1")) / 1e6
        values: dict[str, tuple] = {
            "Manufacturer": ("RS2D", "file format"),
            "SpectrometerFrequency": ([observed], "OBSERVED_FREQUENCY"),
            "ReferencePPM": ((observed - base) / base * 1e6,
                             "(OBSERVED_FREQUENCY - BASE_FREQ_1) / BASE_FREQ_1"),
        }
        nuclei = [self._nucleus(self._first(header, f"NUCLEUS_{n}", "")) for n in
                  range(1, indirect + 2)]
        if None not in nuclei:
            values["ResonantNucleus"] = (nuclei, "NUCLEUS_1")
        for key, field, convert in (
            ("NumberOfAverages", "NUMBER_OF_AVERAGES", int),
            ("Solvent", "SOLVENT", str),
            ("SequenceName", "SEQUENCE_NAME", str),
            ("AcquisitionDateTime", "ACQUISITION_DATE", str),
        ):
            text = self._first(header, field, None)
            if text not in (None, ""):
                values[key] = (convert(text), field)
        return values

    def _nucleus(self, text: str) -> str | None:
        match = self.NUCLEUS.fullmatch(str(text).strip())
        return match.group(1) + match.group(2) if match else None

    @staticmethod
    def _first(header: dict, key: str, default: Any = ...) -> Any:
        if key not in header:
            if default is ...:
                raise ReaderError(f"RS2D header without {key}")
            return default
        value = header[key]["value"]
        return value[0] if isinstance(value, list) else value

    @staticmethod
    def _list(header: dict, key: str) -> list[str]:
        value = header.get(key, {}).get("value", [])
        return value if isinstance(value, list) else [value]

    def _vendor(self, header: dict) -> dict[str, Any]:
        return {k: self.plain(v) for k, v in header.items()}
