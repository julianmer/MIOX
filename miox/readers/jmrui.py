####################################################################################################
#                                             jmrui.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for jMRUI files: text (.txt) and binary (.mrui). The vendor header is every key  #
#          of the text header (also keys with spaces), or the numeric header and the trailing text #
#          of a .mrui file.                                                                        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# own
from miox import __version__
from miox.data import BasisSet, MRSData
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff, Sniffer


#**************************************************************************************************#
#                                           JmruiReader                                            #
#**************************************************************************************************#
#                                                                                                  #
# Reads one jMRUI file. Text: ``Key: value`` lines (keys of word characters for the standard       #
# keys), then per signal a ``Signal n out of N in file`` line and lines of real and imaginary      #
# parts. Binary: 13 big-endian doubles (512 bytes reserved), the signals as complex big-endian     #
# doubles, then text. FIDs are conjugated to NIfTI-MRS's convention; several signals in one file   #
# become a DIM_USER_0 dimension. The nucleus is TypeOfNucleus (a name, or an index: 1 1H, 2 31P, 3 #
# 13C, 4 19F, 5 23Na), else the one whose gyromagnetic ratio is closest to frequency / field and   #
# within 2 % of it, else 1H; a frequency above 100 GHz is in Hz. No geometry: a 10 m voxel. A text #
# file whose SignalNames give one name per signal, most of them metabolites, is a BasisSet.        #
#                                                                                                  #
#**************************************************************************************************#
class JmruiReader(Reader):
    FORMATS = (Format.JMRUI_TEXT, Format.JMRUI_BINARY)
    SOURCE = "jMRUI header"
    SIGNAL = re.compile(r"Signal (?:number: )?\d+ out of \d+ in file")  # start of the data
    STANDARD_LINE = re.compile(r"(\w*):(.*)")  # a header line whose key holds no spaces
    EXTENSION = re.compile(r"\.txt$", re.I)  # on signal names
    NUCLEUS_INDEX = [None, "1H", "31P", "13C", "19F", "23Na"]
    GYROMAGNETIC = {"1H": 42.577, "2H": 6.536, "3H": 45.415, "13C": 10.7084, "23Na": 11.262,
                    "31P": 17.235, "129Xe": -11.777}  # MHz/T
    GYROMAGNETIC_TOLERANCE = 0.02  # relative: 13C, 23Na and 129Xe lie 5 % apart
    BINARY_FIELDS = ("type_of_sig", "number_of_points", "sampling_interval", "begin_time ",
                     "zero_order_phs", "transmitter_frequency", "magnetic_field",
                     "type_of_nucleus", "reference_frequency_hz", "reference_frequency_ppm",
                     "fid_or_echo", "apodizing", "num_zeros_view")
    BINARY_DATA = 512  # bytes before the signals

    def read(self, sniff: Sniff) -> list[MRSData]:
        notes: list[str] = []
        if sniff.format == Format.JMRUI_TEXT:
            data, header, values, info = self._text(sniff.path, notes)
            vendor = self._text_header(sniff.path)
        else:
            data, header, values, info = self._binary(sniff.path, notes)
            vendor = dict(header, text=values["AdditionalInformation"][0]["Value"])
        values["OriginalFile"] = ([sniff.path.name], None)
        values["ConversionMethod"] = (f"miox {__version__}", None)
        values["ConversionTime"] = (datetime.now().isoformat(timespec="milliseconds"), None)
        tags = (("DIM_USER_0", info, None),) if data.ndim > 1 else ()
        data = data.reshape((1, 1, 1) + data.shape)
        data, tags = self.without_zero_indices(data, tags, notes)
        vendor = {k: self.plain(v) for k, v in vendor.items()}
        d = self.dataset(sniff, data, values["DwellTime"][0], values, vendor, sniff.path.stem,
                         notes, self.SOURCE, None, tags)
        return [self._basis(d)]

    # --- text ------------------------------------------------------------------------------------

    def _text(self, path: Path, notes: list[str]) -> tuple:
        header, samples, signals, recording = {}, [], 0, False
        with open(path) as fh:
            for line in fh:
                match = self.STANDARD_LINE.match(line)
                if match:
                    header[match[1]] = self._number(match[2].strip())
                if self.SIGNAL.match(line):
                    recording, signals = True, signals + 1
                    continue
                if recording:
                    samples += [float(v) for v in line.split()[:2]]
        if not signals:
            raise ReaderError(f"{path.name}: no signal in the file")
        floats = np.array(samples)
        data = (floats[0::2] + 1j * floats[1::2]).astype(complex)
        data = data.reshape((signals, -1)).T.squeeze().conj()
        frequency = header["TransmitterFrequency"]
        if frequency > 1e5:
            notes.append(f"TransmitterFrequency {frequency} is in Hz; divided by 1e6")
            frequency = frequency / 1e6
        nucleus, field = self._nucleus(header.get("TypeOfNucleus"), frequency,
                                       header["MagneticField"], notes)
        values = {
            "SpectrometerFrequency": ([float(frequency)], "TransmitterFrequency"),
            "ResonantNucleus": ([nucleus], field),
            "DwellTime": (header["SamplingInterval"] * 1e-3, "SamplingInterval"),
        }
        for key, name in (("ManufacturersModelName", "Spectrometer"),
                          ("PatientName", "NameOfPatient")):
            if name in header:
                values[key] = (header[name], name)
        for name, doc in (("AdditionalInformation", "jMRUI AdditionalInformation field."),
                          ("SignalNames", "jMRUI SignalNames field.")):
            if name in header:
                values[name] = ({"Value": header[name], "Description": doc}, name)
        return data, header, values, "jMRUI signals"

    def _text_header(self, path: Path) -> dict:
        """Every ``key: value`` line before the first signal; keys may contain spaces."""
        header = {}
        with open(path, errors="replace") as fh:
            for line in fh:
                if self.SIGNAL.match(line):
                    break
                key, colon, value = line.partition(":")
                if colon and key.strip():
                    header[key.strip()] = self._number(value.strip())
        return header

    # --- binary ----------------------------------------------------------------------------------

    def _binary(self, path: Path, notes: list[str]) -> tuple:
        raw = path.read_bytes()
        numbers = np.frombuffer(raw[: 13 * 8], dtype=">f8")
        header = dict(zip(self.BINARY_FIELDS, numbers.tolist(), strict=True))
        header["number_of_points"] = points = int(header["number_of_points"])
        frames = int(np.floor((len(raw) - self.BINARY_DATA) / (8 * points * 2)))
        end = self.BINARY_DATA + frames * points * 2 * 8
        floats = np.frombuffer(raw[self.BINARY_DATA:end], dtype=">f8")
        data = (floats[0::2] + 1j * floats[1::2]).reshape((frames, points)).T.squeeze().conj()
        text = raw[end:].decode("utf-8")
        frequency = header["transmitter_frequency"]
        if frequency > 1e5:
            notes.append(f"transmitter_frequency {frequency} is in Hz; divided by 1e6")
            frequency = frequency / 1e6
        nucleus, field = self._nucleus(header["type_of_nucleus"], header["transmitter_frequency"],
                                       header["magnetic_field"], notes)
        values = {
            "SpectrometerFrequency": ([frequency], "transmitter_frequency"),
            "ResonantNucleus": ([nucleus], field),
            "TxOffset": (header["reference_frequency_ppm"], "reference_frequency_ppm"),
            "DwellTime": (header["sampling_interval"] * 1e-3, "sampling_interval"),
            "AdditionalInformation": ({"Value": text, "Description": "jMRUI .mrui string field."},
                                      "text"),
        }
        return data, header, values, "jMRUI frames"

    # --- keys ------------------------------------------------------------------------------------

    def _nucleus(self, stated: Any, frequency: float, field: float,
                 notes: list[str]) -> tuple[str, str]:
        """The nucleus and the header field it comes from."""
        if isinstance(stated, str):
            return stated, "TypeOfNucleus"
        if isinstance(stated, (int, float)) and 0 <= int(stated) < len(self.NUCLEUS_INDEX) and \
                self.NUCLEUS_INDEX[int(stated)] is not None:
            return self.NUCLEUS_INDEX[int(stated)], "TypeOfNucleus"
        try:
            ratio = frequency / field
        except ZeroDivisionError:
            notes.append("nucleus not stated and no field strength: 1H assumed")
            return "1H", "MagneticField"
        ratio = ratio / 1e6 if ratio > 1e4 else ratio
        nucleus = min(self.GYROMAGNETIC, key=lambda n: abs(abs(self.GYROMAGNETIC[n]) - ratio))
        gamma = abs(self.GYROMAGNETIC[nucleus])
        if abs(gamma - ratio) <= self.GYROMAGNETIC_TOLERANCE * gamma:
            return nucleus, "MagneticField"
        notes.append("nucleus not identified from frequency / field: 1H assumed")
        return "1H", "MagneticField"

    @staticmethod
    def _number(text: str) -> int | float | str:
        for kind in (int, float):
            try:
                return kind(text)
            except ValueError:
                pass
        return text

    def _basis(self, d: MRSData) -> MRSData:
        """The dataset as a BasisSet when its signals are named after metabolites."""
        names = str(d.meta.vendor.get("SignalNames", "")).split(";")
        names = [self.EXTENSION.sub("", name.strip()) for name in names if name.strip()]
        if d.data.ndim != 5 or d.shape[4] != len(names):
            return d
        if sum(Sniffer.is_metabolite(name) for name in names) <= len(names) / 2:
            return d
        info = {**d.dim_info, 5: {"info": "basis set metabolites", "header": {"Metabolite": names}}}
        return BasisSet(
            d.data, d.dwelltime, d.affine, d.meta, d.provenance, d.dim_tags, info, d.name
        )
