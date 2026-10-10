####################################################################################################
#                                            nmrml.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 09/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for nmrML files (XML, version 1.0 and its release candidates): the FID of the    #
#          1D acquisition and the processed 1D spectra, decoded from base64 (zlib-compressed or    #
#          not), with the acquisition parameters as standard keys.                                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import base64
import zlib
from typing import Any
from xml.etree import ElementTree

import numpy as np

# own
from miox.data import MRImage, MRSData
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                           NmrmlReader                                            #
#**************************************************************************************************#
#                                                                                                  #
# nmrML keeps the samples of the vendor file it was converted from: Bruker FIDs are conjugated     #
# into the NIfTI-MRS convention as TopSpin's are, Varian FIDs are in it already; FIDs of other     #
# sources are kept as stored, with a note. The carrier is irradiationFrequency; the converters     #
# write the frequency of 0 ppm (Bruker SF, Varian reffrq) as effectiveExcitationField, so the      #
# carrier lies at (irradiation - effective) / effective when both are frequencies. A spectrum's    #
# xAxis runs from the shift of its first point (startValue) to the edge one spectral width further #
# (endValue); it is stored with the shift falling with the index, as TopSpin stores spectra.       #
#                                                                                                  #
#**************************************************************************************************#
class NmrmlReader(Reader):
    FORMATS = (Format.NMRML,)
    DTYPES = {
        "complex128": "<c16", "complex64": "<c8", "float64": "<f8", "float32": "<f4",
        "complex64int": "<i4", "class java.lang.integer": "<i4", "integer32": "<i4",  # int32 pairs
    }
    NUCLEI = {  # ChEBI names of the observed nucleus
        "hydrogen atom": "1H", "proton": "1H", "carbon-13 atom": "13C", "nitrogen-15 atom": "15N",
        "fluorine-19 atom": "19F", "phosphorus-31 atom": "31P", "deuterium atom": "2H",
    }
    UNITS = {"megahertz": 1.0, "hertz": 1e-6, "kilohertz": 1e-3}  # to MHz

    def read(self, sniff: Sniff) -> list[MRSData | MRImage]:
        try:
            root = ElementTree.parse(sniff.path).getroot()
        except ElementTree.ParseError as e:
            raise ReaderError(f"{sniff.path}: not well-formed XML ({e})") from e
        self.ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
        vendor = self._tree(root)
        datasets = []
        fid = self._find(root, "acquisition1D/fidData")
        if fid is not None and (fid.text or "").strip():
            datasets.append(self._fid(sniff, root, fid, vendor))
        for spectrum in root.iter(self.ns + "spectrum1D"):
            data = self._find(spectrum, "spectrumDataArray")
            if data is not None and (data.text or "").strip():
                datasets.append(self._spectrum(sniff, root, spectrum, vendor, len(datasets)))
        if not datasets:
            raise ReaderError(f"{sniff.path}: no FID or spectrum data")
        return datasets

    # --- FID -------------------------------------------------------------------------------------

    def _fid(self, sniff: Sniff, root: ElementTree.Element, element: ElementTree.Element,
             vendor: dict) -> MRSData:
        notes = []
        values = self._decode(element)
        data = values.astype(np.complex128) if values.dtype.kind == "c" else (
            values[0::2] + 1j * values[1::2])
        source = self._source(root)
        if source == "bruker":
            data = np.conj(data)
            notes.append("0 ppm lies at the nominal frequency (BF1): the spectrometer's "
                         "calibrated SF is not stored in nmrML")
        elif source != "varian":
            notes.append("source format not stated as Bruker or Varian: FID kept as stored, its "
                         "sign convention not verified")
        keys, width = self._acquisition(root)
        name = sniff.path.name.removesuffix(".nmrML").removesuffix(".nmrml")
        d = self.dataset(sniff, data.reshape(1, 1, 1, -1), width and 1 / width, keys, vendor, name,
                         notes, "nmrML")
        if width:
            d.meta.set("DwellTime", 1 / width, "nmrML: 1 / sweepWidth", "sweepWidth")
        return d

    def _acquisition(self, root: ElementTree.Element) -> tuple[dict[str, tuple], float | None]:
        """Standard keys of the acquisition parameter set, and the sweep width in Hz."""
        parameters = self._find(root, "acquisition1D/acquisitionParameterSet")
        direct = self._find(parameters, "DirectDimensionParameterSet") if parameters is not None \
            else None
        if direct is None:
            return {}, None
        keys = {}
        carrier = self._frequency(self._find(direct, "irradiationFrequency"))
        zero = self._frequency(self._find(direct, "effectiveExcitationField"))
        if carrier:
            keys["SpectrometerFrequency"] = ([carrier], "irradiationFrequency")
            if zero and abs(carrier - zero) < 1e-3 * zero:
                keys["ReferencePPM"] = ((carrier - zero) / zero * 1e6,
                                        "(irradiationFrequency - effectiveExcitationField) / "
                                        "effectiveExcitationField")
        nucleus = self._find(direct, "acquisitionNucleus")
        if nucleus is not None and nucleus.get("name", "").lower() in self.NUCLEI:
            keys["ResonantNucleus"] = ([self.NUCLEI[nucleus.get("name").lower()]],
                                       "acquisitionNucleus")
        if parameters.get("numberOfScans"):
            keys["NumberOfAverages"] = (int(parameters.get("numberOfScans")), "numberOfScans")
        temperature = self._find(parameters, "sampleAcquisitionTemperature")
        if temperature is not None and temperature.get("unitName", "").lower() == "kelvin":
            keys["SampleTemperature"] = (float(temperature.get("value")),
                                         "sampleAcquisitionTemperature")
        delay = self._find(parameters, "groupDelay")
        if delay is not None and float(delay.get("value", 0) or 0) > 0:
            keys["GroupDelay"] = (float(delay.get("value")), "groupDelay")
        for param in root.iter(self.ns + "userParam"):
            name, value = param.get("name"), (param.get("value") or "").strip()
            if name == "Pulse Program" and value:
                keys["SequenceName"] = (value, "pulseSequence/Pulse Program")
            elif name == "ProbeHead" and value:
                keys["TxCoil"] = keys["RxCoil"] = (value, "instrumentConfiguration/ProbeHead")
        width = self._find(direct, "sweepWidth")
        return keys, float(width.get("value")) if width is not None else None

    # --- spectra ---------------------------------------------------------------------------------

    def _spectrum(self, sniff: Sniff, root: ElementTree.Element, spectrum: ElementTree.Element,
                  vendor: dict, index: int) -> MRImage:
        values = self._decode(self._find(spectrum, "spectrumDataArray"))
        values = values.astype(np.complex128 if values.dtype.kind == "c" else np.float64)
        axis = self._find(spectrum, "xAxis")
        n = values.size
        keys, _ = self._acquisition(root)
        meta = Meta(vendor)
        frequency = keys.get("SpectrometerFrequency", ([None],))[0][0]
        for key in ("SpectrometerFrequency", "ResonantNucleus", "SequenceName"):
            if key in keys:
                meta.set(key, keys[key][0], f"nmrML: {keys[key][1]}", keys[key][1])
        meta.set("NumberOfPoints", int(n), "data shape")
        notes = []
        if axis is not None and "million" in axis.get("unitName", ""):
            start, end = float(axis.get("startValue")), float(axis.get("endValue"))
            if start < end:  # the shift falls with the index
                values, start, end = values[::-1], end, start
            step = (start - end) / n
            meta.set("ReferencePPM", start - step * (n // 2), "nmrML: xAxis startValue, endValue")
            if frequency:
                meta.set("SpectralWidth", (start - end) * frequency,
                         "nmrML: xAxis span * frequency")
                meta.set("DwellTime", 1 / ((start - end) * frequency),
                         "derived: 1 / SpectralWidth")
        else:
            notes.append("no chemical shift xAxis: the spectrum carries no axis")
        stem = sniff.path.name.removesuffix(".nmrML").removesuffix(".nmrml")
        name = f"{stem}_spectrum{spectrum.get('name') or index}"
        data = values.reshape(1, 1, 1, -1)
        return MRImage(data, self.NO_GEOMETRY, meta, self.provenance(sniff, tuple(notes)), name,
                       ("frequency",))

    # --- XML -------------------------------------------------------------------------------------

    def _decode(self, element: ElementTree.Element) -> np.ndarray:
        raw = base64.b64decode(element.text)
        if element.get("compressed", "").lower() == "true":
            raw = zlib.decompress(raw)
        dtype = self.DTYPES.get(element.get("byteFormat", "").lower())
        if dtype is None:
            raise ReaderError(f"unknown byteFormat {element.get('byteFormat')!r}")
        return np.frombuffer(raw, dtype)

    def _source(self, root: ElementTree.Element) -> str | None:
        """The vendor format of the source files: bruker, varian or None."""
        names = " ".join(p.get("name", "") for f in root.iter(self.ns + "sourceFile")
                         for p in f.iter(self.ns + "cvParam")).lower()
        for vendor in ("bruker", "varian"):
            if vendor in names or (vendor == "varian" and "agilent" in names):
                return vendor
        return None

    def _frequency(self, element: ElementTree.Element | None) -> float | None:
        """A frequency in MHz, from its value and unit name; None for other units."""
        if element is None:
            return None
        factor = self.UNITS.get(element.get("unitName", "").lower())
        if factor is None:
            return None
        value = float(element.get("value")) * factor
        return value * 1e6 if factor == 1e-6 and value < 0.01 else value  # drafts: MHz as hertz

    def _find(self, element: ElementTree.Element, path: str) -> ElementTree.Element | None:
        return element.find(".//" + "/".join(self.ns + p for p in path.split("/")))

    def _tree(self, element: ElementTree.Element) -> dict[str, Any]:
        """The XML as nested dicts (attributes, children by tag, lists when repeated), without
        the encoded data arrays."""
        node: dict[str, Any] = dict(element.attrib)
        for child in element:
            tag = child.tag.removeprefix(self.ns)
            if tag in ("fidData", "spectrumDataArray"):
                node[tag] = dict(child.attrib)
                continue
            value = self._tree(child)
            if tag in node:
                node[tag] = node[tag] if isinstance(node[tag], list) else [node[tag]]
                node[tag].append(value)
            else:
                node[tag] = value
        text = (element.text or "").strip()
        if text and len(node) == 0:
            return {"text": text}
        return node
