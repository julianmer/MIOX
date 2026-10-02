####################################################################################################
#                                          vespa_viff.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for Vespa Interchange File Format datasets (.xml, .xml.gz). Arrays are stored as #
#          "xdr zlib base64" (big-endian) or "npy zlib base64" text; Vespa's own reader cannot run #
#          on current Python, so they are decoded here. The vendor header is the whole XML tree,   #
#          arrays summarised by shape.                                                             #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import base64
import gzip
import io
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import numpy as np

# own
from miox.data import MRSData
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                         VespaViffReader                                          #
#**************************************************************************************************#
#                                                                                                  #
# One MRSData per <dataset>, from its raw block: the data as acquired and the acquisition values   #
# (sweep width, frequency in MHz, TE in ms, flip angle). Vespa orders arrays (z, y, x, points).    #
# Experiment, prior and pulse-design files hold no measured data and are refused with a reason.    #
#                                                                                                  #
#**************************************************************************************************#
class VespaViffReader(Reader):
    FORMATS = (Format.VESPA_VIFF,)
    SOURCE = "VIFF raw block"
    RAW = "blocks/block_raw"

    def read(self, sniff: Sniff) -> list[MRSData]:
        root = ET.fromstring(self._text(sniff.path))
        datasets = root.findall("dataset")
        if not datasets:
            kinds = sorted({child.tag for child in root} - {"timestamp", "comment"})
            raise ReaderError(f"{sniff.path.name} holds no measured dataset, only {kinds}")
        names = [sniff.path.stem] if len(datasets) == 1 else [
            f"{sniff.path.stem}_{i}" for i in range(len(datasets))
        ]
        return [self._dataset(sniff, ds, name) for ds, name in zip(datasets, names, strict=True)]

    def _dataset(self, sniff: Sniff, element: ET.Element, name: str) -> MRSData:
        raw = element.find(self.RAW)
        if raw is None:
            raise ReaderError(f"{sniff.path.name}: dataset without a raw block")
        data = self.array(raw.find("data"))
        data = data.reshape(data.shape[:3][::-1] + data.shape[3:]).transpose(2, 1, 0, 3)
        text = {child.tag: (child.text or "").strip() for child in raw}
        field = f"dataset/{self.RAW}/"
        values = {
            "SpectrometerFrequency": ([float(text["frequency"])], field + "frequency"),
            "ResonantNucleus": ([text["nucleus"]], field + "nucleus"),
        }
        if text.get("resppm"):
            values["ReferencePPM"] = (float(text["resppm"]), field + "resppm")
        if text.get("seqte"):
            values["EchoTime"] = (float(text["seqte"]) * 1e-3, field + "seqte")
        if text.get("flip_angle"):
            values["ExcitationFlipAngle"] = (float(text["flip_angle"]), field + "flip_angle")
        notes = ["voxel geometry kept in the vendor header, not converted"]
        dwelltime = 1 / float(text["sweep_width"])
        vendor = {"dataset": self.tree(element)}
        d = self.dataset(sniff, data, dwelltime, values, vendor, name, notes, self.SOURCE)
        d.meta.set("DwellTime", dwelltime, f"{self.SOURCE}: 1 / sweep_width", field + "sweep_width")
        return d

    @staticmethod
    def _text(path: Path) -> bytes:
        raw = path.read_bytes()
        return gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw

    @staticmethod
    def array(element: ET.Element) -> np.ndarray:
        """Decode a VIFF array element (``encoding`` and ``data_type`` attributes)."""
        payload = zlib.decompress(base64.b64decode(element.text or ""))
        encoding = element.get("encoding", "")
        if encoding.startswith("npy"):
            return np.load(io.BytesIO(payload))
        kind = element.get("data_type", "float64")
        dtype = {"complex64": ">f4", "complex128": ">f8"}.get(kind, ">" + np.dtype(kind).str[1:])
        values = np.frombuffer(payload, dtype=dtype)
        if kind.startswith("complex"):
            values = values[0::2] + 1j * values[1::2]
        shape = tuple(int(n) for n in element.get("shape", str(values.size)).split(","))
        return values.reshape(shape)

    def tree(self, element: ET.Element) -> dict | str:
        """The element as nested dict; repeated tags become lists, arrays a shape summary."""
        if "encoding" in element.attrib:
            return f"<array {element.get('data_type')} {element.get('shape')}>"
        children = list(element)
        if not children:
            return (element.text or "").strip()
        out: dict = {}
        for child in children:
            value = self.tree(child)
            if child.tag not in out:
                out[child.tag] = value
            elif isinstance(out[child.tag], list):
                out[child.tag].append(value)
            else:
                out[child.tag] = [out[child.tag], value]
        return out
