####################################################################################################
#                                       philips_dicom_raw.py                                       #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 30/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for Philips DICOM Raw Data Storage: the scan parameters the scanner stores       #
#          beside its images and spectra, or a whole ExamCard.                                     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import base64
import re
import struct
import xml.etree.ElementTree as ET
from typing import Any

import pydicom

# own
from miox.data import Protocol
from miox.meta import Meta
from miox.readers.base import ReaderError
from miox.readers.philips_examcard import PhilipsExamCardReader
from miox.sniff import Format, Sniff, open_dicom


#**************************************************************************************************#
#                                      PhilipsDicomRawReader                                       #
#**************************************************************************************************#
#                                                                                                  #
# The payload is in private (2005,1144) elements. An exam log (XML) gives one Protocol per scan of #
# the ExamCard it holds (ExamCardBlob, base64), as the ExamCard reader reads it, with the scan's   #
# log entry (state, times, series UIDs). Otherwise each element that is a parameter table in the   #
# ExamCard's binary layout (EX_ for the scan's settings, CSC_, CRC_, CPR_, PHW_ and HW_ for system #
# and hardware, ...) is decoded and all of them make one Protocol, named by the series, with the   #
# ExamCard's standard keys; text elements (JSON, XML) are kept as text, others by their size.      #
#                                                                                                  #
#**************************************************************************************************#
class PhilipsDicomRawReader(PhilipsExamCardReader):
    FORMATS = (Format.PHILIPS_DICOM_RAW,)
    PACKAGES = ("miox", "pydicom")
    SOURCE = "Raw Data Storage"
    PAYLOAD = (0x2005, 0x1144)
    NAME = re.compile(rb"^[A-Za-z][A-Za-z0-9_]{2,32}$")  # a parameter name of a table

    def read(self, sniff: Sniff) -> list[Protocol]:
        return self.per_file(sniff, self._one)

    def _one(self, sniff: Sniff) -> list[Protocol]:
        with open_dicom(sniff.path) as fh:
            ds = pydicom.dcmread(fh)
        name = str(ds.get("SeriesDescription") or ds.get("ProtocolName") or sniff.path.stem)
        tables, texts, others = [], [], []
        for element in ds.iterall():
            if element.tag != self.PAYLOAD or not element.value:
                continue
            value = bytes(element.value)
            if value.lstrip().startswith(b"<?xml") and b"<ExamCardBlob>" in value:
                return self._exam(value.rstrip(b"\0"), name, sniff)
            table = self._table(value)
            if table is not None:
                tables.append(table)
            elif (text := self._text(value)) is not None:
                texts.append(text)
            else:
                others.append(len(value))
        if not tables:
            raise ReaderError(f"{sniff.path.name}: no scan parameters in the Raw Data Storage")
        parameters, choices = {}, {}
        for table, table_choices in tables:
            parameters.update(table)
            choices.update(table_choices)
        dicom = {key: str(ds.get(key)) for key in ("SeriesDescription", "ProtocolName",
                                                   "SeriesNumber", "SoftwareVersions",
                                                   "Manufacturer") if ds.get(key) is not None}
        meta = Meta({"parameters": parameters, "choices": choices, "texts": texts,
                     "other_blocks_bytes": others, "dicom": dicom})
        self._keys(meta, parameters, {})
        if isinstance(parameters.get("HW_main_magnetic_field_mT"), (int, float)):
            field = "HW_main_magnetic_field_mT"
            meta.set("FieldStrength", parameters[field] / 1000, f"{self.SOURCE}: {field}",
                     f"parameters/{field}")
        return [Protocol(name, (), meta, self.provenance(sniff))]

    def _exam(self, text: bytes, name: str, sniff: Sniff) -> list[Protocol]:
        """An exam log with its ExamCard: the card's scans, each with its log entry."""
        try:
            root = ET.fromstring(text)
        except ET.ParseError as error:
            raise ReaderError(f"{sniff.path.name}: exam log not readable: {error}") from error
        local = {e.tag.rsplit("}", 1)[-1]: e for e in root.iter()}
        blob = local.get("ExamCardBlob")
        if blob is None or not (blob.text or "").strip():
            raise ReaderError(f"{sniff.path.name}: the exam log holds no ExamCard")
        protocols = self.protocols(base64.b64decode(blob.text), name, sniff)
        log: dict[str, list[dict]] = {}
        for scan in (e for e in root.iter() if e.tag.endswith("SingleScan")):
            fields = {c.tag.rsplit("}", 1)[-1]: (c.text or "").strip() for c in scan
                      if len(c) == 0 and (c.text or "").strip()}
            uids = [u.text.strip() for u in scan.iter() if u.tag.endswith("}string") and u.text]
            if uids:
                fields["SeriesUids"] = uids
            log.setdefault(fields.get("Name", ""), []).append(fields)
        for protocol in protocols:  # scans of one name in their order in both
            if log.get(protocol.name):
                protocol.meta.vendor["exam_log"] = log[protocol.name].pop(0)
        return protocols

    def _table(self, value: bytes) -> tuple[dict[str, Any], dict] | None:
        """A parameter table in the ExamCard's layout, or None: its count fits the element and
        its first entries are named like parameters."""
        if len(value) < 82:
            return None
        count = struct.unpack_from("<I", value, 28)[0]
        if not 0 < count <= (len(value) - 32) // 50:
            return None
        if not all(self.NAME.match(value[32 + 50 * i: 32 + 50 * i + 33].split(b"\0")[0])
                   for i in range(min(count, 3))):
            return None
        try:
            return self.parameters(value)
        except (struct.error, UnicodeDecodeError):
            return None

    @staticmethod
    def _text(value: bytes) -> str | None:
        text = value.rstrip(b"\0")
        printable = all(32 <= c < 127 or c in (9, 10, 13) for c in text[:512])
        if text[:1] in (b"{", b"[", b"<") and printable:
            return text.decode("utf-8", errors="replace")
        return None
