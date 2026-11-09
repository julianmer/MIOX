####################################################################################################
#                                    siemens_dicom_protocol.py                                     #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 30/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for Siemens protocols stored as DICOM: Raw Data Storage of XA systems and the    #
#          PhoenixZIPReport structured report.                                                     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import collections
import html
import re

import pydicom
from nibabel.nicom import csareader

# own
from miox.data import Protocol
from miox.readers.base import ReaderError
from miox.readers.siemens_protocol import SiemensProtocolReader
from miox.sniff import Format, Sniff, open_dicom


#**************************************************************************************************#
#                                    SiemensDicomProtocolReader                                    #
#**************************************************************************************************#
#                                                                                                  #
# Raw Data Storage holds the XProtocol text with its ASCCONV block in (0021,1019); other private   #
# text elements (diffusion tensor XML, ...) are kept as text, binary ones by their size. The       #
# PhoenixZIPReport holds the protocol as an EDF document in its CSA image header (PhoenixZIP),     #
# XML-escaped, with one GUID written in place of the whitespace after every name and before every  #
# value; unescaped and with that GUID (the most frequent) as a space, its ASCCONV block reads as   #
# in a .pro file; on XA systems the report holds the XProtocol text in (0021,1101). One Protocol   #
# each, its keys as the protocol reader gives them.                                                #
#                                                                                                  #
#**************************************************************************************************#
class SiemensDicomProtocolReader(SiemensProtocolReader):
    FORMATS = (Format.SIEMENS_DICOM_RAW, Format.SIEMENS_PHOENIX_REPORT)
    PACKAGES = ("miox", "pydicom", "nibabel")
    XPROTOCOL = (0x0021, 0x1019)
    REPORT_XPROTOCOL = (0x0021, 0x1101)  # the report's protocol on XA systems
    CSA_IMAGE = (0x0029, 0x1010)
    GUID = re.compile(r"[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}")

    def read(self, sniff: Sniff) -> list[Protocol]:
        return self.per_file(sniff, self._one)

    def _one(self, sniff: Sniff) -> list[Protocol]:
        with open_dicom(sniff.path) as fh:
            ds = pydicom.dcmread(fh)
        texts, others = {}, {}
        if sniff.format == Format.SIEMENS_PHOENIX_REPORT:
            text = self._report(ds)
        else:
            text = ""
            for element in ds:
                value = element.value
                if not element.tag.is_private or not isinstance(value, (bytes, bytearray)):
                    continue
                if element.tag == self.XPROTOCOL:
                    text = bytes(value).decode("latin-1")
                elif bytes(value[:1]) == b"<":
                    texts[str(element.tag)] = bytes(value).rstrip(b"\0").decode("latin-1")
                elif len(value) > 64:
                    others[str(element.tag)] = len(value)
        if "### ASCCONV BEGIN" not in text:
            raise ReaderError(f"{sniff.path.name}: no protocol (ASCCONV block) in the DICOM")
        meta = self.meta(text)
        meta.vendor.update({"texts": texts, "other_elements_bytes": others})
        name = meta.get("ProtocolName") or str(ds.get("SeriesDescription") or sniff.path.stem)
        return [Protocol(name, (), meta, self.provenance(sniff))]

    def _report(self, ds: pydicom.Dataset) -> str:
        if self.REPORT_XPROTOCOL in ds:
            return bytes(ds[self.REPORT_XPROTOCOL].value).decode("latin-1")
        if self.CSA_IMAGE not in ds:
            return ""
        tags = csareader.read(bytes(ds[self.CSA_IMAGE].value))["tags"]
        text = html.unescape("".join(tags.get("PhoenixZIP", {}).get("items", [])))
        guid = collections.Counter(self.GUID.findall(text)).most_common(1)
        return text.replace(guid[0][0], " ") if guid else text
