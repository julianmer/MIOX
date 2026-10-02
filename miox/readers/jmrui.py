####################################################################################################
#                                             jmrui.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for jMRUI files: text (.txt) and binary (.mrui). spec2nii converts; the vendor   #
#          header is every key of the text header (also keys with spaces, which spec2nii drops),   #
#          or the numeric header and the trailing text of a .mrui file.                            #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re
from argparse import Namespace
from pathlib import Path

from nifti_mrs.nifti_mrs import NIFTI_MRS
from spec2nii.jmrui import jmrui_format, read_mrui

# own
from miox.readers.base import Spec2niiReader
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                           JmruiReader                                            #
#**************************************************************************************************#
#                                                                                                  #
# Reads one jMRUI file; several signals in one file become a DIM_USER_0 dimension.                 #
#                                                                                                  #
#**************************************************************************************************#
class JmruiReader(Spec2niiReader):
    FORMATS = (Format.JMRUI_TEXT, Format.JMRUI_BINARY)
    CONVERTER = "jmrui"
    SIGNAL = re.compile(r"Signal (?:number: )?\d+ out of \d+ in file")  # start of the data
    SOURCES_TEXT = {
        "SpectrometerFrequency": "TransmitterFrequency",
        "ResonantNucleus": ("TypeOfNucleus", "MagneticField"),
        "ManufacturersModelName": "Spectrometer",
        "PatientName": "NameOfPatient",
        "DwellTime": "SamplingInterval",
    }
    SOURCES_BINARY = {
        "SpectrometerFrequency": "transmitter_frequency",
        "ResonantNucleus": ("type_of_nucleus", "magnetic_field"),
        "TxOffset": "reference_frequency_ppm",
        "DwellTime": "sampling_interval",
    }

    def convert(self, sniff: Sniff) -> tuple[list[NIFTI_MRS], list[str], dict]:
        images, names = jmrui_format(Namespace(file=sniff.path, affine=None, fileout=None))
        if sniff.format == Format.JMRUI_TEXT:
            header = self._text_header(sniff.path)
        else:
            _, header, text = read_mrui(sniff.path)
            header = {**header, "text": text}
        return list(images), list(names), {k: self.plain(v) for k, v in header.items()}

    def sources(self, vendor: dict) -> dict[str, str | tuple[str, ...]]:
        return self.SOURCES_BINARY if "sampling_interval" in vendor else self.SOURCES_TEXT

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

    @staticmethod
    def _number(text: str) -> int | float | str:
        for kind in (int, float):
            try:
                return kind(text)
            except ValueError:
                pass
        return text
