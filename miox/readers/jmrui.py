####################################################################################################
#                                             jmrui.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for jMRUI files: text (.txt) and binary (.mrui). spec2nii converts; the vendor   #
#          header is the parsed text header, or the numeric header and the trailing text of a      #
#          .mrui file.                                                                             #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

from argparse import Namespace

from nifti_mrs.nifti_mrs import NIFTI_MRS
from spec2nii.jmrui import jmrui_format, read_mrui, readjMRUItxt

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
            header = readjMRUItxt(sniff.path)[1]
        else:
            _, header, text = read_mrui(sniff.path)
            header = {**header, "text": text}
        return list(images), list(names), {k: self.plain(v) for k, v in header.items()}

    def sources(self, vendor: dict) -> dict[str, str | tuple[str, ...]]:
        return self.SOURCES_BINARY if "sampling_interval" in vendor else self.SOURCES_TEXT
