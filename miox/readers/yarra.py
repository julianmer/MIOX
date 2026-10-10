####################################################################################################
#                                             yarra.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 10/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for Yarra reconstruction modes (.mode) and module manifests (.ymf): each file a  #
#          Protocol with every section and option in the vendor header.                            #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import configparser

# own
from miox.data import Protocol
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                           YarraReader                                            #
#**************************************************************************************************#
#                                                                                                  #
# Yarra files are INI text (";" comments, case-sensitive option names, "%" placeholders such as    #
# %rid that are not interpolated). A mode's [ClientConfig] Name is the name the scanner client     #
# shows; a module manifest's [YarraModule] Name, Version and Author describe the module. Options   #
# keep their text as written.                                                                      #
#                                                                                                  #
#**************************************************************************************************#
class YarraReader(Reader):
    FORMATS = (Format.YARRA_MODE, Format.YARRA_MODULE)
    PACKAGES = ("miox",)

    def read(self, sniff: Sniff) -> list[Protocol]:
        parser = configparser.ConfigParser(interpolation=None, strict=False,
                                           comment_prefixes=(";", "#"))
        parser.optionxform = str  # keep the case of option names
        try:
            parser.read_string(sniff.path.read_text(errors="replace"))
        except configparser.Error as e:
            raise ReaderError(f"{sniff.path}: not a readable Yarra file ({e})") from e
        vendor = {section: dict(parser[section]) for section in parser.sections()}
        meta = Meta(vendor)
        section = "ClientConfig" if sniff.format == Format.YARRA_MODE else "YarraModule"
        name = vendor.get(section, {}).get("Name")
        if name:
            meta.set("ProtocolName", name, f"Yarra file: [{section}] Name", f"{section}/Name")
        if sniff.format == Format.YARRA_MODULE and vendor.get(section, {}).get("Version"):
            meta.set("SoftwareVersions", vendor[section]["Version"],
                     f"Yarra file: [{section}] Version", f"{section}/Version")
        return [Protocol(name or sniff.path.stem, (), meta, self.provenance(sniff))]
