####################################################################################################
#                                       osprey_container.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for Osprey job containers (.mat with an MRSCont struct): the FID-A structs it    #
#          holds (raw, raw_ref, raw_w, raw_mm, ... and those under processed). The vendor header   #
#          is the whole container, large arrays summarised.                                        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import dataclasses

# own
from miox.data import MRSData
from miox.matfile import MatFile
from miox.readers.base import Reader, ReaderError
from miox.readers.fida_mat import FidaReader
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                      OspreyContainerReader                                       #
#**************************************************************************************************#
#                                                                                                  #
# Every list of FID-A structs in MRSCont (top level and under processed) gives one MRSData per     #
# struct, named after the field and the dataset number, read as FID-A structs are. Fit and         #
# quantification results stay in the vendor header, with a note.                                   #
#                                                                                                  #
#**************************************************************************************************#
class OspreyContainerReader(Reader):
    FORMATS = (Format.OSPREY_MRSCONT,)
    PACKAGES = Reader.PACKAGES + ("mat-io",)
    RESULTS = ("fit", "quantify")

    def read(self, sniff: Sniff) -> list[MRSData]:
        values = MatFile.load(sniff.path)
        container = values.get("MRSCont")
        if not isinstance(container, dict):
            raise ReaderError(f"{sniff.path.name}: MRSCont is not a struct")
        vendor = MatFile.header(values)
        fida = FidaReader()
        datasets = []
        for path, structs in self._struct_lists(container):
            for i, out in enumerate(structs):
                var = f"MRSCont/{path}/{i}"
                name = f"{path.replace('/', '_')}_{i + 1}"
                datasets.append(fida.from_struct(sniff, var, out, vendor, name))
        if not datasets:
            raise ReaderError(f"{sniff.path.name}: MRSCont holds no FID-A structs")
        results = [key for key in self.RESULTS if key in container]
        if results:
            note = f"Osprey {', '.join(results)} results not read; kept in the vendor header"
            for d in datasets:
                d.provenance = dataclasses.replace(d.provenance, notes=d.provenance.notes + (note,))
        return datasets

    @staticmethod
    def _struct_lists(container: dict, prefix: str = "") -> list[tuple[str, list[dict]]]:
        """(path, structs) for every list of FID-A structs, at the top level and in processed."""
        found = []
        for key, value in container.items():
            path = f"{prefix}{key}"
            if isinstance(value, dict) and "fids" in value:
                value = [value]
            if isinstance(value, list) and value and all(
                isinstance(v, dict) and "fids" in v for v in value
            ):
                found.append((path, value))
            elif key == "processed" and isinstance(value, dict):
                found += OspreyContainerReader._struct_lists(value, f"{path}/")
        return found
