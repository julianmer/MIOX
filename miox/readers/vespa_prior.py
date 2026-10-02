####################################################################################################
#                                          vespa_prior.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for Vespa prior knowledge (.xml, .xml.gz with <prior> elements): per metabolite  #
#          a name, three loop indices, the number of spins and lines of group, ppm, area and phase #
#          (degrees). The vendor header is the <prior> tree without the metabolites, whose values  #
#          are the lines.                                                                          #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import xml.etree.ElementTree as ET

# own
from miox.data import PriorSet
from miox.readers.base import Reader, ReaderError
from miox.readers.vespa_viff import VespaViffReader
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                         VespaPriorReader                                         #
#**************************************************************************************************#
#                                                                                                  #
# One PriorSet per <prior>. TE (seqte) is in ms, phases are in degrees, as Vespa writes them.      #
#                                                                                                  #
#**************************************************************************************************#
class VespaPriorReader(Reader):
    FORMATS = (Format.VESPA_PRIOR,)
    SOURCE = "Vespa prior"

    def read(self, sniff: Sniff) -> list[PriorSet]:
        viff = VespaViffReader()
        priors = ET.fromstring(viff.text(sniff.path)).findall("prior")
        if not priors:
            raise ReaderError(f"{sniff.path.name} holds no <prior>")
        names = [sniff.path.stem] if len(priors) == 1 else [
            f"{sniff.path.stem}_{i}" for i in range(len(priors))
        ]
        return [
            self._prior(sniff, prior, viff.tree(prior), name)
            for prior, name in zip(priors, names, strict=True)
        ]

    def _prior(self, sniff: Sniff, prior: ET.Element, tree: dict, name: str) -> PriorSet:
        lines = []
        for metabolite in prior.findall("prior_metabolite"):
            dims = [int(round(float(d.text or 0))) for d in metabolite.findall("dim")]
            spins = int(float(metabolite.findtext("spins") or 0))
            for i, line in enumerate(metabolite.findall("line")):
                lines.append({
                    "metabolite": metabolite.findtext("name"),
                    "line": i,
                    "ppm": float(line.findtext("ppm")),
                    "area": float(line.findtext("area")),
                    "phase": float(line.findtext("phase")),
                    "group": line.findtext("group") or "",
                    "dims": dims,
                    "spins": spins,
                })
        values = {}
        if prior.findtext("nucleus"):
            values["ResonantNucleus"] = ([prior.findtext("nucleus").strip()], "prior/nucleus")
        if float(prior.findtext("seqte") or 0) > 0:
            values["EchoTime"] = (float(prior.findtext("seqte")) * 1e-3, "prior/seqte")
        tree.pop("prior_metabolite", None)
        return self.prior_set(sniff, lines, values, {"prior": tree}, name, [], self.SOURCE)
