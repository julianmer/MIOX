####################################################################################################
#                                         lcmodel_basis.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for LCModel basis sets (.BASIS): $SEQPAR and $BASIS1 namelists, then per         #
#          metabolite an optional $NMUSED (MakeBasis settings) and a $BASIS namelist followed by   #
#          its spectrum. The vendor header keeps every namelist.                                   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re

import f90nml
import numpy as np

# own
from miox.data import MRSData
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                       LcmodelBasisReader                                         #
#**************************************************************************************************#
#                                                                                                  #
# Each metabolite block holds fft(FID zero-filled to NDATAB points), checked against the .RAW      #
# files of the same basis; ISHIFT is a circular shift MakeBasis applied to that spectrum. The FIDs #
# come back as ifft of the unshifted spectrum, NDATAB points long. Encrypted sets (negative        #
# BADELT) are refused. The format stores no nucleus, so 1H is assumed and noted.                   #
#                                                                                                  #
#**************************************************************************************************#
class LcmodelBasisReader(Reader):
    FORMATS = (Format.LCMODEL_BASIS,)
    PACKAGES = Reader.PACKAGES + ("f90nml",)
    SOURCE = "LCModel basis namelist"
    NUCLEUS = "1H"
    GROUP = re.compile(r"\$(\w+)(.*?)\$END", re.S | re.I)
    FIELD = re.compile(r"\(\s*(\d*)\s*[EFGD](\d+)", re.I)

    def read(self, sniff: Sniff) -> list[MRSData]:
        text = sniff.path.read_text(errors="replace")
        groups = list(self.GROUP.finditer(text))
        header, metabolites, spectra = {}, [], []
        for i, group in enumerate(groups):
            name = group[1].lower()
            values = self._namelist(group[1], group[2])
            if name in ("seqpar", "basis1"):
                header[name] = values
            elif name == "nmused":
                metabolites.append({"nmused": values})
            elif name == "basis":
                if not metabolites or "basis" in metabolites[-1]:
                    metabolites.append({})
                metabolites[-1]["basis"] = values
                end = groups[i + 1].start() if i + 1 < len(groups) else len(text)
                spectra.append(self._numbers(text[group.end() : end], header))
        header["metabolites"] = metabolites
        basis1, seqpar = header.get("basis1", {}), header.get("seqpar", {})
        if float(basis1.get("badelt", 0)) < 0:
            raise ReaderError(f"{sniff.path.name} is an encrypted LCModel basis (negative BADELT)")
        names = [str(m["basis"].get("metabo") or m["basis"].get("id")).strip() for m in metabolites]
        fids = [
            np.fft.ifft(np.roll(spectrum, -int(m["basis"].get("ishift", 0) or 0)))
            for spectrum, m in zip(spectra, metabolites, strict=True)
        ]
        fids = np.stack(fids, axis=-1)
        keys = {
            "SpectrometerFrequency": ([float(seqpar["hzpppm"])], "seqpar/hzpppm"),
            "ResonantNucleus": ([self.NUCLEUS], None),
        }
        echo = seqpar.get("echot")
        if isinstance(echo, (int, float)) and echo > 0:
            keys["EchoTime"] = (echo * 1e-3, "seqpar/echot")
        notes = [
            f"nucleus not stored in LCModel basis files; {self.NUCLEUS} assumed",
            f"FIDs are the stored zero-filled length, {fids.shape[0]} points",
        ]
        dwelltime = float(basis1["badelt"])
        d = self.basis_set(
            sniff, fids, names, dwelltime, keys, header, sniff.path.stem, notes, self.SOURCE
        )
        d.meta.set("ResonantNucleus", [self.NUCLEUS], f"assumed: {self.NUCLEUS}")
        d.meta.set("DwellTime", dwelltime, f"{self.SOURCE}: basis1/badelt", "basis1/badelt")
        return [d]

    @staticmethod
    def _namelist(name: str, body: str) -> dict:
        parsed = f90nml.reads(f"${name}{body}$END").todict()
        return dict(next(iter(parsed.values()), {}))

    def _numbers(self, text: str, header: dict) -> np.ndarray:
        """The real/imaginary pairs after a $BASIS namelist, read in FMTBAS fields."""
        match = self.FIELD.search(str(header.get("basis1", {}).get("fmtbas", "")))
        values = []
        for line in text.splitlines():
            line = line.rstrip()
            if not line.strip() or line.lstrip().startswith("$"):
                continue
            if match:
                width = int(match[2])
                fields = [line[i : i + width] for i in range(0, len(line), width)]
            else:
                fields = line.split()
            values.extend(float(f.strip().replace("D", "E")) for f in fields if f.strip())
        values = np.asarray(values)
        return values[0::2] + 1j * values[1::2]
