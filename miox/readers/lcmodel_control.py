####################################################################################################
#                                        lcmodel_control.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for LCModel control files ($LCMODL namelist) and MakeBasis input files ($SEQPAR, #
#          $NMALL and one $NMEACH per metabolite) as analysis protocols.                           #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re

import f90nml

# own
from miox.data import Protocol
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                       LcmodelControlReader                                       #
#**************************************************************************************************#
#                                                                                                  #
# A control file gives LCModel its inputs as Fortran namelists: $LCMODL (or &LCMODL) up to $END    #
# (or a "/" line), the MakeBasis input $SEQPAR, $NMALL and $NMEACH, repeated once per metabolite.  #
# Every namelist goes to the vendor header (repeated groups as a list), the file's name is the     #
# protocol name; HZPPPM (MHz), DELTAT (s), ECHOT (ms) and NUNFIL give the standard keys.           #
#                                                                                                  #
#**************************************************************************************************#
class LcmodelControlReader(Reader):
    FORMATS = (Format.LCMODEL_CONTROL,)
    PACKAGES = ("miox", "f90nml")
    SOURCE = "LCModel control namelist"
    GROUP = re.compile(r"[$&](\w+)\b(.*?)(?:[$&]END\b|^\s*/\s*$)", re.S | re.I | re.M)
    REPEATED = re.compile(r"_grp_(\w+)_\d+")  # f90nml's names of repeated groups
    KEYS = {  # standard key: ((group, field), ...), conversion
        "SpectrometerFrequency": ((("lcmodl", "hzpppm"), ("seqpar", "hzpppm")),
                                  lambda v: [float(v)]),
        "DwellTime": ((("lcmodl", "deltat"),), float),
        "EchoTime": ((("lcmodl", "echot"), ("seqpar", "echot")), lambda v: v * 1e-3),
        "NumberOfPoints": ((("lcmodl", "nunfil"),), int),
    }

    def read(self, sniff: Sniff) -> list[Protocol]:
        text = sniff.path.read_text(errors="replace")
        groups = [f"${m[1]}{m[2]}$END" for m in self.GROUP.finditer(text)]
        if not groups:
            raise ReaderError(f"{sniff.path.name}: no namelist")
        namelists: dict = {}
        for group, values in f90nml.reads("\n".join(groups)).todict().items():
            repeated = self.REPEATED.fullmatch(group)
            if repeated:
                namelists.setdefault(repeated[1], []).append(dict(values))
            else:
                namelists[group] = dict(values)
        meta = Meta({"namelists": namelists})
        for key, (fields, convert) in self.KEYS.items():
            for group, field in fields:
                values = namelists.get(group)
                value = values.get(field) if isinstance(values, dict) else None
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    meta.set(key, convert(value), f"{self.SOURCE}: {group}/{field}",
                             f"namelists/{group}/{field}")
                    break
        return [Protocol(sniff.path.stem, (), meta, self.provenance(sniff))]
