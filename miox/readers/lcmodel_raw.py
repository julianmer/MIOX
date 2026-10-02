####################################################################################################
#                                          lcmodel_raw.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for LCModel .RAW / .H2O files: Fortran namelists ($SEQPAR, $NMID, ...) followed  #
#          by the FID as real/imaginary pairs. LCModel keeps the frequency and dwell time in the   #
#          control file (or a MakeBasis .IN or .BASIS) when the .RAW does not; such a file next to #
#          the .RAW is read too. The vendor header keeps every namelist, any text before the first #
#          namelist, and the side file's namelists.                                                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re
from pathlib import Path

import f90nml
import numpy as np
from nifti_mrs.create_nmrs import gen_nifti_mrs_hdr_ext
from nifti_mrs.hdr_ext import Hdr_Ext

# own
from miox.data import MRSData
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                         LcmodelRawReader                                         #
#**************************************************************************************************#
#                                                                                                  #
# Reads one .RAW file into one MRSData (several FIDs in one file, as LCModel writes for MRSI, go   #
# along DIM_USER_0 when the header states the number of points). The format stores no nucleus and  #
# no voxel, so 1H is assumed and the geometry is a 10 m voxel, both noted. LCModel's sign          #
# convention is NIfTI-MRS's, so the data stay as written unless the file says BRUKER=T.            #
#                                                                                                  #
#**************************************************************************************************#
class LcmodelRawReader(Reader):
    FORMATS = (Format.LCMODEL_RAW,)
    PACKAGES = Reader.PACKAGES + ("f90nml",)
    SOURCE = "LCModel namelist"
    NUCLEUS = "1H"
    NO_GEOMETRY = np.diag([1e4, 1e4, 1e4, 1.0])  # no voxel stored (as spec2nii writes it)
    FIELD = re.compile(r"\(\s*(\d*)\s*[EFGD](\d+)", re.I)  # FMTDAT repeat count and field width
    NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?")
    DWELL = (("dwelltime", 1.0), ("deltat", 1.0), ("badelt", 1.0))  # header keys, factor to s
    SIDE_LIMIT = 50_000_000  # bytes; control, MakeBasis and basis files
    SIDE_MARKERS = ("$LCMODL", "&LCMODL", "$SEQPAR", "&SEQPAR")  # namelists stating HZPPPM
    GROUP = re.compile(r"\$(\w+)(.*?)\$END", re.S | re.I)  # one namelist, data lines skipped

    def read(self, sniff: Sniff) -> list[MRSData]:
        notes = [f"nucleus not stored in LCModel .RAW files; {self.NUCLEUS} assumed"]
        text = sniff.path.read_text(errors="replace")
        end = text.upper().rfind("$END")
        if end < 0:
            raise ReaderError(f"no namelist ($NMID ... $END) in {sniff.path.name}")
        header = {
            "text": [line for line in text[: text.find("$")].splitlines() if line.strip()],
            "namelists": self._namelists(text[: end + 4]),
        }
        side = self._side_file(sniff.path)
        if side is not None:
            header["side_file"] = side.name
            header["side_namelists"] = self._namelists(side.read_text(errors="replace"))
            notes.append(f"frequency and dwell time looked up in {side.name}")
        frequency, frequency_field = self._find(header, "hzpppm")
        dwell, dwell_field = self._dwell(header)
        if frequency is None or dwell is None:
            missing = " and ".join(
                name for name, v in (("HZPPPM", frequency), ("dwell time", dwell)) if v is None
            )
            raise ReaderError(
                f"{sniff.path.name} states no {missing}, and neither does a control file "
                "($LCMODL or $SEQPAR namelist) next to it"
            )
        nmid = header["namelists"].get("nmid", {})
        fid = self._fid(text[end + 4 :], nmid.get("fmtdat"))
        if nmid.get("bruker", False):  # Bruker sign convention: LCModel conjugates these
            fid = fid.conj()
        points = header["namelists"].get("seqpar", {}).get("numberofpoints")
        data = self._shape(fid, points, notes)
        hdr_ext = Hdr_Ext(float(frequency), self.NUCLEUS)
        hdr_ext.set_standard_def("OriginalFile", [sniff.path.name])
        echo, echo_field = self._find(header, "echot")
        if echo is not None:
            hdr_ext.set_standard_def("EchoTime", float(echo) * 1e-3)
        if data.ndim == 5:
            hdr_ext.set_dim_info(0, "DIM_USER_0", info="FIDs in file order (MRSI grid in .CONTROL)")
        image = gen_nifti_mrs_hdr_ext(data, dwell, hdr_ext, self.NO_GEOMETRY, no_conj=True)
        provenance = self.provenance(sniff, tuple(notes))
        d = MRSData.from_nifti_mrs(
            image, provenance, self.SOURCE, vendor=header, name=sniff.path.stem
        )
        d.meta.set("SpectrometerFrequency", [float(frequency)], self.SOURCE, frequency_field)
        d.meta.set("ResonantNucleus", [self.NUCLEUS], f"assumed: {self.NUCLEUS}")
        d.meta.set("OriginalFile", [sniff.path.name], "file name")
        d.meta.set("DwellTime", dwell, self.SOURCE, dwell_field)
        if echo is not None:
            d.meta.set("EchoTime", float(echo) * 1e-3, f"{self.SOURCE} (ms to s)", echo_field)
        self.correct(d)
        return [d]

    def _namelists(self, text: str) -> dict:
        return {group: dict(values) for group, values in f90nml.reads(text).todict().items()}

    def _side_file(self, path: Path) -> Path | None:
        if self._find({"namelists": self._raw(path)}, "hzpppm")[0] is not None:
            return None
        for candidate in sorted(path.parent.iterdir()):
            if candidate == path or not candidate.is_file():
                continue
            if candidate.stat().st_size > self.SIDE_LIMIT:
                continue
            text = candidate.read_text(errors="replace")
            if not any(marker in text.upper() for marker in self.SIDE_MARKERS):
                continue
            namelists = {}
            for group in self.GROUP.finditer(text):
                try:
                    namelists.update(self._namelists(f"${group[1]}{group[2]}$END"))
                except Exception:  # not a namelist after all
                    continue
            if self._find({"namelists": namelists}, "hzpppm")[0] is not None:
                return candidate
        return None

    def _raw(self, path: Path) -> dict:
        text = path.read_text(errors="replace")
        return self._namelists(text[: text.upper().rfind("$END") + 4])

    @staticmethod
    def _find(header: dict, key: str) -> tuple:
        """First value of ``key`` in the file's namelists, then the side file's."""
        for part in ("namelists", "side_namelists"):
            for group, values in header.get(part, {}).items():
                if key in values and values[key] not in (None, ""):
                    return values[key], f"{part}/{group}/{key}"
        return None, None

    def _dwell(self, header: dict) -> tuple:
        for key, factor in self.DWELL:
            value, field = self._find(header, key)
            if value is not None:
                return abs(float(value)) * factor, field
        return None, None

    def _fid(self, text: str, fmtdat: str | None) -> np.ndarray:
        match = self.FIELD.search(fmtdat or "")
        values = []
        for line in text.splitlines():
            line = line.rstrip()
            if not line:
                continue
            if match:
                width = int(match[2])
                fields = [line[i : i + width] for i in range(0, len(line), width)]
            else:
                fields = self.NUMBER.findall(line)
            values.extend(float(f.strip().replace("D", "E").replace("d", "e")) for f in fields)
        values = np.asarray(values)
        return values[0::2] + 1j * values[1::2]

    def _shape(self, fid: np.ndarray, points: int | None, notes: list[str]) -> np.ndarray:
        if points and len(fid) > points and len(fid) % points == 0:
            notes.append(f"{len(fid) // points} FIDs of {points} points in one file")
            return fid.reshape(-1, points).T.reshape(1, 1, 1, points, -1)
        return fid.reshape(1, 1, 1, -1)
