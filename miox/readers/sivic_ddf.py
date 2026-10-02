####################################################################################################
#                                           sivic_ddf.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for UCSF / SIVIC spectroscopy (.ddf text header, .cmplx big-endian float32       #
#          complex data), often MRSI. The vendor header keeps every header line by section.        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

# own
from miox.data import MRSData
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                          SivicDdfReader                                          #
#**************************************************************************************************#
#                                                                                                  #
# Data are stored point fastest, then x, y, z (and frames). Spectra (first dimension of type       #
# frequency) are turned into FIDs: SIVIC orders points from the highest frequency down, which is   #
# the order of fftshift(fft(fid)) in the NIfTI-MRS convention: fid = ifft(ifftshift(spectrum)).    #
# The geometry is LPS (toplc is the centre of the first voxel) and becomes a RAS affine. "ppm      #
# reference" is ReferencePPM: SIVIC reads and writes it as the DICOM Chemical Shift Reference.     #
#                                                                                                  #
#**************************************************************************************************#
class SivicDdfReader(Reader):
    FORMATS = (Format.SIVIC_DDF,)
    SOURCE = "UCSF DDF header"
    DIMENSION = re.compile(
        r"dimension (\d+): type: (\w+) npoints: (\d+)(?: pixel spacing\(mm\): ([-\d.eE+]+))?"
    )
    LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])

    def read(self, sniff: Sniff) -> list[MRSData]:
        ddf, cmplx = self._pair(sniff)
        header, dims = self._header(ddf)
        values = np.fromfile(cmplx, dtype=">f4")
        points = dims[0][1]
        grid = [n for _, n, _ in dims[1:4]] + [1] * (3 - len(dims[1:4]))
        frames = values.size // 2 // (points * int(np.prod(grid)))
        data = (values[0::2] + 1j * values[1::2]).reshape(frames, *grid[::-1], points)
        data = np.moveaxis(data, 0, -1).transpose(2, 1, 0, 3, 4)  # x, y, z, points, frames
        notes = []
        if dims[0][0] == "frequency":
            data = np.fft.ifft(np.fft.ifftshift(data, axes=3), axis=3)
            notes.append("stored as spectra; converted to FIDs")
        else:
            notes.append("stored as FIDs; their sign convention is not verified")
        if frames == 1:
            data = data[..., 0]
        mr, spec = header.get("MR Parameters", {}), header.get("Spectroscopy Parameters", {})
        keys = {
            "SpectrometerFrequency": (
                [float(mr["transmitter frequency(MHz)"])],
                "MR Parameters/transmitter frequency(MHz)",
            ),
            "ResonantNucleus": ([mr["isotope"]], "MR Parameters/isotope"),
        }
        for key, name, factor in (
            ("EchoTime", "echo time(ms)", 1e-3),
            ("RepetitionTime", "repetition time(ms)", 1e-3),
            ("InversionTime", "inversion time(ms)", 1e-3),
            ("ExcitationFlipAngle", "flip angle", 1.0),
        ):
            if mr.get(name):
                keys[key] = (float(mr[name]) * factor, f"MR Parameters/{name}")
        if spec.get("ppm reference"):
            reference = "Spectroscopy Parameters/ppm reference"
            keys["ReferencePPM"] = (float(spec["ppm reference"]), reference)
        dwelltime = float(spec["dwelltime(ms)"]) * 1e-3
        tags = ("DIM_DYN",) if frames > 1 else ()
        d = self.dataset(
            sniff,
            data,
            dwelltime,
            keys,
            header,
            ddf.stem,
            notes,
            self.SOURCE,
            affine=self.affine(header, dims),
            dim_tags=tags,
        )
        field = "Spectroscopy Parameters/dwelltime(ms)"
        d.meta.set("DwellTime", dwelltime, f"{self.SOURCE}: {field}", field)
        return [d]

    def affine(self, header: dict, dims: list[tuple]) -> np.ndarray:
        """RAS affine from the LPS position of the first voxel, spacing and direction cosines."""
        main = header["DATA DESCRIPTOR FILE"]
        toplc = [float(v) for v in main["toplc(lps, mm)"].split()]
        cosines = [[float(v) for v in main[f"dcos{i}"].split()] for i in range(3)]
        spacing = [s for _, _, s in dims[1:4]] + [1.0] * (3 - len(dims[1:4]))
        lps = np.eye(4)
        for axis in range(3):
            lps[:3, axis] = np.asarray(cosines[axis]) * spacing[axis]
        lps[:3, 3] = toplc
        return self.LPS_TO_RAS @ lps

    def _pair(self, sniff: Sniff) -> tuple[Path, Path]:
        files = (sniff.path,) + sniff.companions
        ddf = next((f for f in files if f.suffix.lower() == ".ddf"), None)
        cmplx = next((f for f in files if f.suffix.lower() == ".cmplx"), None)
        if ddf is None or cmplx is None:
            raise ReaderError(f"incomplete .ddf/.cmplx pair: {sniff.path.name}")
        return ddf, cmplx

    def _header(self, path: Path) -> tuple[dict, list[tuple]]:
        """Header lines by section (sections are separated by lines of '='), and the dimensions."""
        lines = path.read_text(errors="replace").splitlines()
        header, section, dims = {}, lines[0].strip(), []
        expect_title = False
        for line in lines[1:]:
            if line.startswith("==="):
                expect_title = True
                continue
            if expect_title and line.strip() and ":" not in line:
                section, expect_title = line.strip(), False
                continue
            expect_title = False
            if match := self.DIMENSION.match(line.strip()):
                number, kind, n, spacing = match.groups()
                dims.append((kind, int(n), float(spacing) if spacing else 1.0))
            key, colon, value = line.partition(":")
            if colon:
                header.setdefault(section, {})[key.strip()] = value.strip()
        if not dims:
            raise ReaderError(f"{path.name}: no dimension lines")
        return header, dims
