####################################################################################################
#                                           fida_mat.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for FID-A structs saved in .mat files (out, outw, outA, ...): fids with the axes #
#          named in dims, txfrq (Hz), dwelltime (s), te and tr (ms). The vendor header keeps every #
#          struct, large arrays summarised.                                                        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import numpy as np

# own
from miox.data import MRSData
from miox.matfile import MatFile
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                            FidaReader                                            #
#**************************************************************************************************#
#                                                                                                  #
# One MRSData per FID-A struct. FID-A keeps FIDs as the conjugate of NIfTI-MRS (its spectra are    #
# fftshift(ifft(fids))): checked against an LCModel basis of the same spectra and against the ppm  #
# axis the files store. The axes named in dims (1-based, 0 when absent) become NIfTI-MRS           #
# dimensions in the order coils, averages, sub-spectra, extras. Without txfrq the frequency is Bo  #
# times gamma (FID-A's 42.577 MHz/T when not stored); without nucleus 1H is assumed.               #
#                                                                                                  #
#**************************************************************************************************#
class FidaReader(Reader):
    FORMATS = (Format.FIDA_MAT,)
    PACKAGES = Reader.PACKAGES + ("scipy", "h5py")
    SOURCE = "FID-A struct"
    NUCLEUS = "1H"
    GAMMA = 42.577e6  # Hz/T
    DIMS = (  # FID-A dims field, NIfTI-MRS tag and info
        ("coils", "DIM_COIL", None),
        ("averages", "DIM_DYN", None),
        ("subSpecs", "DIM_USER_1", "FID-A sub-spectra"),
        ("extras", "DIM_USER_0", "FID-A extras"),
    )

    def read(self, sniff: Sniff) -> list[MRSData]:
        values = MatFile.load(sniff.path)
        structs = {
            name: value
            for name, value in values.items()
            if name.startswith("out") and isinstance(value, dict) and "fids" in value
        }
        if not structs:
            raise ReaderError(f"{sniff.path.name}: no FID-A struct (out... with fids)")
        vendor = MatFile.header(values)
        single = len(structs) == 1
        return [
            self._dataset(sniff, name, out, vendor, sniff.path.stem if single else name)
            for name, out in structs.items()
        ]

    def _dataset(self, sniff: Sniff, var: str, out: dict, vendor: dict, name: str) -> MRSData:
        fids = np.conj(np.asarray(out["fids"]))
        if "sz" in out:
            fids = fids.reshape(tuple(int(n) for n in np.ravel(out["sz"])))
        dims = out.get("dims", {}) if isinstance(out.get("dims"), dict) else {}
        axes = [int(dims.get("t", 1) or 1) - 1]
        tags = []
        for field, tag, info in self.DIMS:
            index = int(dims.get(field, 0) or 0)
            if index > 0 and index - 1 < fids.ndim:
                axes.append(index - 1)
                tags.append((tag, info, None))
        axes += [a for a in range(fids.ndim) if a not in axes and fids.shape[a] == 1]
        if len(axes) != fids.ndim:
            raise ReaderError(f"{sniff.path.name}: {var}.dims does not name every axis of fids")
        data = np.transpose(fids, axes).reshape(1, 1, 1, *[fids.shape[a] for a in axes])
        data = data.reshape(data.shape[: 4 + len(tags)])
        notes = ["stored as the conjugate (FID-A convention); conjugated"]
        keys = {"ResonantNucleus": ([str(out.get("nucleus") or self.NUCLEUS)], f"{var}/nucleus")}
        if "txfrq" in out:
            keys["SpectrometerFrequency"] = ([float(out["txfrq"]) / 1e6], f"{var}/txfrq")
        else:
            gamma = float(out.get("gamma", self.GAMMA))
            keys["SpectrometerFrequency"] = ([float(out["Bo"]) * gamma / 1e6], f"{var}/Bo")
            notes.append(f"frequency from Bo x gamma ({gamma:.6g} Hz/T)")
        for key, field in (("EchoTime", "te"), ("RepetitionTime", "tr")):
            if float(out.get(field, 0) or 0) > 0:
                keys[key] = (float(out[field]) * 1e-3, f"{var}/{field}")
        if not out.get("nucleus"):
            notes.append(f"nucleus not stored; {self.NUCLEUS} assumed")
        dwelltime = float(out["dwelltime"])
        d = self.dataset(
            sniff, data, dwelltime, keys, vendor, name, notes, self.SOURCE, dim_tags=tuple(tags)
        )
        if not out.get("nucleus"):
            d.meta.set("ResonantNucleus", [self.NUCLEUS], f"assumed: {self.NUCLEUS}")
        d.meta.set("DwellTime", dwelltime, f"{self.SOURCE}: {var}/dwelltime", f"{var}/dwelltime")
        return d
