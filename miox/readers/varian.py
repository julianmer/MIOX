####################################################################################################
#                                            varian.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for Varian/Agilent .fid directories (fid and procpar): spectroscopy through      #
#          spec2nii, imaging scans as raw k-space, NMR spectrometer data through nmrglue; the      #
#          vendor header keeps every procpar parameter, the text file and the log.                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re
from argparse import Namespace
from pathlib import Path

import nmrglue as ng
import numpy as np
from nifti_mrs.nifti_mrs import NIFTI_MRS
from spec2nii.varian import read_procpar
from spec2nii.varian_importer import read_varian

# own
from miox.data import MRSData
from miox.readers.base import ReaderError, Spec2niiReader
from miox.readers.varian_raw import VarianRaw
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                           VarianReader                                           #
#**************************************************************************************************#
#                                                                                                  #
# Reads one .fid directory. procpar values stay the strings written in the file; times in procpar  #
# are in s (delays, te, tr) or µs (pulses, alfa). Imaging scans (apptype im2D* or im3D*, or phase  #
# encoding without an apptype) go to VarianRaw, NMR spectrometer data (an apptype not of an        #
# imaging console) to nmrglue, everything else to spec2nii. On a spectrometer the carrier lies at  #
# (sw / 2 - rfl + rfp) / sfrq ppm: rfl is the frequency of the reference line from the low end of  #
# the spectrum, rfp the frequency it is assigned.                                                  #
#                                                                                                  #
#**************************************************************************************************#
class VarianReader(Spec2niiReader):
    FORMATS = (Format.VARIAN,)
    CONVERTER = "varian"
    SOURCES = {
        "SpectrometerFrequency": "procpar/sfrq",
        "ResonantNucleus": "procpar/tn",
        "EchoTime": ("procpar/te", "procpar/p1", "procpar/pw"),
        "RepetitionTime": "procpar/tr",
        "InversionTime": "procpar/ti",
        "ExcitationFlipAngle": "procpar/flip1",
        "ProtocolName": "procpar/seqfil",
        "PatientName": "procpar/comment",
        "SoftwareVersions": "procpar/parver",
        "TxCoil": "procpar/rfcoil",
        "RxCoil": "procpar/rfcoil",
        "DwellTime": "procpar/sw",
    }

    PACKAGES = Spec2niiReader.PACKAGES + ("nmrglue",)
    NUCLEUS = re.compile(r"^([A-Za-z]+)(\d+)$")  # procpar spelling: H1, C13
    INDIRECT = (("ni", "sw1", "phase"), ("ni2", "sw2", "phase2"), ("ni3", "sw3", "phase3"))

    def read(self, sniff: Sniff) -> list:
        folder = sniff.path if sniff.path.is_dir() else sniff.path.parent
        header = self._header(folder)
        if VarianRaw.is_imaging(header["procpar"]):
            return VarianRaw(folder, header, self.provenance(sniff)).datasets()
        if self.is_spectrometer(header["procpar"]):
            return [self._spectrometer(sniff, folder, header)]
        return super().read(sniff)

    @staticmethod
    def is_spectrometer(procpar: dict[str, list[str]]) -> bool:
        """NMR spectrometer data: an apptype that is not an imaging console's (im1D, im2D, ...)."""
        apptype = (procpar.get("apptype") or [""])[0]
        return bool(apptype) and not apptype.startswith("im")

    def _spectrometer(self, sniff: Sniff, folder: Path, header: dict) -> MRSData:
        """The FIDs as nmrglue lays them out (already in the NIfTI-MRS convention), the direct
        dimension first and each indirect one (ni, ni2) on dims 5 to 7 with its rows as
        acquired: increments times the phases of the arrayed phase parameters."""
        notes = []
        with self.captured(notes):
            _, data = ng.varian.read(str(folder))
        data = np.asarray(data, dtype=np.complex128).T
        if data.ndim > 4:
            raise ReaderError(f"{folder}: {data.ndim} dimensions, NIfTI-MRS holds at most 4")
        procpar = header["procpar"]
        number = lambda name: float(procpar[name][0])  # noqa: E731
        width, frequency = number("sw"), number("sfrq")
        nucleus = self.NUCLEUS.sub(r"\2\1", procpar["tn"][0])
        carrier = width / 2 - number("rfl") + number("rfp")  # Hz above 0 ppm
        values = {
            "SpectrometerFrequency": ([frequency], "procpar/sfrq"),
            "ResonantNucleus": ([nucleus], "procpar/tn"),
            "ReferencePPM": (
                carrier / frequency,
                "(procpar/sw / 2 - procpar/rfl + procpar/rfp) / procpar/sfrq",
            ),
        }
        if "reffrq" in procpar and abs((frequency - number("reffrq")) * 1e6 - carrier) > 1:
            stated = (frequency - number("reffrq")) * 1e6 / frequency
            notes.append(f"procpar reffrq puts the carrier at {stated:.4f} ppm, rfl and rfp at "
                         f"{carrier / frequency:.4f} ppm (used)")
        for key, name in (("SequenceName", "seqfil"), ("Solvent", "solvent"),
                          ("SoftwareVersions", "parver"), ("ManufacturersModelName", "console"),
                          ("TxCoil", "probe_"), ("RxCoil", "probe_")):
            if (text := (procpar.get(name) or [""])[0].strip()):
                values[key] = (text, f"procpar/{name}")
        if "nt" in procpar:
            values["NumberOfAverages"] = (int(number("nt")), "procpar/nt")
        if "temp" in procpar and (procpar.get("vttype") or ["0"])[0] != "0":
            values["SampleTemperature"] = (number("temp") + 273.15, "procpar/temp (°C)")
        tags = []
        for count, sw, phase in self.INDIRECT[: data.ndim - 1]:
            phases = ",".join(procpar.get(phase) or ["1"])
            tags.append(("DIM_INDIRECT_" + str(len(tags)),
                         f"{count} {procpar[count][0]} x {phase} {phases}, {sw} "
                         f"{procpar[sw][0]} Hz, rows as acquired; nucleus not stated", None))
        if tags:
            notes.append("procpar does not state the nuclei of the indirect dimensions")
        data = data.reshape((1, 1, 1) + data.shape)
        source = "procpar"
        d = self.dataset(sniff, data, 1 / width, values, header, folder.stem, notes, source,
                         dim_tags=tuple(tags))
        d.meta.set("DwellTime", 1 / width, "procpar: 1 / sw", "procpar/sw")
        return d

    def convert(self, sniff: Sniff) -> tuple[list[NIFTI_MRS], list[str], dict]:
        folder = sniff.path if sniff.path.is_dir() else sniff.path.parent
        args = Namespace(file=str(folder), tag6="DIM_DYN", dump_headers=False, fileout=None)
        images, names = read_varian(args)
        return list(images), list(names), self._header(folder)

    def correct(self, d: MRSData) -> None:
        """spec2nii overwrites te with a pulse time plus alfa, both in µs, stored as s."""
        super().correct(d)
        procpar = d.meta.vendor["procpar"]
        if "te" in procpar:
            te = float(procpar["te"][0])
            d.meta.set("EchoTime", te, "procpar/te (s)", "procpar/te")
        elif "alfa" in procpar and "EchoTime" in d.meta:
            pulse = "p1" if "p1" in procpar else "pw"
            delay = (float(procpar[pulse][0]) + float(procpar["alfa"][0])) * 1e-6
            d.meta.set(
                "EchoTime",
                delay,
                f"procpar/{pulse} + procpar/alfa in µs, converted to s "
                "(acquisition delay of a pulse-acquire sequence)",
                f"procpar/{pulse}",
            )

    def _header(self, folder: Path) -> dict:
        procpar = read_procpar(str(folder / "procpar"))
        header = {"procpar": {name: p["values"] for name, p in procpar.items()}}
        for name in ("text", "log"):
            if (folder / name).is_file():
                header[name] = (folder / name).read_text(errors="replace").splitlines()
        return header
