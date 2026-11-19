####################################################################################################
#                                            varian.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for Varian/Agilent .fid directories (fid and procpar), read through nmrglue:     #
#          spectroscopy, imaging scans as raw k-space, NMR spectrometer data; the vendor header    #
#          keeps every procpar parameter, the text file and the log.                               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import nmrglue as ng
import numpy as np

# own
from miox import __version__
from miox.data import MRSData
from miox.readers.base import Reader, ReaderError
from miox.readers.varian_raw import VarianRaw
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                           VarianReader                                           #
#**************************************************************************************************#
#                                                                                                  #
# Reads one .fid directory. procpar values stay the strings written in the file; times in procpar  #
# are in s (delays, te, tr) or µs (pulses, alfa). Imaging scans (apptype im2D* or im3D*, or phase  #
# encoding without an apptype) go to VarianRaw, NMR spectrometer data (an apptype not of an        #
# imaging console, or none: older VNMR liquids data) to nmrglue, imaging-console spectroscopy      #
# (im1D...) is laid out here: points x receivers (rcvrs y) x arraydim / receivers, the FIDs as     #
# nmrglue reads them (equal to the NIfTI-MRS convention), echo time te, else the pulse (p1, else   #
# pw) plus the acquisition delay alfa, no geometry (a 10 m voxel). On a spectrometer the carrier   #
# lies at (sw / 2 - rfl + rfp) / sfrq ppm: rfl is the frequency of the reference line from the low #
# end of the spectrum, rfp the frequency it is assigned.                                           #
#                                                                                                  #
#**************************************************************************************************#
class VarianReader(Reader):
    FORMATS = (Format.VARIAN,)
    PACKAGES = Reader.PACKAGES + ("nmrglue",)
    NUCLEUS = re.compile(r"^([A-Za-z]+)(\d+)$")  # procpar spelling: H1, C13
    INDIRECT = (("ni", "sw1", "phase"), ("ni2", "sw2", "phase2"), ("ni3", "sw3", "phase3"))

    def read(self, sniff: Sniff) -> list:
        folder = sniff.path if sniff.path.is_dir() else sniff.path.parent
        header = self._header(folder)
        if VarianRaw.is_imaging(header["procpar"]):
            return VarianRaw(folder, header, self.provenance(sniff)).datasets()
        if self.is_spectrometer(header["procpar"]):
            return [self._spectrometer(sniff, folder, header)]
        return [self._console(sniff, folder, header)]

    @staticmethod
    def is_spectrometer(procpar: dict[str, list[str]]) -> bool:
        """NMR spectrometer data: no apptype (older VNMR), or one that is not an imaging
        console's (im1D, im2D, ...); imaging scans are sorted out before."""
        apptype = (procpar.get("apptype") or [""])[0]
        return not apptype.startswith("im")

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
        echo = self._echo_time(procpar, pulse_acquire=False)
        if echo is not None:
            values["EchoTime"] = echo
        if "temp" in procpar and (procpar.get("vttype") or ["0"])[0] != "0":
            values["SampleTemperature"] = (number("temp") + 273.15, "procpar/temp (°C)")
        tags, indirect, arrayed = [], 0, ["DIM_DYN", "DIM_USER_0", "DIM_USER_1"]
        for axis in self._axes(procpar, data.ndim - 1):
            if axis is None:
                names = (procpar.get("array") or [""])[0] or "unnamed"
                tags.append((arrayed.pop(0), f"array elements ({names}) as acquired", None))
                continue
            count, sw, phase = axis
            phases = ",".join(procpar.get(phase) or ["1"])
            tags.append(("DIM_INDIRECT_" + str(indirect),
                         f"{count} {procpar[count][0]} x {phase} {phases}, {sw} "
                         f"{procpar[sw][0]} Hz, rows as acquired; nucleus not stated", None))
            indirect += 1
        if indirect:
            notes.append("procpar does not state the nuclei of the indirect dimensions")
        data = data.reshape((1, 1, 1) + data.shape)
        source = "procpar"
        d = self.dataset(sniff, data, 1 / width, values, header, folder.stem, notes, source,
                         dim_tags=tuple(tags))
        d.meta.set("DwellTime", 1 / width, "procpar: 1 / sw", "procpar/sw")
        return d

    def _axes(self, procpar: dict, count: int) -> list[tuple[str, str, str] | None]:
        """What each of the ``count`` dimensions after the direct one runs over, in nmrglue's
        order: the innermost arrayed parameter (None; an arrayed phase belongs to its
        increments), then every increment count procpar states, as (ni, sw1, phase). Where that
        does not give ``count`` dimensions, the increment counts in turn. An increment count of
        at most 1, or one without its spectral width, is no indirect dimension: None, the rows
        being array elements."""
        def stated(axis: tuple[str, str, str]) -> tuple[str, str, str] | None:
            increments, width = axis[0], axis[1]
            try:
                several = float(procpar[increments][0]) > 1
            except (KeyError, IndexError, ValueError):
                return None
            return axis if several and width in procpar else None

        inner = (procpar.get("array") or [""])[0].split(",")[-1]
        axes = [None] if inner in procpar and not inner.startswith("phase") else []
        axes += [stated(axis) for axis in self.INDIRECT if axis[0] in procpar]
        if len(axes) != count:
            axes = [stated(axis) for axis in self.INDIRECT[:count]]
        return axes

    def _console(self, sniff: Sniff, folder: Path, header: dict) -> MRSData:
        """Imaging-console spectroscopy: (points, receivers, transients); of an acquisition
        that was stopped, the transients the file holds."""
        notes: list[str] = []
        blocks, file_header, _ = VarianRaw(folder, header, self.provenance(sniff))._blocks(notes)
        procpar = header["procpar"]
        receivers = sum(1 for flag in procpar["rcvrs"][0] if flag == "y")
        transients = float(procpar["arraydim"][0]) / receivers
        if not transients.is_integer():
            raise ReaderError(f"{folder}: arraydim {procpar['arraydim'][0]} is not a multiple of "
                              f"{receivers} receivers")
        points, transients = file_header["np"] // 2, int(transients)
        data = blocks.reshape(-1, points)  # one row per block (and trace)
        if data.shape[0] % receivers:
            raise ReaderError(f"{folder}: {data.shape[0]} blocks do not hold {receivers} "
                              "receivers each")
        acquired = data.shape[0] // receivers
        if acquired != transients:
            notes.append(f"{acquired} transients in the file, arraydim states {transients}")
        data = np.ascontiguousarray(
            data.transpose().reshape((1, 1, 1, points, receivers, acquired)))
        if "press" in procpar["seqfil"][0] or "steam" in procpar["seqfil"][0]:
            notes.append("voxel position (pos1-3, vox1-3, vpsi/vtheta/vphi) not read: no checked "
                         "sample")
        values = {
            "SpectrometerFrequency": ([float(procpar["sfrq"][0])], "procpar/sfrq"),
            "ResonantNucleus": ([self.NUCLEUS.sub(r"\2\1", procpar["tn"][0])], "procpar/tn"),
            "RepetitionTime": (float(procpar["tr"][0]), "procpar/tr"),
            "Manufacturer": ("Varian", None),
            "ProtocolName": (procpar["seqfil"][0], "procpar/seqfil"),
            "PatientName": (procpar["comment"][0], "procpar/comment"),
            "ConversionMethod": (f"miox {__version__}", None),
            "ConversionTime": (datetime.now().isoformat(timespec="milliseconds"), None),
            "OriginalFile": ([folder.name], None),
            "kSpace": ([False, False, False], None),
        }
        for key, name, kind in (("SoftwareVersions", "parver", str), ("TxCoil", "rfcoil", str),
                                ("RxCoil", "rfcoil", str), ("InversionTime", "ti", float),
                                ("ExcitationFlipAngle", "flip1", float)):
            if name in procpar:
                values[key] = (kind(procpar[name][0]), f"procpar/{name}")
        echo = self._echo_time(procpar, pulse_acquire=True)
        if echo is not None:
            values["EchoTime"] = echo
        width = float(procpar["sw"][0])
        values["DwellTime"] = (1.0 / width, "procpar/sw")
        return self.dataset(sniff, data, 1.0 / width, values, header, folder.stem, notes,
                            "procpar", dim_tags=("DIM_COIL", "DIM_DYN"))

    @staticmethod
    def _echo_time(procpar: dict, pulse_acquire: bool) -> tuple[float, str] | None:
        """te (s); for a pulse-acquire sequence without one, the pulse (p1, else pw) plus the
        acquisition delay alfa, both in µs."""
        if "te" in procpar:
            return float(procpar["te"][0]), "procpar/te"
        pulse = "p1" if "p1" in procpar else "pw"
        if pulse_acquire and "alfa" in procpar and pulse in procpar:
            delay = (float(procpar[pulse][0]) + float(procpar["alfa"][0])) * 1e-6
            return delay, f"procpar/{pulse}"
        return None

    def _header(self, folder: Path) -> dict:
        procpar = ng.varian.read_procpar(str(folder / "procpar"))
        header = {"procpar": {name: p["values"] for name, p in procpar.items()}}
        for name in ("text", "log"):
            if (folder / name).is_file():
                header[name] = (folder / name).read_text(errors="replace").splitlines()
        return header
