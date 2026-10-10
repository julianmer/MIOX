####################################################################################################
#                                            topspin.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 09/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for Bruker TopSpin (and XWIN-NMR) experiments through nmrglue: the acquired FIDs #
#          (fid, ser) with their acquNs parameters, and the processed spectra of pdata/<n> (1r/1i, #
#          2rr, ...) with their procNs parameters.                                                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import nmrglue as ng
import numpy as np

# own
from miox.data import MRImage, MRSData, Protocol
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff, Sniffer


#**************************************************************************************************#
#                                          TopSpinReader                                           #
#**************************************************************************************************#
#                                                                                                  #
# An experiment folder gives its FID (fid, or ser for nD) and every processed spectrum under       #
# pdata, or its parameters as a Protocol when it holds neither; a pdata/<n> folder gives its       #
# spectrum alone; a folder of experiments gives all of theirs. nmrglue lays out the FIDs with the  #
# direct dimension last and as Bruker stores them; here the direct dimension comes first, each     #
# further dimension (acqu2s, acqu3s, ...) on dims 5 to 7 as DIM_INDIRECT_0 to 2, rows as acquired. #
# FIDs are conjugated into the NIfTI-MRS convention: their spectra then lie on the chemical shift  #
# axis of TopSpin's own processing. Each FID row is cut to TD / 2 complex points (Bruker pads rows #
# to 1024-byte blocks) and an indirect dimension to its TD (an acquisition stopped early keeps the #
# rows it allocated). The first GroupDelay points of a digitally filtered FID hold the filter's    #
# delay; the FIDs are stored as acquired, integers unscaled (acqus NC). Processed spectra are      #
# scaled by 2^NC_proc as TopSpin shows them and keep TopSpin's order: the chemical shift falls     #
# with the index (frequency rises), the centre point (index SI / 2) lies at ReferencePPM.          #
#                                                                                                  #
#**************************************************************************************************#
class TopSpinReader(Reader):
    FORMATS = (Format.TOPSPIN, Format.TOPSPIN_PROCESSED)
    PACKAGES = Reader.PACKAGES + ("nmrglue",)
    ACQUS = ("acqus", "acqu2s", "acqu3s", "acqu4s")  # direct dimension first
    PROCS = ("procs", "proc2s", "proc3s", "proc4s")
    INDIRECT = ("DIM_INDIRECT_0", "DIM_INDIRECT_1", "DIM_INDIRECT_2")
    COMPONENTS = {  # real part first, then the other quadrature components
        1: ("1r", "1i"),
        2: ("2rr", "2ri", "2ir", "2ii"),
        3: ("3rrr", "3rri", "3rir", "3rii", "3irr", "3iri", "3iir", "3iii"),
    }
    QUADRATURE = {0: "qf", 1: "qsim", 2: "qseq", 3: "DQD"}  # acqus AQ_mod
    VERSION = re.compile(r"##TITLE=\s*Parameter file,\s*(.+)")
    ORIGIN = re.compile(r"##ORIGIN=\s*(?:UXNMR,\s*)?(.+)")
    TEXT_FILES = ("pulseprogram", "audita.txt")  # kept in the vendor header, as lines

    def read(self, sniff: Sniff) -> list[MRSData | MRImage | Protocol]:
        folder = sniff.path if sniff.path.is_dir() else sniff.path.parent
        if sniff.format == Format.TOPSPIN_PROCESSED:
            return [self._processed(sniff, folder)]
        if not (folder / "acqus").is_file():  # a folder of experiments
            sniffer = Sniffer()
            return [d for e in sniff.companions for d in self.read(sniffer.sniff(e))]
        datasets = []
        if (folder / "fid").is_file() or (folder / "ser").is_file():
            datasets.append(self._fid(sniff, folder))
        pdata = folder / "pdata"
        processed = sorted(
            (p for p in pdata.iterdir() if p.is_dir() and self._components(p)),
            key=lambda p: (not p.name.isdigit(), int(p.name) if p.name.isdigit() else 0, p.name),
        ) if pdata.is_dir() else []
        datasets += [self._processed(sniff, p) for p in processed]
        return datasets or [self._parameters(sniff, folder)]

    def _parameters(self, sniff: Sniff, folder: Path) -> Protocol:
        """An experiment without data: its acquisition parameters."""
        dic = ng.bruker.read_acqus_file(str(folder))
        acqus = [dic[name] for name in self.ACQUS if name in dic]
        meta = Meta(self._vendor(folder, dic, self.ACQUS))
        for key, (value, field) in self._acquisition(dic, acqus, len(acqus)).items():
            meta.set(key, value, f"TopSpin acquisition parameters: {field}", field)
        notes = ("parameters only: no fid, ser or processed spectrum",)
        name = f"{folder.parent.name}_{folder.name}"
        return Protocol(name, (), meta, self.provenance(sniff, notes))

    # --- FIDs ------------------------------------------------------------------------------------

    def _fid(self, sniff: Sniff, folder: Path) -> MRSData:
        notes = []
        with self.captured(notes):
            dic, data = ng.bruker.read(str(folder), read_procs=False)
        acqus = [dic[name] for name in self.ACQUS if name in dic]
        direct = acqus[0]
        mode = int(direct.get("AQ_mod", 3))
        if mode in (0, 2):
            notes.append(f"{self.QUADRATURE[mode]} acquisition: real samples, stored as the real "
                         "part; the spectrum needs a real (Redfield) transform")
        data = np.asarray(data, dtype=np.complex128)
        points = int(direct["TD"]) // (2 if mode in (1, 3) else 1)
        if data.shape[-1] > points:
            data = data[..., :points]
        data = np.conj(data).T  # direct dimension first, then acqu2s, acqu3s, ...
        for axis, parameters in enumerate(acqus[1:], start=1):
            rows = int(parameters.get("TD", 0))
            if axis < data.ndim and 0 < rows < data.shape[axis]:
                notes.append(f"{self.ACQUS[axis]} TD {rows}: {data.shape[axis]} stored rows cut "
                             "to the acquired ones")
                data = data.take(range(rows), axis=axis)
        if data.ndim > len(acqus):
            notes.append(f"{data.ndim} dimensions, parameters of {len(acqus)}: the shape of the "
                         "others comes from the pulse program loops")
        data = data.reshape((1, 1, 1) + data.shape)
        if data.ndim > 7:
            raise ReaderError(f"{folder}: {data.ndim - 3} dimensions, NIfTI-MRS holds at most 4")
        values = self._acquisition(dic, acqus, data.ndim - 3)
        values.update(self._reference(folder, direct))
        delay = self._group_delay(direct)
        if delay:
            values["GroupDelay"] = delay
        dwelltime = 1.0 / float(direct["SW_h"])
        tags = tuple(
            (self.INDIRECT[i], self._dimension_info(acqus, i + 1), None)
            for i in range(data.ndim - 4)
        )
        vendor = self._vendor(folder, dic, self.ACQUS)
        name = f"{folder.parent.name}_{folder.name}"
        source = "TopSpin acquisition parameters"
        d = self.dataset(sniff, data, dwelltime, values, vendor, name, notes, source,
                         dim_tags=tags)
        d.meta.set("DwellTime", dwelltime, "acqus: 1 / SW_h", "acqus/SW_h")
        return d

    def _acquisition(self, dic: dict, acqus: list[dict], ndim: int) -> dict[str, tuple]:
        """Standard keys from the acquisition parameters; frequencies and nuclei per dimension
        that has its acquNs file."""
        direct = acqus[0]
        stated = acqus[:ndim]
        values = {
            "SpectrometerFrequency": ([float(a["SFO1"]) for a in stated], "acqus/SFO1"),
            "ResonantNucleus": ([str(a["NUC1"]) for a in stated], "acqus/NUC1"),
        }
        for key, field in (("SequenceName", "PULPROG"), ("Solvent", "SOLVENT")):
            if text := str(direct.get(field, "")).strip():
                values[key] = (text, f"acqus/{field}")
        if probe := str(direct.get("PROBHD", "")).strip():
            values["TxCoil"] = (probe, "acqus/PROBHD")
            values["RxCoil"] = (probe, "acqus/PROBHD")
        if "NS" in direct:
            values["NumberOfAverages"] = (int(direct["NS"]), "acqus/NS")
        if float(direct.get("TE", 0)) > 0:
            values["SampleTemperature"] = (float(direct["TE"]), "acqus/TE")
        values.update(self._core(direct, "acqus"))
        return values

    def _reference(self, folder: Path, direct: dict) -> dict[str, tuple]:
        """ReferencePPM: the chemical shift at the carrier SFO1 on the axis of the first
        processing (pdata/1 procs SF, the frequency of 0 ppm), else on the default axis whose
        0 ppm lies at the basic frequency BF1."""
        procs = folder / "pdata" / "1" / "procs"
        if procs.is_file():
            frequency = float(ng.bruker.read_jcamp(str(procs))["SF"])
            field = "pdata/1/procs/SF"
        else:
            frequency, field = float(direct["BF1"]), "acqus/BF1"
        shift = (float(direct["SFO1"]) - frequency) / frequency * 1e6
        return {"ReferencePPM": (shift, f"(acqus/SFO1 - {field}) / {field} * 1e6")}

    @staticmethod
    def _group_delay(direct: dict) -> tuple[float, str] | None:
        """Points the digital filter delays the FID by: GRPDLY (DSPFVS 20 and later), else the
        table nmrglue keeps for the older firmware by DSPFVS and DECIM."""
        if float(direct.get("GRPDLY", -1)) > 0:
            return float(direct["GRPDLY"]), "acqus/GRPDLY"
        version, decimation = int(direct.get("DSPFVS", 0)), direct.get("DECIM")
        table = ng.bruker.bruker_dsp_table.get(version, {})
        if decimation is not None and int(decimation) in table:
            return float(table[int(decimation)]), "acqus/DSPFVS and DECIM, nmrglue's table"
        return None

    def _dimension_info(self, acqus: list[dict], axis: int) -> str:
        """The indirect dimension in words: nucleus, spectral width, frequency, acquisition mode."""
        if axis >= len(acqus):
            return f"no {self.ACQUS[axis]}"
        a = acqus[axis]
        width = float(a["SW"]) * float(a["SFO1"])
        return (f"{self.ACQUS[axis]}: {a['NUC1']}, SW {width:.6g} Hz at SFO1 "
                f"{float(a['SFO1']):.9g} MHz, FnMODE {a.get('FnMODE')}, rows as acquired")

    # --- processed spectra -----------------------------------------------------------------------

    def _processed(self, sniff: Sniff, folder: Path) -> MRImage:
        components = self._components(folder)
        if not components:
            raise ReaderError(f"{folder}: no processed spectrum (1r, 2rr, ...)")
        notes = []
        with self.captured(notes):
            dic, parts = ng.bruker.read_pdata(
                str(folder), bin_files=list(components), all_components=True, scale_data=True
            )
        procs = [dic[name] for name in self.PROCS if name in dic]
        parts = parts if isinstance(parts, list) else [parts]  # one component: an array
        parts = [np.asarray(p, dtype=np.float64).T for p in parts]  # direct dimension first
        dims = ("frequency",) + tuple(f"frequency_indirect_{i}" for i in range(parts[0].ndim - 1))
        if len(parts) == 2 and parts[0].ndim == 1:
            data = parts[0] + 1j * parts[1]
        elif len(parts) > 1:
            data = np.stack(parts, axis=-1)
            dims += ("component",)
            notes.append(f"quadrature components on the last axis: {', '.join(components)}")
        else:
            data = parts[0]
        data = data.reshape((1, 1, 1) + data.shape)
        direct = procs[0]
        width, frequency = float(direct["SW_p"]), float(direct["SF"])
        meta = Meta(self._vendor(folder, dic, self.PROCS + self.ACQUS))
        source = "TopSpin processing parameters"
        meta.set("SpectrometerFrequency", [float(p["SF"]) for p in procs], f"{source}: SF", "SF")
        if all(str(p.get("AXNUC", "")).strip() for p in procs):
            meta.set("ResonantNucleus", [str(p["AXNUC"]) for p in procs],
                     f"{source}: AXNUC", "AXNUC")
        meta.set("NumberOfPoints", int(data.shape[3]), "data shape")
        meta.set("SpectralWidth", width, f"{source}: SW_p", "procs/SW_p")
        meta.set("DwellTime", 1 / width, "derived: 1 / SpectralWidth")
        reference = float(direct["OFFSET"]) - width / (2 * frequency)
        meta.set("ReferencePPM", reference, f"{source}: OFFSET - SW_p / (2 SF)", "procs/OFFSET")
        if "acqus" in dic:
            for key, (value, field) in self._core(dic["acqus"], "acqus").items():
                meta.set(key, value, f"TopSpin acquisition parameters: {field}", field)
        indirect = procs[1:data.ndim - 3]
        if indirect:
            shifts = [float(p["OFFSET"]) - float(p["SW_p"]) / (2 * float(p["SF"]))
                      for p in indirect]
            meta.set("IndirectReferencePPM", shifts,
                     f"{source}: proc2s, ... OFFSET - SW_p / (2 SF)", "proc2s/OFFSET")
            meta.set("IndirectSpectralWidth", [float(p["SW_p"]) for p in indirect],
                     f"{source}: proc2s, ... SW_p", "proc2s/SW_p")
        experiment = folder.parent.parent
        name = f"{experiment.parent.name}_{experiment.name}_pdata{folder.name}"
        return MRImage(data, self.NO_GEOMETRY, meta, self.provenance(sniff, tuple(notes)), name,
                       dims)

    def _components(self, folder: Path) -> tuple[str, ...]:
        """The processed data files in ``folder``: the real part and the further components."""
        for names in self.COMPONENTS.values():
            if (folder / names[0]).is_file():
                return tuple(n for n in names if (folder / n).is_file())
        return ()

    # --- header ----------------------------------------------------------------------------------

    def _core(self, parameters: dict, name: str) -> dict[str, tuple]:
        """Manufacturer and software version from the first lines of a parameter file."""
        values = {}
        for line in parameters.get("_coreheader", []):
            if match := self.VERSION.match(line):
                version = re.sub(r"\s+", " ", match.group(1)).strip()
                values["SoftwareVersions"] = (version, f"{name}/TITLE")
            elif match := self.ORIGIN.match(line):
                values["Manufacturer"] = (match.group(1).strip(), f"{name}/ORIGIN")
        return values

    def _vendor(self, folder: Path, dic: dict, names: tuple[str, ...]) -> dict[str, Any]:
        """The parameter files in ``dic`` and the text files of the folder."""
        vendor = {name: self.plain(dic[name]) for name in names if name in dic}
        for text in self.TEXT_FILES + ("title",):
            if (folder / text).is_file():
                vendor[text] = (folder / text).read_text(errors="replace").splitlines()
        return vendor
