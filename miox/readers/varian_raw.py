####################################################################################################
#                                          varian_raw.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/10/26                                                                                #
#                                                                                                  #
# Purpose: Raw k-space of Varian/Agilent VnmrJ imaging scans: the fid file of a .fid folder as     #
#          readouts with their encoding counters, laid out by the procpar's seqcon.                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import math
import re
import struct
from pathlib import Path

import numpy as np

# own
from miox.data import Provenance, RawData
from miox.meta import Meta
from miox.readers.base import ReaderError


#**************************************************************************************************#
#                                            VarianRaw                                             #
#**************************************************************************************************#
#                                                                                                  #
# A fid file starts with a 32-byte header (big-endian: blocks, traces per block, values per trace, #
# bytes per value, per trace and per block, version, status, block headers per block); every block #
# holds its 28-byte headers, then its traces, real and imaginary interleaved, as 16-bit or 32-bit  #
# integers or 32-bit floats (status bits 0x4 and 0x8). seqcon names for the loops echo, slice,     #
# phase encoding 1, 2 and 3 whether each is compressed into the traces of a block ('c'), runs over #
# the blocks ('s') or is absent ('n'); compressed and standard loops each nest in that order, the  #
# first innermost (multi-echo 2D and 3D scans reconstruct so to the published maps), and the       #
# arrayed parameters (array, arraydim) run over the blocks. Unverified for want of samples: arrays #
# taken to run outside the standard loops, several receivers taken as blocks following each other  #
# for every array element, and fast spin-echo scans (im2Dfse) holding per block the echoes of a    #
# train innermost, then the echo images, the slices and the trains, as vnmrjpy lays them out. The  #
# counters: echo as contrast, the slice by its position (pss) with the acquisition slot in user    #
# counter 1, phase encodings in acquisition order as the encoding steps (the line order of a scan  #
# with a petable is that table's, which the folder does not hold), the position in an echo train   #
# as segment, the array element as repetition. Echo times come from the TE list (ms) when it has   #
# one value per echo, else from te; nothing is reordered or corrected, and the slice geometry      #
# (psi, phi, theta, pro, ppe, pss) stays in the procpar.                                           #
#                                                                                                  #
#**************************************************************************************************#
class VarianRaw:
    SOURCE = "Varian procpar"
    S_32, S_FLOAT = 0x4, 0x8
    LOOPS = ("echo", "slice", "pe1", "pe2", "pe3")  # the positions of seqcon
    SIZES = {"echo": "ne", "slice": "ns", "pe1": "nv", "pe2": "nv2", "pe3": "nv3"}
    COUNTERS = {
        "echo": "contrast", "slice": "slice", "pe1": "kspace_encode_step_1",
        "pe2": "kspace_encode_step_2",
    }
    LIMITS = {
        "pe1": "kspace_encoding_step_1", "pe2": "kspace_encoding_step_2", "echo": "contrast",
        "slice": "slice",
    }
    KEYS = {  # standard key: (procpar parameter, factor to the key's unit)
        "SpectrometerFrequency": ("sfrq", None),
        "RepetitionTime": ("tr", None),
        "ExcitationFlipAngle": ("flip1", None),
        "NumberOfAverages": ("nt", None),
        "EchoTrainLength": ("etl", None),
        "SliceThickness": ("thk", None),
        "PulseSequenceName": ("seqfil", None),
        "SeriesDescription": ("comment", None),
        "PatientName": ("name", None),
        "PatientID": ("ident", None),
        "StudyID": ("studyid_", None),
        "ManufacturersModelName": ("console", None),
        "SoftwareVersions": ("parver", None),
        "TxCoil": ("rfcoil", None),
        "RxCoil": ("rfcoil", None),
    }
    TEXT = {"PulseSequenceName", "SeriesDescription", "PatientName", "PatientID", "StudyID",
            "ManufacturersModelName", "SoftwareVersions", "TxCoil", "RxCoil"}
    GAMMA_H1 = 42.577478518  # MHz/T

    def __init__(self, folder: Path, header: dict, provenance: Provenance):
        self.folder = folder
        self.header = header
        self.procpar: dict[str, list[str]] = header["procpar"]
        self.provenance = provenance

    @staticmethod
    def is_imaging(procpar: dict[str, list[str]]) -> bool:
        """An imaging scan: apptype im2D* or im3D*, or phase encoding (nv > 0) without one."""
        apptype = (procpar.get("apptype") or [""])[0]
        if apptype:
            return apptype.startswith(("im2D", "im3D"))
        return VarianRaw._number(procpar, "nv", 0) > 0

    @staticmethod
    def _number(procpar: dict[str, list[str]], name: str, default: float = 0) -> float:
        try:
            return float(procpar[name][0])
        except (KeyError, IndexError, ValueError):
            return default

    def _int(self, name: str, default: int = 0) -> int:
        return int(self._number(self.procpar, name, default))

    def _text(self, name: str) -> str:
        return (self.procpar.get(name) or [""])[0]

    def datasets(self) -> list[RawData]:
        notes: list[str] = list(self.provenance.notes)
        blocks, file_header, block_headers = self._blocks(notes)  # (blocks, traces, samples)
        receivers = max(1, self._text("rcvrs").count("y"))
        if blocks.shape[0] % receivers:
            notes.append(f"{blocks.shape[0]} blocks for {receivers} receivers; taken as one")
            receivers = 1
        elements = blocks.shape[0] // receivers
        data = blocks.reshape(elements, receivers, blocks.shape[1], blocks.shape[2])
        data = np.transpose(data, (0, 2, 1, 3)).reshape(-1, receivers, blocks.shape[2])
        if receivers > 1:
            notes.append(f"{receivers} receivers taken as following each other block by block")
        headers, encoding = self._headers(elements, blocks.shape[1], data, notes)
        meta = self._meta(data, encoding, notes)
        meta.vendor["fid_header"] = file_header
        meta.vendor["block_headers"] = block_headers
        provenance = Provenance(
            self.provenance.paths, self.provenance.format, self.provenance.reader,
            self.provenance.versions, tuple(notes),
        )
        return [RawData(data, headers, meta, provenance, self.folder.stem, None, None, encoding,
                        [], {})]

    # --- samples ---------------------------------------------------------------------------------

    def _blocks(self, notes: list[str]) -> tuple[np.ndarray, dict, dict]:
        path = self.folder / "fid"
        raw = path.read_bytes()
        if len(raw) < 32:
            raise ReaderError(f"{path}: shorter than its file header")
        fields = ("nblocks", "ntraces", "np", "ebytes", "tbytes", "bbytes", "vers_id", "status",
                  "nbheaders")
        header = dict(zip(fields, struct.unpack(">6i2hi", raw[:32]), strict=True))
        status = header["status"]
        kind = "f4" if status & self.S_FLOAT else ("i4" if status & self.S_32 else "i2")
        dtype = np.dtype(">" + kind)
        traces, values = header["ntraces"], header["np"]
        heads = 28 * max(header["nbheaders"], 1)
        if dtype.itemsize != header["ebytes"] or heads + traces * values * dtype.itemsize != (
            header["bbytes"]
        ):
            raise ReaderError(f"{path}: block layout {header} does not add up")
        count = min(header["nblocks"], (len(raw) - 32) // header["bbytes"])
        if count < header["nblocks"]:
            notes.append(f"{count} of {header['nblocks']} blocks in the file (acquisition stopped)")
        layout = np.dtype([("head", f"V{heads}"), ("data", dtype, (traces, values))])
        table = np.frombuffer(raw, layout, count, 32)
        head = np.dtype([("scale", ">i2"), ("status", ">i2"), ("index", ">i2"), ("mode", ">i2"),
                         ("ctcount", ">i4"), ("lpval", ">f4"), ("rpval", ">f4"), ("lvl", ">f4"),
                         ("tlt", ">f4")])
        bytes_ = np.frombuffer(table["head"].tobytes(), np.uint8).reshape(count, heads)
        first = bytes_[:, :28].copy().view(head).ravel()  # the first header of every block
        block_headers = {name: first[name].tolist() for name in ("index", "ctcount", "status")}
        complex_type = np.complex64 if kind == "f4" else np.complex128
        samples = table["data"][..., 0::2].astype(complex_type)
        samples += 1j * table["data"][..., 1::2]
        return samples, header, block_headers

    # --- counters --------------------------------------------------------------------------------

    def _headers(
        self, elements: int, traces: int, data: np.ndarray, notes: list[str]
    ) -> tuple[np.ndarray, dict]:
        n = data.shape[0]
        heads = np.zeros(n, dtype=RawData.HEADER)
        heads["version"] = 1
        heads["number_of_samples"] = data.shape[2]
        heads["available_channels"] = heads["active_channels"] = data.shape[1]
        heads["scan_counter"] = np.arange(n)
        sw = self._number(self.procpar, "sw")
        if sw > 0:
            heads["sample_time_us"] = 1e6 / sw
        seqcon = self._text("seqcon").ljust(5, "n")
        sizes = {loop: max(1, self._int(self.SIZES[loop], 1)) for loop in self.LOOPS}
        apptype, etl = self._text("apptype"), max(1, self._int("etl", 1))
        inner = [loop for loop, c in zip(self.LOOPS, seqcon, strict=False) if c == "c"]
        outer = [loop for loop, c in zip(self.LOOPS, seqcon, strict=False) if c == "s"]
        compressed = [(loop, sizes[loop]) for loop in inner]
        if apptype == "im2Dfse" and etl > 1 and "pe1" in inner:
            others = [(loop, sizes[loop]) for loop in inner if loop != "pe1"]
            compressed = [("train echo", etl), *others, ("train", sizes["pe1"] // etl)]
        standard = [(loop, sizes[loop]) for loop in outer]
        per_block = math.prod(size for _, size in compressed)
        per_series = math.prod(size for _, size in standard)
        encoding = {"apptype": apptype, "seqcon": seqcon, "loops": sizes, "etl": etl,
                    "petable": self._text("petable")}
        if per_block != traces or elements % per_series:
            notes.append(f"seqcon {seqcon} with {sizes} does not match {elements} blocks of "
                         f"{traces} traces; the readouts carry no counters")
            return heads, encoding
        arrays = elements // per_series
        trace = np.tile(np.arange(traces), elements)
        element = np.repeat(np.arange(elements), traces)
        counters = self._split(trace, compressed) | self._split(element % per_series, standard)
        if "train" in counters:
            counters["pe1"] = counters.pop("train") * etl + counters["train echo"]
            heads["idx"]["segment"] = counters.pop("train echo")
        for loop, values in counters.items():
            if loop == "slice":
                order = self._slice_positions(sizes["slice"])
                heads["idx"]["slice"] = order[values]
                heads["idx"]["user"][:, 1] = values
            elif loop in self.COUNTERS:
                heads["idx"][self.COUNTERS[loop]] = values
        heads["idx"]["repetition"] = element // per_series
        table = encoding["petable"]
        if table and table != "n" and sizes["pe1"] > 1:
            notes.append(f"phase encodings in acquisition order; the line order is petable "
                         f"{table}, which the folder does not hold")
        limits = {}
        for loop, name in self.LIMITS.items():
            if sizes[loop] > 1 and seqcon[self.LOOPS.index(loop)] != "n":
                centre = sizes[loop] // 2 if loop.startswith("pe") else 0
                limits[name] = {"minimum": 0, "maximum": sizes[loop] - 1, "center": centre}
        if arrays > 1:
            limits["repetition"] = {"minimum": 0, "maximum": arrays - 1, "center": 0}
            names = self._text("array")
            notes.append(f"{arrays} array elements ({names or 'unnamed'}) as repetitions")
        encoding["limits"] = limits
        return heads, encoding

    @staticmethod
    def _split(index: np.ndarray, loops: list[tuple[str, int]]) -> dict[str, np.ndarray]:
        """``index`` decomposed over ``loops``, the first innermost."""
        counters = {}
        for loop, size in loops:
            counters[loop] = index % size
            index = index // size
        return counters

    def _slice_positions(self, count: int) -> np.ndarray:
        """The spatial index of every acquisition slot, by its pss position."""
        positions = [float(v) for v in self.procpar.get("pss", [])][:count]
        if len(positions) != count:
            return np.arange(count)
        return np.argsort(np.argsort(positions, kind="stable"), kind="stable")

    # --- keys ------------------------------------------------------------------------------------

    def _meta(self, data: np.ndarray, encoding: dict, notes: list[str]) -> Meta:
        meta = Meta(self.header)
        for key, (name, factor) in self.KEYS.items():
            if name not in self.procpar or not self.procpar[name]:
                continue
            text = self.procpar[name][0]
            if key in self.TEXT:
                if text:
                    meta.set(key, text, f"{self.SOURCE}: {name}", f"procpar/{name}")
                continue
            try:
                value = float(text)
            except ValueError:
                continue
            value = value * factor if factor else value
            value = int(value) if key in ("NumberOfAverages", "EchoTrainLength") else value
            meta.set(key, value, f"{self.SOURCE}: {name}", f"procpar/{name}")
        nucleus = re.fullmatch(r"([A-Za-z]+)(\d+)", self._text("tn"))
        if nucleus:
            value = nucleus.group(2) + nucleus.group(1)
            meta.set("ResonantNucleus", value, f"{self.SOURCE}: tn (element and mass swapped)",
                     "procpar/tn")
        self._echo_times(meta)
        if self._text("ir") == "y" and "ti" in self.procpar:
            values = [float(v) for v in self.procpar["ti"]]
            arrayed = re.split(r"[,()\s]+", self._text("array"))
            values = values if "ti" in arrayed else values[:1]
            meta.set("InversionTime", values[0] if len(values) == 1 else values,
                     f"{self.SOURCE}: ti (ir = y)", "procpar/ti")
        sw = self._number(self.procpar, "sw")
        if sw > 0:
            meta.set("DwellTime", 1 / sw, f"{self.SOURCE}: 1 / sw", "procpar/sw")
        meta.set("NumberOfPoints", int(data.shape[2]), "fid header: np / 2", "fid_header/np")
        meta.set("NumberOfReceiveChannels", int(data.shape[1]), f"{self.SOURCE}: rcvrs",
                 "procpar/rcvrs")
        if meta.get("ResonantNucleus") == "1H" and "SpectrometerFrequency" in meta:
            meta.set("FieldStrength", meta["SpectrometerFrequency"] / self.GAMMA_H1,
                     "sfrq / 42.577478518 MHz/T")
        self._encoding_keys(meta, data, encoding)
        stamp = re.fullmatch(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})",
                             self._text("time_submitted"))
        if stamp:
            y, mo, d, h, mi, s = stamp.groups()
            meta.set("AcquisitionDateTime", f"{y}-{mo}-{d}T{h}:{mi}:{s}",
                     f"{self.SOURCE}: time_submitted", "procpar/time_submitted")
        return meta

    def _echo_times(self, meta: Meta) -> None:
        """The TE list (ms) when it has one value per echo, else te: all its values when te is
        arrayed, its first value when not (VnmrJ keeps unused values of parameters)."""
        echoes = max(1, self._int("ne", 1))
        arrayed = re.split(r"[,()\s]+", self._text("array"))
        if len(self.procpar.get("TE", [])) == echoes:
            values = [float(v) / 1000 for v in self.procpar["TE"]]
            field, source = "procpar/TE", f"{self.SOURCE}: TE (ms)"
        elif self.procpar.get("te"):
            values = [float(v) for v in self.procpar["te"]]
            values = values if "te" in arrayed else values[:1]
            field, source = "procpar/te", f"{self.SOURCE}: te"
        else:
            return
        meta.set("EchoTime", values[0] if len(values) == 1 else values, source, field)

    def _encoding_keys(self, meta: Meta, data: np.ndarray, encoding: dict) -> None:
        sizes, three_d = encoding["loops"], encoding["apptype"].startswith("im3D")
        matrix = [int(data.shape[2]), sizes["pe1"], sizes["pe2"] if three_d else 1]
        meta.set("EncodedMatrixSize", matrix, f"{self.SOURCE}: np / 2, nv, nv2",
                 "procpar/nv")
        fov = [self._number(self.procpar, name) * 10 for name in ("lro", "lpe", "lpe2")]
        if not three_d:
            fov[2] = self._number(self.procpar, "thk")
        if fov[0] > 0 and fov[1] > 0:
            meta.set("EncodedFieldOfView", fov,
                     f"{self.SOURCE}: lro, lpe (cm) and {'lpe2' if three_d else 'thk (mm)'}",
                     "procpar/lro")
        if not three_d and sizes["slice"] > 1 and "gap" in self.procpar:
            spacing = self._number(self.procpar, "thk") + self._number(self.procpar, "gap") * 10
            meta.set("SpacingBetweenSlices", spacing, f"{self.SOURCE}: thk + gap (cm)",
                     "procpar/gap")
        if encoding["apptype"] in ("im2D", "im3D", "im2Dfse"):
            meta.set("KSpaceTrajectory", "cartesian", f"{self.SOURCE}: apptype", "procpar/apptype")
        if encoding.get("limits"):
            meta.set("EncodingLimits", encoding["limits"], f"{self.SOURCE}: seqcon and loop sizes",
                     "procpar/seqcon")
