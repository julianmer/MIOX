####################################################################################################
#                                             toppe.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for TOPPE sequences (GE): the entry file, the module list (modules.txt), the     #
#          scan loop (scanloop.txt) and the waveform modules (.mod) as PulseSequence objects.      #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np

# own
from miox.data import PulseSequence
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                           ToppeReader                                            #
#**************************************************************************************************#
#                                                                                                  #
# TOPPE splits a sequence over files: toppe<N>.entry names the folder on the scanner, the module   #
# list, the scan loop, the modules used for B1 scaling and the receive filter, and the sequence    #
# stamp. The module list (modules.txt: a title line, the module count, a column line, then per     #
# module its .mod file, duration in us, hasRF, hasDAQ and optionally the trigger position) names   #
# the .mod files: big-endian int16 values and text, a description, the number of coils, points     #
# and waveforms, the lines "b1max: <G>" and "gmax: <G/cm>", the int16 parameters (the first two:   #
# discarded samples and RF/ADC samples) and the float parameters (one per text line), then per     #
# waveform rho and theta of every coil and gx, gy, gz, each "points" int16 values of which 32766   #
# is full scale (b1max, pi, gmax). The scan loop (a header line, nt, maxslice, maxecho, maxview,   #
# scan duration in us and version, a column line, then one tab-separated row per module played)    #
# becomes ``blocks``; the modules with their waveforms in SI units (RF in T, gradients in T/m, on  #
# a 4 us raster) are ``events["modules"]`` by their number in the module list. Each file loads     #
# with what it holds; the entry file loads the files it names from its own folder.                 #
#                                                                                                  #
#**************************************************************************************************#
class ToppeReader(Reader):
    FORMATS = (Format.TOPPE_ENTRY, Format.TOPPE_MODULES, Format.TOPPE_LOOP, Format.TOPPE_MOD)
    PACKAGES = ("miox",)
    RASTER = 4e-6  # s, RF, gradients and ADC
    FULL_SCALE = 2**15 - 2
    ENTRY = ("folder", "module_list", "scan_loop", "b1_scaling_module", "receive_filter_module",
             "sequence_stamp")
    LOOP = ("nt", "maxslice", "maxecho", "maxview", "scandur_us", "version")
    COLUMNS = (  # scan loop columns of version 4; version 5 adds trigger_out, 6 core
        "module", "rf_amp", "rf_phase_amp", "gx_amp", "gy_amp", "gz_amp", "slice", "echo", "view",
        "dab_mode", "inplane_rotation", "rf_phase", "receive_phase", "textra_us", "frequency_hz",
        "waveform", *(f"rotation_{i}" for i in range(9)), "trigger", "trigger_out", "core",
    )

    def read(self, sniff: Sniff) -> list[PulseSequence]:
        notes: list[str] = []
        definitions: dict = {"raster": self.RASTER}
        modules: dict[int, dict] = {}
        blocks = np.zeros(0, dtype=[(c, "<i8") for c in self.COLUMNS[:26]])
        folder = sniff.path.parent
        if sniff.format == Format.TOPPE_ENTRY:
            entry = self.entry(sniff.path)
            definitions["entry"] = entry
            module_list = folder / entry.get("module_list", "modules.txt")
            loop = folder / entry.get("scan_loop", "scanloop.txt")
            stamp = folder / entry.get("sequence_stamp", "seqstamp.txt")
            for path in (module_list, loop):
                if not path.is_file():
                    notes.append(f"{path.name} named in the entry file not found beside it")
            if module_list.is_file():
                modules = self.module_list(module_list, notes)
            if loop.is_file():
                definitions["loop"], blocks = self.loop(loop, notes)
            if stamp.is_file():
                definitions["sequence_stamp"] = stamp.read_text(errors="replace").splitlines()
        elif sniff.format == Format.TOPPE_MODULES:
            modules = self.module_list(sniff.path, notes)
        elif sniff.format == Format.TOPPE_LOOP:
            definitions["loop"], blocks = self.loop(sniff.path, notes)
        else:
            modules = {1: {"file": sniff.path.name, **self.mod(sniff.path, notes)}}
        version = str(definitions.get("loop", {}).get("version", ""))
        meta = Meta({"definitions": {k: v for k, v in definitions.items() if k != "raster"}})
        duration = definitions.get("loop", {}).get("scandur_us")
        if duration:
            meta.set("ScanDuration", duration * 1e-6, "scan loop: scan duration (us)",
                     "definitions/loop/scandur_us")
        return [PulseSequence(version, definitions, blocks, {"modules": modules}, {}, {}, {},
                              meta, self.provenance(sniff, tuple(notes)), sniff.path.stem)]

    @classmethod
    def entry(cls, path: Path) -> dict:
        lines = [line.strip() for line in path.read_text(errors="replace").splitlines()]
        return dict(zip(cls.ENTRY, [line for line in lines if line], strict=False))

    def module_list(self, path: Path, notes: list[str]) -> dict[int, dict]:
        lines = [line for line in path.read_text(errors="replace").splitlines() if line.strip()]
        if len(lines) < 3 or not lines[1].strip().isdigit():
            raise ReaderError(f"{path.name}: no module count on the second line")
        count = int(lines[1])
        modules = {}
        for number, line in enumerate(lines[3 : 3 + count], start=1):
            fields = line.split()
            values = [int(v) for v in fields[1:]]
            module = {"file": fields[0], "duration_us": values[0], "has_rf": bool(values[1]),
                      "has_daq": bool(values[2])}
            if len(values) > 3:
                module["trigger_position"] = values[3]
            mod = path.parent / fields[0]
            if mod.is_file():
                module.update(self.mod(mod, notes))
            else:
                notes.append(f"{fields[0]} not found beside {path.name}")
            modules[number] = module
        if len(modules) != count:
            notes.append(f"{path.name} states {count} modules, lists {len(modules)}")
        return modules

    def loop(self, path: Path, notes: list[str]) -> tuple[dict, np.ndarray]:
        lines = [line for line in path.read_text(errors="replace").splitlines() if line.strip()]
        header = dict(zip(self.LOOP, (int(v) for v in lines[1].split()), strict=False))
        rows = [[int(float(v)) for v in line.split()] for line in lines[3:]]
        width = len(rows[0]) if rows else 26
        if any(len(r) != width for r in rows):
            raise ReaderError(f"{path.name}: rows of unequal length")
        names = list(self.COLUMNS[:width]) + [f"column_{i + 1}" for i in
                                             range(len(self.COLUMNS), width)]
        blocks = np.array([tuple(r) for r in rows], dtype=[(n, "<i8") for n in names])
        if header.get("nt") is not None and header["nt"] != len(rows):
            notes.append(f"{path.name} states {header['nt']} rows, holds {len(rows)}")
        return header, blocks

    def mod(self, path: Path, notes: list[str]) -> dict:
        blob = path.read_bytes()
        size = struct.unpack_from(">h", blob, 0)[0]
        description = blob[2 : 2 + size].decode("latin-1")
        coils, points, waveforms = struct.unpack_from(">hhh", blob, 2 + size)
        position = 8 + size
        values = {}
        for key in ("b1max", "gmax"):
            end = blob.index(b"\n", position)
            match = re.match(rf"\s*{key}:\s*([-+.\deE]+)", blob[position:end].decode("latin-1"))
            if match is None:
                raise ReaderError(f"{path.name}: no '{key}:' line")
            values[key] = float(match.group(1))
            position = end + 1
        count = struct.unpack_from(">h", blob, position)[0]
        ints = list(struct.unpack_from(f">{count}h", blob, position + 2))
        position += 2 + 2 * count
        count = struct.unpack_from(">h", blob, position)[0]
        position += 2
        floats = []
        for _ in range(count):
            end = blob.index(b"\n", position)
            floats.append(float(blob[position:end]))
            position = end + 1
        expected = 2 * points * waveforms * (2 * coils + 3)
        if len(blob) - position != expected:
            raise ReaderError(f"{path.name}: {len(blob) - position} bytes of waveforms, "
                              f"{expected} expected")
        raw = np.frombuffer(blob, ">i2", offset=position).astype(np.float64)
        raw = raw.reshape(waveforms, 2 * coils + 3, points)
        rho = raw[:, :coils] * values["b1max"] / self.FULL_SCALE * 1e-4  # G to T
        theta = raw[:, coils : 2 * coils] * np.pi / self.FULL_SCALE
        gradients = raw[:, 2 * coils :] * values["gmax"] / self.FULL_SCALE * 1e-2  # G/cm to T/m
        return {
            "description": description, "coils": coils, "points": points,
            "waveforms": waveforms, "b1max_gauss": values["b1max"],
            "gmax_gauss_per_cm": values["gmax"],
            "discarded_samples": ints[0] if ints else None,
            "rf_samples": ints[1] if len(ints) > 1 else None,
            "int_parameters": ints[2:], "float_parameters": floats,
            "rf": np.transpose(rho * np.exp(1j * theta), (2, 0, 1)),  # (points, waveforms, coils)
            "gx": gradients[:, 0].T, "gy": gradients[:, 1].T, "gz": gradients[:, 2].T,
            "raw": raw.astype(np.int16),
        }
