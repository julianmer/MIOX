####################################################################################################
#                                       philips_list_raw.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 31/10/26                                                                                #
#                                                                                                  #
# Purpose: Raw k-space of Philips imaging scans exported as .data/.list: the readouts with their   #
#          counters, as RawData.                                                                   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

from pathlib import Path

import numpy as np

# own
from miox.data import Provenance, RawData
from miox.meta import Meta
from miox.readers.base import ReaderError


#**************************************************************************************************#
#                                          PhilipsListRaw                                          #
#**************************************************************************************************#
#                                                                                                  #
# Each vector line of the .list names one readout of one coil: its type (STD imaging, NOI noise,   #
# PHX/FRX/NAV correction and navigator data, REJ rejected, ...), counters (mix, dynamic, cardiac   #
# phase, echo, location, channel, extra attributes, ky, kz, average, sign, rf, grad, enc, rtop,    #
# rr) and the size and offset of its complex float32 samples in the .data file. The coils of one   #
# readout follow each other; they are gathered into (readouts, coils, samples) per type, the       #
# imaging readouts first and the other types after them as their own RawData (noise readouts are   #
# far longer). The counters land in the ISMRMRD header: ky and kz from the bottom of their ranges  #
# as encoding steps 1 and 2, echo as contrast, location as slice, average, dynamic as repetition,  #
# cardiac phase as phase, mix as set, the extra attributes as user 0 and 1, the readout sign as    #
# user_int 0; the rest stays in the vendor table. Noise readouts carry the ISMRMRD noise flag.     #
# The readouts are as stored (a readout of reversed sign keeps its order); nothing is              #
# reconstructed.                                                                                   #
#                                                                                                  #
#**************************************************************************************************#
class PhilipsListRaw:
    COLUMNS = ("typ", "mix", "dyn", "card", "echo", "loca", "chan", "extr1", "extr2", "ky", "kz",
               "na", "aver", "sign", "rf", "grad", "enc", "rtop", "rr", "size", "offset")
    NOISE_FLAG = 19  # ISMRMRD ACQ_IS_NOISE_MEASUREMENT (bits counted from 1)
    INDEX = {"kspace_encode_step_1": "ky", "kspace_encode_step_2": "kz", "contrast": "echo",
             "slice": "loca", "average": "aver", "repetition": "dyn", "phase": "card",
             "set": "mix"}

    def __init__(self, data: Path, list_file: Path, info: dict, provenance: Provenance):
        self.data, self.list_file, self.info, self.provenance = data, list_file, info, provenance

    @classmethod
    def vectors(cls, lines: list[str]) -> list[dict]:
        """The vector lines as dicts by column name."""
        rows = []
        for line in lines:
            parts = line.split()
            if len(parts) == len(cls.COLUMNS) and parts[0].isalpha():
                rows.append(dict(zip(cls.COLUMNS, [parts[0]] + [int(v) for v in parts[1:]],
                                     strict=True)))
        return rows

    @classmethod
    def imaging(cls, info: dict, rows: list[dict]) -> bool:
        """Imaging k-space: the readout of an imaging line spans the kx range (in spectroscopy the
        readout is the FID, and kx, if any, a phase-encoding counter)."""
        standard = [r for r in rows if r["typ"] == "STD"]
        if not standard:
            return False
        ranges = [v.split() for v in info.get("kx_range", {}).values()]
        widths = {abs(int(v[1]) - int(v[0])) + 1 for v in ranges if len(v) == 2}
        return standard[0]["size"] // 8 in widths

    def datasets(self, rows: list[dict]) -> list[RawData]:
        blob = self.data.read_bytes()
        types = list(dict.fromkeys(r["typ"] for r in rows))
        types.sort(key=lambda t: t != "STD")
        datasets = []
        for typ in types:
            datasets.append(self._dataset(blob, [r for r in rows if r["typ"] == typ], typ))
        return datasets

    def _dataset(self, blob: bytes, rows: list[dict], typ: str) -> RawData:
        groups, keys = [], []
        for row in rows:  # the coils of one readout follow each other
            key = tuple(row[c] for c in self.COLUMNS if c not in ("chan", "size", "offset"))
            if not keys or keys[-1] != key:
                keys.append(key)
                groups.append([])
            groups[-1].append(row)
        coils = sorted({r["chan"] for r in rows})
        samples = max(r["size"] for r in rows) // 8
        data = np.zeros((len(groups), len(coils), samples), np.complex64)
        lengths = np.zeros(len(groups), int)
        for i, group in enumerate(groups):
            for row in group:
                count = row["size"] // 4
                if row["offset"] + 4 * count > len(blob):
                    raise ReaderError(f"{self.data.name}: readout beyond the end of the file")
                values = np.frombuffer(blob, "<f4", count, row["offset"])
                data[i, coils.index(row["chan"]), :count // 2] = values[0::2] + 1j * values[1::2]
            lengths[i] = group[0]["size"] // 8
        headers = np.zeros(len(groups), RawData.HEADER)
        firsts = [g[0] for g in groups]
        lows = {c: min(r[c] for r in firsts) for c in ("ky", "kz")}
        headers["scan_counter"] = np.arange(len(groups))
        headers["number_of_samples"] = lengths
        headers["active_channels"] = headers["available_channels"] = len(coils)
        for field, column in self.INDEX.items():
            values = np.array([r[column] for r in firsts])
            headers["idx"][field] = values - lows.get(column, 0)
        headers["idx"]["user"][:, 0] = [r["extr1"] for r in firsts]
        headers["idx"]["user"][:, 1] = [r["extr2"] for r in firsts]
        headers["user_int"][:, 0] = [r["sign"] for r in firsts]
        if typ == "NOI":
            headers["flags"] = np.uint64(1) << np.uint64(self.NOISE_FLAG - 1)
        encoding = self._encoding() if typ == "STD" else {}
        meta = Meta({"list_info": self.info, "type": typ, "coils": coils,
                     "vectors": [{c: r[c] for c in self.COLUMNS if c not in ("size", "offset")}
                                 for r in firsts], "offsets_from": dict(lows)})
        name = f"{self.data.stem} {typ}"
        return RawData(data, headers, meta, self.provenance, name, None, lengths, encoding, [],
                       {self.NOISE_FLAG: "ACQ_IS_NOISE_MEASUREMENT"})

    def _encoding(self) -> dict:
        """Encoded and reconstructed matrix sizes from the .list's ranges and resolutions."""
        def width(name: str) -> int | None:
            values = next(iter(self.info.get(name, {}).values()), "").split()
            return abs(int(values[1]) - int(values[0])) + 1 if len(values) == 2 else None

        def number(name: str) -> int | None:
            values = next(iter(self.info.get(name, {}).values()), "").split()
            return int(float(values[0])) if values else None

        return {
            "encodedSpace": {"matrixSize": {"x": width("kx_range"), "y": width("ky_range"),
                                            "z": width("kz_range") or 1}},
            "reconSpace": {"matrixSize": {"x": number("X-resolution"),
                                          "y": number("Y-resolution"),
                                          "z": number("Z-resolution") or 1}},
            "oversampling": {"x": float(next(iter(self.info.get("kx_oversample_factor",
                                                                 {}).values()), "1"))},
        }
