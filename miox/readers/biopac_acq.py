####################################################################################################
#                                          biopac_acq.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for BIOPAC AcqKnowledge files (.acq) of AcqKnowledge 2 to 5, uncompressed and    #
#          compressed: the channels (one dataset per sampling frequency), markers and journal.     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import math
import struct
import zlib
from datetime import UTC, datetime, timedelta

import numpy as np

# own
from miox.data import PhysioData
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                        AcqKnowledgeReader                                        #
#**************************************************************************************************#
#                                                                                                  #
# An AcqKnowledge file, in the byte order of the computer that wrote it, holds in turn: the graph  #
# header (file revision, channel count, sample time in ms, from revision 41 a compression flag;    #
# its own length), from revision 124 padding headers, one header per channel (name, units, point   #
# count, scale and offset of integer samples, frequency divider), a foreign data block, one        #
# (size, type) pair per channel (type 0 or 1: float64, 2: int16), then the samples, the markers    #
# and the journal. Uncompressed samples are interleaved: at every base sample t, each channel      #
# whose divider divides t in channel order, the channels with fewer points left out once they are  #
# complete. Compressed files keep the samples after the journal, one zlib stream per channel,      #
# little-endian whatever the byte order of the headers. Revisions from 61 (AcqKnowledge 4) change  #
# the layouts of the headers, markers and journal; from 128 markers carry their creation time.     #
# The layouts are those bioread reads (MIT, research of N. Vack and J. Ollinger).                  #
#                                                                                                  #
# Integer samples are scaled (raw x scale + offset); float samples are taken as stored. Channels   #
# of the same divider form one dataset; a file with several dividers gives one per sampling        #
# frequency. Markers become events named by their text, at the sample of their dataset (channel    #
# markers only in the dataset of their channel); the start time is the creation time of the marker #
# at the first sample, when the file has one.                                                      #
#                                                                                                  #
#**************************************************************************************************#
class AcqKnowledgeReader(Reader):
    FORMATS = (Format.BIOPAC_ACQ,)
    PACKAGES = ("miox",)
    REVISION_4 = 61  # first revision of AcqKnowledge 4
    DTYPES = {0: "f8", 1: "f8", 2: "i2"}
    JOURNAL_TAG = b"\x44\x33\x22\x11"
    DTYPE_SCAN = 4096  # bytes searched past the foreign data for the (size, type) pairs

    def read(self, sniff: Sniff) -> list[PhysioData]:
        blob = sniff.path.read_bytes()
        order, revision = self.order_and_revision(blob)
        if revision is None:
            raise ReaderError(f"{sniff.path.name}: no AcqKnowledge graph header")
        self.order, self.revision = order, revision
        self.encoding = "latin-1" if revision < self.REVISION_4 else "utf-8"
        notes: list[str] = []
        graph = self._graph(blob)
        position = graph["header_length"]
        for _ in range(graph["paddings"]):
            position += self._int(blob, position)
        channels = []
        for _ in range(graph["channels"]):
            channel = self._channel(blob, position)
            channels.append(channel)
            position += channel.pop("header_length")
        foreign = self._short(blob, position) if revision < self.REVISION_4 else self._int(
            blob, position)
        position = self._dtypes(blob, position + foreign, channels)
        data_start = position
        if not graph["compressed"]:
            position += sum(c["points"] * c["size"] for c in channels)
        markers, journal = [], None
        try:
            markers, position = self._markers(blob, position)
            journal, position = self._journal(blob, position)
        except (struct.error, ValueError) as error:
            if graph["compressed"]:
                raise ReaderError(f"{sniff.path.name}: markers or journal unreadable ({error}), "
                                  f"the compressed samples follow them") from error
            notes.append(f"markers or journal unreadable: {error}")
        if graph["compressed"]:
            samples = self._compressed(blob, position, channels)
        else:
            samples = self._interleaved(blob, data_start, channels)
        vendor = {"graph_header": graph, "channels": channels, "markers": markers,
                  "journal": journal}
        return self._datasets(sniff, samples, channels, graph, markers, vendor, notes)

    @classmethod
    def order_and_revision(cls, blob: bytes) -> tuple[str, int | None]:
        """The byte order and file revision: the revision read in both orders, the smaller
        value; None when it is no plausible revision or no channel count follows."""
        if len(blob) < 12:
            return "<", None
        readings = sorted(
            (struct.unpack_from(f"{o}I", blob, 2)[0], o) for o in ("<", ">")
        )
        revision, order = readings[0]
        channels = struct.unpack_from(f"{order}h", blob, 10)[0]
        if not 30 <= revision <= 1000 or not 0 < channels <= 1024:
            return order, None
        return order, revision

    def _int(self, blob: bytes, at: int) -> int:
        return struct.unpack_from(f"{self.order}i", blob, at)[0]

    def _short(self, blob: bytes, at: int) -> int:
        return struct.unpack_from(f"{self.order}h", blob, at)[0]

    def _double(self, blob: bytes, at: int) -> float:
        return struct.unpack_from(f"{self.order}d", blob, at)[0]

    def _text(self, raw: bytes) -> str:
        return raw.split(b"\0", 1)[0].decode(self.encoding, errors="replace").strip()

    def _graph(self, blob: bytes) -> dict:
        o = self.order
        _, revision, length, count, axis, current = struct.unpack_from(f"{o}hIihhh", blob, 0)
        sample_time, time_offset, time_scale = struct.unpack_from(f"{o}ddd", blob, 16)
        if self.revision >= self.REVISION_4:
            compressed = self._int(blob, 972)
        elif self.revision >= 41:
            compressed = self._int(blob, 1936)
        else:
            compressed = 0
        return {
            "revision": revision, "byte_order": "little" if o == "<" else "big",
            "header_length": length, "channels": count, "horizontal_axis_type": axis,
            "current_channel": current, "sample_time_ms": sample_time,
            "time_offset": time_offset, "time_scale": time_scale,
            "first_time_offset": self._double(blob, 78),
            "horizontal_units": [self._text(blob[88:128]), self._text(blob[128:138])],
            "compressed": bool(compressed),
            "paddings": self._short(blob, 2398) if self.revision >= 124 else 0,
        }

    def _channel(self, blob: bytes, at: int) -> dict:
        post4 = self.revision >= self.REVISION_4
        if post4:
            divider = self._short(blob, at + 152)
        else:
            divider = self._short(blob, at + 250) if self.revision >= 38 else 1
        channel = {
            "header_length": self._int(blob, at),
            "number": self._short(blob, at + 4),
            "name": self._text(blob[at + 6 : at + 46]),
            "volt_offset": self._double(blob, at + 52),
            "volt_scale": self._double(blob, at + 60),
            "units": self._text(blob[at + 68 : at + 88]),
            "points": self._int(blob, at + 88),
            "scale": self._double(blob, at + 92),
            "offset": self._double(blob, at + 100),
            "order": self._short(blob, at + 108),
            "divider": divider or 1,
        }
        if not post4 and self.revision >= 38:
            channel["description"] = self._text(blob[at + 122 : at + 250])
        return channel

    def _dtypes(self, blob: bytes, at: int, channels: list[dict]) -> int:
        """Find the (size, type) pairs, which may lie some bytes past the foreign data; returns
        the offset after them."""
        count = len(channels)
        for start in range(at, at + self.DTYPE_SCAN):
            pairs = [struct.unpack_from(f"{self.order}hh", blob, start + 4 * i)
                     for i in range(count)]
            if all(t in self.DTYPES and int(self.DTYPES[t][1]) == s for s, t in pairs):
                for channel, (size, code) in zip(channels, pairs, strict=True):
                    channel["size"], channel["type"] = size, self.DTYPES[code]
                return start + 4 * count
        raise ReaderError("no channel data types found after the foreign data")

    def _markers(self, blob: bytes, at: int) -> tuple[list[dict], int]:
        if self.revision < self.REVISION_4:
            return self._markers_pre4(blob, at)
        extra = self._int(blob, at + 4)
        position = at + 25 + (8 if self.revision >= 121 else 0) + (
            8 if self.revision >= 128 else 0)
        markers = []
        for _ in range(extra - 1):
            sample = struct.unpack_from(f"{self.order}I", blob, position)[0]
            channel = self._short(blob, position + 8)
            style = blob[position + 10 : position + 14].decode("latin-1")
            position += 14
            created = None
            if self.revision >= 128:
                created = struct.unpack_from(f"{self.order}Q", blob, position)[0]
                position += 8
            if self.revision >= 121:
                position += 8
            length = self._short(blob, position)
            position += 2
            markers.append({
                "sample": sample, "text": self._text(blob[position : position + length]),
                "channel": None if channel == -1 else channel, "type": style,
                "created": self._date(created),
            })
            position += length
        return markers, position

    def _markers_pre4(self, blob: bytes, at: int) -> tuple[list[dict], int]:
        count = self._int(blob, at + 4)
        position = at + 8
        selected = self.revision >= 36
        markers = []
        for _ in range(count):
            sample = self._int(blob, position)
            length = self._short(blob, position + (10 if selected else 8)) + (1 if selected else 0)
            position += 12 if selected else 10
            markers.append({"sample": sample,
                            "text": self._text(blob[position : position + length]),
                            "channel": None, "type": None, "created": None})
            position += length
        if self.revision >= 41 and blob[position : position + 4] != self.JOURNAL_TAG:
            position += 84  # item count and unknown values
            for _ in range(count):
                number = self._int(blob, position + 4)
                if 0 < number <= count:
                    markers[number - 1]["tag"] = self._short(blob, position + 24)
                position += 28
        return markers, position

    def _journal(self, blob: bytes, at: int) -> tuple[str | None, int]:
        if self.revision < self.REVISION_4:
            if self.revision < 38:
                return None, at
            if blob[at : at + 4] != self.JOURNAL_TAG:
                raise ValueError(f"journal tag {blob[at : at + 4].hex()} at {at}")
            length = self._int(blob, at + 6)
            text = blob[at + 10 : at + 10 + length]
            return self._text(text), at + 10 + length
        total = self._int(blob, at)
        late = self.revision >= 108
        header = 262 + 4 + 290 + (26 + 8 if late else 0) + (4 if self.revision >= 128 else 0)
        journal = None
        if header <= total:
            start = at + 4
            length = self._int(blob, start + header - 4) if late else self._int(blob, start + 262)
            journal = self._text(blob[start + header : start + header + length])
        return journal, at + total

    def _compressed(self, blob: bytes, at: int, channels: list[dict]) -> list[np.ndarray]:
        if self.revision < self.REVISION_4:
            position = at + 38 + self._int(blob, at + 34)
        else:
            position = at + (58 if self.revision >= 108 else 52) + self._int(
                blob, at + 24) + self._int(blob, at + 28)
        samples = []
        for channel in channels:
            label, unit = self._int(blob, position + 44), self._int(blob, position + 48)
            length = self._int(blob, position + 56)
            start = position + 60 + label + unit
            raw = zlib.decompress(blob[start : start + length])
            samples.append(np.frombuffer(raw, f"<{channel['type']}")[: channel["points"]])
            position = start + length
        return samples

    def _interleaved(self, blob: bytes, at: int, channels: list[dict]) -> list[np.ndarray]:
        """Samples of interleaved channels: whole repetitions of the divider pattern read as
        records, the remainder (channels ending at different points) sample by sample."""
        dividers = [c["divider"] for c in channels]
        sizes = np.array([c["size"] for c in channels])
        counts = np.array([c["points"] for c in channels])
        base = math.lcm(*dividers)
        pattern = np.array([k for t in range(base) for k, d in enumerate(dividers) if t % d == 0])
        uses = np.bincount(pattern, minlength=len(channels))
        if at + int((counts * sizes).sum()) > len(blob):
            raise ReaderError(f"file ends before the {int((counts * sizes).sum())} bytes of "
                              f"samples")
        whole = int((counts // uses).min())
        offsets = np.concatenate([[0], np.cumsum(sizes[pattern])])
        record = np.dtype({
            "names": [f"s{i}" for i in range(len(pattern))],
            "formats": [self.order + channels[k]["type"] for k in pattern],
            "offsets": offsets[:-1].tolist(), "itemsize": int(offsets[-1]),
        })
        records = np.frombuffer(blob, record, count=whole, offset=at)
        rest = counts - whole * uses
        reps = int(np.max(-(-rest // uses)))
        tail = np.tile(pattern, reps)
        rank = np.zeros(len(tail), dtype=np.int64)
        for k in range(len(channels)):
            rank[tail == k] = np.arange(int((tail == k).sum()))
        tail = tail[rank < rest[tail]]
        starts = at + whole * record.itemsize + np.concatenate(
            [[0], np.cumsum(sizes[tail])])[:-1]
        raw = np.frombuffer(blob, np.uint8)
        samples = []
        for k, channel in enumerate(channels):
            dtype = np.dtype(self.order + channel["type"])
            fields = [records[f"s{i}"] for i in np.flatnonzero(pattern == k)]
            head = np.stack(fields, axis=1).ravel() if whole else np.empty(0, dtype)
            where = starts[tail == k]
            last = raw[where[:, None] + np.arange(dtype.itemsize)].copy().view(dtype).ravel()
            samples.append(np.concatenate([head.astype(dtype), last]))
        return samples

    @staticmethod
    def _date(milliseconds: int | None) -> str | None:
        if not milliseconds:
            return None
        try:
            return (datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=milliseconds)
                    ).isoformat()
        except OverflowError:
            return None

    def _datasets(self, sniff: Sniff, samples: list[np.ndarray], channels: list[dict],
                  graph: dict, markers: list[dict], vendor: dict,
                  notes: list[str]) -> list[PhysioData]:
        base_rate = 1000.0 / graph["sample_time_ms"]
        values = []
        for channel, raw in zip(channels, samples, strict=True):
            if raw.dtype.kind == "f":
                values.append(raw.astype(np.float64))
            else:
                values.append(raw * channel["scale"] + channel["offset"])
        start_time, reference = None, ""
        first = [m for m in markers if m["sample"] == 0 and m["created"]]
        if first:
            start_time = datetime.fromisoformat(first[0]["created"]).timestamp()
            reference = "s since 1970-01-01 UTC"
            notes.append(f"start time: creation time of the marker {first[0]['text']!r} at the "
                         f"first sample")
        by_order = {c["order"]: i for i, c in enumerate(channels)}
        total = max(c["points"] * c["divider"] for c in channels)
        if any(m["sample"] >= total for m in markers):
            notes.append(f"markers past the last of {total} samples: their positions are not "
                         f"reliable, the markers are kept in the header only")
            markers = []
        dividers = list(dict.fromkeys(c["divider"] for c in channels))
        datasets = []
        for divider in dividers:
            members = [i for i, c in enumerate(channels) if c["divider"] == divider]
            length = max(len(values[i]) for i in members)
            data = np.full((length, len(members)), np.nan)
            for column, i in enumerate(members):
                data[: len(values[i]), column] = values[i]
            group_notes = list(notes)
            short = [channels[i]["name"] for i in members if len(values[i]) < length]
            if short:
                group_notes.append(f"channels ending early, NaN after their last sample: "
                                   f"{', '.join(short)}")
            events: dict[str, list[int]] = {}
            late = 0
            for marker in markers:
                if marker["channel"] is not None and by_order.get(marker["channel"]) not in members:
                    continue
                index = marker["sample"] // divider
                if index >= length:
                    late += 1
                    continue
                events.setdefault(marker["text"] or "marker", []).append(index)
            if late:
                group_notes.append(f"{late} markers after the last sample of these channels, "
                                   f"kept in the header only")
            rate = base_rate / divider
            name = sniff.path.stem if len(dividers) == 1 else f"{sniff.path.stem}_{rate:g}Hz"
            datasets.append(PhysioData(
                data, tuple(channels[i]["name"] for i in members), rate, start_time, reference,
                {k: np.unique(np.asarray(v, dtype=np.int64)) for k, v in events.items()},
                Meta(vendor), self.provenance(sniff, tuple(group_notes)), name,
            ))
        return datasets
