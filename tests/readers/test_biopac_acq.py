####################################################################################################
#                                        test_biopac_acq.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/10/26                                                                                #
#                                                                                                  #
# Purpose: BIOPAC AcqKnowledge reader on a synthetic file (revision 45): interleaved channels of   #
#          three frequency dividers ending at different points, int16 and float64 samples,         #
#          scaling, markers and journal.                                                           #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import struct

import numpy as np

# own
import miox as gu
from miox.sniff import Format, Kind, Sniffer

CHANNELS = (  # name, units, divider, points, type (2 int16, 1 float64), scale, offset
    ("Pulse", "Volts", 1, 13, 2, 0.5, 1.0),
    ("EDA", "µS", 1, 12, 1, 7.0, 3.0),
    ("ECG", "mV", 2, 7, 2, 0.25, 0.0),
    ("Resp", "Volts", 3, 4, 2, 2.0, -1.0),
)


def _raw(k: int, i: int) -> float:
    return 100 * (k + 1) + i


def _acq(path, markers=((0, "Segment 1"), (6, "Breath")), journal="Three deep breaths."):
    graph = bytearray(2000)
    struct.pack_into("<hIihhh", graph, 0, 0, 45, len(graph), len(CHANNELS), 0, 0)
    struct.pack_into("<d", graph, 16, 2.0)  # ms per base sample: 500 Hz
    blob = bytearray(graph)
    for k, (name, units, divider, points, _, scale, offset) in enumerate(CHANNELS):
        header = bytearray(252)
        struct.pack_into("<ih", header, 0, len(header), k)
        header[6 : 6 + len(name)] = name.encode("latin-1")
        header[68 : 68 + len(units.encode("latin-1"))] = units.encode("latin-1")
        struct.pack_into("<iddh", header, 88, points, scale, offset, k + 1)
        struct.pack_into("<h", header, 250, divider)
        blob += header
    blob += struct.pack("<hh", 4, 0)  # foreign data: its length only
    for *_, kind, _, _ in CHANNELS:
        blob += struct.pack("<hh", 8 if kind == 1 else 2, kind)
    done = [0] * len(CHANNELS)
    t = 0
    while any(n < c[3] for n, c in zip(done, CHANNELS, strict=True)):
        for k, (_, _, divider, points, kind, _, _) in enumerate(CHANNELS):
            if t % divider == 0 and done[k] < points:
                blob += struct.pack("<d" if kind == 1 else "<h", _raw(k, done[k]))
                done[k] += 1
        t += 1
    blob += struct.pack("<ii", 0, len(markers))
    for sample, text in markers:
        blob += struct.pack("<ihhhh", sample, 0, 0, 0, len(text)) + text.encode() + b"\0"
    blob += b"\x44\x33\x22\x11" + struct.pack("<hi", 1, len(journal)) + journal.encode()
    path.write_bytes(bytes(blob))
    return path


def test_channels_by_divider(tmp_path):
    path = _acq(tmp_path / "rec.acq")
    sniff = Sniffer().sniff(path)
    assert (sniff.format, sniff.kind, sniff.vendor) == (Format.BIOPAC_ACQ, Kind.PHYSIO, "biopac")
    fast, mid, slow = gu.load(path)
    assert fast.channels == ("Pulse", "EDA")
    assert (fast.sampling_frequency, mid.sampling_frequency) == (500.0, 250.0)
    assert slow.sampling_frequency == 500.0 / 3
    assert fast.name == "rec_500Hz"
    assert np.array_equal(fast.data[:, 0], [_raw(0, i) * 0.5 + 1.0 for i in range(13)])
    assert np.array_equal(fast.data[:12, 1], [_raw(1, i) for i in range(12)])  # float: unscaled
    assert np.isnan(fast.data[12, 1])
    assert any("EDA" in note for note in fast.provenance.notes)
    assert np.array_equal(mid.data[:, 0], [_raw(2, i) * 0.25 for i in range(7)])
    assert np.array_equal(slow.data[:, 0], [_raw(3, i) * 2.0 - 1.0 for i in range(4)])


def test_markers_journal_and_header(tmp_path):
    fast, mid, slow = gu.load(_acq(tmp_path / "rec.acq"))
    assert {k: v.tolist() for k, v in fast.events.items()} == {"Segment 1": [0], "Breath": [6]}
    assert mid.events["Breath"].tolist() == [3]
    assert slow.events["Breath"].tolist() == [2]
    assert fast.start_time is None
    vendor = fast.meta.vendor
    assert vendor["journal"] == "Three deep breaths."
    assert vendor["graph_header"]["revision"] == 45
    assert vendor["graph_header"]["byte_order"] == "little"
    assert [c["units"] for c in vendor["channels"]] == ["Volts", "µS", "mV", "Volts"]


def test_markers_past_the_recording_stay_in_the_header(tmp_path):
    datasets = gu.load(_acq(tmp_path / "rec.acq", markers=((2, "a"), (99, "b"))))
    assert all(not d.events for d in datasets)
    assert [m["sample"] for m in datasets[0].meta.vendor["markers"]] == [2, 99]
    assert any("not reliable" in note for note in datasets[0].provenance.notes)
