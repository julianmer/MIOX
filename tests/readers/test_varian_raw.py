####################################################################################################
#                                       test_varian_raw.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/10/26                                                                                #
#                                                                                                  #
# Purpose: Varian/Agilent imaging k-space on synthetic .fid folders: the fid layout (float and     #
#          integer samples, block headers, truncation), the seqcon loop nesting for 2D and 3D      #
#          scans, arrays, receivers, fast spin-echo trains and the keys from the procpar.          #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import struct

import numpy as np

# own
import miox
from miox.readers.varian_raw import VarianRaw


#**************************************************************************************************#
#                                             helpers                                              #
#**************************************************************************************************#
def _procpar(path, params: dict) -> None:
    lines = []
    for name, value in params.items():
        values = value if isinstance(value, list) else [value]
        if isinstance(values[0], str):
            lines += [f"{name} 2 2 8 0 0 2 1 0 1 64", f'{len(values)} "{values[0]}"']
            lines += [f'"{v}"' for v in values[1:]]
        else:
            lines += [f"{name} 1 1 1e+09 -1e+09 0 2 1 0 1 64",
                      f"{len(values)} " + " ".join(repr(float(v)) for v in values)]
        lines.append("0")
    path.write_text("\n".join(lines) + "\n")


def _fid(path, blocks: np.ndarray, kind: str = "f4", cut: int = 0) -> None:
    """``blocks`` complex (blocks, traces, samples) as a fid file of ``kind`` samples."""
    nblocks, traces, samples = blocks.shape
    values = np.empty((nblocks, traces, 2 * samples))
    values[..., 0::2], values[..., 1::2] = blocks.real, blocks.imag
    size = int(kind[1])
    status = 0x1 | (0x8 if kind == "f4" else 0) | (0x4 if kind == "i4" else 0)
    head = struct.pack(">6i2hi", nblocks, traces, 2 * samples, size, 2 * samples * size,
                       28 + traces * 2 * samples * size, 0, status, 1)
    body = b"".join(struct.pack(">4hi4f", 1, status, b + 1, 0, 1, 0, 0, 0, 0)
                    + values[b].astype(">" + kind).tobytes() for b in range(nblocks))
    raw = head + body
    path.write_bytes(raw[: len(raw) - cut])


def _folder(tmp_path, name, params, blocks, kind="f4", cut=0):
    folder = tmp_path / f"{name}.fid"
    folder.mkdir()
    _procpar(folder / "procpar", params)
    _fid(folder / "fid", blocks, kind, cut)
    return folder


BASE = {"sfrq": 400.29, "tn": "H1", "sw": 100000.0, "tr": 1.5, "flip1": 90.0, "nt": 1.0,
        "thk": 1.0, "gap": 0.05, "lro": 3.0, "lpe": 2.0, "lpe2": 1.0, "rcvrs": "y",
        "seqfil": "test", "comment": "scan", "name": "subject", "ident": "id7",
        "time_submitted": "20220824T112048", "array": "", "petable": "n", "ne": 1.0,
        "ns": 1.0, "nv": 1.0, "nv2": 0.0}


#**************************************************************************************************#
#                                              tests                                               #
#**************************************************************************************************#
def test_varian_2d_multi_echo_multi_slice(tmp_path):
    ne, ns, nv, samples = 2, 3, 4, 8
    blocks = np.zeros((nv, ns * ne, samples), np.complex64)
    for p in range(nv):
        for s in range(ns):
            for e in range(ne):
                blocks[p, s * ne + e] = e + 10 * s + 100 * p + 1j * np.arange(samples)
    params = {**BASE, "apptype": "im2D", "seqcon": "ccsnn", "ne": ne, "ns": ns, "nv": nv,
              "arraydim": nv, "np": 2 * samples, "pss": [0.2, -0.2, 0.0], "te": 0.01,
              "TE": [10.0, 20.0]}
    folder = _folder(tmp_path, "mems", params, blocks)
    assert miox.sniff(folder).format == miox.Format.VARIAN
    [raw] = miox.load(folder)
    assert isinstance(raw, miox.RawData) and raw.shape == (nv * ns * ne, 1, samples)
    assert raw.name == "mems" and raw.data.dtype == np.complex64
    e, slot = raw.counters("contrast"), raw.headers["idx"]["user"][:, 1]
    p = raw.counters("kspace_encode_step_1")
    assert np.array_equal(raw.data[:, 0, 0].real, e + 10 * slot + 100 * p)
    assert e[:4].tolist() == [0, 1, 0, 1] and slot[:4].tolist() == [0, 0, 1, 1]
    assert raw.counters("slice")[[0, 2, 4]].tolist() == [2, 0, 1]  # by position pss
    assert np.allclose(raw.data[0, 0].imag, np.arange(samples))
    m = raw.meta
    assert m["EchoTime"] == [0.01, 0.02] and m["RepetitionTime"] == 1.5
    assert m["ResonantNucleus"] == "1H" and m["DwellTime"] == 1e-5
    assert m["EncodedMatrixSize"] == [samples, nv, 1] and m["EncodedFieldOfView"] == [30, 20, 1]
    assert m["SpacingBetweenSlices"] == 1.5 and m["KSpaceTrajectory"] == "cartesian"
    assert m["EncodingLimits"]["kspace_encoding_step_1"] == {"minimum": 0, "maximum": 3,
                                                             "center": 2}
    assert m["EncodingLimits"]["contrast"]["maximum"] == 1
    assert m["AcquisitionDateTime"] == "2022-08-24T11:20:48" and m["PatientName"] == "subject"
    assert m["SeriesDescription"] == "scan" and m["PulseSequenceName"] == "test"
    assert m.vendor["fid_header"]["nblocks"] == nv and m.vendor["block_headers"]["index"][1] == 2
    assert raw.headers["sample_time_us"][0] == 10.0


def test_varian_3d_integer_samples_and_truncation(tmp_path):
    nv, nv2, samples = 3, 2, 4
    blocks = np.zeros((nv2, nv, samples), np.complex64)
    for b in range(nv2):
        for t in range(nv):
            blocks[b, t] = 7 * b + t + 1j
    params = {**BASE, "apptype": "im3D", "seqcon": "nccsn", "nv": nv, "nv2": nv2,
              "arraydim": nv2, "np": 2 * samples, "te": 0.002}
    [raw] = miox.load(_folder(tmp_path, "gre3d", params, blocks, kind="i2"))
    assert raw.data.dtype == np.complex128 and raw.shape == (nv * nv2, 1, samples)
    assert raw.counters("kspace_encode_step_1").tolist() == [0, 1, 2, 0, 1, 2]
    assert raw.counters("kspace_encode_step_2").tolist() == [0, 0, 0, 1, 1, 1]
    assert raw.data[4, 0, 0] == 7 + 1 + 1j and raw.meta["EchoTime"] == 0.002
    assert raw.meta["EncodedMatrixSize"] == [samples, nv, nv2]
    assert raw.meta["EncodedFieldOfView"] == [30, 20, 10]
    # a file cut inside its last block keeps the complete blocks
    [cut] = miox.load(_folder(tmp_path, "cut", params, blocks, kind="i4", cut=5))
    assert cut.shape == (nv, 1, samples)
    assert any("1 of 2 blocks" in n for n in cut.provenance.notes)


def test_varian_arrays_receivers_and_fse_trains(tmp_path):
    # inversion times arrayed over the standard phase-encoding loop, two receivers
    nv, ns, samples = 2, 2, 4
    elements = nv * 2
    blocks = np.zeros((elements * 2, ns, samples), np.complex64)
    for element in range(elements):
        for receiver in range(2):
            blocks[element * 2 + receiver] = 100 * element + receiver
    params = {**BASE, "apptype": "im2D", "seqcon": "ncsnn", "ns": ns, "nv": nv, "np": 2 * samples,
              "arraydim": elements, "array": "ti", "ti": [0.1, 0.5], "ir": "y", "rcvrs": "yy",
              "te": [0.01, 0.02], "pss": [0.0, 0.1]}
    [raw] = miox.load(_folder(tmp_path, "ir", params, blocks))
    assert raw.shape == (elements * ns, 2, samples)
    assert raw.data[ns * 3, 1, 0] == 301 and raw.data[0, 0, 0] == 0
    assert raw.counters("repetition").tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert raw.counters("kspace_encode_step_1").tolist() == [0, 0, 1, 1, 0, 0, 1, 1]
    assert raw.meta["InversionTime"] == [0.1, 0.5] and raw.meta["EchoTime"] == 0.01  # te unarrayed
    assert raw.meta["NumberOfReceiveChannels"] == 2
    notes = " ".join(raw.provenance.notes)
    assert "2 array elements (ti) as repetitions" in notes and "2 receivers" in notes
    # fast spin echo: per block the echoes of a train, then the slices, then the trains
    etl, nv, ns = 2, 4, 2
    blocks = np.arange(etl * ns * (nv // etl) * samples, dtype=np.float32).reshape(1, -1, samples)
    params = {**BASE, "apptype": "im2Dfse", "seqcon": "nccnn", "ns": ns, "nv": nv, "etl": etl,
              "np": 2 * samples, "arraydim": 1, "petable": "fse4_2_1", "pss": [0.0, 0.1]}
    [fse] = miox.load(_folder(tmp_path, "fse", params, blocks.astype(np.complex64)))
    assert fse.counters("segment").tolist() == [0, 1] * 4
    assert fse.counters("slice").tolist() == [0, 0, 1, 1, 0, 0, 1, 1]
    assert fse.counters("kspace_encode_step_1").tolist() == [0, 1, 0, 1, 2, 3, 2, 3]
    assert fse.meta["EchoTrainLength"] == 2
    assert any("petable fse4_2_1" in n for n in fse.provenance.notes)


def test_varian_layouts_that_do_not_add_up_and_routing(tmp_path):
    blocks = np.ones((2, 3, 4), np.complex64)
    params = {**BASE, "apptype": "im2D", "seqcon": "ccsnn", "ne": 2, "ns": 2, "nv": 2,
              "arraydim": 2, "np": 8}
    [raw] = miox.load(_folder(tmp_path, "odd", params, blocks))
    assert raw.shape == (6, 1, 4) and not raw.counters("contrast").any()
    assert any("carry no counters" in n for n in raw.provenance.notes)
    assert VarianRaw.is_imaging({"apptype": ["im2Dfse"]})
    assert VarianRaw.is_imaging({"apptype": ["im3D"]})
    assert not VarianRaw.is_imaging({"apptype": ["im1Dglobal"], "nv": ["0"]})
    assert VarianRaw.is_imaging({"nv": ["64"]}) and not VarianRaw.is_imaging({"nv": ["0"]})
