####################################################################################################
#                                     test_sequence_formats.py                                     #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/10/26                                                                                #
#                                                                                                  #
# Purpose: Pulseq files of revisions 1.3, 1.4 and 1.5 (column layouts, shape decompression,        #
#          extensions, definitions as keys) and BART arrays (.cfl with .hdr, .ra) on synthetic     #
#          files.                                                                                  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import struct

import numpy as np
import pytest

import miox as gu
from miox.readers.pulseq import PulseqReader

SEQ_15 = """# Pulseq sequence file
# Created by hand

[VERSION]
major 1
minor 5
revision 1

[DEFINITIONS]
AdcRasterTime 1e-07
BlockDurationRaster 1e-05
GradientRasterTime 1e-05
RadiofrequencyRasterTime 1e-06
Name fid
FOV 0.256 0.256 0.004
TE 0.005
TR 0.01
TotalDuration 0.10786

# Format of blocks:
# NUM DUR RF  GX  GY  GZ  ADC  EXT
[BLOCKS]
1 42 1 0 0 0 0 1
2 500 0 0 0 2 0 0
3 10244 0 0 0 0 1 0

[RF]
1 833.333 1 2 0 150 100 0 0 0 0 e

[GRADIENTS]
3 790127 0 -550073 6 0 980

[TRAP]
2 25000 30 940 30 100

[ADC]
1 1024 100000 20 0 0 0 0 0

[EXTENSIONS]
1 1 1 0
2 1 2 1

extension TRIGGERS 1
1 2 1 0 2000
2 1 3 500 100

extension DELAYS 2
1 0 -7840 2 TE

[SHAPES]

shape_id 1
num_samples 300
1
0
0
297

shape_id 2
num_samples 300
0
0
298

shape_id 6
num_samples 4
0
0.5
1
0.5

[SIGNATURE]
Type md5
Hash 217d8a34f69ccfab42d754cefa333d7f
"""
SEQ_13 = """[VERSION]
major 1
minor 3
revision 0

[DEFINITIONS]
Name old

[BLOCKS]
1 0 1 0 0 0 0 0
2 1 0 0 0 0 0 0

[RF]
1 500 1 2 10 0 0

[DELAYS]
1 2000

[GRADIENTS]
4 1000 3 0

[SHAPES]

shape_id 1
num_samples 2
1
0

shape_id 2
num_samples 4
0
0
2

shape_id 3
num_samples 3
0.1
0.1
1
"""


def test_pulseq_15(tmp_path):
    path = tmp_path / "fid.seq"
    path.write_text(SEQ_15)
    sniff = gu.sniff(path)
    assert (sniff.format, sniff.kind) == (gu.Format.PULSEQ, gu.Kind.SEQUENCE)
    [seq] = gu.load(path)
    assert isinstance(seq, gu.PulseSequence) and seq.version == "1.5.1"
    assert seq.n_blocks == 3 and seq.duration == pytest.approx((42 + 500 + 10244) * 1e-5)
    assert seq.blocks["duration"].tolist() == [42, 500, 10244]
    assert seq.blocks["ext"].tolist() == [1, 0, 0]
    rf = seq.events["rf"][1]
    assert rf["amp"] == pytest.approx(833.333) and rf["center"] == 150 and rf["use"] == "e"
    assert rf["freq_ppm"] == 0 and rf["time_id"] == 0
    gradient = seq.events["gradients"][3]
    assert gradient["last"] == -550073 and gradient["delay"] == 980
    trap = {"amp": 25000, "rise": 30, "flat": 940, "fall": 30, "delay": 100}
    assert seq.events["trap"][2] == trap
    adc = seq.events["adc"][1]
    assert adc["num"] == 1024 and adc["dwell"] == 100000 and adc["phase_shape_id"] == 0
    assert seq.extensions["list"][2] == {"type": 1, "ref": 2, "next": 1}
    specs = seq.extensions["specifications"]
    assert specs["TRIGGERS"]["type"] == 1 and specs["TRIGGERS"]["rows"][1] == [2, 1, 3, 500, 100]
    assert specs["DELAYS"]["rows"] == [[1, 0, -7840, 2, "TE"]]
    assert seq.shapes[1].tolist() == [1.0] * 300  # compressed: 1, then 0 repeated 299 times
    assert seq.shapes[2].tolist() == [0.0] * 300
    assert seq.shapes[6].tolist() == [0, 0.5, 1, 0.5]  # stored uncompressed
    assert seq.signature == {"Type": "md5", "Hash": "217d8a34f69ccfab42d754cefa333d7f"}
    m = seq.meta
    assert m["SequenceName"] == "fid" and m["EchoTime"] == 0.005 and m["RepetitionTime"] == 0.01
    assert m["ScanDuration"] == 0.10786 and m["EncodedFieldOfView"] == [256.0, 256.0, 4.0]
    assert m.vendor["definitions"]["AdcRasterTime"] == 1e-7
    assert m.fields["EchoTime"] == "definitions/TE"
    assert seq.to_dict()["events"] == {"rf": 1, "gradients": 1, "trap": 1, "adc": 1}
    assert not seq.provenance.notes


def test_pulseq_13_delays(tmp_path):
    path = tmp_path / "old.seq"
    path.write_text(SEQ_13)
    [seq] = gu.load(path)
    assert seq.version == "1.3.0" and seq.duration is None
    assert seq.blocks.dtype.names == ("id", "delay", "rf", "gx", "gy", "gz", "adc", "ext")
    assert seq.blocks["delay"].tolist() == [0, 1] and seq.events["delays"][1] == {"delay": 2000}
    rf = {"amp": 500, "mag_id": 1, "phase_id": 2, "delay": 10, "freq": 0, "phase": 0}
    assert seq.events["rf"][1] == rf
    assert seq.events["gradients"][4] == {"amp": 1000, "shape_id": 3, "delay": 0}
    assert seq.shapes[1].tolist() == [1.0, 1.0] and seq.shapes[2].tolist() == [0.0] * 4
    assert seq.shapes[3].tolist() == pytest.approx([0.1, 0.2, 0.3])  # compressed, same count
    assert seq.to_dict()["duration"] is None


def test_shape_decompression_examples():
    """The specification's examples: a ramp, 100 zeros, 100 ones."""
    ramp = PulseqReader.decompress(np.array([0, 0.1, 0.15, 0.25, 0.5, 0, 0, 4, -0.25, -0.25, 2]))
    assert ramp.tolist() == pytest.approx(
        [0, 0.1, 0.25, 0.5, 1, 1, 1, 1, 1, 1, 1, 0.75, 0.5, 0.25, 0]
    )
    assert PulseqReader.decompress(np.array([0, 0, 98])).tolist() == [0.0] * 100
    assert PulseqReader.decompress(np.array([1, 0, 0, 97])).tolist() == [1.0] * 100


def test_bart_cfl_and_ra(tmp_path):
    values = (np.arange(24, dtype=np.float32) + 1j * np.arange(24, dtype=np.float32)[::-1])
    values = values.astype(np.complex64)
    (tmp_path / "ksp.cfl").write_bytes(values.tobytes())
    header = "# Dimensions\n2 3 4 1 1 1 1 1 1 1 1 1 1 1 1 1\n# Command\nbart phantom -k\n"
    (tmp_path / "ksp.hdr").write_text(header)
    for suffix in ("hdr", "cfl"):
        sniff = gu.sniff(tmp_path / f"ksp.{suffix}")
        assert (sniff.format, sniff.kind) == (gu.Format.BART_CFL, gu.Kind.IMAGING)
        assert len(sniff.companions) == 1
    [image] = gu.load(tmp_path / "ksp.hdr")
    assert isinstance(image, gu.MRImage) and image.shape == (2, 3, 4)
    assert image.dims == ("i", "j", "k")
    assert image.data[1, 0, 0] == values[1] and image.data[0, 1, 0] == values[2]  # column-major
    assert image.data[0, 0, 1] == values[6] and image.data.dtype == np.complex64
    assert image.meta.vendor["header"]["Command"] == "bart phantom -k"
    assert image.meta.vendor["dimensions"][:4] == [2, 3, 4, 1] and image.name == "ksp"
    assert any("k-space" in note for note in image.provenance.notes)
    # a coil dimension is named, trailing singletons dropped
    (tmp_path / "coils.cfl").write_bytes(np.ones(2 * 3 * 1 * 4, dtype=np.complex64).tobytes())
    (tmp_path / "coils.hdr").write_text("# Dimensions\n2 3 1 4 1 1 1 1 1 1 1 1 1 1 1 1\n")
    [coils] = gu.load(tmp_path / "coils.cfl")
    assert coils.shape == (2, 3, 1, 4) and coils.dims == ("i", "j", "k", "coil")
    # rawarray
    header = struct.pack("<6Q", 0x7961727261776172, 0, 4, 8, values.nbytes, 3)
    header += struct.pack("<3Q", 2, 3, 4)
    (tmp_path / "ksp.ra").write_bytes(header + values.tobytes())
    sniff = gu.sniff(tmp_path / "ksp.ra")
    assert sniff.format == gu.Format.BART_CFL and sniff.evidence == ("rawarray magic",)
    [ra] = gu.load(tmp_path / "ksp.ra")
    assert np.array_equal(ra.data, image.data)
    assert ra.meta.vendor["header"]["data_bytes"] == values.nbytes


def _mod(path, rho, gz, b1max=0.25, gmax=4.0, floats=(1.0,)):
    """A .mod file as TOPPE's writemod.m writes it: one coil, one waveform."""
    description = b"test module\n"
    blob = struct.pack(">h", len(description)) + description
    blob += struct.pack(">hhh", 1, len(rho), 1)
    blob += f"b1max:  {b1max:f}\ngmax:   {gmax:f}\n".encode()
    blob += struct.pack(">h", 3) + struct.pack(">3h", 0, len(rho), 7)
    blob += struct.pack(">h", len(floats)) + "".join(f"{v:f}\n" for v in floats).encode()
    zeros = [0] * len(rho)
    for wave in (rho, zeros, zeros, zeros, gz):
        blob += struct.pack(f">{len(rho)}h", *wave)
    path.write_bytes(blob)


def test_toppe_sequence(tmp_path):
    _mod(tmp_path / "ex.mod", [0, 32766, 16382, 0], [16382, 16382, 16382, 16382])
    _mod(tmp_path / "ro.mod", [0, 0, 0, 0], [0, 32766, -32766, 0])
    (tmp_path / "modules.txt").write_text(
        "Total number of unique cores\n2\nwavfile_name\tduration(us)\thasRF?\thasDAQ?\n"
        "ex.mod\t0\t1\t0\nro.mod\t100\t0\t1\t-1\n")
    row = [1, 32766, 32766, 0, 0, 16382, 0, 0, 1, 0, 0, 100, 100, 0, 250, 1,
           32766, 0, 0, 0, 32766, 0, 0, 0, 32766, 0, 0, 3]
    (tmp_path / "scanloop.txt").write_text(
        "nt\tmaxslice\tmaxecho\tmaxview\tscandur\tversion\n2\t1\t0\t1\t5000\t6\n"
        "Core a_rf a_th a_gx a_gy a_gz dabslice dabecho dabview dabmode rot rfphase recphase "
        "textra freq waveform rotmat...\n"
        + "\n".join("\t".join(str(v) for v in r) for r in (row, [2] + row[1:])) + "\n")
    (tmp_path / "toppe1.entry").write_text(
        "/usr/g/research/pulseq/test/\nmodules.txt\nscanloop.txt\nex.mod\nro.mod\n"
        "seqstamp.txt\n")
    formats = {p.name: gu.sniff(p).format for p in tmp_path.iterdir()}
    assert formats == {"ex.mod": gu.Format.TOPPE_MOD, "ro.mod": gu.Format.TOPPE_MOD,
                       "modules.txt": gu.Format.TOPPE_MODULES,
                       "scanloop.txt": gu.Format.TOPPE_LOOP,
                       "toppe1.entry": gu.Format.TOPPE_ENTRY}
    [seq] = gu.load(tmp_path / "toppe1.entry")
    assert seq.version == "6" and seq.n_blocks == 2
    assert seq.blocks.dtype.names[-3:] == ("trigger", "trigger_out", "core")
    assert seq.blocks["module"].tolist() == [1, 2] and seq.blocks["core"].tolist() == [3, 3]
    assert seq.blocks["rotation_0"][0] == 32766 and seq.blocks["frequency_hz"][0] == 250
    assert seq.definitions["entry"]["b1_scaling_module"] == "ex.mod"
    assert seq.meta["ScanDuration"] == pytest.approx(5e-3)
    assert "sequence_stamp" not in seq.definitions and seq.provenance.notes == ()
    ex, ro = seq.events["modules"][1], seq.events["modules"][2]
    assert ex["has_rf"] and not ex["has_daq"] and ro["duration_us"] == 100
    assert ro["trigger_position"] == -1
    assert ex["rf"][:, 0, 0].real == pytest.approx([0, 0.25e-4, 0.25e-4 * 16382 / 32766, 0])
    assert ex["gz"][:, 0] == pytest.approx([4e-2 * 16382 / 32766] * 4)
    assert ro["gz"][:, 0] == pytest.approx([0, 4e-2, -4e-2, 0])
    assert ex["discarded_samples"] == 0 and ex["rf_samples"] == 4 and ex["int_parameters"] == [7]
    [loop] = gu.load(tmp_path / "scanloop.txt")
    assert loop.n_blocks == 2 and loop.events["modules"] == {}
