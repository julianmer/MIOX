####################################################################################################
#                                       test_philips_raw.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/10/26                                                                                #
#                                                                                                  #
# Purpose: Philips raw/lab/sin exports on synthetic files: bit-packed blocks written by a small    #
#          encoder, labels of several control types, a block whose size on disk differs from its   #
#          label, the sin parameters as keys, and the old plain-sample layout; imaging k-space     #
#          exported as .data/.list.                                                                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import struct

import numpy as np
import pytest

import miox
from miox.readers.philips_raw_lab import MiraDecoder, PhilipsRawLabReader

CHANNELS, SAMPLES = 2, 8
SIN = """ CAUTION - Investigational device

 00 00 00: scan_name                    : COR GRE
 00 00 00: start_scan_date_time         : 17-JUN-2025 21:06:21.61
 00 00 00: sample_time_interval         :      22.1600
 01 00 00: nr_dynamic_scans             :            1
 01 00 00: nr_locations                 :            2
 01 00 00: nr_echoes                    :            1
 01 00 00: nr_measurements              :            1
 01 00 00: recon_resolutions            :            8            4            1            1
 01 00 00: repetition_times             :      29.7091
 01 00 00: inversion_delays             :       0.0000       0.0000
 01 00 00: voxel_sizes                  :       1.0000       2.0000       5.0000
 01 01 00: min_encoding_numbers         :           -4           -2            0            0
 01 01 00: max_encoding_numbers         :            3            1            0            0
 01 01 00: oversample_factors           :       2.0000       1.0000       1.0000       1.0000
 01 01 00: echo_times                   :       9.2081
 01 00 00: channel_names                : _ODU_DCC1/SMALL_BODY_ARRAY.E01
 02 00 00: channel_names                : _ODU_DCC1/SMALL_BODY_ARRAY.E02
 01 00 00: receiving_coils              : SMALL_BODY_ARRAY
 01 00 00: nr_measured_channels         :            2
 01 00 00: flip_angles                  :      15.0000
 00 00 00: slice_thickness              :       5.0000
 01 00 00: spacing_between_slices_arr   :       6.0000
 00 00 01: loc_ap_rl_fh_row_image_oris  :       0.0000       1.0000       0.0000
 00 00 01: loc_ap_rl_fh_col_image_oris  :       0.0000       0.0000      -1.0000
 00 00 01: loc_ap_rl_fh_offcentres      :      -6.1686       2.0000      -0.7711
 00 00 02: loc_ap_rl_fh_offcentres      :      -6.1686       2.0000       5.2289
"""


def _pack(values: np.ndarray, resolution: int, shift: int = 0) -> bytes:
    """A MIRA chunk payload: groups of 16 values of one bit resolution and shift."""
    bits = []
    values = np.asarray(values, dtype=np.int64)
    addend = (1 << shift) >> 1
    for start in range(0, len(values), 16):
        group = (values[start : start + 16] - addend) >> shift
        bits += [int(b) for b in f"{resolution:05b}{shift:05b}"]
        for v in group:
            bits += [int(b) for b in format(int(v) & ((1 << resolution) - 1), f"0{resolution}b")]
    bits += [0] * (-len(bits) % 32)
    words = [int("".join(map(str, bits[i : i + 32])), 2) for i in range(0, len(bits), 32)]
    return struct.pack(f"<{len(words)}I", *words)


def _block(values: np.ndarray, chunks: int = 2, resolution: int = 12, shift: int = 0) -> bytes:
    """A bit-packed block: ``chunks`` chunks in reversed order, then the 8 trailing bytes."""
    parts = np.array_split(np.asarray(values, dtype=np.int64), chunks)
    out = b""
    offset = 0
    pieces = []
    for part in parts:
        pieces.append((offset, part))
        offset += 4 * len(part)
    for offset, part in reversed(pieces):
        payload = _pack(part, resolution, shift)
        out += struct.pack("<HHI", 4 * len(part), len(payload), offset) + payload
    return out + bytes(8)


def _label(control: int, data_size: int, coded_size: int, **fields) -> bytes:
    label = np.zeros(1, dtype=PhilipsRawLabReader.NEW_LABEL)
    label["data_size"], label["coded_data_size"] = data_size, coded_size
    label["label_type"], label["control"], label["raw_format"] = 0x7F01, control, 6
    label["normalization_factor"] = 157.88
    for name, value in fields.items():
        label[name] = value
    return label.tobytes()


def _values(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(-2000, 2000, n, dtype=np.int64)


def _write(tmp_path, stem="scan"):
    """Noise block, a DC-fixed block whose size on disk is three times its label, an echo-phase
    block of one channel and four normal profiles of two slices."""
    labels, raw = b"", bytes(512)
    noise = _values(1, CHANNELS * 6 * 2)
    block = _block(noise, chunks=3)
    labels += _label(16, 4 * len(noise), len(block), seq_nr=100, e1_profile_nr=2)
    raw += block
    labels += _label(18, 16, 16 + 8, seq_nr=101, raw_format=2)  # on disk: 3 x 16 bytes + 8
    raw += bytes(48) + bytes(8)
    labels += _label(8, 0, 0, seq_nr=102)  # a control event without data
    echo = _values(2, SAMPLES * 2)
    block = _block(echo, chunks=1)
    labels += _label(3, 4 * len(echo), len(block), seq_nr=103, channels_active=1, e1_profile_nr=2)
    raw += block
    profiles = {}
    for n, (location, e1) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
        shift = n % 2  # a shift drops the low bit: the values are stored as the encoder keeps them
        values = _values(10 + n, CHANNELS * SAMPLES * 2)
        values = (values - (1 << shift >> 1) >> shift << shift) + (1 << shift >> 1)
        profiles[location, e1] = values
        block = _block(values, chunks=2, resolution=13, shift=shift)
        labels += _label(
            0, 4 * len(values), len(block), seq_nr=200 + n, e1_profile_nr=e1, location_nr=location,
            random_phase=1000 * n, rtop_offset=7, rr_interval=800, mix_nr=1, dynamic_scan_nr=0,
        )
        raw += block
    (tmp_path / f"{stem}.lab").write_bytes(labels)
    (tmp_path / f"{stem}.raw").write_bytes(raw)
    (tmp_path / f"{stem}.sin").write_text(SIN)
    return profiles, noise, echo


def test_decoder_roundtrip():
    values = _values(5, 70)
    for resolution, shift in ((12, 0), (14, 2), (5, 0)):
        scaled = values if resolution >= 12 else (values >> 7)
        if shift:
            scaled = (scaled >> shift << shift) + ((1 << shift) >> 1)
        payload = np.frombuffer(_pack(scaled, resolution, shift), dtype=np.uint8)
        assert MiraDecoder.chunk(payload, len(scaled)).tolist() == scaled.tolist()


def test_philips_raw_export(tmp_path):
    profiles, noise, echo = _write(tmp_path)
    for suffix in ("lab", "raw", "sin"):
        sniff = miox.sniff(tmp_path / f"scan.{suffix}")
        assert sniff.format == miox.Format.PHILIPS_RAW_LAB and sniff.kind == miox.Kind.RAW
        assert sniff.vendor == "philips"
        assert len(sniff.companions) == 2 and sniff.confidence == 1.0
    image, echo_phase, noise_data = miox.load(tmp_path / "scan.lab")
    assert image.name == "scan" and noise_data.name == "scan_frc_noise_data"
    assert echo_phase.name == "scan_echo_phase_data"
    assert image.shape == (4, CHANNELS, SAMPLES) and image.data.dtype == np.complex128
    for r, (location, e1) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
        values = profiles[location, e1].reshape(CHANNELS, SAMPLES, 2)
        assert np.array_equal(image.data[r].real, values[..., 0])
        assert np.array_equal(image.data[r].imag, values[..., 1])
    assert image.counters("kspace_encode_step_1").tolist() == [0, 0, 1, 1]
    assert image.counters("slice").tolist() == [0, 1, 0, 1]
    assert image.headers["idx"]["user"][:, 0].tolist() == [1] * 4  # mix number
    assert image.headers["scan_counter"].tolist() == [200, 201, 202, 203]
    assert image.headers["physiology_time_stamp"][0].tolist() == [7, 800, 0]
    assert image.headers["user_int"][2].tolist() == [0, 2000, 0, 0, 0, 6, 0, 0]
    assert image.headers["user_float"][0, 0] == pytest.approx(157.88)
    assert image.headers["center_sample"][0] == SAMPLES // 2 and image.flags_of(0) == []
    assert image.headers["position"][0].tolist() == pytest.approx([2.0, -6.1686, -0.7711])
    assert image.headers["position"][1].tolist() == pytest.approx([2.0, -6.1686, 5.2289])
    assert image.headers["slice_dir"][0].tolist() == pytest.approx([0, 1, 0])
    assert noise_data.shape == (1, CHANNELS, 6)
    assert noise_data.flags_of(0) == ["IS_NOISE_MEASUREMENT"]
    assert np.array_equal(noise_data.data[0].real, noise.reshape(CHANNELS, 6, 2)[..., 0])
    assert echo_phase.shape == (1, 1, SAMPLES) and echo_phase.headers["active_channels"][0] == 1
    assert np.array_equal(echo_phase.data[0, 0].imag, echo.reshape(SAMPLES, 2)[:, 1])
    m = image.meta
    assert m["Manufacturer"] == "Philips" and m["ProtocolName"] == "COR GRE"
    assert m["RepetitionTime"] == pytest.approx(0.0297091)
    assert m["EchoTime"] == pytest.approx(0.0092081)
    assert "InversionTime" not in m and m["ExcitationFlipAngle"] == 15.0
    assert m["DwellTime"] == pytest.approx(22.16e-6) and m["NumberOfReceiveChannels"] == 2
    assert m["ReceiveCoilName"] == "SMALL_BODY_ARRAY" and m["SliceThickness"] == 5.0
    assert m["SpacingBetweenSlices"] == 6.0 and m["NumberOfAverages"] == 1
    assert m["AcquisitionDateTime"] == "2025-06-17T21:06:21.61" and m["NumberOfPoints"] == SAMPLES
    assert m["ReconMatrixSize"] == [8, 4, 1] and m["EncodedMatrixSize"] == [16, 4, 1]
    assert m["ReconFieldOfView"] == [8.0, 8.0, 5.0] and m["EncodedFieldOfView"] == [16.0, 8.0, 5.0]
    limits = m["EncodingLimits"]
    assert limits["kspace_encoding_step_1"] == {"minimum": 0, "maximum": 3, "center": 1}
    assert limits["kspace_encoding_step_2"] == {"minimum": 0, "maximum": 0, "center": 0}
    assert m["EncodingLimits"]["slice"] == {"minimum": 0, "maximum": 1, "center": 0}
    assert m["ImageOrientationPatientDICOM"] == [1.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    assert m.fields["EchoTime"] == "sin/echo_times/0/values"
    assert m.vendor["sin"]["channel_names"][1]["index"] == [2, 0, 0]
    assert m.vendor["labels"]["control"] == [16, 18, 8, 3, 0, 0, 0, 0]
    assert m.vendor["label_layout"] == "new"
    assert any("32 bytes skipped before label 3" in note for note in image.provenance.notes)
    assert image.encoding["controls"] == [0] and image.encoding["sample_bytes"] == 4


def test_old_layout_with_plain_samples(tmp_path):
    values = np.arange(CHANNELS * SAMPLES * 2, dtype="<i2") - 7
    label = np.zeros(1, dtype=PhilipsRawLabReader.OLD_LABEL)
    label["data_size"], label["coded_data_size"] = values.nbytes, values.nbytes
    label["label_type"], label["control"], label["seq_nr"], label["e1_profile_nr"] = 0x7F01, 0, 5, 3
    (tmp_path / "old.lab").write_bytes(label.tobytes())
    (tmp_path / "old.raw").write_bytes(bytes(512) + values.tobytes())
    (tmp_path / "old.sin").write_text(SIN + " 00 00 00: enable_pda : 1\n")
    [image] = miox.load(tmp_path / "old.lab")
    assert image.shape == (1, CHANNELS, SAMPLES) and image.data.dtype == np.complex64
    assert image.data[0, 1, 0] == complex(16 - 7, 17 - 7)
    assert image.counters("kspace_encode_step_1").tolist() == [3]
    assert image.meta.vendor["label_layout"] == "old" and image.encoding["sample_bytes"] == 2


def test_missing_companion_is_an_error(tmp_path):
    (tmp_path / "x.lab").write_bytes(_label(0, 0, 0))
    with pytest.raises(miox.ReaderError, match="no .raw file"):
        miox.load(tmp_path / "x.lab")


def test_data_list_imaging_kspace(tmp_path):
    """Two ky lines of two coils (the readout spans the kx range) and one noise readout."""
    header = (
        "# === DATA DESCRIPTION FILE ======================================================\n"
        ".    0    0    0  number_of_echoes                   :     1\n"
        ".    0    0    0  number of coil channels            :     2\n"
        ".    0    0    0  kx_range                           :    -2     1\n"
        ".    0    0    0  ky_range                           :    -1     0\n"
        ".    0    0    0  kx_oversample_factor               :    2.0000\n"
        ".    0    0    0  X-resolution                       :     2\n"
        ".    0    0    0  Y-resolution                       :     2\n"
    )
    values, lines, offset = [], [], 0
    rows = [("NOI", 0, 0, 0, 6), ("NOI", 0, 1, 0, 6), ("STD", 0, 0, -1, 4), ("STD", 0, 1, -1, 4),
            ("STD", 1, 0, 0, 4), ("STD", 1, 1, 0, 4)]
    for typ, echo, coil, ky, count in rows:
        samples = np.arange(count) + 10 * coil + 100 * (ky + 1) + 1j * echo
        values.append(np.column_stack([samples.real, samples.imag]).astype("<f4").ravel())
        lines.append(f"  {typ}  0  0  0  {echo}  0  {coil}  0  0  {ky}  0  0  0  1  0  0  0  0  0"
                     f"  {8 * count}  {offset}")
        offset += 8 * count
    (tmp_path / "raw_001.list").write_text(header + "\n".join(lines) + "\n# END OF DATA\n")
    (tmp_path / "raw_001.data").write_bytes(np.concatenate(values).tobytes())
    imaging, noise = miox.load(tmp_path / "raw_001.data")
    assert imaging.data.shape == (2, 2, 4) and noise.data.shape == (1, 2, 6)
    assert np.allclose(imaging.data[1, 1], np.arange(4) + 10 + 100 + 1j)
    assert imaging.headers["idx"]["kspace_encode_step_1"].tolist() == [0, 1]
    assert imaging.headers["idx"]["contrast"].tolist() == [0, 1]
    assert imaging.encoding["encodedSpace"]["matrixSize"] == {"x": 4, "y": 2, "z": 1}
    assert int(noise.headers["flags"][0]) == 1 << 18  # ISMRMRD noise measurement
