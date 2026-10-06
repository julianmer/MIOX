####################################################################################################
#                                          test_ge_raw.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 06/10/26                                                                                #
#                                                                                                  #
# Purpose: GE raw k-space on a synthetic imaging P-file: frame layout, baseline stream, counters,  #
#          flags, slice geometry and header keys.                                                  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import ctypes
import struct

import numpy as np
import pytest
from spec2nii.GE.ge_hdr_fields import get_pfile_hdr_fields

import miox as gu
from miox.readers.ge_rdb import GeRdbHeader

VERSION = 26.002
HEADER_SIZE = 213684
SLICE_TABLE = 36864
SLICE_STRIDE = 72
COILS, SLICES, VIEWS, SAMPLES = 2, 2, 4, 8
CORNERS = (  # per slice in spatial order: three corner points (R, A, S) of a coronal scan
    ((131.418, -23.888, -128.0), (131.418, -23.888, 128.0), (-124.582, -23.888, -128.0)),
    ((131.418, -5.888, -128.0), (131.418, -5.888, 128.0), (-124.582, -5.888, -128.0)),
)
ACQUIRED = (2, 1)  # acquisition number (from 1) of each slice: the second slice was acquired first
EXTRAS = {  # fields beyond spec2nii's layout: value (offsets from the reader's 26.002 map)
    "rhi_slthick": 3.0, "rhi_dfov_rect": 256.0, "rhr_rh_rc_xres": 16, "rhr_rh_rc_yres": 16,
    "rhs_prtcl": b"VR prototyping template", "rhe_patage": 37, "rhe_patweight": 74840,
    "rhi_sctime": 35550000.0,
}
def _header(slice_table: bool = True, psd: bytes = b"2dfast") -> bytes:
    fields = {"_pack_": 1, "_fields_": get_pfile_hdr_fields(VERSION)}
    h = type("Hdr", (ctypes.LittleEndianStructure,), fields)()
    h.rhr_rh_rdbm_rev = VERSION
    h.rhr_rdb_hdr_off_data = HEADER_SIZE
    h.rhr_rdb_hdr_off_data_acq_tab = SLICE_TABLE if slice_table else 0
    h.rhr_rh_logo = b"GE_MED_NMR"
    h.rhr_rh_scan_date, h.rhr_rh_scan_time = b"04/11/118", b"08:59"
    h.rhr_rh_npasses, h.rhr_rh_nslices, h.rhr_rh_nechoes, h.rhr_rh_nframes = 1, SLICES, 1, VIEWS
    h.rhr_rh_frame_size, h.rhr_rh_point_size = SAMPLES, 2
    h.rhr_rh_da_xres, h.rhr_rh_da_yres = SAMPLES, VIEWS + 1
    h.rhr_rh_raw_pass_size = COILS * SLICES * (VIEWS + 1) * SAMPLES * 4
    setattr(h, "rhr_rh_dab[0]_start_rcv", 0)
    setattr(h, "rhr_rh_dab[0]_stop_rcv", COILS - 1)
    h.rhr_rh_ps_mps_freq = 1276969780
    h.rhe_magstrength, h.rhe_ex_no = 30000, 26356
    h.rhe_hospname, h.rhe_ex_sysid, h.rhe_ex_verscre = b"NIH FMRIF", b"fmrif3tb", b"27"
    h.rhe_uniq_sys_id, h.rhe_patname, h.rhe_patid = b"000301496MR3T6MR", b"Phantom^FBIRN", b"P01"
    h.rhe_dateofbirth = b"19800909"
    h.rhs_position, h.rhs_entry, h.rhs_se_no, h.rhs_se_desc = 1, 1, 2, b"Basic GRE scan"
    h.rhi_dfov, h.rhi_scanspacing, h.rhi_freq_dir = 256.0, 15.0, 2
    h.rhi_tr, h.rhi_te, h.rhi_ti, h.rhi_mr_flip, h.rhi_nex = 250000, 5100, 0, 10, 1.0
    h.rhi_numecho, h.rhi_rawrunnum = 1, 20480
    h.rhi_psdname, h.rhi_cname = psd, b"8HRBRAIN"
    h.rhi_ctr_R, h.rhi_ctr_A, h.rhi_ctr_S = 3.418, -23.888, 13.393
    raw = bytearray(bytes(h)).ljust(HEADER_SIZE, b"\0")
    offsets = {name: (offset, fmt) for name, offset, fmt in GeRdbHeader.EXTRAS_26}
    for name, value in EXTRAS.items():
        offset, fmt = offsets[name]
        struct.pack_into("<" + fmt, raw, offset, value)
    for k, corners in enumerate(CORNERS):
        flat = [x for point in corners for x in point]
        entry = SLICE_TABLE + k * SLICE_STRIDE
        struct.pack_into("<hh9fhh", raw, entry, 0, ACQUIRED[k], *flat, 3, 1)
    return bytes(raw)


def _frames() -> np.ndarray:
    """(coil, slice, view, sample); view 0 is the baseline."""
    frames = np.zeros((COILS, SLICES, VIEWS + 1, SAMPLES), dtype=np.complex64)
    for c in range(COILS):
        for s in range(SLICES):
            frames[c, s, 0] = complex(-(c + 1), s)
            for v in range(1, VIEWS + 1):
                frames[c, s, v] = 100 * c + 10 * s + v + 1j * np.arange(SAMPLES)
    return frames


def _pfile(path, **kwargs) -> np.ndarray:
    frames = _frames()
    data = np.stack([frames.real, frames.imag], axis=-1).astype("<i2")
    path.write_bytes(_header(**kwargs) + data.tobytes())
    return frames


def test_pfile_imaging(tmp_path):
    path = tmp_path / "P12345.7"
    frames = _pfile(path)
    sniff = gu.sniff(path)
    assert (sniff.format, sniff.kind) == (gu.Format.GE_PFILE, gu.Kind.RAW)
    image, baseline = gu.load(path)
    assert image.name == "P12345" and baseline.name == "P12345_baseline"
    assert image.shape == (SLICES * VIEWS, COILS, SAMPLES)
    assert baseline.shape == (SLICES, COILS, SAMPLES)
    assert image.counters("slice").tolist() == [1] * VIEWS + [0] * VIEWS  # spatial index
    assert image.headers["idx"]["user"][:, 1].tolist() == [0] * VIEWS + [1] * VIEWS  # slot
    assert image.counters("kspace_encode_step_1").tolist() == list(range(VIEWS)) * SLICES
    assert image.data[5, 1, 3] == complex(100 + 10 + 2, 3)  # slice 1, view 2, coil 1, sample 3
    assert np.array_equal(image.data[5], frames[:, 1, 2])
    assert baseline.data[1, 0, 0] == complex(-1, 1)
    assert baseline.counters("slice").tolist() == [1, 0]
    assert image.flags_of(0) == ["FIRST_IN_SLICE"] and image.flags_of(3) == ["LAST_IN_SLICE"]
    assert image.flags_of(1) == [] and baseline.flags_of(0) == []
    head = image.headers
    assert head["measurement_uid"][0] == 20480 and head["center_sample"][0] == SAMPLES // 2
    assert head["scan_counter"].tolist() == list(range(8)) and head["active_channels"][0] == COILS
    # slot 0 holds the second slice: centre (3.418, -5.888, 0) in the scanner frame, head first
    # supine to LPS, shifted by the image header's centre S of 13.393 that the table's points lack
    assert head["position"][0].tolist() == pytest.approx([-3.418, 5.888, 13.393], abs=1e-3)
    assert head["position"][4].tolist() == pytest.approx([-3.418, 23.888, 13.393], abs=1e-3)
    assert head["read_dir"][0].tolist() == [0, 0, 1] and head["phase_dir"][0].tolist() == [1, 0, 0]
    assert head["slice_dir"][0].tolist() == [0, 1, 0]
    m = image.meta
    assert m["EchoTime"] == pytest.approx(0.0051) and m["RepetitionTime"] == 0.25
    assert "InversionTime" not in m and m["ExcitationFlipAngle"] == 10
    assert m["FieldStrength"] == 3.0 and m["SpectrometerFrequency"] == pytest.approx(127.696978)
    assert m["PatientPosition"] == "HFS" and m["Manufacturer"] == "GE" and m["StudyID"] == "26356"
    assert m["SeriesNumber"] == 2 and m["PatientAge"] == 37 and m["PatientWeight"] == 74.84
    assert m["SliceThickness"] == 3.0 and m["SpacingBetweenSlices"] == 18.0
    assert m["ScanDuration"] == 35.55 and m["ProtocolName"] == "VR prototyping template"
    assert m["PulseSequenceName"] == "2dfast" and m["ReceiveCoilName"] == "8HRBRAIN"
    assert m["InPlanePhaseEncodingDirectionDICOM"] == "ROW"
    assert m["AcquisitionDateTime"] == "2018-04-11T08:59"
    assert m["NumberOfReceiveChannels"] == COILS and m["NumberOfPoints"] == SAMPLES
    assert m["EncodedMatrixSize"] == [SAMPLES, VIEWS, 1] and m["ReconMatrixSize"] == [16, 16, 1]
    assert m["EncodedFieldOfView"] == [256.0, 256.0, 3.0]
    assert m["EncodingLimits"]["slice"] == {"minimum": 0, "maximum": 1, "center": 1}
    assert m["EncodingLimits"]["kspace_encoding_step_1"]["center"] == 2
    assert m["PatientName"] == "Phantom^FBIRN" and m["PatientDoB"] == "19800909"
    assert m.fields["EchoTime"] == "rhi_te"
    vendor = m.vendor
    assert vendor["rhi_slthick"] == 3.0 and vendor["rhs_prtcl"] == "VR prototyping template"
    assert len(vendor["slice_table"]) == 2 and vendor["slice_table"][1]["slice_in_pass"] == 1
    assert vendor["slice_table"][0]["transpose"] == 3 and vendor["slice_table"][0]["rotate"] == 1
    assert vendor["version"] == pytest.approx(VERSION)
    assert image.encoding["readout_order"] == "pass, slice, echo, view"
    assert image.encoding["views"] == VIEWS and baseline.encoding["baseline_frames"] == 1


def test_pfile_without_slice_table(tmp_path):
    path = tmp_path / "P00001.7"
    _pfile(path, slice_table=False)
    image, _ = gu.load(path)
    assert not image.headers["position"].any() and not image.headers["read_dir"].any()
    assert any("no slice geometry" in note for note in image.provenance.notes)


def test_spectroscopy_sequences_stay_with_spec2nii(tmp_path):
    path = tmp_path / "P00002.7"
    _pfile(path, psd=b"probe-p")
    assert GeRdbHeader.from_file(path).spectroscopy_mapper()
    path = tmp_path / "P00003.7"
    _pfile(path)
    assert not GeRdbHeader.from_file(path).spectroscopy_mapper()
