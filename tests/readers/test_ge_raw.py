####################################################################################################
#                                          test_ge_raw.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 06/10/26                                                                                #
#                                                                                                  #
# Purpose: GE raw k-space on synthetic files: an imaging P-file (frame layout, baseline stream,    #
#          counters, flags, slice geometry, header keys) and ScanArchives (control packets with    #
#          one frame each or hyper-frame blocks, the archive's own metadata and saved files).      #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import base64
import ctypes
import struct

import h5py
import numpy as np
import pytest
from spec2nii.GE.ge_hdr_fields import get_pfile_hdr_fields

import miox
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
HEADER_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>
<!DOCTYPE boost_serialization>
<boost_serialization signature="serialization::archive" version="10">
<Header class_id="0" tracking_level="0" version="1">
<px class_id="1" tracking_level="1" version="0" object_id="_0">
<ArchiveType>ScanArchive</ArchiveType><ExamNumber>26356</ExamNumber><SeriesNumber>2</SeriesNumber>
<DateTime><ptime_date><date>20180411</date></ptime_date><ptime_time_duration>
<time_duration_hours>8</time_duration_hours><time_duration_minutes>59</time_duration_minutes>
<time_duration_seconds>46</time_duration_seconds>
<time_duration_fractional_seconds>822405144</time_duration_fractional_seconds>
</ptime_time_duration></DateTime>
<SystemID><HospitalName>NIH FMRIF</HospitalName><SoftwareLabel>DV26.0_R01_1725.a</SoftwareLabel>
</SystemID></px></Header></boost_serialization>
"""
UPDATE_XML = '<?xml version="1.0"?>\n<Update><Map><count>1</count></Map></Update>\n'


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


def _packet(opcode: int, index: int, time: int, slice_number: int = 0, view: int = 0,
            echo: int = 0, frames: int = 0) -> bytes:
    raw = bytearray(44)
    struct.pack_into("<IIH", raw, 0, 44, time, opcode)
    raw[10] = echo
    if opcode == 6:
        raw[13], raw[14], raw[20] = slice_number >> 8, slice_number & 255, frames
    elif opcode == 1:
        raw[14] = 1
        raw[15], raw[16] = slice_number >> 8, slice_number & 255
        raw[19], raw[20] = view >> 8, view & 255
    struct.pack_into("<I", raw, 40, index)
    return bytes(raw)


def _frame(index: int, time: int, values: np.ndarray) -> bytes:
    coils, samples = values.shape
    data = np.stack([values.real, values.imag], axis=-1).astype("<i2").tobytes()
    size = 48 + len(data) + 6 * coils
    head = struct.pack("<12I", size, time, 0, 29, 0, 13, 0, 1, coils, samples, index, 0)
    return head + data + bytes(6 * coils)


def _archive(path, packets: list[bytes], frames: list[bytes], header: bytes) -> None:
    pool = base64.b64encode(header).decode()
    download = (
        f"<DownloadData><px><PoolHeaderBuffer><Size>{len(header)}</Size><Data>\n{pool}\n</Data>"
        "</PoolHeaderBuffer></px></DownloadData>"
    )
    with h5py.File(path, "w") as fh:
        fh.attrs["H5ArchiveVersion"] = np.array([1], dtype=np.uint32)
        fh.create_dataset("DownloadData.xml", data=np.array([download.encode()]))
        fh.create_dataset("Header.xml", data=np.array([HEADER_XML.encode()]))
        group = fh.create_group("Acquisition")
        for k, elements in enumerate((packets, frames)):
            stream = np.frombuffer(b"".join(elements) + bytes(8), np.uint8)
            group.create_dataset(f"S_Block{k}_Segment0", data=stream)
        update = struct.pack("<II", 8 + len(UPDATE_XML), 7) + UPDATE_XML.encode()
        group.create_dataset("S_Block2_Segment0", data=np.frombuffer(update + bytes(8), np.uint8))
        fh.create_group("ArchiveMetaData")
        fh.create_dataset("w/config/MRconfig.cfg", data=np.array([b"# config\n"]))
        fh.create_dataset("usr/g/bin/kernel.dat", data=np.arange(4, dtype=np.uint8))


def test_pfile_imaging(tmp_path):
    path = tmp_path / "P12345.7"
    frames = _pfile(path)
    sniff = miox.sniff(path)
    assert (sniff.format, sniff.kind) == (miox.Format.GE_PFILE, miox.Kind.RAW)
    image, baseline = miox.load(path)
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
    image, _ = miox.load(path)
    assert not image.headers["position"].any() and not image.headers["read_dir"].any()
    assert any("no slice geometry" in note for note in image.provenance.notes)


def test_spectroscopy_sequences_stay_with_spec2nii(tmp_path):
    path = tmp_path / "P00002.7"
    _pfile(path, psd=b"probe-p")
    assert GeRdbHeader.from_file(path).spectroscopy_mapper()
    path = tmp_path / "P00003.7"
    _pfile(path)
    assert not GeRdbHeader.from_file(path).spectroscopy_mapper()


def test_scanarchive_frames_follow_their_control_packets(tmp_path):
    pfile = tmp_path / "P12345.7"
    frames = _pfile(pfile)
    [reference, _] = miox.load(pfile)
    packets, elements = [], []
    order = [(s, v) for v in range(1, VIEWS + 1) for s in range(SLICES)]  # view-major order
    order += [(1, 257), (0, 0)]  # the view number's high byte; a baseline frame
    for i, (s, v) in enumerate(order):
        packets.append(_packet(1, i, 1000 + 10 * i, slice_number=s, view=v))
        values = frames[:, s, min(v, VIEWS)] if v else frames[:, s, 0]
        elements.append(_frame(i, 1005 + 10 * i, values))
    packets.append(_packet(3072, len(order), 2000))
    path = tmp_path / "ScanArchive_GRE.h5"
    _archive(path, packets, elements, _header())
    sniff = miox.sniff(path)
    expected = (miox.Format.GE_SCANARCHIVE, miox.Kind.RAW, "ge")
    assert (sniff.format, sniff.kind, sniff.vendor) == expected
    image, baseline = miox.load(path)
    assert image.name == "ScanArchive_GRE" and baseline.name == "ScanArchive_GRE_baseline"
    assert image.shape == (9, COILS, SAMPLES) and baseline.shape == (1, COILS, SAMPLES)
    assert image.counters("slice").tolist() == [1, 0] * VIEWS + [0]  # spatial index
    assert image.headers["idx"]["user"][:, 1].tolist() == [0, 1] * VIEWS + [1]
    assert image.counters("kspace_encode_step_1").tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 256]
    assert image.headers["acquisition_time_stamp"].tolist() == [1005 + 10 * i for i in range(9)]
    assert image.headers["scan_counter"].tolist() == list(range(9))
    assert image.headers["user_int"][:, 0].tolist() == [1] * 9
    assert image.flags_of(0) == ["FIRST_IN_SLICE"] and image.flags_of(7) == ["LAST_IN_SLICE"]
    for i, (s, v) in enumerate(order[:8]):
        j = s * VIEWS + v - 1
        assert np.array_equal(image.data[i], reference.data[j])
        assert image.headers["position"][i].tolist() == reference.headers["position"][j].tolist()
    assert np.array_equal(baseline.data[0], frames[:, 0, 0])
    m = image.meta
    assert m["AcquisitionDateTime"] == "2018-04-11T08:59:46.822405"
    assert m["SoftwareVersions"] == "DV26.0_R01_1725.a" and m["EchoTime"] == pytest.approx(0.0051)
    assert m["StudyID"] == "26356" and m["PatientPosition"] == "HFS"
    assert m.vendor["header"]["Header"]["px"]["ExamNumber"] == 26356
    assert m.vendor["rdb"]["rhi_psdname"] == "2dfast" and len(m.vendor["slice_table"]) == 2
    assert m.vendor["files"]["w/config/MRconfig.cfg"] == "# config\n"
    assert m.vendor["files"]["usr/g/bin/kernel.dat"] == bytes(range(4))
    assert m.vendor["updates"] == [UPDATE_XML] and len(m.vendor["control_packets"]) == 11
    assert m.vendor["frame_headers"][0][8:11] == [COILS, SAMPLES, 0]
    assert image.encoding["control_packets"] == 11 and image.encoding["frames"] == 9


def test_scanarchive_hyper_frame_packets(tmp_path):
    frames = _frames()
    packets, elements, index = [], [], 0
    for _ in range(2):
        for s in range(SLICES):
            packets.append(_packet(6, len(packets), 500, slice_number=s, frames=3))
            for v in range(3):
                elements.append(_frame(index, 600 + index, frames[:, s, v + 1]))
                index += 1
        packets.append(_packet(1024, len(packets), 700))
    packets.append(_packet(3072, len(packets), 800))
    path = tmp_path / "ScanArchive_EPI.h5"
    _archive(path, packets, elements, _header())
    [image] = miox.load(path)
    assert image.shape == (12, COILS, SAMPLES)
    assert image.counters("kspace_encode_step_1").tolist() == [0, 1, 2] * 4
    assert image.counters("slice").tolist() == [1, 1, 1, 0, 0, 0] * 2
    assert image.counters("repetition").tolist() == [0] * 6 + [1] * 6
    assert image.headers["user_int"][:, 0].tolist() == [6] * 12
    assert np.array_equal(image.data[4], frames[:, 1, 2])
    position = image.headers["position"][3].tolist()
    assert position == pytest.approx([-3.418, 23.888, 13.393], abs=1e-3)
    assert not image.provenance.notes


def test_scanarchive_frame_count_mismatch_is_noted(tmp_path):
    frames = _frames()
    packets = [_packet(1, 0, 1, view=1), _packet(1, 1, 2, view=2), _packet(3072, 2, 3)]
    elements = [_frame(i, i, frames[:, 0, 1]) for i in range(3)]
    path = tmp_path / "ScanArchive_X.h5"
    _archive(path, packets, elements, _header())
    [image] = miox.load(path)
    assert image.shape == (3, COILS, SAMPLES)
    assert any("name 2 frames, 3 stored" in note for note in image.provenance.notes)
    assert image.headers["user_int"][2].tolist() == [0] * 8  # no packet for the third frame
