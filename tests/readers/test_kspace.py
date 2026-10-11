####################################################################################################
#                                          test_kspace.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 05/10/26                                                                                #
#                                                                                                  #
# Purpose: ISMRMRD reader on synthetic files: the XML header to keys, acquisitions of unequal      #
#          length with trajectories, flags, waveforms, image groups and the fastMRI layout in      #
#          the sniffer. No test data needed.                                                       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import h5py
import numpy as np
import pytest

# own
import miox
from miox.readers.ismrmrd import IsmrmrdReader
from miox.readers.siemens_twix_raw import SiemensTwixRaw

HEADER = IsmrmrdReader.ACQUISITION_FLAGS
XML = """<?xml version="1.0"?>
<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD">
  <subjectInformation><patientName>phantom</patientName><patientWeight_kg>70</patientWeight_kg>
    <patientGender>O</patientGender></subjectInformation>
  <measurementInformation><measurementID>42</measurementID><seriesDate>2026-10-05</seriesDate>
    <seriesTime>10:11:12</seriesTime><patientPosition>HFS</patientPosition>
    <protocolName>gre</protocolName><initialSeriesNumber>7</initialSeriesNumber>
    <relativeTablePosition><x>0</x><y>0</y><z>-1.5</z></relativeTablePosition>
  </measurementInformation>
  <acquisitionSystemInformation><systemVendor>ACME</systemVendor><systemModel>One</systemModel>
    <receiverChannels>2</receiverChannels>
    <coilLabel><coilNumber>0</coilNumber><coilName>A</coilName></coilLabel>
    <coilLabel><coilNumber>1</coilNumber><coilName>B</coilName></coilLabel>
  </acquisitionSystemInformation>
  <experimentalConditions><H1resonanceFrequency_Hz>123200000</H1resonanceFrequency_Hz>
  </experimentalConditions>
  <encoding>
    <encodedSpace><matrixSize><x>8</x><y>4</y><z>1</z></matrixSize>
      <fieldOfView_mm><x>200</x><y>100</y><z>5</z></fieldOfView_mm></encodedSpace>
    <reconSpace><matrixSize><x>4</x><y>4</y><z>1</z></matrixSize>
      <fieldOfView_mm><x>100</x><y>100</y><z>5</z></fieldOfView_mm></reconSpace>
    <encodingLimits><kspace_encoding_step_1><minimum>0</minimum><maximum>3</maximum>
      <center>2</center></kspace_encoding_step_1></encodingLimits>
    <trajectory>radial</trajectory>
    <parallelImaging><accelerationFactor><kspace_encoding_step_1>2</kspace_encoding_step_1>
      <kspace_encoding_step_2>1</kspace_encoding_step_2></accelerationFactor></parallelImaging>
    <echoTrainLength>3</echoTrainLength>
  </encoding>
  <sequenceParameters><TR>500</TR><TE>2.5</TE><TE>5</TE><flipAngle_deg>12</flipAngle_deg>
    <sequence_type>GRE</sequence_type><echo_spacing>0.8</echo_spacing>
    <diffusion><gradientDirection><rl>1</rl><ap>0</ap><fh>0</fh></gradientDirection>
      <bvalue>1000</bvalue></diffusion>
  </sequenceParameters>
  <userParameters><userParameterLong><name>k</name><value>3</value></userParameterLong>
  </userParameters>
</ismrmrdHeader>
"""


def _acquisition_dtype():
    return np.dtype(
        [
            ("head", SiemensTwixRaw.HEADER),
            ("traj", h5py.special_dtype(vlen=np.dtype("float32"))),
            ("data", h5py.special_dtype(vlen=np.dtype("float32"))),
        ]
    )


def _write_raw(path, readouts, two_dimensional=False):
    """``readouts``: (samples, channels, trajectory dims, flags, line) per readout."""
    table = np.zeros(len(readouts), dtype=_acquisition_dtype())
    for i, (samples, channels, dims, flags, line) in enumerate(readouts):
        head = table["head"][i]
        head["version"], head["flags"] = 1, flags
        head["number_of_samples"], head["active_channels"] = samples, channels
        head["available_channels"], head["trajectory_dimensions"] = 2, dims
        head["sample_time_us"], head["scan_counter"] = 2.5, i
        head["idx"]["kspace_encode_step_1"] = line
        head["idx"]["user"][5] = 64
        data = np.arange(channels * samples, dtype=np.float32) + 100 * i
        table["data"][i] = np.stack([data, -data], axis=-1).ravel()  # interleaved re, im
        table["traj"][i] = np.arange(samples * dims, dtype=np.float32) / 10
    with h5py.File(path, "w") as fh:
        group = fh.create_group("dataset")
        group.create_dataset("xml", data=[XML.encode()], dtype=h5py.special_dtype(vlen=str))
        group.create_dataset("data", data=table.reshape(-1, 1) if two_dimensional else table)
        return group


def test_header_keys_and_acquisitions(tmp_path):
    path = tmp_path / "raw.h5"
    _write_raw(path, [(8, 2, 0, 1 << (19 - 1), 0), (8, 2, 2, 0, 1), (6, 1, 2, 1 << (2 - 1), 2)])
    sniff = miox.sniff(path)
    assert (sniff.format, sniff.kind) == (miox.Format.ISMRMRD, miox.Kind.RAW)
    [raw] = miox.load(path)
    assert isinstance(raw, miox.RawData)
    assert raw.shape == (3, 2, 8) and raw.samples.tolist() == [8, 8, 6]
    assert raw.data[1, 1, 3] == complex(111, -111)  # readout 1, channel 1, sample 3
    assert raw.data[2, 1, 0] == 0 and raw.data[2, 0, 6] == 0  # padded
    assert raw.trajectory.shape == (3, 8, 2)
    assert raw.trajectory[1, 1].tolist() == pytest.approx([0.2, 0.3])
    assert raw.trajectory[0].tolist() == [[0, 0]] * 8
    assert raw.flags_of(0) == ["IS_NOISE_MEASUREMENT"]
    assert raw.flagged("LAST_IN_ENCODE_STEP1").tolist() == [False, False, True]
    assert raw.counters("kspace_encode_step_1").tolist() == [0, 1, 2]
    meta = raw.meta
    assert meta["PatientName"] == "phantom" and meta["PatientWeight"] == 70
    assert meta["PatientSex"] == "O" and meta["PatientPosition"] == "HFS"
    assert meta["AcquisitionDateTime"] == "2026-10-05T10:11:12"
    assert meta["SeriesNumber"] == 7 and meta["TablePosition"] == [0, 0, -1.5]
    assert meta["NumberOfReceiveChannels"] == 2
    assert meta["H1ResonanceFrequency"] == pytest.approx(123.2)
    assert meta["FieldStrength"] == pytest.approx(123.2 / 42.577478518)
    assert meta["RepetitionTime"] == 0.5 and meta["EchoTime"] == [0.0025, 0.005]
    assert meta["ExcitationFlipAngle"] == 12 and meta["PulseSequenceType"] == "GRE"
    assert meta["VendorReportedEchoSpacing"] == pytest.approx(0.0008)
    assert meta["DiffusionBValue"] == [1000.0]
    assert meta["EncodedMatrixSize"] == [8, 4, 1] and meta["EncodedFieldOfView"] == [200, 100, 5]
    assert meta["ReconMatrixSize"] == [4, 4, 1]
    assert meta["KSpaceTrajectory"] == "radial" and meta["EchoTrainLength"] == 3
    assert meta["ParallelReductionFactorInPlane"] == 2
    assert meta["EncodingLimits"]["kspace_encoding_step_1"] == {
        "minimum": 0, "maximum": 3, "center": 2
    }
    assert meta["DwellTime"] == pytest.approx(2.5e-6)
    assert "NumberOfPoints" not in meta  # readouts differ in length
    assert meta.vendor["acquisitionSystemInformation"]["coilLabel"][1]["coilName"] == "B"
    assert meta.vendor["userParameters"]["userParameterLong"]["value"] == 3
    assert meta.fields["EchoTime"] == "sequenceParameters/TE"
    assert raw.encoding[0]["trajectory"] == "radial"
    assert any("gradient directions" in note for note in raw.provenance.notes)
    assert any("2 lengths" in note for note in raw.provenance.notes)


def test_two_dimensional_table_and_waveforms(tmp_path):
    path = tmp_path / "old.h5"
    with h5py.File(path, "w") as fh:
        pass
    group = _write_raw(path, [(4, 1, 0, 0, 0), (4, 1, 0, 0, 1)], two_dimensional=True)
    waveform_dtype = np.dtype(
        [
            ("head", np.dtype([
                ("version", "<u2"), ("flags", "<u8"), ("measurement_uid", "<u4"),
                ("scan_counter", "<u4"), ("time_stamp", "<u4"), ("number_of_samples", "<u2"),
                ("channels", "<u2"), ("sample_time_us", "<f4"), ("waveform_id", "<u2"),
            ], align=True)),
            ("data", h5py.special_dtype(vlen=np.dtype("uint32"))),
        ]
    )
    table = np.zeros(1, dtype=waveform_dtype)
    table["head"][0]["number_of_samples"], table["head"][0]["channels"] = 3, 2
    table["head"][0]["waveform_id"], table["head"][0]["sample_time_us"] = 1, 400
    table["data"][0] = np.arange(6, dtype=np.uint32)
    with h5py.File(path, "a") as fh:
        fh["dataset"].create_dataset("waveforms", data=table)
    del group
    [raw] = miox.load(path)
    assert raw.shape == (2, 1, 4) and raw.meta["NumberOfPoints"] == 4
    assert len(raw.waveforms) == 1
    assert raw.waveforms[0]["waveform_id"] == 1
    assert raw.waveforms[0]["data"].tolist() == [[0, 1, 2], [3, 4, 5]]


def _image_dtype():
    return np.dtype(
        [
            ("version", "<u2"), ("data_type", "<u2"), ("flags", "<u8"), ("measurement_uid", "<u4"),
            ("matrix_size", "<u2", (3,)), ("field_of_view", "<f4", (3,)), ("channels", "<u2"),
            ("position", "<f4", (3,)), ("read_dir", "<f4", (3,)), ("phase_dir", "<f4", (3,)),
            ("slice_dir", "<f4", (3,)), ("patient_table_position", "<f4", (3,)),
            ("average", "<u2"), ("slice", "<u2"), ("contrast", "<u2"), ("phase", "<u2"),
            ("repetition", "<u2"), ("set", "<u2"), ("acquisition_time_stamp", "<u4"),
            ("physiology_time_stamp", "<u4", (3,)), ("image_type", "<u2"), ("image_index", "<u2"),
            ("image_series_index", "<u2"), ("user_int", "<i4", (8,)), ("user_float", "<f4", (8,)),
            ("attribute_string_len", "<u4"),
        ]
    )


def test_image_groups_without_xml(tmp_path):
    path = tmp_path / "images.mrd"
    heads = np.zeros(2, dtype=_image_dtype())
    for i, head in enumerate(heads):
        head["matrix_size"], head["field_of_view"], head["channels"] = (4, 2, 1), (40, 20, 5), 1
        head["position"] = (10, 20, 30)  # LPS, the image centre
        head["read_dir"], head["phase_dir"], head["slice_dir"] = (1, 0, 0), (0, 1, 0), (0, 0, 1)
        head["image_type"], head["image_index"], head["image_series_index"] = 1, i + 1, 3
        head["data_type"] = 5
    data = np.arange(2 * 1 * 1 * 2 * 4, dtype=np.float32).reshape(2, 1, 1, 2, 4)  # n, ch, z, y, x
    with h5py.File(path, "w") as fh:
        group = fh.create_group("recon").create_group("image_0")
        group.create_dataset("header", data=heads)
        group.create_dataset("data", data=data)
        group.create_dataset(
            "attributes", data=[b"<meta/>", b"<meta/>"], dtype=h5py.special_dtype(vlen=str)
        )
    sniff = miox.sniff(path)
    assert (sniff.format, sniff.kind) == (miox.Format.ISMRMRD, miox.Kind.IMAGING)
    [image] = miox.load(path)
    assert isinstance(image, miox.MRImage)
    assert image.shape == (4, 2, 1, 2) and image.dims == ("i", "j", "k", "image")
    assert image.data[3, 1, 0, 1] == 15  # x=3, y=1 of the second image
    assert image.voxel_sizes == pytest.approx((10, 10, 5))
    # centre (10, 20, 30) LPS = (-10, -20, 30) RAS; voxel (0, 0, 0) is 1.5 voxels left, 0.5 up
    assert image.affine[:3, 3].tolist() == pytest.approx([5, -15, 30])
    assert image.affine[0, 0] == -10 and image.affine[1, 1] == -10 and image.affine[2, 2] == 5
    assert image.meta["ImageType"] == ["MAGNITUDE"] and image.meta["SeriesNumber"] == 3
    assert image.meta.vendor["image_headers"]["image_index"] == [1, 2]
    assert image.meta.vendor["image_attributes"] == ["<meta/>", "<meta/>"]


def test_fastmri_layout(tmp_path):
    path = tmp_path / "file.h5"
    kspace = np.ones((2, 3, 4, 5), dtype=np.complex64)  # slices, coils, frequency, phase
    kspace[1, 2, :, 3] = np.arange(4) + 1j  # slice 1, coil 2, line 3
    kspace[:, :, :, 0] = 0  # line 0 of both slices not acquired
    with h5py.File(path, "w") as fh:
        fh.create_dataset("kspace", data=kspace)
        fh.create_dataset("ismrmrd_header", data=XML.encode())
        fh.create_dataset("reconstruction_rss", data=np.ones((2, 4, 5), dtype=np.float32))
        fh.attrs["acquisition"] = "AXT2"
    sniff = miox.sniff(path)
    assert (sniff.format, sniff.kind) == (miox.Format.FASTMRI_H5, miox.Kind.RAW)
    raw, image = miox.load(path)
    assert isinstance(raw, miox.RawData) and raw.shape == (10, 3, 4)
    assert raw.counters("slice").tolist() == [0] * 5 + [1] * 5
    assert raw.counters("kspace_encode_step_1").tolist() == [0, 1, 2, 3, 4] * 2
    assert raw.data[8, 2].tolist() == [0 + 1j, 1 + 1j, 2 + 1j, 3 + 1j]
    assert raw.meta["EncodedMatrixSize"] == [4, 5, 2] and raw.meta["NumberOfReceiveChannels"] == 3
    assert raw.meta["RepetitionTime"] == 0.5 and raw.meta["PatientPosition"] == "HFS"
    assert raw.meta.vendor["file_attributes"] == {"acquisition": "AXT2"}
    assert any("2 of 10" in note for note in raw.provenance.notes)
    assert isinstance(image, miox.MRImage) and image.shape == (5, 4, 2)
    assert image.meta["ImageType"] == ["RSS"]


def test_other_hdf5_is_unknown(tmp_path):
    path = tmp_path / "file.h5"
    with h5py.File(path, "w") as fh:
        fh.create_dataset("x", data=np.zeros(3))
    sniff = miox.sniff(path)
    assert sniff.format == miox.Format.UNKNOWN and sniff.detail["names"] == ["x"]


def test_twix_directions_follow_ismrmrd():
    """As siemens_to_ismrmrd maps them: the identity quaternion (w, x, y, z) = (1, 0, 0, 0)
    puts the phase direction along x and the read direction along y; a 90 degree rotation
    about z gives read along -x and phase along y (checked against a converted file)."""
    phase, read, slab = SiemensTwixRaw.directions(0, 0, 0, 1)
    assert (phase, read, slab) == ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    half = np.sqrt(0.5)
    phase, read, slab = SiemensTwixRaw.directions(0, 0, half, half)
    assert np.allclose(read, (-1, 0, 0)) and np.allclose(phase, (0, 1, 0))
    assert np.allclose(slab, (0, 0, 1))
