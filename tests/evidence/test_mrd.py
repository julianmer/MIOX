####################################################################################################
#                                           test_mrd.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/10/26                                                                                #
#                                                                                                  #
# Purpose: MRD (version 2) streams written by the MRD project's own tools against independent      #
#          evidence: ISMRMRD files converted by ismrmrd_to_mrd against the same files read as      #
#          ISMRMRD (readouts, every header field, trajectories, keys, images and their geometry),  #
#          a Pulseq file converted by seq_to_mrd against the .seq file, the phantom's k-space      #
#          against the reconstruction of stream_recon, the NDJSON copies against the binary        #
#          streams, and a noise covariance written by MRD's writer against the noise readouts.     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import numpy as np
import pytest

# own
from miox import RawData, load
from tests.conftest import PUBLIC_DATA

MRD = PUBLIC_DATA / "mrd"
GADGETRON = PUBLIC_DATA / "gadgetron"
CONVERTED_RAW = {  # MRD stream: the ISMRMRD file it was converted from
    "generic_gre.mrd": "generic/meas_MID01095_FID40115/meas_MID01095_FID40115.mrd",
    "simple_gre.mrd": "simple_gre/simple_gre_in_20220831.mrd",
    "kidney_spiral.mrd": "matlab/Kidney4cmSlice.mrd",
}
CONVERTED_IMAGES = {
    "simple_gre_images.mrd": "simple_gre/simple_gre_out_20210909_klk.mrd",
    "epi_images.mrd": "epi/epi_2d_out_20161020_pjv.mrd",
    "radial_images.mrd": "radial_phantom/fixed_radial_mode1_20210923_klk.mrd",
    "hyper_images.mrd": "hyper/Kidney4cmSlice_sb_outV4.mrd",
}
CONVERTER_LOSSES = {  # header fields ismrmrd_to_mrd does not carry over
    ("idx", "user"),  # the encoding counters' user values are not copied
    ("available_channels", None),  # MRD has no such field; the active channel count is kept
}
KEYS_NOT_CONVERTED = {"TablePosition"}  # relativeTablePosition is not converted


def _need(*paths):
    for path in paths:
        if not path.is_file():
            pytest.skip(f"{path.relative_to(PUBLIC_DATA)} not fetched")


def _header_fields(headers: np.ndarray):
    for name in headers.dtype.names:
        subfields = headers.dtype[name].names
        for sub in subfields or (None,):
            yield (name, sub), headers[name] if sub is None else headers[name][sub]


@pytest.mark.parametrize("stream, source", CONVERTED_RAW.items())
def test_converted_acquisitions_equal_the_ismrmrd_file(stream, source):
    _need(MRD / stream, GADGETRON / source)
    [ismrmrd] = [d for d in load(GADGETRON / source) if isinstance(d, RawData)]
    [mrd] = load(MRD / stream)
    assert np.array_equal(mrd.data, ismrmrd.data) and np.array_equal(mrd.samples, ismrmrd.samples)
    theirs = dict(_header_fields(ismrmrd.headers))
    for field, ours in _header_fields(mrd.headers):
        if field in CONVERTER_LOSSES:
            continue
        if field == ("sample_time_us", None):  # the converter truncates to whole ns
            assert np.all(np.abs(ours - theirs[field]) < 1e-3), field
            continue
        assert np.array_equal(ours, theirs[field]), field
    if ismrmrd.trajectory is None:
        assert mrd.trajectory is None
    else:  # ismrmrd_to_mrd stores trajectories (samples, basis); the reader transposes them
        assert np.array_equal(mrd.trajectory, ismrmrd.trajectory)
        assert any("transposed" in note for note in mrd.provenance.notes)
    for key in (set(mrd.meta) | set(ismrmrd.meta)) - KEYS_NOT_CONVERTED:
        ours, theirs_value = mrd.meta.get(key), ismrmrd.meta.get(key)
        if isinstance(theirs_value, float):  # float32 in MRD, decimal text in the ISMRMRD XML
            assert ours == pytest.approx(theirs_value, rel=1e-6), key
        else:
            assert ours == theirs_value, key


@pytest.mark.parametrize("stream, source", CONVERTED_IMAGES.items())
def test_converted_images_equal_the_ismrmrd_file(stream, source):
    _need(MRD / stream, GADGETRON / source)
    ismrmrd, mrd = load(GADGETRON / source), load(MRD / stream)
    assert len(mrd) == len(ismrmrd)
    for theirs in ismrmrd:  # the image groups' names are not in the stream; match by content
        [ours] = [m for m in mrd if m.shape == theirs.shape and np.array_equal(m.data, theirs.data)]
        assert ours.dims == theirs.dims
        assert np.allclose(ours.affine, theirs.affine, atol=1e-6)
        for key in ("ImageType", "SeriesNumber"):
            assert ours.meta.get(key) == theirs.meta.get(key), key


def test_pulseq_stream_equals_the_seq_file():
    _need(MRD / "haste.mrd", MRD / "haste.seq")
    [seq], [stream] = load(MRD / "haste.seq"), load(MRD / "haste.mrd")
    assert np.array_equal(stream.blocks, seq.blocks) and stream.duration == seq.duration
    assert stream.definitions == seq.definitions
    for table, events in seq.events.items():  # 1.4 columns; MRD adds those of 1.5
        assert events.keys() == stream.events[table].keys()
        for i, event in events.items():
            assert {c: stream.events[table][i][c] for c in event} == event, (table, i)
    assert stream.shapes.keys() == seq.shapes.keys()
    for i, shape in seq.shapes.items():
        assert np.allclose(stream.shapes[i], shape), i


def test_phantom_kspace_reconstructs_to_the_tools_image():
    _need(MRD / "phantom.mrd", MRD / "phantom_recon.mrd")
    [raw], [image] = load(MRD / "phantom.mrd"), load(MRD / "phantom_recon.mrd")
    assert raw.meta["EncodedMatrixSize"] == [64, 32, 1]
    assert raw.meta["ReconMatrixSize"] == [32, 32, 1]
    assert raw.trajectory.shape == (96, 64, 2) and raw.flagged("IS_NOISE_MEASUREMENT").sum() == 32
    data = ~raw.flagged("IS_NOISE_MEASUREMENT")
    lines, repetitions = raw.counters("kspace_encode_step_1"), raw.counters("repetition")
    for repetition in (0, 1):
        chosen = data & (repetitions == repetition)
        kspace = np.zeros((raw.n_coils, 32, 64), dtype=np.complex64)
        kspace[:, lines[chosen], :] = np.transpose(raw.data[chosen], (1, 0, 2))
        coils = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace, axes=(1, 2))), axes=(1, 2))
        ours = np.sqrt((np.abs(coils) ** 2).sum(axis=0))[:, 16:48]  # readout oversampling cut
        theirs = image.data[:, :, 0, repetition].T  # (x, y) to (y, x)
        scale = np.sqrt(64 * 32)  # stream_recon's orthonormal FFT
        assert np.allclose(theirs, scale * ours, atol=1e-6 * theirs.max())


@pytest.mark.parametrize("stem", ["phantom", "phantom_recon", "haste"])
def test_ndjson_copies_equal_the_binary_streams(stem):
    _need(MRD / f"{stem}.mrd", MRD / f"{stem}.ndjson")
    binary, text = load(MRD / f"{stem}.mrd"), load(MRD / f"{stem}.ndjson")
    assert len(binary) == len(text)
    for a, b in zip(binary, text, strict=True):
        assert dict(a.meta) == dict(b.meta) and a.name == b.name
        if isinstance(a, RawData):
            assert np.array_equal(a.data, b.data) and a.headers.tobytes() == b.headers.tobytes()
            assert np.array_equal(a.trajectory, b.trajectory)
        elif hasattr(a, "affine"):
            assert np.array_equal(a.data, b.data) and np.array_equal(a.affine, b.affine)
        else:
            assert np.array_equal(a.blocks, b.blocks) and a.events == b.events


def test_noise_covariance_equals_the_streams_noise_readouts():
    """MRD's own writer stored the covariance of the stream's noise readout; it equals the one
    computed from those readouts as read here, with the stream's bandwidth and noise dwell time."""
    _need(MRD / "simple_gre.mrd", MRD / "simple_gre_noise_covariance.mrd")
    [raw] = load(MRD / "simple_gre.mrd")
    [covariance] = load(MRD / "simple_gre_noise_covariance.mrd")
    noise = raw.data[raw.flagged("IS_NOISE_MEASUREMENT")]  # (readouts, coils, samples)
    samples = np.concatenate(list(noise), axis=1)
    expected = samples @ samples.conj().T / samples.shape[1]
    assert covariance.matrix.shape == (32, 32) and covariance.sample_count == samples.shape[1]
    assert np.allclose(covariance.matrix, expected, rtol=1e-5, atol=1e-6 * np.abs(expected).max())
    stated = raw.meta.vendor["acquisitionSystemInformation"]["relativeReceiverNoiseBandwidth"]
    assert covariance.bandwidth == stated
    first = int(np.argmax(raw.flagged("IS_NOISE_MEASUREMENT")))
    assert covariance.dwell_time == raw.meta.vendor["acquisitions"]["sample_time_ns"][first] / 1e9
