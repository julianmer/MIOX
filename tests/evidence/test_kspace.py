####################################################################################################
#                                          test_kspace.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 05/10/26                                                                                #
#                                                                                                  #
# Purpose: Raw k-space against independent evidence: a twix imaging scan against its ISMRMRD       #
#          conversion by siemens_to_ismrmrd (readouts and every header field), the ISMRMRD         #
#          project's 2013 sample against the Gadgetron store's 2022 conversion of the same scan,   #
#          the public ISMRMRD files of UIH, OSI2ONE and Gadgetron reconstructions, and GE: the     #
#          P-files of ge_to_ismrmrd against the ScanArchives of the same scans frame for frame,    #
#          their slice geometry against the image header, and a spectroscopy P-file read through   #
#          the imaging frame layout against spec2nii, and the Philips raw export against the       #
#          published reading of Philips' reader and its random phase as a ramp along the readout;  #
#          a Philips Dixon scan exported as .data/.list against the scanner's DICOM of it; the    #
#          PMU waveforms of two twix scans against the waveform counts siemens_to_ismrmrd's test   #
#          cases state for their conversions; Quick Look's plain reconstruction against            #
#          Gadgetron's images and M4Raw's stored rss.                                              #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import numpy as np
import pydicom
import pytest

# own
from miox import MRImage, RawData, load
from miox.readers.ge_pfile import GePfile
from miox.readers.ge_pfile_raw import GePfileRaw
from miox.readers.ge_rdb import GeRdbHeader
from tests.conftest import PUBLIC_DATA

GADGETRON = PUBLIC_DATA / "gadgetron"
ISMRMRD = PUBLIC_DATA / "ismrmrd"
GE = PUBLIC_DATA / "ge_to_ismrmrd"
PHILIPS = PUBLIC_DATA / "labrawsin/20250617_210621_COR_GRE.lab"
GE_MRS = PUBLIC_DATA / "spec2nii_test_data/ge/pFiles/svMRS/P03072.7"
TWIX_GRE = GADGETRON / "simple_gre/meas_MiniGadgetron_GRE.dat"
ISMRMRD_GRE = GADGETRON / "simple_gre/simple_gre_in_20220831.mrd"
ISMRMRD_GRE_2013 = ISMRMRD / "simple_gre.h5"
SAME_HEADER_FIELDS = (  # equal between twix and conversion; the directions to float32 rounding
    "scan_counter", "acquisition_time_stamp", "physiology_time_stamp", "number_of_samples",
    "active_channels", "discard_pre", "discard_post", "center_sample", "sample_time_us",
    "position", "slice_dir", "user_int", "user_float", "measurement_uid",
)
TWIX_FLAGS = {  # Siemens evaluation-mask flag: the ISMRMRD flag siemens_to_ismrmrd sets for it
    "NOISEADJSCAN": "IS_NOISE_MEASUREMENT",
    "FIRSTSCANINSLICE": "FIRST_IN_SLICE",
    "LASTSCANINSLICE": "LAST_IN_SLICE",
    "PATREFSCAN": "IS_PARALLEL_CALIBRATION",
    "PATREFANDIMASCAN": "IS_PARALLEL_CALIBRATION_AND_IMAGING",
}


def _need(*paths):
    for path in paths:
        if not path.is_file():
            pytest.skip(f"{path.relative_to(PUBLIC_DATA)} not fetched")


def _converted_without_noise(path):
    [raw] = load(path)
    noise = raw.flagged("IS_NOISE_MEASUREMENT")
    return raw, noise


def test_twix_readouts_equal_the_ismrmrd_conversion():
    _need(TWIX_GRE, ISMRMRD_GRE)
    image, noise = load(TWIX_GRE)
    converted, is_noise = _converted_without_noise(ISMRMRD_GRE)
    assert image.name.endswith("GRE") and noise.name.endswith("_noise")
    assert np.array_equal(image.data, converted.data[~is_noise])
    assert np.array_equal(noise.data, converted.data[is_noise])
    ours, theirs = image.headers, converted.headers[~is_noise]
    for field in SAME_HEADER_FIELDS:
        assert np.array_equal(ours[field], theirs[field]), field
    for field in ("read_dir", "phase_dir"):
        assert np.allclose(ours[field], theirs[field], atol=1e-6), field
    for counter in ours["idx"].dtype.names:
        assert np.array_equal(ours["idx"][counter], theirs["idx"][counter]), counter
    for twix_flag, ismrmrd_flag in TWIX_FLAGS.items():
        assert np.array_equal(image.flagged(twix_flag), converted.flagged(ismrmrd_flag)[~is_noise])
    assert noise.flagged("NOISEADJSCAN").all()
    for key in ("EchoTime", "RepetitionTime", "Manufacturer", "ManufacturersModelName",
                "FieldStrength", "InstitutionName", "PatientPosition", "NumberOfPoints",
                "NumberOfReceiveChannels", "EncodedMatrixSize"):
        assert image.meta[key] == pytest.approx(converted.meta[key]), key
    frequency = converted.meta["H1ResonanceFrequency"]
    assert image.meta["SpectrometerFrequency"] == pytest.approx(frequency)
    assert image.meta["DwellTime"] == pytest.approx(converted.meta["DwellTime"], rel=1e-2)


@pytest.mark.parametrize("relpath, waveforms, acquisitions, channels", [
    # siemens_to_ismrmrd test/cases/*.yaml: num_waveforms and num_acquisitions of the conversion
    ("cmr/T1SRmapping/meas_MID00613_FID40909_pre_SASHA_10pt_TS600_256/"
     "meas_MID00613_FID40909_pre_SASHA_10pt_TS600_256.dat", 1115, 719, 14),
    ("tse/meas_MID00450_FID76726_SAX_TE62_DIR_TSE/meas_MID00450_FID76726_SAX_TE62_DIR_TSE.dat",
     1090, 136, 30),
])
def test_twix_pmu_waveforms_match_the_converters_counts(relpath, waveforms, acquisitions,
                                                        channels):
    """siemens_to_ismrmrd writes five waveforms per PMU packet (ECG with its leads, PULS, RESP,
    EXT1, EXT2); its test cases state how many each scan yields. The ECG waveform carries four
    leads and their combined trigger, the others one signal and its trigger; sample times follow
    the stated periods (2.5, 5 and 20 ms) and the learning phase gets ids 5-9."""
    path = GADGETRON / relpath
    if not path.is_file():
        pytest.skip(f"{relpath} not fetched")
    datasets = load(path)
    raw = datasets[0]
    assert raw.n_coils == channels
    counters = np.concatenate([d.headers["scan_counter"] for d in datasets])
    assert len(set(counters.tolist())) == acquisitions  # lines in two streams counted once
    assert len(raw.waveforms) == waveforms and all(not d.waveforms for d in datasets[1:])
    by_name = {}
    for w in raw.waveforms:
        by_name.setdefault(w["name"], []).append(w)
    assert sorted(by_name) == ["ECG", "EXT1", "EXT2", "PULS", "RESP"]
    assert {len(v) for v in by_name.values()} == {waveforms // 5}
    full = [w for w in by_name["ECG"] if w["number_of_samples"] == 40]
    assert full and all(w["channels"] == 5 and w["sample_time_us"] == 2500 for w in full)
    assert all(w["data"].shape == (w["channels"], w["number_of_samples"]) for w in raw.waveforms)
    assert all(w["channels"] == 2 for n in ("PULS", "RESP") for w in by_name[n])
    assert {w["sample_time_us"] for w in by_name["PULS"] if w["number_of_samples"] == 20} == {5000}
    assert {w["sample_time_us"] for w in by_name["RESP"] if w["number_of_samples"] == 5} == {20000}
    ids = {(w["name"], w["phase"]): w["waveform_id"] for w in raw.waveforms}
    assert ids[("ECG", "Learning")] == 5 and ids[("ECG", "Acquisition")] == 0
    assert ids[("EXT2", "Acquisition")] == 4 and ids[("RESP", "Learning")] == 7
    stamps = [w["time_stamp"] for w in raw.waveforms]
    assert stamps == sorted(stamps)
    assert all(len(np.unique(w["data"][-1])) <= 8 for w in raw.waveforms)  # trigger flag bits


def test_2013_sample_equals_the_2022_conversion():
    _need(ISMRMRD_GRE_2013, ISMRMRD_GRE)
    [old] = load(ISMRMRD_GRE_2013)
    [new] = load(ISMRMRD_GRE)
    assert np.array_equal(old.data, new.data)
    for counter in old.headers["idx"].dtype.names:
        if counter != "user":  # the 2013 converter left the k-space centre line out
            assert np.array_equal(old.headers["idx"][counter], new.headers["idx"][counter])
    stamps = "acquisition_time_stamp"
    assert np.array_equal(old.headers[stamps], new.headers[stamps])
    assert old.meta["EncodedMatrixSize"] == new.meta["EncodedMatrixSize"] == [256, 128, 1]
    assert old.meta["KSpaceTrajectory"] == "cartesian"


def test_spiral_sample_has_a_trajectory():
    path = ISMRMRD / "simple_spiral.h5"
    _need(path)
    [raw] = load(path)
    assert raw.meta["KSpaceTrajectory"] == "spiral"
    assert raw.trajectory.shape == (raw.n_readouts, raw.data.shape[-1], 2)
    assert np.abs(raw.trajectory).max() <= 0.5  # k-space in cycles per field of view


def test_gadgetron_reconstruction_images():
    path = GADGETRON / "simple_gre/ref_20240810.mrd"
    _need(path, ISMRMRD_GRE)
    [image] = load(path)
    [raw] = load(ISMRMRD_GRE)
    assert isinstance(image, MRImage) and image.dims == ("i", "j", "k", "image")
    assert image.shape == (128, 128, 1, 10)  # the readout oversampling removed, 10 repetitions
    assert image.shape[-1] == raw.counters("repetition").max() + 1
    assert image.meta["ImageType"] == ["MAGNITUDE"]
    fov = raw.meta["ReconFieldOfView"]
    assert image.voxel_sizes == pytest.approx((fov[0] / 128, fov[1] / 128, fov[2]))


@pytest.mark.parametrize("relpath", [
    "gadgetron/simple_gre/meas_MiniGadgetron_GRE.dat",
    "gadgetron/simple_gre/simple_gre_in_20220831.mrd",
    "mrd/simple_gre.mrd",
], ids=["twix", "ismrmrd", "mrd"])
def test_plain_reconstruction_of_the_gre_equals_gadgetrons_image(relpath):
    """The Quick Look page's plain reconstruction (echo centred, 2-D FFT, root sum of squares,
    readout cropped to the matrix) of the fully sampled GRE, in its twix, ISMRMRD and MRD
    forms, is Gadgetron's reconstruction of it (the uint16 series of simple_gre_images.mrd;
    correlation above 0.999), stored with the readout along the first axis."""
    from miox.quicklook_pages import RawPage

    images = PUBLIC_DATA / "mrd/simple_gre_images.mrd"
    _need(PUBLIC_DATA / relpath, images)
    raw = next(d for d in load(PUBLIC_DATA / relpath) if isinstance(d, RawData))  # image stream
    reference = next(d for d in load(images) if d.name == "imageUint16/series_1")
    theirs = np.asarray(reference.data, dtype=np.float64)[:, :, 0, 0]
    ours, cropped = RawPage.plain_image(raw)
    assert cropped == "readout cropped to 128" and ours.shape == (128, 128)
    assert np.corrcoef(ours.T.ravel(), theirs.ravel())[0, 1] > 0.999


def test_plain_reconstruction_equals_the_m4raw_stored_rss():
    """M4Raw stores its root-sum-of-squares reconstruction beside the k-space: the page's
    plain reconstruction of the first slice equals it up to the FFT's scale."""
    from miox.quicklook_pages import RawPage

    path = PUBLIC_DATA / "m4raw/multicoil_test/2022101103_T206.h5"
    _need(path)
    datasets = load(path)
    [raw] = [d for d in datasets if isinstance(d, RawData)]
    [stored] = [d for d in datasets if isinstance(d, MRImage)]
    ours, cropped = RawPage.plain_image(raw)
    theirs = np.asarray(stored.data, dtype=np.float64)[:, :, 0]
    assert cropped == "as stored" and ours.shape == theirs.shape
    scale = theirs.max() / ours.max()
    assert np.allclose(ours * scale, theirs, rtol=1e-4, atol=1e-4 * theirs.max())


@pytest.mark.parametrize(
    "relpath",
    [
        "spec2nii_test_data/UIH/mrs_data/UID_191716063826303_svs_press_te144.h5",
        "spec2nii_test_data/UIH/mrs_3d/UID_191716063827984_csi_hise_3d_te144.h5",
    ],
)
def test_uih_spectroscopy_raw(relpath):
    path = PUBLIC_DATA / relpath
    _need(path)
    [raw] = load(path)
    assert isinstance(raw, RawData)
    assert raw.meta["Manufacturer"] == "UIH" and raw.meta["ProtocolName"] in relpath
    assert raw.meta["DwellTime"] == pytest.approx(raw.headers["sample_time_us"][0] * 1e-6)
    assert raw.meta.vendor["userParameters"]["userParameterDouble"][0]["name"] == "PosdTra"


def test_osi2one_raw_with_trajectory():
    folders = sorted((PUBLIC_DATA / "osi2one").glob("**/ismrmrd_data.h5"))
    if not folders:
        pytest.skip("osi2one not fetched")
    [raw] = load(folders[0])
    assert raw.meta["Manufacturer"] == "osi2" and raw.meta["FieldStrength"] < 0.1
    assert raw.trajectory is not None and raw.n_coils == 1
    assert raw.meta["EncodedMatrixSize"][0] == raw.data.shape[-1]


# --- GE -----------------------------------------------------------------------------------------


def _by_slice_and_view(raw):
    """Row of every readout by (slice counter, first encoding step)."""
    slices, views = raw.counters("slice"), raw.counters("kspace_encode_step_1")
    return {(int(s), int(v)): i for i, (s, v) in enumerate(zip(slices, views, strict=True))}


@pytest.mark.parametrize(
    ("pfile", "archive", "exact"),
    [("P20480_GRE.7", "ScanArchive_GRE.h5", 896), ("P21504_FSE.7", "ScanArchive_FSE.h5", 728)],
)
def test_ge_pfile_equals_its_scanarchive(pfile, archive, exact):
    """The archive stores the frames as acquired with the slice and view of their control packets;
    the P-file stores them in slots by slice and view. The FRFSE scanner writes 64 of its 792
    frames into the P-file with values up to 5 units off (int32 data up to 2e8), the first echo
    train (views 1 to 8 of each slice) into the slots 257 to 264, so no slot is written twice."""
    _need(GE / pfile, GE / archive)
    image, _ = load(GE / pfile)
    [converted] = load(GE / archive)
    assert image.shape == converted.shape
    rows = _by_slice_and_view(image)
    assert len(rows) == image.n_readouts
    same = 0
    for i, key in enumerate(_by_slice_and_view(converted)):
        j = rows[key]
        assert np.allclose(converted.data[i], image.data[j], atol=5)
        same += np.array_equal(converted.data[i], image.data[j])
        assert np.array_equal(converted.headers["position"][i], image.headers["position"][j])
    assert same == exact
    for key in ("EchoTime", "RepetitionTime", "FieldStrength", "SpectrometerFrequency",
                "SliceThickness", "SpacingBetweenSlices", "EncodedMatrixSize", "ReconMatrixSize",
                "PatientPosition", "ProtocolName", "StudyID"):
        assert image.meta[key] == converted.meta[key], key
    assert converted.meta["SoftwareVersions"] == "DV26.0_R01_1725.a"
    stamps = converted.headers["acquisition_time_stamp"]
    assert stamps[-1] > stamps[0]  # frame times in µs


def test_ge_gre_slice_geometry():
    """Seven coronal slices 18 mm apart (3 mm thick, 15 mm gap) acquired interleaved; the first
    slice's centre is the image header's centre, read along S, phase along L."""
    _need(GE / "P20480_GRE.7")
    image, baseline = load(GE / "P20480_GRE.7")
    assert image.meta["SpacingBetweenSlices"] == 18.0
    first = [int(np.argmax(image.counters("slice") == k)) for k in range(7)]
    positions = image.headers["position"][first]
    assert positions[0].tolist() == pytest.approx([-3.418, 23.888, 13.393], abs=1e-3)
    assert np.allclose(np.diff(positions, axis=0), [[0, -18, 0]] * 6, atol=1e-3)
    assert image.headers["read_dir"][0].tolist() == [0, 0, 1]
    assert image.headers["phase_dir"][0].tolist() == [1, 0, 0]
    assert image.headers["slice_dir"][0].tolist() == [0, 1, 0]
    assert image.headers["idx"]["user"][first, 1].tolist() == [0, 4, 1, 5, 2, 6, 3]  # slots
    assert image.meta["InPlanePhaseEncodingDirectionDICOM"] == "ROW"
    assert not baseline.data.any()  # the baseline frames of this scan are empty
    energy = np.abs(image.data).sum(axis=(1, 2))
    for k in range(7):
        lines = image.counters("kspace_encode_step_1")[image.counters("slice") == k]
        assert lines[np.argmax(energy[image.counters("slice") == k])] == 63  # the k-space centre


def test_ge_fse_slice_geometry():
    _need(GE / "P21504_FSE.7")
    image, _ = load(GE / "P21504_FSE.7")
    assert image.meta["SpacingBetweenSlices"] == 27.0 and image.meta["SliceThickness"] == 2.0
    first = [int(np.argmax(image.counters("slice") == k)) for k in range(3)]
    positions = image.headers["position"][first]
    assert positions[:, 0].tolist() == pytest.approx([-8.548, -35.548, -62.548], abs=1e-3)
    assert positions[0, 1:].tolist() == pytest.approx([-26.147, 19.819], abs=1e-3)
    assert image.headers["read_dir"][0].tolist() == [0, 0, -1]
    assert image.headers["phase_dir"][0].tolist() == [0, 1, 0]
    assert image.headers["slice_dir"][0].tolist() == [1, 0, 0]
    assert set(image.counters("kspace_encode_step_1").tolist()) == set(range(264))


def test_ge_mrs_pfile_frames_through_the_imaging_layout():
    """The frames of a spectroscopy P-file read with the imaging layout are spec2nii's FIDs."""
    _need(GE_MRS)
    header = GeRdbHeader.from_file(GE_MRS)
    assert header.spectroscopy_mapper()
    provenance = load(GE_MRS)[0].provenance
    image, baseline = GePfileRaw(GE_MRS, header, provenance).datasets()
    theirs = GePfile(str(GE_MRS)).map.raw_data[0, 0, 0]  # frames, coils, points
    assert image.shape == theirs.shape == (4, 32, 4096)
    assert np.array_equal(image.data, theirs)
    assert baseline.shape == (1, 32, 4096)
    assert image.meta["SpectrometerFrequency"] == pytest.approx(127.76365, abs=1e-5)


def test_ge_epi_scanarchive():
    """EPI archives hold hyper-frame packets: one per slice and volume, 100 frames each (96 views
    and 4 reference frames); the scanner's reference scan and row-flip table travel with it."""
    _need(GE / "ScanArchive_EPI.h5")
    [raw] = load(GE / "ScanArchive_EPI.h5")
    assert raw.shape == (2100, 32, 238)
    assert raw.counters("repetition").max() == 6 and raw.counters("slice").max() == 2
    assert raw.counters("kspace_encode_step_1").max() == 99
    assert raw.encoding["control_packets"] == 28 and len(raw.meta.vendor["updates"]) == 7
    assert raw.headers["user_int"][0, 0] == 6
    files = raw.meta.vendor["files"]
    assert files["usr/g/bin/rowflip.param"].startswith("# EPI recon control")
    assert files["usr/g/bin/ref.h5"][:8] == b"\x89HDF\r\n\x1a\n"
    assert raw.meta["AcquisitionDateTime"].startswith("2019-05-29T09:11:02")
    assert raw.meta["InPlanePhaseEncodingDirectionDICOM"] == "COL"
    assert raw.headers["position"].any() and not raw.provenance.notes


# --- Philips ------------------------------------------------------------------------------------


def test_philips_raw_export_matches_the_published_reading():
    """The repository of the sample shows what Philips' closed reader returns for it: k-space
    (12 channels, 320 profiles, 640 samples), noise (12 x 19840), the channel names and the
    scan parameters; the k-space centre lies at the encoding centre and the echo at the centre
    sample."""
    _need(PHILIPS)
    image, echo_phase, noise = load(PHILIPS)
    assert image.shape == (320, 12, 640) and noise.shape == (1, 12, 19840)
    assert echo_phase.shape == (1, 2, 640) and noise.flags_of(0) == ["IS_NOISE_MEASUREMENT"]
    assert sorted(image.counters("kspace_encode_step_1").tolist()) == list(range(320))
    m = image.meta
    assert m["RepetitionTime"] == pytest.approx(0.0297091)
    assert m["EchoTime"] == pytest.approx(0.0092081) and m["ExcitationFlipAngle"] == 15.0
    assert m["DwellTime"] == pytest.approx(22.16e-6)
    assert m["NumberOfReceiveChannels"] == 12 and m["ProtocolName"] == "COR GRE"
    assert m["AcquisitionDateTime"] == "2025-06-17T21:06:21.61"
    assert m["EncodedMatrixSize"] == [640, 320, 1] and m["ReconMatrixSize"] == [320, 320, 1]
    assert m["EncodingLimits"]["kspace_encoding_step_1"]["center"] == 159
    names = [e["values"][0] for e in m.vendor["sin"]["channel_names"]]
    assert len(names) == 14 and names[0] == "_ODU_DCC1/SMALL_BODY_ARRAY.E01"
    assert names[-1] == "_ConnQBC/BODY_QUAD.E2"
    energy = np.abs(image.data).sum(axis=(1, 2))
    assert image.counters("kspace_encode_step_1")[np.argmax(energy)] == 159
    echo = int(np.abs(image.data).sum(axis=(0, 1)).argmax())
    assert echo == image.headers["center_sample"][0] == 320
    covariance = noise.data[0] @ noise.data[0].conj().T / noise.data.shape[-1]
    diagonal = np.abs(np.diag(covariance)).max()
    off = np.abs(covariance - np.diag(np.diag(covariance))).max() / diagonal
    assert off < 0.5  # channels mostly independent, as the published covariance picture shows
    assert any("skipped before label 19" in note for note in image.provenance.notes)
    assert any("not settled" in note for note in image.provenance.notes)


def _philips_image(kspace: np.ndarray) -> np.ndarray:
    coils = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace, axes=(1, 2))), axes=(1, 2))
    return np.sqrt((np.abs(coils) ** 2).sum(axis=0))


def _entropy(image: np.ndarray) -> float:
    p = image / image.sum()
    return float(-(p * np.log(p + 1e-30)).sum())


def test_philips_random_phase_is_a_ramp_along_the_readout():
    """Philips' own reader (run locally on the sample) removes the random phase r of each readout
    as exp(i·π·r·(s + 10)/327680) along its samples s, r read as a signed 16-bit number; its
    corrected k-space equals ours corrected so to 0.6 % of the maximum. With that ramp the image
    sharpens (its entropy drops); the constant per-profile phase π·r/32767 of philips_to_ismrmrd
    leaves it as blurred as without a correction."""
    _need(PHILIPS)
    image = load(PHILIPS)[0]
    lines = image.counters("kspace_encode_step_1")
    r = image.headers["user_int"][:, 1].astype(np.int64)
    r = np.where(r > 32767, r - 65536, r)
    samples = np.arange(image.data.shape[-1])
    variants = {
        "none": np.ones((len(r), 1)),
        "ramp": np.exp(1j * np.pi * r[:, None] * (samples + 10) / 327680),
        "constant": np.exp(-1j * np.pi * r[:, None] / 32767) * np.ones(len(samples)),
    }
    entropy = {}
    for name, factor in variants.items():
        kspace = np.zeros((image.n_coils, len(lines), len(samples)), dtype=np.complex128)
        kspace[:, lines] = np.transpose(image.data * factor[:, None, :], (1, 0, 2))
        entropy[name] = _entropy(_philips_image(kspace))
    assert entropy["ramp"] < entropy["none"] - 0.1
    assert abs(entropy["constant"] - entropy["none"]) < 0.02


DIXON = PUBLIC_DATA / "philips_dixon/dixon_2p"


def _dixon_image(raw: RawData, echo: int) -> np.ndarray:
    """A plain reconstruction on the scanner's grid: the lines placed by ky on the Y-resolution,
    the readout zero-filled to twice the X-resolution (2x oversampled), 2D FFT, the central half
    of the readout, root sum of squares over the coils: (Y-resolution, X-resolution)."""
    recon = raw.encoding["reconSpace"]["matrixSize"]
    ny, nx = recon["y"], 2 * recon["x"]
    samples = raw.data.shape[2]
    rows = raw.headers["idx"]["contrast"] == echo
    steps = raw.headers["idx"]["kspace_encode_step_1"][rows].astype(int)
    ky = raw.meta.vendor["offsets_from"]["ky"] + steps
    kspace = np.zeros((raw.data.shape[1], ny, nx), complex)
    kspace[:, ky + ny // 2, nx // 2 - samples // 2:nx // 2 + samples // 2] = \
        np.moveaxis(raw.data[rows], 0, 1)
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace, axes=(1, 2))), axes=(1, 2))
    return np.sqrt((np.abs(image[:, :, nx // 4:3 * nx // 4]) ** 2).sum(0))


@pytest.mark.data
def test_philips_dixon_raw_against_the_scanner_image():
    """The scanner's DICOM holds, per echo, imaginary, magnitude and real frames of 320 x 320,
    the 232 reconstructed rows in its middle. The first echo matches its magnitude frame as
    read; the second echo was read out the other way (the .list's X_range runs backwards for
    it), so it matches mirrored along the readout, better than as read."""
    if not DIXON.is_dir():
        pytest.skip("Philips Dixon sample not fetched")
    raw, noise = load(DIXON / "RAW/raw_000.data")
    assert raw.data.shape == (274, 4, 376) and noise.headers["flags"][0] == 1 << 18
    frames = pydicom.dcmread(DIXON / "DICOM/IM_000.dcm").pixel_array.astype(float)
    middle = frames[:, 44:44 + 232, :]

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])

    first, second = _dixon_image(raw, 0), _dixon_image(raw, 1)
    assert corr(first, middle[2]) > 0.7
    assert corr(second[:, ::-1], middle[3]) > 0.75
    assert corr(first, middle[2]) > corr(first[:, ::-1], middle[2]) + 0.05  # the object is
    assert corr(second[:, ::-1], middle[3]) > corr(second, middle[3]) + 0.05  # near symmetric
