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
#          and the public ISMRMRD files of UIH, OSI2ONE and Gadgetron reconstructions.             #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import numpy as np
import pytest

# own
from miox import MRImage, RawData, load
from tests.conftest import PUBLIC_DATA

GADGETRON = PUBLIC_DATA / "gadgetron"
ISMRMRD = PUBLIC_DATA / "ismrmrd"
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
