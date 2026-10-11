####################################################################################################
#                                       test_corrections.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Value corrections on synthetic metadata: float text, quotes, and the units spec2nii     #
#          leaves in ms or µs. No test data needed.                                                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import numpy as np
import pytest
from nifti_mrs.create_nmrs import gen_nifti_mrs

# own
import miox
from miox.data import MRSData, Provenance
from miox.meta import Meta
from miox.readers import (
    DicomMrsReader,
    PhilipsDataListReader,
    PhilipsSparSdatReader,
    SiemensTwixReader,
    VarianReader,
)


def _dataset(vendor: dict, values: dict[str, tuple]) -> MRSData:
    """MRSData with ``values`` as key -> (value, vendor field)."""
    meta = Meta(vendor)
    for key, (value, field) in values.items():
        meta.set(key, value, f"spec2nii test: {field}", field)
    provenance = Provenance((), miox.Format.UNKNOWN, "test", {})
    return MRSData(np.zeros((1, 1, 1, 8), complex), 1e-3, np.eye(4), meta, provenance)


def test_float_text_and_quotes(tmp_path):
    image = gen_nifti_mrs(np.ones((1, 1, 1, 64), complex), 2.5e-4, 123.2, "1H", no_conj=True)
    image.add_hdr_field("DeviceSerialNumber", "66050.0")
    image.add_hdr_field("PatientDoB", "19890117.0")
    image.add_hdr_field("RxCoil", '"Head_32"')
    path = tmp_path / "svs.nii.gz"
    image.save(path)
    [d] = miox.load(path)
    assert d.meta["DeviceSerialNumber"] == "66050" and d.meta["PatientDoB"] == "19890117"
    assert d.meta["RxCoil"] == "Head_32"
    assert d.meta.source["RxCoil"].endswith("corrected: surrounding quotes removed")


def test_twix_inversion_time_from_protocol():
    d = _dataset({"MeasYaps": {"alTI[0]": 300000.0}}, {"InversionTime": (300000.0, "x")})
    SiemensTwixReader().correct(d)
    assert d.meta["InversionTime"] == pytest.approx(0.3)
    assert d.meta.fields["InversionTime"] == "MeasYaps/alTI[0]"


def test_siemens_dicom_inversion_time_in_ms():
    vendor = {"dicom": {"Manufacturer": "SIEMENS"}, "csa_image": {"InversionTime": 300.0}}
    d = _dataset(vendor, {"InversionTime": (300.0, "csa_image/InversionTime")})
    DicomMrsReader().correct(d)
    assert d.meta["InversionTime"] == pytest.approx(0.3)


def test_philips_classic_dicom_times_in_ms():
    vendor = {
        "dicom": {"Manufacturer": "Philips", "FlipAngle": 90.0},
        "file_meta": {"MediaStorageSOPClassUID": "1.3.46.670589.11.0.0.12.1"},
    }
    d = _dataset(
        vendor,
        {
            "EchoTime": (34.0, "dicom/(2005,1310)"),
            "RepetitionTime": (910.0, "dicom/RepetitionTime"),
            "ExcitationFlipAngle": (90.0, "dicom/FlipAngle"),
        },
    )
    DicomMrsReader().correct(d)
    assert d.meta["EchoTime"] == pytest.approx(0.034)
    assert d.meta["RepetitionTime"] == pytest.approx(0.910)
    assert d.meta["ExcitationFlipAngle"] == 90.0


def test_philips_enhanced_dicom_untouched():
    vendor = {
        "dicom": {"Manufacturer": "Philips"},
        "file_meta": {"MediaStorageSOPClassUID": "1.2.840.10008.5.1.4.1.1.4.2"},
    }
    d = _dataset(vendor, {"RepetitionTime": (0.91, "dicom/RepetitionTime")})
    DicomMrsReader().correct(d)
    assert d.meta["RepetitionTime"] == 0.91


def test_spar_inversion_time_in_ms():
    d = _dataset({}, {"InversionTime": (150.0, "spectrum_inversion_time")})
    PhilipsSparSdatReader().correct(d)
    assert d.meta["InversionTime"] == pytest.approx(0.15)


def test_varian_echo_time_from_te_or_pulse_delay():
    procpar = {"te": ["0.03"], "p1": ["0"], "alfa": ["20"]}
    d = _dataset({"procpar": procpar}, {"EchoTime": (20.0, "x")})
    VarianReader().correct(d)
    assert d.meta["EchoTime"] == pytest.approx(0.03)
    d = _dataset({"procpar": {"p1": ["0"], "alfa": ["20"]}}, {"EchoTime": (20.0, "x")})
    VarianReader().correct(d)
    assert d.meta["EchoTime"] == pytest.approx(20e-6)


def test_hyper_echo_times():
    d = _dataset({}, {"EchoTime": (0.08, "echo_time")})
    d.name = "raw_226_hyper_short_te"
    PhilipsDataListReader().correct(d)
    assert d.meta["EchoTime"] == pytest.approx(0.035)
    d = _dataset({}, {"EchoTime": (0.08, "echo_time")})
    d.name, d.data = "raw_226_hyper_water_ref", np.zeros((1, 1, 1, 8, 2, 4), complex)
    PhilipsDataListReader().correct(d)
    assert "EchoTime" not in d.meta
    assert d.dim_info[6]["header"]["EchoTime"] == [0.08, 0.035, 0.08, 0.035]
