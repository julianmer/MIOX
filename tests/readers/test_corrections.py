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

from pathlib import Path

import numpy as np
import pydicom
import pytest
from nifti_mrs.create_nmrs import gen_nifti_mrs

# own
import miox
from miox.data import MRSData, Provenance
from miox.meta import Meta
from miox.readers import (
    PhilipsDataListReader,
    PhilipsSparSdatReader,
    VarianReader,
)
from miox.readers.philips_dicom_mrs import PhilipsDicomMrs
from miox.readers.siemens_dicom_mrs import SiemensDicomMrs
from miox.readers.siemens_twix_mrs import TwixMrs


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
    """MeasYaps alTI[0] in µs, in s (spec2nii: µs on XA, none on VB to VE)."""
    hdr = {
        "Dicom": {"SoftwareVersions": "syngo MR XA30", "lFrequency": 123.2e6, "Manufacturer": "",
                  "ManufacturersModelName": "", "DeviceSerialNumber": 1.0,
                  "InstitutionName": "", "InstitutionAddress": "", "tProtocolName": ""},
        "Meas": {"tSequenceString": "svs_se", "tPatientPosition": "HFS", "tPatientName": "",
                 "PatientSex": 3.0, "PatientBirthDay": 1.0, "tICEProgramName": "",
                 "flUsedPatientWeight": 70.0},
        "MeasYaps": {("alTI", "0"): 300000.0, ("alTR", "0"): 2e6,
                     ("adFlipAngleDegree", "0"): 90.0,
                     ("sTXSPEC", "asNucleusInfo", "0", "tNucleus"): '"1H"'},
        "Phoenix": {("alTE", "0"): 30000.0},
        "Config": {"SequenceFileName": ""},
    }
    twix = {"hdr": hdr, "image": type("Stream", (), {"filename": "/x/meas.dat"})()}
    values = TwixMrs(twix, Path("meas.dat")).values("image")
    assert values["InversionTime"] == (pytest.approx(0.3), "MeasYaps/alTI[0]")


def test_siemens_dicom_inversion_time_in_ms():
    """The CSA inversion time in ms, in s (spec2nii: ms)."""
    reader = SiemensDicomMrs()
    reader.protocol = lambda ds: {("alTE", "0"): 30000.0}
    csa = {"ImagingFrequency": [123.2], "ImagedNucleus": ["1H"], "RepetitionTime": [2000.0],
           "FlipAngle": [90.0], "TransmittingCoil": ["Body"], "SequenceName": ["svs_se"],
           "InversionTime": [300.0], "ReceivingCoil": ["Head_32"]}
    values = reader._values_vx(pydicom.Dataset(), csa)
    assert values["InversionTime"] == (pytest.approx(0.3), "csa_image/InversionTime")


def test_philips_classic_dicom_times_in_ms():
    ds = pydicom.Dataset()
    ds.file_meta = pydicom.dataset.FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = PhilipsDicomMrs.CLASSIC
    ds.add_new((0x2001, 0x1083), "DS", "127.7")
    ds.add_new((0x2001, 0x1087), "CS", "1H")
    ds.add_new((0x2005, 0x1310), "FL", 34.0)
    ds.add_new((0x2005, 0x1357), "FL", 2000.0)
    ds.RepetitionTime, ds.FlipAngle = 910.0, 90.0
    values = PhilipsDicomMrs().values(ds, ["x"])
    assert values["EchoTime"][0] == pytest.approx(0.034)
    assert values["RepetitionTime"][0] == pytest.approx(0.910)
    assert values["ExcitationFlipAngle"][0] == 90.0


def test_spar_inversion_time_in_ms():
    header = {"synthesizer_frequency": 127.7e6, "nucleus": "1H", "echo_time": 30,
              "repetition_time": 2000, "offset_frequency": 0, "spectrum_inversion_time": 150}
    values = PhilipsSparSdatReader()._values(header, Path("x.SDAT"), [])
    assert values["InversionTime"][0] == pytest.approx(0.15)


def test_varian_echo_time_from_te_or_pulse_delay():
    procpar = {"te": ["0.03"], "p1": ["0"], "alfa": ["20"]}
    d = _dataset({"procpar": procpar}, {"EchoTime": (20.0, "x")})
    VarianReader().correct(d)
    assert d.meta["EchoTime"] == pytest.approx(0.03)
    d = _dataset({"procpar": {"p1": ["0"], "alfa": ["20"]}}, {"EchoTime": (20.0, "x")})
    VarianReader().correct(d)
    assert d.meta["EchoTime"] == pytest.approx(20e-6)


def test_hyper_echo_times():
    reader, values = PhilipsDataListReader(), {"EchoTime": (0.08, "echo_time")}
    act = np.zeros((1, 1, 1, 8, 2, 36), complex)
    short, edited = reader._parts("STD_0", act, ["chan", "extr1"], values, hyper=True)
    assert short[2]["EchoTime"][0] == pytest.approx(0.035)
    assert edited[2]["EchoTime"][0] == pytest.approx(0.08)
    ref = np.zeros((1, 1, 1, 8, 2, 4), complex)
    ((_, _, ref_values, tags, _),) = reader._parts("STD_1", ref, ["chan", "extr1"], values, True)
    assert "EchoTime" not in ref_values
    assert tags[1][2]["EchoTime"] == [0.08, 0.035, 0.08, 0.035]
