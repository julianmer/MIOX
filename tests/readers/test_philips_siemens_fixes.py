####################################################################################################
#                                  test_philips_siemens_fixes.py                                   #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 19/11/26                                                                                #
#                                                                                                  #
# Purpose: Philips SPAR/SDAT and data/list, Siemens and UIH DICOM spectroscopy: SPAR texts kept as #
#          written, a short SDAT, an inversion time of 0, the SPAR of a data/list among several,   #
#          rows that are not the stated grid. Synthetic files, and the public HYPER sample.        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import numpy as np
import pydicom
import pytest

# own
import miox
from miox.readers import PhilipsSparSdatReader
from miox.readers.base import ReaderError
from miox.readers.siemens_dicom_mrs import SiemensDicomMrs
from miox.readers.uih_dicom_mrs import UihDicomMrs

SPAR = {"samples": 8, "rows": 1, "synthesizer_frequency": 127748760, "nucleus": "1H",
        "echo_time": 30, "repetition_time": 2000, "offset_frequency": 0, "sample_frequency": 2000,
        "volume_selection_enable": '"no"'}


def _pair(folder: Path, sdat_bytes: int, **fields) -> Path:
    """A SPAR/SDAT pair of zeros with these fields; the SDAT's path."""
    lines = [f"{key} : {value}" for key, value in (SPAR | fields).items()]
    (folder / "scan.SPAR").write_text("\n".join(lines) + "\n")
    (folder / "scan.SDAT").write_bytes(bytes(sdat_bytes))
    return folder / "scan.SDAT"


#**************************************************************************************************#
#                                            SPAR values                                           #
#**************************************************************************************************#
def test_spar_texts_stay_as_written(tmp_path):
    sdat = _pair(tmp_path, 64, patient_name="1024_2", scan_id="2020_01_15", examination_name="1e3",
                 patient_birth_date="01011970", scan_date="1,5", equipment_sw_verions="5_7",
                 patient_position='"head_first"', patient_orientation='""')
    header = PhilipsSparSdatReader.read_spar(sdat.with_suffix(".SPAR"))
    assert header["patient_name"] == "1024_2" and header["scan_id"] == "2020_01_15"
    assert header["examination_name"] == "1e3" and header["patient_birth_date"] == "01011970"
    assert header["scan_date"] == "1,5" and header["equipment_sw_verions"] == "5_7"
    assert header["patient_position"] == "head_first" and header["patient_orientation"] is None
    d = miox.load(sdat)[0]
    assert d.meta["PatientName"] == "1024_2" and d.meta["ProtocolName"] == "2020_01_15"
    assert d.meta["PatientDoB"] == "01011970" and d.meta["SoftwareVersions"] == "5_7"


def test_spar_numbers_are_numbers():
    value = PhilipsSparSdatReader.value
    assert value("rows", "16") == 16 and isinstance(value("rows", "16"), int)
    assert value("echo_time", "-30.500000") == -30.5 and value("x", "1.5e-3") == 0.0015
    assert value("x", "1e3") == 1000.0 and value("x", ".5") == 0.5 and value("x", "+2") == 2
    assert value("x", "") is None and value("nucleus", "1H") == "1H"
    assert value("patient_name", "12") == "12" and value("x", "0x10") == "0x10"


#**************************************************************************************************#
#                                               SDAT                                               #
#**************************************************************************************************#
def test_short_sdat_is_a_reader_error(tmp_path):
    sdat = _pair(tmp_path, 60)
    with pytest.raises(ReaderError, match=r"scan\.SDAT: 60 bytes.*1 rows of 8 samples \(64 bytes"):
        miox.load(sdat)


def test_rows_off_the_stated_grid_are_noted(tmp_path):
    fields = {"phase_encoding_enable": '"yes"', "dim2_direction": "x", "dim2_pnts": 2,
              "dim3_direction": "y", "dim3_pnts": 2}
    d = miox.load(_pair(tmp_path, 3 * 64, rows=3, **fields))[0]
    assert d.data.shape == (1, 1, 1, 8, 3)
    assert any("3 rows are not the 2 x 2 x 1 grid" in note for note in d.provenance.notes)
    (tmp_path / "grid").mkdir()
    d = miox.load(_pair(tmp_path / "grid", 4 * 64, rows=4, **fields))[0]
    assert d.data.shape == (2, 2, 1, 8)
    assert not any("are not the" in note for note in d.provenance.notes)


#**************************************************************************************************#
#                                          inversion time                                          #
#**************************************************************************************************#
def _siemens_values(inversion: list) -> dict:
    reader = SiemensDicomMrs()
    reader.protocol = lambda ds: {("alTE", "0"): 30000}
    reader._dicom_values = lambda ds, patient_id: {}
    csa = {"ImagingFrequency": [123.2], "ImagedNucleus": ["1H"], "RepetitionTime": [2000.0],
           "FlipAngle": [90.0], "TransmittingCoil": ["Body"], "SequenceName": ["svs_se"],
           "ReceivingCoil": ["Head"], "InversionTime": inversion}
    return reader._values_vx(pydicom.Dataset(), csa)


def test_siemens_inversion_time_of_zero_is_left_out():
    assert "InversionTime" not in _siemens_values([0.0])
    assert "InversionTime" not in _siemens_values([])
    assert _siemens_values([150.0])["InversionTime"][0] == pytest.approx(0.15)


def _uih_values(inversion) -> dict:
    ds = pydicom.Dataset()
    ds.TransmitterFrequency, ds.ResonantNucleus, ds.EchoTime = 128.2, "1H", 30.0
    ds.RepetitionTime, ds.FlipAngle, ds.SpectralWidth = 2000.0, 90.0, 2000.0
    for keyword in ("Manufacturer", "ManufacturerModelName", "DeviceSerialNumber",
                    "SoftwareVersions", "InstitutionName", "InstitutionAddress", "SequenceName",
                    "ProtocolName", "PatientPosition"):
        setattr(ds, keyword, "X")
    if inversion is not None:
        ds.InversionTime = inversion
    return UihDicomMrs().values(ds, Path("x.dcm"))


def test_uih_inversion_time_of_zero_is_left_out():
    assert "InversionTime" not in _uih_values(0.0)
    assert "InversionTime" not in _uih_values(None)
    assert _uih_values(150.0)["InversionTime"][0] == pytest.approx(0.15)


#**************************************************************************************************#
#                                     the SPAR of a data/list                                      #
#**************************************************************************************************#
ACT = "HBCD_HYPER_r5712_Export_WIP_HYPER_5_2_raw_act.SPAR"


def _hyper(public_data: Path, folder: Path, other: str | None = None, stem: str = "raw_226"):
    """The public HYPER data/list read from a copy of its folder, with another scan's SPAR (its
    echo time 144 ms, its protocol PRESS) beside it under the name ``other``."""
    source = public_data / "spec2nii_test_data" / "philips" / "hyper"
    if not (source / "raw_226.data").exists():
        pytest.skip("spec2nii test data not fetched")
    folder.mkdir()
    for suffix in (".data", ".list"):
        shutil.copy(source / f"raw_226{suffix}", folder / f"{stem}{suffix}")
    for path in source.glob("*.SPAR"):
        shutil.copy(path, folder / path.name)
    if other:
        text = (source / ACT).read_text()
        assert "\nscan_id : HYPER\n" in text and "\necho_time : 80\n" in text
        text = text.replace("\nscan_id : HYPER\n", "\nscan_id : PRESS\n")
        (folder / other).write_text(text.replace("\necho_time : 80\n", "\necho_time : 144\n"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return miox.load(folder / f"{stem}.data")


def test_data_list_does_not_take_another_scans_spar(public_data, tmp_path):
    """The other scan's SPAR sorts first; the names do not tell which is raw_226's."""
    alone = _hyper(public_data, tmp_path / "alone")
    assert {"raw_226_hyper_short_te", "raw_226_hyper_edited"} < {d.name for d in alone}
    assert alone[0].meta["ProtocolName"] == "HYPER" and alone[0].meta["OriginalFile"][2] == ACT
    assert not any("several SPAR" in note for note in alone[0].provenance.notes)
    both = _hyper(public_data, tmp_path / "both", "AAA_PRESS_7_2_raw_act.SPAR")
    assert not any("hyper" in d.name for d in both)
    for d in both:
        assert "EchoTime" not in d.meta and "ProtocolName" not in d.meta
        assert d.meta["SpectrometerFrequency"] == alone[0].meta["SpectrometerFrequency"]
        assert d.dwelltime == alone[0].dwelltime
        np.testing.assert_array_equal(d.affine, alone[0].affine)
        assert d.meta["OriginalFile"][2:] == ["AAA_PRESS_7_2_raw_act.SPAR", ACT]
        note = next(n for n in d.provenance.notes if "several SPAR" in n)
        assert "EchoTime" in note and "ProtocolName" in note


def test_data_list_takes_the_spar_of_its_scan_number(public_data, tmp_path):
    """raw_5.data belongs to <name>_5_2_raw_act.SPAR, whatever else lies beside it."""
    got = _hyper(public_data, tmp_path / "named", "AAA_PRESS_7_2_raw_act.SPAR", stem="raw_5")
    short = next(d for d in got if d.name == "raw_5_hyper_short_te")
    assert short.meta["EchoTime"] == 0.035 and short.meta["ProtocolName"] == "HYPER"
    assert got[0].meta["OriginalFile"] == ["raw_5.data", "raw_5.list", ACT]
    assert not any("several SPAR" in note for note in got[0].provenance.notes)
