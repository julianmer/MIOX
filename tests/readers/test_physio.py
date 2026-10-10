####################################################################################################
#                                         test_physio.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/10/26                                                                                #
#                                                                                                  #
# Purpose: Physiological logs on synthetic files: Siemens PMU (markers, text sections, stated and  #
#          derived rates), CMRR logs as files and inside a DICOM file (tic grid, gaps, triggers,   #
#          volume starts), Philips SCANPHYSLOG (mark bits), GE PPG/RESP files, BIDS physio and an  #
#          HCP physio log.                                                                         #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import gzip
import json
import struct

import numpy as np
import pydicom
import pytest
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

# own
import miox as gu


def _pmu_footer(start_ms: int, stop_ms: int) -> str:
    return (
        "ECG  Freq Per: 0 0\nPULS Freq Per: 72 833\nNrTrig NrMP NrArr AcqWin: 0 0 0 0\n"
        f"LogStartMDHTime:  {start_ms}\nLogStopMDHTime:   {stop_ms}\n"
        f"LogStartMPCUTime: {start_ms + 7}\nLogStopMPCUTime:  {stop_ms + 7}\n6003\n"
    )


def test_siemens_pmu(tmp_path):
    samples = list(range(1000, 1400))  # 400 samples over 1 s
    values = samples[:10] + [5000] + samples[10:20] + [6000] + samples[20:]
    text = ("1 2 40 280 5002 LOGVERSION_PULS   1 6002 " + " ".join(map(str, values)) + " 5003\n"
            + _pmu_footer(36_000_000, 36_001_000))
    (tmp_path / "scan.puls").write_text(text)
    sniff = gu.sniff(tmp_path / "scan.puls")
    assert (sniff.format, sniff.kind) == (gu.Format.SIEMENS_PMU, gu.Kind.PHYSIO)
    [physio] = gu.load(tmp_path / "scan.puls")
    assert isinstance(physio, gu.PhysioData) and physio.channels == ("cardiac",)
    assert physio.data[:, 0].tolist() == samples  # markers are no samples
    assert physio.events["trigger"].tolist() == [10]
    assert physio.events["marker_6000"].tolist() == [20]
    assert physio.sampling_frequency == 400 and physio.start_time == 36000.0
    assert physio.duration == 1.0 and physio.times()[1] == pytest.approx(36000.0025)
    footer = physio.meta.vendor["footer"]
    assert footer["LogStartMPCUTime"] == [36_000_007] and footer["PULS Freq Per"] == [72, 833]
    assert physio.meta.vendor["header_values"] == [1, 2, 40, 280]
    assert physio.meta.vendor["text_sections"] == ["LOGVERSION_PULS   1"]
    # a stated rate wins over the logged duration
    stated = ("1 2 20 2 5002 Logging RESP signal: RESP_SAMPLES_PER_SECOND = 50; 6002 "
              + " ".join(map(str, samples)) + " 5003\n" + _pmu_footer(1000, 9000))
    (tmp_path / "scan.resp").write_text(stated)
    [resp] = gu.load(tmp_path / "scan.resp")
    assert resp.sampling_frequency == 50 and resp.channels == ("respiratory",)


def _cmrr_log(kind: str, step: int, rows: list[tuple]) -> str:
    lines = ["UUID        = 1234", "ScanDate    = 20190114_143828", "LogVersion  = EJA_1",
             f"LogDataType = {kind}", f"SampleTime  = {step}", "",
             "ACQ_TIME_TICS  CHANNEL  VALUE  SIGNAL", ""]
    lines += ["  ".join(str(v) for v in row) for row in rows]
    return "\n".join(lines) + "\n"


INFO = ("UUID        = 1234\nLogDataType = ACQUISITION_INFO\nNumSlices   = 2\nNumVolumes  = 2\n\n"
        "VOLUME   SLICE   ACQ_START_TICS  ACQ_FINISH_TICS  ECHO\n\n"
        "0 0 1008 1010 0\n0 1 1012 1014 0\n1 0 1020 1022 0\n1 1 1024 1026 0\n")


def test_cmrr_logs_and_dicom(tmp_path):
    rows = [(1001, "PULS", 10), (1002, "PULS", 11, "PULS_TRIGGER"), (1004, "PULS", 12),
            (1008, "PULS", 14), (1010, "PULS", 15)]  # first row off the grid, 1006 missing
    (tmp_path / "Physio_x_PULS.log").write_text(_cmrr_log("PULS", 2, rows))
    (tmp_path / "Physio_x_Info.log").write_text(INFO)
    assert gu.sniff(tmp_path / "Physio_x_PULS.log").format == gu.Format.CMRR_PHYSIO
    [puls] = gu.load(tmp_path / "Physio_x_PULS.log")
    assert puls.sampling_frequency == 200 and puls.channels == ("cardiac",)
    assert np.array_equal(puls.data[:, 0], [10, 11, 12, np.nan, 14, 15], equal_nan=True)
    assert puls.start_time == pytest.approx(1000 * 2.5e-3)
    assert puls.events["puls_trigger"].tolist() == [1]
    assert puls.events["volume"].tolist() == [4]  # 1008; the volume at 1020 is after the samples
    notes = " ".join(puls.provenance.notes)
    assert "1 samples missing" in notes and "1 rows off the SampleTime grid" in notes
    [info] = gu.load(tmp_path / "Physio_x_Info.log")
    assert info.sampling_frequency is None and info.data.shape == (4, 5)
    assert info.events["volume"].tolist() == [0, 2] and info.channels[2] == "acq_start_tics"
    # the same logs inside a DICOM file, the last line cut off
    ecg = _cmrr_log("ECG", 1, [(2000, "ECG1", 1), (2000, "ECG2", 2), (2001, "ECG1", 3),
                               (2001, "ECG2", 4)]) + "  2002  ECG1"
    texts = [("Physio_x_ECG.log", ecg), ("Physio_x_Info.log", INFO)]
    blob = b""
    for name, text in texts:
        data = text.encode()
        chunk = struct.pack("<II", len(data), len(name)) + name.encode()
        blob += chunk.ljust(1024, b"\0") + data.ljust(2048 - 1024, b"\0")
    ds = pydicom.Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.3.12.2.1107.5.9.1"
    ds.SOPClassUID = "1.3.12.2.1107.5.9.1"
    ds.SOPInstanceUID = "1.2.3"
    ds.ImageType = ["ORIGINAL", "PRIMARY", "RAWDATA", "PHYSIO"]
    ds.AcquisitionNumber = 2
    ds.add_new(0x7FE10010, "LO", "SIEMENS CSA NON-IMAGE")
    ds.add_new(0x7FE11010, "OB", blob)
    path = tmp_path / "physio.dcm"
    ds.save_as(path, enforce_file_format=True)
    sniff = gu.sniff(path)
    assert (sniff.format, sniff.kind) == (gu.Format.CMRR_PHYSIO, gu.Kind.PHYSIO)
    ecg_data, info_data = gu.load(path)
    assert ecg_data.channels == ("ecg1", "ecg2") and ecg_data.sampling_frequency == 400
    assert ecg_data.data.tolist() == [[1, 2], [3, 4]]
    assert any("last line cut off" in n for n in ecg_data.provenance.notes)
    assert info_data.data.shape == (4, 5) and ecg_data.name == "ECG"


def test_philips_physlog(tmp_path):
    lines = ["## Site (SRN = 1), Release r5 (SWID 1)", "## Wed 08-05-2019 09:37:29",
             "# v1raw v2raw  v1 v2  ppu resp  gx gy gz mark"]
    marks = ["0000", "0010", "0202", "0000", "0020"]
    lines += [f"0 0  1 2  {100 + i} {200 + i}  0 0 0 {m}" for i, m in enumerate(marks)]
    (tmp_path / "SCANPHYSLOG.log").write_text("\n".join(lines) + "\n")
    assert gu.sniff(tmp_path / "SCANPHYSLOG.log").format == gu.Format.PHILIPS_PHYSLOG
    [log] = gu.load(tmp_path / "SCANPHYSLOG.log")
    assert log.channels == ("v1raw", "v2raw", "v1", "v2", "cardiac", "respiratory", "gx", "gy",
                            "gz")
    assert log.data[:, 4].tolist() == [100, 101, 102, 103, 104]
    assert log.sampling_frequency == 500 and log.start_time is None
    assert log.events["scan_start"].tolist() == [1] and log.events["volume"].tolist() == [2]
    assert log.events["ppu_trigger"].tolist() == [2] and log.events["scan_end"].tolist() == [4]
    assert log.meta.vendor["header"][1] == "Wed 08-05-2019 09:37:29"


def test_ge_physio_and_bids_physio(tmp_path):
    (tmp_path / "PPGData_epiRT_0000000000_00_00_000").write_text(" 1\n -2\n 3\n")
    assert gu.sniff(tmp_path / "PPGData_epiRT_0000000000_00_00_000").format == gu.Format.GE_PHYSIO
    [ppg] = gu.load(tmp_path / "PPGData_epiRT_0000000000_00_00_000")
    assert ppg.data[:, 0].tolist() == [1, -2, 3] and ppg.sampling_frequency == 100
    assert ppg.start_time == -30 and ppg.channels == ("cardiac",)
    (tmp_path / "RESPData_epiRT_0000000000_00_00_000.gep").write_text("5\n6\n")
    [resp] = gu.load(tmp_path / "RESPData_epiRT_0000000000_00_00_000.gep")
    assert resp.sampling_frequency == 25 and resp.name == "RESPData_epiRT_0000000000_00_00_000"
    # BIDS physio
    with gzip.open(tmp_path / "sub-1_task-rest_physio.tsv.gz", "wt") as fh:
        fh.write("1.5\t0\nn/a\t1\n2.5\t0\n")
    sidecar = {"Columns": ["cardiac", "trigger"], "SamplingFrequency": 100.0, "StartTime": -1.5,
               "Manufacturer": "x"}
    (tmp_path / "sub-1_task-rest_physio.json").write_text(json.dumps(sidecar))
    sniff = gu.sniff(tmp_path / "sub-1_task-rest_physio.tsv.gz")
    assert sniff.format == gu.Format.BIDS_PHYSIO and len(sniff.companions) == 1
    [bids] = gu.load(tmp_path / "sub-1_task-rest_physio.tsv.gz")
    assert bids.channels == ("cardiac", "trigger") and bids.start_time == -1.5
    assert np.array_equal(bids.data[:, 0], [1.5, np.nan, 2.5], equal_nan=True)
    assert bids.name == "sub-1_task-rest_physio" and bids.meta.vendor["sidecar"]["Manufacturer"]


def test_hcp_physio(tmp_path):
    path = tmp_path / "rfMRI_REST1_LR_Physio_log.txt"
    path.write_text("".join(f"{int(i % 4 == 0)}\t{1900 + i}\t{1750 - i}\n" for i in range(8)))
    sniff = gu.sniff(path)
    assert (sniff.format, sniff.kind) == (gu.Format.HCP_PHYSIO, gu.Kind.PHYSIO)
    [d] = gu.load(path)
    assert d.channels == ("trigger", "respiratory", "cardiac")
    assert d.sampling_frequency == 400.0 and d.data.shape == (8, 3)
    assert d.data[4, 0] == 1 and d.data[2, 1] == 1902
