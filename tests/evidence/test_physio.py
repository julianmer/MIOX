####################################################################################################
#                                         test_physio.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/10/26                                                                                #
#                                                                                                  #
# Purpose: Physiological logs against independent evidence: CMRR physio DICOM files against        #
#          bidsphysio's BIDS conversions of them, Siemens PMU files against bidsphysio's samples   #
#          and their own logged durations, Philips SCANPHYSLOG volume markers against the TR and   #
#          volume count of the scans they were recorded with, GE PPG and respiration files of one  #
#          scan against each other, an HCP physio log against the HCP's TR and volume count.       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

# own
from miox import load
from tests.conftest import PUBLIC_DATA

BIDSPHYSIO = PUBLIC_DATA / "bidsphysio"
CMRR = BIDSPHYSIO / "bidsphysio.dcm2bids/tests/data"
PMU = BIDSPHYSIO / "bidsphysio.pmu2bids/tests/data"
PHYSLOG = PUBLIC_DATA / "scanphyslog2bids/scanphyslog2bids/data"
GE = PUBLIC_DATA / "phys2bids_ge"


def _need(*paths):
    for path in paths:
        if not path.is_file():
            pytest.skip(f"{path.relative_to(PUBLIC_DATA)} not fetched")


def test_cmrr_physio_dicom_equals_the_bids_conversion():
    """bidsphysio converted the CMRR physio DICOM to BIDS: the cardiac, respiratory and external
    trigger signals equal it sample for sample (missing samples NaN in both), its trigger column
    marks the volume starts read here, and its start times relative to the first volume agree to
    one sample (the grid here starts at the sampling phase most rows lie on)."""
    _need(CMRR / "samplePhysioCMRR.dcm")
    puls, resp, ext, info = load(CMRR / "samplePhysioCMRR.dcm")
    first_volume = info.data[0, 2] * 2.5e-3
    for ours, name in ((puls, "dcm_cardiac"), (resp, "dcm_respiratory"),
                       (ext, "dcm_external_trigger")):
        [theirs] = load(CMRR / f"{name}.tsv")
        assert np.array_equal(ours.data[:, 0], theirs.data[:, 0], equal_nan=True), name
        assert np.array_equal(np.flatnonzero(theirs.data[:, 1] == 1), ours.events["volume"])
        assert ours.sampling_frequency == theirs.sampling_frequency
        period = 1 / ours.sampling_frequency
        assert ours.start_time - first_volume == pytest.approx(theirs.start_time, abs=period)


def test_siemens_pmu_files_match_bidsphysio_and_their_logged_durations():
    """The 5000 markers lie between the samples: without them the VB15A files hold exactly 50
    samples per second of their logged MPCU duration (51.4 with them), and the samples equal
    bidsphysio's (a 100-sample window of its VE11C output, the start of its VB15A and VBX ones)."""
    _need(PMU / "sample_VB15A.puls", PMU / "sample_VE11C.puls")
    for name in ("sample_VB15A.puls", "sample_VB15A.resp"):
        [physio] = load(PMU / name)
        footer = physio.meta.vendor["footer"]
        duration = (footer["LogStopMPCUTime"][0] - footer["LogStartMPCUTime"][0]) / 1000
        assert physio.n_samples / duration == pytest.approx(50, abs=0.01)
        assert physio.sampling_frequency == 50 and len(physio.events["trigger"]) > 100
    for sample, reference, offset in (("sample_VE11C.puls", "pmu_VE11C_cardiac.tsv", 205),
                                      ("sample_VE11C.resp", "pmu_VE11C_respiratory.tsv", 205),
                                      ("sample_VB15A.resp", "pmu_VB15A_respiratory.tsv", 0),
                                      ("sample_VBX.puls", "pmu_VBX_cardiac.tsv", 0)):
        [physio] = load(PMU / sample)
        theirs = np.loadtxt(PMU / reference, ndmin=1)
        assert np.array_equal(physio.data[offset : offset + len(theirs), 0], theirs), sample
    [ve11c] = load(PMU / "sample_VE11C.puls")
    assert ve11c.sampling_frequency == 400 and ve11c.start_time == 39008.572


def test_philips_physlog_volume_markers_match_the_scan():
    """The volume markers (0x0200) of the SCANPHYSLOG number the scan's volumes and lie one TR
    apart at 500 samples per second (496 would put them 653 samples apart, not 658)."""
    _need(PHYSLOG / "example_for_vol_markers.log", PHYSLOG / "example_for_vol_markers.nii.gz")
    [log] = load(PHYSLOG / "example_for_vol_markers.log")
    scan = nib.load(PHYSLOG / "example_for_vol_markers.nii.gz")
    tr = float(scan.header["pixdim"][4])
    volumes = log.events["volume"]
    assert len(volumes) == scan.shape[3] == 193
    assert np.median(np.diff(volumes)) == round(tr * log.sampling_frequency) == 658
    assert len(log.events["scan_start"]) == len(log.events["scan_end"]) == 1


def test_ge_ppg_and_respiration_of_one_scan_cover_the_same_time():
    """GE's files state no rates; at its fixed 100 Hz (PPG) and 25 Hz (respiration) the two files
    of one scan cover the same time to one respiration sample."""
    _need(GE / "PPGData_epiRT_0000000000_00_00_000", GE / "RESPData_epiRT_0000000000_00_00_000")
    [ppg] = load(GE / "PPGData_epiRT_0000000000_00_00_000")
    [resp] = load(GE / "RESPData_epiRT_0000000000_00_00_000")
    assert abs(ppg.duration - resp.duration) <= 1 / resp.sampling_frequency
    [gep] = load(GE / "PPGData_epiRT_0000000000_00_00_000.gep")
    assert np.array_equal(gep.data, ppg.data)


def test_hcp_physio_trigger_follows_the_hcp_tr():
    """The HCP motor task (tfMRI_MOTOR_LR: 284 volumes, TR 0.72 s): at the 400 Hz the reader
    sets, the trigger rises every 288 samples (0.72 s), it starts high and rises 283 more times
    (284 volumes), and the log lasts 284 TRs to within one TR."""
    path = PUBLIC_DATA / "physio_examples/Siemens_HCP/PPU3T/tfMRI_MOTOR_LR_Physio_log.txt"
    if not path.is_file():
        pytest.skip("not fetched")
    [d] = load(path)
    trigger = d.data[:, d.channels.index("trigger")]
    rises = np.flatnonzero(np.diff(trigger) > 0)
    assert trigger[0] == 1 and rises.size + 1 == 284
    assert np.median(np.diff(rises)) / d.sampling_frequency == pytest.approx(0.72, abs=1e-9)
    assert d.data.shape[0] / d.sampling_frequency == pytest.approx(284 * 0.72, abs=0.72)
