####################################################################################################
#                                       test_orientation.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Spectrum orientation per reader: in data stored as NIfTI-MRS prescribes, NAA (2.0 ppm)  #
#          and creatine (3.0 ppm) must appear at their own positions, not mirrored about water.    #
#          A reader that conjugates when it should not (or forgets to) fails here.                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import numpy as np
import pytest

# own
from miox import MRSData, load
from tests.conftest import PUBLIC_DATA

# one dataset with clear metabolite signal per format: (path, dataset name)
REFERENCES = [
    ("nifti_mrs_examples/example_07.nii.gz", "example_07"),
    (
        "spec2nii_test_data/Siemens/special_cases_slaser_dkd/twix/"
        "meas_MID00053_FID16506_svs_slaser_dkd_von_wrsoff.dat",
        "meas_MID00053_FID16506_svs_slaser_dkd_von_wrsoff",
    ),
    (
        "spec2nii_test_data/Siemens/special_cases_slaser_dkd/DICOM/svs_slaser_dkd_von_wrs2_17_None/"
        "UNKNOWN.1.3.12.2.1107.5.2.43.66050.2024041811441798283502419",
        "svs_slaser_dkd_von_wrs2",
    ),
    (
        "spec2nii_test_data/philips/DICOM/SV_phantom_center/IM-0018-0002-0001.dcm",
        "SV_phantom centro 16NSA",
    ),
    (
        "spec2nii_test_data/UIH/mrs_data/dicom/svs_press_te144_SVS_801/00000001.dcm",
        "svs_press_te144_SVS",
    ),
    ("spec2nii_test_data/Siemens/rda/latin1.rda", "latin1"),
    ("spec2nii_test_data/philips/P3/SV_PRESS_sh_4_2_raw_act.SDAT", "SV_PRESS_sh_4_2_raw_act"),
    ("spec2nii_test_data/philips/hyper/raw_226.list", "raw_226_hyper_edited"),
    (
        "spec2nii_test_data/ge/pFiles/sLASER/sub-01_ses-01_acq-slaser_svs_noID.7",
        "sub-01_ses-01_acq-slaser_svs_noID",
    ),
    ("spec2nii_test_data/bruker/PV-6.0.1/9/pdata/2/2dseq", "2DSEQ_9_2_lego_rod_3"),
    ("spec2nii_test_data/jmrui/metab_jmrui.txt", "metab_jmrui"),
    ("spec2nii_test_data/jmrui/press2.mrui", "press2"),
    ("spec2nii_test_data/other/metab.RAW", "metab"),
    ("lcmodel/test_lcm/data.raw", "data"),
    ("tarquin/data/phillips/svs_30_ws.dpt", "svs_30_ws"),
    (
        "vespa_io/docs/user_manuals/files/tutorial_data_analysis/tutorial_analysis/"
        "tutorial_01_import-file/press_cp0.rsp",
        "press_cp0",
    ),
    (
        "vespa_io/docs/user_manuals/files/tutorial_data_analysis/tutorial_analysis/"
        "tutorial_03_open-multiple-datasets/press_cp0.xml",
        "press_cp0",
    ),
]


def orientation(d: MRSData) -> float:
    """Signal at NAA and creatine over signal at their mirror positions about 4.65 ppm."""
    n = d.shape[3]
    fid = np.moveaxis(d.data, 3, 0).reshape(n, -1)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(fid, axis=0), axes=0)).sum(axis=1)
    ppm = 4.65 - np.fft.fftshift(np.fft.fftfreq(n, d.dwelltime)) / d.spectrometer_frequency

    def peak(low: float, high: float) -> float:
        return float(spectrum[(ppm > low) & (ppm < high)].max())

    own = peak(1.9, 2.1) + peak(2.9, 3.1)
    mirrored = peak(9.3 - 2.1, 9.3 - 1.9) + peak(9.3 - 3.1, 9.3 - 2.9)
    return own / mirrored


@pytest.mark.data
@pytest.mark.parametrize("relpath,name", REFERENCES, ids=[r[1] for r in REFERENCES])
def test_metabolites_are_not_mirrored(relpath, name):
    path = PUBLIC_DATA / relpath
    if not path.exists():
        pytest.skip("file not fetched")
    [d] = [d for d in load(path) if d.name == name]
    assert orientation(d) > 1.5
