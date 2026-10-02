####################################################################################################
#                                          test_basis.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Basis sets: the NAA singlet of every set at 2.008 ppm (not mirrored, not shifted), and  #
#          the same spectra stored in different formats read as the same FIDs.                     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import numpy as np
import pytest

# own
from miox import BasisSet, load
from tests.conftest import PUBLIC_DATA

TOOLBOX = "basis_toolbox/sample_basis_sets/"
CHALLENGE = "mrs_fitting_challenge/"
OSPREY = "osprey/fit/basissets/"
SLASER = TOOLBOX + "LCModel/BASIS_file/SLASER_Siemens26.basis"

# (basis set, ppm at 0 Hz): most basis formats store no reference, each tool picks its own; None:
# the ReferencePPM the set states
NAA_SETS = [
    (SLASER, 4.65),
    ("lcmodel/test_lcm/3t.basis", 4.65),
    ("suspect/tests/test_data/lcmodel/press_30ms_3T.basis", 4.65),
    (CHALLENGE + "basisset_LCModel/press3T_30ms.BASIS", 4.65),
    (CHALLENGE + "basisset_LCModel", 4.65),
    (CHALLENGE + "basisset_JMRUI", 4.65),
    (OSPREY + "3T/siemens/unedited/special/8_5/basis_siemens_special8_5.basis", 4.65),
    (OSPREY + "7T/siemens/unedited/special/9/basis_siemens_special9.basis", 4.65),
    (OSPREY + "7T/siemens/unedited/steam/5/basis_siemens_steam5.BASIS", 4.65),
    # sets made by Osprey (named so in IDBASI) put 0 Hz at 4.68 ppm
    (OSPREY + "3T/siemens/unedited/press/35/3T_PRESS_Siemens_35ms_noMM.BASIS", 4.68),
    (OSPREY + "3T/philips/unedited/press/35/3T_PRESS_Philips_35ms_noMM.BASIS", 4.68),
    (OSPREY + "3T/ge/unedited/press/35/3T_PRESS_GE_35ms_noMM.BASIS", 4.68),
    (TOOLBOX + "LCModel/BASIS_file/LCModel_Siemens_UnEdited_sLASER_105_.BASIS", 4.68),
    (TOOLBOX + "FSL-MRS", 4.65),
    (TOOLBOX + "FSL-MRS/NAA.json", 4.65),
    (TOOLBOX + "JMRUI", 4.65),
    # the same spectra as SLASER; these state 4.675 (BasisRFOffset.txt, ppmCalib, centerFreq) but
    # the data keep SLASER's 4.65
    (TOOLBOX + "SPINWIZARD", 4.65),
    (TOOLBOX + "JET/from_marss_converted", 4.65),
    (TOOLBOX + "INSPECTOR/Basis_sLASER_26.mat", 4.65),
    (TOOLBOX + "MARSS/basis_functions_metab_mm_mat", 4.65),
    (TOOLBOX + "OSPREY/ospreybasis_marss_to_cows_26.mat", 4.65),
    (TOOLBOX + "FIDA", 4.65),
    (TOOLBOX + "PROFIT/profit_struct.mat", 4.65),
    (TOOLBOX + "JET/from_spinwizard_mouse", None),
    (TOOLBOX + "MRSCLOUD/BASIS_Siemens_UnEdited_sLASER_105_wMM.mat", None),
    ("profit/sampleBasisSet/sLASER_7ppm_moiety_UF_basis_TE24.mat", 4.70),  # not stored; ProFit's
] + [
    (path.relative_to(PUBLIC_DATA).as_posix(), None)
    for path in sorted((PUBLIC_DATA / OSPREY).rglob("*.mat"))
]

# (basis set, basis set, conjugate, names to skip): the same spectra in different formats
SAME_SPECTRA = [
    (SLASER, TOOLBOX + "FSL-MRS", False, ()),
    (SLASER, TOOLBOX + "JMRUI", False, ()),
    (SLASER, TOOLBOX + "SPINWIZARD", False, ()),
    (SLASER, TOOLBOX + "JET/from_marss_converted", False, ()),
    (SLASER, TOOLBOX + "NIFTI-MRS", True, ()),  # spec2nii 0.8.7 conjugated the .RAW it converted
    (SLASER, TOOLBOX + "INSPECTOR/Basis_sLASER_26.mat", False, ()),
    (SLASER, TOOLBOX + "MARSS/basis_functions_metab_mm_mat", False, ()),
    (SLASER, TOOLBOX + "OSPREY/ospreybasis_marss_to_cows_26.mat", False, ("MM09",)),  # Osprey-made
    (SLASER, TOOLBOX + "FIDA", False, ()),
    (SLASER, TOOLBOX + "PROFIT/profit_struct.mat", False, ()),
    (CHALLENGE + "basisset_LCModel/press3T_30ms.BASIS", CHALLENGE + "basisset_LCModel", False, ()),
    (CHALLENGE + "basisset_LCModel", CHALLENGE + "basisset_JMRUI", False, ()),
]


def basis(relpath: str) -> BasisSet:
    path = PUBLIC_DATA / relpath
    if not path.exists():
        pytest.skip("file not fetched")
    [b] = load(path)
    assert isinstance(b, BasisSet)
    return b


def naa_ppm(b: BasisSet, centre: float) -> float:
    """Position of the largest NAA peak between 1 and 3 ppm (first sub-spectrum if edited)."""
    fid = b.fid(next(m for m in b.metabolites if m.upper() in ("NAA", "NAA_AC")))  # ProFit: acetyl
    fid = fid.reshape(fid.shape[0], -1)[:, 0]
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(fid)))
    ppm = centre - np.fft.fftshift(np.fft.fftfreq(fid.size, b.dwelltime)) / b.spectrometer_frequency
    window = (ppm > 1) & (ppm < 3)
    return float(ppm[window][np.argmax(spectrum[window])])


@pytest.mark.data
@pytest.mark.parametrize("relpath,centre", NAA_SETS, ids=[r[0] for r in NAA_SETS])
def test_naa_at_2_008_ppm(relpath, centre):
    b = basis(relpath)
    centre = b.meta["ReferencePPM"] if centre is None else centre
    assert naa_ppm(b, centre) == pytest.approx(2.008, abs=0.015)


@pytest.mark.data
@pytest.mark.parametrize("first,second,conjugate,skip", SAME_SPECTRA)
def test_same_spectra_in_different_formats(first, second, conjugate, skip):
    a, b = basis(first), basis(second)
    shared = [m for m in a.metabolites if m in b.metabolites and m not in skip]
    assert len(shared) >= 10
    for m in shared:
        x, y = a.fid(m), b.fid(m)
        n = min(x.size, y.size)
        x, y = x[:n], np.conj(y[:n]) if conjugate else y[:n]
        assert abs(np.vdot(x, y)) / np.linalg.norm(x) / np.linalg.norm(y) > 0.999, m


@pytest.mark.data
def test_mrscloud_single_spectrum_is_its_basis_column():
    [single] = load(PUBLIC_DATA / TOOLBOX / "MRSCLOUD/Siemens_UnEdited_sLASER_TE105_NAA.mat")
    b = basis(TOOLBOX + "MRSCLOUD/BASIS_Siemens_UnEdited_sLASER_105_wMM.mat")
    assert single.spectrometer_frequency == pytest.approx(b.spectrometer_frequency)
    x, y = b.fid("NAA"), single.data.ravel()
    assert abs(np.vdot(x, y)) / np.linalg.norm(x) / np.linalg.norm(y) > 0.999
