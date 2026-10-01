####################################################################################################
#                                           test_keys.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 01/10/26                                                                                #
#                                                                                                  #
# Purpose: Key table tests: the standard keys are present with their units and the sheet           #
#          conversions are consistent.                                                             #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#

# own
from miox.meta import KEYS
from miox.meta.keys import DIM_TAGS, NIFTI_MRS_VERSION


def test_standard_keys_present():
    assert NIFTI_MRS_VERSION >= (0, 11)
    assert "DIM_COIL" in DIM_TAGS
    for name in ("SpectrometerFrequency", "ResonantNucleus", "EchoTime", "SpectralWidth"):
        assert name in KEYS
    assert KEYS["EchoTime"].unit == "s"
    assert KEYS["SpectrometerFrequency"].origin == "nifti-mrs:required"


def test_sheet_conversion_is_consistent():
    te = KEYS["EchoTime"]
    assert (te.sheet, te.sheet_unit, te.sheet_factor) == ("TE", "ms", 1e3)
    assert KEYS["FieldStrength"].sheet == "Bfield"
    for key in KEYS.values():
        if key.sheet_factor is None:
            assert key.sheet is None or key.origin == "miox"
