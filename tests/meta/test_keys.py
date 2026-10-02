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
from miox.meta import KeyTable


def test_standard_keys_present():
    keys = KeyTable()
    assert keys.nifti_mrs_version >= (0, 11)
    assert "DIM_COIL" in keys.dim_tags
    for name in ("SpectrometerFrequency", "ResonantNucleus", "EchoTime", "SpectralWidth"):
        assert name in keys
    assert keys["EchoTime"].unit == "s"
    assert keys["SpectrometerFrequency"].origin == "nifti-mrs:required"


def test_sheet_conversion_is_consistent():
    keys = KeyTable()
    te = keys["EchoTime"]
    assert (te.sheet, te.sheet_unit, te.sheet_factor) == ("TE", "ms", 1e3)
    assert keys["FieldStrength"].sheet == "Bfield"
    for key in keys.values():
        if key.sheet_factor is None:
            assert key.sheet is None or key.origin == "miox"
