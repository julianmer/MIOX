####################################################################################################
#                                          test_audit.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Coverage audit on a synthetic NIfTI-MRS file: standard, other and unused fields, and a  #
#          nested block covered by the key read from it. No test data needed.                      #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import numpy as np
from nifti_mrs.create_nmrs import gen_nifti_mrs

# own
from miox.audit import Auditor, Status


def test_audit_statuses(tmp_path):
    image = gen_nifti_mrs(np.ones((1, 1, 1, 64), complex), 2.5e-4, 123.2, "1H", no_conj=True)
    image.add_hdr_field("EchoTime", 0.03)
    image.add_hdr_field("my_block", {"a": 1, "b": 2}, doc="test block")
    path = tmp_path / "svs.nii.gz"
    image.save(path)
    coverage = Auditor().audit(path)
    fields = {f.path: f for f in coverage.fields}
    assert fields["header_extension/EchoTime"].status == Status.STANDARD
    assert fields["nifti/pixdim"].key == "DwellTime"
    assert fields["header_extension/my_block/a"].status == Status.OTHER
    assert fields["header_extension/my_block/a"].key == "my_block"
    assert fields["nifti/descrip"].status == Status.UNUSED
    assert sum(coverage.summary().values()) == len(coverage.fields)
