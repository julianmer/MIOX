####################################################################################################
#                                        test_sequences.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/10/26                                                                                #
#                                                                                                  #
# Purpose: Pulseq files against independent evidence: the OSI2ONE scans ship the sequence that    #
#          acquired them beside the ISMRMRD raw data, so the ADC events, dwell time, matrix and    #
#          field of view of the sequence must match the acquired readouts; the block durations     #
#          must add up to the declared total.                                                      #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import pytest

# own
from miox import PulseSequence, load
from tests.conftest import PUBLIC_DATA

OSI2ONE = sorted(set(p.parent for p in (PUBLIC_DATA / "osi2one").glob("**/sequence.seq")))


@pytest.mark.parametrize("folder", OSI2ONE[:6], ids=lambda p: p.name[:40])
def test_osi2one_sequence_matches_its_raw_data(folder):
    raw_file = folder / "ismrmrd_data.h5"
    if not raw_file.is_file():
        pytest.skip("no raw data beside the sequence")
    [seq] = load(folder / "sequence.seq")
    [raw] = load(raw_file)
    assert isinstance(seq, PulseSequence) and seq.version.startswith("1.4")
    adc_blocks = seq.blocks[seq.blocks["adc"] != 0]
    assert len(adc_blocks) == raw.n_readouts
    dwells = {seq.events["adc"][int(b)]["dwell"] * 1e-9 for b in adc_blocks["adc"]}
    assert any(dwell == pytest.approx(raw.meta["DwellTime"]) for dwell in dwells)
    matrix = seq.definitions["encoding_dim"]
    oversampling = seq.definitions.get("ro_oversampling", 1)
    assert [matrix[0] * oversampling, matrix[1], matrix[2]] == raw.meta["EncodedMatrixSize"]
    fov = [v * 1000 for v in seq.definitions["FOV"]]
    encoded = raw.meta["EncodedFieldOfView"]
    assert fov == pytest.approx([encoded[0] / oversampling] + encoded[1:])
    assert seq.duration == pytest.approx(seq.definitions["TotalDuration"], rel=1e-9)
    assert seq.meta["EchoTime"] == seq.definitions["TE"]
    assert "LABELSET" in seq.extensions["specifications"]
    gradients = set(seq.events["trap"]) | set(seq.events.get("gradients", {}))
    for channel in ("gx", "gy", "gz"):
        assert set(seq.blocks[channel].tolist()) - {0} <= gradients
