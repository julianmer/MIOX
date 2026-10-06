####################################################################################################
#                                        test_bruker_raw.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/10/26                                                                                #
#                                                                                                  #
# Purpose: Bruker imaging k-space on synthetic scan folders: a ParaVision 6 fid with padded        #
#          blocks, two channels, an echo train and interleaved objects, and a ParaVision 360       #
#          rawdata.job0; the counters follow the encoding steps and the object order.             #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import numpy as np
import pytest

import miox as gu

POINTS, CHANNELS, OBJECTS, STEPS, ECHOES = 6, 2, 2, 8, 2  # 8 phase steps in trains of 2 echoes


def _jcamp(params: dict) -> str:
    lines = ["##TITLE=Parameter List, ParaVision 6.0.1", "##JCAMPDX=4.24"]
    for key, value in params.items():
        if isinstance(value, list):
            lines.append(f"##${key}=( {len(value)} )")
            lines.append(" ".join(str(v) for v in value))
        elif isinstance(value, str) and value.startswith("<"):
            lines.append(f"##${key}=( {len(value)} )")
            lines.append(value)
        else:
            lines.append(f"##${key}={value}")
    lines.append("##END=")
    return "\n".join(lines) + "\n"


def _scan(tmp_path, pv360: bool = False):
    scan = tmp_path / "7"
    scan.mkdir()
    steps1 = [-4, -2, 0, 2, -3, -1, 1, 3]  # two echoes per train, interleaved trains
    acqp = {
        "ACQ_sw_version": "<PV-360.3.5>" if pv360 else "<PV 6.0.1>",
        "PULPROG": "<RARE.ppg>", "ACQ_dim": 2, "ACQ_size": [2 * POINTS, STEPS],
        "ACQ_phase_factor": ECHOES, "NI": OBJECTS, "NR": 1, "ACQ_obj_order": [1, 0],
        "ACQ_n_echo_images": 1, "BYTORDA": "little", "BF1": 400.3, "NUCLEUS": "<1H>",
        "SW_h": 50000, "ACQ_flip_angle": 90, "ACQ_protocol_name": "<rare_qa>",
        "ACQ_method": "<Bruker:RARE>", "ACQ_station": "<BioSpec94/30>",
        "ACQ_institution": "<UPT>", "ACQ_time": "<2020-12-08T10:58:32,053+0100>",
        "ACQ_slice_sepn": [1.5],
    }
    if pv360:
        jobs = [2 * POINTS, 1, 1, 32, 1, 50000, 32, 1, "<job0>"]
        acqp.update({"ACQ_word_size": "_32_BIT", "ACQ_jobs": jobs})
    else:
        acqp["GO_raw_data_format"] = "GO_32BIT_SGN_INT"
        acqp["GO_block_size"] = "Standard_KBlock_Format"
    method = {
        "Method": "<Bruker:RARE>", "PVM_EchoTime": 11, "PVM_RepetitionTime": 2500,
        "PVM_EncNReceivers": CHANNELS, "PVM_EncMatrix": [POINTS, STEPS], "PVM_Matrix": [8, 8],
        "PVM_EncSteps1": steps1, "PVM_EncSteps2": [0], "PVM_RareFactor": ECHOES,
        "PVM_Fov": [40, 40], "PVM_SliceThick": 1, "PVM_NAverages": 1, "PVM_EffSWh": 50000,
    }
    (scan / "acqp").write_text(_jcamp(acqp))
    (scan / "method").write_text(_jcamp(method))
    blocks = OBJECTS * STEPS
    values = np.zeros((blocks, CHANNELS, POINTS, 2), dtype="<i4")
    for b in range(blocks):
        for c in range(CHANNELS):
            values[b, c, :, 0] = 1000 * b + 100 * c + np.arange(POINTS)
            values[b, c, :, 1] = -b
    per_block = CHANNELS * POINTS * 2  # 24 values = 96 bytes
    if pv360:
        raw = values.reshape(blocks, -1)
        name = "rawdata.job0"
    else:
        padded = 1024 // 4  # 256 values per block
        raw = np.zeros((blocks, padded), dtype="<i4")
        raw[:, :per_block] = values.reshape(blocks, -1)
        name = "fid"
    (scan / name).write_bytes(raw.tobytes())
    return scan / name, steps1


@pytest.mark.parametrize("pv360", [False, True])
def test_bruker_imaging_raw(tmp_path, pv360):
    path, steps1 = _scan(tmp_path, pv360)
    sniff = gu.sniff(path)
    assert sniff.format == gu.Format.BRUKER and sniff.kind == gu.Kind.RAW
    [raw] = gu.load(path)
    assert isinstance(raw, gu.RawData) and raw.shape == (OBJECTS * STEPS, CHANNELS, POINTS)
    assert raw.data[5, 1, 2] == complex(5000 + 100 + 2, -5)
    assert raw.headers["scan_counter"].tolist() == list(range(16))
    # block b: echo b % 2, object (b // 2) % 2, train b // 4; line = echo + 2 * train
    assert raw.counters("segment").tolist() == [0, 1] * 8
    assert raw.headers["idx"]["user"][:, 0].tolist() == [0, 0, 1, 1] * 4  # acquired object
    assert raw.counters("slice").tolist() == [1, 1, 0, 0] * 4  # geometric slice by ACQ_obj_order
    expected = [steps1[(b % 2) + 2 * (b // 4)] + 4 for b in range(16)]
    assert raw.counters("kspace_encode_step_1").tolist() == expected
    assert raw.headers["center_sample"][0] == POINTS // 2
    assert raw.headers["active_channels"][0] == 2
    m = raw.meta
    assert m["Manufacturer"] == "Bruker" and m["EchoTime"] == pytest.approx(0.011)
    assert m["RepetitionTime"] == 2.5 and m["DwellTime"] == pytest.approx(2e-5)
    assert m["EncodedMatrixSize"] == [POINTS, STEPS, 1] and m["ReconMatrixSize"] == [8, 8, 1]
    assert m["EncodedFieldOfView"] == [40.0, 40.0, 1.0] and m["SpacingBetweenSlices"] == 1.5
    assert m["EchoTrainLength"] == ECHOES and m["ProtocolName"] == "rare_qa"
    assert m["PulseSequenceName"] == "RARE.ppg"
    assert m["AcquisitionDateTime"] == "2020-12-08T10:58:32.053+0100"
    assert m["SpectrometerFrequency"] == [400.3] and m["ResonantNucleus"] == ["1H"]
    limits = m["EncodingLimits"]["kspace_encoding_step_1"]
    assert limits == {"minimum": 0, "maximum": 7, "center": 4}
    assert m.vendor["acqp"]["NI"] == OBJECTS and m.fields["EchoTime"] == "method/PVM_EchoTime"
    assert raw.encoding["object_order"] == [1, 0] and raw.encoding["echoes_per_train"] == ECHOES
    assert raw.name == f"7/{path.name}"
    assert raw.provenance.notes == ("no slice geometry: no pdata/1/visu_pars beside the raw file",)
    assert not raw.headers["position"].any()
