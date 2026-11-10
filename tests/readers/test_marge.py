####################################################################################################
#                                          test_marge.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 10/11/26                                                                                #
#                                                                                                  #
# Purpose: MaRGE acquisitions: a Localizer and a RARE .mat as k-space readouts with the stored     #
#          reconstruction, a Noise .mat as one noise readout, the parameter .csv on its own, and   #
#          the sniff of synthetic files in MaRGE's folder layout.                                  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import numpy as np
import pytest
from scipy.io import savemat

# own
import miox
from miox.cli import CLI
from miox.quicklook import page
from miox.viewer import ImageView, RawView, view_class

LOCALIZER = "marge/FENCE_InVivoSNR/FENCE/Spectrum/Localizer.972.mat"
RARE = "marge/FENCE_Phantom_OSIIcoil/NoShield/Imaging/Grounding_RARE.2025.11.07.15.38.47.924.mat"
NOISE = "marge/FENCE_InVivoSNR/FENCE/NoiseFactor/Noise.673.mat"


def _need(public_data, relpath):
    path = public_data / relpath
    if not path.is_file():
        pytest.skip("MaRGE samples absent")
    return path


@pytest.mark.data
def test_localizer_readouts_and_stored_image(public_data):
    path = _need(public_data, LOCALIZER)
    s = miox.sniff(path)
    assert (s.format, s.kind, s.vendor) == (miox.Format.MARGE_MAT, miox.Kind.RAW, "marge")
    assert [c.name for c in s.companions] == ["Localizer.972.csv"]
    raw, image = miox.load(path)
    assert isinstance(raw, miox.RawData) and raw.shape == (1000, 1, 100)
    assert np.array_equal(raw.counters("kspace_encode_step_1")[:3], [0, 1, 2])
    assert np.array_equal(raw.counters("kspace_encode_step_2")[[0, 50, 999]], [0, 1, 19])
    assert raw.meta["EncodedMatrixSize"] == [100, 50, 20]
    assert raw.meta["EncodedFieldOfView"] == [200.0, 200.0, 200.0]
    assert raw.meta["EchoTime"] == pytest.approx(0.020)  # the centre line in the first echo
    assert raw.meta["RepetitionTime"] == 0.3 and raw.meta["EchoTrainLength"] == 5
    assert raw.meta["DwellTime"] == pytest.approx(1 / 0.04995121951219512e6)
    assert raw.meta["SpectrometerFrequency"] == [pytest.approx(2.11048227487121)]
    assert raw.meta["AcquisitionDateTime"] == "2025-11-09T15:49:18.972"
    assert raw.meta.vendor["parameter_names"]["nScans"] == "Number of scans"
    assert raw.meta.vendor["overData"] == "<array complex128 (720000,)>"
    assert raw.headers["read_dir"][0].tolist() == [0.0, 0.0, 1.0]  # axesOrientation [2, 1, 0]
    assert isinstance(image, miox.MRImage) and image.data.shape == (20, 50, 100)
    assert image.voxel_sizes == (10.0, 4.0, 2.0)
    assert image.name == "Localizer.972/image3D" and image.meta["ImageType"] == ["COMPLEX"]
    assert "1000 readouts" in CLI()._describe(raw)
    assert view_class(raw) is RawView and view_class(image) is ImageView
    assert "stored in the file" in page([raw, image], path.name)


@pytest.mark.data
def test_rare_with_a_shifted_field_of_view(public_data):
    path = _need(public_data, RARE)
    raw, image = miox.load(path)
    assert raw.shape == (2560, 1, 80) and raw.meta["ScanDuration"] == pytest.approx(102.4)
    assert raw.headers["position"][0].tolist() == pytest.approx([0.0, -6.0, 7.0])  # dfov in mm
    assert image.data.shape == (32, 80, 80) and image.voxel_sizes == (5.0, 2.0, 2.0)
    assert image.affine[:3, 3].tolist() == pytest.approx([-80.0, -86.0, -73.0])


@pytest.mark.data
def test_noise_readout(public_data):
    path = _need(public_data, NOISE)
    [raw] = miox.load(path)
    assert raw.shape == (1, 1, 2500) and raw.flags_of(0) == ["IS_NOISE_MEASUREMENT"]
    assert raw.meta["DwellTime"] == pytest.approx(20e-6)  # bw 50 kHz
    assert raw.meta["SequenceName"] == "Noise" and raw.meta["RepetitionTime"] == 0.5
    assert raw.meta.vendor["RMS noise"] == pytest.approx(0.3885298)
    assert RawView(raw) is not None and "1 readouts" in page([raw], path.name)


@pytest.mark.data
def test_parameter_table_on_its_own(public_data):
    path = _need(public_data, RARE).with_suffix(".csv")
    s = miox.sniff(path)
    assert (s.format, s.kind) == (miox.Format.MARGE_CSV, miox.Kind.PROTOCOL)
    [protocol] = miox.load(path)
    assert isinstance(protocol, miox.Protocol) and protocol.name == "RARE"
    assert protocol.meta["EncodedMatrixSize"] == [80, 80, 32]
    assert protocol.meta["EncodedFieldOfView"] == [160.0, 160.0, 160.0]
    assert protocol.meta.vendor["parameters"]["shimming"] == [105.0, 15.0, 55.0]
    assert protocol.meta.vendor["parameter_names"]["fov"] == "FOV[x,y,z] (cm)"
    assert "EchoTime" not in protocol.meta  # no sweepOrder in the table


def test_sniff_of_marge_folder_layout(tmp_path):
    """MaRGE saves <name>.mat under mat/ and <name>.csv under csv/."""
    (tmp_path / "mat").mkdir()
    (tmp_path / "csv").mkdir()
    mat = tmp_path / "mat" / "Noise.2025.01.02.03.04.05.006.mat"
    savemat(mat, {"seqName": "Noise", "name_string": "2025.01.02.03.04.05.006", "bw": 50.0,
                  "nPoints": 8, "data": np.arange(8) * (1 + 1j)})
    table = tmp_path / "csv" / "Noise.2025.01.02.03.04.05.006.csv"
    table.write_text("seqName,nPoints,bw\nNoiseInfo,Number of points,Bandwidth (kHz)\n"
                     "Noise,8,50.0\n")
    s = miox.sniff(mat)
    assert s.format == miox.Format.MARGE_MAT and s.companions == (table,)
    [raw] = miox.load(mat)
    assert raw.shape == (1, 1, 8) and raw.meta.vendor["parameter_names"]["bw"] == "Bandwidth (kHz)"
    assert np.array_equal(raw.data[0, 0], np.arange(8) * (1 + 1j))
    assert miox.sniff(table).format == miox.Format.MARGE_CSV
    [protocol] = miox.load(table)
    assert protocol.meta["DwellTime"] == pytest.approx(20e-6) and protocol.name == "Noise"
