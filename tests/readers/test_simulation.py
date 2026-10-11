####################################################################################################
#                                        test_simulation.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/10/26                                                                                #
#                                                                                                  #
# Purpose: Simulation readers on synthetic files: a KomaMRI .phantom with motion and a spin list,  #
#          a 3D JEMRIS phantom grid with offset, a JEMRIS signal of two channels with readouts of  #
#          unequal length, an MRiLab phantom, a JEMRIS motion trajectory (units to s and m).       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import h5py
import numpy as np
import pytest
import scipy.io

# own
import miox


def test_koma_phantom_with_motion(tmp_path):
    path = tmp_path / "ring.phantom"
    with h5py.File(path, "w") as fh:
        fh.attrs.update({"Name": "ring", "Ns": 3, "Dims": 2, "Version": "0.9.0"})
        for c, values in zip("xyz", ([0.0, 0.01, 0.02], [0.0, -0.01, 0.01], [0.0] * 3),
                             strict=True):
            fh[f"position/{c}"] = values
        for k, values in (("ρ", [1.0, 0.5, 0.2]), ("T1", [1.0] * 3), ("T2", [0.1] * 3),
                          ("T2s", [0.05] * 3), ("Δw", [0.0, 10.0, -10.0])):
            fh[f"contrast/{k}"] = values
        fh.create_group("motion/motion_1/action").attrs.update({"type": "Translate", "dy": 0.02})
        fh.create_group("motion/motion_1/time").attrs.update(
            {"type": "TimeRange", "t_start": 0.0, "t_end": 0.5})
        fh.create_group("motion/motion_1/spins").attrs["type"] = "AllSpins"
    sniff = miox.sniff(path)
    assert (sniff.format, sniff.kind) == (miox.Format.KOMA_PHANTOM, miox.Kind.PHANTOM)
    [phantom] = miox.load(path)
    assert isinstance(phantom, miox.Phantom) and phantom.name == "ring"
    assert phantom.positions[1].tolist() == [0.01, -0.01, 0.0]
    assert phantom.properties["rho"].tolist() == [1.0, 0.5, 0.2]
    assert phantom.properties["dw"].tolist() == [0.0, 10.0, -10.0]
    assert phantom.units["dw"] == "rad/s" and phantom.grid is None
    [motion] = phantom.motion
    assert motion["action"] == {"type": "Translate", "dy": 0.02}
    assert motion["time"]["t_end"] == 0.5


def test_koma_spin_list(tmp_path):
    path = tmp_path / "spins.h5"
    with h5py.File(path, "w") as fh:
        for k, values in (("x", [0.0, 0.1]), ("y", [0.0, 0.0]), ("z", [0.0, 0.0]),
                          ("ρ", [1.0, 1.0]), ("T1", [0.5, 0.6])):
            fh[k] = values
    [phantom] = miox.load(path)
    assert phantom.n_spins == 2 and list(phantom.properties) == ["T1", "rho"]


def test_jemris_grid(tmp_path):
    path = tmp_path / "sample.h5"
    data = np.zeros((5, 3, 2, 2))  # (values, x, y, z) as JEMRIS writes it
    data[:, 2, 1, 0] = [1.0, 0.001, 0.01, 0.0, 5.0]
    data[:, 0, 0, 1] = [0.5, 0.002, 0.02, 0.04, 0.0]
    with h5py.File(path, "w") as fh:
        fh["sample/data"] = data.T
        fh["sample/resolution"] = [[2.0], [1.0], [4.0]]
        fh["sample/offset"] = [[10.0], [0.0], [-2.0]]
    [phantom] = miox.load(path)
    assert phantom.n_spins == 2
    assert phantom.grid["shape"] == [3, 2, 2]
    assert phantom.grid["indices"].tolist() == [[0, 0, 1], [2, 1, 0]]
    assert np.allclose(phantom.positions, [[0.008, -0.0005, 0.0], [0.012, 0.0005, -0.004]])
    assert phantom.properties["T1"].tolist() == pytest.approx([0.5, 1.0])
    assert phantom.properties["T2s"].tolist() == [0.025, np.inf]
    assert phantom.properties["DB"].tolist() == [0.0, 5.0]


def test_jemris_signal(tmp_path):
    path = tmp_path / "signals.h5"
    times = np.concatenate([np.arange(4) * 0.01 + 1.0, np.arange(3) * 0.01 + 2.0])  # ms
    values = np.arange(7 * 3, dtype=np.float64).reshape(7, 3)
    with h5py.File(path, "w") as fh:
        fh["signal/times"] = times
        fh["signal/channels/00"] = values
        fh["signal/channels/01"] = -values
    sniff = miox.sniff(path)
    assert (sniff.format, sniff.kind) == (miox.Format.JEMRIS_SIGNAL, miox.Kind.RAW)
    [raw] = miox.load(path)
    assert raw.data.shape == (2, 2, 4) and raw.samples.tolist() == [4, 3]
    assert raw.data[1, 0, :3].tolist() == [12 + 13j, 15 + 16j, 18 + 19j]
    assert raw.data[1, 1, 0] == -12 - 13j and raw.data[1, 0, 3] == 0
    mz, time = raw.waveforms
    assert mz["data"][0, 0].tolist() == [2.0, 5.0, 8.0, 11.0]
    assert time["data"][1, :3].tolist() == pytest.approx([2e-3, 2.01e-3, 2.02e-3])
    assert raw.meta["DwellTime"] == 1e-5


def test_mrilab_phantom(tmp_path):
    rho = np.zeros((4, 2, 2))
    rho[3, 1, 0] = 0.8
    vobj = {"Rho": rho, "T1": rho * 2, "T2": rho * 0.1, "T2Star": rho * 0.05, "Mz": rho,
            "XDimRes": 0.002, "YDimRes": 0.001, "ZDimRes": 0.003, "XDim": 2.0, "YDim": 4.0,
            "ZDim": 2.0, "Gyro": 267538030.38, "ChemShift": 0.0, "Type": "Water"}
    scipy.io.savemat(tmp_path / "obj.mat", {"VObj": vobj})
    sniff = miox.sniff(tmp_path / "obj.mat")
    assert (sniff.format, sniff.kind) == (miox.Format.MRILAB_PHANTOM, miox.Kind.PHANTOM)
    [phantom] = miox.load(tmp_path / "obj.mat")
    assert phantom.n_spins == 1
    assert np.allclose(phantom.positions, [[(3 - 2) * 0.002, (1 - 1) * 0.001, (0 - 1) * 0.003]])
    assert phantom.properties["T1"].tolist() == pytest.approx([1.6])
    assert phantom.meta.vendor["VObj"]["Type"] == "Water"


def test_jemris_motion_trajectory(tmp_path):
    path = tmp_path / "motion.dat"
    path.write_text("1 2 3\n" + "".join(
        f"{t} {t / 10} 0 0 0 0 {t}\n" for t in (0.0, 10.0, 20.0)))
    sniff = miox.sniff(path)
    assert (sniff.format, sniff.kind) == (miox.Format.JEMRIS_MOTION, miox.Kind.PHANTOM)
    [phantom] = miox.load(path)
    motion = phantom.motion[0]
    assert phantom.n_spins == 0
    assert np.allclose(motion["origin"], [1e-3, 2e-3, 3e-3])
    assert np.allclose(motion["time"], [0.0, 0.01, 0.02])
    assert np.allclose(motion["translation"][:, 0], [0.0, 1e-3, 2e-3])
    assert np.allclose(motion["rotation"][:, 2], [0.0, 10.0, 20.0])  # degrees as written
