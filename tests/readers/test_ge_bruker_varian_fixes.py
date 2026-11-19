####################################################################################################
#                                 test_ge_bruker_varian_fixes.py                                   #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 19/11/26                                                                                #
#                                                                                                  #
# Purpose: Edge cases of the GE P-file, Varian, Bruker spectroscopy, jMRUI and LCModel .RAW        #
#          readers on synthetic files and modified copies of public samples: header layouts        #
#          without a frame count, stopped acquisitions, arrayed experiments, methods without       #
#          localisation, values the files do not state, nuclei by gyromagnetic ratio, side files.  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re
import shutil
import struct
from pathlib import Path

import nmrglue as ng
import numpy as np
import pytest

# own
import miox
from miox.readers.base import Reader, ReaderError
from miox.readers.bruker_mrs import BrukerMrs
from miox.readers.ge_pfile_mrs import GePfileMrs
from miox.readers.jmrui import JmruiReader
from miox.readers.lcmodel_raw import LcmodelRawReader


#**************************************************************************************************#
#                                             helpers                                              #
#**************************************************************************************************#
def _sample(public_data: Path, name: str) -> Path:
    path = public_data / name
    if not path.exists():
        pytest.skip(f"{name} not fetched")
    return path


def _bruker_scan(public_data: Path, tmp_path: Path, acqp=(), method=()) -> Path:
    """A copy of the PV 6.0.1 STEAM scan (acqp, method, fid) with (pattern, replacement) edits;
    the fid of the copy."""
    source = _sample(public_data, "spec2nii_test_data/bruker/PV-6.0.1/10")
    scan = tmp_path / "10"
    scan.mkdir()
    shutil.copy(source / "fid", scan / "fid")
    for name, edits in (("acqp", acqp), ("method", method)):
        text = (source / name).read_text()
        for pattern, replacement in edits:
            text, count = re.subn(pattern, replacement, text, flags=re.S)
            assert count, pattern
        (scan / name).write_text(text)
    return scan / "fid"


def _without(parameter: str) -> tuple[str, str]:
    """The edit that takes the JCAMP-DX parameters starting with ``parameter`` out of a file."""
    return rf"##\${parameter}\w*=.*?(?=\n##|\n\$\$)\n?", ""


def _spectrometer(folder: Path, data: np.ndarray, **parameters: list[str]) -> Path:
    """A Varian spectrometer .fid folder of ``data`` (rows, points) written by nmrglue."""
    udic = ng.fileiobase.create_blank_udic(data.ndim)
    for axis, size in enumerate(data.shape):
        udic[axis].update(size=size, complex=True)
    dic = ng.varian.create_dic(udic)
    values = {"apptype": ["std1D"], "tn": ["H1"], "sfrq": ["599.83"], "sw": ["12000"],
              "rfl": ["2000"], "rfp": ["1000"], "seqfil": ["s2pul"], **parameters}
    for name, value in values.items():
        param = ng.varian.create_pdic_param(name, value)
        if not value[0].replace(".", "").isdigit():
            param["basictype"] = "2"
        dic["procpar"][name] = param
    ng.varian.write(str(folder), dic, data.astype(np.complex64), overwrite=True)
    return folder


def _raw(path: Path, header: str = "") -> Path:
    """An LCModel .RAW file of four points, ``header`` before its $NMID namelist."""
    points = "".join(f"{k + 1:15.6E}{-k:15.6E}\n" for k in range(4))
    path.write_text(f"{header} $NMID\n id='x', fmtdat='(2E15.6)'\n volume=1\n tramp=1\n $END\n"
                    + points)
    return path


#**************************************************************************************************#
#                                            GE P-file                                             #
#**************************************************************************************************#
def test_pfile_raw_without_a_stated_frame_count(tmp_path):
    # the layout of revision 20.006 has no rh_nframes: no frames can be laid out
    path = tmp_path / "P00001.7"
    raw = bytearray(200000)
    struct.pack_into("<f", raw, 0, 20.006)
    path.write_bytes(bytes(raw))
    assert miox.sniff(path).format == miox.Format.GE_PFILE
    with pytest.raises(ReaderError, match="frame count"):
        miox.load(path)


@pytest.mark.parametrize("frames", [250, 262, 265])
def test_isthmus_of_a_stopped_scan(frames):
    # after the noise readout: blocks of a reference and 32 transients, the first block short TE
    reading = object.__new__(GePfileMrs)
    reading.values = lambda suppressed: {}
    raw = np.arange(1 + frames, dtype=complex).reshape(1, 1, 1, 1, -1, 1) * np.ones((1, 2))
    edited, short, ref_edited, ref_short = reading._isthmus(raw)
    frame = np.arange(1, 1 + frames)  # what each kept frame holds
    hercules = [f for f in frame[33:] if (f - 1) % 33]
    sets = len(hercules) // 4
    assert edited.data.shape == (1, 1, 1, 1, sets, 2, 4)
    assert np.array_equal(edited.data[0, 0, 0, 0, :, 0, :].real,
                          np.reshape(hercules[: sets * 4], (sets, 4)))
    cut = len(hercules) % 4
    assert edited.notes == ([f"80 ms HERCULES: {len(hercules)} averages cut to {sets * 4}"]
                            if cut else [])
    assert short.data.shape[4] == 32 and short.data[0, 0, 0, 0, 0, 0] == 2
    assert ref_edited.data[0, 0, 0, 0, :, 0].real.tolist() == frame[0::66].tolist()
    assert ref_short.data[0, 0, 0, 0, :, 0].real.tolist() == frame[33::66].tolist()


#**************************************************************************************************#
#                                              Varian                                              #
#**************************************************************************************************#
def test_varian_console_stopped_acquisition(public_data, tmp_path):
    source = _sample(public_data, "spec2nii_test_data/varian/13C_spuls_01.fid")
    [full] = miox.load(source)
    folder = tmp_path / "stopped.fid"
    folder.mkdir()
    shutil.copy(source / "procpar", folder / "procpar")
    raw = (source / "fid").read_bytes()
    block = struct.unpack(">i", raw[20:24])[0]
    (folder / "fid").write_bytes(raw[: 32 + 50 * block + 100])  # 50 blocks and a part of one
    [d] = miox.load(folder)
    assert full.shape == (1, 1, 1, 2048, 1, 80) and d.shape == (1, 1, 1, 2048, 1, 50)
    assert np.array_equal(d.data, full.data[..., :50])
    assert "50 transients in the file, arraydim states 80" in d.provenance.notes
    assert d.dim_tags[:2] == ("DIM_COIL", "DIM_DYN")


def test_varian_arrayed_1d_experiment(tmp_path):
    data = np.arange(3 * 16).reshape(3, 16) * (1 - 1j)
    pulses = {"array": ["pw"], "pw": ["5", "10", "15"], "arraydim": ["3"]}
    [d] = miox.load(_spectrometer(tmp_path / "arrayed.fid", data, **pulses))
    assert d.shape == (1, 1, 1, 16, 3) and np.array_equal(d.data[0, 0, 0], data.T)
    assert d.dim_tags[0] == "DIM_DYN" and "pw" in d.dim_info[5]["info"]
    assert not any("indirect" in note for note in d.provenance.notes)
    # an increment count of 1 does not make the array an indirect dimension
    single = dict(pulses, ni=["1"], sw1=["4000"], phase=["1"])
    [d] = miox.load(_spectrometer(tmp_path / "increment.fid", data, **single))
    assert d.shape[:5] == (1, 1, 1, 16, 3) and d.dim_tags[0] == "DIM_DYN"
    # several increments are one
    indirect = {"ni": ["3"], "sw1": ["4000"], "phase": ["1"], "array": [""]}
    [d] = miox.load(_spectrometer(tmp_path / "indirect.fid", data, **indirect))
    assert d.dim_tags[0] == "DIM_INDIRECT_0" and d.dim_info[5]["info"].startswith("ni 3 x")


#**************************************************************************************************#
#                                              Bruker                                              #
#**************************************************************************************************#
def test_bruker_press_fid_of_paravision_6(public_data, tmp_path):
    [steam] = miox.load(_sample(public_data, "spec2nii_test_data/bruker/PV-6.0.1/10") / "fid")
    fid = _bruker_scan(public_data, tmp_path, acqp=[(r"<STEAM\.ppg>", "<PRESS.ppg>")])
    [d] = miox.load(fid)
    assert d.shape == steam.shape == (1, 1, 1, 1980) and np.array_equal(d.data, steam.data)


def test_bruker_method_without_localisation(public_data, tmp_path):
    fid = _bruker_scan(public_data, tmp_path,
                       method=[_without("PVM_VoxArr"), _without("PVM_EchoTime")])
    [d] = miox.load(fid)
    assert d.shape == (1, 1, 1, 1980) and "EchoTime" not in d.meta
    assert d.meta["RepetitionTime"] == 2.5
    assert np.array_equal(d.affine, Reader.NO_GEOMETRY)
    assert any("no voxel" in note for note in d.provenance.notes)


def test_bruker_values_the_files_do_not_state(public_data, tmp_path):
    fid = _bruker_scan(public_data, tmp_path, acqp=[_without("NUCLEUS")],
                       method=[_without("PVM_FrqWorkOffsetPpm")])
    [d] = miox.load(fid)
    assert d.shape == (1, 1, 1, 1980)
    assert "ResonantNucleus" not in d.meta and "TxOffset" not in d.meta
    fields = ("acqp/BF1", "acqp/NUCLEUS", "acqp/ACQ_sw_version", "method/Method")
    values = BrukerMrs(fid, {})._values(None, None, None, None, fields)
    for key in ("SpectrometerFrequency", "ResonantNucleus", "TxOffset", "SoftwareVersions",
                "SequenceName"):
        assert key not in values
    assert all(value is not None for value, _ in values.values())
    stated = BrukerMrs(fid, {})._values(400.3, "<1H>", "<PV 6.0.1>", "<Bruker:STEAM>", fields)
    assert stated["ResonantNucleus"] == (["1H"], "acqp/NUCLEUS")
    assert stated["SoftwareVersions"][0] == "<PV 6.0.1>"


#**************************************************************************************************#
#                                              jMRUI                                               #
#**************************************************************************************************#
def test_jmrui_nucleus_from_the_gyromagnetic_ratio(public_data, tmp_path):
    reader, notes = JmruiReader(), []
    for nucleus, megahertz in (("13C", 32.125), ("23Na", 33.786), ("31P", 51.705),
                               ("2H", 19.608), ("1H", 127.731), ("129Xe", 35.331)):
        assert reader._nucleus(None, megahertz, 3.0, notes) == (nucleus, "MagneticField")
        assert reader._nucleus(-1.0, megahertz * 1e6, 3.0, notes) == (nucleus, "MagneticField")
    assert notes == []
    assert reader._nucleus(5.0, 33.786, 3.0, notes) == ("23Na", "TypeOfNucleus")
    assert reader._nucleus(6.0, 32.125, 3.0, notes) == ("13C", "MagneticField")  # no such index
    # a ratio between 13C and 23Na is neither
    assert reader._nucleus(None, 33.0, 3.0, notes)[0] == "1H" and len(notes) == 1
    source = _sample(public_data, "spec2nii_test_data/jmrui/P31rest_proc_jmrui.txt")
    text = source.read_text()
    assert "TypeOfNucleus: 2E0" in text
    path = tmp_path / source.name
    path.write_text(text.replace("TypeOfNucleus: 2E0", "TypeOfNucleus: 9E0"))
    [d] = miox.load(path)
    assert d.meta["ResonantNucleus"] == ["31P"]  # 51.707 MHz at 3 T


#**************************************************************************************************#
#                                           LCModel .RAW                                           #
#**************************************************************************************************#
def test_lcmodel_raw_side_file_belongs_to_the_raw(tmp_path):
    control = " $LCMODL\n hzpppm={}\n deltat={}\n filraw='{}'\n $END\n"
    raw = _raw(tmp_path / "scan.RAW")
    (tmp_path / "another.control").write_text(control.format(63.9, 1e-3, "another.RAW"))
    reader = LcmodelRawReader()
    assert reader._side_file(raw) is None
    [d] = miox.load(raw)
    assert d.shape == (1, 1, 1, 4) and "SpectrometerFrequency" not in d.meta
    assert d.dwelltime is None
    # the control file that names the .RAW, whatever its own name
    (tmp_path / "run7.control").write_text(control.format(123.2, 5e-4, "/data/scan.RAW"))
    assert reader._side_file(raw) == tmp_path / "run7.control"
    # a file of the same stem comes with the .RAW; a stem that only starts alike does not
    (tmp_path / "scan2.control").write_text(control.format(400.0, 1e-4, "elsewhere.RAW"))
    (tmp_path / "run7.control").unlink()
    assert reader._side_file(raw) is None
    (tmp_path / "SCAN.CONTROL").write_text(control.format(297.2, 2.5e-4, "elsewhere.RAW"))
    [d] = miox.load(raw)
    assert d.meta["SpectrometerFrequency"] == [297.2] and d.dwelltime == 2.5e-4
    assert d.meta.vendor["side_file"] == "SCAN.CONTROL"


def test_lcmodel_raw_side_file_of_a_basis_folder(tmp_path):
    # the basis set made of the folder's .RAW files states their field and sampling
    for name in ("Cr", "NAA", "Ace"):
        _raw(tmp_path / f"{name}.RAW")
    (tmp_path / "set.BASIS").write_text(
        " $SEQPAR\n HZPPPM = 123.26,\n $END\n $BASIS1\n BADELT = 2.5E-4,\n $END\n"
        " $NMUSED\n FILRAW = 'Cr.RAW',\n $END\n $NMUSED\n FILRAW = 'NAA.RAW',\n $END\n")
    reader = LcmodelRawReader()
    for name in ("Cr", "NAA", "Ace"):
        assert reader._side_file(tmp_path / f"{name}.RAW") == tmp_path / "set.BASIS"
