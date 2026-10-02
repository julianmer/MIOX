####################################################################################################
#                                          test_sniff.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 01/10/26                                                                                #
#                                                                                                  #
# Purpose: Sniffer tests on synthetic files: every rule fires on its magic bytes and on nothing    #
#          else. No test data needed.                                                              #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import struct

import pytest

# own
from miox import sniff
from miox.sniff import Format, Kind


def test_rda(tmp_path):
    f = tmp_path / "x.rda"
    f.write_bytes(b">>> Begin of header <<<\r\nPatientName: anon\r\n>>> End of header <<<\r\n")
    s = sniff(f)
    assert (s.format, s.vendor, s.kind, s.confidence) == (
        Format.SIEMENS_RDA,
        "siemens",
        Kind.SPECTROSCOPY,
        1.0,
    )


def test_spar_sdat_pair(tmp_path):
    spar = tmp_path / "scan.SPAR"
    sdat = tmp_path / "scan.SDAT"
    spar.write_bytes(b"!--------------\nexamination_name : x\nsamples : 2048\n")
    sdat.write_bytes(b"\x81\x3a\x85\x78" * 16)
    assert sniff(spar).companions == (sdat,)
    s = sniff(sdat)
    assert s.format == Format.PHILIPS_SPAR_SDAT and s.companions == (spar,) and s.confidence == 1.0


def test_sdat_without_spar_is_low_confidence(tmp_path):
    sdat = tmp_path / "scan.sdat"
    sdat.write_bytes(b"\x00" * 64)
    s = sniff(sdat)
    assert s.format == Format.PHILIPS_SPAR_SDAT and s.confidence < 1.0


def test_data_list_pair(tmp_path):
    lst = tmp_path / "raw_1.list"
    data = tmp_path / "raw_1.data"
    lst.write_bytes(b"# === DATA DESCRIPTION FILE ======\n")
    data.write_bytes(b"\x00" * 64)
    assert sniff(lst).format == Format.PHILIPS_DATA_LIST
    s = sniff(data)
    assert s.format == Format.PHILIPS_DATA_LIST and s.companions == (lst,) and s.kind == Kind.RAW


def test_twix_vb(tmp_path):
    f = tmp_path / "meas.dat"
    body = b"\x00" * 1024
    head = struct.pack("<II", 512, 6) + b"Config\x00" + b"\x00" * 100
    f.write_bytes(head + body)
    s = sniff(f)
    assert s.format == Format.SIEMENS_TWIX and s.detail["twix_layout"] == "VB"


def test_twix_vd(tmp_path):
    f = tmp_path / "meas.dat"
    head = struct.pack("<II", 0, 1) + struct.pack("<IIQQ", 1, 2, 10240, 4096)
    f.write_bytes(head + b"\x00" * 20000)
    s = sniff(f)
    assert s.format == Format.SIEMENS_TWIX and s.detail["twix_layout"] == "VD"
    assert s.detail["n_measurements"] == 1


def test_pfile(tmp_path):
    f = tmp_path / "P12345.7"
    f.write_bytes(struct.pack("<f", 24.0) + b"\x00" * 64)
    s = sniff(f)
    assert s.format == Format.GE_PFILE and s.detail["pfile_revision"] == 24.0 and s.kind == Kind.RAW


def test_pfile_unknown_revision_needs_name(tmp_path):
    f = tmp_path / "blob.bin"
    f.write_bytes(struct.pack("<f", 13.0) + b"\x00" * 64)
    assert sniff(f).format == Format.UNKNOWN
    g = tmp_path / "P00001.7"
    g.write_bytes(struct.pack("<f", 13.0) + b"\x00" * 64)
    assert sniff(g).confidence == 1.0


def test_bruker_scan_dir(tmp_path):
    scan = tmp_path / "5"
    scan.mkdir()
    (scan / "acqp").write_text(
        "##TITLE=Parameter List\n##JCAMPDX=4.24\n##$ACQ_sw_version=( 65 )\n<PV 6.0.1>\n"
        "##$ACQ_method=( 40 )\n<Bruker:PRESS>\n"
    )
    (scan / "method").write_text("##TITLE=Parameter List\n##JCAMPDX=4.24\n")
    (scan / "fid").write_bytes(b"\x00" * 64)
    (scan / "pdata" / "1").mkdir(parents=True)
    (scan / "pdata" / "1" / "visu_pars").write_text("##TITLE=Parameter List\n##JCAMPDX=4.24\n")
    (scan / "pdata" / "1" / "2dseq").write_bytes(b"\x00" * 64)
    s = sniff(scan)
    assert s.format == Format.BRUKER and s.detail["paravision_version"] == "PV 6.0.1"
    assert s.kind == Kind.SPECTROSCOPY
    assert sniff(scan / "fid").kind == Kind.RAW
    assert sniff(scan / "pdata" / "1" / "2dseq").kind == Kind.SPECTROSCOPY
    assert sniff(scan / "acqp").format == Format.BRUKER


def test_bruker_imaging_method(tmp_path):
    (tmp_path / "acqp").write_text(
        "##TITLE=Parameter List\n##JCAMPDX=4.24\n##$ACQ_method=( 40 )\n<Bruker:FLASH>\n"
    )
    assert sniff(tmp_path / "acqp").kind == Kind.IMAGING


def test_varian(tmp_path):
    d = tmp_path / "scan.fid"
    d.mkdir()
    (d / "procpar").write_text("np 1 1\n1 2048\n")
    (d / "fid").write_bytes(b"\x00" * 64)
    assert sniff(d).format == Format.VARIAN
    fid = sniff(d / "fid")
    assert fid.format == Format.VARIAN and fid.companions == (d / "procpar",)


def test_lcmodel_raw(tmp_path):
    f = tmp_path / "metab.RAW"
    f.write_text(" $SEQPAR\n echot= 30.\n $END\n $NMID\n id='x'\n $END\n 1.0 0.0\n")
    assert sniff(f).format == Format.LCMODEL_RAW


def test_jmrui(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("jMRUI Data Textfile\n\nFilename: x\n")
    assert sniff(f).format == Format.JMRUI_TEXT
    g = tmp_path / "x.mrui"
    g.write_bytes(b"\x00" * 64)
    assert sniff(g).format == Format.JMRUI_BINARY and sniff(g).confidence == 0.5


def test_hdf5(tmp_path):
    magic = b"\x89HDF\r\n\x1a\n"
    f = tmp_path / "x.h5"
    f.write_bytes(magic + b"\x00" * 64)
    assert sniff(f).format == Format.ISMRMRD
    g = tmp_path / "x.bin"
    g.write_bytes(magic + b"\x00" * 64)
    assert sniff(g).format == Format.UNKNOWN


def test_nifti(tmp_path):
    nib = pytest.importorskip("nibabel")
    import numpy as np

    f = tmp_path / "img.nii.gz"
    nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4)).to_filename(f)
    s = sniff(f)
    assert s.format == Format.NIFTI and s.kind == Kind.IMAGING and s.detail["shape"] == (2, 2, 2)


def test_nifti_mrs(tmp_path):
    nib = pytest.importorskip("nibabel")
    import json

    import numpy as np

    f = tmp_path / "svs.nii.gz"
    img = nib.Nifti2Image(np.zeros((1, 1, 1, 8), dtype=np.complex64), np.eye(4))
    hdr = {"SpectrometerFrequency": [123.2], "ResonantNucleus": ["1H"], "dim_5": "DIM_DYN"}
    img.header.extensions.append(nib.nifti1.Nifti1Extension(44, json.dumps(hdr).encode()))
    img.to_filename(f)
    s = sniff(f)
    assert s.format == "nifti_mrs" and s.kind == Kind.SPECTROSCOPY
    assert s.detail["nifti_version"] == 2 and s.detail["dim_tags"] == {}


def test_unknown_and_missing(tmp_path):
    f = tmp_path / "noise.bin"
    f.write_bytes(b"hello world " * 10)
    assert sniff(f).format == Format.UNKNOWN
    with pytest.raises(FileNotFoundError):
        sniff(tmp_path / "nope")


def test_empty_dir(tmp_path):
    assert sniff(tmp_path).format == Format.UNKNOWN
