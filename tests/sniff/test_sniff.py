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


def test_text_is_not_a_pfile(tmp_path):
    f = tmp_path / "README.md"
    f.write_text("## About\n\nThis repository holds test data.\n")  # "## A" reads as 10.0086
    assert sniff(f).format == Format.UNKNOWN


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
    img.header.set_intent("none", (), name="mrs_v0_2")
    img.to_filename(f)
    s = sniff(f)
    assert s.format == Format.NIFTI_MRS and s.kind == Kind.SPECTROSCOPY
    assert s.detail["nifti_version"] == 2 and s.detail["nifti_mrs_version"] == "0.2"
    assert s.detail["dim_tags"] == {"dim_5": "DIM_DYN"}


def test_unknown_and_missing(tmp_path):
    f = tmp_path / "noise.bin"
    f.write_bytes(b"hello world " * 10)
    assert sniff(f).format == Format.UNKNOWN
    with pytest.raises(FileNotFoundError):
        sniff(tmp_path / "nope")


def test_empty_dir(tmp_path):
    assert sniff(tmp_path).format == Format.UNKNOWN


@pytest.mark.parametrize(
    "name,text,fmt,kind",
    [
        (
            "x.BASIS",
            " $SEQPAR\n HZPPPM=123.2\n $END\n $BASIS1\n ID='NAA'\n $END\n",
            "lcmodel_basis",
            "basis_set",
        ),
        ("control.file", " $LCMODL\n HZPPPM=123.2\n $END\n", "lcmodel_control", "protocol"),
        ("makebasis.in", " $SEQPAR\n $END\n $NMALL\n $END\n", "lcmodel_control", "protocol"),
        (
            "x.COORD",
            " LCModel (Version 6.3-1B)\n 38 lines in following concentration table\n",
            "lcmodel_coord",
            "fit_result",
        ),
        (
            "x.TABLE",
            " LCModel (Version 6.3-1B)\n x\n$$CONC 38 lines in following\n",
            "lcmodel_table",
            "fit_result",
        ),
        ("x.CSV", "Row, Col, NAA, NAA %SD\n 1, 1, 1.0, 3\n", "lcmodel_csv", "fit_result"),
        ("x.ps", "%!PS-Adobe-2.0\n%%Creator: LCModel\n", "lcmodel_ps", "report"),
        ("x.dpt", "Dangerplot_version\t1.0\nNumber_of_points\t2\n", "tarquin_dpt", "spectroscopy"),
        ("r.txt", "Signal     Conc (au)       %SD   SD (au)\n", "tarquin_results", "fit_result"),
        ("f.txt", "Row : 1, Col : 1, Slice : 1\nPPMScale,Data,Fit\n", "tarquin_fit", "fit_result"),
        (
            "x.xml",
            '<vespa_export version="1.0.0">\n<dataset id="a">\n',
            "vespa_viff",
            "spectroscopy",
        ),
        ("x.ddf", "DATA DESCRIPTOR FILE\nversion: 6.1\n", "sivic_ddf", "spectroscopy"),
    ],
)
def test_analysis_text_formats(tmp_path, name, text, fmt, kind):
    f = tmp_path / name
    f.write_text(text)
    s = sniff(f)
    assert (s.format, s.kind) == (fmt, kind)


def test_header_and_binary_pairs(tmp_path):
    (tmp_path / "press.rsp").write_text("[FILE INFORMATION]\ndata_format= xdr\n")
    (tmp_path / "press.rsd").write_bytes(b"\x3f\x80\x00\x00" * 16)
    (tmp_path / "csi.ddf").write_text("DATA DESCRIPTOR FILE\nversion: 6.1\n")
    (tmp_path / "csi.cmplx").write_bytes(b"\x00" * 64)
    assert sniff(tmp_path / "press.rsp").companions == (tmp_path / "press.rsd",)
    assert sniff(tmp_path / "press.rsd").format == Format.VASF
    assert sniff(tmp_path / "csi.cmplx").format == Format.SIVIC_DDF
