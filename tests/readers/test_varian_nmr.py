####################################################################################################
#                                        test_varian_nmr.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 09/10/26                                                                                #
#                                                                                                  #
# Purpose: Varian/Agilent NMR spectrometer data on synthetic .fid folders written by nmrglue: a 1D #
#          FID with its referencing, a 2D FID with its rows as acquired, and the routing of        #
#          imaging consoles' data away from the spectrometer path.                                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

from pathlib import Path

import nmrglue as ng
import numpy as np
import pytest

# own
import miox


def spectrometer(folder: Path, data: np.ndarray, **parameters) -> Path:
    udic = ng.fileiobase.create_blank_udic(data.ndim)
    for axis, size in enumerate(data.shape):
        udic[axis].update(size=size, complex=True)
    dic = ng.varian.create_dic(udic)
    values = {"apptype": "std1D", "tn": "H1", "sfrq": "599.8311617", "sw": "12000",
              "rfl": "2000", "rfp": "1000", "reffrq": "599.8311617", "seqfil": "s2pul",
              "solvent": "cd3od", "nt": "16", "console": "vnmrs", "temp": "25", "vttype": "2",
              **parameters}
    for name, value in values.items():
        param = ng.varian.create_pdic_param(name, [value])
        if not value.replace(".", "").replace("-", "").isdigit():
            param["basictype"] = "2"
        dic["procpar"][name] = param
    ng.varian.write(str(folder), dic, data.astype(np.complex64), overwrite=True)
    return folder


def test_spectrometer_fid(tmp_path):
    fid = (np.arange(64) - 1j * np.arange(64)).astype(complex)
    folder = spectrometer(tmp_path / "proton.fid", fid)
    [d] = miox.load(folder)
    assert isinstance(d, miox.MRSData) and d.provenance.reader == "VarianReader"
    assert d.shape == (1, 1, 1, 64) and np.array_equal(d.data[0, 0, 0], fid)
    assert d.meta["SpectrometerFrequency"] == [599.8311617]
    assert d.meta["ResonantNucleus"] == ["1H"] and d.dwelltime == pytest.approx(1 / 12000)
    # the carrier lies sw / 2 - rfl + rfp = 5000 Hz above 0 ppm
    assert d.meta["ReferencePPM"] == pytest.approx(5000 / 599.8311617)
    assert d.meta["SequenceName"] == "s2pul" and d.meta["Solvent"] == "cd3od"
    assert d.meta["NumberOfAverages"] == 16 and d.meta["SampleTemperature"] == 298.15
    assert any("reffrq puts the carrier" in n for n in d.provenance.notes)


def test_spectrometer_2d_rows_as_acquired(tmp_path):
    rows = np.arange(4 * 8).reshape(4, 8) * (1 + 1j)
    folder = spectrometer(tmp_path / "hsqc.fid", rows, ni="2", sw1="3000", reffrq="599.8261617")
    dic = ng.varian.read_procpar(str(folder / "procpar"))
    dic["phase"] = ng.varian.create_pdic_param("phase", ["1", "2"])
    dic["array"] = ng.varian.create_pdic_param("array", ["phase"])
    dic["array"]["basictype"] = "2"
    ng.varian.write_procpar(str(folder / "procpar"), dic, overwrite=True)
    [d] = miox.load(folder)
    assert d.shape == (1, 1, 1, 8, 4) and d.dim_tags[0] == "DIM_INDIRECT_0"
    assert np.array_equal(d.data[0, 0, 0], rows.T)
    assert d.dim_info[5]["info"].startswith("ni 2 x phase 1,2, sw1 3000 Hz")
    assert d.meta["SpectrometerFrequency"] == [599.8311617]  # indirect nuclei not stated
    assert not any("reffrq puts the carrier" in n for n in d.provenance.notes)


def test_imaging_console_is_not_spectrometer():
    assert miox.readers.varian.VarianReader.is_spectrometer({"apptype": ["std1D"]})
    assert not miox.readers.varian.VarianReader.is_spectrometer({"apptype": ["im1Dglobal"]})
    assert miox.readers.varian.VarianReader.is_spectrometer({})  # older VNMR: no apptype
