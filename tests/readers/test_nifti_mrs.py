####################################################################################################
#                                        test_nifti_mrs.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Loader, MRSData and NIfTI-MRS reader tests on synthetic files. No test data needed.     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pytest
from nifti_mrs.create_nmrs import gen_nifti_mrs

# own
import miox
from miox.meta import Meta


def _write(tmp_path, data, **header):
    tags = ["DIM_COIL", "DIM_DYN", None]
    image = gen_nifti_mrs(data, 2.5e-4, 123.2, "1H", dim_tags=tags, no_conj=True)  # stored as given
    for key, value in header.items():
        image.add_hdr_field(key, value)
    path = tmp_path / "svs.nii.gz"
    image.save(path)
    return path


def test_nifti_mrs_reader(tmp_path):
    data = (np.arange(2048 * 4 * 3) * (1 + 1j)).reshape(1, 1, 1, 2048, 4, 3)
    [d] = miox.load(_write(tmp_path, data, EchoTime=0.03))
    assert d.name == "svs" and d.shape == (1, 1, 1, 2048, 4, 3)
    assert d.dims == ("x", "y", "z", "t", "DIM_COIL", "DIM_DYN")
    assert d.nucleus == "1H" and d.spectrometer_frequency == pytest.approx(123.2)
    assert d.dwelltime == pytest.approx(2.5e-4) and d.spectral_width == pytest.approx(4000)
    assert d.meta["EchoTime"] == 0.03
    assert d.meta.source["EchoTime"] == "NIfTI-MRS header extension"
    assert d.meta.source["SpectralWidth"].startswith("derived")
    assert d.meta["FieldStrength"] == pytest.approx(2.894, abs=1e-3)
    assert d.meta.vendor["header_extension"]["dim_5"] == "DIM_COIL"
    assert d.provenance.reader == "NiftiMrsReader" and d.provenance.notes == ()
    assert np.allclose(d.data, data)


def test_invalid_header_is_read_with_a_note(tmp_path):
    path = _write(tmp_path, np.ones((1, 1, 1, 64, 2, 1), complex))
    image = nib.load(path)
    header = json.loads(image.header.extensions[0].get_content())
    header["InversionTime"] = None  # breaks the standard: must be a number
    image.header.extensions[0] = nib.nifti1.Nifti1Extension(44, json.dumps(header).encode())
    nib.save(image, path)
    [d] = miox.load(path)
    assert d.meta["InversionTime"] is None
    assert any("not valid NIfTI-MRS" in note for note in d.provenance.notes)


def test_frequency_in_hz_is_corrected(tmp_path):
    path = _write(tmp_path, np.ones((1, 1, 1, 64, 2, 1), complex))
    image = nib.load(path)
    header = json.loads(image.header.extensions[0].get_content())
    header["SpectrometerFrequency"] = [123.2e6]
    image.header.extensions[0] = nib.nifti1.Nifti1Extension(44, json.dumps(header).encode())
    nib.save(image, path)
    [d] = miox.load(path)
    assert d.spectrometer_frequency == pytest.approx(123.2)
    assert d.meta["FieldStrength"] == pytest.approx(2.894, abs=1e-3)
    assert "corrected: stored in Hz" in d.meta.source["SpectrometerFrequency"]
    assert d.meta.vendor["header_extension"]["SpectrometerFrequency"] == [123.2e6]


def test_bids_sidecar_adds_keys_and_notes_disagreements(tmp_path):
    path = _write(tmp_path, np.ones((1, 1, 1, 64, 2, 1), complex), EchoTime=0.03)
    sidecar = {"Manufacturer": "Philips", "EchoTime": 0.08, "SpectrometerFrequency": 123.2}
    (tmp_path / "svs.json").write_text(json.dumps(sidecar))
    [d] = miox.load(path)
    assert d.meta["Manufacturer"] == "Philips"
    assert d.meta.source["Manufacturer"] == "BIDS sidecar: Manufacturer"
    assert d.meta["EchoTime"] == 0.03 and d.meta.vendor["sidecar"] == sidecar
    [note] = [n for n in d.provenance.notes if "svs.json" in n]
    assert "EchoTime" in note and "header's is kept" in note


def test_vendor_fields_flatten_nested_headers():
    meta = Meta({"a": {"b": [1, 2], "c": [{"d": 1}, [3, 4]]}, "e": "x"})
    assert meta.vendor_fields() == {"a/b": [1, 2], "a/c/0/d": 1, "a/c/1": [3, 4], "e": "x"}


def test_no_reader(tmp_path):
    f = tmp_path / "noise.bin"
    f.write_bytes(b"hello world " * 10)
    with pytest.raises(miox.ReaderError):
        miox.load(f)
