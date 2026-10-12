####################################################################################################
#                                       test_quicklook.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 12/10/26                                                                                #
#                                                                                                  #
# Purpose: The Quick Look page on synthetic data: five tabs, the processed spectrum with its       #
#          steps and the Full | Metabolites switch, coils as combined, averages before and after   #
#          alignment, edited data with their difference, the empty states (one coil, no water, no  #
#          spectral axis), water scans (by name or by water dominance) and water references found  #
#          next to the data; the MRSI grid around the centre voxel; the image page: the volume     #
#          turned to RAS, planes, slices, details.                                                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re

import numpy as np

# own
import miox
from miox.data import MRImage, MRSData, Provenance
from miox.meta import Meta
from miox.quicklook import ImagePage, Page, Water

SW, SF, N = 2000.0, 123.2, 1024
T = np.arange(N) / SW


def fid():
    """NAA and creatine as NIfTI-MRS stores them (the conjugate of FSL-MRS's orientation)."""
    return sum(a * np.exp(2j * np.pi * (ppm - 4.65) * SF * T - np.pi * 6 * T)
               for ppm, a in ((2.01, 1.0), (3.027, 0.8))).conj()


def dataset(data, tags=(None, None, None), dwell=1 / SW, name="test"):
    meta = Meta({})
    meta.set("SpectrometerFrequency", [SF], "test")
    meta.set("ResonantNucleus", ["1H"], "test")
    return MRSData(data.reshape(1, 1, 1, *data.shape), dwell, np.eye(4), meta,
                   Provenance((), miox.Format.NIFTI_MRS, "test", {}), tuple(tags), name=name)


def test_page_with_coils_and_averages():
    rng = np.random.default_rng(0)
    data = fid()[:, None, None] * rng.uniform(0.5, 1, (1, 4, 1)) + 0.01 * rng.normal(
        size=(N, 4, 8))
    page = Page([dataset(data, ("DIM_COIL", "DIM_DYN", None))], "a.dat").html()
    for tab in ("Spectrum", "Coils", "Averages", "Water", "Details"):
        assert f">{tab}</label>" in page
    assert "4 coils combined (wSVD)" in page and "8 transients aligned" in page
    assert page.count("<title>coil ") == 2 * 4 and "Before alignment" in page  # full, metabolites
    assert page.count('<g class="thin"><title>transient ') == 16  # before and after alignment
    assert "<label for=sm>Metabolites</label>" in page
    assert "No water reference in this file." in page


def test_edited_page_shows_the_difference():
    data = np.stack([fid(), 0.5 * fid()], axis=1)[:, :, None] * np.ones((1, 1, 4))
    page = Page([dataset(data, ("DIM_EDIT", "DIM_DYN", None))], "mega.dat").html()
    assert "difference 1 − 2" in page and "each edit condition processed on its own" in page


def test_single_fid_and_missing_axis():
    page = Page([dataset(fid())], "one.rda").html()
    assert "One channel" in page and "One transient" in page
    unknown = Page([dataset(fid(), dwell=None)], "raw.txt").html()
    assert "nothing is processed" in unknown


def water(shift_hz=0.0):
    return 1000 * np.exp(2j * np.pi * shift_hz * T - np.pi * 6 * T).conj()


def test_water_scans_take_the_water_path():
    named = Page([dataset(water()[:, None] * np.ones((1, 4)), ("DIM_DYN", None, None),
                          name="S01_PRESS_35_H2O")], "S01_PRESS_35_H2O.dat").html()
    unsuppressed = Page([dataset(water() + fid())], "unknown.dat").html()
    for page in (named, unsuppressed):
        assert "water scan" in page and "zero-order phase on water" in page
        assert "This file is a water scan" in page and "creatine" not in page
    suppressed = Page([dataset(0.1 * water() + fid())], "metabolites.dat").html()
    assert "water scan" not in suppressed and "shifted to creatine" in suppressed


def test_water_reference_next_to_the_data(tmp_path):
    for name in ("S01_PRESS_35.dat", "S01_PRESS_35_H2O.dat", "S02_act.SDAT", "S02_ref.sdat",
                 "sub-01_svs.nii.gz", "sub-01_mrsref.nii.gz", "alone.rda"):
        (tmp_path / name).touch()
    found = {name: Water.partner(tmp_path / name) for name in
             ("S01_PRESS_35.dat", "S02_act.SDAT", "sub-01_svs.nii.gz", "alone.rda")}
    assert {k: v.name if v else None for k, v in found.items()} == {
        "S01_PRESS_35.dat": "S01_PRESS_35_H2O.dat", "S02_act.SDAT": "S02_ref.sdat",
        "sub-01_svs.nii.gz": "sub-01_mrsref.nii.gz", "alone.rda": None}


def test_image_page_turns_the_volume_to_ras():
    volume = np.zeros((4, 5, 6))
    volume[0, 0, 0] = 1  # the voxel at the file's origin
    affine = np.diag([-2.0, 1.0, 3.0, 1.0])  # i runs right to left
    image = MRImage(volume, affine, Meta({}), Provenance((), miox.Format.NIFTI, "test", {}))
    turned, sizes = ImagePage.ras(volume, affine)
    assert turned[3, 0, 0] == 1 and list(sizes) == [2.0, 1.0, 3.0]
    page = ImagePage(image, "t1.nii.gz").html()
    for tab in ("Planes", "Slices", "Details"):
        assert f">{tab}</label>" in page
    assert page.count("<figure>") == 3 + 6  # three planes, every one of the 6 axial slices


def test_image_page_of_a_single_slice_and_of_a_series():
    flat = MRImage(np.ones((8, 8, 1)), np.eye(4), Meta({}),
                   Provenance((), miox.Format.DICOM_IMAGE, "test", {}))
    page = ImagePage(flat, "slice.dcm").html()
    assert page.count("<figure>") == 1 and ">Slices</label>" not in page
    series = MRImage(np.ones((4, 4, 4, 10)), np.eye(4), Meta({}),
                     Provenance((), miox.Format.NIFTI, "test", {}))
    assert "first of 10 volumes" in ImagePage(series, "bold.nii").html()


def test_command_line_writes_the_image_page(tmp_path):
    import nibabel as nib

    from miox.cli import main

    path = tmp_path / "t1.nii.gz"
    nib.Nifti1Image(np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6), np.eye(4)).to_filename(
        path)
    assert main(["preview", str(path), "--to", str(tmp_path / "t1.html")]) == 0
    assert ">Planes</label>" in (tmp_path / "t1.html").read_text()
    assert main(["preview", str(path), "--to", str(tmp_path / "t1.pdf")]) == 2


def test_mrsi_grid_marks_the_centre_voxel():
    data = fid()[None, None, None, :] * np.arange(1, 10).reshape(3, 3, 1, 1)
    meta = Meta({})
    meta.set("SpectrometerFrequency", [SF], "test")
    meta.set("ResonantNucleus", ["1H"], "test")
    grid = MRSData(data, 1 / SW, np.eye(4), meta,
                   Provenance((), miox.Format.NIFTI_MRS, "test", {}), (None, None, None))
    page = Page([grid], "mrsi.nii.gz").html()
    assert ">Grid</label>" in page and "voxel (1, 1, 0) of 3 x 3 x 1 (centre)" in page
    assert page.count("<title>voxel (") == 2 * 9  # full and metabolites
    assert len(re.findall(r'class="focus" style="[^"]*"><title>voxel \(1, 1, 0\)', page)) == 2
