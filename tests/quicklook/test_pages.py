####################################################################################################
#                                          test_pages.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/11/26                                                                                #
#                                                                                                  #
# Purpose: The Quick Look pages of the kinds beyond spectroscopy and images: every page builds     #
#          from a public sample with the tabs it should have, the dispatcher picks the page by     #
#          the datasets, and the pieces the pages share behave on synthetic input.                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re

import numpy as np
import pytest

# own
from miox import load
from miox.quicklook import page
from miox.quicklook_pages import Panel, RawPage, SequencePage
from tests.conftest import PUBLIC_DATA

SURF = "nitest_freesurfer/fsaverage/surf/"
PULSEQ = "komamri/KomaMRIFiles/test/test_files/pulseq/basic_tests/"
SAMPLES = [  # file, the tabs of its page
    ("gadgetron/T2W/meas_MID00057_T2w.dat", ["k-space", "Coils", "Readouts", "Waveforms",
                                             "Details"]),
    ("gadgetron/generic/meas_MID01095_FID40115/meas_MID01095_FID40115.mrd",
     ["k-space", "Coils", "Readouts", "Details"]),
    ("m4raw/multicoil_test/2022101103_T206.h5", ["k-space", "Coils", "Readouts", "Details"]),
    ("ge_to_ismrmrd/P20480_GRE.7", ["k-space", "Coils", "Readouts", "Details"]),
    ("spec2nii_test_data/bruker/PV-360.3.5/3/rawdata.job0",
     ["k-space", "Coils", "Readouts", "Details"]),
    (PULSEQ + "v1.4/epi.seq", ["Sequence", "Start", "Details"]),
    (PULSEQ + "v1.4/spiral.seq", ["Sequence", "Start", "Details"]),
    (PULSEQ + "v1.2/epi_JEMRIS.seq", ["Sequence", "Start", "Details"]),
    ("toppe/toppe0.entry", ["Details"]),
    ("bidsphysio/bidsphysio.physio2bids/tests/data/sample_VB15A.puls",
     ["Recording", "First 30 s", "Details"]),
    ("bidsphysio/bidsphysio.acq2bids/tests/data/sample.acq", ["Recording", "Details"]),
    ("niivue/packages/niivue/demos/images/TR_S_R.tt.gz", ["Views", "Details"]),
    ("nibabel/nibabel/tests/data/empty.tck", ["Details"]),
    (SURF + "lh.inflated", ["Views", "Details"]),
    (SURF + "lh.curv", ["Views", "Details"]),
    ("nibabel/nibabel/gifti/tests/data/ascii.gii", ["Views", "Details"]),
    ("bmrb/18465/o157_nhsqcv6.ucsf", ["Spectrum", "Details"]),
    ("bruker_data_test/data/flat/coffee/UV1010_M1-1003-1002_6268756_ErISKLIoeB/11/pdata/1/1r",
     ["Spectrum", "Details"]),
    ("nmrglue/bruker_2d/ser", ["Rows", "Details"]),
    ("spant/inst/extdata/press_refocus.pta", ["Pulse", "Details"]),
    ("nitransforms/nitransforms/tests/data/affine-LAS.itk.tfm", ["Matrix", "Details"]),
    ("komamri/examples/2.phantoms/brain.h5", ["Maps", "Details"]),
    ("mrd/simple_gre_noise_covariance.mrd", ["Covariance", "Details"]),
    ("dicompare/dicompare/tests/fixtures/pro_files/C2P__cmrr_mbep2d_bold__p3_mb2.pro",
     ["C2P__cmrr_mbep2d_bold__p3_mb"]),
    # the edges found by building every recorded sample's page
    ("bruker_data_test/data/zipped/cyclosporin/cyclosporin_hsqc/cyclosporin_hsqc/3/pdata/1/2rr",
     ["Spectrum", "Details"]),  # four components: the first shown
    ("nmrglue_repo/nmrglue/fileio/tests/data/nmrpipe_3d_freq.ft3", ["Spectrum", "Details"]),
    ("mrsimulator/tests/spectral_integration_tests/simpson_simulated_lineshapes/"
     "sideband_sideband/test00/test00.csdf", ["Spectrum", "Details"]),  # no f0: Hz axes
    ("jcamp_data_test/data/nmr/cytisine/hmbc.jdx", ["Spectrum", "Details"]),  # indirect index
    ("komamri/KomaMRICore/test/test_files/jemris_signals_epi_brain_motion.h5",
     ["k-space", "Coils", "Readouts", "Details"]),  # waveforms without a clock
    ("komamri/KomaMRIFiles/test/test_files/phantom/brain_mrilab.mat", ["Maps", "Details"]),
    ("niivue/packages/niivue/demos/images/mni152.SLF1_R.tsf", ["Details"]),  # no points
    ("nitest_freesurfer/fsaverage/label/lh.V1.label", ["Details"]),  # no faces
    ("nibabel/nibabel/gifti/tests/data/ascii_flat_data.gii", ["Details"]),  # faces beyond
    ("mrsi_bids/sub-01/mrs/sub-01_run-1_mrsi.nii.gz", None),  # TE as a list
    ("osi2one/OSIIONE_Reference_System_Data_v1/OSIIONE_Reference_System_Data_v1/"
     "2026-08-16_113205_simulation_tse_3d_T1-mapping_single_slice_CaliberMRI/ismrmrd_data.h5",
     ["k-space", "Coils", "Readouts", "Details"]),  # no recon matrix stated
]


def tabs(html: str) -> list[str]:
    return re.findall(r"<label for=t\d+>([^<]+)</label>", html)


@pytest.mark.data
@pytest.mark.parametrize("relpath,expected", SAMPLES, ids=[s[0].rsplit("/", 1)[-1]
                                                            for s in SAMPLES])
def test_page_has_its_tabs(relpath, expected):
    path = PUBLIC_DATA / relpath
    if not path.exists():
        pytest.skip("file not fetched")
    html = page(load(path), path.name, path)
    if expected is not None:  # None: the page builds, its tabs are the spectroscopy page's
        assert tabs(html) == expected
    assert "<svg" in html or "<img" in html or "<table" in html


@pytest.mark.data
def test_values_only_surface_is_drawn_on_the_inflated_surface_beside_it():
    path = PUBLIC_DATA / SURF / "lh.curv"
    if not path.exists():
        pytest.skip("file not fetched")
    html = page(load(path), path.name, path)
    assert "shown on lh.inflated beside it" in html
    assert "163842 vertices" in html


@pytest.mark.data
def test_sequence_blocks_add_up_to_the_stated_duration():
    path = PUBLIC_DATA / PULSEQ / "v1.4/epi.seq"
    if not path.exists():
        pytest.skip("file not fetched")
    [seq] = load(path)
    traces, adcs, total = SequencePage.expand(seq)
    assert abs(total - seq.duration) < 1e-9
    assert len(adcs) == sum(1 for b in seq.blocks if b["adc"])
    assert all(traces[g][0][-1] == total for g in ("gx", "gy", "gz"))
    assert traces["rf"][1].max() == pytest.approx(abs(seq.events["rf"][1]["amp"]), rel=1e-6)


def test_faces_are_filled_between_their_vertices():
    """A square of two triangles seen face-on covers every pixel between its corners, with
    one flat shade: no stipple, no holes."""
    from miox.data import Provenance, Surface
    from miox.meta import Meta
    from miox.quicklook_pages import SurfacePage
    from miox.sniff import Format

    vertices = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    square = Surface(vertices, faces, "test", {}, {}, Meta(),
                     Provenance((), Format.FREESURFER_SURFACE, "test", {}), "rh.test")
    image = SurfacePage(square, "square").render(((1, 0), (1, 1), (1, 2)))
    inside = image[3:-3, 3:-3]
    assert not np.isnan(inside).any()
    assert np.ptp(inside) == pytest.approx(0, abs=1e-9)
    assert np.isnan(image[0]).all() and np.isnan(image[:, 0]).all()


def test_plain_image_recovers_a_box_from_its_k_space():
    """A 32 x 48 box, phase encoded along 64 lines, read out over 128 samples with twofold
    oversampling: the plain image is the box, cropped to the 64-sample matrix."""
    from miox.data import Provenance, RawData
    from miox.meta import Meta
    from miox.sniff import Format

    lines, samples = 64, 128
    box = np.zeros((lines, samples))
    box[16:48, 40:88] = 1.0
    k = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(box)))
    headers = np.zeros(lines, dtype=RawData.HEADER)
    headers["idx"]["kspace_encode_step_1"] = np.arange(lines)
    headers["center_sample"] = samples // 2
    headers["number_of_samples"] = samples
    data = k[:, None, :].astype(np.complex64)
    meta = Meta()
    meta.set("BaseResolution", 64, "test")
    raw = RawData(data, headers, meta, Provenance((), Format.SIEMENS_TWIX, "test", {}), "box")
    image, cropped = RawPage.plain_image(raw)
    assert cropped == "readout cropped to 64"
    assert image.shape == (lines, 64)
    inside = image[16:48, 8:56]
    assert inside.min() > 0.9 * inside.max()
    assert image[:8].max() < 0.05 * inside.max()


def test_scale_numbers_are_short():
    assert [Panel.number(v) for v in (0, 1090.4, -0.25, 3.14159, 240000, 0.000289)] == [
        "0", "1090", "-0.25", "3.14", "2.4e+05", "0.000289"]
