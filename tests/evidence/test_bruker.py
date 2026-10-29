####################################################################################################
#                                          test_bruker.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 04/10/26                                                                                #
#                                                                                                  #
# Purpose: Bruker 2dseq images against the vendor's own DICOM exports, complex 2dseq spectra       #
#          against the magnitude spectra and single-voxel ones against the scan's fid, rawdata     #
#          transients as dynamics, and imaging k-space (fid, rawdata.job0) against the             #
#          2dseq images of the same scans: the lines placed by the encoding steps and objects      #
#          reconstruct, with zero-filling to the recon matrix, to the vendor's image of every      #
#          slice up to the in-plane orientation. The fid and rawdata of CSI scans say they are     #
#          k-space and point to the scanner's reconstruction, which their folder and Quick Look    #
#          show.                                                                                   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import numpy as np
import pytest

# own
from miox import MRImage, MRSData, RawData, load
from tests.conftest import PUBLIC_DATA
from tests.evidence.test_dicom_images import grid_mapping, on_reference_grid, same

BRUKER = PUBLIC_DATA / "spec2nii_test_data/bruker"
EXPORTED = sorted(p.parent.relative_to(BRUKER).as_posix() for p in BRUKER.glob("*/*/pdata/*/dicom"))
COMPLEX = ["PV-6.0.1/9", "PV-6.0.1/10", "PV-6.0.1/5", "PV-6.0.1/6"]  # complex pdata/2, magnitude 1
TO_THE_SECOND = ("AcquisitionTime", "AcquisitionDateTime")  # the export drops the fraction
KNOWN_DIFFERENCES = {
    "AcquisitionMatrixPE": "the export swaps the acquisition matrix of transposed frames (it "
    "then states the 213 read points, not the 256 phase-encoding steps)",
}


def _magnitude(scan: str) -> MRImage:
    [image] = load(BRUKER / scan / "pdata/1/2dseq")
    return image


@pytest.mark.data
@pytest.mark.parametrize("pdata", EXPORTED)
def test_2dseq_image_equals_dicom_export(pdata):
    images = load(BRUKER / pdata / "2dseq")
    references = load(BRUKER / pdata / "dicom")
    assert len(images) == len(references)
    for image in images:
        matches = []
        for reference in references:
            try:
                order, flipped = grid_mapping(image, reference)
            except AssertionError:
                continue
            matches.append((reference, order, flipped))
        assert len(matches) == 1, f"{image.name}: {len(matches)} exports on its grid"
        reference, order, flipped = matches[0]
        data = on_reference_grid(image, reference)
        assert data.shape == reference.shape
        scale = float(np.abs(reference.data).max()) or 1.0
        assert np.allclose(data, reference.data, rtol=1e-4, atol=1e-3 * scale)
        differ = []
        for key in sorted(set(image.meta) & set(reference.meta)):
            if key in KNOWN_DIFFERENCES:
                continue
            ours, theirs = image.meta[key], reference.meta[key]
            if key in TO_THE_SECOND:
                ours, theirs = str(ours)[:-7], str(theirs)[:-7]
            if not same(key, ours, theirs):
                differ.append(f"{key}: {ours!r} vs {theirs!r}")
        assert not differ, differ


@pytest.mark.data
@pytest.mark.parametrize("scan", COMPLEX)
def test_2dseq_spectrum_fid_matches_magnitude_spectrum(scan):
    """The FID made from the complex 2dseq spectrum has the vendor's magnitude spectrum, in the
    stored frequency order (zero-filled as the reconstruction did; the vendor's also holds the
    digital filter's points, which the FID leaves out), and the raw fid of the scan agrees with
    that order."""
    [fid] = load(BRUKER / scan / "pdata/2/2dseq")
    magnitude = _magnitude(scan)
    assert isinstance(fid, MRSData) and magnitude.dims[3] == "frequency"
    m = magnitude.shape[3]
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(fid.data, n=m, axis=3), axes=3))
    assert spectrum.shape == magnitude.shape
    assert np.corrcoef(spectrum.ravel(), magnitude.data.ravel())[0, 1] > 0.999
    [raw] = load(BRUKER / scan / "fid")
    index = np.unravel_index(np.argmax(magnitude.data.sum(axis=3)), magnitude.shape[:3])
    reference = magnitude.data[index]
    raw_spectrum = np.abs(np.fft.fftshift(np.fft.fft(raw.data[index])))
    n, m = raw_spectrum.size, reference.size
    width = raw.meta["SpectralWidth"]
    on_grid = np.interp(
        (np.arange(m) - m // 2) * width / m, (np.arange(n) - n // 2) * width / n, raw_spectrum
    )
    same_order = np.corrcoef(on_grid, reference)[0, 1]
    reversed_order = np.corrcoef(on_grid[::-1], reference)[0, 1]
    assert same_order > 0.98 and same_order > reversed_order


@pytest.mark.data
@pytest.mark.parametrize("scan", ["PV-6.0.1/9", "PV-6.0.1/10"])
def test_2dseq_single_voxel_fid_equals_the_scan_fid(scan):
    """The reconstruction writes the scan's fid and its spectrum: the FID made from the spectrum
    equals that fid point for point (up to scale)."""
    [ours] = load(BRUKER / scan / "pdata/2/2dseq")
    [fid] = load(BRUKER / scan / "fid")
    a, b = np.asarray(fid.data).ravel(), np.asarray(ours.data).ravel()
    head = b[: a.size]
    assert abs(np.vdot(head, a)) / (np.linalg.norm(head) * np.linalg.norm(a)) > 1 - 1e-6
    assert b.size == a.size


@pytest.mark.data
@pytest.mark.parametrize("scan, transients", [("PV-6.0.1/9", 16), ("PV-360.3.5/19", 32),
                                              ("PV-360.1.1/12", 384)])
def test_rawdata_transients_are_dynamics(scan, transients):
    """rawdata.job0 stores every transient: averages (PVM_NAverages) x repetitions (NR)."""
    [d] = load(BRUKER / scan / "rawdata.job0")
    assert d.shape[4 + d.dim_tags.index("DIM_DYN")] == transients


@pytest.mark.data
def test_magnitude_spectrum_keys():
    image = _magnitude("PV-6.0.1/9")
    assert image.shape == (1, 1, 1, 2048) and image.dims == ("i", "j", "k", "frequency")
    assert image.meta["SpectralWidth"] == pytest.approx(4401.4085, rel=1e-6)
    assert image.meta["ReferencePPM"] == 4.7 and image.meta["ResonantNucleus"] == ["1H"]
    assert image.meta["SpectrometerFrequency"] == pytest.approx([400.3225099], rel=1e-8)
    csi = _magnitude("PV-6.0.1/5")
    assert csi.shape == (16, 16, 1, 2048) and csi.voxel_sizes == pytest.approx((2.5, 2.5, 2.0))
    cycles = _magnitude("PV-360.1.1/12")
    assert cycles.shape == (1, 1, 1, 2060, 12) and cycles.dims[4] == "cycle"
    assert cycles.meta.source["SpectralWidth"] == "method: PVM_SpecSWH"  # VisuCoreExtent is 0


@pytest.mark.data
@pytest.mark.parametrize("scan", ["PV-6.0.1/5", "PV-6.0.1/6"])
def test_csi_kspace_says_so_and_points_to_the_reconstruction(scan, tmp_path):
    """The fid and rawdata of a CSI scan are k-space: they say so and name the scanner's
    reconstructions (complex first); the scan folder and Quick Look show the complex one."""
    from miox.quicklook import load as shown, page

    for raw_file in ("fid", "rawdata.job0"):
        [raw] = load(BRUKER / scan / raw_file)
        notes = [n for n in raw.provenance.notes if n.startswith("k-space, not reconstructed")]
        assert len(notes) == 1 and notes[0].endswith("is pdata/2/2dseq or pdata/1/2dseq")
        assert not any("might not be reconstructed" in n for n in raw.provenance.notes)
    [folder] = load(BRUKER / scan)
    [complex_spectra] = load(BRUKER / scan / "pdata/2/2dseq")
    assert np.array_equal(folder.data, complex_spectra.data)
    datasets, page_notes = shown(BRUKER / scan / "fid")
    assert np.array_equal(datasets[0].data, complex_spectra.data)
    assert page_notes == ("fid holds k-space: shown is the scanner's reconstruction "
                          "pdata/2/2dseq",)
    copy = tmp_path / "5"
    copy.mkdir()
    for name in ("acqp", "method", "fid"):
        (copy / name).write_bytes((BRUKER / scan / name).read_bytes())
    datasets, page_notes = shown(copy / "fid")
    assert page_notes == ()
    assert "The data are k-space: nothing to show" in page(datasets, "fid", copy / "fid")
    assert any(n.endswith("the scan holds no reconstruction (pdata/<n>/2dseq)")
               for n in datasets[0].provenance.notes)


# --- k-space ------------------------------------------------------------------------------------

RAW_SCANS = [  # scan, raw file: Cartesian imaging scans with their reconstructed 2dseq
    ("PV-6.0.1/1", "fid"),  # FLASH, three orthogonal slice packs of three slices
    ("PV-6.0.1/3", "fid"),  # RARE, echo trains of 8, ten interleaved slices
    ("PV-360.3.5/3", "rawdata.job0"),  # FLASH, four receive channels, three packs
    ("PV-360.1.1/2", "rawdata.job0"),
]
ORIENTATIONS = (
    lambda a: a, lambda a: a.T, lambda a: a[::-1], lambda a: a[:, ::-1], lambda a: a.T[::-1],
    lambda a: a.T[:, ::-1], lambda a: a[::-1, ::-1], lambda a: a.T[::-1, ::-1],
)


def _reconstruct(raw: RawData, slice_number: int) -> np.ndarray:
    """Root sum of squares image of one slice: the lines at their encoding steps, zero-filled
    to the reconstruction matrix, inverse Fourier transformed."""
    h = raw.headers
    rows = np.nonzero((h["idx"]["slice"] == slice_number) & (h["idx"]["repetition"] == 0))[0]
    lines = int(h["idx"]["kspace_encode_step_1"].max()) + 1
    matrix = raw.meta["ReconMatrixSize"]
    k = np.zeros((raw.n_coils, matrix[1], matrix[0]), dtype=complex)
    top, left = (matrix[1] - lines) // 2, (matrix[0] - raw.data.shape[-1]) // 2
    for row in rows:
        line = int(h["idx"]["kspace_encode_step_1"][row])
        k[:, top + line, left : left + raw.data.shape[-1]] = raw.data[row]
    axes = (1, 2)
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k, axes=axes), axes=axes), axes=axes)
    return np.sqrt((np.abs(image) ** 2).sum(axis=0))


@pytest.mark.data
@pytest.mark.parametrize("scan, raw_file", RAW_SCANS)
def test_imaging_kspace_reconstructs_to_the_2dseq_image(scan, raw_file):
    [raw] = load(BRUKER / scan / raw_file)
    images = load(BRUKER / scan / "pdata/1/2dseq")
    slices = [np.abs(image.data[:, :, k]) for image in images for k in range(image.shape[2])]
    assert raw.n_readouts == raw.encoding["blocks"] and not raw.provenance.notes
    assert raw.encoding["geometry"].startswith("pdata/1/visu_pars")
    counters = sorted(set(raw.counters("slice").tolist()))
    assert counters == list(range(len(slices)))  # one geometric slice per 2dseq slice
    frames = [(image, k) for image in images for k in range(image.shape[2])]
    for s in counters:
        image, k = frames[s]
        rows = raw.counters("slice") == s
        voxel = np.array([(image.shape[0] - 1) / 2, (image.shape[1] - 1) / 2, k, 1.0])
        centre_lps = np.diag([-1, -1, 1, 1]) @ image.affine @ voxel
        assert np.allclose(raw.headers["position"][rows], centre_lps[:3], atol=1e-3)
        axes = np.diag([-1.0, -1.0, 1.0]) @ image.affine[:3, :3]
        axes /= np.linalg.norm(axes, axis=0)
        read = raw.headers["read_dir"][rows][0]
        assert np.allclose(read, axes[:, 0], atol=1e-6) or np.allclose(read, axes[:, 1], atol=1e-6)
        assert np.allclose(np.abs(raw.headers["slice_dir"][rows][0]), np.abs(axes[:, 2]), atol=1e-6)
        reconstruction = _reconstruct(raw, s)
        best = max(
            np.corrcoef(orient(reconstruction).ravel(), slices[s].ravel())[0, 1]
            for orient in ORIENTATIONS
            if orient(reconstruction).shape == slices[s].shape
        )
        assert best > 0.999, f"{scan} slice {s}: correlation {best:.4f}"
