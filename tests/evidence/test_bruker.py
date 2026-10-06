####################################################################################################
#                                          test_bruker.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 04/10/26                                                                                #
#                                                                                                  #
# Purpose: Bruker 2dseq images against the vendor's own DICOM export, and 2dseq spectra against    #
#          the magnitude spectra and raw FIDs of the same scans.                                   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import numpy as np
import pytest

# own
from miox import MRImage, MRSData, load
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
    stored frequency order, and the raw fid of the scan agrees with that order."""
    [fid] = load(BRUKER / scan / "pdata/2/2dseq")
    magnitude = _magnitude(scan)
    assert isinstance(fid, MRSData) and magnitude.dims[3] == "frequency"
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(fid.data, axis=3), axes=3))
    assert spectrum.shape == magnitude.shape
    assert np.allclose(spectrum, magnitude.data, rtol=1e-4, atol=1e-4 * float(magnitude.data.max()))
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
