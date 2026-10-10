####################################################################################################
#                                           test_nmr.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 09/10/26                                                                                #
#                                                                                                  #
# Purpose: NMR spectrometer files against independent evidence: Bruker FIDs against TopSpin's own  #
#          processing of them (the spectrum of the FID read lies on the chemical shift axis of the #
#          processed spectrum), an acquisition stopped early against its audit trail, Varian FIDs  #
#          against the tabulated shifts of their solvent and reference signals, TopSpin's JCAMP-DX #
#          exports against the TopSpin files they were written from, a Nanalysis JCAMP-DX FID      #
#          against the shift of its compound's tallest signal, nmrML files against the vendor      #
#          files they were converted from, JEOL Delta files against spectra other software made    #
#          of them, a Tecmag file against TNMR's own text export of it.                            #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re

import numpy as np
import pytest
from scipy.ndimage import maximum_filter

# own
from miox import MRImage, MRSData, load
from tests.conftest import PUBLIC_DATA

BRUKER = PUBLIC_DATA / "bruker_data_test/data"
COFFEE = BRUKER / "flat/coffee/UV1009_M1-1003-1002_6268712_73uEjPg4XR"
ZIPPED = BRUKER / "zipped"
MTBLS1 = PUBLIC_DATA / "nmrml/examples/MTBLS1/FIDs"
JCAMP = PUBLIC_DATA / "jcamp_data_test/data/nmr"
EXPORTS = [  # (TopSpin's JCAMP-DX FID export, the experiment; the same acquisition by data hash)
    (JCAMP / "aspirin/1h.fid.dx", ZIPPED / "aspirin-1h/1"),
    (JCAMP / "naphtoicAcid/1h.fid.dx", ZIPPED / "naphtoicAcid-1h/1"),
]
HOPS = PUBLIC_DATA / "nmrml/examples/IPB_HopExample/FIDs"
HMDB = PUBLIC_DATA / "nmrml/examples/reference_spectra_examples/hmdb"
SIGNALS = [  # (Varian .fid folder, window in ppm, shift of its tallest signal, tolerance)
    # residual CHD2OD and HOD in methanol-d4 (Gottlieb, Kotlyar, Nudelman 1997)
    (HOPS / "FAM013_AHTM.PROTON_04.fid", (3.2, 3.4), 3.31, 0.02),
    (HOPS / "FAM013_AHTM.PROTON_04.fid", (4.7, 5.0), 4.87, 0.03),
    (HOPS / "FAM013_CZAG.PROTON_02.fid", (3.2, 3.4), 3.31, 0.02),
    (HOPS / "FAM013_TPE.PROTON_02.fid", (3.2, 3.4), 3.31, 0.02),
    # the DSS reference of an HMDB spectrum in water
    (HMDB / "HMDB00005.fid", (-0.2, 0.2), 0.0, 0.01),
]
EXPERIMENTS = [  # 1D experiments with a fid and TopSpin's processed spectrum in pdata/1
    COFFEE / "20", COFFEE / "21", COFFEE / "22",
    ZIPPED / "cyclosporin/cyclosporin_1h/cyclosporin_1h/1",
    ZIPPED / "ibuprofen/processed/proton/1", ZIPPED / "ibuprofen/processed/carbon/2",
    ZIPPED / "rb87/rb87/13", ZIPPED / "strychnine-1h/strychnine/10",
    ZIPPED / "topspin365/topspin365/10",
    MTBLS1 / "ADG10003u_007/10", MTBLS1 / "ADG10003u_051/10",
]


def tallest(d: MRSData, low: float, high: float) -> float:
    """Shift of the tallest point of the magnitude spectrum between ``low`` and ``high``."""
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(d.data[0, 0, 0])))
    ppm = ppm_axis(d.shape[3], d.spectral_width, d.spectrometer_frequency, d.meta["ReferencePPM"])
    window = (ppm > low) & (ppm < high)
    return float(ppm[window][np.argmax(spectrum[window])])


def ppm_axis(n: int, width: float, frequency: float, reference: float) -> np.ndarray:
    """Chemical shift of each point of fftshift(fft(fid)): ReferencePPM - f / frequency."""
    return reference - np.fft.fftshift(np.fft.fftfreq(n, 1 / width)) / frequency


@pytest.mark.parametrize("folder", EXPERIMENTS, ids=lambda p: str(p.relative_to(PUBLIC_DATA)))
def test_fid_spectrum_lies_on_topspin_axis(folder):
    """The magnitude spectrum of the FID read, resampled from its stated axis onto the axis of
    TopSpin's processed spectrum, matches that spectrum (correlation above 0.95) best without
    any shift (lags of up to 30 points tried). The FID as nmrglue reads it (unconjugated)
    mirrors the axis and matches nowhere."""
    if not folder.is_dir():
        pytest.skip("not fetched")
    fid, image = load(folder)[:2]
    assert isinstance(fid, MRSData) and isinstance(image, MRImage)
    processed = np.abs(image.data[0, 0, 0])
    theirs = ppm_axis(image.shape[3], image.meta["SpectralWidth"],
                      image.meta["SpectrometerFrequency"][0], image.meta["ReferencePPM"])
    ours = ppm_axis(fid.shape[3], fid.spectral_width, fid.spectrometer_frequency,
                    fid.meta["ReferencePPM"])
    order = np.argsort(ours)
    lags = np.arange(-30, 31)
    for samples, aligned in ((fid.data[0, 0, 0], True), (fid.data[0, 0, 0].conj(), False)):
        spectrum = np.abs(np.fft.fftshift(np.fft.fft(samples)))
        resampled = np.interp(theirs, ours[order], spectrum[order])
        match = [np.corrcoef(np.roll(resampled, lag), processed)[0, 1] for lag in lags]
        if aligned:
            assert lags[np.argmax(match)] == 0 and max(match) > 0.95
        else:
            assert max(match) < 0.3


def test_stopped_acquisition_keeps_the_rows_of_its_audit_trail():
    """The HSQC was stopped after 101 of 256 increments: acqu2s TD and the audit trail ("data
    hash MD5: 2K * 101") state 101 rows, the ser holds 256."""
    folder = ZIPPED / "hsqc-incompleted-acquisition/5"
    if not folder.is_dir():
        pytest.skip("not fetched")
    audit = (folder / "audita.txt").read_text()
    rows = int(re.search(r"data hash MD5: \d+K \* (\d+)", audit).group(1))
    d = load(folder)[0]
    assert rows == 101 and d.shape[3:] == (1024, rows)
    assert (folder / "ser").stat().st_size == 256 * 2048 * 4


@pytest.mark.parametrize("folder,window,shift,tolerance", SIGNALS,
                         ids=[f"{s[0].name}-{s[2]}" for s in SIGNALS])
def test_varian_signal_at_its_tabulated_shift(folder, window, shift, tolerance):
    """On the axis stated by rfl, rfp and sfrq the solvent and reference signals lie at their
    tabulated shifts; the FIDs are not conjugated (mirrored, the methanol signal of the hop
    extracts would lie near 12.7 ppm)."""
    if not folder.is_dir():
        pytest.skip("not fetched")
    [d] = load(folder)
    assert tallest(d, *window) == pytest.approx(shift, abs=tolerance)


@pytest.mark.parametrize("export,folder", EXPORTS, ids=[e[0].parent.name for e in EXPORTS])
def test_jcamp_fid_export_equals_topspin_fid(export, folder):
    """TopSpin wrote the FID of the experiment to JCAMP-DX: both reads give the same samples,
    dwell time, carrier shift and digital filter delay."""
    if not (export.is_file() and folder.is_dir()):
        pytest.skip("not fetched")
    jcamp, topspin = load(export)[0], load(folder)[0]
    assert np.array_equal(jcamp.data, topspin.data)
    assert jcamp.dwelltime == pytest.approx(topspin.dwelltime, rel=1e-12)
    for key in ("ReferencePPM", "GroupDelay", "SpectrometerFrequency"):
        assert jcamp.meta[key] == pytest.approx(topspin.meta[key], abs=1e-9), key


def test_jcamp_spectrum_export_matches_topspin_processing():
    """TopSpin's JCAMP-DX export of the aspirin spectrum (X in Hz from the spectral width down
    to 0, .SHIFT REFERENCE placing point 1) and the processed spectrum of the same acquisition
    in pdata (OFFSET, SW_p, SI; another phase correction) lie on the same axis within a quarter
    point and match in magnitude."""
    export, pdata = JCAMP / "aspirin/1h.dx", ZIPPED / "aspirin-1h-processed/1/pdata/1"
    if not (export.is_file() and pdata.is_dir()):
        pytest.skip("not fetched")
    jcamp, topspin = load(export)[0], load(pdata)[0]
    assert jcamp.shape == topspin.shape
    width = topspin.meta["SpectralWidth"]
    step = width / topspin.shape[3] / topspin.meta["SpectrometerFrequency"][0]
    assert jcamp.meta["SpectralWidth"] == pytest.approx(width, rel=1e-9)
    assert jcamp.meta["ReferencePPM"] == pytest.approx(topspin.meta["ReferencePPM"],
                                                       abs=step / 4)
    magnitudes = np.abs(jcamp.data.ravel()), np.abs(topspin.data.ravel())
    assert np.corrcoef(*magnitudes)[0, 1] > 0.9999


def test_nanalysis_fid_shows_ibuprofen_at_its_shift():
    """Ibuprofen in CDCl3 at 60 MHz: the tallest signal, the isobutyl methyl doublet at 0.89 ppm
    (lines 0.055 ppm apart at 60 MHz), lies there on the stated axis; conjugation into the
    NIfTI-MRS convention is needed (as stored, it would lie near 11.1 ppm)."""
    path = JCAMP / "nanalysis/1h.jdx"
    if not path.is_file():
        pytest.skip("not fetched")
    [d] = load(path)
    assert tallest(d, -1.0, 13.0) == pytest.approx(0.89, abs=0.07)


NMRML = PUBLIC_DATA / "nmrml/examples"


def test_nmrml_varian_fid_equals_the_vnmr_fid():
    """The IPB nmrML (v2) holds the samples of the Varian fid it was converted from, with the
    same dwell time, frequency and scans; 0 ppm agrees to 1e-4 ppm."""
    nmrml = NMRML / "IPB_HopExample/nmrMLs.v2/FAM013_AHTM.PROTON_04.fid.nmrML"
    folder = HOPS / "FAM013_AHTM.PROTON_04.fid"
    if not (nmrml.is_file() and folder.is_dir()):
        pytest.skip("not fetched")
    [a], [b] = load(nmrml), load(folder)
    assert np.array_equal(a.data, b.data)
    assert a.dwelltime == pytest.approx(b.dwelltime, rel=1e-12)
    assert a.spectrometer_frequency == pytest.approx(b.spectrometer_frequency, rel=1e-12)
    assert a.meta["ReferencePPM"] == pytest.approx(b.meta["ReferencePPM"], abs=1e-4)
    assert a.meta["NumberOfAverages"] == b.meta["NumberOfAverages"]


def test_nmrml_bruker_fid_and_spectrum_equal_the_topspin_files():
    """The MTBLS1 jnmrML FID equals the Bruker fid sample for sample (both conjugated), and its
    spectrum lies on the axis of TopSpin's processed spectrum (same first point, same shift at
    the centre, tallest signal within a point; the processing itself differs)."""
    nmrml = NMRML / "MTBLS1/nmrMLs/ADG10003u_007-full-jnmrML.nmrML"
    if not (nmrml.is_file() and (MTBLS1 / "ADG10003u_007/10").is_dir()):
        pytest.skip("not fetched")
    fid, spectrum = load(nmrml)
    bruker_fid, processed = load(MTBLS1 / "ADG10003u_007/10")
    assert np.array_equal(fid.data, bruker_fid.data)
    assert fid.dwelltime == pytest.approx(bruker_fid.dwelltime, rel=1e-12)
    assert spectrum.meta["ReferencePPM"] == pytest.approx(processed.meta["ReferencePPM"], abs=1e-5)
    assert spectrum.meta["SpectralWidth"] == pytest.approx(processed.meta["SpectralWidth"],
                                                           rel=1e-5)
    ours, theirs = np.abs(spectrum.data.ravel()), np.abs(processed.data.ravel())
    assert abs(int(np.argmax(ours)) - int(np.argmax(theirs))) <= 1
    assert np.corrcoef(ours, theirs)[0, 1] > 0.85


JEOL = PUBLIC_DATA / "jeol_data_test/data"
RUTIN = "Rutin_3080ug200uL_DMSOd6_{}_400MHz_Jeol.jdf"


def peak_shifts(ppm: np.ndarray, spectrum: np.ndarray, count: int, size: int) -> np.ndarray:
    """Shifts of the ``count`` tallest local maxima (within ``size`` points), sorted."""
    maxima = np.flatnonzero(spectrum == maximum_filter(spectrum, size=size))
    return np.sort(ppm[maxima[np.argsort(spectrum[maxima])[::-1][:count]]])


def fid_axis(d: MRSData) -> np.ndarray:
    return ppm_axis(d.shape[3], d.spectral_width, d.spectrometer_frequency, d.meta["ReferencePPM"])


def image_axis(image: MRImage) -> np.ndarray:
    """Shift of each point of a spectrum from ReferencePPM (point N / 2) and SpectralWidth."""
    n = image.shape[3]
    step = image.meta["SpectralWidth"] / image.meta["SpectrometerFrequency"][0] / n
    return image.meta["ReferencePPM"] - (np.arange(n) - n // 2) * step


def test_jeol_fid_shows_the_peaks_of_the_spectrum_mestrenova_processed():
    """The eight tallest lines of the Rutin proton FID's spectrum lie within 0.003 ppm of
    those in the spectrum MestReNova exported from the same .jdf, with the FID as read; the
    FID as nmrglue returns it (conjugated) puts the tallest line at 2.58 ppm instead of 2.46."""
    path, export = JEOL / RUTIN.format("qHNMR"), JCAMP / "rutin/qH.jdx"
    if not (path.is_file() and export.is_file()):
        pytest.skip("not fetched")
    [fid], [spectrum] = load(path), load(export)
    ours = peak_shifts(fid_axis(fid), np.abs(np.fft.fftshift(np.fft.fft(fid.data[0, 0, 0]))), 8,
                       201)
    theirs = peak_shifts(image_axis(spectrum), np.abs(spectrum.data.ravel()), 8, 801)
    shared = [t for t in theirs if np.min(np.abs(ours - t)) < 0.003]
    assert len(shared) >= 7 and tallest(fid, 0.0, 12.0) == pytest.approx(2.461, abs=0.003)
    flipped = np.abs(np.fft.fftshift(np.fft.fft(fid.data[0, 0, 0].conj())))
    assert abs(fid_axis(fid)[np.argmax(flipped)] - 2.461) > 0.05


def test_jeol_fid_spectrum_lies_on_the_axis_of_its_processed_spectrum():
    """The same proton experiment as an FID and as a spectrum processed by Delta: three of the
    four tallest lines of the spectrum lie at shifts of the FID's eight tallest, and the magnitude
    spectra correlate."""
    fid_path = JEOL / "PM032220_3000U200U_Bin_180218_qH-1-1.jdf"
    spectrum_path = JEOL / "PM032220_3000U200U_Bin_180218_qH-1-4.jdf"
    if not (fid_path.is_file() and spectrum_path.is_file()):
        pytest.skip("not fetched")
    [fid], [image] = load(fid_path), load(spectrum_path)
    axis, mine = fid_axis(fid), np.abs(np.fft.fftshift(np.fft.fft(fid.data[0, 0, 0])))
    theirs = np.abs(image.data.ravel())
    ours = peak_shifts(axis, mine, 8, 201)
    reference = peak_shifts(image_axis(image), theirs, 4, 401)
    assert (np.abs(reference[:, None] - ours[None, :]).min(axis=1) < 0.005).sum() >= 3
    order = np.argsort(axis)
    resampled = np.interp(image_axis(image), axis[order], mine[order])
    assert np.corrcoef(resampled, theirs)[0, 1] > 0.8


def test_jeol_spectrum_axis_is_that_of_the_points_kept():
    """Two proton spectra of one sample, processed with different carriers (6 and 5 ppm), show
    their lines at the same shifts only if the header's axis range is taken as spanning the
    points kept; the water line in DMSO-d6 lies at 3.33 ppm (Gottlieb et al. 1997)."""
    on = JEOL / "8PA_SynLK_5360u150uDMSO_snc1811_qH_SpinOn-1-2.jdf"
    off = JEOL / "8PA_SynLK_5360u150uDMSO_snc1811_qH_spinOff-1-2.jdf"
    if not (on.is_file() and off.is_file()):
        pytest.skip("not fetched")
    [a], [b] = load(on), load(off)
    lines = [peak_shifts(image_axis(x), np.abs(x.data.ravel()), 5, 801) for x in (a, b)]
    for shift in (1.581, 1.712, 6.239, 6.709):
        assert all(np.min(np.abs(found - shift)) < 0.003 for found in lines)
    assert np.min(np.abs(peak_shifts(image_axis(b), np.abs(b.data.ravel()), 6, 801) - 3.33)) < 0.01


def test_jeol_carbon_fid_shows_dmso_and_the_temperatures_follow_the_file_names():
    """The DMSO-d6 septet of the carbon FID lies within 0.6 ppm of 39.52 ppm; temp_get is in
    degrees Celsius: the files named 278k and 255K state 4.8 and -18.1 degrees."""
    path = JEOL / RUTIN.format("13CNMR")
    cold = ((JEOL / "12_pm033236_400JEOL_278k carbon_C.jdf", 278.0),
            (JEOL / "PM031528 255K_400JEOL_C.jdf", 255.0))
    if not (path.is_file() and all(p.is_file() for p, _ in cold)):
        pytest.skip("not fetched")
    [carbon] = load(path)
    assert tallest(carbon, 30.0, 50.0) == pytest.approx(39.52, abs=0.6)
    for file, kelvin in cold:
        assert load(file)[0].meta["SampleTemperature"] == pytest.approx(kelvin, abs=0.3)


@pytest.mark.parametrize("name", ["HSQC", "HMBC", "COSY"])
def test_jeol_2d_rows_are_direct_fids_in_the_same_convention(name):
    """The rows of a 2D FID, transformed along the direct dimension and summed, resemble the
    proton spectrum of the same sample (correlation above 0.3, in the even and in the odd rows);
    mirrored about the carrier they do not (below 0.1). The other pairing of the sections
    mirrors the direct axis."""
    path, proton = JEOL / RUTIN.format(name), JEOL / RUTIN.format("qHNMR")
    if not (path.is_file() and proton.is_file()):
        pytest.skip("not fetched")
    [d], [q] = load(path), load(proton)
    assert d.meta["ResonantNucleus"][0] == "1H" and len(d.meta["SpectrometerFrequency"]) == 2
    reference = np.argsort(fid_axis(q))
    q_axis, q_spectrum = fid_axis(q)[reference], np.abs(np.fft.fftshift(np.fft.fft(
        q.data[0, 0, 0])))[reference]
    carrier = d.meta["ReferencePPM"]
    axis = ppm_axis(d.shape[3], d.spectral_width, d.meta["SpectrometerFrequency"][0], carrier)
    rows = d.data[0, 0, 0]
    window = (axis > -0.5) & (axis < 11.0)
    for part in ((rows[:, 0::2], rows[:, 1::2]) if name == "HSQC" else (rows,)):
        summed = np.abs(np.fft.fftshift(np.fft.fft(part, axis=0), axes=0)).sum(axis=1)
        same = np.interp(axis, q_axis, q_spectrum)
        mirrored = np.interp(2 * carrier - axis, q_axis, q_spectrum)
        assert np.corrcoef(summed[window], same[window])[0, 1] > 0.3
        assert np.corrcoef(summed[window], mirrored[window])[0, 1] < 0.1


def test_jeol_increment_times_follow_the_indirect_sweep():
    """Increment times come from the header: 256 increments at 1 / y_sweep, two rows each."""
    path = JEOL / RUTIN.format("HSQC")
    if not path.is_file():
        pytest.skip("not fetched")
    [d] = load(path)
    times = np.asarray(d.dim_info[5]["header"]["IndirectTime"]["Value"])
    assert times.size == d.shape[4] == 512
    assert np.array_equal(times[0::2], times[1::2])
    assert np.diff(times[0::2]).mean() == pytest.approx(1 / 17099.86, rel=1e-5)


def test_tecmag_fid_equals_tnmr_text_export():
    """TNMR listed the FID of LiCl_ref1.tnt as text (real, imaginary, time in us): the samples
    read equal the listing and the dwell time equals its time step."""
    path = PUBLIC_DATA / "nmrglue/tecmag/LiCl_ref1.tnt"
    listing = path.with_suffix(".txt")
    if not (path.is_file() and listing.is_file()):
        pytest.skip("not fetched")
    [d] = load(path)
    table = np.loadtxt(listing, skiprows=3)
    assert np.array_equal(d.data[0, 0, 0], table[:, 0] + 1j * table[:, 1])
    assert d.dwelltime == pytest.approx((table[1, 2] - table[0, 2]) * 1e-6, rel=1e-12)
    assert d.meta["ResonantNucleus"] == ["7Li"]
