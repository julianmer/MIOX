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
#          of them, a Tecmag file against TNMR's own text export of it, Oxford Instruments and Q   #
#          Magnetics JCAMP-DX FIDs against the line patterns of their compounds, TopSpin zip       #
#          exports against their unpacked copies, NMRPipe FIDs against the Varian FIDs they were   #
#          converted from and against NMRPipe's own processing of them, a Sparky 15N-HSQC against #
#          the amide shifts assigned in its BMRB entry, SIMPSON output against the spin system it  #
#          simulates and against SIMPSON's own transform.                                          #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import re

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator
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


def fid_lines(d: MRSData, count: int, conjugate: bool = False) -> np.ndarray:
    """Shifts of the ``count`` tallest lines of the FID's magnitude spectrum, sorted."""
    samples = d.data[0, 0, 0]
    samples = samples.conj() if conjugate else samples
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(samples)))
    return peak_shifts(fid_axis(d), spectrum, count, max(3, d.shape[3] // 2000))


def matches(found: np.ndarray, expected: list[float], tolerance: float) -> bool:
    return bool(np.all(np.abs(found[:, None] - np.asarray(expected)[None, :]).min(axis=1)
                       < tolerance))


def test_oxford_fid_is_stored_in_the_convention():
    """Ethyl crotonate in CDCl3 at 60 MHz ($REVERSE= yes): the seven tallest lines, shifted by one
    common offset below 0.15 ppm (the file's SF is not calibrated on a reference), fall within
    0.02 ppm of the lines of its ethyl triplet (1.28 ppm, J 7.1 Hz), crotyl CH3 doublet of
    doublets (1.87 ppm, J 6.9 and 1.7 Hz) and the inner lines of the OCH2 quartet (4.18 ppm); the
    FID conjugated mirrors them to 5.8 - 8.6 ppm."""
    path = JCAMP / "oxfordInstruments/1h.jdx"
    if not path.is_file():
        pytest.skip("not fetched")
    [d] = load(path)
    hz = 1 / d.spectrometer_frequency
    expected = [1.28, 1.28 + 7.1 * hz] + [1.87 + a * 6.9 * hz / 2 + b * 1.7 * hz / 2
                                          for a in (-1, 1) for b in (-1, 1)] + \
        [4.18 - 7.1 * hz / 2, 4.18 + 7.1 * hz / 2]
    offsets = np.arange(-0.15, 0.15, 0.002)
    assert any(matches(fid_lines(d, 7) - o, expected, 0.02) for o in offsets)
    assert not any(matches(fid_lines(d, 7, conjugate=True) - o, expected, 0.02) for o in offsets)


def test_q_magnetics_fid_shows_ibuprofen_from_its_tms_line():
    """Ibuprofen with TMS in CDCl3 at 124 MHz, conjugated: measured from the TMS line (the lowest
    of the seven tallest), the lines lie within 0.03 ppm of the CH3 doublet (0.89 ppm), the
    CH-CH3 doublet (1.49 ppm, J 7.2 Hz) and the inner aromatic lines (7.10, 7.20 ppm); as stored
    they do not. The file's .SHIFT REFERENCE puts the TMS line at -13.38 ppm."""
    path = JCAMP / "qmagnetics/ibuprofen_j_ave_qmagnetics.jdx"
    if not path.is_file():
        pytest.skip("not fetched")
    [d] = load(path)
    hz = 1 / d.spectrometer_frequency
    expected = [0.0, 0.89 - 6.6 * hz / 2, 0.89 + 6.6 * hz / 2, 1.49 - 7.2 * hz / 2,
                1.49 + 7.2 * hz / 2, 7.10, 7.20]
    lines = fid_lines(d, 7)
    assert matches(lines - lines.min(), expected, 0.03)
    assert lines.min() == pytest.approx(-13.38, abs=0.02)
    stored = fid_lines(d, 7, conjugate=True)
    assert not matches(stored - stored.min(), expected, 0.03)


ZIPS = sorted(ZIPPED.rglob("*.zip")) if ZIPPED.is_dir() else []


@pytest.mark.parametrize("path", ZIPS, ids=lambda p: str(p.relative_to(ZIPPED)))
def test_topspin_zip_equals_its_unpacked_copy(path):
    """A zip export reads to the datasets of the folder the fetch script unpacked it into: same
    names, data and header; provenance paths point into the archive."""
    datasets = load(path)
    assert datasets
    for d in datasets:
        member = d.provenance.paths[0].relative_to(path)
        [twin] = [t for t in load(path.with_suffix("") / member) if t.name == d.name]
        assert np.array_equal(d.data, twin.data)
        assert dict(d.meta) == dict(twin.meta)


BMRB = PUBLIC_DATA / "bmrb"


def test_nmrpipe_fid_is_the_varian_fid_converted():
    """var2pipe converted the Rance-Kay 15N-HSQC of BMRB 15089: the first NMRPipe row read
    (conjugated) equals the difference of the first two Varian rows exactly (the echo and
    anti-echo combination), with the Varian frequency and spectral width."""
    pipe, folder = BMRB / "15089/test.fid", BMRB / "15089"
    if not pipe.is_file():
        pytest.skip("not fetched")
    [p], [v] = load(pipe), load(folder)
    rows, raw = p.data[0, 0, 0], v.data[0, 0, 0]
    assert np.array_equal(rows[:, 0], raw[:512, 0] - raw[:512, 1])
    assert p.spectrometer_frequency == pytest.approx(v.spectrometer_frequency, rel=1e-7)
    assert p.spectral_width == pytest.approx(v.spectral_width, rel=1e-6)


def test_nmrpipe_fid_transforms_to_nmrpipes_spectrum():
    """BMRB 17169 holds a 13C-HSQC both as converted by var2pipe and as processed by NMRPipe:
    transformed in both dimensions (States pairs as even - i odd) and resampled onto the axes
    of the processed spectrum, the FID read correlates with it (outside the water line) above
    0.5; with the direct dimension mirrored it does not reach 0.4."""
    fid_path = BMRB / "17169/pipe_conversion/C13_hsqc.fid"
    spectrum_path = BMRB / "17169/C13_hsqcB.ft"
    if not (fid_path.is_file() and spectrum_path.is_file()):
        pytest.skip("not fetched")
    [fid], [image] = load(fid_path), load(spectrum_path)
    header = image.meta.vendor
    direct = image_axis(image)
    rows = image.shape[4]
    indirect = (float(header["FDF1ORIG"]) + float(header["FDF1SW"]) * (
        1 - (np.arange(rows) + 1) / rows)) / float(header["FDF1OBS"])  # NMRPipe's F1 axis
    v = fid.meta.vendor
    x_axis = fid_axis(fid)
    n1 = fid.shape[4] // 2
    y_axis = float(v["FDF1CAR"]) - np.fft.fftshift(np.fft.fftfreq(n1, 1 / float(v["FDF1SW"]))) / \
        float(v["FDF1OBS"])
    keep = (direct < 4.4) | (direct > 5.1)
    reference = np.abs(image.data[0, 0, 0])

    def correlation(samples: np.ndarray) -> float:
        f2 = np.fft.fftshift(np.fft.fft(samples, axis=0), axes=0)
        spectrum = np.abs(np.fft.fftshift(np.fft.fft(f2[:, 0::2] - 1j * f2[:, 1::2], axis=1),
                                          axes=1))
        grid = RegularGridInterpolator((x_axis[::-1], y_axis[::-1]), spectrum[::-1, ::-1],
                                       bounds_error=False, fill_value=0.0)
        resampled = grid(np.stack(np.meshgrid(direct, indirect, indexing="ij"), axis=-1))
        return np.corrcoef(resampled[keep].ravel(), reference[keep].ravel())[0, 1]

    assert correlation(fid.data[0, 0, 0]) > 0.5
    assert correlation(fid.data[0, 0, 0].conj()) < 0.4


def assigned_shifts(path) -> dict[tuple[str, str], dict[str, float]]:
    """The assigned shifts of a BMRB NMR-STAR 3 entry by (entity, residue) and atom name."""
    lines = path.read_text().splitlines()
    i = next(k for k, line in enumerate(lines) if line.strip() == "_Atom_chem_shift.ID")
    columns = []
    while lines[i].strip().startswith("_Atom_chem_shift."):
        columns.append(lines[i].strip().split(".")[1])
        i += 1
    shifts: dict[tuple[str, str], dict[str, float]] = {}
    while lines[i].strip() != "stop_":
        fields = lines[i].split()
        if len(fields) == len(columns):
            row = dict(zip(columns, fields, strict=True))
            key = (row["Entity_ID"], row["Comp_index_ID"])
            shifts.setdefault(key, {})[row["Atom_ID"]] = float(row["Val"])
        i += 1
    return shifts


def test_sparky_hsqc_peaks_lie_on_the_assigned_amides():
    """The 15N-HSQC of BMRB 18465 as a Sparky file: with one referencing offset (within 0.1 ppm
    for 1H, 1 ppm for 15N), at least 75 % of its 60 tallest peaks lie within 0.03 / 0.3 ppm of
    an assigned amide (H, N) pair of the entry; with either axis mirrored no offset brings
    more than 40 %."""
    path, star = BMRB / "18465/o157_nhsqcv6.ucsf", BMRB / "18465/bmr18465_3.str"
    if not (path.is_file() and star.is_file()):
        pytest.skip("not fetched")
    pairs = np.array([(a["H"], a["N"]) for a in assigned_shifts(star).values()
                      if "H" in a and "N" in a])
    [image] = load(path)
    proton = image_axis(image)
    w1 = image.meta.vendor["w1"]
    rows = image.shape[4]
    nitrogen = float(w1["xmtr_freq"]) + (rows // 2 - np.arange(rows)) * float(
        w1["spectral_width"]) / float(w1["spectrometer_freq"]) / rows
    width = float(w1["spectral_width"]) / float(w1["spectrometer_freq"])
    magnitude = np.abs(image.data[0, 0, 0])
    maxima = np.argwhere(magnitude == maximum_filter(magnitude, size=9))
    tallest = maxima[np.argsort(magnitude[tuple(maxima.T)])[::-1][:60]]
    h, n = proton[tallest[:, 0]], nitrogen[tallest[:, 1]]

    def best(h: np.ndarray, n: np.ndarray) -> float:
        """The largest fraction of peaks on an amide pair over the offsets (15N folded)."""
        fractions = []
        for dh in np.arange(-0.1, 0.101, 0.01):
            for dn in np.arange(-1.0, 1.01, 0.1):
                near_h = np.abs(pairs[:, 0][None, :] - (h + dh)[:, None]) < 0.03
                dn_folded = (pairs[:, 1][None, :] - (n + dn)[:, None] + width / 2) % width
                near_n = np.abs(dn_folded - width / 2) < 0.3
                fractions.append(np.mean(np.any(near_h & near_n, axis=1)))
        return max(fractions)

    centre_h, centre_n = proton[proton.size // 2], nitrogen[rows // 2]
    assert best(h, n) >= 0.75
    assert best(2 * centre_h - h, n) < 0.4
    assert best(h, 2 * centre_n - n) < 0.4


SIMPYSON = PUBLIC_DATA / "simpyson/examples/read"


def test_simpson_fid_shows_ethanol_at_its_simulated_shifts():
    """ethanol.in simulates CH3, CH2 and OH at 1, 3.5 and 7 ppm (3:2:1, J 30 Hz between CH3 and
    CH2) at 400 MHz relative to the carrier: the FID read (conjugated) puts its tallest line at
    -400 Hz in the transform, i.e. at +1 ppm on the NIfTI-MRS axis (ppm = -f / 400 MHz), the OH
    singlet at +7 ppm; SIMPSON's CSV export of the same FID equals the samples as stored."""
    path = SIMPYSON / "ethanol.fid"
    if not path.is_file():
        pytest.skip("not fetched")
    [d] = load(path)
    samples = d.data[0, 0, 0]
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(samples)))
    ppm = -np.fft.fftshift(np.fft.fftfreq(samples.size, d.dwelltime)) / 400.0
    lines = peak_shifts(ppm, spectrum, 2, 21)
    assert lines == pytest.approx([1.0, 7.0], abs=0.01)
    assert ppm[np.argmax(spectrum)] == pytest.approx(1.0, abs=0.01)
    table = np.loadtxt(SIMPYSON / "fid.csv", delimiter=",", skiprows=1)
    assert np.allclose(samples, table[:, 0] - 1j * table[:, 1], rtol=1e-6, atol=1e-6)


def test_simpson_spectrum_is_the_transform_of_its_fid():
    """ethanol.spe is SIMPSON's transform of ethanol.fid (saved after it): read, it equals the
    transform of the FID read with its first point halved, to single precision."""
    fid_path, spe_path = SIMPYSON / "ethanol.fid", SIMPYSON / "ethanol.spe"
    if not (fid_path.is_file() and spe_path.is_file()):
        pytest.skip("not fetched")
    [d], [image] = load(fid_path), load(spe_path)
    samples = d.data[0, 0, 0].copy()
    samples[0] /= 2
    expected = np.fft.fftshift(np.fft.fft(samples))
    assert np.allclose(image.data[0, 0, 0], expected, atol=1e-4 * np.abs(expected).max())

