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
#          against the tabulated shifts of their solvent and reference signals.                    #
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
from miox import MRImage, MRSData, load
from tests.conftest import PUBLIC_DATA

BRUKER = PUBLIC_DATA / "bruker_data_test/data"
COFFEE = BRUKER / "flat/coffee/UV1009_M1-1003-1002_6268712_73uEjPg4XR"
ZIPPED = BRUKER / "zipped"
MTBLS1 = PUBLIC_DATA / "nmrml/examples/MTBLS1/FIDs"
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
