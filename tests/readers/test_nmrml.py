####################################################################################################
#                                          test_nmrml.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 10/10/26                                                                                #
#                                                                                                  #
# Purpose: nmrML reader on synthetic files: a compressed FID from a Bruker and from a Varian       #
#          source, a spectrum written from low to high shift, a complex spectrum without axis, a   #
#          draft file without sweep width, and the sniffing of nmrML files.                        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import base64
import zlib

import numpy as np
import pytest

# own
import miox as gu

SAMPLES = np.arange(8) + 1j * (10 - np.arange(8))


def encode(values: np.ndarray, compressed: bool) -> str:
    raw = values.tobytes()
    return base64.b64encode(zlib.compress(raw) if compressed else raw).decode()


def document(source: str = "Bruker TopSpin", fid: str = "", spectrum: str = "",
             sweep: str = '<sweepWidth value="1000" unitName="hertz"/>',
             frequency: str = '<irradiationFrequency value="100.0001" unitName="megaHertz"/>'
             ) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<nmrML xmlns="http://nmrml.org/schema" version="1.0.rc1">
  <sourceFileList><sourceFile id="a" name="fid"><cvParam name="{source}"/></sourceFile>
  </sourceFileList>
  <acquisition><acquisition1D>
    <acquisitionParameterSet numberOfScans="16">
      <sampleAcquisitionTemperature value="298.0" unitName="kelvin"/>
      <DirectDimensionParameterSet numberOfDataPoints="8">
        <acquisitionNucleus name="hydrogen atom"/>
        {frequency}
        <effectiveExcitationField value="100.0" unitName="megaHertz"/>
        {sweep}
      </DirectDimensionParameterSet>
    </acquisitionParameterSet>
    {fid}
  </acquisition1D></acquisition>
  <spectrumList>{spectrum}</spectrumList>
</nmrML>"""


def fid_element(values: np.ndarray, byte_format: str = "Complex128", compressed=True) -> str:
    return (f'<fidData byteFormat="{byte_format}" compressed="{str(compressed).lower()}">'
            f"{encode(values, compressed)}</fidData>")


def test_bruker_fid_is_conjugated(tmp_path):
    path = tmp_path / "bruker.nmrML"
    path.write_text(document(fid=fid_element(SAMPLES)))
    sniff = gu.sniff(path)
    assert (sniff.format, sniff.kind) == (gu.Format.NMRML, gu.Kind.SPECTROSCOPY)
    [d] = gu.load(path)
    assert isinstance(d, gu.MRSData) and d.shape == (1, 1, 1, 8)
    assert np.array_equal(d.data[0, 0, 0], SAMPLES.conj())
    assert d.dwelltime == pytest.approx(1e-3)
    assert d.spectrometer_frequency == pytest.approx(100.0001)
    assert d.meta["ReferencePPM"] == pytest.approx(1.0)  # (100.0001 - 100) / 100, in ppm
    assert d.meta["ResonantNucleus"] == ["1H"]
    assert d.meta["NumberOfAverages"] == 16 and d.meta["SampleTemperature"] == 298.0


def test_varian_fid_is_kept_and_int32_pairs_are_read(tmp_path):
    pairs = np.empty(16, "<i4")
    pairs[0::2], pairs[1::2] = SAMPLES.real, SAMPLES.imag
    path = tmp_path / "varian.nmrML"
    path.write_text(document(source="Varian VNMR format",
                             fid=fid_element(pairs, "Complex64Int", compressed=False)))
    [d] = gu.load(path)
    assert np.array_equal(d.data[0, 0, 0], SAMPLES)


def test_other_source_is_kept_with_a_note(tmp_path):
    path = tmp_path / "other.nmrML"
    path.write_text(document(source="JEOL", fid=fid_element(SAMPLES)))
    [d] = gu.load(path)
    assert np.array_equal(d.data[0, 0, 0], SAMPLES)
    assert any("not verified" in note for note in d.provenance.notes)


def test_spectrum_rising_axis_is_reversed(tmp_path):
    values = np.arange(8, dtype="<f8")
    spectrum = (f'<spectrum1D numberOfDataPoints="8"><xAxis unitName="parts per million" '
                f'startValue="-4.0" endValue="4.0"/><spectrumDataArray byteFormat="float64" '
                f'compressed="false">{encode(values, False)}</spectrumDataArray></spectrum1D>')
    path = tmp_path / "spectrum.nmrML"
    path.write_text(document(spectrum=spectrum))
    [image] = gu.load(path)
    assert isinstance(image, gu.MRImage) and image.dims[3] == "frequency"
    assert np.array_equal(image.data[0, 0, 0], values[::-1])
    # the span 8 ppm over 8 points: the first point at 4 ppm falls by 1 ppm per point
    assert image.meta["ReferencePPM"] == pytest.approx(0.0)
    assert image.meta["SpectralWidth"] == pytest.approx(8 * 100.0001)


def test_complex_spectrum_without_axis(tmp_path):
    spectrum = (f'<spectrum1D numberOfDataPoints="8"><spectrumDataArray byteFormat="complex128" '
                f'compressed="false">{encode(SAMPLES, False)}</spectrumDataArray></spectrum1D>')
    path = tmp_path / "bare.nmrML"
    path.write_text(document(spectrum=spectrum))
    [image] = gu.load(path)
    assert np.array_equal(image.data[0, 0, 0], SAMPLES)
    assert "ReferencePPM" not in image.meta and image.provenance.notes


def test_draft_without_sweep_width_lacks_the_dwell_time(tmp_path):
    path = tmp_path / "draft.nmrML"
    path.write_text(document(fid=fid_element(SAMPLES), sweep="",
                             frequency='<irradiationFrequency value="600.0" unitName="hertz"/>'))
    [d] = gu.load(path)
    assert "DwellTime" in d.missing
    assert d.spectrometer_frequency == pytest.approx(600.0)  # the draft's MHz labelled as hertz


def test_malformed_xml_is_a_reader_error(tmp_path):
    path = tmp_path / "broken.nmrML"
    path.write_text('<nmrML xmlns="http://nmrml.org/schema"><a <!--x--> /></nmrML>')
    with pytest.raises(gu.ReaderError):
        gu.load(path)
