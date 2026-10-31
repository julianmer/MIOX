####################################################################################################
#                                        test_sequences.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/10/26                                                                                #
#                                                                                                  #
# Purpose: Pulseq files against independent evidence: the OSI2ONE scans ship the sequence that     #
#          acquired them beside the ISMRMRD raw data, so the ADC events, dwell time, matrix and    #
#          field of view of the sequence must match the acquired readouts; the block durations     #
#          must add up to the declared total. KomaMRI's comparison files (Pulseq 1.2 to 1.5) hold  #
#          MATLAB Pulseq's reading of every block as sampled waveforms: the events read here must  #
#          give the same amplitudes, timing and ADC samples. TOPPE's public RF module against what #
#          its writer guarantees and the flip angle its waveform gives; RF pulses: Bruker shape    #
#          files against the integral factor they state, a Siemens pulse table against the         #
#          integrals its header states, and the pulses of FSL-MRS's example sequences against      #
#          their flip angles.                                                                      #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import numpy as np
import pytest
import scipy.io

# own
from miox import PulseSequence, RFPulse, load
from tests.conftest import PUBLIC_DATA

OSI2ONE = sorted(set(p.parent for p in (PUBLIC_DATA / "osi2one").glob("**/sequence.seq")))
KOMA = PUBLIC_DATA / "komamri/KomaMRIFiles/test/test_files/pulseq/read_comparison"
KOMA_FILES = sorted(KOMA.glob("v*/*.seq"))
GAMMA = 42577468.8  # Hz/T, as the comparison files convert Hz to T
TOPPE = PUBLIC_DATA / "toppe"


@pytest.mark.parametrize("folder", OSI2ONE[:6], ids=lambda p: p.name[:40])
def test_osi2one_sequence_matches_its_raw_data(folder):
    raw_file = folder / "ismrmrd_data.h5"
    if not raw_file.is_file():
        pytest.skip("no raw data beside the sequence")
    [seq] = load(folder / "sequence.seq")
    [raw] = load(raw_file)
    assert isinstance(seq, PulseSequence) and seq.version.startswith("1.4")
    adc_blocks = seq.blocks[seq.blocks["adc"] != 0]
    assert len(adc_blocks) == raw.n_readouts
    dwells = {seq.events["adc"][int(b)]["dwell"] * 1e-9 for b in adc_blocks["adc"]}
    assert any(dwell == pytest.approx(raw.meta["DwellTime"]) for dwell in dwells)
    matrix = seq.definitions["encoding_dim"]
    oversampling = seq.definitions.get("ro_oversampling", 1)
    assert [matrix[0] * oversampling, matrix[1], matrix[2]] == raw.meta["EncodedMatrixSize"]
    fov = [v * 1000 for v in seq.definitions["FOV"]]
    encoded = raw.meta["EncodedFieldOfView"]
    assert fov == pytest.approx([encoded[0] / oversampling] + encoded[1:])
    assert seq.duration == pytest.approx(seq.definitions["TotalDuration"], rel=1e-9)
    assert seq.meta["EchoTime"] == seq.definitions["TE"]
    assert "LABELSET" in seq.extensions["specifications"]
    gradients = set(seq.events["trap"]) | set(seq.events.get("gradients", {}))
    for channel in ("gx", "gy", "gz"):
        assert set(seq.blocks[channel].tolist()) - {0} <= gradients



def _samples(event) -> tuple[np.ndarray, np.ndarray]:
    return np.atleast_1d(np.asarray(event.t, float)), np.atleast_1d(np.asarray(event.A))


@pytest.mark.parametrize("path", KOMA_FILES, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_pulseq_blocks_equal_matlab_pulseq(path):
    """Block by block against MATLAB Pulseq's waveforms (amplitudes in T and T/m): the same events
    present (MATLAB leaves out gradients of zero amplitude); RF peak = amplitude x peak of the
    magnitude shape; trapezoid peak, delay, rise, flat and fall; arbitrary gradient peak =
    amplitude x peak of its shape; ADC sample count and dwell. From version 1.4, where blocks
    state their duration, the block starts add up from the durations: RF and gradients start at
    their delay, ADC samples at delay + (k + 1/2) x dwell."""
    [seq] = load(path)
    matlab = np.atleast_1d(
        scipy.io.loadmat(path.with_suffix(".mat"), squeeze_me=True,
                         struct_as_record=False)["sequence"])
    assert seq.n_blocks == len(matlab)
    raster = seq.definitions.get("BlockDurationRaster")
    timed = seq.version >= "1.4"
    start = 0.0
    for block, theirs in zip(seq.blocks, matlab, strict=True):
        for name in ("rf", "gx", "gy", "gz", "adc"):
            t, a = _samples(getattr(theirs, name))
            event_id = int(block[name])
            table = "trap" if event_id in seq.events.get("trap", {}) else "gradients"
            if name[0] == "g" and event_id and seq.events[table][event_id]["amp"] == 0:
                event_id = 0
            assert bool(event_id) == (len(t) > 0), (int(block["id"]), name)
            if not event_id:
                continue
            if name == "rf":
                rf = seq.events["rf"][event_id]
                peak = rf["amp"] * np.abs(seq.shapes[rf["mag_id"]]).max()
                assert np.abs(a).max() * GAMMA == pytest.approx(peak, rel=1e-6)
                if timed:
                    rf_raster = seq.definitions["RadiofrequencyRasterTime"]
                    first = seq.shapes[rf["time_id"]][0] * rf_raster if rf["time_id"] else 0
                    assert t[0] - start == pytest.approx(rf["delay"] * 1e-6 + first,
                                                         abs=rf_raster)
            elif name == "adc":
                adc = seq.events["adc"][event_id]
                dwell = adc["dwell"] * 1e-9
                assert len(t) == adc["num"]
                assert np.diff(t) == pytest.approx(dwell, rel=1e-6)
                if timed:
                    assert t[0] - start == pytest.approx(adc["delay"] * 1e-6 + dwell / 2,
                                                         abs=1e-9)
            elif table == "trap":
                trap = seq.events["trap"][event_id]
                assert np.abs(a).max() * GAMMA == pytest.approx(abs(trap["amp"]), rel=1e-6)
                steps = [trap[k] * 1e-6 for k in ("rise", "flat", "fall")]
                assert np.diff(t)[:3] == pytest.approx(steps, abs=1e-9)
                if timed:
                    assert t[0] - start == pytest.approx(trap.get("delay", 0) * 1e-6, abs=1e-9)
            else:
                grad = seq.events["gradients"][event_id]
                peak = abs(grad["amp"]) * np.abs(seq.shapes[grad["shape_id"]]).max()
                assert np.abs(a).max() * GAMMA == pytest.approx(peak, rel=1e-6)
        if timed:
            start += block["duration"] * raster


def test_toppe_rf_module():
    """TOPPE's writer (writemod.m) stores every sample as an even int16 and the file must hold
    exactly the declared waveforms; the RF module's 608 samples of 4 us last 2.432 ms (its first
    float parameter), and the integral of its B1 gives a flip angle of 15.00 degrees, a round
    angle that a wrong amplitude scale or a shifted waveform would not give."""
    path = TOPPE / "tipdown.mod"
    if not path.is_file():
        pytest.skip("toppe/tipdown.mod not fetched")
    [seq] = load(path)
    module = seq.events["modules"][1]
    assert (module["raw"] % 2 == 0).all()
    assert module["points"] * 4e-6 == pytest.approx(module["float_parameters"][0] * 1e-3)
    flip = np.degrees(2 * np.pi * GAMMA * abs(module["rf"][:, 0, 0].sum()) * 4e-6)
    assert flip == pytest.approx(15.0, abs=0.01)


PTA = PUBLIC_DATA / "spant/inst/extdata/press_refocus.pta"
FSL_MRS = PUBLIC_DATA / "fsl_mrs_testdata/fsl_mrs/tests/testdata"


@pytest.mark.data
def test_siemens_pulse_table_against_its_integrals():
    """The pulse tool states AMPINT (sum of amplitude x cos(phase)), POWERINT (sum of
    amplitude^2) and ABSINT (sum of |amplitude|) of the samples it wrote."""
    if not PTA.exists():
        pytest.skip("spant not fetched")
    [pulse] = load(PTA)
    header = pulse.meta.vendor["header"]
    assert isinstance(pulse, RFPulse) and pulse.n_samples == 400 and pulse.amplitude.max() == 1
    assert np.sum(pulse.waveform.real) == pytest.approx(header["AMPINT"], abs=1e-6)
    assert np.sum(pulse.amplitude ** 2) == pytest.approx(header["POWERINT"], abs=1e-6)
    assert np.sum(np.abs(pulse.amplitude)) == pytest.approx(header["ABSINT"], abs=1e-6)


@pytest.mark.data
@pytest.mark.parametrize("relpath,flips", [("sim/example.json", (90, 90, 90)),
                                           ("mrs_io/fittingChallengePRESS.json", (90, 180, 180))])
def test_fsl_mrs_sequence_pulses_flip_as_designed(relpath, flips):
    """STEAM is three 90 degree pulses, PRESS 90-180-180: amplitudes in Hz, each sample lasting
    duration / samples, give flip = 360 x |sum of amplitude x exp(i phase) x dt| degrees."""
    path = FSL_MRS / relpath
    if not path.exists():
        pytest.skip("FSL-MRS test data not fetched")
    pulses = [d for d in load(path) if isinstance(d, RFPulse)]
    assert [p.amplitude_unit for p in pulses] == ["Hz"] * len(flips)
    for pulse, flip in zip(pulses, flips, strict=True):
        step = pulse.duration / pulse.n_samples
        assert 360 * abs(np.sum(pulse.waveform) * step) == pytest.approx(flip, abs=0.05)


SHAPES = sorted(PUBLIC_DATA.glob("**/spnam*"))


@pytest.mark.data
def test_bruker_shapes_against_their_integral_factor():
    """SHAPE_INTEGFAC is the shape's integral relative to a rectangle of the same length:
    |sum of amplitude x exp(i phase)| / (100 x samples), given to the digits written. Adiabatic
    shapes state a factor of their own and a few user-made shapes leave it at 1: not compared."""
    if not SHAPES:
        pytest.skip("no Bruker shape files fetched")
    compared = 0
    for path in SHAPES:
        [pulse] = load(path)
        header = pulse.meta.vendor["header"]
        stated, mode = header.get("SHAPE_INTEGFAC"), str(header.get("SHAPE_EXMODE", ""))
        if not isinstance(stated, float) or stated == 1.0 or "adiabatic" in mode.lower():
            continue
        digits = len(str(path.read_text().split("SHAPE_INTEGFAC=")[1].split()[0]).split("e")[0])
        integral = abs(np.sum(pulse.waveform)) / (100 * pulse.n_samples)
        assert integral == pytest.approx(stated, rel=10.0 ** -(digits - 3)), path.name
        compared += 1
    assert compared >= 20
