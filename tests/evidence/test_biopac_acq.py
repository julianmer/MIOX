####################################################################################################
#                                        test_biopac_acq.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/10/26                                                                                #
#                                                                                                  #
# Purpose: BIOPAC AcqKnowledge files against independent evidence: bidsphysio's BIDS conversion    #
#          of its sample, the compressed twin of every file (one zlib stream per channel instead   #
#          of interleaved samples), the same recordings saved by AcqKnowledge 3.8.1 to 5.0.1, and  #
#          the journal text bioread's authors saved from AcqKnowledge.                             #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import html.parser

import numpy as np
import pytest

# own
from miox import load
from tests.conftest import PUBLIC_DATA

ACQ2BIDS = PUBLIC_DATA / "bidsphysio/bidsphysio.acq2bids/tests/data"
BIOREAD = PUBLIC_DATA / "bioread/test"
VERSIONS = ("3.8.1", "3.9.1", "4.1.0", "4.2.0", "4.3.0", "4.4.0", "5.0.1")
RECORDINGS = ("physio", "nojournal")


def _need(*paths):
    for path in paths:
        if not path.is_file():
            pytest.skip(f"{path.relative_to(PUBLIC_DATA)} not fetched")


def _file(recording: str, version: str, compressed: bool = False):
    path = BIOREAD / "data" / recording / f"{recording}-{version}{'-c' if compressed else ''}.acq"
    _need(path)
    return path


def _whole_patterns(datasets) -> list[int]:
    """Rows of each dataset up to the end of its slowest channel's last sample: the part every
    version stores alike (versions differ after it, see the test below)."""
    vendor = datasets[0].meta.vendor
    end = min(c["points"] * c["divider"] for c in vendor["channels"])
    base = 1000 / vendor["graph_header"]["sample_time_ms"]
    return [-(-end // round(base / d.sampling_frequency)) for d in datasets]


def test_sample_equals_the_bids_conversion():
    """bidsphysio converted its sample to BIDS (values to 4 decimals): pulse, respiration and GSR
    agree to the rounding, the trigger column rises where the digital input does, at sample 300,
    which is the StartTime of -0.6 s it states at 500 Hz."""
    _need(ACQ2BIDS / "sample.acq", ACQ2BIDS / "acq_physio.tsv")
    [ours] = load(ACQ2BIDS / "sample.acq")
    theirs = np.loadtxt(ACQ2BIDS / "acq_physio.tsv")
    assert ours.channels == ("Pulse", "Resp", "GSR", "Digital input")
    assert ours.sampling_frequency == 500.0
    assert ours.data.shape == theirs.shape
    assert np.abs(ours.data[:, :3] - theirs[:, :3]).max() <= 5e-5 + 1e-12
    rise = np.flatnonzero(theirs[:, 3] > 0)[0]
    assert rise == np.flatnonzero(ours.data[:, 3] > 0)[0] == 300
    assert -rise / ours.sampling_frequency == pytest.approx(-0.6)


@pytest.mark.parametrize("recording", RECORDINGS)
@pytest.mark.parametrize("version", VERSIONS)
def test_interleaved_samples_equal_the_compressed_twin(recording, version):
    """AcqKnowledge saved each file also compressed, every channel as its own zlib stream: the
    interleaved samples equal them (channels with dividers 1, 2 and 512). Files of 3.8.1 agree
    up to the end of the slowest channel; after it AcqKnowledge 3.8.1 wrote the remaining
    samples of the faster channels differently in its two files."""
    plain = load(_file(recording, version))
    packed = load(_file(recording, version, compressed=True))
    assert [d.channels for d in plain] == [d.channels for d in packed]
    rows = _whole_patterns(plain) if version == "3.8.1" else [len(d.data) for d in plain]
    for a, b, n in zip(plain, packed, rows, strict=True):
        assert a.sampling_frequency == b.sampling_frequency
        assert np.array_equal(a.data[:n], b.data[:n], equal_nan=True)


@pytest.mark.parametrize("recording", RECORDINGS)
@pytest.mark.parametrize("version", VERSIONS)
def test_versions_hold_the_same_recording(recording, version):
    """The recordings were saved by AcqKnowledge 3.8.1 to 5.0.1 (little- and big-endian, header
    layouts before and after version 4): channels, units, rates, samples (up to the end of the
    slowest channel) and markers equal those of the 4.4.0 file. The compressed 5.0.1 physio
    file stores every marker shifted (some past the recording), so its markers stay out of the
    events."""
    reference = load(_file(recording, "4.4.0"))
    for compressed in (False, True):
        ours = load(_file(recording, version, compressed))
        assert [d.channels for d in ours] == [d.channels for d in reference]
        units = [c["units"] for c in ours[0].meta.vendor["channels"]]
        assert units == [c["units"] for c in reference[0].meta.vendor["channels"]]
        for a, b, n in zip(ours, reference, _whole_patterns(reference), strict=True):
            assert a.sampling_frequency == b.sampling_frequency
            assert np.array_equal(a.data[:n], b.data[:n], equal_nan=True)
        markers = [(m["sample"], m["text"]) for m in ours[0].meta.vendor["markers"]]
        expected = [(m["sample"], m["text"]) for m in reference[0].meta.vendor["markers"]]
        if (recording, version, compressed) == ("physio", "5.0.1", True):
            assert [t for _, t in markers] == [t for _, t in expected]
            assert all(not d.events for d in ours)
        else:
            assert markers == expected


class _Text(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = ""

    def handle_data(self, data):
        self.text += data


@pytest.mark.parametrize("version", VERSIONS)
def test_journals_equal_the_saved_text(version):
    """The journal of the physio recording equals the text saved from AcqKnowledge: plain text
    up to 4.1.0, HTML from 4.2.0 (compared by its text)."""
    _need(BIOREAD / "fixtures/journal.txt", BIOREAD / "fixtures/journal.html")
    for compressed in (False, True):
        [d, *_] = load(_file("physio", version, compressed))
        journal = d.meta.vendor["journal"]
        if version < "4.2.0":
            saved = (BIOREAD / "fixtures/journal.txt").read_text()
            assert journal.replace("\r\n", "\n").strip() == saved.replace("\r\n", "\n").strip()
        else:
            ours, saved = _Text(), _Text()
            ours.feed(journal)
            saved.feed((BIOREAD / "fixtures/journal.html").read_text())
            assert ours.text == saved.text
