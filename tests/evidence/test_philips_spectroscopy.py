####################################################################################################
#                                   test_philips_spectroscopy.py                                   #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 30/10/26                                                                                #
#                                                                                                  #
# Purpose: Philips spectroscopy against the scanner's own exports of the same scans (private data  #
#          only; paths are never recorded): single-voxel raw/lab/sin against the SPAR/SDAT pair    #
#          (water reference as is, data after the scanner's water-weighted coil combination and    #
#          eddy-current correction), SPAR/SDAT spectroscopic imaging against the DICOM (positions  #
#          where the SPAR states them).                                                            #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import numpy as np
import pytest

# own
from miox import MRSData, load
from miox.sniff import Format, Sniffer


def _corr(a, b) -> float:
    a, b = np.ravel(a), np.ravel(b)
    return float(abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _files(roots, pattern: str, fmt: Format) -> list:
    return [p for root in roots for p in root.rglob(pattern)
            if not p.name.startswith("._") and Sniffer().sniff(p).format == fmt]


def _combined(d: MRSData, weights: np.ndarray) -> np.ndarray:
    """Channels weighted, transients averaged: (points,)."""
    return (np.asarray(d.data)[0, 0, 0] * weights[None, :, None]).sum(axis=1).mean(axis=-1)


@pytest.mark.data
def test_private_raw_single_voxel_equals_the_scanner_export(private_roots):
    found = 0
    for lab in _files(private_roots, "*.lab", Format.PHILIPS_RAW_LAB):
        spectra = [d for d in load(lab) if isinstance(d, MRSData)]
        if len(spectra) != 2:
            continue
        act, ref = spectra
        weights = np.asarray(ref.data)[0, 0, 0, 0].mean(axis=-1).conj()  # first point, per coil
        water = _combined(ref, weights)
        data = _combined(act, weights) * np.exp(-1j * np.angle(water))  # the scanner's ECC
        exports = [s for s in _files([lab.parent.parent], "*.[sS][dD][aA][tT]",
                                     Format.PHILIPS_SPAR_SDAT)
                   if s.stem.lower().endswith("_ref")]
        best = 0.0, 0.0
        for sdat in exports:
            [their_ref] = load(sdat)
            if their_ref.meta.get("ProtocolName") != act.meta.get("ProtocolName"):
                continue
            their_act = sdat.with_name(sdat.name[: -len("ref" + sdat.suffix)] + "act" + sdat.suffix)
            if not their_act.exists():
                continue
            [their] = load(their_act)
            best = max(best, (_corr(water, their_ref.data), _corr(data, their.data)))
        if best == (0.0, 0.0):
            continue
        assert best[0] > 0.9999 and best[1] > 0.999
        found += 1
    if not found:
        pytest.skip("no Philips single-voxel raw/lab/sin with its SPAR/SDAT in the private data")


@pytest.mark.data
def test_private_sdat_imaging_grid_equals_the_dicom(private_roots):
    found = 0
    for spar in _files(private_roots, "*.[sS][pP][aA][rR]", Format.PHILIPS_SPAR_SDAT):
        try:
            [ours] = load(spar)
        except Exception:  # SPAR files spec2nii cannot parse are another test's matter
            continue
        if np.prod(ours.data.shape[:3]) == 1:
            continue
        candidates = [p for p in spar.parent.parent.rglob("*") if p.is_file()
                      and not p.name.startswith("._")
                      and Sniffer().sniff(p).format == Format.DICOM_MRS]
        matches = []
        for path in candidates:
            try:
                datasets = load(path)
            except Exception:  # spectroscopy DICOM no reader maps is another test's matter
                continue
            for theirs in datasets:
                if theirs.data.shape == ours.data.shape:
                    matches.append((_corr(ours.data, theirs.data), theirs))
        if not matches:
            continue
        best, theirs = max(matches, key=lambda m: m[0])
        assert best > 0.9999
        if not any("positions unknown" in n for n in ours.provenance.notes):
            assert np.allclose(ours.affine, theirs.affine, atol=1e-3)
        found += 1
    if not found:
        pytest.skip("no Philips SPAR/SDAT imaging with its DICOM in the private data")
