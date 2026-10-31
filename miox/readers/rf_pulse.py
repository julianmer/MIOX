####################################################################################################
#                                           rf_pulse.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 30/10/26                                                                                #
#                                                                                                  #
# Purpose: Readers for RF pulse shapes: Siemens pulse tables (.pta), Bruker shape files (JCAMP-DX  #
#          Shape Data: TopSpin waves, ParaVision spnam) and the sequence descriptions of the       #
#          FSL-MRS simulator (JSON: RF pulses, delays, gradients, receiver).                       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import json
import re

import numpy as np

# own
from miox.data import Protocol, RFPulse
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                         SiemensPtaReader                                         #
#**************************************************************************************************#
#                                                                                                  #
# "NAME: value" header lines (PULSENAME, COMMENT, REFGRAD, MINSLICE, MAXSLICE, AMPINT, POWERINT,   #
# ABSINT; in some files the first is written ePULSENAME), then one "amplitude phase ; (index)"     #
# line per sample: the amplitude scaled to a peak of 1, the phase in radians (checked: AMPINT =    #
# sum of amplitude x cos(phase), POWERINT = sum of amplitude^2, ABSINT = sum of |amplitude|). The  #
# duration is not stored; the sequence sets it.                                                    #
#                                                                                                  #
#**************************************************************************************************#
class SiemensPtaReader(Reader):
    FORMATS = (Format.SIEMENS_PTA,)
    PACKAGES = ("miox",)
    HEADER = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*):\s*(.*?)\s*$")
    SAMPLE = re.compile(r"^\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*;")

    def read(self, sniff: Sniff) -> list[RFPulse]:
        header, samples = {}, []
        for line in sniff.path.read_text(errors="replace").splitlines():
            if match := self.SAMPLE.match(line):
                samples.append((float(match[1]), float(match[2])))
            elif match := self.HEADER.match(line):
                header[match[1]] = self._value(match[2])
        if not samples:
            raise ReaderError(f"{sniff.path.name}: no amplitude and phase samples")
        amplitude, phase = np.array(samples).T
        names = [v for k, v in header.items() if k.endswith("PULSENAME")]  # also "ePULSENAME"
        name = str(names[0]) if names else sniff.path.stem
        notes = ("duration not stored (set by the sequence)",)
        return [RFPulse(amplitude, np.degrees(phase), "relative", None, Meta({"header": header}),
                        self.provenance(sniff, notes), name)]

    @staticmethod
    def _value(text: str) -> float | str:
        try:
            return float(text)
        except ValueError:
            return text


#**************************************************************************************************#
#                                        FslMrsSimReader                                           #
#**************************************************************************************************#
#                                                                                                  #
# Read as the FSL-MRS simulator (fsl_denmatsim) reads it: each RF entry's "time" is the pulse      #
# duration, its samples (amp in RFUnits, times ampScale when given; phase in radians, plus         #
# phaseOffset) each last time / samples (checked: the pulses of the example STEAM and PRESS        #
# sequences come out at 90 and 180 degrees). One RFPulse per entry, its slice gradient (GradUnits  #
# per metre) and frequency offset in its vendor header; entries given as complex samples ("cmplx") #
# are kept there and noted. The rest is one Protocol: B0 as FieldStrength, Rx_SW and Rx_Points as  #
# SpectralWidth and NumberOfPoints, centralShift (the ppm at 0 Hz) as ReferencePPM.                #
#                                                                                                  #
#**************************************************************************************************#
class FslMrsSimReader(Reader):
    FORMATS = (Format.FSL_MRS_SIM,)
    PACKAGES = ("miox",)
    SOURCE = "FSL-MRS sequence description"
    KEYS = {"FieldStrength": "B0", "SpectralWidth": "Rx_SW", "NumberOfPoints": "Rx_Points",
            "ReferencePPM": "centralShift"}

    def read(self, sniff: Sniff) -> list[RFPulse | Protocol]:
        try:
            description = json.loads(sniff.path.read_text())
        except ValueError as e:
            raise ReaderError(f"{sniff.path.name}: {e}") from e
        name = str(description.get("sequenceName") or sniff.path.stem)
        unit = str(description.get("RFUnits", "Hz"))
        pulses, notes = [], []
        for index, entry in enumerate(description.get("RF", []), start=1):
            vendor = {k: v for k, v in entry.items() if k not in ("amp", "phase")}
            vendor["GradUnits"] = description.get("GradUnits", "T")
            if "amp" not in entry:
                notes.append(f"RF {index} given as complex samples: kept in the vendor header")
                continue
            amplitude = np.atleast_1d(np.asarray(entry["amp"], dtype=float))
            amplitude = amplitude * float(entry.get("ampScale", 1.0))
            phase = np.atleast_1d(np.asarray(entry.get("phase", 0.0), dtype=float))
            phase = np.broadcast_to(phase, amplitude.shape) + float(entry.get("phaseOffset", 0.0))
            pulse_notes = ("ampScale applied",) if "ampScale" in entry else ()
            if float(entry.get("phaseOffset", 0.0)):
                pulse_notes += ("phaseOffset added to the phase",)
            pulses.append(RFPulse(amplitude, np.degrees(phase), unit, float(entry["time"]),
                                  Meta(vendor), self.provenance(sniff, pulse_notes),
                                  f"{name} RF {index}"))
        if not pulses:
            raise ReaderError(f"{sniff.path.name}: no RF pulse with amplitude samples")
        meta = Meta({k: v for k, v in description.items() if k != "RF"}
                    | {"RF": [{k: v for k, v in e.items() if k not in ("amp", "phase")}
                              for e in description.get("RF", [])]})
        for key, field in self.KEYS.items():
            if isinstance(description.get(field), (int, float)):
                value = int(description[field]) if key == "NumberOfPoints" else description[field]
                meta.set(key, value, f"{self.SOURCE}: {field}", field)
        protocol = Protocol(name, (), meta, self.provenance(sniff, tuple(notes)))
        return [*pulses, protocol]


#**************************************************************************************************#
#                                        BrukerShapeReader                                         #
#**************************************************************************************************#
#                                                                                                  #
# JCAMP-DX "##KEY= value" lines (TITLE, $SHAPE_EXMODE, $SHAPE_TOTROT, $SHAPE_BWFAC,                #
# $SHAPE_INTEGFAC, $SHAPE_PARAMETERS, ...), then ##XYPOINTS= (XY..XY): one "amplitude, phase"      #
# pair per sample, the amplitude in percent of the maximum, the phase in degrees (checked:         #
# SHAPE_INTEGFAC = |sum of amplitude x exp(i phase)| / (100 x samples) for excitation, inversion   #
# and universal shapes; adiabatic shapes state a factor of their own). The duration is set by the  #
# experiment, not the file.                                                                        #
#                                                                                                  #
#**************************************************************************************************#
class BrukerShapeReader(Reader):
    FORMATS = (Format.BRUKER_SHAPE,)
    PACKAGES = ("miox",)
    HEADER = re.compile(r"^##\$?([A-Za-z][\w ]*?)=\s*(.*?)\s*$")
    SAMPLE = re.compile(r"^\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*$")

    def read(self, sniff: Sniff) -> list[RFPulse]:
        header, samples, points = {}, [], False
        for line in sniff.path.read_text(errors="replace").splitlines():
            if line.startswith("##"):
                points = line.startswith("##XYPOINTS")
                if match := self.HEADER.match(line):
                    header[match[1]] = SiemensPtaReader._value(match[2])
            elif points and (match := self.SAMPLE.match(line)):
                samples.append((float(match[1]), float(match[2])))
        if not samples:
            raise ReaderError(f"{sniff.path.name}: no XYPOINTS samples")
        amplitude, phase = np.array(samples).T
        notes = ("duration not stored (set by the experiment)",)
        if isinstance(header.get("NPOINTS"), float) and int(header["NPOINTS"]) != len(samples):
            notes += (f"NPOINTS {int(header['NPOINTS'])}, {len(samples)} samples read",)
        name = str(header.get("TITLE") or sniff.path.name)
        return [RFPulse(amplitude, phase, "%", None, Meta({"header": header}),
                        self.provenance(sniff, notes), name)]
