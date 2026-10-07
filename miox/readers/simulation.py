####################################################################################################
#                                          simulation.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/10/26                                                                                #
#                                                                                                  #
# Purpose: Readers for MR simulation files: KomaMRI phantoms (.phantom, and spin lists in HDF5),   #
#          JEMRIS phantoms (a voxel grid in HDF5) and JEMRIS simulated signals.                    #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import h5py
import numpy as np

# own
from miox.data import Phantom, RawData
from miox.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                        KomaPhantomReader                                         #
#**************************************************************************************************#
#                                                                                                  #
# A KomaMRI .phantom file (HDF5) holds the attributes Name, Ns, Dims and Version, the spin         #
# positions x, y, z (m) in the group "position", one array per contrast property in "contrast"     #
# (ρ, T1, T2, T2s in s, Δw in rad/s, diffusion Dλ1, Dλ2, Dθ) and, optionally, "motion" with one    #
# group per motion of three parts (action, time, spins), each a type attribute and its parameters  #
# as attributes or datasets. A spin list may also hold x, y, z and the properties at the root.     #
# Property names are spelled in ASCII (rho, dw, Dlambda1, Dlambda2, Dtheta).                       #
#                                                                                                  #
#**************************************************************************************************#
class KomaPhantomReader(Reader):
    FORMATS = (Format.KOMA_PHANTOM,)
    PACKAGES = ("miox", "h5py")
    NAMES = {"ρ": "rho", "Δw": "dw", "Dλ1": "Dlambda1", "Dλ2": "Dlambda2", "Dθ": "Dtheta"}
    UNITS = {"rho": "", "T1": "s", "T2": "s", "T2s": "s", "dw": "rad/s"}

    def read(self, sniff: Sniff) -> list[Phantom]:
        with h5py.File(sniff.path, "r") as fh:
            attributes = {k: self._plain(v) for k, v in fh.attrs.items()}
            nested = "position" in fh and "contrast" in fh
            position = fh["position"] if nested else fh
            contrast = fh["contrast"] if nested else fh
            positions = np.stack([np.asarray(position[c][()], dtype=np.float64).ravel()
                                  for c in "xyz"], axis=1)
            properties = {
                self.NAMES.get(k, k): np.asarray(contrast[k][()]).ravel()
                for k in contrast if k not in "xyz" and isinstance(contrast[k], h5py.Dataset)
            }
            motion = [self._motion(fh["motion"][k]) for k in fh["motion"]] if "motion" in fh else []
        if any(len(v) != len(positions) for v in properties.values()):
            raise ReaderError(f"{sniff.path.name}: properties of unequal length")
        units = {k: self.UNITS[k] for k in properties if k in self.UNITS}
        name = str(attributes.get("Name") or sniff.path.stem)
        meta = Meta({"attributes": attributes, "motion": motion})
        return [Phantom(positions, properties, units, motion, meta, self.provenance(sniff), name)]

    def _motion(self, group: h5py.Group) -> dict:
        motion = {}
        for part in group:  # action, time, spins
            item = {k: self._plain(v) for k, v in group[part].attrs.items()}
            item.update({k: group[part][k][()] for k in group[part]})
            motion[part] = item
        return motion

    @staticmethod
    def _plain(value):
        if isinstance(value, bytes):
            return value.decode(errors="replace")
        if isinstance(value, np.generic):
            return value.item()
        return value


#**************************************************************************************************#
#                                       JemrisPhantomReader                                        #
#**************************************************************************************************#
#                                                                                                  #
# A JEMRIS phantom (sample) file holds sample/data, five values per voxel (M0, 1/T1, 1/T2, 1/T2*   #
# in 1/ms and DB), sample/resolution and sample/offset in mm. Every voxel with M0 not zero is a    #
# spin; T1, T2 and T2s are the inverse rates in s (a rate of 0, no relaxation, gives infinity).    #
# DB is kept as stored (JEMRIS's off-resonance; its unit is not stated in the file). The spins lie #
# on the grid centred on the offset, the first index along x, as KomaMRI places them.              #
#                                                                                                  #
#**************************************************************************************************#
class JemrisPhantomReader(Reader):
    FORMATS = (Format.JEMRIS_PHANTOM,)
    PACKAGES = ("miox", "h5py")
    CHANNELS = ("rho", "R1", "R2", "R2s", "DB")

    def read(self, sniff: Sniff) -> list[Phantom]:
        with h5py.File(sniff.path, "r") as fh:
            data = np.transpose(fh["sample/data"][()])  # (values, x, y[, z]) as JEMRIS writes it
            resolution = np.asarray(fh["sample/resolution"][()], dtype=np.float64).ravel() * 1e-3
            offset = np.asarray(fh["sample/offset"][()], dtype=np.float64).ravel() * 1e-3
        if data.shape[0] != len(self.CHANNELS):
            raise ReaderError(f"{sniff.path.name}: {data.shape[0]} values per voxel, not 5")
        grid = data.reshape(data.shape + (1,) * (4 - data.ndim))  # (5, x, y, z)
        mask = grid[0] != 0
        indices = np.argwhere(mask)
        shape = grid.shape[1:]
        centres = [(np.arange(n) - (n - 1) / 2) * resolution[i] + offset[i]
                   for i, n in enumerate(shape)]
        positions = np.stack([centres[i][indices[:, i]] for i in range(3)], axis=1)
        properties = {"rho": grid[0][mask]}
        with np.errstate(divide="ignore"):
            for name, rate in zip(("T1", "T2", "T2s"), grid[1:4], strict=True):
                properties[name] = 1e-3 / rate[mask]
        properties["DB"] = grid[4][mask]
        units = {"rho": "", "T1": "s", "T2": "s", "T2s": "s", "DB": "as stored"}
        notes = ("DB as stored: the file does not state its unit",
                 "spin positions on the grid centred on the offset, as KomaMRI places them")
        grid_info = {"shape": list(shape), "resolution": resolution.tolist(),
                     "offset": offset.tolist(), "indices": indices}
        meta = Meta({"resolution_mm": (resolution * 1e3).tolist(),
                     "offset_mm": (offset * 1e3).tolist(), "shape": list(shape)})
        return [Phantom(positions, properties, units, [], meta, self.provenance(sniff, notes),
                        sniff.path.stem, grid_info)]


#**************************************************************************************************#
#                                        JemrisSignalReader                                        #
#**************************************************************************************************#
#                                                                                                  #
# A JEMRIS signal file holds signal/times (ms) and one dataset per receive channel in              #
# signal/channels (00, 01, ...) with Mx, My and Mz of every ADC sample. The file stores no         #
# readouts: a readout is a run of equally spaced times, a new one starts where the step to the     #
# next sample exceeds 1.5 times the smallest step. Mx + iMy are the samples; Mz and the sample     #
# times (s) are kept as waveforms of the same layout.                                              #
#                                                                                                  #
#**************************************************************************************************#
class JemrisSignalReader(Reader):
    FORMATS = (Format.JEMRIS_SIGNAL,)
    PACKAGES = ("miox", "h5py")
    GAP = 1.5  # times the smallest step

    def read(self, sniff: Sniff) -> list[RawData]:
        with h5py.File(sniff.path, "r") as fh:
            times = np.asarray(fh["signal/times"][()], dtype=np.float64).ravel() * 1e-3
            names = sorted(fh["signal/channels"])
            channels = np.stack([np.asarray(fh["signal/channels"][n][()], dtype=np.float64)
                                 for n in names])  # (coils, samples, 3)
        if channels.shape[1] != len(times) or channels.shape[2] != 3:
            raise ReaderError(f"{sniff.path.name}: channels {channels.shape} for {len(times)} "
                              f"times")
        steps = np.diff(times)
        smallest = steps[steps > 0].min() if (steps > 0).any() else 0.0
        starts = np.concatenate([[0], np.flatnonzero(steps > self.GAP * smallest) + 1])
        ends = np.concatenate([starts[1:], [len(times)]])
        lengths = ends - starts
        n, coils, longest = len(starts), channels.shape[0], int(lengths.max())
        data = np.zeros((n, coils, longest), dtype=np.complex128)
        mz = np.zeros((n, coils, longest))
        sample_times = np.zeros((n, longest))
        headers = np.zeros(n, dtype=RawData.HEADER)
        for i, (a, b) in enumerate(zip(starts, ends, strict=True)):
            data[i, :, : b - a] = channels[:, a:b, 0] + 1j * channels[:, a:b, 1]
            mz[i, :, : b - a] = channels[:, a:b, 2]
            sample_times[i, : b - a] = times[a:b]
            headers[i]["sample_time_us"] = (times[a + 1] - times[a]) * 1e6 if b - a > 1 else 0
        headers["scan_counter"] = np.arange(n)
        headers["number_of_samples"] = lengths
        headers["available_channels"] = coils
        headers["active_channels"] = coils
        notes = [f"{n} readouts: runs of equally spaced sample times"]
        if len(set(lengths.tolist())) > 1:
            notes.append("readouts of unequal length, zero-padded")
        waveforms = [{"name": "Mz", "data": mz}, {"name": "time", "unit": "s",
                                                 "data": sample_times}]
        meta = Meta({"channels": names})
        dwell = round(float(steps[0]), 12) if len(steps) and lengths[0] > 1 else None
        if dwell is not None and np.allclose(headers["sample_time_us"][lengths > 1], dwell * 1e6):
            meta.set("DwellTime", dwell, "signal/times: step within a readout (to 1 ps)",
                     "signal/times")
        provenance = self.provenance(sniff, tuple(notes))
        return [RawData(data, headers, meta, provenance, sniff.path.stem, None, lengths, {},
                        waveforms, {})]
