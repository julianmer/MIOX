####################################################################################################
#                                         ge_pfile_raw.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 06/10/26                                                                                #
#                                                                                                  #
# Purpose: Raw k-space of a GE P-file: every acquired frame with its slice, echo and view, the     #
#          frame geometry of the data acquisition table, and the baseline frames as their own      #
#          stream.                                                                                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

# own
from miox.data import Provenance, RawData
from miox.readers.base import ReaderError
from miox.readers.ge_rdb import GeRdbHeader
from miox.readers.ismrmrd import IsmrmrdReader


#**************************************************************************************************#
#                                            GePfileRaw                                            #
#**************************************************************************************************#
#                                                                                                  #
# The frames of a P-file follow the header as, for every pass, receive coil, slice and echo, one   #
# baseline frame (view 0) and then the acquired views, each frame ``frame_size`` complex points of #
# 16- or 32-bit integers in the header's byte order; a pass holds ``raw_pass_size`` bytes. The     #
# views become one RawData of readouts ordered by pass, slice slot, echo and view with the coils   #
# as channels; the baseline frames a second one. The readout headers follow the ISMRMRD layout     #
# with ge_to_ismrmrd's choices: the view index as the first encoding step, the echo as the         #
# contrast, the centre sample at half the frame size, the first and last view of a slice flagged,  #
# the slice counter the spatial index of the slice in the data acquisition table (the slot in the  #
# file is the acquisition order and kept in the second user counter) and the position and          #
# directions of the slice from that table; the pass number is kept in the first user counter, the  #
# run number as the measurement id.                                                                #
#                                                                                                  #
#**************************************************************************************************#
class GePfileRaw:
    FLAGS = IsmrmrdReader.ACQUISITION_FLAGS
    FIRST_IN_SLICE = 7  # flag bits as in ismrmrd.h
    LAST_IN_SLICE = 8
    GEOMETRY = ("position", "read_dir", "phase_dir", "slice_dir")

    def __init__(self, path: Path, header: GeRdbHeader, provenance: Provenance):
        self.path = path
        self.header = header
        self.provenance = provenance

    def datasets(self) -> list[RawData]:
        v = self.header.values
        passes, slices = v.get("rhr_rh_npasses") or 1, v.get("rhr_rh_nslices") or 1
        echoes, views = v.get("rhr_rh_nechoes") or 1, v.get("rhr_rh_nframes") or 0
        frames = self._frames(passes, slices, echoes, views)
        notes = list(self.provenance.notes)
        geometry = self.header.geometry()
        if geometry is None:
            notes.append("no slice geometry: the header does not locate the data acquisition table")
        provenance = dataclasses.replace(self.provenance, notes=tuple(notes))
        image = self._stream(frames[..., 1:, :], 1, provenance, geometry, self.path.stem)
        name = self.path.stem + "_baseline"
        baseline = self._stream(frames[..., :1, :], 0, provenance, geometry, name)
        return [image, baseline]

    def _frames(self, passes: int, slices: int, echoes: int, views: int) -> np.ndarray:
        """All frames as (pass, coil, slice, echo, view, sample) complex values, view 0 the
        baseline."""
        v, coils = self.header.values, self.header.coils
        samples, point_size = v.get("rhr_rh_frame_size") or 0, v.get("rhr_rh_point_size") or 2
        offset = self.header.data_offset
        size = self.path.stat().st_size
        per_pass = coils * slices * echoes * (views + 1) * samples * 2
        needed = per_pass * point_size
        if offset is None:
            offset = size - needed * passes  # the frames fill the end of the file
        available = size - offset
        if needed * passes > available or needed == 0:
            raise ReaderError(
                f"{self.path.name}: {needed * passes} bytes of frames expected after the header, "
                f"{available} present"
            )
        dtype = self.header.prefix + ("i2" if point_size == 2 else "i4")
        pass_size = v.get("rhr_rh_raw_pass_size") or needed
        stride = pass_size if passes > 1 and offset + pass_size * passes <= size else needed
        raw = np.stack(
            [
                np.fromfile(self.path, dtype, per_pass, offset=offset + p * stride)
                for p in range(passes)
            ]
        )
        raw = raw.reshape(passes, coils, slices, echoes, views + 1, samples, 2)
        raw = raw.astype(np.float32 if point_size == 2 else np.float64)  # 32-bit integers exactly
        return raw[..., 0] + 1j * raw[..., 1]

    def _stream(
        self, frames: np.ndarray, first_view: int, provenance: Provenance,
        geometry: list[dict] | None, name: str,
    ) -> RawData:
        passes, coils, slices, echoes, views, samples = frames.shape
        data = np.ascontiguousarray(
            np.transpose(frames, (0, 2, 3, 4, 1, 5)).reshape(-1, coils, samples)
        )
        index = np.indices((passes, slices, echoes, views)).reshape(4, -1)
        pass_number, slice_number, echo, view = index
        n = data.shape[0]
        headers = np.zeros(n, dtype=RawData.HEADER)
        headers["measurement_uid"] = self.header.values.get("rhi_rawrunnum") or 0
        headers["scan_counter"] = np.arange(n)
        headers["number_of_samples"] = samples
        headers["available_channels"] = coils
        headers["active_channels"] = coils
        headers["center_sample"] = samples // 2
        geometric = self.header.geometric_slices() if geometry else {}
        slots = zip(pass_number.tolist(), slice_number.tolist(), strict=True)
        table_index = np.array([geometric.get(slot, -1) for slot in slots])
        headers["idx"]["slice"] = np.where(table_index >= 0, table_index, slice_number)
        headers["idx"]["contrast"] = echo
        headers["idx"]["user"][:, 0] = pass_number
        headers["idx"]["user"][:, 1] = slice_number
        if first_view:
            total = self.header.values.get("rhr_rh_nframes") or views
            headers["idx"]["kspace_encode_step_1"] = view
            first = np.uint64(1 << (self.FIRST_IN_SLICE - 1))
            last = np.uint64(1 << (self.LAST_IN_SLICE - 1))
            headers["flags"] = np.where(view == 0, first, np.uint64(0))
            headers["flags"] |= np.where(view + 1 == total, last, np.uint64(0))
        if geometry:
            for i in np.nonzero(table_index >= 0)[0]:
                for field in self.GEOMETRY:
                    headers[field][i] = geometry[table_index[i]][field]
        meta = self.header.meta()
        meta.vendor["slice_table"] = self.header.slice_table()
        encoding = {
            "readout_order": "pass, slice, echo, view",
            "passes": passes, "slices_per_pass": slices, "echoes": echoes,
            "views": views if first_view else 0, "baseline_frames": 0 if first_view else views,
            "frame_size": samples, "coils": coils,
            "point_size": self.header.values.get("rhr_rh_point_size"),
        }
        return RawData(data, headers, meta, provenance, name, None, None, encoding, [], self.FLAGS)
