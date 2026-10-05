####################################################################################################
#                                            fastmri.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 05/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for HDF5 files in the fastMRI layout (fastMRI, M4Raw): a gridded multi-coil      #
#          k-space per slice, the ISMRMRD header text and the root-sum-of-squares reconstruction.  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import xml.etree.ElementTree as ET

import h5py
import numpy as np

# own
from miox.data import MRImage, RawData
from miox.readers.base import Reader, ReaderError
from miox.readers.ismrmrd import IsmrmrdReader
from miox.readers.siemens_twix_raw import SiemensTwixRaw
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                          FastMriReader                                           #
#**************************************************************************************************#
#                                                                                                  #
# ``kspace`` is stored as (slices, coils, frequency encodes, phase encodes); every phase-encoding  #
# column of every slice becomes one readout of a RawData, with the slice and the line in its       #
# header counters. Lines the file holds as all zeros were not acquired; their count is noted.      #
# The ``ismrmrd_header`` text gives the standard keys as for ISMRMRD files; the file's root        #
# attributes join the vendor header. ``reconstruction_rss`` (and any other ``reconstruction_*``    #
# array) becomes an MRImage of (x, y, slice) without geometry, since the file states none.         #
#                                                                                                  #
#**************************************************************************************************#
class FastMriReader(Reader):
    FORMATS = (Format.FASTMRI_H5,)
    PACKAGES = ("miox", "h5py")

    def __init__(self):
        self.ismrmrd = IsmrmrdReader()

    def read(self, sniff: Sniff) -> list[RawData | MRImage]:
        notes: list[str] = []
        with h5py.File(sniff.path, "r") as fh:
            kspace = fh["kspace"][()]
            text = fh["ismrmrd_header"][()]
            text = text.decode() if isinstance(text, bytes) else str(text)
            attributes = {k: self.plain(v) for k, v in fh.attrs.items()}
            reconstructions = {
                name: fh[name][()] for name in fh if name.startswith("reconstruction")
            }
        if kspace.ndim != 4:
            raise ReaderError(f"{sniff.path.name}: kspace has {kspace.ndim} dimensions, not 4")
        header = self.ismrmrd._element(ET.fromstring(text))
        header["file_attributes"] = attributes
        datasets: list[RawData | MRImage] = [self._raw(sniff, kspace, header, notes)]
        for name, values in reconstructions.items():
            datasets.append(self._image(sniff, name, values, header))
        return datasets

    def _raw(self, sniff: Sniff, kspace: np.ndarray, header: dict, notes: list[str]) -> RawData:
        slices, coils, samples, lines = kspace.shape
        data = np.ascontiguousarray(
            np.transpose(kspace, (0, 3, 1, 2)).reshape(slices * lines, coils, samples)
        )
        headers = np.zeros(slices * lines, dtype=SiemensTwixRaw.HEADER)
        headers["idx"]["slice"] = np.repeat(np.arange(slices), lines)
        headers["idx"]["kspace_encode_step_1"] = np.tile(np.arange(lines), slices)
        headers["number_of_samples"] = samples
        headers["available_channels"] = headers["active_channels"] = coils
        headers["center_sample"] = samples // 2
        headers["scan_counter"] = np.arange(slices * lines)
        empty = int((np.abs(data).sum(axis=(1, 2)) == 0).sum())
        if empty:
            notes.append(f"{empty} of {slices * lines} k-space lines are all zero (not acquired)")
        meta = self.ismrmrd._meta(header, notes)
        meta.set("NumberOfPoints", int(samples), "kspace frequency-encoding extent")
        meta.set("NumberOfReceiveChannels", int(coils), "kspace coil extent")
        matrix = [int(samples), int(lines), int(slices)]
        meta.set("EncodedMatrixSize", matrix, "kspace extents: frequency, phase, slices")
        provenance = self.provenance(sniff, tuple(notes))
        notes.clear()
        return RawData(
            data, headers, meta, provenance, sniff.path.stem, None, None,
            self.ismrmrd._encoding(header), [], IsmrmrdReader.ACQUISITION_FLAGS,
        )

    def _image(self, sniff: Sniff, name: str, values: np.ndarray, header: dict) -> MRImage:
        data = np.transpose(values, tuple(range(values.ndim - 1, -1, -1)))  # (x, y, slice)
        meta = self.ismrmrd._meta(header, [])
        meta.set("ImageType", [name.removeprefix("reconstruction_").upper()], "dataset name")
        provenance = self.provenance(sniff, ("no geometry stated; the affine is a placeholder",))
        return MRImage(data, self.NO_GEOMETRY.copy(), meta, provenance, f"{sniff.path.stem}/{name}")
