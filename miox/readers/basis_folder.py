####################################################################################################
#                                         basis_folder.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for basis sets stored as a folder of single-spectrum files (LCModel .RAW, jMRUI  #
#          text, FSL-MRS JSON, NIfTI-MRS), one file per metabolite. The vendor header keeps the    #
#          vendor header of every file, by metabolite.                                             #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

# own
from miox.data import MRSData
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff

if TYPE_CHECKING:
    from miox.readers.loader import Loader


#**************************************************************************************************#
#                                        BasisFolderReader                                         #
#**************************************************************************************************#
#                                                                                                  #
# Each file is read by the reader of its own format, so the sign convention and units are the ones #
# checked for that format. The files must agree in frequency, nucleus, dwell time and length; each #
# is named after its file. Standard keys equal in every file are kept, with the first file's       #
# source.                                                                                          #
#                                                                                                  #
#**************************************************************************************************#
class BasisFolderReader(Reader):
    FORMATS = (Format.BASIS_FOLDER,)
    SOURCE = "basis-set member files"
    DERIVED = ("NumberOfPoints", "FieldStrength")  # recomputed from the stacked data

    def __init__(self, loader: Loader):
        self.loader = loader

    def read(self, sniff: Sniff) -> list[MRSData]:
        members = [self._member(path) for path in sniff.companions]
        names = [self.metabolite(path) for path in sniff.companions]
        first = members[0]
        self._check(sniff, members)
        fids = np.stack([m.data.reshape(-1) for m in members], axis=-1)
        vendor = {name: m.meta.vendor for name, m in zip(names, members, strict=True)}
        keys = {key: (first.meta[key], None) for key in MRSData.LIST_KEYS if key in first.meta}
        notes = [f"{len(members)} files read by {first.provenance.reader}"]
        notes.extend(dict.fromkeys(note for m in members for note in m.provenance.notes))
        d = self.basis_set(
            sniff, fids, names, first.dwelltime, keys, vendor, sniff.path.name, notes, self.SOURCE
        )
        for key, value in first.meta.items():
            if key in self.DERIVED:
                continue
            if any(self.plain(m.meta.get(key)) != self.plain(value) for m in members):
                continue
            field = first.meta.fields.get(key)
            source = f"{names[0]}: {first.meta.source[key]}"
            d.meta.set(key, value, source, f"{names[0]}/{field}" if field else None)
        d.provenance.versions.update(first.provenance.versions)
        return [d]

    @staticmethod
    def metabolite(path: Path) -> str:
        """The file name without its extension (and without .gz)."""
        return Path(path.name.removesuffix(".gz")).stem

    def _member(self, path: Path) -> MRSData:
        try:
            datasets = self.loader.load(path)
        except ReaderError as e:
            raise ReaderError(f"basis-set file {path.name}: {e}") from e
        if len(datasets) != 1 or datasets[0].data.size != datasets[0].n_points:
            raise ReaderError(f"basis-set file {path.name} holds more than one spectrum")
        return datasets[0]

    def _check(self, sniff: Sniff, members: list[MRSData]) -> None:
        """The files must describe the same acquisition grid."""
        first = members[0]
        for m in members[1:]:
            same = (
                self._close(m.spectrometer_frequency, first.spectrometer_frequency)
                and self._close(m.dwelltime, first.dwelltime)
                and m.nucleus == first.nucleus
                and m.n_points == first.n_points
            )
            if not same:
                raise ReaderError(
                    f"{sniff.path.name}: {m.name} differs from {first.name} in frequency, "
                    "nucleus, dwell time or length"
                )

    @staticmethod
    def _close(a: float | None, b: float | None) -> bool:
        """Equal within rounding; a value not stated equals only another not stated."""
        return a == b if a is None or b is None else bool(np.isclose(a, b, rtol=1e-6))
