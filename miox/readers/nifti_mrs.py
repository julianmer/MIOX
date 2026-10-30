####################################################################################################
#                                           nifti_mrs.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for NIfTI-MRS files. The header extension is the metadata record; the vendor     #
#          header keeps it verbatim together with every field of the NIfTI header.                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
from nifti_mrs import validator
from nifti_mrs.nifti_mrs import NIFTI_MRS

# own
from miox.data import MRSData
from miox.readers.base import Reader
from miox.readers.nibabel_image import NibabelImageReader
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                          NiftiMrsReader                                          #
#**************************************************************************************************#
#                                                                                                  #
# Reads one NIfTI-MRS file into one MRSData. Files that break the standard are still read; why     #
# they are invalid goes into the provenance notes. A BIDS sidecar (.json of the same name) adds    #
# its standard keys where the header extension lacks them and is kept whole in the vendor header.  #
#                                                                                                  #
#**************************************************************************************************#
class NiftiMrsReader(Reader):
    FORMATS = (Format.NIFTI_MRS,)
    SOURCE = "NIfTI-MRS header extension"

    def read(self, sniff: Sniff) -> list[MRSData]:
        notes = []
        with self.captured(notes):
            image = NIFTI_MRS(sniff.path, validate_on_creation=False)
            try:
                validator.validate_nifti_mrs(image)
            except Exception as e:
                notes.append(f"not valid NIfTI-MRS: {e}")
        vendor = {
            "nifti": {key: self.plain(image.header[key]) for key in image.header.keys()},
            "header_extension": image.hdr_ext.to_dict(),
        }
        name = sniff.path.name.removesuffix(".gz").removesuffix(".nii")
        in_hz = self._frequency_in_mhz([image], notes)
        d = MRSData.from_nifti_mrs(
            image, self.provenance(sniff, tuple(notes)), self.SOURCE, vendor=vendor, name=name
        )
        for key in d.meta:
            if d.meta.source[key] == self.SOURCE:
                d.meta.fields[key] = f"header_extension/{key}"
        d.meta.set("DwellTime", d.dwelltime, "NIfTI header: pixdim[4]", "nifti/pixdim")
        if in_hz:
            frequency = d.meta["SpectrometerFrequency"]
            self.fix(d, "SpectrometerFrequency", frequency, "stored in Hz; divided by 1e6")
        if sniff.companions:
            self._sidecar(d, sniff.companions[0])
        self.correct(d)
        return [d]

    @staticmethod
    def _sidecar(d: MRSData, path: Path) -> None:
        """A BIDS sidecar's standard keys where the header extension lacks them; a value both
        state differently stays as the header extension has it, with a note."""
        try:
            sidecar = json.loads(path.read_text())
        except ValueError as e:
            notes = (f"{path.name} not read: {e}",)
            d.provenance = dataclasses.replace(d.provenance, notes=d.provenance.notes + notes)
            return
        d.meta.vendor["sidecar"] = sidecar
        notes = ()
        for key, (value, field, source) in NibabelImageReader().bids_values(sidecar).items():
            if key not in d.meta:
                d.meta.set(key, value, source, field)
                continue
            ours, theirs = np.ravel(d.meta[key]), np.ravel(value)
            same = ours.shape == theirs.shape and all(
                np.isclose(a, b) if isinstance(a, (int, float, np.number))
                and isinstance(b, (int, float, np.number)) else a == b
                for a, b in zip(ours.tolist(), theirs.tolist(), strict=True))
            if not same:
                notes += (f"{path.name}: {key} {value!r} differs from the header extension's "
                          f"{d.meta[key]!r}; the header's is kept",)
        d.provenance = dataclasses.replace(d.provenance, notes=d.provenance.notes + notes)
