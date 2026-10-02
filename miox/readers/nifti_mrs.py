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

from nifti_mrs import validator
from nifti_mrs.nifti_mrs import NIFTI_MRS

# own
from miox.data import MRSData
from miox.readers.base import Reader
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                          NiftiMrsReader                                          #
#**************************************************************************************************#
#                                                                                                  #
# Reads one NIfTI-MRS file into one MRSData. Files that break the standard are still read; why     #
# they are invalid goes into the provenance notes.                                                 #
#                                                                                                  #
#**************************************************************************************************#
class NiftiMrsReader(Reader):
    FORMATS = (Format.NIFTI_MRS,)

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
        return [
            MRSData.from_nifti_mrs(
                image,
                self.provenance(sniff, tuple(notes)),
                source="NIfTI-MRS header extension",
                vendor=vendor,
                name=name,
            )
        ]
