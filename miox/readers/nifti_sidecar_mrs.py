####################################################################################################
#                                       nifti_sidecar_mrs.py                                       #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for spectroscopy written as a plain complex NIfTI with its MRS values in a BIDS  #
#          .json sidecar (as dcm2niix writes it): SpectrometerFrequency, ResonantNucleus,          #
#          SpectralWidth, dim_5 to dim_7 and the BIDS fields. The vendor header keeps the NIfTI    #
#          header and the whole sidecar.                                                           #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import json

import nibabel as nib
import numpy as np

# own
from miox.data import MRSData
from miox.readers.base import Reader, ReaderError
from miox.readers.nibabel_image import NibabelImageReader
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                      NiftiSidecarMrsReader                                       #
#**************************************************************************************************#
#                                                                                                  #
# The data are stored as NIfTI-MRS stores them: checked against MIOX's reading of the DICOM        #
# dcm2niix converted (identical data and affine). The dwell time is 1 / SpectralWidth, else the    #
# sidecar's DwellTime, else pixdim[4].                                                             #
#                                                                                                  #
#**************************************************************************************************#
class NiftiSidecarMrsReader(Reader):
    FORMATS = (Format.NIFTI_SIDECAR_MRS,)
    PACKAGES = Reader.PACKAGES + ("nibabel",)
    SOURCE = "BIDS sidecar"
    DIMS = ("dim_5", "dim_6", "dim_7")

    def read(self, sniff: Sniff) -> list[MRSData]:
        img = nib.load(sniff.path)
        sidecar_path = next(c for c in sniff.companions if c.suffix == ".json")
        sidecar = json.loads(sidecar_path.read_text())
        data = np.asanyarray(img.dataobj)
        if data.ndim < 4:
            raise ReaderError(f"{sniff.path.name}: {data.ndim}-D data has no spectral dimension")
        values = {
            key: (value, field)
            for key, (value, field, _) in NibabelImageReader().bids_values(sidecar).items()
        }
        if "SpectralWidth" in sidecar:
            dwelltime, field = 1 / float(sidecar["SpectralWidth"]), "sidecar/SpectralWidth"
        elif "DwellTime" in sidecar:
            dwelltime, field = float(sidecar["DwellTime"]), "sidecar/DwellTime"
        else:
            dwelltime, field = float(img.header.get_zooms()[3]), "nifti/pixdim"
        tags = tuple(
            (sidecar[k], sidecar.get(f"{k}_info"), sidecar.get(f"{k}_header"))
            for k in self.DIMS[: data.ndim - 4]
            if k in sidecar
        )
        vendor = {
            "nifti": {k: self.plain(img.header[k]) for k in img.header.keys()},
            "sidecar": sidecar,
        }
        name = sniff.path.name.removesuffix(".gz").removesuffix(".nii")
        d = self.dataset(
            sniff, data, dwelltime, values, vendor, name, [], self.SOURCE,
            affine=img.affine, dim_tags=tags,
        )
        d.meta.set("DwellTime", dwelltime, f"{self.SOURCE}: {field}", field)
        return [d]
