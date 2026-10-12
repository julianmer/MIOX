####################################################################################################
#                                          nifti_zarr.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 13/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for NIfTI-Zarr stores (.nii.zarr): an OME-NGFF multiscale zarr group that keeps  #
#          the NIfTI header, as written, in its "nifti" array.                                     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import io
import warnings

import nibabel as nib
import numpy as np

# own
from miox.data import MRImage
from miox.meta.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                         NiftiZarrReader                                          #
#**************************************************************************************************#
#                                                                                                  #
# The full-resolution level (the first dataset of the first multiscale) holds the voxels with the  #
# axes reversed (t, z, y, x); the "nifti" array the NIfTI-1 or NIfTI-2 header bytes, whose best    #
# affine (sform, else qform) and scaling apply as in the NIfTI file. The coarser levels are noted, #
# not read; the header as JSON (the array's attributes) and the multiscales are kept.              #
#                                                                                                  #
#**************************************************************************************************#
class NiftiZarrReader(Reader):
    FORMATS = (Format.NIFTI_ZARR,)
    HEADERS = {348: nib.Nifti1Header, 540: nib.Nifti2Header}  # by sizeof_hdr

    def read(self, sniff: Sniff) -> list[MRImage]:
        import zarr  # only for zarr stores

        try:
            with warnings.catch_warnings():  # files beside the arrays (e.g. .npy) are skipped
                warnings.simplefilter("ignore")
                group = zarr.open_group(str(sniff.path), mode="r")
                attrs = dict(group.attrs)
                levels = attrs.get("multiscales", [{}])[0].get("datasets", [{"path": "0"}])
                raw = bytes(np.asarray(group["nifti"][:], dtype=np.uint8))
                header = self.header(raw)
                data = np.asarray(group[levels[0]["path"]][:])
                nifti_json = dict(group["nifti"].attrs)
        except (KeyError, ValueError, OSError) as e:
            raise ReaderError(f"{sniff.path.name}: {type(e).__name__}: {e}") from e
        data = np.transpose(data, tuple(range(data.ndim))[::-1])  # (x, y, z, t, ...)
        slope, inter = header.get_slope_inter()
        if slope not in (None, 1.0) or inter not in (None, 0.0):
            data = data * (1.0 if slope is None else slope) + (inter or 0.0)
        notes = [f"{len(levels)} resolution levels; the full resolution read"] if len(
            levels) > 1 else []
        meta = Meta({"nifti": nifti_json, "multiscales": attrs.get("multiscales", [])})
        dims = ("time",) * (data.ndim > 3) + ("volume",) * max(data.ndim - 4, 0)
        name = sniff.path.name.removesuffix(".zarr").removesuffix(".nii")
        return [MRImage(data, header.get_best_affine(), meta,
                        self.provenance(sniff, tuple(notes)), name, dims)]

    @classmethod
    def header(cls, raw: bytes):
        for order in ("<", ">"):
            size = int(np.frombuffer(raw[:4], f"{order}i4")[0])
            if size in cls.HEADERS:
                return cls.HEADERS[size].from_fileobj(io.BytesIO(raw))
        raise ValueError("the nifti array holds no NIfTI header")
