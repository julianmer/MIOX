####################################################################################################
#                                         nibabel_image.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for the image formats nibabel reads: NIfTI-1/2 (with a BIDS .json sidecar),      #
#          Analyze, MGH/MGZ, MINC 1/2, Philips PAR/REC and AFNI. The vendor header keeps the       #
#          format's whole header (and the sidecar).                                                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import nibabel as nib
import numpy as np
from nibabel.externals.netcdf import netcdf_file

# own
from miox.data import MRImage
from miox.meta.keys import KeyTable
from miox.meta.meta import Meta
from miox.readers.base import Reader, ReaderError
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                        NibabelImageReader                                        #
#**************************************************************************************************#
#                                                                                                  #
# One MRImage per file: nibabel's data (scaling applied; PAR/REC as display values) and affine.    #
# Standard keys: a BIDS sidecar's fields (renamed where Miox has the same key: FlipAngle,          #
# MagneticFieldStrength, ImagingFrequency, ...), MGH times (ms) and flip angle (rad), PAR/REC      #
# general and per-image values (ms). MINC and AFNI attributes stay in the vendor header; MINC data #
# get their spatial axes first, as nibabel builds the affine from those alone.                     #
#                                                                                                  #
#**************************************************************************************************#
class NibabelImageReader(Reader):
    FORMATS = (
        Format.NIFTI,
        Format.ANALYZE,
        Format.MGH,
        Format.MINC1,
        Format.MINC2,
        Format.PARREC,
        Format.AFNI,
    )
    PACKAGES = Reader.PACKAGES + ("nibabel",)
    HEADER_FILES = (".hdr", ".par", ".head")  # of formats split in header and data files
    BIDS_RENAMED = {  # BIDS sidecar field -> Miox key, (scale)
        "FlipAngle": ("ExcitationFlipAngle", None),
        "MagneticFieldStrength": ("FieldStrength", None),
        "ImagingFrequency": ("SpectrometerFrequency", "list"),
        "ReceiveCoilName": ("RxCoil", None),
        "TransmitCoilName": ("TxCoil", None),
        "PatientSex": ("PatientSex", None),
    }
    MGH = {"tr": "RepetitionTime", "te": "EchoTime", "ti": "InversionTime"}  # ms
    POSITIONS = {"head": "H", "feet": "F", "first": "F", "supine": "S", "prone": "P",
                 "decubitus": "D", "right": "R", "left": "L"}

    def read(self, sniff: Sniff) -> list[MRImage]:
        files = (sniff.path,) + sniff.companions
        path = next((f for f in files if f.suffix.lower() in self.HEADER_FILES), sniff.path)
        try:
            img = nib.load(path)
            data = np.asanyarray(img.dataobj)
        except FileNotFoundError as e:
            raise ReaderError(f"{path.name}: data file missing ({Path(e.filename).name})") from e
        except Exception as e:
            if not type(e).__module__.startswith("nibabel"):
                raise
            raise ReaderError(f"{path.name}: {e}") from e
        notes = []
        affine = getattr(img, "affine", None)
        if affine is None:
            affine = np.eye(4)
            notes.append("no spatial affine (e.g. CIFTI); identity used")
        color = data.dtype.names is not None  # RGB(A) samples stored as a structured type
        if color:
            data = np.stack([data[name] for name in data.dtype.names], axis=-1)
        dims = ("time",) if self._timed(img) else ("volume",) * max(data.ndim - 3 - color, 0)
        dims += ("rgb",) if color else ()
        if np.iscomplexobj(data) and data.ndim == 4 and data.shape[:3] == (1, 1, 1):
            notes.append("complex 1x1x1xN data look like spectroscopy, but the file has no "
                         "NIfTI-MRS extension or MRS sidecar")
        if sniff.format in (Format.MINC1, Format.MINC2):
            data, dims = self._minc_axes(img, data)
        values, vendor = self._header(sniff.format, img, files, notes)
        meta = Meta(vendor)
        for key, (value, field, source) in values.items():
            meta.set(key, value, source, field)
        name = path.name.removesuffix(".gz")
        name = name.rsplit(".", 1)[0] if "." in name else name
        return [MRImage(data, affine, meta, self.provenance(sniff, tuple(notes)), name, dims)]

    def _header(self, fmt: Format, img, files: tuple[Path, ...], notes: list[str]) -> tuple:
        """Standard values as {key: (value, vendor field, source)} and the vendor header."""
        if fmt in (Format.NIFTI, Format.ANALYZE):
            header = getattr(img, "nifti_header", img.header)  # CIFTI keeps it apart
            vendor = {"header": {k: self.plain(header[k]) for k in header.keys()}}
            extensions = getattr(header, "extensions", [])
            if extensions:
                vendor["extensions"] = [
                    {"code": int(e.get_code()), "size": int(e.get_sizeondisk())} for e in extensions
                ]
            sidecar = next((f for f in files if f.suffix.lower() == ".json"), None)
            values = {}
            if sidecar:
                vendor["sidecar"] = json.loads(sidecar.read_text())
                values = self.bids_values(vendor["sidecar"])
            values.update(self.diffusion_values(img, files, vendor))
            return values, vendor
        if fmt == Format.MGH:
            vendor = {"header": {k: self.plain(img.header[k]) for k in img.header.keys()}}
            return self._mgh(img.header), vendor
        if fmt == Format.PARREC:
            notes.append("PAR/REC values scaled as display values (nibabel's default)")
            return self._parrec(img.header)
        if fmt == Format.AFNI:
            return {}, {"header": self.plain(img.header.info)}
        return {}, {"header": self._minc(files[0], fmt)}

    def diffusion_values(self, img, files: tuple[Path, ...], vendor: dict) -> dict:
        """B-values and gradient directions from .bval/.bvec files next to the image; the
        directions, given along the voxel axes, are turned into RAS world unit vectors."""
        bval = next((f for f in files if f.suffix.lower() == ".bval"), None)
        bvec = next((f for f in files if f.suffix.lower() == ".bvec"), None)
        if bval is None or bvec is None:
            return {}
        values = np.atleast_1d(np.loadtxt(bval, dtype=float))
        vectors = np.atleast_2d(np.loadtxt(bvec, dtype=float))
        if vectors.shape[0] != 3 or vectors.shape[1] != values.size:
            return {}
        vendor["bval"], vendor["bvec"] = values.tolist(), vectors.tolist()
        axes = img.affine[:3, :3] / np.linalg.norm(img.affine[:3, :3], axis=0)
        world = axes @ vectors
        norms = np.linalg.norm(world, axis=0)
        world = np.where(norms > 0, world / np.where(norms > 0, norms, 1), 0.0)
        return {
            "DiffusionBValue": (values.tolist(), "bval", f"{bval.name}"),
            "DiffusionGradientOrientation": (
                world.T.round(6).tolist(),
                "bvec",
                f"{bvec.name}: voxel-axis vectors turned into RAS world vectors",
            ),
        }

    def bids_values(self, sidecar: dict) -> dict:
        """Sidecar fields that are standard keys, as {key: (value, vendor field, source)}."""
        keys = KeyTable()
        values = {}
        for field, value in sidecar.items():
            key, scale = self.BIDS_RENAMED.get(field, (field, None))
            if key not in keys:
                continue
            value = [value] if scale == "list" else value
            values[key] = (value, f"sidecar/{field}", f"BIDS sidecar: {field}")
        return values

    def _mgh(self, header) -> dict:
        values = {}
        for field, key in self.MGH.items():
            if float(header[field]) > 0:
                values[key] = (float(header[field]) * 1e-3, f"header/{field}", f"MGH {field} (ms)")
        if float(header["flip_angle"]) > 0:
            angle = float(np.degrees(header["flip_angle"]))
            values["ExcitationFlipAngle"] = (angle, "header/flip_angle", "MGH flip_angle (rad)")
        return values

    def _parrec(self, header) -> tuple[dict, dict]:
        info, defs = header.general_info, header.image_defs
        columns = {}
        for name in defs.dtype.names:
            unique = np.unique(defs[name], axis=0)
            columns[name] = self.plain(unique) if len(unique) <= 16 else f"<{len(unique)} values>"
        vendor = {"general_info": self.plain(info), "image_defs": columns}
        source = "PAR"
        values = {"Manufacturer": ("Philips", None, "PAR/REC is a Philips format")}

        def single(column: str, scale: float = 1.0) -> Any:
            unique = np.unique(defs[column]) * scale
            return float(unique[0]) if len(unique) == 1 else [float(v) for v in unique]

        tr = np.unique(info["repetition_time"]) * 1e-3
        values["RepetitionTime"] = (
            float(tr[0]) if len(tr) == 1 else [float(v) for v in tr],
            "general_info/repetition_time",
            f"{source}: repetition time (ms)",
        )
        for key, column, scale in (
            ("EchoTime", "echo_time", 1e-3),
            ("ExcitationFlipAngle", "image_flip_angle", 1.0),
            ("SliceThickness", "slice thickness", 1.0),
        ):
            values[key] = (single(column, scale), f"image_defs/{column}", f"{source}: {column}")
        if np.any(defs["Inversion delay"] > 0):
            values["InversionTime"] = (
                single("Inversion delay", 1e-3),
                "image_defs/Inversion delay",
                f"{source}: Inversion delay (ms)",
            )
        spacing = np.unique(defs["slice thickness"] + defs["slice gap"])
        if len(spacing) == 1:
            values["SpacingBetweenSlices"] = (
                float(spacing[0]),
                "image_defs/slice gap",
                f"{source}: slice thickness + slice gap",
            )
        for key, field in (("ProtocolName", "protocol_name"), ("SequenceName", "tech")):
            if info.get(field):
                values[key] = (str(info[field]), f"general_info/{field}", f"{source}: {field}")
        mode = {"MS": "2D", "3D": "3D"}.get(str(info.get("scan_mode")))
        if mode:
            values["MRAcquisitionType"] = (mode, "general_info/scan_mode", f"{source}: scan_mode")
        position = self._position(str(info.get("patient_position", "")))
        if position:
            values["PatientPosition"] = (
                position,
                "general_info/patient_position",
                f"{source}: patient_position, written out (e.g. Head First Supine = HFS)",
            )
        return values, vendor

    def _position(self, text: str) -> str:
        """DICOM patient position code from the written form, e.g. Head First Supine -> HFS."""
        letters = [self.POSITIONS.get(word.lower()) for word in text.split()]
        return "" if not letters or None in letters else "".join(letters)

    def _minc(self, path: Path, fmt: Format) -> dict:
        """All MINC variable attributes, by variable."""
        if fmt == Format.MINC1:
            with netcdf_file(path, "r", mmap=False) as nc:
                return {
                    name: {k: self.plain(v) for k, v in var._attributes.items()}
                    for name, var in nc.variables.items()
                }
        header = {}
        with h5py.File(path, "r") as h:
            h["minc-2.0"].visititems(
                lambda name, node: header.__setitem__(
                    name, {k: self.plain(v) for k, v in node.attrs.items()}
                )
            )
        return header

    @staticmethod
    def _minc_axes(img, data: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
        """Spatial axes first, in file order (as nibabel's affine), then the others."""
        names = list(img.dataobj.minc_file._dim_names)
        spatial = [i for i, n in enumerate(names) if n.endswith("space")]
        other = [i for i, n in enumerate(names) if not n.endswith("space")]
        return np.transpose(data, spatial + other), tuple(names[i] for i in other)

    @staticmethod
    def _timed(img) -> bool:
        """A 4D NIfTI whose fourth dimension has time units."""
        header = img.header
        if len(img.shape) != 4 or not hasattr(header, "get_xyzt_units"):
            return False
        return header.get_xyzt_units()[1] not in ("unknown", None)
