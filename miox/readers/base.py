####################################################################################################
#                                             base.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Base classes of the readers (plain, and converting through spec2nii) and the error a    #
#          reader raises when a file of its format cannot be read.                                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import contextlib
import io
import re
import sys
import warnings
from collections.abc import Iterator
from importlib.metadata import version
from types import SimpleNamespace
from typing import Any

import numpy as np
from nifti_mrs.create_nmrs import gen_nifti_mrs_hdr_ext
from nifti_mrs.hdr_ext import Hdr_Ext
from nifti_mrs.nifti_mrs import NIFTI_MRS
from spec2nii.spec2nii import remove_zero_higher_dim_indices
from spec2nii.spec2nii import spec2nii as Spec2niiCommand

# own
from miox.data import MRSData, Provenance
from miox.meta.meta import Meta
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                           ReaderError                                            #
#**************************************************************************************************#
#                                                                                                  #
# A file of a supported format that cannot be read.                                                #
#                                                                                                  #
#**************************************************************************************************#
class ReaderError(Exception):
    pass


#**************************************************************************************************#
#                                              Reader                                              #
#**************************************************************************************************#
#                                                                                                  #
# Defines the reader structure. A reader handles the formats in ``FORMATS``, takes the sniff of a  #
# path and returns one MRSData per dataset in it (a file may hold several, e.g. a water            #
# reference next to the metabolite data).                                                          #
#                                                                                                  #
#**************************************************************************************************#
class Reader:
    FORMATS: tuple[Format, ...] = ()
    PACKAGES = ("miox", "nifti-mrs")  # packages whose versions go into the provenance
    PROGRESS = re.compile(r"\d+%\|")  # progress-bar lines
    LIBRARY_WARNINGS = (DeprecationWarning, PendingDeprecationWarning, FutureWarning)
    INTEGER_TEXT = re.compile(r"-?\d+\.0")  # an integer written as float text
    NO_GEOMETRY = np.diag([1e4, 1e4, 1e4, 1.0])  # a 10 m voxel: the file stores no geometry
    INTEGER_KEYS = ("DeviceSerialNumber", "PatientDoB")
    QUOTED_KEYS = ("RxCoil", "TxCoil")

    def read(self, sniff: Sniff) -> list[MRSData]:
        raise NotImplementedError

    def correct(self, d: MRSData) -> None:
        """Fix values the parsers or converters write wrongly; the reason goes into the source."""
        for key in self.INTEGER_KEYS:
            value = d.meta.get(key)
            if isinstance(value, str) and self.INTEGER_TEXT.fullmatch(value):
                self.fix(d, key, value.removesuffix(".0"), "integer written as float text")
        for key in self.QUOTED_KEYS:
            value = d.meta.get(key)
            if isinstance(value, str) and len(value) > 1 and value[0] == value[-1] == '"':
                self.fix(d, key, value[1:-1], "surrounding quotes removed")

    def dataset(
        self,
        sniff: Sniff,
        data: np.ndarray,
        dwelltime: float,
        values: dict[str, tuple[Any, str | None]],
        vendor: dict,
        name: str,
        notes: list[str],
        source: str,
        affine: np.ndarray | None = None,
        dim_tags: tuple[str, ...] = (),
    ) -> MRSData:
        """MRSData from data and header values a reader parsed itself.

        ``data`` is in NIfTI-MRS order and convention; ``values`` maps standard keys to
        (value, vendor field), and must hold SpectrometerFrequency and ResonantNucleus.
        """
        frequency = values["SpectrometerFrequency"][0]
        nucleus = values["ResonantNucleus"][0]
        hdr_ext = Hdr_Ext(frequency, nucleus)
        for index, tag in enumerate(dim_tags):
            hdr_ext.set_dim_info(index, tag)
        geometry = self.NO_GEOMETRY if affine is None else affine
        image = gen_nifti_mrs_hdr_ext(data, dwelltime, hdr_ext, geometry, no_conj=True)
        provenance = self.provenance(sniff, tuple(notes))
        d = MRSData.from_nifti_mrs(image, provenance, source, vendor=vendor, name=name)
        for key, (value, field) in values.items():
            d.meta.set(key, value, f"{source}: {field}" if field else source, field)
        self.correct(d)
        return d

    @staticmethod
    def fix(d: MRSData, key: str, value: Any, reason: str) -> None:
        source = d.meta.source.get(key, "")
        d.meta.set(key, value, f"{source}; corrected: {reason}", d.meta.fields.get(key))

    def provenance(self, sniff: Sniff, notes: tuple[str, ...] = ()) -> Provenance:
        return Provenance(
            paths=(sniff.path,) + sniff.companions,
            format=sniff.format,
            reader=type(self).__name__,
            versions={name: version(name) for name in self.PACKAGES},
            notes=notes,
        )

    @contextlib.contextmanager
    def captured(self, notes: list[str]) -> Iterator[None]:
        """Collect what wrapped libraries print or warn into ``notes`` instead of the terminal.

        Progress bars and warnings about library code (deprecations) are dropped; stdin is empty,
        so a library that asks a question fails instead of waiting for an answer.
        """
        out = io.StringIO()
        stdin = sys.stdin
        sys.stdin = io.StringIO()
        try:
            with (
                contextlib.redirect_stdout(out),
                contextlib.redirect_stderr(out),
                warnings.catch_warnings(record=True) as caught,
            ):
                warnings.simplefilter("always")
                yield
        finally:
            sys.stdin = stdin
            lines = (line.strip() for line in out.getvalue().splitlines())
            notes.extend(line for line in lines if line and not self.PROGRESS.search(line))
            notes.extend(
                str(w.message) for w in caught if not issubclass(w.category, self.LIBRARY_WARNINGS)
            )

    @classmethod
    def plain(cls, value: Any) -> Any:
        """Header value as plain Python (dict, list, str, int, float, bool, None)."""
        if isinstance(value, dict):
            return {str(k): cls.plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls.plain(v) for v in value]
        if isinstance(value, np.ndarray):
            return cls.plain(value.tolist())
        if isinstance(value, np.generic):
            return cls.plain(value.item())
        if isinstance(value, bytes):
            return value.decode("latin-1").rstrip("\0")
        return value


#**************************************************************************************************#
#                                          Spec2niiReader                                          #
#**************************************************************************************************#
#                                                                                                  #
# Base of the readers that convert through spec2nii. A subclass parses the file once, keeps the    #
# complete vendor header and calls the spec2nii converter; this class then does what spec2nii's    #
# command line does after every converter (spectral width from the dwell time, all-zero indices of #
# the higher dimensions removed) and builds the MRSData. ``SOURCES`` names the vendor field        #
# spec2nii takes each standard key from, so the metadata says where every value came from.         #
#                                                                                                  #
#**************************************************************************************************#
class Spec2niiReader(Reader):
    PACKAGES = Reader.PACKAGES + ("spec2nii",)
    CONVERTER = ""  # spec2nii subcommand
    SOURCES: dict[str, str | tuple[str, ...]] = {}  # standard key -> vendor field(s), first found
    MAX_MHZ = 1e5  # a larger spectrometer frequency is in Hz

    def convert(self, sniff: Sniff) -> tuple[list[NIFTI_MRS], list[str], dict]:
        """The converted images, their names and the complete vendor header."""
        raise NotImplementedError

    def converter(self, sniff: Sniff) -> str:
        """The spec2nii subcommand that converts this file."""
        return self.CONVERTER

    def sources(self, vendor: dict) -> dict[str, str | tuple[str, ...]]:
        """Where spec2nii takes each standard key from in this vendor header."""
        return self.SOURCES

    def read(self, sniff: Sniff) -> list[MRSData]:
        notes = []
        with self.captured(notes):
            images, names, vendor = self.convert(sniff)
            self._frequency_in_mhz(images, notes)
            images = self._as_spec2nii_command(images, notes)
        provenance = self.provenance(sniff, tuple(notes))
        source = f"spec2nii {self.converter(sniff)}"
        fields = self._fields(vendor)
        datasets = []
        for image, name in zip(images, names, strict=True):
            d = MRSData.from_nifti_mrs(image, provenance, source, vendor=vendor, name=name)
            for key, field in fields.items():
                if d.meta.source.get(key) == source:
                    d.meta.set(key, d.meta[key], f"{source}: {field}", field)
            self.correct(d)
            datasets.append(d)
        return datasets

    def _fields(self, vendor: dict) -> dict[str, str]:
        """The source field of every key: the first alternative present in the header."""
        present = Meta(vendor).vendor_fields()
        fields = {}
        for key, alternatives in self.sources(vendor).items():
            options = (alternatives,) if isinstance(alternatives, str) else alternatives
            fields[key] = next((f for f in options if f in present), options[0])
        return fields

    def _frequency_in_mhz(self, images: list[NIFTI_MRS], notes: list[str]) -> None:
        """Some converters store the frequency in Hz; no spectrometer runs above 100 GHz."""
        for image in images:
            frequencies = image.hdr_ext.SpectrometerFrequency
            if any(f > self.MAX_MHZ for f in frequencies):
                image.hdr_ext.SpectrometerFrequency = [f / 1e6 for f in frequencies]
                notes.append(f"SpectrometerFrequency {frequencies} is in Hz; divided by 1e6")

    def _as_spec2nii_command(self, images: list[NIFTI_MRS], notes: list[str]) -> list[NIFTI_MRS]:
        Spec2niiCommand.insert_spectralwidth(SimpleNamespace(imageOut=images))
        cleaned = []
        for image in images:
            before = image.shape
            image, zeros = remove_zero_higher_dim_indices(image, remove=True)
            removed = {d: i for d, i in zeros.items() if image.shape[d - 1] < before[d - 1]}
            if removed:
                notes.append(f"all-zero indices removed (dimension: indices): {removed}")
            cleaned.append(image)
        return cleaned

