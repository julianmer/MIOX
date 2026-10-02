####################################################################################################
#                                       philips_spar_sdat.py                                       #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Reader for Philips SPAR/SDAT pairs. Either file of the pair can be given; spec2nii      #
#          converts and parses the SPAR, which is kept whole as the vendor header.                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

from pathlib import Path

from nifti_mrs.nifti_mrs import NIFTI_MRS
from spec2nii.Philips.philips import read_sdat_spar_pair, read_spar

# own
from miox.data import MRSData
from miox.readers.base import ReaderError, Spec2niiReader
from miox.sniff import Format, Sniff


#**************************************************************************************************#
#                                      PhilipsSparSdatReader                                       #
#**************************************************************************************************#
#                                                                                                  #
# Reads one SPAR/SDAT pair with spec2nii's defaults (dimension tags chosen from the data).         #
#                                                                                                  #
#**************************************************************************************************#
class PhilipsSparSdatReader(Spec2niiReader):
    FORMATS = (Format.PHILIPS_SPAR_SDAT,)
    CONVERTER = "philips"
    HYPER_SHORT_TE = 0.035  # s; HYPER acquires its short-TE block at 35 ms, the edited one at 80 ms
    HYPER_EDITED_TE = 0.08
    SOURCES = {
        "SpectrometerFrequency": "synthesizer_frequency",
        "ResonantNucleus": "nucleus",
        "EchoTime": "echo_time",
        "RepetitionTime": "repetition_time",
        "InversionTime": "spectrum_inversion_time",
        "TxOffset": "offset_frequency",
        "SoftwareVersions": "equipment_sw_verions",
        "ProtocolName": "scan_id",
        "PatientPosition": "patient_position",
        "PatientName": "patient_name",
        "PatientDoB": "patient_birth_date",
        "DwellTime": "sample_frequency",
    }

    def convert(self, sniff: Sniff) -> tuple[list[NIFTI_MRS], list[str], dict]:
        spar, sdat = self._pair(sniff)
        images, names = read_sdat_spar_pair(sdat, spar, None, [None, None, None], None, None)
        header = {key: self.plain(value) for key, value in read_spar(spar).items()}
        return list(images), list(names), header

    def correct(self, d: MRSData) -> None:
        """The SPAR inversion time unscaled (ms) and the HYPER short-TE reference set to 80 ms."""
        super().correct(d)
        if d.meta.fields.get("InversionTime") == "spectrum_inversion_time":
            self.fix(d, "InversionTime", d.meta["InversionTime"] * 1e-3, "SPAR value in ms")
        if d.name.endswith("_hyper_ref_short_te"):
            self.fix(d, "EchoTime", self.HYPER_SHORT_TE, "HYPER short-TE reference is 35 ms")

    def _pair(self, sniff: Sniff) -> tuple[Path, Path]:
        files = (sniff.path,) + sniff.companions
        spar = next((f for f in files if f.suffix.lower() == ".spar"), None)
        sdat = next((f for f in files if f.suffix.lower() == ".sdat"), None)
        if spar is None or sdat is None:
            raise ReaderError(f"incomplete SPAR/SDAT pair: {sniff.path.name}")
        return spar, sdat
