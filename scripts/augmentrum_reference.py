####################################################################################################
#                                     augmentrum_reference.py                                      #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 12/10/26                                                                                #
#                                                                                                  #
# Purpose: Reference results for MIOX's raw pipeline from Augmentrum's RawProcessor (torch engine, #
# defaults), in tests/data/augmentrum_reference. "export" (MIOX's environment) writes the cases    #
# from public files, conjugated into FSL-MRS's orientation: data (B, X, Y, Z, dims, T), water in   #
# NIfTI order, spectral width and frequency. "reference" (an environment with Augmentrum, torch    #
# and fsl-mrs; Augmentrum on PYTHONPATH) processes them and writes *.reference.npz.                #
#                                                                                                  #
####################################################################################################


#*************#
# imports                                                                                          #
#*************#
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "data" / "augmentrum_reference"
CASES = {
    "siemens_drift": "suspect/tests/test_data/siemens/example_frequency_correction.dat",
    "siemens_xa20": "spec2nii_test_data/Siemens/XAData/XA20/twix/"
                    "meas_MID00027_FID07653_svs_se_30_WS_on.dat",
    "ge_press": "spec2nii_test_data/ge/pFiles/PRESS/Big_GABA/G1/S01_PRESS_35_noID.7",
    "nifti_7t": "nifti_mrs_examples/example_01.nii.gz",
    "ge_press_48": "spec2nii_test_data/ge/pFiles/PRESS/MR30.1/P30101.7",
}


#**************************************************************************************************#
#                                        ReferenceBuilder                                          #
#**************************************************************************************************#
#                                                                                                  #
# Writes the cases (from MIOX's reads) and Augmentrum's results for them.                          #
#                                                                                                  #
#**************************************************************************************************#
class ReferenceBuilder:
    def export(self) -> None:
        import miox

        OUT.mkdir(parents=True, exist_ok=True)
        for name, rel in CASES.items():
            path = ROOT / "tests" / "data" / rel
            if not path.exists():
                print(f"{name}: {rel} not fetched")
                continue
            datasets = [d for d in miox.load(path) if isinstance(d, miox.MRSData)]
            met = next(d for d in datasets if not d.name.endswith("_ref"))
            wat = next((d for d in datasets if d.name.endswith("_ref")), None)
            # FSL-MRS's orientation: NIfTI-MRS as stored, conjugated (as nifti_mrs hands it out)
            arrays = {"met": np.moveaxis(np.asarray(met.data, np.complex128), 3, -1)[None].conj(),
                      "sw": 1 / met.dwelltime, "sf": met.spectrometer_frequency}
            info = {"source": rel, "tags": list(met.dim_tags[: met.data.ndim - 4])}
            if wat is not None:
                arrays["wat"] = np.asarray(wat.data, np.complex128)[None].conj()
                info["wtags"] = list(wat.dim_tags[: wat.data.ndim - 4])
            np.savez(OUT / f"{name}.npz", **arrays)
            (OUT / f"{name}.json").write_text(json.dumps(info))
            print(f"{name}: exported")

    def reference(self) -> None:
        from augmentrum.processing.raw_processing import RawProcessor

        for case in sorted(OUT.glob("*.json")):
            name = case.stem
            arrays = np.load(OUT / f"{name}.npz")
            info = json.loads(case.read_text())
            pad = lambda tags: tags + [None] * (3 - len(tags))  # noqa: E731
            kwargs = {"sw_hz": float(arrays["sw"]), "sf_mhz": float(arrays["sf"]),
                      "dim_tags": pad(info["tags"])}
            water = arrays["wat"] if "wat" in arrays else None
            if water is not None:
                kwargs["water_dim_tags"] = pad(info["wtags"])
            processor = RawProcessor(registration_method="torch")
            met, wat = processor.process_tensor(arrays["met"], water, **kwargs)
            out = {"met": np.asarray(met)}
            if wat is not None:
                out["wat"] = np.asarray(wat)
            if getattr(processor, "last_alignment_", None) is not None:
                out["phi"], out["eps"] = (np.asarray(v) for v in processor.last_alignment_)
            if getattr(processor, "last_keep_mask_", None) is not None:
                out["keep"] = np.asarray(processor.last_keep_mask_)
            np.savez(OUT / f"{name}.reference.npz", **out)
            print(f"{name}: reference written")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("export", "reference"))
    args = parser.parse_args()
    getattr(ReferenceBuilder(), args.step)()
    return 0


if __name__ == "__main__":
    sys.exit(main())
