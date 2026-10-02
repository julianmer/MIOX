####################################################################################################
#                                        fetch_test_data.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 01/10/26                                                                                #
#                                                                                                  #
# Purpose: Fetches the public test data into tests/data (gitignored): the spec2nii test set (git   #
#          LFS, a small per-format subset by default, --all for everything, retried on the         #
#          server's rate limit) and the NIfTI-MRS example data from Zenodo record 5085449.         #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


#**************************************************************************************************#
#                                           DataFetcher                                            #
#**************************************************************************************************#
#                                                                                                  #
# Downloads the public test data. Both sources resume: LFS objects already present and Zenodo      #
# files of the right size are skipped.                                                             #
#                                                                                                  #
#**************************************************************************************************#
class DataFetcher:
    SPEC2NII_URL = "https://git.fmrib.ox.ac.uk/wclarke/spec2nii_test_data.git"
    ZENODO_RECORD = "5085449"
    ZENODO_SMALL_LIMIT = 5_000_000  # bytes; larger Zenodo files only with --all

    # LFS include patterns for the subset: one small example per format and vendor
    SUBSET = [
        "Siemens/VBData/Twix/meas_MID153_svs_se_Tra_sat_FID108743.dat",
        "Siemens/VBData/DICOM/svs_se_Tra_sat_13_1/*",
        "Siemens/VBData/tim_trio_sop_class_uid/*",
        "Siemens/VEData/Twix/meas_MID00242_FID62747_svs_se_iso_tra_sat.dat",
        "Siemens/VEData/rda/svs_se_iso_tra_sat.rda",
        "Siemens/VEData/DICOM/svs_se_iso_tra_sat_13_1/*",
        "Siemens/VEData/T1.nii.gz",
        "Siemens/XAData/XA20/DICOM/*",
        "Siemens/XAData/XA20/rda/*",
        "Siemens/XAData/XA30/*.dcm",
        "Siemens/XAData/XA31/rda/*",
        "Siemens/anon/*",
        "Siemens/fid/*",
        "Siemens/rda/*",
        "UIH/mrs_data/dicom/svs_press_te144_SVS_801/*",
        "UIH/mrs_data/dicom/csi_hise_te144_CSI_1201/*",
        "bruker/PV-360.1.1/15/**",
        "bruker/PV-360.3.5/19/**",
        "bruker/PV-6.0.1/10/**",
        "ge/pFiles/svMRS/P03072.7",
        "ge/from_dicom/*",
        "philips/P1/*",
        "philips/DICOM/SV_phantom_center/*",
        "philips/DICOM/structural/*",
        "philips/hyper/*.list",
        "philips/hyper/*.SPAR",
        "philips/hyper/*.SDAT",
        "philips/hyper/*.dcm",
        "philips/HERCULES_spar_sdat/*",
        "jmrui/*",
        "other/*",
        "varian/**",
    ]

    def __init__(self, dest: Path, everything: bool, retries: int):
        self.dest = dest
        self.everything = everything
        self.retries = retries

    def run(self) -> None:
        self.dest.mkdir(parents=True, exist_ok=True)
        self.spec2nii(self.dest / "spec2nii_test_data")
        self.zenodo(self.dest / "nifti_mrs_examples")

    def spec2nii(self, dest: Path) -> None:
        """Clone the LFS pointers once, then pull the objects, waiting out the rate limit."""
        env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
        if not (dest / ".git").is_dir():
            print(f"cloning {self.SPEC2NII_URL} (pointers only)")
            subprocess.run(
                ["git", "clone", "--depth", "1", self.SPEC2NII_URL, str(dest)], check=True, env=env
            )
        cmd = ["git", "lfs", "pull"]
        if not self.everything:
            cmd += ["--include", ",".join(self.SUBSET)]
        for attempt in range(1, self.retries + 1):
            print(f"git lfs pull (attempt {attempt}/{self.retries})")
            if subprocess.run(cmd, cwd=dest, env=env).returncode == 0:
                break
            pause = min(60 * attempt, 300)
            print(f"LFS pull failed (rate limit?), waiting {pause}s")
            time.sleep(pause)
        else:
            sys.exit("LFS pull kept failing; rerun later, downloads resume where they stopped")
        left = subprocess.run(
            ["git", "lfs", "ls-files"], cwd=dest, capture_output=True, text=True
        ).stdout.count(" - ")
        print(f"spec2nii test data in {dest}; {left} objects still pointers")

    def zenodo(self, dest: Path) -> None:
        """Download the record's files, skipping those already complete."""
        dest.mkdir(parents=True, exist_ok=True)
        url = f"https://zenodo.org/api/records/{self.ZENODO_RECORD}"
        with urllib.request.urlopen(url) as resp:
            record = json.load(resp)
        for entry in record["files"]:
            if not self.everything and entry["size"] > self.ZENODO_SMALL_LIMIT:
                continue
            target = dest / entry["key"]
            if target.is_file() and target.stat().st_size == entry["size"]:
                continue
            print(f"downloading {entry['key']} ({entry['size'] / 1e6:.1f} MB)")
            urllib.request.urlretrieve(entry["links"]["self"], target)
        print(f"NIfTI-MRS examples in {dest}")


#**************************************************************************************************#
#                                               main                                               #
#**************************************************************************************************#
#                                                                                                  #
# Command line: --all for everything, --dest for another target folder.                            #
#                                                                                                  #
#**************************************************************************************************#
def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch the public test data into tests/data.")
    parser.add_argument("--all", action="store_true", help="fetch everything, not the subset")
    parser.add_argument(
        "--dest", type=Path, default=Path(__file__).resolve().parents[1] / "tests" / "data"
    )
    parser.add_argument("--retries", type=int, default=8)
    args = parser.parse_args()
    DataFetcher(args.dest, args.all, args.retries).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
