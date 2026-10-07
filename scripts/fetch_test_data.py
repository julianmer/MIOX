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
#          server's rate limit), the NIfTI-MRS examples and the OSI2ONE k-space data from Zenodo,  #
#          the test folders of further public repositories, single files by URL (ITK's GE images   #
#          through an IPFS gateway, ISMRMRD and Gadgetron k-space samples, one M4Raw file, the GE  #
#          P-files and ScanArchives of ge_to_ismrmrd, a Philips raw/lab/sin export), NEMA's        #
#          spectroscopy DICOM examples, MRD (version 2) streams made from fetched files by the     #
#          MRD project's own tools, and single members of large Zenodo archives (Varian imaging    #
#          scans) read through HTTP range requests.                                                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


#**************************************************************************************************#
#                                            fid_folder                                            #
#**************************************************************************************************#
#                                                                                                  #
# The files of a VnmrJ .fid folder inside an archive, by their target paths.                       #
#                                                                                                  #
#**************************************************************************************************#
def fid_folder(target: str, member: str) -> dict[str, str]:
    return {f"{target}/{name}": f"{member}/{name}" for name in ("fid", "procpar", "text", "log")}


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
    ZENODO_SMALL_LIMIT = 5_000_000  # bytes; larger files of a Zenodo record only with --all
    TIMEOUT = 120  # seconds without data before a download is retried
    ZENODO = {  # folder: (record, files always fetched; None for every file up to the limit)
        "nifti_mrs_examples": ("5085449", None),
        "osi2one": ("21807140", ("OSIIONE_Reference_System_Data_v1.zip",)),
        "osi2one_lumc": ("19661402", ()),
    }
    ZENODO_ALL = {  # with --all: further files of large records (CC BY 4.0)
        "osi2one_lumc": ("partial_fourier.zip",),
    }

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

    # further public sets: folder -> (git URL, paths to check out, or None for the whole repo)
    REPOSITORIES = {
        "mrs_fitting_challenge": ("https://github.com/wtclarke/mrs_fitting_challenge.git", None),
        "suspect": ("https://github.com/openmrslab/suspect.git", ["tests/test_data"]),
        "tarquin": ("https://github.com/martin3141/tarquin.git", ["data"]),
        "dcm_qa_spec": ("https://github.com/neurolabusc/dcm_qa_spec.git", None),
        "lcmodel": ("https://github.com/schorschinho/LCModel.git", ["test_lcm"]),
        "vespa_io": (
            "https://github.com/vespa-mrs/vespa.io.git",
            ["docs/user_manuals/files/tutorial_data_analysis.zip"],
        ),
        "hyperpolarized_toolbox": (
            "https://github.com/LarsonLab/hyperpolarized-mri-toolbox.git",
            ["reconstruction/EPSI demo/dummy_ddf"],
        ),
        "basis_toolbox": (
            "https://github.com/igweckay/MRS-Basis-Set-Conversion-Toolbox.git",
            ["sample_basis_sets"],
        ),
        "osprey": ("https://github.com/schorschinho/osprey.git", ["fit/basissets"]),
        "bidsphysio": (  # physio logs with their BIDS conversions (MIT)
            "https://github.com/cbinyu/bidsphysio.git",
            [f"bidsphysio.{m}/tests/data" for m in ("acq2bids", "base", "dcm2bids", "edf2bids",
                                                     "physio2bids", "pmu2bids")],
        ),
        "scanphyslog2bids": (  # Philips SCANPHYSLOG files with their scans (MIT)
            "https://github.com/lukassnoek/scanphyslog2bids.git", ["scanphyslog2bids/data"],
        ),
        "pyamares": ("https://github.com/HawkMRS/pyAMARES.git", ["pyAMARES/examples", "tests"]),
        "profit": ("https://gitlab.tuebingen.mpg.de/AG_Henning/ProFit-1D.git", ["sampleBasisSet"]),
        "nibabel": (
            "https://github.com/nipy/nibabel.git",
            ["nibabel/tests/data", "nibabel/nicom/tests/data"],
        ),
        "dcm_qa_nih": ("https://github.com/neurolabusc/dcm_qa_nih.git", None),
        "dcm_qa_enh": ("https://github.com/neurolabusc/dcm_qa_enh.git", None),
        "dcm_qa_mosaic": ("https://github.com/neurolabusc/dcm_qa_mosaic.git", None),
        "dcm_qa_ts": ("https://github.com/neurolabusc/dcm_qa_ts.git", None),
        "dcm_qa_12bit": ("https://github.com/neurolabusc/dcm_qa_12bit.git", None),
        "dcm_qa_frag": ("https://github.com/neurolabusc/dcm_qa_frag.git", None),
        "dcm_qa_me": ("https://github.com/neurolabusc/dcm_qa_me.git", None),
        "dcm_qa_pdt2": ("https://github.com/neurolabusc/dcm_qa_pdt2.git", None),
        "dcm_qa": ("https://github.com/neurolabusc/dcm_qa.git", None),
        "dcm_qa_ge": ("https://github.com/neurolabusc/dcm_qa_ge.git", None),
        "dcm_qa_xa30": ("https://github.com/neurolabusc/dcm_qa_xa30.git", None),
        "dcm_qa_philips": ("https://github.com/neurolabusc/dcm_qa_philips.git", None),
        "dcm_qa_philips_enh": ("https://github.com/neurolabusc/dcm_qa_philips_enh.git", None),
        "dcm_qa_philips_asl": ("https://github.com/neurolabusc/dcm_qa_philips_asl.git", None),
        "dcm_qa_uih": ("https://github.com/neurolabusc/dcm_qa_uih.git", None),
        "dcm_qa_canon_enh": ("https://github.com/neurolabusc/dcm_qa_canon_enh.git", None),
        "dcm_qa_toshiba": ("https://github.com/neurolabusc/dcm_qa_toshiba.git", None),
        "dcm_qa_agfa": ("https://github.com/neurolabusc/dcm_qa_agfa.git", None),
        "dcm_qa_fmap": ("https://github.com/neurolabusc/dcm_qa_fmap.git", None),
        "dcm_qa_dti": ("https://github.com/neurolabusc/dcm_qa_dti.git", None),
        "dcm_qa_asl": ("https://github.com/neurolabusc/dcm_qa_asl.git", None),
        "dcm_qa_stc": ("https://github.com/neurolabusc/dcm_qa_stc.git", None),
        "mrtrix_test_data": (
            "https://github.com/MRtrix3/test_data.git",
            ["dwi.mif", "mask.mif", "template.mif.gz", "mrmath", "mrconvert"],
        ),
    }
    # single files: folder -> (base URL, {relative path: name or content id under the base})
    PYNRRD = "https://raw.githubusercontent.com/mhe/pynrrd/master/nrrd/tests/data"  # MIT
    SWELLING = (  # fdf images and their ImageJ text exports, CC BY 4.0 (Beata Wereszczyńska)
        "https://raw.githubusercontent.com/BeataWereszczynska/"
        "Swelling_tablet_fronts_D_k_from_MRI_T2_or_img/main"
    )
    IPFS = "https://ipfs.filebase.io/ipfs"  # gateway to ITK's test data store, Apache 2.0
    PYNRRD_FILES = (
        "BallBinary30x30x30.nhdr", "BallBinary30x30x30.nii.gz", "BallBinary30x30x30.nrrd",
        "BallBinary30x30x30.raw", "BallBinary30x30x30_byteskip_minus_five.nhdr",
        "BallBinary30x30x30_byteskip_minus_one.nhdr",
        "BallBinary30x30x30_byteskip_minus_one_nifti.nhdr", "BallBinary30x30x30_bz2.nrrd",
        "BallBinary30x30x30_gz.nrrd", "BallBinary30x30x30_gz_byteskip_minus_one.nrrd",
        "BallBinary30x30x30_gz_lineskip.nrrd", "BallBinary30x30x30_nifti.nhdr",
        "test1d_ascii.nrrd", "test2d_ascii.nrrd", "test_customFields.nrrd",
        "test_simple4d_raw.nrrd",
    )
    SWELLING_FILES = (
        "MRI_FDF/0009.fdf", "MRI_FDF/0030.fdf", "MRI_TXTimages/0009.txt", "MRI_TXTimages/0030.txt",
        "LICENSE",
    )
    ITK_GE_FILES = {  # relative path: content id
        "test_ge5/113766.003.001": "bafkreiecl6v2p36p5ewgmmptvjajopa2bod3bcnqdgq2rosq4nmd25fl3e",
        "test_ge5/113766.003.002": "bafkreicbywnrrjcl2wehhbqmff73bjcwwoucopxutj53g5virc5x2qal6m",
        "test_ge5/113766.003.003": "bafkreihvzsg2qtvsy7pbdry25tzm5gzjqwyo5gyfw36vix26chj5eu4oe4",
        "test_ge5/113766.003.004": "bafkreiffchdk6cqi2tkyysxfwbaw2kqit6k7fq7exnsbopn76ehrtp2jfq",
        "test_ge5/113766.003.005": "bafkreiafcd4siu7wrrm7c6k5ou5a7wm4p2j7iyczgngjh5h5a74rmk4eem",
        "test_ge5/113766.003.006": "bafkreicv562xhqs7mdeasawxh5p5kjnhua4boomkvjp2i4ob6fskhvnv5i",
        "test_ge4/19771.002.001": "bafkreidl5lywjunmfxfy6kbaocfok2fuehj5bx4ciiozw46pcxaa44rzte",
        "test_ge4/19771.002.002": "bafkreib5x4r4zcroilk36n2bsx3wo7y5grmoosudojifnn2dcf2viwieiu",
        "test_ge4/19771.002.003": "bafkreickggjzritwhvbdqj64w6s4olnsh5f4icp6ycqyvtidpzybqitndm",
        "test_ge4/19771.002.004": "bafkreia6nz4nrxgal47ns5epex7wxgaag4gughg5xorjem7oewdj7ck6nu",
        "test_ge4/19771.002.005": "bafkreic4zoyqnybqogydtkiwidosy3v6xwnofzjnofir7txgd553bel4zq",
        "test_ge4/19771.002.006": "bafkreiftiogu3h7dpyk55svoz6fw7ybzlfd3u3odancrs23rhgljrbjvdq",
        "ct_genesis/c_vf1122.fre": "bafkreia6hxgzn43ez65h4tren4dxlcoheh665pfaq4rysylc6tygazztd4",
        "ct_genesis/c_vf1123.fre": "bafkreidcnq2bohnkvzvmexfbpedg4xlxvov77a2nqcvoyfn2deu2eqptmy",
        "ct_genesis/c_vf1210.fre": "bafkreih35u5ecdox6sbbwphke7f3xsq7dea7gdeeeizorgyo67e6sabcwi",
        "ct_genesis/c_vf1211.fre": "bafkreif26rvs3wdnvsoiyycpidrirr4fygob3fsn3bfhlde6e7frllfpgq",
        "baseline/c_vf1122.fre.mha": "bafybeieqjaqvkwbubml5lsnwzy25wlgadhwqtytt6mig6g6n6f7yb3tfcu",
        "baseline/c_vf1210.fre.mha": "bafybeifecxwzugdiethtnupgvc5uumlkyucp7zbkokfbzpzgb6nk2xecri",
        "baseline/113766.003.001.mha":
            "bafkreighxzakrzvtd5ykjw5ju4eycxwmxcruphkqi6weza2zayhx2ub33a",
        "baseline/19771.002.001.mha": "bafkreidvxasu7bdqrqvsicgkmirp4tzunpgnm6z4y6egoo5ch3f6yyrxyi",
    }
    ISMRMRD = "https://downloads.sourceforge.net/project/ismrmrd/data"  # ISMRMRD's own samples
    ISMRMRD_FILES = ("simple_gre.h5", "simple_spiral.h5")
    ISMRMRD_FILES_ALL = ("3D_partial_fourier.h5",)
    GADGETRON = "https://gadgetrondata.blob.core.windows.net/gadgetrontestdata"
    GADGETRON_FILES = (  # twix imaging .dat with their ISMRMRD conversions and reconstructions
        "simple_gre/meas_MiniGadgetron_GRE.dat", "simple_gre/simple_gre_in_20220831.mrd",
        "simple_gre/ismrmrd_dump_output_20240811.h5", "simple_gre/simple_gre_out_20210909_klk.mrd",
        "simple_gre/ref_20240810.mrd",
        "radial_phantom/meas_MID00133_FID20080_CV_Radial_Fixed_Angle_128_x8_32phs.dat",
        "radial_phantom/fixed_radial_mode1_20210923_klk.mrd",
        "T2W/meas_MID00057_T2w.dat", "T2W/generic_grappa_T2W_ref_20220817_klk.mrd",
        "generic/meas_MID01095_FID40115/meas_MID01095_FID40115.mrd",
        "generic/meas_MID01095_FID40115/ref_20210923_klk.mrd",
        "matlab/Kidney4cmSlice.mrd", "hyper/Kidney4cmSlice_sb_outV4.mrd",
        "spiral/simple_spiral_out_v2.mrd", "gre_3d/simple_gre_out_3d_20210910_klk.mrd",
        "grappa_3d/grappa2x1_ref_20210917.mrd", "epi/epi_2d_out_20161020_pjv.mrd",
        "rtgrappa/grappa_rate2_cpu_out.mrd",
    )
    GADGETRON_FILES_ALL = (  # the large raw files (0.1 to 1.2 GB each)
        "gre_3d/meas_MID248_gre_FID30644.dat", "spiral/meas_MID00080_FID16005_MiniIRT_B0.dat",
        "spiral/meas_MID1132_MiniIRT_spiral_16int_tr500_acc1_10reps_FID13142.dat",
        "rtgrappa/acc_data_with_device_2.dat", "epi/meas_MID517_nih_ep2d_bold_fa60_FID82077.dat",
        "grappa_3d/gre_3D_Grappa2x1.mrd", "grappa_3d/gre_3D_Grappa2x2.mrd",
        "grappa_3d/grappa2x2_ref_20210917.mrd",
        "generic/meas_MID836_1slice_FID22517/meas_MID836_1slice_FID22517.dat",
        "generic/meas_MID836_1slice_FID22517/ref_20220817_klk.mrd",
    )
    M4RAW = "https://huggingface.co/datasets/mylyu/M4Raw_brain/resolve/main"  # CC BY 4.0
    M4RAW_FILES = ("multicoil_test/2022101103_T206.h5", "H5_FORMAT.md", "LICENSE")
    LABRAWSIN = "https://raw.githubusercontent.com/welcheb/labrawsin_env/main"  # MIT
    LABRAWSIN_FILES = {  # one Philips raw/lab/sin export (coronal GRE)
        **{f"20250617_210621_COR_GRE.{s}": f"20250617_210621_COR_GRE.{s}"
           for s in ("lab", "raw", "sin")},
        "LICENSE": "LICENSE",
    }
    GE_TO_ISMRMRD = "https://raw.githubusercontent.com/ismrmrd/ge_to_ismrmrd/master"  # NIH
    GE_TO_ISMRMRD_FILES = {  # P-files and ScanArchives of GE's converter, public domain
        "P20480_GRE.7": "sampleData/P20480_GRE.7", "P21504_FSE.7": "sampleData/P21504_FSE.7",
        "ScanArchive_GRE.h5": "sampleData/ScanArchive_GRE.h5",
        "ScanArchive_FSE.h5": "sampleData/ScanArchive_FSE.h5",
        "ScanArchive_EPI.h5": "sampleData/ScanArchive_EPI.h5", "LICENSE": "LICENSE",
    }
    OSF = "https://osf.io"  # phys2bids' test files (project 3txqr, no licence stated)
    OSF_FILES = {  # GE physio logs: file: OSF file id
        "PPGData_epiRT_0000000000_00_00_000.gep": "wb84d/download",
        "RESPData_epiRT_0000000000_00_00_000.gep": "qawjv/download",
        "PPGData_epiRT_0000000000_00_00_000": "u9wsr/download",
        "RESPData_epiRT_0000000000_00_00_000": "49xpw/download",
    }
    MRD_REPOSITORY = "https://raw.githubusercontent.com/ismrmrd/mrd/main"  # MIT
    MRD_FILES = {  # a Pulseq file the MRD tools convert, and the licence
        "haste.seq": "test/test_data/pulseq/1.4.2/haste.seq", "LICENSE": "LICENSE",
    }
    FILES = {
        "pynrrd": (PYNRRD, dict(zip(PYNRRD_FILES, PYNRRD_FILES, strict=True))),
        "agilent_fdf": (SWELLING, dict(zip(SWELLING_FILES, SWELLING_FILES, strict=True))),
        "itk_ge": (IPFS, ITK_GE_FILES),
        "ismrmrd": (ISMRMRD, dict(zip(ISMRMRD_FILES, ISMRMRD_FILES, strict=True))),
        "gadgetron": (GADGETRON, dict(zip(GADGETRON_FILES, GADGETRON_FILES, strict=True))),
        "m4raw": (M4RAW, dict(zip(M4RAW_FILES, M4RAW_FILES, strict=True))),
        "ge_to_ismrmrd": (GE_TO_ISMRMRD, GE_TO_ISMRMRD_FILES),
        "labrawsin": (LABRAWSIN, LABRAWSIN_FILES),
        "mrd": (MRD_REPOSITORY, MRD_FILES),
        "phys2bids_ge": (OSF, OSF_FILES),
    }
    FILES_ALL = {  # with --all
        "ismrmrd": (ISMRMRD, dict(zip(ISMRMRD_FILES_ALL, ISMRMRD_FILES_ALL, strict=True))),
        "gadgetron": (
            GADGETRON, dict(zip(GADGETRON_FILES_ALL, GADGETRON_FILES_ALL, strict=True))
        ),
    }
    STRAWBERRY = "zenodo_repo_strawberry_mri"  # zenodo 7442381 (CC BY 4.0), one 13 GB zip
    STRAWBERRY_SCANS = f"{STRAWBERRY}/hydroref/s_20220824_strawberry_stol01_day01_hydroref_01"
    ZIP_MEMBERS = {  # folder: (Zenodo record, archive, {target: member})
        "varian_strawberry": ("7442381", f"{STRAWBERRY}.zip", {
            **fid_folder("mems.fid", f"{STRAWBERRY_SCANS}/mems_T2_MESE_strawberries_ms_T2_MESE_"
                         "sagittal_01.fid"),
            **fid_folder("fsems.fid", f"{STRAWBERRY_SCANS}/fsems_varianIR_prep_strawb_varianIR_"
                         "strawberry_sagittal_01.fid"),
            "t2_map.MAT": f"{STRAWBERRY}/qmaps/sb_href/s0_t2_2par_s01d01_hydroref.MAT",
            "readme.txt": f"{STRAWBERRY}/readme.txt",
        }),
    }
    ZIP_MEMBERS_ALL = {  # with --all
        "varian_strawberry": ("7442381", f"{STRAWBERRY}.zip", {
            **fid_folder("mge3d.fid", f"{STRAWBERRY_SCANS}/mge3d_qsm_strawberry_lowres_mge_"
                         "6echo_qsm_01.fid"),
            "t2star_map.MAT":
                f"{STRAWBERRY}/qmaps/sb_href/s0_t2star_2par_s01d01_href_intmaskautoset.MAT",
        }),
    }
    MRD_PYTHON = "mrd-python==2.2.1"  # the MRD project's tools (MIT), run through uv only here
    MRD_CONVERSIONS = {  # MRD stream: ISMRMRD file it is converted from (ismrmrd_to_mrd)
        "generic_gre.mrd": "gadgetron/generic/meas_MID01095_FID40115/meas_MID01095_FID40115.mrd",
        "simple_gre.mrd": "gadgetron/simple_gre/simple_gre_in_20220831.mrd",
        "kidney_spiral.mrd": "gadgetron/matlab/Kidney4cmSlice.mrd",
        "simple_gre_images.mrd": "gadgetron/simple_gre/simple_gre_out_20210909_klk.mrd",
        "epi_images.mrd": "gadgetron/epi/epi_2d_out_20161020_pjv.mrd",
        "radial_images.mrd": "gadgetron/radial_phantom/fixed_radial_mode1_20210923_klk.mrd",
        "hyper_images.mrd": "gadgetron/hyper/Kidney4cmSlice_sb_outV4.mrd",
    }
    MRD_NDJSON = (  # binary streams also written as NDJSON
        "import sys, mrd\n"
        "with mrd.BinaryMrdReader(sys.argv[1]) as r, mrd.NDJsonMrdWriter(sys.argv[2]) as w:\n"
        "    w.write_header(r.read_header())\n"
        "    w.write_data(r.read_data())\n"
    )
    MRD_NOISE = (  # the noise covariance of a stream's noise readouts, by MRD's own writer
        "import sys, numpy as np, mrd\n"
        "with mrd.BinaryMrdReader(sys.argv[1]) as r:\n"
        "    head = r.read_header()\n"
        "    noise = [item.value for item in r.read_data()\n"
        "             if isinstance(item, mrd.StreamItem.Acquisition)\n"
        "             and item.value.head.flags & mrd.AcquisitionFlags.IS_NOISE_MEASUREMENT]\n"
        "data = np.concatenate([a.data for a in noise], axis=1)\n"
        "info = head.acquisition_system_information\n"
        "covariance = mrd.NoiseCovariance(\n"
        "    coil_labels=list(info.coil_label),\n"
        "    receiver_noise_bandwidth=info.relative_receiver_noise_bandwidth or 1.0,\n"
        "    noise_dwell_time_ns=noise[0].head.sample_time_ns or 0, sample_count=data.shape[1],\n"
        "    matrix=(data @ data.conj().T / data.shape[1]).astype(np.complex64))\n"
        "with mrd.BinaryMrdNoiseCovarianceWriter(sys.argv[2]) as w:\n"
        "    w.write_noise_covariance(covariance)\n"
    )
    NEMA = "ftp://medical.nema.org/medical/dicom/DataSets/WG16"
    NEMA_FILES = {  # spectroscopy DICOM examples of the DICOM working group 16
        "Siemens/Siemens_spectroEnhDICOM": None,  # None: every file in the folder
        "Philips/Philips%20MR%20Spectroscopy/DICOM": "XX_",  # only files starting with this
    }

    def __init__(self, dest: Path, everything: bool, retries: int):
        self.dest = dest
        self.everything = everything
        self.retries = retries
        socket.setdefaulttimeout(self.TIMEOUT)

    def run(self) -> None:
        self.dest.mkdir(parents=True, exist_ok=True)
        self.spec2nii(self.dest / "spec2nii_test_data")
        for name, (record, keys) in self.ZENODO.items():
            more = self.ZENODO_ALL.get(name, ()) if self.everything else ()
            self.zenodo(self.dest / name, record, None if keys is None else keys + more)
        for name, (url, paths) in self.REPOSITORIES.items():
            self.repository(self.dest / name, url, paths)
        groups = (self.FILES, self.FILES_ALL) if self.everything else (self.FILES,)
        for group in groups:
            for name, (base, files) in group.items():
                self.files(self.dest / name, base, files)
        self.nema(self.dest / "nema_wg16")
        groups = (self.ZIP_MEMBERS, self.ZIP_MEMBERS_ALL) if self.everything else (
            self.ZIP_MEMBERS,)
        for group in groups:
            for name, (record, archive, members) in group.items():
                self.zip_members(self.dest / name, record, archive, members)
        self.mrd(self.dest / "mrd")

    def repository(self, dest: Path, url: str, paths: list[str] | None) -> None:
        """Shallow clone; with ``paths`` only those are checked out."""
        if (dest / ".git").is_dir():
            print(f"{dest.name} already present")
            return
        if paths is None:
            subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)
            return
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--no-checkout", url, str(dest)],
            check=True,
        )
        subprocess.run(["git", "sparse-checkout", "set", "--no-cone", *paths], cwd=dest, check=True)
        subprocess.run(["git", "checkout"], cwd=dest, check=True)
        for archive in dest.rglob("*.zip"):
            shutil.unpack_archive(archive, archive.with_suffix(""))

    def files(self, dest: Path, base: str, files: dict[str, str]) -> None:
        """Download single files, skipping those present; gateways are retried with a pause."""
        for relative, name in files.items():
            url, target = f"{base}/{name}", dest / relative
            if target.is_file():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part")
            for attempt in range(1, self.retries + 1):
                try:
                    print(f"downloading {dest.name}/{relative}")
                    urllib.request.urlretrieve(url, partial)
                    partial.replace(target)
                    break
                except OSError as e:
                    print(f"  failed ({e}), attempt {attempt}/{self.retries}")
                    time.sleep(min(10 * attempt, 60))
            else:
                sys.exit(f"could not fetch {url}")
            time.sleep(0.5)  # the gateways rate-limit bursts
        print(f"{dest.name}: {len(files)} files")

    def mrd(self, dest: Path) -> None:
        """MRD files written by the MRD project's tools: conversions of fetched ISMRMRD files,
        a phantom (noise scans, trajectories) with its reconstruction, a Pulseq file, NDJSON
        copies and a noise covariance; files already present are skipped."""
        python = ["uv", "run", "--no-project", "--quiet", "--with", self.MRD_PYTHON, "python"]
        jobs = [
            (name, ["-m", "mrd.tools.ismrmrd_to_mrd", "-d", str(self.dest / source), "-o"],
             self.dest / source)
            for name, source in self.MRD_CONVERSIONS.items()
        ]
        phantom = ["-m", "mrd.tools.phantom", "-c", "4", "-m", "32", "-r", "2", "-s", "2",
                   "-n", "0", "-C", "-K", "-o"]
        jobs += [
            ("phantom.mrd", phantom, None),
            ("phantom_recon.mrd", ["-m", "mrd.tools.stream_recon", "-i", str(dest / "phantom.mrd"),
                                   "-o"], dest / "phantom.mrd"),
            ("haste.mrd", ["-m", "mrd.tools.seq_to_mrd", "-i", str(dest / "haste.seq"), "-o"],
             dest / "haste.seq"),
        ]
        jobs += [
            (f"{stem}.ndjson", ["-c", self.MRD_NDJSON, str(dest / f"{stem}.mrd")],
             dest / f"{stem}.mrd")
            for stem in ("phantom", "phantom_recon", "haste")
        ]
        jobs.append(("simple_gre_noise_covariance.mrd",
                     ["-c", self.MRD_NOISE, str(dest / "simple_gre.mrd")], dest / "simple_gre.mrd"))
        dest.mkdir(parents=True, exist_ok=True)
        for name, args, source in jobs:
            target = dest / name
            if target.is_file():
                continue
            if source is not None and not source.is_file():
                print(f"mrd/{name}: {source.name} not fetched, skipped")
                continue
            partial = target.with_name(name + ".part")
            print(f"writing mrd/{name}")
            subprocess.run(python + args + [str(partial)], check=True)
            partial.replace(target)
        print(f"MRD streams in {dest}")

    def zip_members(self, dest: Path, record: str, archive: str, members: dict[str, str]) -> None:
        """Extract single members of a Zenodo zip archive without downloading the archive: the
        central directory and the members are read through HTTP range requests."""
        missing = {t: m for t, m in members.items() if not (dest / t).is_file()}
        if not missing:
            print(f"{dest.name}: {len(members)} files")
            return
        url = f"https://zenodo.org/api/records/{record}/files/{archive}/content"
        with zipfile.ZipFile(io.BufferedReader(RangeFile(url), buffer_size=1 << 20)) as zf:
            for target, member in missing.items():
                path = dest / target
                path.parent.mkdir(parents=True, exist_ok=True)
                partial = path.with_name(path.name + ".part")
                print(f"extracting {dest.name}/{target}")
                with zf.open(member) as source, open(partial, "wb") as out:
                    shutil.copyfileobj(source, out, 1 << 20)
                partial.replace(path)
        print(f"{dest.name}: {len(members)} files")

    def nema(self, dest: Path) -> None:
        """Download the selected files of the NEMA FTP folders, skipping those already present."""
        for folder, prefix in self.NEMA_FILES.items():
            target = dest / urllib.parse.unquote(folder)
            target.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(f"{self.NEMA}/{folder}/") as resp:
                names = [line.split()[-1] for line in resp.read().decode().splitlines() if line]
            for name in names:
                if (prefix and not name.startswith(prefix)) or (target / name).is_file():
                    continue
                print(f"downloading {folder}/{name}")
                urllib.request.urlretrieve(f"{self.NEMA}/{folder}/{name}", target / name)
        print(f"NEMA spectroscopy DICOM in {dest}")

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

    def zenodo(self, dest: Path, record: str, keys: tuple[str, ...] | None) -> None:
        """Download the named files of a Zenodo record (all of them up to the size limit when
        ``keys`` is None), skipping those already complete; zip archives are unpacked."""
        dest.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"https://zenodo.org/api/records/{record}") as resp:
            entries = json.load(resp)["files"]
        for entry in entries:
            if keys is None:
                if not self.everything and entry["size"] > self.ZENODO_SMALL_LIMIT:
                    continue
            elif entry["key"] not in keys:
                continue
            target = dest / entry["key"]
            if target.is_file() and target.stat().st_size == entry["size"]:
                continue
            print(f"downloading {entry['key']} ({entry['size'] / 1e6:.1f} MB)")
            urllib.request.urlretrieve(entry["links"]["self"], target)
            if target.suffix == ".zip":
                shutil.unpack_archive(target, target.with_suffix(""))
        print(f"Zenodo record {record} in {dest}")


#**************************************************************************************************#
#                                            RangeFile                                             #
#**************************************************************************************************#
#                                                                                                  #
# A remote file as a seekable read-only stream; every read is one HTTP range request.              #
#                                                                                                  #
#**************************************************************************************************#
class RangeFile(io.RawIOBase):
    def __init__(self, url: str):
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as response:
            self.size = int(response.headers["Content-Length"])
            self.url = response.url
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.position, 2: self.size}[whence]
        self.position = base + offset
        return self.position

    def readinto(self, buffer) -> int:
        count = min(len(buffer), self.size - self.position)
        if count <= 0:
            return 0
        span = f"bytes={self.position}-{self.position + count - 1}"
        request = urllib.request.Request(self.url, headers={"Range": span})
        with urllib.request.urlopen(request) as response:
            data = response.read()
        buffer[: len(data)] = data
        self.position += len(data)
        return len(data)


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
