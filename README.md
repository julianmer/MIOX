<div align="center">
  <img src="assets/logo.png" alt="MIOX Logo" width="100"/>
  <h1 style="margin-top: 5px; margin-bottom: 5px;">MIOX</h1>
  <p style="margin-top: 0px;"><em>Open any MR file from any vendor into one object: data, geometry, all metadata</em></p>

  [![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
  [![License](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](LICENSE)
</div>

---

MIOX reads MR spectroscopy, images, raw k-space, sequences, protocols, physiological recordings
and NMR data into one well-defined object each. It keeps the complete vendor header and records
where every standard key came from.

```python
import miox
[d] = miox.load("meas.dat")      # one object per dataset in the file
d.data, d.dims, d.dwelltime      # complex FIDs in NIfTI-MRS order
d.meta["EchoTime"], d.meta.source["EchoTime"], d.meta.vendor
miox.view(d)                     # quick look in a window
```

## Installation

```
pip install pymiox
```

or, for development,

```
git clone https://github.com/julianmer/MIOX.git
cd MIOX
uv sync --extra dev
```

## What it loads

| | Formats |
|---|---|
| **Spectroscopy** | Siemens twix, RDA and DICOM; Philips SPAR/SDAT, data/list and DICOM; UIH DICOM; GE P-file; Bruker; Varian; NIfTI-MRS; dcm2niix NIfTI with an MRS sidecar; LCModel .RAW; jMRUI text and binary; TARQUIN .dpt; VASF; Vespa VIFF; SIVIC DDF; FID-A .mat; spant .rds; text FIDs |
| **Images** | DICOM (classic, mosaic, enhanced, study folders); NIfTI with BIDS sidecar; NIfTI-Zarr; JNIfTI; Analyze; MGH; MINC 1 and 2; PAR/REC; AFNI; NRRD; MetaImage; GIPL; Vista; Stimulate; VTK; MRtrix; SIVIC IDF; DSI Studio src and fib; BrainVoyager VMR, V16, VTC, FMR, VMP, GLM and VOI; Bruker 2dseq; Varian fdf; GE Signa; ISMRMRD image series |
| **Raw k-space** | ISMRMRD; MRD v2; Siemens twix; GE P-file and ScanArchive; Philips raw/lab/sin; Bruker fid and rawdata; Varian fid; fastMRI; BART .cfl; Gadgetron array dumps; MaRGE .mat (low-field MaRCoS consoles) |
| **Basis sets and priors** | LCModel .BASIS; FSL-MRS; SpinWizard; Osprey/MRSCloud; INSPECTOR; MARSS; ProFit; MRS-Sim; spant; metabolite folders; Vespa, GAVA, MIDAS and PyAMARES priors |
| **Fit results** | LCModel COORD, TABLE, CSV and PostScript tables; TARQUIN results and fit; PyAMARES; Osprey MRSCont; FSL-MRS dynamic; spant; DMIWizard maps |
| **Sequences and protocols** | Pulseq .seq; TOPPE; FSL-MRS simulator sequences; Siemens .pta and Bruker RF pulse shapes; Siemens .pro, .exar and printed protocols, raw-data DICOMs and Phoenix reports; Philips ExamCards and raw-data DICOMs; LCModel control files; MaRGE parameter tables; Yarra modes and modules; SPM.mat, FSL and BrainVoyager designs and stimulation protocols |
| **Physio** | Siemens PMU; CMRR; Philips SCANPHYSLOG; GE; BIDS; HCP; BIOPAC AcqKnowledge |
| **Simulation** | KomaMRI, JEMRIS and MRiLab phantoms; JEMRIS signals and motion; SIMPSON |
| **NMR** | Bruker TopSpin; Varian/Agilent; JCAMP-DX; nmrML; JEOL Delta; Tecmag TNMR; NMRPipe; Sparky; CSDM; Magritek Spinsolve; RS2D; RNMRTK |
| **Relaxometry** | Bruker minispec CPMG decays (.dps) with their fit reports (.mdt) and CONTIN T2 distributions |
| **Tractograms** | TrackVis .trk; MRtrix .tck and .tsf; TRX; VTK polydata; DSI Studio .tt |
| **Transforms** | FreeSurfer LTA; ITK/ANTs .tfm and .h5; AFNI affine matrices; FSL FLIRT; BrainVoyager TRF |
| **Surfaces** | FreeSurfer surfaces, per-vertex values, annotations and labels; GIFTI; BrainVoyager SRF, SMP, MTC, SSM and POI; MZ3 |
| **Transforms** | FreeSurfer LTA; ITK; AFNI; FSL FLIRT; BrainVoyager TRF |
