<div align="center">
  <img src="assets/logo.png" alt="Miox Logo" width="130"/>
  <h1 style="margin-top: 5px; margin-bottom: 5px;">Miox</h1>
  <p style="margin-top: 0px;"><em>Open any MR file from any vendor into one object: data, geometry, all metadata</em></p>

  [![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
  [![License](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](LICENSE)
</div>

---

Miox reads MR spectroscopy, images, raw k-space, sequences, protocols, physiological recordings
and NMR data into one well-defined object each. It keeps the complete vendor header and records
where every standard key came from.

```python
import miox
[d] = miox.load("meas.dat")      # one object per dataset in the file
d.data, d.dims, d.dwelltime      # complex FIDs in NIfTI-MRS order
d.meta["EchoTime"], d.meta.source["EchoTime"], d.meta.vendor
```

```
miox info  path/to/file      # what a file holds and where each value came from
miox sniff path/to/file      # identify a file or folder without reading it whole
miox audit path/to/file      # every vendor header field and whether it is read
```

## Installation

```
git clone https://github.com/julianmer/miox.git
cd miox
uv sync --extra dev
```

## What it loads

| | Formats |
|---|---|
| **Spectroscopy** | Siemens twix, RDA and DICOM; Philips SPAR/SDAT, data/list and DICOM; UIH DICOM; GE P-file; Bruker; Varian; NIfTI-MRS; dcm2niix NIfTI with an MRS sidecar; LCModel .RAW; jMRUI text and binary; TARQUIN .dpt; VASF; Vespa VIFF; SIVIC DDF; FID-A .mat; text FIDs |
| **Images** | DICOM (classic, mosaic, enhanced, study folders); NIfTI with BIDS sidecar; Analyze; MGH; MINC 1 and 2; PAR/REC; AFNI; NRRD; MetaImage; MRtrix; Bruker 2dseq; Varian fdf; GE Signa |
| **Raw k-space** | ISMRMRD; MRD v2; Siemens twix; GE P-file and ScanArchive; Philips raw/lab/sin; Bruker fid and rawdata; Varian fid; fastMRI; BART .cfl |
| **Basis sets and priors** | LCModel .BASIS; FSL-MRS; SpinWizard; Osprey/MRSCloud; INSPECTOR; MARSS; ProFit; metabolite folders; Vespa, GAVA, MIDAS and PyAMARES priors |
| **Fit results** | LCModel COORD, TABLE and CSV; TARQUIN results and fit; PyAMARES; Osprey MRSCont |
| **Sequences and protocols** | Pulseq .seq; TOPPE; Siemens .pro, .exar and printed protocols; Philips ExamCards; LCModel control files; Yarra modes and modules |
| **Physio** | Siemens PMU; CMRR; Philips SCANPHYSLOG; GE; BIDS; HCP; BIOPAC AcqKnowledge |
| **Simulation** | KomaMRI, JEMRIS and MRiLab phantoms; JEMRIS signals and motion; SIMPSON |
| **NMR** | Bruker TopSpin; Varian/Agilent; JCAMP-DX; nmrML; JEOL Delta; Tecmag TNMR; NMRPipe; Sparky; Magritek Spinsolve; RS2D; RNMRTK |
| **Transforms** | FreeSurfer LTA; ITK; AFNI; FSL FLIRT |

Zip archives of any of these open as well. `docs/coverage/` lists, for every public test file,
each vendor header field and whether it is read.
