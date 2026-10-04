# MIOX

A Python package that opens any MR file from any vendor — spectroscopy first, imaging too —
into one well-defined object with data, geometry and all metadata, and shows it in one command.

Spectroscopy readers: Siemens twix, RDA and DICOM; Philips SPAR/SDAT and DICOM; UIH DICOM;
GE P-file; Bruker; NIfTI-MRS; LCModel .RAW; jMRUI. Data and geometry are assembled by spec2nii;
MIOX keeps the complete vendor header and records where every standard key came from.

```python
import miox as gu
[d] = gu.load("meas.dat")      # one MRSData per dataset in the file
d.data, d.dims, d.dwelltime    # complex FIDs in NIfTI-MRS order
d.meta["EchoTime"], d.meta.source["EchoTime"], d.meta.vendor
```

`load` returns what a file states; a file without frequency, nucleus or dwell time (headerless
text columns, a bare LCModel .RAW) gives a dataset whose `missing` names those keys. Pass them in
`given` and ask for `complete=True` when everything NIfTI-MRS needs must be there:

```python
[d] = gu.load("fid.txt", given={"SpectrometerFrequency": 123.2, "ResonantNucleus": "1H",
                                "SpectralWidth": 4000}, complete=True)   # IncompleteError if not
```

`miox info` prints what a file holds and where each value came from; `miox sniff`
identifies a file or directory without reading it whole; `miox audit` lists every field of
the vendor header and whether a key is read from it. `docs/coverage/` has that audit for every
public test file, one page per format.

```
uv sync --extra dev
uv run miox info path/to/file.dat
uv run miox sniff path/to/file.dat
uv run miox audit path/to/file.dat
uv run pytest                        # synthetic tests, no data needed
uv run python scripts/fetch_test_data.py   # public test data (subset; --all for everything)
uv run pytest -m data                # golden-record tests on the fetched data
```

Private test sets are picked up from `MIOX_PRIVATE_DATA` (directories separated by the
path separator); their paths never enter the repository.

Licence: BSD-3-Clause.
