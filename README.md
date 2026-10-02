# MIOX

A Python package that opens any MR file from any vendor — spectroscopy first, imaging too —
into one well-defined object with data, geometry and all metadata, and shows it in one command.

`miox sniff` identifies the format of a file or directory without reading it whole;
`miox audit` lists the metadata coverage of a NIfTI-MRS file.

```
uv sync --extra dev
uv run miox sniff path/to/file.dat
uv run miox audit path/to/svs.nii.gz
uv run pytest                        # synthetic tests, no data needed
uv run python scripts/fetch_test_data.py   # public test data (subset; --all for everything)
uv run pytest -m data                # golden-record tests on the fetched data
```

Private test sets are picked up from `MIOX_PRIVATE_DATA` (directories separated by the
path separator); their paths never enter the repository.

Licence: BSD-3-Clause.
