# Server retirement backup

The final circuit correctness audit is in `results/negative_audit_20261007`.
Both honest controls (Groth16 and PLONK) passed. Six original corruption cases
and two isolated tie/input-boundary cases were rejected by circuit constraints.
The server service exited with status 0 in 864.78 seconds. These are correctness
checks, separate from the 560 measured campaign proofs.

The backup directory is `results/server_retirement_20261007`. Its
`transfer_state.json` records verified archives and active transfer progress.
`compact_download_complete.json` is written only after every selected compact
archive has been downloaded and checked against its server SHA256 receipt.
An archive ending in `.partial` is not a completed, verified archive.

The transfer is now complete: all 19 compact archives (2,920,811,271 bytes)
passed SHA256 verification. Local validation also passed for all 40,638 unique
stored or regenerated arrays, covering every indexed NPZ file. The verification
receipts are `compact_download_complete.json` and `array_verification.json`.
All remaining arrays were subsequently checked, with no missing objects.
Incomplete downloads and duplicated payloads were removed during final cleanup.

`two_reports_paper_core_20261007.tar.gz` is the portable final bundle containing
all downloaded campaign rows, logs, proof records, both report editions, shared derived tables,
figures, Persian/English PDFs and current manuscript. Its receipt lists its hash and
file count. Extract into `remote_project_snapshot/` to place the core records
beside the preserved source. Recovery tools are installed in that snapshot's
`tools/` directory. The complete numerical backup is separate, in the 19
compact archives and the extracted content-addressed object directory.

## Already preserved essential records

- All previously downloaded result rows, aggregate statistics, confidence
  intervals, figures, manifests and proof/verification records for the three
  campaigns; the existing core bundles contain 2,248 checked files.
- The circuit audit's actual instances, detailed rejection errors, timings and
  systemd exit status.
- The server's source/configuration snapshot, original Linux prover, verifier
  probe, dependency freeze, service definitions, journals and environment.
- The complete 135 MiB dataset cache, packed into an 81 MB archive.
- The complete small quick campaign, including its NPZ arrays and saved
  proof/public-witness/verifying-key artifacts.

The source and dataset archives have also been extracted into
`remote_project_snapshot/` inside the backup directory. The original prover's
SHA256 is `a66dd0ddfd788ebbb4f4dee224f3c5c15bd134d884c89d58bff902ba591aaafa`.

## Compact numerical array bundle

This bundle indexes 13,772 NPZ files from the campaigns and smoke runs. Identical
arrays are stored once as compressed NPY objects, retaining their original
shape, dtype and expected NPY SHA256. The 39,907 stored objects are divided into
19 `compact_arrays_*.tar.gz` archives. The index is
`priority_compact_index.tar.gz`.

The 1,444 DBSCAN squared-distance arrays are represented by their preserved
coordinates (X or Z), pair indices, original calculation function and expected
hash. Server-side reconstruction was checked for all 722 DBSCAN files:
every reconstructed array and NPY hash matched the original. See
`distance_regeneration_validation.json`. No point sampling or rounding was
introduced for this backup.

The 3,646 RQ1 files under `artifacts/maps/` are Z projection caches and are
excluded from the compact transfer. Their exact paths, original file hashes
and sizes remain indexed. Input subsets, PCA components, configuration/seeds
and numerical result arrays are retained. A cache can be rebuilt without
rerunning clustering or proving; see the command below. All six implemented
RQ1 projection methods passed a recovery check against an actual small run,
including equality of the recovered NPZ file hashes on the test environment.

From the repository's `code/` directory:

```powershell
python tools/restore_retirement_arrays.py --compact --archive-dir results/server_retirement_20261007 --verify-only
python tools/restore_retirement_arrays.py --compact --archive-dir results/server_retirement_20261007 --destination .
python -m tools.rebuild_rq1_map --dataset-dir results/server_paper_20261006/rq1/gaussian_easy --method sparse_jl --m 32 --q 2 --seed 0
```

The restoration tool maps the original remote `paper_server_20261006` name to
the existing local `server_paper_20261006` directory and merges its arrays with
the already downloaded core records. It refuses to overwrite a differing
existing file. Files from the follow-up and quick campaigns keep their names.

The recorded server numerical environment used NumPy 2.5.3. If a regenerated
floating-point array fails its expected hash on another environment, use the
recorded Linux package versions; the tool rejects a mismatch rather than
silently accepting a changed result. The `.npz` container hash can differ after
restoration while its arrays are identical; array hashes are checked separately.

## Current report editions

- Comprehensive Persian: `results/reports/full_persian_20261007/report.html`;
  PDF: `output/pdf/commitjl_full_report_fa.pdf`.
- English manuscript-table digest: `results/reports/paper_tables_english_20261007/report.html`;
  PDF: `output/pdf/commitjl_paper_tables_en.pdf`.
- Shared scientific exports: `results/server_study_data_20261007/`.
- Report-only delivery: `output/commitjl_two_reports_20261007.zip`.

The earlier combined report was replaced. Its scientific CSVs and figures were
preserved as shared inputs; removing that report does not remove server results.
See `report_replacement_receipt.json` in the shared data folder for cleanup mode.
The four server campaign folders, compact archive/object backup and datasets remain.

## Current report editions

- Comprehensive Persian: `results/reports/full_persian_20261007/report.html`;
  PDF: `output/pdf/commitjl_full_report_fa.pdf`.
- English manuscript-table digest: `results/reports/paper_tables_english_20261007/report.html`;
  PDF: `output/pdf/commitjl_paper_tables_en.pdf`.
- Shared scientific exports: `results/server_study_data_20261007/`.
- Report-only delivery: `output/commitjl_two_reports_20261007.zip`.

The earlier combined report was replaced. Its scientific CSVs and figures were
preserved as shared inputs; removing that report does not remove server results.
See `report_replacement_receipt.json` in the shared data folder for cleanup mode.
The four server campaign folders, compact archive/object backup and datasets remain.

## Current report editions

- Comprehensive Persian: `results/reports/full_persian_20261007/report.html`;
  PDF: `output/pdf/commitjl_full_report_fa.pdf`.
- English manuscript-table digest: `results/reports/paper_tables_english_20261007/report.html`;
  PDF: `output/pdf/commitjl_paper_tables_en.pdf`.
- Shared scientific exports: `results/server_study_data_20261007/`.
- Report-only delivery: `output/commitjl_two_reports_20261007.zip`.

The earlier combined report was replaced. Its scientific CSVs and figures were
preserved as shared inputs; removing that report does not remove server results.
See `report_replacement_receipt.json` in the shared data folder for cleanup mode.
The four server campaign folders, compact archive/object backup and datasets remain.

## Current report editions

- Comprehensive Persian: `results/reports/full_persian_20261007/report.html`;
  PDF: `output/pdf/commitjl_full_report_fa.pdf`.
- English manuscript-table digest: `results/reports/paper_tables_english_20261007/report.html`;
  PDF: `output/pdf/commitjl_paper_tables_en.pdf`.
- Shared scientific exports: `results/server_study_data_20261007/`.
- Report-only delivery: `output/commitjl_two_reports_20261007.zip`.

The earlier combined report was replaced. Its scientific CSVs and figures were
preserved as shared inputs; removing that report does not remove server results.
See `report_replacement_receipt.json` in the shared data folder for cleanup mode.
The four server campaign folders, compact archive/object backup and datasets remain.

## Current report editions

- Comprehensive Persian: `results/reports/full_persian_20261007/report.html`;
  PDF: `output/pdf/commitjl_full_report_fa.pdf`.
- English manuscript-table digest: `results/reports/paper_tables_english_20261007/report.html`;
  PDF: `output/pdf/commitjl_paper_tables_en.pdf`.
- Shared scientific exports: `results/server_study_data_20261007/`.
- Report-only delivery: `output/commitjl_two_reports_20261007.zip`.

The earlier combined report was replaced. Its scientific CSVs and figures were
preserved as shared inputs; removing that report does not remove server results.
See `report_replacement_receipt.json` in the shared data folder for cleanup mode.
The four server campaign folders, compact archive/object backup and datasets remain.
