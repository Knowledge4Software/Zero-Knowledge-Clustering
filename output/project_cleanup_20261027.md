# Project cleanup, 27 October 2026

Deleted 228,657 files across 1,175 targets, totaling 61.665 GiB.
Actual deletion was requested by the user; no replacement archive or backup copy was created. Git history was not rewritten.

Removed: superseded raw experiment campaigns, retirement/recovery caches, copied paper versions, old PDF/source deliveries, obsolete reports and binaries, inactive proving keys and compiled/preparation caches.

Retained: current full paper and methodology, supervisor input ZIPs and source material, code/tests, canonical datasets, numerical and negative-result records, public proofs/verification keys, source receipts, and every cache needed by active/queued work.

Validation: 95 protected file hashes; 29 current manuscript dependencies; two current delivery ZIP manifests; 11 public manifests covering 3,811 proof files and 3,873 artifact hashes.
This is a preservation/integrity check, not a new cryptographic replay or security audit.

Research supervisor live at validation: True; phase: proof_har561_ordered_head_base. No proof worker was intentionally paused, restarted or deleted.
Inactive proving caches can be regenerated if a new experiment is later authorized. Their deletion leaves measured resource metadata and existing public proofs intact.

The JSON receipt lists every removed target, its reason, file count and byte count. Python/test caches may be recreated by subsequent imports and tests.

Targeted regression tests: 15 passed, 0 failed.

The filesystem deletion overlapped the HAR Full timing phase. Its proofs/inputs remain valid. The interference ledger marks this comparison; automatic exports retain raw times but exclude its ratio from clean speed conclusions. No replacement proof workload was launched by cleanup.
