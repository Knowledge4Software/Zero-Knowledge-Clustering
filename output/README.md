# Submitted evidence

Keep these two local submission attachments in this directory:

- zk-audit-evidence-20261118-v2.zip
- zk-audit-supplementary-evidence-final-20261119.zip

The primary release contains 42 primary archives, follow-up archives, checking
scripts, and public verifier examples. The supplemental release includes
additional proof timings, the Digits lifecycle runs, the large-catalog native
check, and the utility analysis. Both ZIP files are kept outside Git.

After extracting the primary ZIP, run the following from its extracted root:

    python evidence/check_primary_archives.py archives

Expect 42 verified archive hashes and 400 recomputed integer objective changes.
Follow evidence/README.md in the primary release for public replay and
evidence/REVIEWER_EVIDENCE_INDEX.md in the supplemental release for the
follow-up results.
