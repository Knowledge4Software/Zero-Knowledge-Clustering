"""Check moved research artifacts, source provenance, and active document links."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

try:
    from .audit import CODE, PAPER_RESULT_PATHS
except ImportError:
    from audit import CODE, PAPER_RESULT_PATHS
from experiments.common import source_fingerprint


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    results = CODE / "results"
    source = source_fingerprint()
    problems = []
    files = []
    expected_root_files = {"README.md", "Makefile", "pyproject.toml", "requirements.txt"}
    actual_root_files = {path.name for path in CODE.iterdir() if path.is_file()}
    if actual_root_files != expected_root_files:
        problems.append(f"Unexpected code-root files: {sorted(actual_root_files ^ expected_root_files)}")
    validation_files = {path.name for path in (CODE / "validation").iterdir() if path.is_file()}
    if validation_files != {"README.md", "audit.py", "check_layout.py"}:
        problems.append(f"Unexpected validation-root files: {sorted(validation_files)}")
    for relative in ("campaigns/stages.ps1", "campaigns/native.ps1",
                     "campaigns/native_parallel.ps1", "campaigns/supplements.ps1",
                     "analysis/paper_results.py", "analysis/author_overview.py",
                     "notebooks/kaggle/commitjl.ipynb", "env_capture/__init__.py"):
        if not (CODE / relative).is_file():
            problems.append(f"Missing organized entry point: {relative}")
    engine = CODE / "campaigns/supplements/engine.py"
    engine_sha256 = sha256(engine) if engine.exists() else None
    if engine_sha256 != "6db5d96235adf22c6cfac4c4653ffd39f50d49536db940dc33a19c4794597768":
        problems.append("Supplemental engine no longer matches the original checkpoint implementation")
    native = {"rq1_fidelity.parquet", "rq3_overhead.parquet", "rq3_grinding.parquet",
              "rq4_certificates.parquet", "rq5_scaling.parquet"}
    supplement = {"rq3_paired.parquet", "rq4_real_certificates.parquet",
                  "rq5_gamma.parquet", "rq5_precision_24_reference.parquet"}
    for filename, relative in PAPER_RESULT_PATHS.items():
        path = results / relative
        if not path.exists():
            problems.append(f"Missing result: {relative}")
            continue
        data = pd.read_parquet(path)
        entry = {
            "file": relative,
            "sha256": sha256(path),
            "rows": len(data),
            "statuses": data["_status"].value_counts().to_dict(),
            "source_matches": bool(data["_source_sha256"].eq(source).all()),
            "duplicate_row_hashes": int(data["_hash"].duplicated().sum()),
        }
        if not entry["source_matches"] or entry["duplicate_row_hashes"]:
            problems.append(f"Invalid provenance or duplicate rows: {relative}")
        bundle = ("paper_native_parallel_v1" if filename in native else
                  "paper_supplement_v1" if filename in supplement else None)
        if bundle:
            archived = results / "archive" / bundle / filename
            matches = archived.exists() and entry["sha256"] == sha256(archived)
            entry["archive"] = archived.relative_to(results).as_posix()
            entry["archive_bytes_match"] = matches
            if not matches:
                problems.append(f"Archived bytes differ or are missing: {relative}")
        files.append(entry)

    documents = [CODE.parent / "README.md", CODE / "README.md", results / "README.md",
                 CODE / "validation/README.md",
                 CODE / "validation/archive/review_20260928/READINESS_REVIEW.md"]
    documents += list((CODE / "validation/reports").glob("*.md"))
    documents += list((CODE / "validation/archive").glob("*.md"))
    documents += list((CODE / "docs").glob("*.md"))
    for folder in ("campaigns", "analysis", "notebooks"):
        documents += list((CODE / folder).glob("*.md"))
    documents += list(results.glob("rq?/README.md"))
    documents += list((results / "overview").glob("*.md"))
    documents += list((results / "overview/analysis").glob("*.md"))
    links_checked = 0
    for document in documents:
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)",
                                 document.read_text(encoding="utf-8")):
            if "://" in target or target.startswith(("#", "mailto:")):
                continue
            target = target.strip("<>").split("#", 1)[0]
            if not target:
                continue
            links_checked += 1
            if not (document.parent / target).exists():
                problems.append(f"Broken link in {document.relative_to(CODE.parent)}: {target}")

    report = {
        "checked_at": datetime.now(timezone(timedelta(hours=3, minutes=30))).isoformat(),
        "status": "PASS" if not problems else "FAIL",
        "source_sha256": source,
        "code_root_files": sorted(actual_root_files),
        "supplement_engine_sha256": engine_sha256,
        "raw_files_checked": len(files),
        "documents_checked": len(documents),
        "local_links_checked": links_checked,
        "raw_files": files,
        "problems": problems,
        "scope": "Layout and byte preservation; proof failures remain recorded outcomes.",
    }
    output = CODE / "validation/outputs/layout_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{report['status']}: {len(files)} raw files, {links_checked} local links; {output}")
    for problem in problems:
        print(problem)
    raise SystemExit(bool(problems))


if __name__ == "__main__":
    main()
