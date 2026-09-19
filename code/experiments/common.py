"""Shared experiment-runner infrastructure (plan P4): YAML config loading
and grid expansion, a resumable parquet row writer keyed by a config hash,
`--shard i/N` splitting, and lightweight environment capture.

Config file shape (see experiments/configs/*.yaml):
    fixed: {key: value, ...}          # constant across all rows
    grid:  {key: [v1, v2, ...], ...}  # cartesian product axes
Each row = {**fixed, **one combination of grid}.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("fixed", {})
    cfg.setdefault("grid", {})
    return cfg


def load_prereg(path: Optional[str] = None) -> Dict[str, Any]:
    path = path or str(REPO_ROOT / "experiments" / "configs" / "prereg.yaml")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def generate_grid(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Cartesian product of cfg['grid'] axes, each row merged with cfg['fixed']."""
    fixed = cfg.get("fixed", {})
    grid = cfg.get("grid", {})
    if not grid:
        return [dict(fixed)]
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    rows = []
    for combo in combos:
        row = dict(fixed)
        row.update(dict(zip(keys, combo)))
        rows.append(row)
    return rows


def row_hash(row: Dict[str, Any], extra_keys: Optional[Iterable[str]] = None) -> str:
    """Stable hash identifying a config row, used as the resumability key."""
    keys = sorted(row.keys()) if extra_keys is None else sorted(extra_keys)
    blob = json.dumps({k: row.get(k) for k in keys}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def apply_shard(rows: List[Dict[str, Any]], shard: Optional[str]) -> List[Dict[str, Any]]:
    """`shard` is "i/N" (1-indexed i); returns the i-th of N contiguous-ish
    (round-robin) slices so sharded runs stay load-balanced."""
    if not shard:
        return rows
    i_s, n_s = shard.split("/")
    i, n = int(i_s), int(n_s)
    assert 1 <= i <= n
    return [r for idx, r in enumerate(rows) if (idx % n) == (i - 1)]


class ResultWriter:
    """Resumable parquet row writer keyed by row_hash. `open()` loads any
    existing file's hashes; `should_skip(h)` lets a runner skip completed
    rows; `write(row)` appends one row and flushes to disk incrementally
    (rewrites the file; fine at the row counts used here)."""

    def __init__(self, path: str, key_field: str = "_hash"):
        self.path = Path(path)
        self.key_field = key_field
        self.rows: List[Dict[str, Any]] = []
        self._keys = set()
        self._load()

    def _load(self):
        if self.path.exists() and pd is not None:
            try:
                df = pd.read_parquet(self.path)
                self.rows = df.to_dict(orient="records")
                self._keys = {r.get(self.key_field) for r in self.rows}
            except Exception:
                self.rows, self._keys = [], set()

    def should_skip(self, h: str) -> bool:
        return h in self._keys

    def write(self, row: Dict[str, Any]):
        self.rows.append(row)
        self._keys.add(row.get(self.key_field))
        self.flush()

    def flush(self):
        if pd is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.rows)
        tmp = self.path.with_suffix(".tmp.parquet")
        df.to_parquet(tmp, index=False)
        tmp.replace(self.path)


def capture_env() -> Dict[str, Any]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import env_capture  # type: ignore
    return env_capture.capture()


def common_arg_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True, help="path to a YAML grid config")
    p.add_argument("--prereg", default=None, help="path to prereg.yaml (defaults to experiments/configs/prereg.yaml)")
    p.add_argument("--out", default=None, help="output parquet path (default: results/<script>.parquet)")
    p.add_argument("--shard", default=None, help="i/N shard selector, e.g. 2/4")
    p.add_argument("--limit", type=int, default=None, help="cap number of grid rows processed (debug)")
    return p


def default_out_path(script_name: str) -> str:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return str(RESULTS_DIR / f"{script_name}.parquet")


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        return result, time.perf_counter() - t0, None
    except Exception as e:  # pragma: no cover
        return None, time.perf_counter() - t0, str(e)
