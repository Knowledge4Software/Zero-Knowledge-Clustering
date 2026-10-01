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
import atexit
import ctypes
import gc
import hashlib
import itertools
import json
import os
import sys
import time
import tempfile
import weakref
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(os.environ.get("COMMITJL_RESULTS_DIR") or REPO_ROOT / "results")


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
    if cfg.get("blocks"):
        rows = []
        for block in cfg["blocks"]:
            block_fixed = {**fixed, **block.get("fixed", {})}
            rows.extend(generate_grid({"fixed": block_fixed, "grid": block.get("grid", {})}))
        return rows
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


def progress(iterable, *, total=None, desc="experiment"):
    """Visible progress with a dependency-free fallback for minimal installs."""
    try:
        from tqdm.auto import tqdm
        # stdout avoids Windows PowerShell 5 converting tqdm's ordinary
        # stderr display into NativeCommandError records.
        return tqdm(iterable, total=total, desc=desc, unit="row",
                    dynamic_ncols=True, file=sys.stdout)
    except ImportError:  # pragma: no cover - tqdm is a declared dependency
        return iterable


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
    if not 1 <= i <= n:
        raise ValueError("shard must satisfy 1 <= i <= N")
    return [r for idx, r in enumerate(rows) if (idx % n) == (i - 1)]


def source_fingerprint():
    """Include dirty local source, dependencies and configurations in provenance."""
    digest = hashlib.sha256()
    for folder in ("commitjl", "experiments", "data", "zk"):
        for path in sorted((REPO_ROOT / folder).rglob("*")):
            if path.is_file() and (path.suffix in {".py", ".go", ".yaml"} or path.name in {"go.mod", "go.sum"}):
                digest.update(path.relative_to(REPO_ROOT).as_posix().encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


class ResultWriter:
    """Resumable parquet row writer keyed by row_hash. `open()` loads any
    existing file's hashes; `should_skip(h)` lets a runner skip completed
    rows; `write(row)` appends one row and flushes to disk incrementally
    (rewrites the file; fine at the row counts used here)."""

    def __init__(self, path: str, key_field: str = "_hash", context=None, flush_every: Optional[int] = None):
        if pd is None:
            raise RuntimeError("pandas and a parquet engine are required to save results")
        self.path = Path(path)
        self.provenance = source_fingerprint()
        self.context_hash = row_hash(context or {})
        self.key_field = key_field
        self.rows: List[Dict[str, Any]] = []
        self._keys = set()
        self._dirty = 0
        self.flush_every = int(flush_every or os.environ.get("COMMITJL_CHECKPOINT_ROWS", "1"))
        if self.flush_every < 1:
            raise ValueError("flush_every must be positive")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = open(str(self.path) + ".lock", "a+b")
        try:
            if os.name == "nt":
                import msvcrt
                if lock.tell() == 0:
                    lock.write(b"0")
                    lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            lock.close()
            raise RuntimeError(f"another writer is using {self.path}; give each shard a separate output") from None
        self._release_lock = weakref.finalize(self, lock.close)
        atexit.register(self.close)
        try:
            self._load()
        except Exception:
            self.close()
            raise

    def close(self):
        if self._dirty:
            self.flush()
        self._release_lock()

    def _load(self):
        if self.path.exists() and pd is not None:
            df = pd.read_parquet(self.path)
            self.rows = df.to_dict(orient="records")
            if any(r.get("_source_sha256") != self.provenance or r.get("_context_hash") != self.context_hash for r in self.rows):
                raise ValueError(f"{self.path} contains results from different or unrecorded code/configuration; use a fresh --out")
            self._keys = {r.get(self.key_field) for r in self.rows if r.get("_status") in {"ok", "native_only"}}

    def should_skip(self, h: str) -> bool:
        return h in self._keys

    def write(self, row: Dict[str, Any]):
        row = {**row, "_source_sha256": self.provenance, "_context_hash": self.context_hash,
               "_written_unix_s": time.time()}
        old_rows = self.rows
        key = row.get(self.key_field)
        self.rows = [r for r in self.rows if r.get(self.key_field) != key] + [row]
        self._dirty += 1
        if self._dirty >= self.flush_every:
            try:
                self.flush()
            except Exception:
                self.rows = old_rows
                self._dirty -= 1
                raise
        if row.get("_status") in {"ok", "native_only"}:
            self._keys.add(key)
        else:
            self._keys.discard(key)

    def flush(self):
        if pd is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.rows)
        fd, tmp_name = tempfile.mkstemp(prefix=self.path.stem + ".", suffix=".tmp.parquet",
                                        dir=str(self.path.parent))
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            df.to_parquet(tmp, index=False)
            for attempt in range(6):
                try:
                    os.replace(tmp, self.path)
                    break
                except PermissionError:
                    if attempt == 5:
                        raise
                    time.sleep(0.05 * (attempt + 1))
        finally:
            if tmp.exists():
                tmp.unlink()
        self._dirty = 0


def release_process_memory() -> None:
    """Collect cyclic garbage and return unused pages to the OS when supported.

    Experiment rows are seed repetitions. Calling this after each row prevents
    temporary NumPy/sklearn allocations from accumulating over long campaigns;
    bounded PipelineCache entries remain live and reusable.
    """
    gc.collect()
    try:
        if os.name == "nt":
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.psapi.EmptyWorkingSet(handle)
        else:
            libc = ctypes.CDLL(None)
            trim = getattr(libc, "malloc_trim", None)
            if trim is not None:
                trim(0)
    except (AttributeError, OSError):
        pass


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


def default_out_path(script_name: str, shard=None) -> str:
    directory = Path(os.environ.get("COMMITJL_RESULTS_DIR") or RESULTS_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    suffix = "_shard_" + shard.replace("/", "_of_") if shard else ""
    return str(directory / f"{script_name}{suffix}.parquet")


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        return result, time.perf_counter() - t0, None
    except Exception as e:  # pragma: no cover
        return None, time.perf_counter() - t0, str(e)
