"""Environment capture (plan P6): CPU/RAM, package versions, git hash --
attached to every experiment run for reproducibility."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent


def _git_hash() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
                              capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _pkg_versions() -> Dict[str, str]:
    versions = {}
    for pkg in ("numpy", "scipy", "sklearn", "pandas", "pyarrow", "yaml", "matplotlib", "psutil"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[pkg] = "not_installed"
    return versions


def _cpu_ram() -> Dict[str, Any]:
    info: Dict[str, Any] = {"platform": platform.platform(), "processor": platform.processor(),
                             "python": sys.version}
    try:
        import psutil
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["ram_total_bytes"] = psutil.virtual_memory().total
    except Exception:
        import os
        info["cpu_count_logical"] = os.cpu_count()
    return info


def _go_version() -> str:
    try:
        out = subprocess.run(["go", "version"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else "not_installed"
    except Exception:
        return "not_installed"


def capture() -> Dict[str, Any]:
    return {
        "git_hash": _git_hash(),
        "packages": _pkg_versions(),
        "go_version": _go_version(),
        "system": _cpu_ram(),
    }


if __name__ == "__main__":
    print(json.dumps(capture(), indent=2))
