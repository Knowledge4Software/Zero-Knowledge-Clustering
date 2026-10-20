"""Dataset loaders for the CommitJL experiment suite (manuscript Table
datasets-a): synthetic Gaussian mixtures, a DBSCAN margin generator, and
real datasets (MNIST, Fashion-MNIST, Gisette, Arcene, 20 Newsgroups TF-IDF,
and an optional sentence-embedding corpus). All real-data loaders cache
under `COMMITJL_DATA` (default: `<code>/.cache/commitjl_data`) and produce a
SHA256 manifest entry so results are traceable to a fixed data hash.

Every loader returns a dict with at least: X (np.ndarray, n x d, float64),
optionally y (labels), and `meta` (name, n, d, k, source, sha256).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np


DBSCAN_MARGIN_BENCHMARK = {
    "kwargs": {"n": 1000, "d": 128, "rho": 1.0, "eps": 0.15,
               "frac_near_band": 0.5, "seed": 0},
    "sha256": "853ac376614a4f8fcc672358f165c9e1b7e04955a3d019f56813d7949dd2cf05",
}


def data_home() -> Path:
    p = Path(os.environ.get("COMMITJL_DATA", Path(__file__).resolve().parent.parent / ".cache" / "commitjl_data"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sha256_array(X: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(memoryview(np.ascontiguousarray(X)).cast("B"))
    return h.hexdigest()


def _manifest_path() -> Path:
    return data_home() / "manifest.json"


def write_manifest_entry(name: str, meta: Dict) -> None:
    """Append/update one dataset's entry in the SHA256 manifest
    (`data_home()/manifest.json`)."""
    path = _manifest_path()
    manifest = {}
    if path.exists():
        manifest = json.loads(path.read_text())
    manifest[name] = {**meta, "recorded_at": time.time()}
    # Preserve each dataset/sample separately; the summary's latest-name entry
    # may change when another grid seed or sample size is loaded.
    entries = path.parent / "manifest_entries"
    entries.mkdir(exist_ok=True)
    entry = entries / f"{name}_{meta['sha256']}.json"
    for target, content in ((entry, meta), (path, manifest)):
        fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(content, stream, indent=2, sort_keys=True)
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


# --------------------------------------------------------------------------
# Synthetic generators
# --------------------------------------------------------------------------

def gaussian_mixture(n: int, d: int, k: int, gamma: float = 3.0, imbalance: float = 1.0,
                      seed: int = 0) -> Dict:
    """k-component isotropic Gaussian mixture with center separation
    controlled by `gamma` (mean center norm is gamma * sqrt(d) * sigma;
    this does not guarantee a minimum pairwise separation)
    and cluster-size imbalance controlled by `imbalance` (ratio of largest
    to smallest expected cluster size; 1.0 = balanced)."""
    if not 1 <= k <= n or d < 1 or gamma <= 0 or imbalance < 1:
        raise ValueError("require n >= k >= 1, d >= 1, gamma > 0 and imbalance >= 1")
    rng = np.random.default_rng(seed)
    sigma = 1.0
    # place centers on a simplex-like well-separated lattice, scaled by gamma
    centers = rng.normal(size=(k, d))
    centers *= (gamma * sigma * np.sqrt(d)) / max(np.linalg.norm(centers, axis=1).mean(), 1e-9)
    if imbalance > 1.0:
        weights = np.geomspace(1.0, 1.0 / imbalance, k)
    else:
        weights = np.ones(k)
    weights = weights / weights.sum()
    counts = np.maximum((weights * n).astype(int), 1)
    counts[-1] += n - counts.sum()
    if counts[-1] < 1:
        deficit = 1 - counts[-1]
        counts[-1] = 1
        for j in np.argsort(-counts):
            take = min(deficit, counts[j] - 1)
            counts[j] -= take
            deficit -= take
            if deficit == 0:
                break
    X_parts, y_parts = [], []
    for j in range(k):
        X_parts.append(centers[j] + sigma * rng.normal(size=(counts[j], d)))
        y_parts.append(np.full(counts[j], j))
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    meta = {"name": "gaussian_mixture", "n": n, "d": d, "k": k, "gamma": gamma,
            "imbalance": imbalance, "seed": seed, "source": "synthetic",
            "sha256": _sha256_array(X)}
    return {"X": X, "y": y, "centers": centers, "meta": meta}


def dbscan_margin_dataset(n: int, d: int, rho: float = 1.0, eps: float = 0.1,
                           frac_near_band: float = 0.5, seed: int = 0) -> Dict:
    """Generator with pairwise distances deliberately placed around the
    DBSCAN neighbor radius rho: a `frac_near_band` fraction of points are
    placed in pairs at distance ~rho*(1 +- small offsets spanning the
    ambiguity band [rho^2/(1+eps), rho^2/(1-eps)]), the remainder scattered
    uniformly at large/small distances (unambiguous)."""
    if n < 1 or d < 1 or rho <= 0 or not 0 <= eps < 1 or not 0 <= frac_near_band <= 1:
        raise ValueError("invalid DBSCAN generator dimensions, radius, epsilon or band fraction")
    rng = np.random.default_rng(seed)
    n_pairs = int((n // 2) * frac_near_band)
    pts = []
    base = rng.normal(size=(max(n_pairs, 1), d))
    band_lo = rho / np.sqrt(1 + eps)
    band_hi = rho / np.sqrt(max(1 - eps, 1e-6))
    for i in range(n_pairs):
        anchor = base[i]
        direction = rng.normal(size=d)
        direction /= np.linalg.norm(direction) + 1e-12
        r = rng.uniform(band_lo, band_hi)
        pts.append(anchor)
        pts.append(anchor + direction * r)
    remaining = n - len(pts)
    if remaining > 0:
        scattered = rng.normal(size=(remaining, d)) * (rho * 5.0)
        pts.extend(list(scattered))
    X = np.asarray(pts[:n], dtype=np.float64)
    meta = {"name": "dbscan_margin", "n": n, "d": d, "rho": rho, "eps": eps,
            "frac_near_band": frac_near_band, "seed": seed, "source": "synthetic",
            "generator_version": "dbscan-margin-v1", "sha256": _sha256_array(X)}
    return {"X": X, "meta": meta}


# --------------------------------------------------------------------------
# Real datasets
# --------------------------------------------------------------------------

def _fetch_openml(name: str, version=1, **kw):
    from sklearn.datasets import fetch_openml
    return fetch_openml(name, version=version, data_home=str(data_home()), as_frame=False, **kw)


def load_mnist(n_max: Optional[int] = None, seed: int = 0) -> Dict:
    ds = _fetch_openml("mnist_784")
    X, y = ds.data.astype(np.float64), ds.target.astype(int)
    return _finish("mnist", X, y, n_max, seed, source="openml:mnist_784")


def load_fashion_mnist(n_max: Optional[int] = None, seed: int = 0) -> Dict:
    ds = _fetch_openml("Fashion-MNIST")
    X, y = ds.data.astype(np.float64), ds.target.astype(int)
    return _finish("fashion_mnist", X, y, n_max, seed, source="openml:Fashion-MNIST")


_UCI_GISETTE = "https://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/"
_UCI_ARCENE = "https://archive.ics.uci.edu/ml/machine-learning-databases/arcene/"


def _try_uci_download(base_url: str, files, cache_subdir: str) -> Optional[Path]:
    import urllib.request
    outdir = data_home() / cache_subdir
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        for fn in files:
            dest = outdir / fn
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".download")
                os.close(fd)
                try:
                    urllib.request.urlretrieve(base_url + fn, tmp)
                    os.replace(tmp, dest)
                finally:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
        return outdir
    except Exception:
        return None


def load_gisette(n_max: Optional[int] = None, seed: int = 0) -> Dict:
    outdir = _try_uci_download(_UCI_GISETTE, ["gisette_train.data", "gisette_train.labels"], "gisette")
    if outdir is not None:
        X = np.loadtxt(outdir / "gisette_train.data")
        y = np.loadtxt(outdir / "gisette_train.labels").astype(int)
        if X.shape != (6000, 5000) or y.shape != (6000,):
            raise ValueError("incomplete Gisette training cache: expected 6000 x 5000 with 6000 labels")
        return _finish("gisette", X, y, n_max, seed, source="uci")
    ds = _fetch_openml("gisette")
    X, y = ds.data.astype(np.float64), ds.target.astype(str)
    y = np.array([1 if v in ("1", "1.0") else 0 for v in y])
    return _finish("gisette", X, y, n_max, seed, source="openml:gisette(fallback)")


def load_arcene(n_max: Optional[int] = None, seed: int = 0) -> Dict:
    outdir = _try_uci_download(_UCI_ARCENE, ["ARCENE/arcene_train.data", "ARCENE/arcene_train.labels"], "arcene")
    if outdir is not None:
        try:
            X = np.loadtxt(outdir / "ARCENE" / "arcene_train.data")
            y = np.loadtxt(outdir / "ARCENE" / "arcene_train.labels").astype(int)
            if X.shape != (100, 10000) or y.shape != (100,):
                raise ValueError("incomplete Arcene training cache: expected 100 x 10000 with 100 labels")
            return _finish("arcene", X, y, n_max, seed, source="uci")
        except Exception:
            pass
    ds = _fetch_openml("arcene")
    X, y = ds.data.astype(np.float64), ds.target.astype(str)
    y = np.array([1 if v in ("1", "1.0") else 0 for v in y])
    return _finish("arcene", X, y, n_max, seed, source="openml:arcene(fallback)")


def load_20ng_tfidf(n_max: Optional[int] = None, max_features: int = 2000, seed: int = 0) -> Dict:
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    raw = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"),
                              data_home=str(data_home()))
    vec = TfidfVectorizer(max_features=max_features, stop_words="english")
    Xs = vec.fit_transform(raw.data)
    X = np.asarray(Xs.todense(), dtype=np.float64)
    y = np.asarray(raw.target)
    return _finish("20ng_tfidf", X, y, n_max, seed, source="sklearn:fetch_20newsgroups+tfidf")


def load_embedding_corpus(n_max: Optional[int] = None, seed: int = 0) -> Optional[Dict]:
    """Sentence-transformer embeddings of the 20NG corpus (768-d), skipped
    (returns None) if `sentence-transformers` is not installed."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    from sklearn.datasets import fetch_20newsgroups
    raw = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"),
                              data_home=str(data_home()))
    texts = raw.data[: (n_max or 2000)]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    X = np.asarray(model.encode(texts), dtype=np.float64)
    y = np.asarray(raw.target[: len(texts)])
    return _finish("embedding_corpus_20ng", X, y, None, seed, source="sentence-transformers:all-MiniLM-L6-v2")


def _load_cached_embedding(name: str, n_max: Optional[int], seed: int) -> Dict:
    """Neural features precomputed by `python -m data.build_embeddings <name>`."""
    path = data_home() / "embeddings" / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python -m data.build_embeddings {name}` "
                                "in an environment with torch/sentence-transformers")
    with np.load(path) as z:
        X, y, meta = z["X"].astype(np.float64), z["y"].astype(int), json.loads(str(z["meta"]))
    return _finish(name, X, y, n_max, seed, source=f"embedding:{meta['model']}")


def load_cifar10_resnet50(n_max: Optional[int] = None, seed: int = 0) -> Dict:
    return _load_cached_embedding("cifar10_resnet50", n_max, seed)


def load_20ng_minilm(n_max: Optional[int] = None, seed: int = 0) -> Dict:
    return _load_cached_embedding("20ng_minilm", n_max, seed)


def _finish(name: str, X: np.ndarray, y: Optional[np.ndarray], n_max: Optional[int], seed: int,
            source: str) -> Dict:
    if X.ndim != 2 or min(X.shape) < 1 or not np.isfinite(X).all():
        raise ValueError(f"{name}: expected a nonempty finite feature matrix")
    if y is not None and y.shape != (len(X),):
        raise ValueError(f"{name}: feature and label counts do not match")
    if n_max is not None and n_max < 1:
        raise ValueError("n_max must be positive")
    source_n, source_d = X.shape
    if n_max is not None and n_max < X.shape[0]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=n_max, replace=False)
        X = X[idx]
        if y is not None:
            y = y[idx]
    meta = {"name": name, "n": int(X.shape[0]), "d": int(X.shape[1]),
            "k": int(len(np.unique(y))) if y is not None else None,
            "source": source, "seed": seed, "sha256": _sha256_array(X),
            "source_n": source_n, "source_d": source_d,
            "labels_sha256": _sha256_array(y) if y is not None else None}
    write_manifest_entry(name, meta)
    return {"X": X, "y": y, "meta": meta}


LOADERS = {
    "gaussian_mixture": gaussian_mixture,
    "dbscan_margin": dbscan_margin_dataset,
    "mnist": load_mnist,
    "fashion_mnist": load_fashion_mnist,
    "gisette": load_gisette,
    "arcene": load_arcene,
    "20ng_tfidf": load_20ng_tfidf,
    "embedding_corpus": load_embedding_corpus,
    "cifar10_resnet50": load_cifar10_resnet50,
    "20ng_minilm": load_20ng_minilm,
}


def load_dataset(name: str, **kwargs) -> Optional[Dict]:
    if name not in LOADERS:
        raise KeyError(f"unknown dataset loader {name!r}; known: {sorted(LOADERS)}")
    return LOADERS[name](**kwargs)
