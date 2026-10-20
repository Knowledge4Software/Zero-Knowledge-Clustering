"""Build cached neural-embedding corpora for the native experiments.

Run once in a separate environment that has torch/torchvision and
sentence-transformers (the experiment venv stays numpy/sklearn only):

    python -m data.build_embeddings cifar10_resnet50 20ng_minilm

Outputs `<data_home>/embeddings/<name>.npz` with X (float32), y and a JSON
`meta` string (model, preprocessing, source indices). The loaders in
`data.loaders` read only these files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np

from data.loaders import data_home

CIFAR_N = 10_000      # random subset of the 50,000 CIFAR-10 training images
CIFAR_SEED = 0
BATCH = 64


def _out(name):
    p = data_home() / "embeddings"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{name}.npz"


def _save(name, X, y, meta):
    path = _out(name)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, X=X.astype(np.float32), y=np.asarray(y), meta=json.dumps(meta))
    tmp.replace(path)
    print(name, X.shape, f"{meta['seconds']:.0f}s", path, flush=True)


def build_cifar10_resnet50():
    import torch
    from torchvision import datasets, transforms
    from torchvision.models import resnet50, ResNet50_Weights

    t0 = time.time()
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()          # 2048-d global-average-pooled features
    model.eval()
    tf = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    ds = datasets.CIFAR10(root=str(data_home() / "cifar10"), train=True, download=True, transform=tf)
    idx = np.sort(np.random.default_rng(CIFAR_SEED).choice(len(ds), CIFAR_N, replace=False))
    feats, labels = [], []
    with torch.inference_mode():
        for s in range(0, len(idx), BATCH):
            batch = [ds[int(i)] for i in idx[s:s + BATCH]]
            feats.append(model(torch.stack([b[0] for b in batch])).numpy())
            labels += [b[1] for b in batch]
            if s // BATCH % 20 == 0:
                print(f"cifar {s}/{len(idx)} {time.time() - t0:.0f}s", flush=True)
    meta = dict(model="torchvision resnet50 IMAGENET1K_V2, fc removed (avgpool 2048-d)",
                preprocessing="Resize(224), ImageNet normalization", source="CIFAR-10 train",
                n_source=len(ds), subset_seed=CIFAR_SEED, indices_sha256=hashlib.sha256(idx.astype(np.int64).tobytes()).hexdigest(),
                torch=torch.__version__, seconds=time.time() - t0)
    _save("cifar10_resnet50", np.concatenate(feats), np.asarray(labels), meta)


def build_20ng_minilm():
    from sentence_transformers import SentenceTransformer
    from sklearn.datasets import fetch_20newsgroups

    t0 = time.time()
    raw = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"),
                             data_home=str(data_home()))
    keep = [i for i, t in enumerate(raw.data) if t.strip()]   # empty documents carry no text
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    X = model.encode([raw.data[i] for i in keep], batch_size=BATCH, show_progress_bar=False,
                     convert_to_numpy=True)
    meta = dict(model="sentence-transformers/all-MiniLM-L6-v2 (384-d, mean pooling, normalized)",
                preprocessing="headers/footers/quotes removed; empty documents dropped",
                source="sklearn fetch_20newsgroups train", n_source=len(raw.data), n_kept=len(keep),
                seconds=time.time() - t0)
    _save("20ng_minilm", X, np.asarray(raw.target)[keep], meta)


BUILDERS = {"cifar10_resnet50": build_cifar10_resnet50, "20ng_minilm": build_20ng_minilm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+", choices=sorted(BUILDERS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for name in a.names:
        if _out(name).exists() and not a.force:
            print(name, "exists; skip", flush=True)
            continue
        BUILDERS[name]()


if __name__ == "__main__":
    main()
