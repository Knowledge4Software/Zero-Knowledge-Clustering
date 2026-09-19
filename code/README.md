# CommitJL

Code that produces every table and figure in `manuscript.tex` (Cluster
Computing submission): commit-then-sample sparse-JL projected clustering
with a ZK proof of correct computation.

## English quickstart

```bash
cd code
python -m venv .venv && .venv/bin/pip install -r requirements.txt -e .
make zk           # builds zk/bin/commitjl-prove (needs Go >= 1.22)
make test         # pytest: protocol, projections, certificates, negative tests
make smoke        # tiny end-to-end run (<3 min): 1 native grid + 1 real ZK proof
make tables figures   # LaTeX tables + PDF/PNG figures from results/*.parquet
```

Full grids (not required for the smoke check, may take hours):

```bash
make native   # Table matrix-a native/symbolic sweep -> results/rq1_fidelity.parquet, rq3/rq4/rq5
make core     # preregistered small proof grid -> results/rq2_proof_cost.parquet, ablation.parquet
```

Each `experiments/rq*.py` script is resumable (skips rows already present in
its output parquet, keyed by a config hash) and shardable:

```bash
.venv/bin/python -m experiments.rq2_proof_cost --config experiments/configs/core.yaml --shard 1/4
```

Dataset cache (MNIST/Fashion-MNIST/Gisette/Arcene/20NG; SHA256 manifest) is
under `COMMITJL_DATA` (default `code/.cache/commitjl_data`).

### Kaggle

Upload `kaggle/commitjl_kaggle.ipynb` to a Kaggle notebook (internet on).
It installs Go, builds the circuit binary, runs `pytest -q` and `make smoke`,
then lets you run selected configs/shards and `make tables figures`, and
zips `results/`.

## راهنمای سریع (فارسی)

```bash
cd code
python -m venv .venv && .venv/bin/pip install -r requirements.txt -e .
make zk        # ساخت باینری اثبات ZK (نیازمند Go نسخه ۱٫۲۲ یا بالاتر)
make test      # اجرای تست‌های واحد
make smoke     # اجرای دود: یک شبکه‌ی کوچک ارزیابی + یک اثبات واقعی ZK (کمتر از سه دقیقه)
make tables figures   # تولید جداول LaTeX و شکل‌های PDF/PNG از نتایج
```

اجرای شبکه‌های کامل (اختیاری، ممکن است ساعت‌ها طول بکشد): `make native` و `make core`.
هر اسکریپت `experiments/rq*.py` قابل ازسرگیری است (سطرهای انجام‌شده را بر اساس هش پیکربندی رد می‌کند)
و با `--shard i/N` قابل تقسیم بین چند نشست Kaggle است.

برای اجرا روی Kaggle، نوت‌بوک `kaggle/commitjl_kaggle.ipynb` را بارگذاری کنید؛ این نوت‌بوک
Go را نصب، باینری مدار را می‌سازد، تست‌ها و اجرای دود را انجام می‌دهد و سپس امکان اجرای
پیکربندی‌های انتخابی و تولید جداول/شکل‌ها را فراهم می‌کند.

## Layout

- `commitjl/` -- core Python package (P1-P3): encoding, hashing, commit,
  sparse_gen, randomness, quantize, protocol, projections, clustering,
  certificates, metrics, stats, cost_model, zk_bridge.
- `zk/` -- Go/gnark circuit (`zk/circuit`) and CLI (`zk/cmd/commitjl-prove`).
- `data/loaders.py` -- dataset loaders + SHA256 manifest (Table datasets-a).
- `experiments/` -- config-driven grid runners (`rq1_fidelity.py` ...
  `ablation.py`), `common.py` (grid/resumable-parquet/shard/env plumbing),
  `pipeline.py` (shared native-clustering helper), `make_tables.py`,
  `make_figures.py`, `summarize.py`.
- `figures/schematics.py` -- conceptual diagrams (protocol, security game,
  generator, dataflow, architecture, workflow, motivation).
- `results/{tables,figures}` -- generated LaTeX/PDF/PNG outputs;
  `results/*.parquet` -- raw per-row experiment results;
  `results/headline.json` -- abstract-ready summary numbers.
- `env_capture.py` -- CPU/RAM/package/git-hash capture attached to every run.
- `kaggle/commitjl_kaggle.ipynb` -- Kaggle packaging (P6).

## Verification

1. `make test` (incl. Python<->Go MiMC golden vectors, Prop. 1 Monte Carlo,
   negative ZK tests).
2. `make smoke`: proof verifies, native Psi_hat == circuit Psi_hat, zero
   range rejections on the smoke config.
3. `make tables figures`: every manuscript table/figure label is produced
   (as LaTeX/PDF, using whatever `results/*.parquet` rows exist; tables/
   figures degrade gracefully to a "no data" placeholder when a grid has
   not been run yet).
