"""Statistical and methodological regressions for paper campaign execution."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from commitjl.quantize import scale_for_precision
from experiments.common import ResultWriter
from experiments.rq5_precision import measure_precision_row
from experiments.statistics import mean_std_ci95, summarize_repetitions
from experiments.select_proof_shapes import clean_json_value


def test_t_confidence_interval_uses_ten_seed_sample_standard_deviation():
    values = np.arange(10, dtype=float)
    stats = mean_std_ci95(values)
    assert stats["n"] == 10
    assert stats["mean"] == pytest.approx(4.5)
    assert stats["std"] == pytest.approx(np.std(values, ddof=1))
    assert stats["ci95_low"] < stats["mean"] < stats["ci95_high"]
    assert stats["ci95_high"] - stats["mean"] == pytest.approx(stats["mean"] - stats["ci95_low"])


def test_repetition_summary_keeps_configuration_axes_separate():
    rows = [{"dataset_name": "a", "m": m, "q": 2, "precision_bits": 16,
             "seed": seed, "ari": float(m + seed), "delta_q95": float(seed) / 10,
             "r_phi": 1 + float(seed) / 100, "_status": "ok"}
            for m in (4, 8) for seed in range(10)]
    summary = summarize_repetitions(pd.DataFrame(rows), ("dataset_name", "m", "q", "precision_bits"))
    assert len(summary) == 2
    assert set(summary.repetitions) == {10}
    assert set(summary.ari_mean) == {8.5, 12.5}
    assert summary["delta_q95_ci95_low"].notna().all()


def test_precision_sweep_holds_sparse_q_and_constraint_estimate_fixed():
    row = {"dataset": {"loader": "gaussian_mixture", "kwargs": {
        "n": 30, "d": 12, "k": 2, "gamma": 2.0, "imbalance": 1.0, "seed": 12345}},
        "projection": "sparse_jl", "m": 4, "q": 2, "k": 2, "seed": 0,
        "B": 4096, "normalization_reference_bits": 20}
    measured = [measure_precision_row({**row, "precision_bits": bits}, {"pair_set_size": 50})
                for bits in (8, 12, 16, 20)]
    assert [m["quantization_scale_S"] for m in measured] == [scale_for_precision(b) for b in (8, 12, 16, 20)]
    assert {m["q"] for m in measured} == {2}
    assert len({m["estimated_constraints"] for m in measured}) == 1
    assert measured[0]["mean_abs_quantization_error"] > measured[-1]["mean_abs_quantization_error"]
    assert all(np.isfinite(m["ari"]) and np.isfinite(m["delta_q95"]) and np.isfinite(m["r_phi"]) for m in measured)


def test_batched_writer_persists_on_close_and_resumes(tmp_path):
    path = tmp_path / "batched.parquet"
    writer = ResultWriter(path, flush_every=3)
    writer.write({"_hash": "one", "_status": "ok", "ari": .5})
    writer.write({"_hash": "two", "_status": "ok", "ari": .7})
    assert not path.exists()
    writer.close()
    assert len(pd.read_parquet(path)) == 2
    again = ResultWriter(path, flush_every=3)
    assert again.should_skip("one") and again.should_skip("two")
    again.close()


def test_selected_proof_dataset_restores_integer_loader_arguments():
    encoded = {"loader": "gaussian_mixture", "kwargs": {"n": 2000.0, "d": 256.0,
               "gamma": 3.0, "eps": .15, "unused": None}}
    cleaned = clean_json_value(encoded)
    assert cleaned["kwargs"] == {"n": 2000, "d": 256, "gamma": 3, "eps": .15}
