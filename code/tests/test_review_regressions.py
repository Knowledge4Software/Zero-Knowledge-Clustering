"""Independent regressions discovered during the experiment-readiness review."""
import numpy as np
import pandas as pd
import pytest

from commitjl.clustering import nearest_centroid_assign, psi_hat
from commitjl.certificates import centroid_margin_certificate
from commitjl.metrics import sample_pair_set, delta_uv
from commitjl.quantize import quantize
from commitjl.protocol import run_protocol
from commitjl.zk_bridge import build_sparse_instance_from_transcript, res_ok
from experiments.common import ResultWriter
from experiments.cache import ByteLRU
from experiments.pipeline import PipelineCache, run_native, load_row_dataset
from experiments.rq2_proof_cost import build_zk_instance
from experiments.rq4_certificates import run_dbscan_cert


def row(variant="sparse"):
    return {"dataset": {"loader": "gaussian_mixture", "kwargs": {"n": 12, "d": 8, "k": 2}},
            "seed": 3, "m": 4, "q": 2, "k": 2, "precision_bits": 8,
            "projection": "sparse_jl", "variant": variant}


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 1e30, -1e30, 2**63])
def test_quantization_rejects_before_cast(value):
    with pytest.raises(OverflowError):
        quantize(np.array([value]), 1, 4096)


def test_python_integer_and_uint64_distances_are_exact():
    for dtype in (object, np.uint64):
        X = np.array([[2**63 + 10]], dtype=dtype)
        C = np.array([[2**63 + 9], [2**63 + 11]], dtype=dtype)
        assert nearest_centroid_assign(X, C).tolist() == [0]
        assert psi_hat(X, C, np.array([0])) == 1


def test_chunked_assignment_matches_independent_integer_oracle(monkeypatch):
    monkeypatch.setattr("commitjl.clustering.DISTANCE_WORKSPACE_BYTES", 256)
    rng = np.random.default_rng(13)
    X = rng.integers(-4096, 4097, (17, 1024))
    C = rng.integers(-4096, 4097, (4, 1024))
    expected = [min(range(len(C)), key=lambda j: sum((int(x)-int(c))**2 for x,c in zip(point,C[j]))) for point in X]
    assert nearest_centroid_assign(X,C).tolist() == expected


def test_coincident_centroids_are_not_margin_certified():
    assert not centroid_margin_certificate(np.zeros((1,2)), np.zeros((2,2)), 0.15)[0]


def test_pair_sampling_including_diagonal_has_exact_capacity():
    for n in (0,1,5):
        assert sample_pair_set(n,100,7,exclude_identical=False) == [(i,j) for i in range(n) for j in range(i,n)]
        assert len(sample_pair_set(n,100,7)) == n*(n-1)//2


def test_lru_does_not_retain_large_arrays():
    cache = ByteLRU(1024)
    cache["a"] = np.zeros(60)
    cache["b"] = np.zeros(60)
    assert cache.bytes_used <= cache.max_bytes
    assert "a" not in cache and "b" in cache
    cache["oversized"] = np.zeros(1024)
    assert "oversized" not in cache


def test_tiny_cache_still_returns_uncached_values_and_identity():
    cache = PipelineCache(max_bytes=1)
    result = run_native({**row(), "projection":"identity"}, {"pair_set_size":10}, cache)
    assert result["m"] == 8
    assert result["ari"] == 1
    assert result["delta_max"] == 0


def test_native_pipeline_reports_ground_truth_and_fidelity_ari_separately():
    result = run_native(row(), {"pair_set_size": 10})
    assert -1 <= result["ari"] <= 1
    assert -1 <= result["ari_ground_truth_full"] <= 1
    assert -1 <= result["ari_ground_truth_projected"] <= 1


def test_rq4_uses_requested_dataset_instead_of_synthetic_substitution():
    result = run_dbscan_cert({**row(),"projection":"identity"}, {"eps":0.15})
    assert result["dataset_sha256"] == load_row_dataset(row())["meta"]["sha256"]
    assert result["pairs_total"] == 66
    assert result["violation_rate"] == 0
    assert result["m"] == 8


@pytest.mark.parametrize("variant", ["sparse","dense","coord","pca-public","pisot","prepublished","full"])
def test_proof_metrics_measure_the_exact_witness(variant):
    inst, *_ = build_zk_instance(row(variant), {"pair_set_size":66})
    from commitjl.hashing import BN254_FR
    signed = lambda a: np.array([[int(v) if int(v)<BN254_FR//2 else int(v)-BN254_FR for v in r] for r in a],dtype=np.int64)
    X, C = signed(inst.X), signed(inst.centroids)
    if inst.sparse_records is not None:
        A = np.zeros((inst.m,inst.d),dtype=np.int64)
        for r in inst.sparse_records:
            A[r.row,r.t] = r.sign
    else:
        A = inst.dense_A if inst.dense_A is not None else inst.constant_A
    Z, Cz = X @ A.T, C @ A.T
    assert np.array_equal(nearest_centroid_assign(Z,Cz), inst.assign)
    assert int(inst.benchmark_metrics["psi_hat_exact"]) == psi_hat(Z,Cz,np.array(inst.assign))
    pairs = [(i,j) for i in range(len(X)) for j in range(i+1,len(X))]
    delta = delta_uv(X,Z*inst.benchmark_metrics["projection_normalization"],pairs)
    assert inst.benchmark_metrics["delta_max"] == pytest.approx(np.max(np.abs(delta)))
    assert inst.benchmark_metrics["m"] == inst.m


def test_bridge_rejects_consistent_matrix_with_wrong_transcript_challenge():
    X,C = np.array([[1,2],[3,4]]), np.array([[0,0]])
    result = run_protocol(X,C,m=1,q=1,scale=1)
    with pytest.raises(ValueError,match="commitment/challenge"):
        build_sparse_instance_from_transcript(X,C, seed=result["seed"],rho=b"wrong"*8,
            records=result["records"],A_dense=result["A_dense"],assignments=result["assignments"],
            psi_hat=result["Psi_hat"],c_x=result["C_X"],r_x=result["r_X"],c_theta=result["C_theta"],
            c_a=result["C_A"],q=1,m=1)


def test_bridge_success_requires_verification():
    assert not res_ok(0, {"verify_ok":False})
    assert not res_ok(0, {})
    assert res_ok(0, {"verify_ok":True})


def test_external_challenge_cannot_be_replayed():
    from commitjl.randomness import TaskLog
    log = TaskLog(Q=2)
    X,C = np.array([[1,2]]), np.array([[0,0]])
    challenge = lambda: b"x"*32
    run_protocol(X,C,m=1,q=1,scale=1,task_log=log,challenge_fn=challenge)
    with pytest.raises(ValueError,match="replay"):
        run_protocol(X,C,m=1,q=1,scale=1,task_log=log,challenge_fn=challenge)


def test_memory_watchdog_stops_over_budget_process():
    import subprocess
    import sys
    from commitjl.zk_bridge import _psutil_peak_rss
    process = subprocess.Popen([sys.executable,"-c","import time; x=bytearray(64*1024**2); time.sleep(5)"])
    try:
        result = _psutil_peak_rss(process,max_rss_bytes=32*1024**2)
        assert result.get("memory_limit_exceeded") == 1
    finally:
        process.kill()
        process.wait()


def test_writer_retries_failures_without_duplicate_rows(tmp_path):
    path=tmp_path/"results.parquet"
    writer=ResultWriter(path)
    writer.write({"_hash":"one","_status":"failed","value":1})
    assert not writer.should_skip("one")
    writer.write({"_hash":"one","_status":"ok","value":2})
    assert writer.should_skip("one")
    writer.close()
    df=pd.read_parquet(path)
    assert len(df)==1 and df.iloc[0]["value"]==2


def test_native_and_proof_metrics_share_a_parquet_schema(tmp_path):
    instance, *_ = build_zk_instance(row(), {"pair_set_size":10})
    native = run_native(row(), {"pair_set_size":10})
    writer = ResultWriter(tmp_path / "mixed.parquet")
    writer.write({**instance.benchmark_metrics,"_hash":"proof","_status":"ok"})
    writer.write({**native,"_hash":"native","_status":"native_only"})
    writer.close()
    df = pd.read_parquet(tmp_path / "mixed.parquet")
    assert len(df)==2 and pd.api.types.is_numeric_dtype(df.psi_hat)
    assert df.iloc[0].psi_hat_exact == str(instance.psi_hat)


def test_writer_flush_failure_does_not_mark_complete(tmp_path,monkeypatch):
    writer=ResultWriter(tmp_path/"results.parquet")
    def fail(): raise OSError("disk failure")
    monkeypatch.setattr(writer,"flush",fail)
    with pytest.raises(OSError): writer.write({"_hash":"one","_status":"ok"})
    assert not writer.should_skip("one") and not writer.rows
    writer.close()


def test_writer_rejects_corrupt_legacy_and_concurrent_output(tmp_path):
    path=tmp_path/"results.parquet"
    path.write_bytes(b"incomplete parquet")
    with pytest.raises(Exception): ResultWriter(path)
    assert path.read_bytes()==b"incomplete parquet"
    pd.DataFrame([{"_hash":"old","_status":"ok"}]).to_parquet(path)
    with pytest.raises(ValueError,match="fresh --out"): ResultWriter(path)
    writer=ResultWriter(tmp_path/"fresh.parquet")
    with pytest.raises(RuntimeError,match="another writer"): ResultWriter(writer.path)
    writer.close()
