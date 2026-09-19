import numpy as np
import pytest

from commitjl.quantize import (
    make_quant_params, quantize, dequantize, quantize_error_bound,
    scale_for_precision, QuantizationParams,
)


def test_roundtrip_within_error_bound():
    S = scale_for_precision(16)
    d = 8
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, size=d)
    xi = quantize(x, S, B=S * 2)
    x_hat = dequantize(xi, S)
    err = np.linalg.norm(x - x_hat)
    bound = quantize_error_bound(d, S)
    assert err <= bound + 1e-12


def test_p_greater_than_2M_invariant_holds_for_small_params():
    qp = make_quant_params(precision_bits=8, B=256, n=5, d=4, m=8)
    assert qp.p > 2 * qp.M


def test_p_greater_than_2M_invariant_violated_raises():
    # deliberately huge B/n/d/m to blow up M past the field size
    huge = QuantizationParams(S=2**24, B=2**60, m=2**40, d=2**40, n=2**40)
    with pytest.raises(AssertionError):
        huge.check_no_wraparound()


def test_overflow_caught_for_out_of_range_values():
    S = scale_for_precision(8)
    B = 10
    x = np.array([100.0])  # way beyond bound after scaling
    with pytest.raises(OverflowError):
        quantize(x, S, B)
