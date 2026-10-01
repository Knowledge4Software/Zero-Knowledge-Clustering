"""Quantization and no-wraparound conditions.

Manuscript reference: Sec. "Quantization and no-wraparound conditions".
Real coordinates are scaled by S and rounded to integers with per-coordinate
error <= 1/(2S). Bound B on |x_t|, |c_t|. D_max = 4*m*d^2*B^2,
Psi_max = n*D_max. Range checks use M = max(2dB, 2*D_max, Psi_max) and the
field must satisfy p > 2M.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hashing import BN254_FR

SUPPORTED_PRECISION_BITS = (8, 12, 16, 20, 24)


@dataclass(frozen=True)
class QuantizationParams:
    S: int          # scale (2**precision_bits, or an explicit integer scale)
    B: int          # bound on |x_t|, |c_t| after quantization
    m: int
    d: int
    n: int
    p: int = BN254_FR

    @property
    def D_max(self) -> int:
        return 4 * self.m * (self.d ** 2) * (self.B ** 2)

    @property
    def Psi_max(self) -> int:
        return self.n * self.D_max

    @property
    def M(self) -> int:
        return max(2 * self.d * self.B, 2 * self.D_max, self.Psi_max)

    def check_no_wraparound(self) -> None:
        if self.p <= 2 * self.M:
            raise AssertionError(
                f"no-wraparound violated: field modulus p={self.p} must exceed 2*M={2*self.M} "
                f"(M={self.M}, D_max={self.D_max}, Psi_max={self.Psi_max})"
            )


def scale_for_precision(precision_bits: int) -> int:
    if precision_bits not in SUPPORTED_PRECISION_BITS:
        raise ValueError(f"precision_bits must be one of {SUPPORTED_PRECISION_BITS}, got {precision_bits}")
    return 1 << precision_bits


def make_quant_params(precision_bits: int, B: int, n: int, d: int, m: int, p: int = BN254_FR) -> QuantizationParams:
    S = scale_for_precision(precision_bits)
    qp = QuantizationParams(S=S, B=B, m=m, d=d, n=n, p=p)
    qp.check_no_wraparound()
    return qp


def quantize(x: np.ndarray, S: int, B: int) -> np.ndarray:
    """Round real x by scale S to integers and reject values outside [-B, B].

    Raises OverflowError if any scaled+rounded coordinate exceeds the bound B
    (this is the host-side analogue of the circuit's range check failing).
    """
    x = np.asarray(x, dtype=np.float64)
    if not isinstance(S, (int, np.integer)) or S <= 0 or not 0 <= B < 2**63:
        raise ValueError("S must be a positive integer and B must fit signed int64")
    with np.errstate(over="ignore", invalid="ignore"):
        rounded = np.rint(x * S)
    if not np.all(np.isfinite(rounded)):
        raise OverflowError("quantization requires finite, representable coordinates")
    if np.any(np.abs(rounded) > B) or np.any(rounded >= 2**63) or np.any(rounded < -(2**63)):
        bad = np.abs(rounded).max()
        raise OverflowError(f"quantized value {bad} exceeds bound B={B} (scale S={S})")
    return rounded.astype(np.int64)


def dequantize(xi: np.ndarray, S: int) -> np.ndarray:
    return np.asarray(xi, dtype=np.float64) / S


def quantize_error_bound(d: int, S: int) -> float:
    """Per-vector L2 quantization error bound: ||x - x~/S||_2 <= sqrt(d)/(2S)."""
    return (d ** 0.5) / (2 * S)


def pairwise_distance_perturbation_bound(d: int, S: int) -> float:
    """Distance perturbation bound for a pair u,v: <= sqrt(d)/S."""
    return (d ** 0.5) / S


def squared_distance_error_bound(u_v_norm: float, d: int, S: int) -> float:
    """Squared-distance error bound: <= 2*||u-v||_2*sqrt(d)/S + d/S^2."""
    return 2 * u_v_norm * (d ** 0.5) / S + d / (S ** 2)
