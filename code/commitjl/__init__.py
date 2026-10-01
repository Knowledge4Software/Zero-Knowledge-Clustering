"""CommitJL: zero-knowledge clustering reference implementation (P1: core protocol, P2: projections/clustering/certificates/metrics/stats).

This package is a pure-Python, host-side reference implementation matching the
mathematical definitions in manuscript.tex (Sections "Background, model, and
problem statement" through "Geometric and cryptographic guarantees"). It does
NOT implement the ZK circuit itself (that is gnark/Go, out of scope for P1/P2);
it implements the native/host computations, commitments, sparse generator,
protocol orchestration, projections, clustering, certificates, metrics and
statistics that the circuit must match bit-for-bit on the integer path.
"""

__version__ = "0.1.0"

VERSION_TAG = "CommitJL-v1"
