"""TaskLog and randomness instantiations.

Manuscript reference: Sec. "Randomness instantiations" and "Threat model and
commit-then-sample requirement". Three challenge sources: verifier nonce,
public randomness beacon (referenced by round, loaded from a recorded
trace), and two-party commit-reveal. A TaskLog enforces: (C_X, C_theta) is
recorded before rho is learned; nonces/beacon rounds are single-use; at most
Q eligible commitments per authenticated prover/task.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

from .hashing import sha256


@dataclass
class TaskLog:
    """Tracks commit-then-sample eligibility for one task.

    deadline: opaque marker (e.g. unix timestamp or round number) after which
        no new commitments are eligible; enforcement is left to the caller
        (this class just stores it and exposes `is_open`).
    Q: per-prover attempt budget (max eligible commitments).
    """
    deadline: Optional[float] = None
    Q: int = 1
    _now: float = 0.0
    _attempts: Dict[str, int] = field(default_factory=dict)
    _used_nonces: Set[bytes] = field(default_factory=set)
    _committed: Dict[str, list] = field(default_factory=dict)  # prover_id -> [(C_X, C_theta), ...]

    def is_open(self, now: Optional[float] = None) -> bool:
        t = self._now if now is None else now
        return self.deadline is None or t <= self.deadline

    def record_commitment(self, prover_id: str, C_X: int, C_theta: int, now: Optional[float] = None) -> None:
        """Record (C_X, C_theta) as an eligible-attempt commitment for prover_id.

        Raises if the task is closed or the prover has exhausted their budget Q.
        """
        if not self.is_open(now):
            raise PermissionError("TaskLog is closed (past deadline); commitment not eligible")
        used = self._attempts.get(prover_id, 0)
        if used >= self.Q:
            raise PermissionError(f"prover {prover_id!r} has exhausted budget Q={self.Q}")
        self._attempts[prover_id] = used + 1
        self._committed.setdefault(prover_id, []).append((C_X, C_theta))

    def attempts_used(self, prover_id: str) -> int:
        return self._attempts.get(prover_id, 0)

    def budget_remaining(self, prover_id: str) -> int:
        return self.Q - self.attempts_used(prover_id)

    def fresh_nonce(self, nbytes: int = 32) -> bytes:
        """Generate and register a fresh single-use nonce rho."""
        while True:
            candidate = os.urandom(nbytes)
            if candidate not in self._used_nonces:
                self._used_nonces.add(candidate)
                return candidate

    def consume_nonce(self, rho: bytes) -> None:
        """Mark an externally supplied nonce as used; raises if already used."""
        if rho in self._used_nonces:
            raise ValueError("nonce rho has already been consumed (replay)")
        self._used_nonces.add(rho)


@dataclass
class BeaconTrace:
    """Loads a public-randomness-beacon trace (round -> 32-byte value) from a
    recorded file, one 'round_index,hex_value' CSV line per round, or an
    in-memory dict for tests."""
    rounds: Dict[int, bytes]
    _used_rounds: Set[int] = field(default_factory=set)

    @classmethod
    def from_file(cls, path: str) -> "BeaconTrace":
        rounds = {}
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                idx_s, hex_s = line.split(",")
                rounds[int(idx_s)] = bytes.fromhex(hex_s)
        return cls(rounds=rounds)

    def get_round(self, round_index: int) -> bytes:
        if round_index in self._used_rounds:
            raise ValueError(f"beacon round {round_index} already used (single-use)")
        if round_index not in self.rounds:
            raise KeyError(f"beacon round {round_index} not present in trace")
        self._used_rounds.add(round_index)
        return self.rounds[round_index]


class CommitRevealChallenge:
    """Two-party commit-reveal coin flip for deriving rho jointly.

    Each party picks a random value; each commits (SHA-256) to their value
    before seeing the other's; after both commitments are recorded, values
    are revealed and checked against their commitments; rho = XOR of both
    revealed values.
    """

    def __init__(self):
        self._commitments: Dict[str, bytes] = {}
        self._values: Dict[str, bytes] = {}

    def commit(self, party: str, value: Optional[bytes] = None) -> bytes:
        value = value if value is not None else os.urandom(32)
        self._values[party] = value
        c = sha256(value)
        self._commitments[party] = c
        return c

    def reveal(self, party: str) -> bytes:
        if party not in self._values:
            raise ValueError(f"party {party!r} has not committed")
        return self._values[party]

    def finalize(self) -> bytes:
        if len(self._commitments) < 2:
            raise ValueError("need commitments from both parties before finalize")
        for party, c in self._commitments.items():
            if sha256(self._values[party]) != c:
                raise ValueError(f"party {party!r} revealed a value not matching its commitment")
        vals = list(self._values.values())
        rho = bytes(a ^ b for a, b in zip(vals[0], vals[1]))
        return rho
