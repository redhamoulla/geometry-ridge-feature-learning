"""Exact factorization and strict non-scalar orbit examples."""
from __future__ import annotations
import numpy as np
from geometric_register.core import orbit_signature, register_diagnostics


def random_orthogonal(rng: np.random.Generator, p: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(p, p)))
    return q @ np.diag(np.sign(np.diag(r)))


def main() -> None:
    rng = np.random.default_rng(7)
    B = np.diag([0.2, 0.8, 2.0]); b = np.array([1.0, -0.5, 2.0])
    q = random_orthogonal(rng, 3)
    sig1 = orbit_signature(B, b); sig2 = orbit_signature(q @ B @ q.T, q @ b)
    assert np.allclose([(s.eigenvalue, s.effort_energy) for s in sig1], [(s.eigenvalue, s.effort_energy) for s in sig2])

    # Same trace of contraction, different spectra: scalar leverage is non-injective.
    B1 = np.diag([1.0, 0.0]); B2 = np.eye(2) / 3.0
    c1 = register_diagnostics(B1, np.array([1.0, 0.0]))
    c2 = register_diagnostics(B2, np.array([1.0, 0.0]))
    assert np.isclose(c1["trace_contraction"], c2["trace_contraction"])
    assert not np.allclose(np.linalg.eigvalsh(B1), np.linalg.eigvalsh(B2))
    print("orbit invariance: ok")
    print("non-injective scalarization: ok")


if __name__ == "__main__": main()
