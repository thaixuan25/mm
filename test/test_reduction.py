"""Tests cho `ml_dsa.reduction`."""

from __future__ import annotations

import random

from ml_dsa.params import D, ML_DSA_44_PARAMS, ML_DSA_65_PARAMS, ML_DSA_87_PARAMS, Q
from ml_dsa.reduction import (
    Decompose,
    HighBits,
    LowBits,
    MakeHint,
    Power2Round,
    UseHint,
)


def test_power2round_decomposition_identity():
    rng = random.Random(123)
    for _ in range(200):
        r = rng.randint(0, Q - 1)
        r1, r0 = Power2Round(r)
        assert -(1 << (D - 1)) < r0 <= (1 << (D - 1))
        assert (r1 * (1 << D) + r0) % Q == r


def test_power2round_known_values():
    r1, r0 = Power2Round(0)
    assert (r1, r0) == (0, 0)
    r1, r0 = Power2Round(1 << D)
    assert (r1, r0) == (1, 0)


def test_decompose_identity_for_44():
    gamma2 = ML_DSA_44_PARAMS.gamma2
    rng = random.Random(7)
    for _ in range(200):
        r = rng.randint(0, Q - 1)
        r1, r0 = Decompose(r, gamma2)
        m = (Q - 1) // (2 * gamma2)
        assert -gamma2 <= r0 <= gamma2
        assert 0 <= r1 < m
        assert (r1 * (2 * gamma2) + r0) % Q == r


def test_decompose_identity_for_65():
    gamma2 = ML_DSA_65_PARAMS.gamma2
    rng = random.Random(8)
    for _ in range(200):
        r = rng.randint(0, Q - 1)
        r1, r0 = Decompose(r, gamma2)
        m = (Q - 1) // (2 * gamma2)
        assert -gamma2 <= r0 <= gamma2
        assert 0 <= r1 < m
        assert (r1 * (2 * gamma2) + r0) % Q == r


def test_high_low_bits_split():
    gamma2 = ML_DSA_65_PARAMS.gamma2
    rng = random.Random(99)
    for _ in range(50):
        r = rng.randint(0, Q - 1)
        h = HighBits(r, gamma2)
        l = LowBits(r, gamma2)
        r1, r0 = Decompose(r, gamma2)
        assert h == r1
        assert l == r0


def test_use_hint_inverts_perturbation():
    """UseHint(MakeHint(z, r), r) ≡ HighBits(r + z) khi |z| nhỏ."""
    gamma2 = ML_DSA_65_PARAMS.gamma2
    rng = random.Random(55)
    for _ in range(200):
        r = rng.randint(0, Q - 1)
        z = rng.randint(-gamma2 // 2, gamma2 // 2)
        h = MakeHint(z, r, gamma2)
        recovered = UseHint(h, r, gamma2)
        expected = HighBits((r + z) % Q, gamma2)
        assert recovered == expected


def test_use_hint_zero_hint_returns_high_bits():
    gamma2 = ML_DSA_65_PARAMS.gamma2
    rng = random.Random(77)
    for _ in range(50):
        r = rng.randint(0, Q - 1)
        assert UseHint(0, r, gamma2) == HighBits(r, gamma2)


def test_make_hint_zero_for_zero_perturbation():
    gamma2 = ML_DSA_65_PARAMS.gamma2
    rng = random.Random(13)
    for _ in range(20):
        r = rng.randint(0, Q - 1)
        assert MakeHint(0, r, gamma2) == 0


def test_decompose_identity_at_boundaries_for_all_parameter_sets():
    boundary_values = [
        0,
        1,
        Q - 1,
        Q // 2,
        (Q - 1) // 88,
        (Q - 1) // 32,
        Q - ((Q - 1) // 88),
        Q - ((Q - 1) // 32),
    ]
    for params in (ML_DSA_44_PARAMS, ML_DSA_65_PARAMS, ML_DSA_87_PARAMS):
        for r in boundary_values:
            r1, r0 = Decompose(r, params.gamma2)
            assert (r1 * (2 * params.gamma2) + r0) % Q == r % Q
            assert HighBits(r, params.gamma2) == r1
            assert LowBits(r, params.gamma2) == r0
