"""Unit tests cho `ml_dsa.field` và `ml_dsa.poly`."""

from __future__ import annotations

import random

import pytest

from ml_dsa.field import CenterModQ, InfinityNorm, ModPM, ModQ
from ml_dsa.params import N, Q
from ml_dsa.poly import (
    PolyAdd,
    PolyNeg,
    PolyReduce,
    PolySchoolbookMul,
    PolySub,
    VectorAdd,
    VectorInfinityNorm,
    VectorSub,
    ZeroPoly,
    ZeroVector,
)


def test_mod_q_normalization():
    assert ModQ(0) == 0
    assert ModQ(Q) == 0
    assert ModQ(Q + 5) == 5
    assert ModQ(-1) == Q - 1


def test_center_mod_q_balanced_range():
    assert CenterModQ(0) == 0
    assert CenterModQ(Q - 1) == -1
    assert CenterModQ((Q + 1) // 2) == (Q + 1) // 2 - Q
    assert CenterModQ(Q // 2) == Q // 2


def test_mod_pm_even_alpha():
    alpha = 8
    for x in range(-100, 100):
        r = ModPM(x, alpha)
        assert -alpha // 2 < r <= alpha // 2
        assert (x - r) % alpha == 0


def test_mod_pm_odd_alpha():
    alpha = 7
    for x in range(-100, 100):
        r = ModPM(x, alpha)
        assert -(alpha - 1) // 2 <= r <= (alpha - 1) // 2
        assert (x - r) % alpha == 0


def test_infinity_norm_picks_largest_abs():
    coeffs = [0, 5, Q - 3, Q - 1, 100]
    assert InfinityNorm(coeffs) == 100


def test_zero_poly_and_vec_dimensions():
    p = ZeroPoly()
    assert len(p) == N
    assert all(c == 0 for c in p)
    v = ZeroVector(4)
    assert len(v) == 4
    assert all(len(p) == N for p in v)


def test_poly_add_sub_neg_consistency():
    rng = random.Random(0xC0DE)
    a = [rng.randint(0, Q - 1) for _ in range(N)]
    b = [rng.randint(0, Q - 1) for _ in range(N)]
    s = PolyAdd(a, b)
    diff = PolySub(a, b)
    neg = PolyNeg(b)
    for i in range(N):
        assert s[i] == (a[i] + b[i]) % Q
        assert diff[i] == (a[i] - b[i]) % Q
        assert neg[i] == (-b[i]) % Q


def test_poly_schoolbook_mul_matches_definition():
    rng = random.Random(42)
    n_small_test = 8
    a = [rng.randint(0, Q - 1) for _ in range(N)]
    b = [rng.randint(0, Q - 1) for _ in range(N)]

    expected = [0] * N
    for i in range(N):
        for j in range(N):
            k = i + j
            if k < N:
                expected[k] = (expected[k] + a[i] * b[j]) % Q
            else:
                expected[k - N] = (expected[k - N] - a[i] * b[j]) % Q

    actual = PolySchoolbookMul(a, b)
    assert actual == expected
    assert len(actual) == N


def test_poly_schoolbook_mul_x_n_equals_minus_one():
    """Trong R_q, X · X^{n-1} = X^n ≡ -1 (mod X^n + 1)."""
    x = [0] * N
    x[1] = 1
    x_nm1 = [0] * N
    x_nm1[N - 1] = 1
    product = PolySchoolbookMul(x, x_nm1)
    expected = [0] * N
    expected[0] = (-1) % Q
    assert product == expected


def test_vec_ops_consistent():
    rng = random.Random(7)
    u = [[rng.randint(0, Q - 1) for _ in range(N)] for _ in range(3)]
    v = [[rng.randint(0, Q - 1) for _ in range(N)] for _ in range(3)]
    s = VectorAdd(u, v)
    d = VectorSub(u, v)
    for i in range(3):
        for j in range(N):
            assert s[i][j] == (u[i][j] + v[i][j]) % Q
            assert d[i][j] == (u[i][j] - v[i][j]) % Q


def test_vec_infinity_norm_picks_global_max():
    p1 = [0] * N
    p2 = [0] * N
    p1[10] = 100
    p2[20] = Q - 50
    assert VectorInfinityNorm([p1, p2]) == 100


def test_mismatched_vec_lengths_raise():
    with pytest.raises(ValueError):
        VectorAdd([ZeroPoly()], [ZeroPoly(), ZeroPoly()])


def test_poly_operations_keep_coefficients_canonical_at_boundaries():
    a = [0, 1, Q - 1, Q, -1] + [Q + i for i in range(N - 5)]
    b = [Q - 1, Q - 2, 1, -Q, 2 * Q + 7] + [-(i + 1) for i in range(N - 5)]
    for poly in (PolyAdd(a, b), PolySub(a, b), PolyNeg(a), PolyReduce(a)):
        assert len(poly) == N
        assert all(0 <= coeff < Q for coeff in poly)
