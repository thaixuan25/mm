"""Unit tests cho `ml_dsa.field` và `ml_dsa.poly`."""

from __future__ import annotations

import random

import pytest

from ml_dsa.field import center_mod_q, infinity_norm, mod_pm, mod_q
from ml_dsa.params import N, Q
from ml_dsa.poly import (
    poly_add,
    poly_neg,
    poly_schoolbook_mul,
    poly_sub,
    vec_add,
    vec_infinity_norm,
    vec_sub,
    zero_poly,
    zero_vec,
)


def test_mod_q_normalization():
    assert mod_q(0) == 0
    assert mod_q(Q) == 0
    assert mod_q(Q + 5) == 5
    assert mod_q(-1) == Q - 1


def test_center_mod_q_balanced_range():
    assert center_mod_q(0) == 0
    assert center_mod_q(Q - 1) == -1
    assert center_mod_q((Q + 1) // 2) == (Q + 1) // 2 - Q
    assert center_mod_q(Q // 2) == Q // 2


def test_mod_pm_even_alpha():
    alpha = 8
    for x in range(-100, 100):
        r = mod_pm(x, alpha)
        assert -alpha // 2 < r <= alpha // 2
        assert (x - r) % alpha == 0


def test_mod_pm_odd_alpha():
    alpha = 7
    for x in range(-100, 100):
        r = mod_pm(x, alpha)
        assert -(alpha - 1) // 2 <= r <= (alpha - 1) // 2
        assert (x - r) % alpha == 0


def test_infinity_norm_picks_largest_abs():
    coeffs = [0, 5, Q - 3, Q - 1, 100]
    assert infinity_norm(coeffs) == 100


def test_zero_poly_and_vec_dimensions():
    p = zero_poly()
    assert len(p) == N
    assert all(c == 0 for c in p)
    v = zero_vec(4)
    assert len(v) == 4
    assert all(len(p) == N for p in v)


def test_poly_add_sub_neg_consistency():
    rng = random.Random(0xC0DE)
    a = [rng.randint(0, Q - 1) for _ in range(N)]
    b = [rng.randint(0, Q - 1) for _ in range(N)]
    s = poly_add(a, b)
    diff = poly_sub(a, b)
    neg = poly_neg(b)
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

    actual = poly_schoolbook_mul(a, b)
    assert actual == expected
    assert len(actual) == N


def test_poly_schoolbook_mul_x_n_equals_minus_one():
    """Trong R_q, X · X^{n-1} = X^n ≡ -1 (mod X^n + 1)."""
    x = [0] * N
    x[1] = 1
    x_nm1 = [0] * N
    x_nm1[N - 1] = 1
    product = poly_schoolbook_mul(x, x_nm1)
    expected = [0] * N
    expected[0] = (-1) % Q
    assert product == expected


def test_vec_ops_consistent():
    rng = random.Random(7)
    u = [[rng.randint(0, Q - 1) for _ in range(N)] for _ in range(3)]
    v = [[rng.randint(0, Q - 1) for _ in range(N)] for _ in range(3)]
    s = vec_add(u, v)
    d = vec_sub(u, v)
    for i in range(3):
        for j in range(N):
            assert s[i][j] == (u[i][j] + v[i][j]) % Q
            assert d[i][j] == (u[i][j] - v[i][j]) % Q


def test_vec_infinity_norm_picks_global_max():
    p1 = [0] * N
    p2 = [0] * N
    p1[10] = 100
    p2[20] = Q - 50
    assert vec_infinity_norm([p1, p2]) == 100


def test_mismatched_vec_lengths_raise():
    with pytest.raises(ValueError):
        vec_add([zero_poly()], [zero_poly(), zero_poly()])
