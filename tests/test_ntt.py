"""NTT/INTT tests và đối chiếu với schoolbook."""

from __future__ import annotations

import random

import pytest

from ml_dsa.ntt import ZETAS, intt, ntt, ntt_pointwise
from ml_dsa.params import N, Q, ZETA
from ml_dsa.poly import poly_schoolbook_mul


def test_zeta_is_primitive_512th_root():
    assert pow(ZETA, 256, Q) == Q - 1
    assert pow(ZETA, 512, Q) == 1


def test_zetas_table_size_and_first_entries():
    assert len(ZETAS) == N
    assert ZETAS[0] == 1


def test_ntt_round_trip():
    rng = random.Random(2024)
    for trial in range(5):
        a = [rng.randint(0, Q - 1) for _ in range(N)]
        a_hat = ntt(a)
        a_back = intt(a_hat)
        assert a_back == a, f"trial {trial}: round-trip failed"


def test_ntt_zero_polynomial():
    zero = [0] * N
    assert ntt(zero) == zero
    assert intt(zero) == zero


def test_ntt_constant_polynomial():
    """NTT(c · 1) = (c, c, ..., c) where 1 is the constant polynomial."""
    a = [0] * N
    a[0] = 5
    a_hat = ntt(a)
    assert all(x == 5 for x in a_hat)


def test_ntt_pointwise_matches_schoolbook():
    rng = random.Random(11)
    a = [rng.randint(0, Q - 1) for _ in range(N)]
    b = [rng.randint(0, Q - 1) for _ in range(N)]

    expected = poly_schoolbook_mul(a, b)
    a_hat = ntt(a)
    b_hat = ntt(b)
    prod_hat = ntt_pointwise(a_hat, b_hat)
    actual = intt(prod_hat)
    assert actual == expected


def test_ntt_pointwise_two_random_pairs():
    rng = random.Random(99)
    for _ in range(3):
        a = [rng.randint(0, Q - 1) for _ in range(N)]
        b = [rng.randint(0, Q - 1) for _ in range(N)]
        expected = poly_schoolbook_mul(a, b)
        a_hat = ntt(a)
        b_hat = ntt(b)
        actual = intt(ntt_pointwise(a_hat, b_hat))
        assert actual == expected


def test_ntt_returns_new_list():
    a = [0] * N
    a[0] = 1
    a_hat = ntt(a)
    assert a_hat is not a
    a[0] = 99
    a_hat2 = ntt(a)
    assert a_hat != a_hat2
