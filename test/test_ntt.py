"""NTT/INTT tests và đối chiếu với schoolbook."""

from __future__ import annotations

import random

import pytest

from ml_dsa.ntt import ZETAS, INTT, NTT, MultiplyNTT
from ml_dsa.params import N, Q, ZETA
from ml_dsa.poly import PolySchoolbookMul


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
        a_hat = NTT(a)
        a_back = INTT(a_hat)
        assert a_back == a, f"trial {trial}: round-trip failed"


def test_ntt_zero_polynomial():
    zero = [0] * N
    assert NTT(zero) == zero
    assert INTT(zero) == zero


def test_ntt_constant_polynomial():
    """NTT(c · 1) = (c, c, ..., c) where 1 is the constant polynomial."""
    a = [0] * N
    a[0] = 5
    a_hat = NTT(a)
    assert all(x == 5 for x in a_hat)


def test_ntt_pointwise_matches_schoolbook():
    rng = random.Random(11)
    a = [rng.randint(0, Q - 1) for _ in range(N)]
    b = [rng.randint(0, Q - 1) for _ in range(N)]

    expected = PolySchoolbookMul(a, b)
    a_hat = NTT(a)
    b_hat = NTT(b)
    prod_hat = MultiplyNTT(a_hat, b_hat)
    actual = INTT(prod_hat)
    assert actual == expected


def test_ntt_pointwise_two_random_pairs():
    rng = random.Random(99)
    for _ in range(3):
        a = [rng.randint(0, Q - 1) for _ in range(N)]
        b = [rng.randint(0, Q - 1) for _ in range(N)]
        expected = PolySchoolbookMul(a, b)
        a_hat = NTT(a)
        b_hat = NTT(b)
        actual = INTT(MultiplyNTT(a_hat, b_hat))
        assert actual == expected


def test_ntt_returns_new_list():
    a = [0] * N
    a[0] = 1
    a_hat = NTT(a)
    assert a_hat is not a
    a[0] = 99
    a_hat2 = NTT(a)
    assert a_hat != a_hat2


def test_ntt_and_intt_outputs_are_canonical_for_boundary_inputs():
    a = [0, Q - 1, Q, -1] + [(i * Q + i) for i in range(N - 4)]
    a_reduced = [x % Q for x in a]
    a_hat = NTT(a)
    a_back = INTT(a_hat)
    assert all(0 <= x < Q for x in a_hat)
    assert all(0 <= x < Q for x in a_back)
    assert a_back == a_reduced
