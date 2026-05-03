"""Tests cho `ml_dsa.sampling`."""

from __future__ import annotations

import hashlib

from ml_dsa.params import (
    ML_DSA_44_PARAMS,
    ML_DSA_65_PARAMS,
    ML_DSA_87_PARAMS,
    N,
    Q,
)
from ml_dsa.sampling import (
    SHAKEStream,
    coeff_from_half_byte,
    coeff_from_three_bytes,
    expand_a,
    expand_mask,
    expand_s,
    rej_bounded_poly,
    rej_ntt_poly,
    sample_in_ball,
)


def test_shake_stream_consistent_with_oneshot():
    seed = b"shake-stream-test"
    direct = hashlib.shake_256(seed).digest(123)
    stream = SHAKEStream(hashlib.shake_256, seed)
    chunks = []
    for size in (1, 2, 5, 32, 16, 67):
        chunks.append(stream.read(size))
    out = b"".join(chunks)
    assert out == direct


def test_shake_stream_grows_buffer_correctly():
    seed = b"x"
    stream = SHAKEStream(hashlib.shake_128, seed)
    a = stream.read(200)
    b = stream.read(200)
    assert len(a) == 200 and len(b) == 200
    direct = hashlib.shake_128(seed).digest(400)
    assert a + b == direct


def test_coeff_from_three_bytes_accepts_below_q():
    z = coeff_from_three_bytes(0xFF, 0xFF, 0x7F)
    assert z is None
    z2 = coeff_from_three_bytes(0x00, 0x00, 0x00)
    assert z2 == 0
    z3 = coeff_from_three_bytes(0x01, 0x00, 0x00)
    assert z3 == 1


def test_coeff_from_half_byte_eta2():
    counts = [0] * 5
    rejects = 0
    for b in range(16):
        v = coeff_from_half_byte(b, 2)
        if v is None:
            rejects += 1
        else:
            counts[v + 2] += 1
    assert rejects == 1
    assert sum(counts) == 15


def test_coeff_from_half_byte_eta4():
    rejects = 0
    seen = set()
    for b in range(16):
        v = coeff_from_half_byte(b, 4)
        if v is None:
            rejects += 1
        else:
            seen.add(v)
    assert rejects == 7
    assert seen == set(range(-4, 5))


def test_rej_ntt_poly_deterministic_and_in_range():
    seed = b"a" * 34
    p1 = rej_ntt_poly(seed)
    p2 = rej_ntt_poly(seed)
    assert p1 == p2
    assert len(p1) == N
    assert all(0 <= x < Q for x in p1)


def test_rej_bounded_poly_eta_2_range():
    seed = b"b" * 66
    p = rej_bounded_poly(seed, 2)
    assert len(p) == N
    centered = [(x if x <= Q // 2 else x - Q) for x in p]
    assert all(-2 <= v <= 2 for v in centered)


def test_rej_bounded_poly_eta_4_range():
    seed = b"c" * 66
    p = rej_bounded_poly(seed, 4)
    centered = [(x if x <= Q // 2 else x - Q) for x in p]
    assert all(-4 <= v <= 4 for v in centered)


def test_expand_a_dimensions_and_determinism():
    rho = b"\x01" * 32
    A1 = expand_a(rho, ML_DSA_44_PARAMS)
    A2 = expand_a(rho, ML_DSA_44_PARAMS)
    assert len(A1) == ML_DSA_44_PARAMS.k
    assert all(len(row) == ML_DSA_44_PARAMS.l for row in A1)
    assert A1 == A2
    for row in A1:
        for poly in row:
            assert len(poly) == N
            assert all(0 <= x < Q for x in poly)


def test_expand_s_dimensions_and_range():
    rho = b"\x02" * 64
    s1, s2 = expand_s(rho, ML_DSA_65_PARAMS)
    eta = ML_DSA_65_PARAMS.eta
    assert len(s1) == ML_DSA_65_PARAMS.l
    assert len(s2) == ML_DSA_65_PARAMS.k
    for v in s1 + s2:
        centered = [(x if x <= Q // 2 else x - Q) for x in v]
        assert all(-eta <= c <= eta for c in centered)


def test_expand_mask_range_and_dimensions():
    rho_pp = b"\x03" * 64
    y = expand_mask(rho_pp, 0, ML_DSA_87_PARAMS)
    gamma1 = ML_DSA_87_PARAMS.gamma1
    assert len(y) == ML_DSA_87_PARAMS.l
    for poly in y:
        assert len(poly) == N
        for x in poly:
            v = x if x <= Q // 2 else x - Q
            assert -gamma1 + 1 <= v <= gamma1 or x == 0


def test_sample_in_ball_has_tau_nonzero_entries():
    for params in (ML_DSA_44_PARAMS, ML_DSA_65_PARAMS, ML_DSA_87_PARAMS):
        rho = bytes(range(params.c_tilde_bytes))
        c = sample_in_ball(rho, params)
        nonzero = sum(1 for x in c if x != 0)
        assert nonzero == params.tau
        for x in c:
            assert x in (0, 1, Q - 1)


def test_sample_in_ball_deterministic_for_same_seed():
    rho = b"\x00" * ML_DSA_65_PARAMS.c_tilde_bytes
    c1 = sample_in_ball(rho, ML_DSA_65_PARAMS)
    c2 = sample_in_ball(rho, ML_DSA_65_PARAMS)
    assert c1 == c2
