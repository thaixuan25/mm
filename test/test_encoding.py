"""Round-trip tests cho `ml_dsa.encoding`."""

from __future__ import annotations

import random

import pytest

from ml_dsa.encoding import (
    BitPack,
    BitUnpack,
    BitsToBytes,
    BitsToInteger,
    BytesToBits,
    HintBitPack,
    HintBitUnpack,
    IntegerToBits,
    IntegerToBytes,
    pkDecode,
    pkEncode,
    sigDecode,
    sigEncode,
    SimpleBitPack,
    SimpleBitUnpack,
    skDecode,
    skEncode,
    w1Encode,
)
from ml_dsa.params import (
    D,
    ML_DSA_44_PARAMS,
    ML_DSA_65_PARAMS,
    ML_DSA_87_PARAMS,
    N,
    Q,
)


def test_integer_bits_round_trip():
    for x in [0, 1, 7, 255, 1024, 2 ** 20 - 1]:
        bits = IntegerToBits(x, 24)
        assert BitsToInteger(bits) == x


def test_bytes_bits_round_trip():
    rng = random.Random(7)
    for _ in range(5):
        data = bytes(rng.randint(0, 255) for _ in range(32))
        bits = BytesToBits(data)
        assert len(bits) == 256
        assert BitsToBytes(bits) == data


def test_integer_to_bytes_little_endian():
    assert IntegerToBytes(1, 2) == b"\x01\x00"
    assert IntegerToBytes(258, 2) == b"\x02\x01"
    assert IntegerToBytes(255, 1) == b"\xff"


def test_bits_to_bytes_uses_little_endian_bit_order():
    assert BitsToBytes([1, 0, 0, 0, 0, 0, 0, 1]) == b"\x81"
    assert BytesToBits(b"\x02")[:8] == [0, 1, 0, 0, 0, 0, 0, 0]


def test_bit_pack_round_trip_eta_2():
    rng = random.Random(0)
    coeffs = [rng.randint(-2, 2) for _ in range(N)]
    packed = BitPack(coeffs, 2, 2)
    assert len(packed) == 32 * 3
    decoded = BitUnpack(packed, 2, 2)
    assert decoded == coeffs


def test_bit_pack_round_trip_eta_4():
    rng = random.Random(1)
    coeffs = [rng.randint(-4, 4) for _ in range(N)]
    packed = BitPack(coeffs, 4, 4)
    assert len(packed) == 32 * 4
    decoded = BitUnpack(packed, 4, 4)
    assert decoded == coeffs


def test_bit_pack_round_trip_t0_range():
    a = (1 << (D - 1)) - 1
    b = 1 << (D - 1)
    rng = random.Random(2)
    coeffs = [rng.randint(-a, b) for _ in range(N)]
    packed = BitPack(coeffs, a, b)
    assert len(packed) == 32 * D
    decoded = BitUnpack(packed, a, b)
    assert decoded == coeffs


def test_bit_pack_round_trip_gamma1_range():
    for params in (ML_DSA_44_PARAMS, ML_DSA_65_PARAMS, ML_DSA_87_PARAMS):
        a = params.gamma1 - 1
        b = params.gamma1
        rng = random.Random(params.gamma1)
        coeffs = [rng.randint(-a, b) for _ in range(N)]
        packed = BitPack(coeffs, a, b)
        assert len(packed) == 32 * params.gamma1_bits
        decoded = BitUnpack(packed, a, b)
        assert decoded == coeffs


def test_simple_bit_pack_round_trip():
    bound = (1 << 10) - 1
    rng = random.Random(3)
    coeffs = [rng.randint(0, bound) for _ in range(N)]
    packed = SimpleBitPack(coeffs, bound)
    assert len(packed) == 32 * 10
    decoded = SimpleBitUnpack(packed, bound)
    assert decoded == coeffs


def test_hint_bit_pack_round_trip():
    rng = random.Random(13)
    params = ML_DSA_65_PARAMS
    h = [[0] * N for _ in range(params.k)]
    for i in range(params.k):
        positions = sorted(rng.sample(range(N), rng.randint(0, 5)))
        for pos in positions:
            h[i][pos] = 1
    packed = HintBitPack(h, params)
    assert len(packed) == params.omega + params.k
    decoded = HintBitUnpack(packed, params)
    assert decoded is not None
    assert decoded == h


def test_hint_bit_unpack_rejects_corrupt_data():
    params = ML_DSA_65_PARAMS
    bad = b"\x10" + b"\x00" * (params.omega + params.k - 1)
    bad = bytearray(bad)
    bad[params.omega] = 200
    assert HintBitUnpack(bytes(bad), params) is None


def test_hint_bit_unpack_rejects_nonzero_unused_tail():
    params = ML_DSA_44_PARAMS
    packed = bytearray(params.omega + params.k)
    packed[params.omega] = 1
    packed[0] = 7
    packed[1] = 99
    assert HintBitUnpack(bytes(packed), params) is None


def test_decode_functions_reject_wrong_lengths():
    params = ML_DSA_65_PARAMS
    with pytest.raises(ValueError):
        BitUnpack(b"\x00" * 3, params.eta, params.eta)
    with pytest.raises(ValueError):
        SimpleBitUnpack(b"\x00" * 3, (1 << params.t1_bits) - 1)
    with pytest.raises(ValueError):
        pkDecode(b"\x00" * (params.pk_bytes - 1), params)
    with pytest.raises(ValueError):
        skDecode(b"\x00" * (params.sk_bytes - 1), params)
    assert sigDecode(b"\x00" * (params.sig_bytes - 1), params) is None


def test_pk_encode_round_trip():
    params = ML_DSA_65_PARAMS
    rho = bytes(range(32))
    bound = (1 << params.t1_bits) - 1
    rng = random.Random(0xBABE)
    t1 = [
        [rng.randint(0, bound) for _ in range(N)] for _ in range(params.k)
    ]
    pk = pkEncode(rho, t1, params)
    assert len(pk) == params.pk_bytes
    rho2, t1_2 = pkDecode(pk, params)
    assert rho2 == rho
    assert t1_2 == t1


def test_sk_encode_round_trip():
    params = ML_DSA_44_PARAMS
    rng = random.Random(0xC0FFEE)
    rho = bytes(rng.randint(0, 255) for _ in range(32))
    K = bytes(rng.randint(0, 255) for _ in range(32))
    tr = bytes(rng.randint(0, 255) for _ in range(64))
    eta = params.eta
    s1 = [[rng.randint(-eta, eta) for _ in range(N)] for _ in range(params.l)]
    s2 = [[rng.randint(-eta, eta) for _ in range(N)] for _ in range(params.k)]
    a = (1 << (D - 1)) - 1
    b = 1 << (D - 1)
    t0 = [[rng.randint(-a, b) for _ in range(N)] for _ in range(params.k)]
    sk = skEncode(rho, K, tr, s1, s2, t0, params)
    assert len(sk) == params.sk_bytes
    rho2, K2, tr2, s1_2, s2_2, t0_2 = skDecode(sk, params)
    assert rho2 == rho
    assert K2 == K
    assert tr2 == tr
    assert s1_2 == s1
    assert s2_2 == s2
    assert t0_2 == t0


def test_sig_encode_round_trip():
    params = ML_DSA_87_PARAMS
    rng = random.Random(0xFEED)
    c_tilde = bytes(rng.randint(0, 255) for _ in range(params.c_tilde_bytes))
    a = params.gamma1 - 1
    b = params.gamma1
    z = [[rng.randint(-a, b) for _ in range(N)] for _ in range(params.l)]
    h = [[0] * N for _ in range(params.k)]
    total_ones = 0
    for i in range(params.k):
        ones_here = rng.randint(0, min(5, params.omega - total_ones))
        positions = sorted(rng.sample(range(N), ones_here))
        for pos in positions:
            h[i][pos] = 1
        total_ones += ones_here
    sig = sigEncode(c_tilde, z, h, params)
    assert len(sig) == params.sig_bytes
    decoded = sigDecode(sig, params)
    assert decoded is not None
    c_tilde_2, z_2, h_2 = decoded
    assert c_tilde_2 == c_tilde
    assert z_2 == z
    assert h_2 == h


def test_w1_encode_size():
    params = ML_DSA_65_PARAMS
    bound = (Q - 1) // (2 * params.gamma2) - 1
    rng = random.Random(444)
    w1 = [[rng.randint(0, bound) for _ in range(N)] for _ in range(params.k)]
    encoded = w1Encode(w1, params)
    expected_len = 32 * params.k * params.w1_bits
    assert len(encoded) == expected_len
