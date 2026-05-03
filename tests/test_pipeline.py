"""End-to-end keygen/sign/verify cho mọi mức an toàn ML-DSA."""

from __future__ import annotations

import os

import pytest

from ml_dsa import ML_DSA_44, ML_DSA_65, ML_DSA_87
from ml_dsa.api import ML_DSA


PARAMS = [
    pytest.param(ML_DSA_44, id="ML-DSA-44"),
    pytest.param(ML_DSA_65, id="ML-DSA-65"),
    pytest.param(ML_DSA_87, id="ML-DSA-87"),
]


@pytest.mark.parametrize("scheme", PARAMS)
def test_keygen_returns_correct_sizes(scheme: ML_DSA):
    pk, sk = scheme.keygen(xi=b"\x00" * 32)
    assert len(pk) == scheme.pk_bytes
    assert len(sk) == scheme.sk_bytes


@pytest.mark.parametrize("scheme", PARAMS)
def test_sign_verify_round_trip(scheme: ML_DSA):
    pk, sk = scheme.keygen(xi=b"\x01" * 32)
    msg = b"the quick brown fox"
    sig = scheme.sign(sk, msg, deterministic=True)
    assert len(sig) == scheme.sig_bytes
    assert scheme.verify(pk, msg, sig) is True


@pytest.mark.parametrize("scheme", PARAMS)
def test_verify_fails_on_modified_message(scheme: ML_DSA):
    pk, sk = scheme.keygen(xi=b"\x02" * 32)
    msg = b"original"
    sig = scheme.sign(sk, msg, deterministic=True)
    assert scheme.verify(pk, b"tampered", sig) is False


@pytest.mark.parametrize("scheme", PARAMS)
def test_verify_fails_on_modified_signature(scheme: ML_DSA):
    pk, sk = scheme.keygen(xi=b"\x03" * 32)
    msg = b"hello"
    sig = bytearray(scheme.sign(sk, msg, deterministic=True))
    sig[scheme.params.c_tilde_bytes + 5] ^= 0x01
    assert scheme.verify(pk, msg, bytes(sig)) is False


@pytest.mark.parametrize("scheme", PARAMS)
def test_verify_fails_on_wrong_public_key(scheme: ML_DSA):
    pk1, sk1 = scheme.keygen(xi=b"\x04" * 32)
    pk2, _ = scheme.keygen(xi=b"\x05" * 32)
    msg = b"context-aware"
    sig = scheme.sign(sk1, msg, deterministic=True)
    assert scheme.verify(pk1, msg, sig) is True
    assert scheme.verify(pk2, msg, sig) is False


@pytest.mark.parametrize("scheme", PARAMS)
def test_deterministic_signing_is_repeatable(scheme: ML_DSA):
    _, sk = scheme.keygen(xi=b"\x06" * 32)
    msg = b"replay"
    sig1 = scheme.sign(sk, msg, deterministic=True)
    sig2 = scheme.sign(sk, msg, deterministic=True)
    assert sig1 == sig2


@pytest.mark.parametrize("scheme", PARAMS)
def test_randomized_signing_produces_distinct_sigs(scheme: ML_DSA):
    _, sk = scheme.keygen(xi=b"\x07" * 32)
    msg = b"random"
    sig1 = scheme.sign(sk, msg, rnd=os.urandom(32))
    sig2 = scheme.sign(sk, msg, rnd=os.urandom(32))
    assert sig1 != sig2


@pytest.mark.parametrize("scheme", PARAMS)
def test_keygen_deterministic_with_xi(scheme: ML_DSA):
    pk1, sk1 = scheme.keygen(xi=b"\x08" * 32)
    pk2, sk2 = scheme.keygen(xi=b"\x08" * 32)
    assert pk1 == pk2 and sk1 == sk2


@pytest.mark.parametrize("scheme", PARAMS)
def test_context_string_separates_signatures(scheme: ML_DSA):
    pk, sk = scheme.keygen(xi=b"\x09" * 32)
    msg = b"shared message"
    sig_a = scheme.sign(sk, msg, ctx=b"app-A", deterministic=True)
    sig_b = scheme.sign(sk, msg, ctx=b"app-B", deterministic=True)
    assert sig_a != sig_b
    assert scheme.verify(pk, msg, sig_a, ctx=b"app-A") is True
    assert scheme.verify(pk, msg, sig_a, ctx=b"app-B") is False
    assert scheme.verify(pk, msg, sig_b, ctx=b"app-B") is True


@pytest.mark.parametrize("scheme", PARAMS)
def test_verify_rejects_oversized_ctx(scheme: ML_DSA):
    pk, sk = scheme.keygen(xi=b"\x0a" * 32)
    msg = b"x"
    sig = scheme.sign(sk, msg, deterministic=True)
    assert scheme.verify(pk, msg, sig, ctx=b"a" * 256) is False


@pytest.mark.parametrize("scheme", PARAMS)
def test_verify_rejects_wrong_size_inputs(scheme: ML_DSA):
    pk, sk = scheme.keygen(xi=b"\x0b" * 32)
    msg = b"hi"
    sig = scheme.sign(sk, msg, deterministic=True)
    assert scheme.verify(pk[:-1], msg, sig) is False
    assert scheme.verify(pk, msg, sig[:-1]) is False


def test_signing_with_all_three_schemes_using_same_seed_yields_distinct_keys():
    """Domain separation by (k, l) means different schemes share xi but diverge."""
    xi = b"\x42" * 32
    pks = {scheme.name: scheme.keygen(xi=xi)[0] for scheme in (ML_DSA_44, ML_DSA_65, ML_DSA_87)}
    assert len({bytes(pk) for pk in pks.values()}) == 3
