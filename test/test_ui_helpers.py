from __future__ import annotations

import pytest

from ml_dsa import ML_DSA_44, ML_DSA_65, ML_DSA_87
from ui_helpers import (
    InputValidationError,
    bytes_to_hex,
    parse_hex_bytes,
    parse_message,
    resolve_pre_hash,
    validate_context,
)


def test_parse_hex_bytes_accepts_wrapped_hex_and_prefix():
    assert parse_hex_bytes("0xAA bb\ncc", field_name="Value") == b"\xaa\xbb\xcc"


def test_parse_hex_bytes_rejects_invalid_hex():
    with pytest.raises(InputValidationError):
        parse_hex_bytes("abc", field_name="Seed")
    with pytest.raises(InputValidationError):
        parse_hex_bytes("zz", field_name="Seed")


def test_parse_hex_bytes_rejects_wrong_length():
    with pytest.raises(InputValidationError):
        parse_hex_bytes("00" * 31, field_name="Seed", expected_len=32)


def test_bytes_to_hex_round_trip():
    value = bytes(range(32))
    wrapped = bytes_to_hex(value, line_chars=16)
    assert parse_hex_bytes(wrapped, field_name="Value", expected_len=32) == value


def test_parse_message_modes():
    assert parse_message("hello", "UTF-8 text") == b"hello"
    assert parse_message("68656c6c6f", "Hex bytes") == b"hello"


def test_validate_context_limit():
    assert validate_context("ctx") == b"ctx"
    with pytest.raises(InputValidationError):
        validate_context("a" * 256)


def test_resolve_pre_hash_modes():
    assert resolve_pre_hash("Pure", "sha256") is None
    assert resolve_pre_hash("HashML-DSA", "sha256") == "sha256"
    with pytest.raises(InputValidationError):
        resolve_pre_hash("HashML-DSA", "md5")


@pytest.mark.parametrize("scheme", (ML_DSA_44, ML_DSA_65, ML_DSA_87))
def test_round_trip_using_helper_formatted_values(scheme):
    pk, sk = scheme.keygen(xi=bytes([scheme.params.k]) * 32)
    msg = parse_message("demo message", "UTF-8 text")
    ctx = validate_context("demo")
    sig = scheme.sign(sk, msg, ctx=ctx, deterministic=True)

    pk2 = parse_hex_bytes(bytes_to_hex(pk), field_name="Public key", expected_len=scheme.pk_bytes)
    sk2 = parse_hex_bytes(bytes_to_hex(sk), field_name="Secret key", expected_len=scheme.sk_bytes)
    sig2 = parse_hex_bytes(bytes_to_hex(sig), field_name="Signature", expected_len=scheme.sig_bytes)

    assert sk2 == sk
    assert scheme.verify(pk2, msg, sig2, ctx=ctx) is True
