"""ML-DSA.Sign / Sign_internal (FIPS 204 §6.2, Algorithm 2 & 7)."""

from __future__ import annotations

import os
from typing import Optional

from ml_dsa.encoding import sig_encode, sk_decode, w1_encode
from ml_dsa.ntt import intt_vec, ntt_matvec, ntt_pointwise, ntt_vec, ntt
from ml_dsa.params import MLDSAParams, Q
from ml_dsa.poly import (
    poly_to_centered,
    vec_add,
    vec_infinity_norm,
    vec_reduce,
    vec_sub,
)
from ml_dsa.reduction import vec_high_bits, vec_low_bits, vec_make_hint
from ml_dsa.sampling import H_shake256, expand_a, expand_mask, sample_in_ball


def _ntt_each(vec):
    return ntt_vec(vec)


def _intt_each(vec):
    return intt_vec(vec)


def _hat_pointwise_with_scalar_hat(scalar_hat, vec_hat):
    return [ntt_pointwise(scalar_hat, p) for p in vec_hat]


def _vec_neg(v):
    return [[(-x) % Q for x in p] for p in v]


def _hint_popcount(h):
    return sum(sum(x for x in p) for p in h)


def sign_internal(
    sk: bytes, message_prime: bytes, rnd: bytes, params: MLDSAParams
) -> bytes:
    """Sign_internal (Algorithm 7).

    `message_prime` đã bao gồm domain separator + context theo Algorithm 2.
    `rnd` là 32 byte ngẫu nhiên (dùng b"\\x00" * 32 cho deterministic mode).
    """
    if len(rnd) != 32:
        raise ValueError("sign_internal: rnd must be 32 bytes")

    rho, K, tr, s1, s2, t0 = sk_decode(sk, params)

    s1_hat = _ntt_each(s1)
    s2_hat = _ntt_each(s2)
    t0_hat = _ntt_each(t0)
    A_hat = expand_a(rho, params)

    mu = H_shake256(tr + message_prime, 64)
    rho_pp = H_shake256(K + rnd + mu, 64)

    kappa = 0
    while True:
        y = expand_mask(rho_pp, kappa, params)
        y_hat = _ntt_each(y)
        Ay_hat = ntt_matvec(A_hat, y_hat)
        w = vec_reduce(_intt_each(Ay_hat))
        w1 = vec_high_bits(w, params.gamma2)
        c_tilde = H_shake256(mu + w1_encode(w1, params), params.c_tilde_bytes)

        c = sample_in_ball(c_tilde, params)
        c_hat = ntt(c)

        cs1_hat = _hat_pointwise_with_scalar_hat(c_hat, s1_hat)
        cs2_hat = _hat_pointwise_with_scalar_hat(c_hat, s2_hat)
        cs1 = vec_reduce(_intt_each(cs1_hat))
        cs2 = vec_reduce(_intt_each(cs2_hat))

        z = vec_reduce(vec_add(y, cs1))
        r0 = vec_low_bits(vec_reduce(vec_sub(w, cs2)), params.gamma2)

        if (
            vec_infinity_norm(z) >= params.gamma1 - params.beta
            or max(abs(x) for p in r0 for x in p) >= params.gamma2 - params.beta
        ):
            kappa += params.l
            continue

        ct0_hat = _hat_pointwise_with_scalar_hat(c_hat, t0_hat)
        ct0 = vec_reduce(_intt_each(ct0_hat))

        if vec_infinity_norm(ct0) >= params.gamma2:
            kappa += params.l
            continue

        neg_ct0 = _vec_neg(ct0)
        r_for_hint = vec_reduce(
            vec_add(vec_sub(w, cs2), ct0)
        )
        h = vec_make_hint(neg_ct0, r_for_hint, params.gamma2)

        if _hint_popcount(h) > params.omega:
            kappa += params.l
            continue

        z_centered = [poly_to_centered(p) for p in z]
        return sig_encode(c_tilde, z_centered, h, params)


def _format_message_prime(message: bytes, ctx: bytes) -> bytes:
    if len(ctx) > 255:
        raise ValueError("ctx exceeds 255 bytes")
    return bytes([0, len(ctx)]) + ctx + message


def sign(
    sk: bytes,
    message: bytes,
    params: MLDSAParams,
    *,
    ctx: bytes = b"",
    deterministic: bool = False,
    rnd: Optional[bytes] = None,
) -> bytes:
    """ML-DSA.Sign (Algorithm 2).

    - `ctx`: chuỗi context tùy chọn (≤ 255 byte) cho domain separation.
    - `deterministic=True`: dùng rnd toàn 0, chữ ký lặp lại được.
    - `rnd`: tham số 32 byte rõ ràng (ưu tiên hơn `deterministic`).
    """
    message_prime = _format_message_prime(message, ctx)
    if rnd is None:
        rnd = b"\x00" * 32 if deterministic else os.urandom(32)
    return sign_internal(sk, message_prime, rnd, params)


__all__ = ["sign", "sign_internal"]
