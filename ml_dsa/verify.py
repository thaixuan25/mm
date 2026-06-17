from __future__ import annotations

from .encoding import pkDecode, sigDecode, w1Encode
from .ntt import INTT, MatrixVectorNTT, NTT, MultiplyNTT
from .params import D, MLDSAParams, Q
from .prehash import FormatMessagePrime
from .poly import VectorInfinityNorm, VectorReduce, VectorSub
from .reduction import VectorUseHint
from .sampling import HShake256, ExpandA, SampleInBall


def _scale_t1(t1, factor):
    """Nhân từng hệ số của vector ``t1`` với hằng số (mod q).

    Trong verify ta cần ``t1 · 2^d`` để đối chiếu với ``A·z``, vì t1 là
    phần "high" sau Power2Round.
    """
    return [[(x * factor) % Q for x in p] for p in t1]


def _hint_popcount(h):
    """Đếm số bit hint = 1 toàn vector."""
    return sum(sum(p) for p in h)


def Verify_internal(
    pk: bytes, message_prime: bytes, sig: bytes, params: MLDSAParams
) -> bool:
    """
    Verify_internal (Algorithm 8).
    """
    decoded_sig = sigDecode(sig, params)
    if decoded_sig is None:
        return False
    c_tilde, z, h = decoded_sig

    if _hint_popcount(h) > params.omega:
        return False

    if VectorInfinityNorm(z) >= params.gamma1 - params.beta:
        return False

    rho, t1 = pkDecode(pk, params)
    A_hat = ExpandA(rho, params)
    tr = HShake256(pk, 64)
    mu = HShake256(tr + message_prime, 64)

    c = SampleInBall(c_tilde, params)
    c_hat = NTT(c)

    # Tính A·z - c·t1·2^d trong miền NTT để khớp với cách signer dựng w.
    z_hat = [NTT(p) for p in z]
    Az_hat = MatrixVectorNTT(A_hat, z_hat)

    t1_scaled = _scale_t1(t1, 1 << D)
    t1_scaled_hat = [NTT(p) for p in t1_scaled]
    ct1_hat = [MultiplyNTT(c_hat, p) for p in t1_scaled_hat]

    diff_hat = [
        [(Az_hat[i][j] - ct1_hat[i][j]) % Q for j in range(len(Az_hat[i]))]
        for i in range(len(Az_hat))
    ]
    w_approx = VectorReduce([INTT(p) for p in diff_hat])

    # UseHint phục hồi ``w1`` xấp xỉ; nếu hint chuẩn từ signer thì kết quả
    # sẽ trùng với w1 ban đầu, từ đó c_tilde tái sinh đúng.
    w1_prime = VectorUseHint(h, w_approx, params.gamma2)
    c_tilde_prime = HShake256(
        mu + w1Encode(w1_prime, params), params.c_tilde_bytes
    )

    return c_tilde == c_tilde_prime


def Verify(
    pk: bytes,
    message: bytes,
    sig: bytes,
    params: MLDSAParams,
    *,
    ctx: bytes = b"",
    pre_hash: str | None = None,
) -> bool:
    """
    ML-DSA.Verify (Algorithm 3).
    """
    if len(ctx) > 255:
        return False
    if len(pk) != params.pk_bytes or len(sig) != params.sig_bytes:
        return False
    try:
        message_prime = FormatMessagePrime(message, ctx, pre_hash=pre_hash)
    except ValueError:
        return False
    return Verify_internal(pk, message_prime, sig, params)


__all__ = ["Verify", "Verify_internal"]
