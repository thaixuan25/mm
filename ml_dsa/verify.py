"""ML-DSA.Verify / Verify_internal (FIPS 204 §6.3, Algorithm 3 & 8).

Verifier tái lập ``w1'`` từ ``z`` và ``c`` rồi kiểm tra
``H(mu, w1') == c_tilde``. Vì verifier không có ``t0``, hint ``h``
được dùng để bù phần sai lệch nhỏ giữa ``HighBits(w - c·s2)`` và
``HighBits(A·z - c·t1·2^d)``.
"""

from __future__ import annotations

from ml_dsa.encoding import pk_decode, sig_decode, w1_encode
from ml_dsa.ntt import intt_vec, ntt_matvec, ntt_pointwise, ntt_vec, ntt
from ml_dsa.params import D, MLDSAParams, Q
from ml_dsa.poly import vec_infinity_norm, vec_reduce, vec_sub
from ml_dsa.reduction import vec_use_hint
from ml_dsa.sampling import H_shake256, expand_a, sample_in_ball


def _scale_t1(t1, factor):
    """Nhân từng hệ số của vector ``t1`` với hằng số (mod q).

    Trong verify ta cần ``t1 · 2^d`` để đối chiếu với ``A·z``, vì t1 là
    phần "high" sau Power2Round.
    """
    return [[(x * factor) % Q for x in p] for p in t1]


def _hint_popcount(h):
    """Đếm số bit hint = 1 toàn vector."""
    return sum(sum(p) for p in h)


def verify_internal(
    pk: bytes, message_prime: bytes, sig: bytes, params: MLDSAParams
) -> bool:
    """Verify_internal (Algorithm 8).

    Trả về True khi cả ba điều kiện sau cùng thỏa mãn:

    1. ``sig`` decode hợp lệ và có popcount(h) ≤ omega.
    2. ``||z||_∞ < γ1 - β``.
    3. ``H(mu || w1Encode(UseHint(h, A·z - c·t1·2^d)))`` trùng với
       ``c_tilde`` trong chữ ký.
    """
    decoded_sig = sig_decode(sig, params)
    if decoded_sig is None:
        return False
    c_tilde, z, h = decoded_sig

    if _hint_popcount(h) > params.omega:
        return False

    if vec_infinity_norm(z) >= params.gamma1 - params.beta:
        return False

    rho, t1 = pk_decode(pk, params)
    A_hat = expand_a(rho, params)
    tr = H_shake256(pk, 64)
    mu = H_shake256(tr + message_prime, 64)

    c = sample_in_ball(c_tilde, params)
    c_hat = ntt(c)

    # Tính A·z - c·t1·2^d trong miền NTT để khớp với cách signer dựng w.
    z_hat = ntt_vec(z)
    Az_hat = ntt_matvec(A_hat, z_hat)

    t1_scaled = _scale_t1(t1, 1 << D)
    t1_scaled_hat = ntt_vec(t1_scaled)
    ct1_hat = [ntt_pointwise(c_hat, p) for p in t1_scaled_hat]

    diff_hat = [
        [(Az_hat[i][j] - ct1_hat[i][j]) % Q for j in range(len(Az_hat[i]))]
        for i in range(len(Az_hat))
    ]
    w_approx = vec_reduce(intt_vec(diff_hat))

    # UseHint phục hồi ``w1`` xấp xỉ; nếu hint chuẩn từ signer thì kết quả
    # sẽ trùng với w1 ban đầu, từ đó c_tilde tái sinh đúng.
    w1_prime = vec_use_hint(h, w_approx, params.gamma2)
    c_tilde_prime = H_shake256(
        mu + w1_encode(w1_prime, params), params.c_tilde_bytes
    )

    return c_tilde == c_tilde_prime


def _format_message_prime(message: bytes, ctx: bytes) -> bytes:
    """Tiền xử lý thông điệp giống bên ký để băm cho ra cùng ``mu``."""
    if len(ctx) > 255:
        raise ValueError("ctx exceeds 255 bytes")
    return bytes([0, len(ctx)]) + ctx + message


def verify(
    pk: bytes,
    message: bytes,
    sig: bytes,
    params: MLDSAParams,
    *,
    ctx: bytes = b"",
) -> bool:
    """ML-DSA.Verify (Algorithm 3).

    Hàm này từ chối thay vì raise khi gặp dữ liệu sai dạng (sai độ dài
    pk/sig hoặc context > 255 byte) để hành vi xác thực luôn trả về bool
    đơn giản, tránh tạo nhánh ngoại lệ ảnh hưởng tới timing.
    """
    if len(ctx) > 255:
        return False
    if len(pk) != params.pk_bytes or len(sig) != params.sig_bytes:
        return False
    message_prime = _format_message_prime(message, ctx)
    return verify_internal(pk, message_prime, sig, params)


__all__ = ["verify", "verify_internal"]
