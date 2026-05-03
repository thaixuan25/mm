"""ML-DSA.Sign / Sign_internal (FIPS 204 §6.2, Algorithm 2 & 7).

Hàm ký dùng kỹ thuật "Fiat-Shamir with aborts": lặp lại quá trình sinh
mặt nạ ``y`` tới khi ``z = y + c·s1`` và ``r0 = LowBits(w - c·s2)`` đủ
nhỏ; nếu không đủ nhỏ, lặp lại với ``kappa`` mới (tăng theo bội số
``params.l``). Cách lặp này giữ cho phân phối ``z`` không tiết lộ
``s1`` (zero-knowledge).
"""

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
    """Áp NTT lên từng đa thức của một vector."""
    return ntt_vec(vec)


def _intt_each(vec):
    """Áp NTT-1 lên từng đa thức của một vector."""
    return intt_vec(vec)


def _hat_pointwise_with_scalar_hat(scalar_hat, vec_hat):
    """Nhân ``c_hat`` (một đa thức) lần lượt với từng phần tử của vector NTT.

    Đây là cách diễn đạt "scalar-vector multiplication" trong miền NTT:
    ``(c · v)_i = c · v_i`` cho mọi ``i``.
    """
    return [ntt_pointwise(scalar_hat, p) for p in vec_hat]


def _vec_neg(v):
    """Đổi dấu mọi hệ số của vector (vẫn giữ trong ``[0, q)``)."""
    return [[(-x) % Q for x in p] for p in v]


def _hint_popcount(h):
    """Đếm số bit hint = 1 trên toàn vector (giới hạn bởi ``omega``)."""
    return sum(sum(x for x in p) for p in h)


def sign_internal(
    sk: bytes, message_prime: bytes, rnd: bytes, params: MLDSAParams
) -> bytes:
    """Sign_internal (Algorithm 7).

    Parameters
    ----------
    sk:
        Secret key đã encode (bytes), độ dài ``params.sk_bytes``.
    message_prime:
        Thông điệp đã được tiền xử lý theo Algorithm 2 (gồm domain
        separator + context + message). Phải dùng cùng tiền xử lý với
        bên xác thực.
    rnd:
        32 byte ngẫu nhiên (sử dụng ``b"\\x00" * 32`` cho deterministic).
    params:
        Bộ tham số ML-DSA tương ứng.
    """
    if len(rnd) != 32:
        raise ValueError("sign_internal: rnd must be 32 bytes")

    rho, K, tr, s1, s2, t0 = sk_decode(sk, params)

    # Đưa các vector bí mật sang miền NTT một lần để dùng trong vòng lặp.
    s1_hat = _ntt_each(s1)
    s2_hat = _ntt_each(s2)
    t0_hat = _ntt_each(t0)
    A_hat = expand_a(rho, params)

    # mu là "label" của thông điệp (kết hợp tr với message_prime).
    mu = H_shake256(tr + message_prime, 64)
    # rho'' phục vụ làm seed cho mặt nạ y; trộn cả K (per-key) lẫn rnd
    # (per-signature) đảm bảo mỗi chữ ký dùng phân phối y khác nhau.
    rho_pp = H_shake256(K + rnd + mu, 64)

    # `kappa` là counter cho ExpandMask. Tăng theo bội ``params.l`` mỗi
    # lần reject để tránh tái sử dụng cùng seed cho y.
    kappa = 0
    while True:
        # Bước 1: sinh mặt nạ y và w = A·y, sau đó lấy phần cao của w.
        y = expand_mask(rho_pp, kappa, params)
        y_hat = _ntt_each(y)
        Ay_hat = ntt_matvec(A_hat, y_hat)
        w = vec_reduce(_intt_each(Ay_hat))
        w1 = vec_high_bits(w, params.gamma2)

        # Bước 2: rút thử thách c từ băm của (mu, w1).
        c_tilde = H_shake256(mu + w1_encode(w1, params), params.c_tilde_bytes)
        c = sample_in_ball(c_tilde, params)
        c_hat = ntt(c)

        # Bước 3: tính z = y + c·s1 và r0 = LowBits(w - c·s2).
        cs1_hat = _hat_pointwise_with_scalar_hat(c_hat, s1_hat)
        cs2_hat = _hat_pointwise_with_scalar_hat(c_hat, s2_hat)
        cs1 = vec_reduce(_intt_each(cs1_hat))
        cs2 = vec_reduce(_intt_each(cs2_hat))

        z = vec_reduce(vec_add(y, cs1))
        r0 = vec_low_bits(vec_reduce(vec_sub(w, cs2)), params.gamma2)

        # Reject nếu z hoặc r0 vượt ngưỡng — đây là điều kiện zero-knowledge,
        # ngăn rò rỉ thông tin về s1/s2 qua phân phối chữ ký.
        if (
            vec_infinity_norm(z) >= params.gamma1 - params.beta
            or max(abs(x) for p in r0 for x in p) >= params.gamma2 - params.beta
        ):
            kappa += params.l
            continue

        # Bước 4: tính c·t0 để tạo hint, đồng thời kiểm tra ||c·t0||_∞.
        ct0_hat = _hat_pointwise_with_scalar_hat(c_hat, t0_hat)
        ct0 = vec_reduce(_intt_each(ct0_hat))

        if vec_infinity_norm(ct0) >= params.gamma2:
            kappa += params.l
            continue

        # MakeHint mã hoá sự khác biệt giữa HighBits(w - cs2) và
        # HighBits(w - cs2 + ct0) để verifier phục hồi w1 mà không cần t0.
        neg_ct0 = _vec_neg(ct0)
        r_for_hint = vec_reduce(
            vec_add(vec_sub(w, cs2), ct0)
        )
        h = vec_make_hint(neg_ct0, r_for_hint, params.gamma2)

        # Reject nếu hint quá thưa-quá dày: số bit 1 phải ≤ omega để khớp
        # giới hạn dung lượng trong sig_encode/sig_decode.
        if _hint_popcount(h) > params.omega:
            kappa += params.l
            continue

        # z được centered hoá trước khi pack để giữ tính đối xứng quanh 0.
        z_centered = [poly_to_centered(p) for p in z]
        return sig_encode(c_tilde, z_centered, h, params)


def _format_message_prime(message: bytes, ctx: bytes) -> bytes:
    """Bọc thông điệp theo Algorithm 2 (domain separator = 0)."""
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

    Parameters
    ----------
    sk:
        Secret key đã encode.
    message:
        Thông điệp gốc.
    params:
        Bộ tham số ML-DSA.
    ctx:
        Chuỗi context tuỳ chọn (≤ 255 byte) giúp domain separation giữa
        các giao thức/ứng dụng dùng chung khoá.
    deterministic:
        Khi True, dùng rnd toàn 0 — chữ ký sẽ lặp lại nếu thông điệp và
        khoá giống nhau. Hữu ích cho test KAT.
    rnd:
        Override rnd 32 byte cụ thể; ưu tiên hơn cờ ``deterministic``.
    """
    message_prime = _format_message_prime(message, ctx)
    if rnd is None:
        rnd = b"\x00" * 32 if deterministic else os.urandom(32)
    return sign_internal(sk, message_prime, rnd, params)


__all__ = ["sign", "sign_internal"]
