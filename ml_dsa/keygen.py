"""ML-DSA.KeyGen / KeyGen_internal (FIPS 204 §6.1, Algorithm 1 & 6).

Quy trình sinh khóa gồm 3 giai đoạn lớn:

1. Mở rộng seed ``xi`` thành ``(rho, rho', K)`` qua SHAKE-256.
2. Sinh ma trận công khai ``A`` từ ``rho`` và bộ vector bí mật
   ``(s1, s2)`` từ ``rho'``, sau đó tính ``t = A·s1 + s2`` (mod q).
3. Phân tách ``t = t1·2^d + t0``, đóng gói ``pk = (rho, t1)`` và
   ``sk = (rho, K, tr, s1, s2, t0)`` theo §8.2.

Toàn bộ phép nhân ma trận-vector được thực hiện trong miền NTT để giữ
chi phí ở mức O(n log n).
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

from ml_dsa.encoding import pk_encode, sk_encode
from ml_dsa.ntt import intt_vec, ntt_matvec, ntt_vec
from ml_dsa.params import MLDSAParams
from ml_dsa.poly import vec_add, vec_reduce
from ml_dsa.reduction import vec_power2round
from ml_dsa.sampling import H_shake256, expand_a, expand_s


def keygen_internal(xi: bytes, params: MLDSAParams) -> Tuple[bytes, bytes]:
    """KeyGen_internal (Algorithm 6) với seed ``xi`` xác định.

    Parameters
    ----------
    xi:
        32 byte seed. Trong chế độ thường, ``xi`` là output của TRNG; đối
        với KAT, ``xi`` được chỉ định trước.
    params:
        Bộ tham số ML-DSA tương ứng (44/65/87).

    Returns
    -------
    (pk, sk):
        Public key đã encode (bytes) và secret key đã encode (bytes).
    """
    if len(xi) != 32:
        raise ValueError("keygen_internal: xi must be 32 bytes")

    # Mở rộng seed: thêm domain separator (k, l) trước khi băm để tránh
    # va chạm seed giữa các bộ tham số khác nhau.
    expanded = H_shake256(xi + bytes([params.k, params.l]), 128)
    rho = expanded[:32]
    rho_prime = expanded[32:96]
    K = expanded[96:128]

    A_hat = expand_a(rho, params)
    s1, s2 = expand_s(rho_prime, params)

    # Nhân A·s1 trong miền NTT rồi đưa về miền chuẩn để cộng s2.
    s1_hat = ntt_vec(s1)
    As_hat = ntt_matvec(A_hat, s1_hat)
    As = intt_vec(As_hat)
    t = vec_reduce(vec_add(As, s2))

    # Tách phần cao/thấp của t: chỉ t1 đi vào public key (giảm 13 bit/hệ số),
    # còn t0 nằm trong secret key để khử "noise" khi verify.
    t1, t0 = vec_power2round(t)

    pk = pk_encode(rho, t1, params)
    # tr = H(pk) được dùng làm "label" của public key trong các bước sau.
    tr = H_shake256(pk, 64)
    sk = sk_encode(rho, K, tr, s1, s2, t0, params)
    return pk, sk


def keygen(
    params: MLDSAParams, *, xi: Optional[bytes] = None
) -> Tuple[bytes, bytes]:
    """Sinh cặp khóa ML-DSA.

    Khi ``xi`` không được truyền, hàm tự lấy 32 byte ngẫu nhiên từ
    ``os.urandom`` (CSPRNG do hệ điều hành cung cấp).
    """
    if xi is None:
        xi = os.urandom(32)
    return keygen_internal(xi, params)


__all__ = ["keygen", "keygen_internal"]
