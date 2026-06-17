from __future__ import annotations

import os
from typing import Optional, Tuple

from .encoding import pkEncode, skEncode
from .ntt import INTT, MatrixVectorNTT, NTT
from .params import MLDSAParams
from .poly import VectorAdd, VectorReduce
from .reduction import VectorPower2Round
from .sampling import HShake256, ExpandA, ExpandS


def KeyGen_internal(xi: bytes, params: MLDSAParams) -> Tuple[bytes, bytes]:
    """
    KeyGen_internal (Algorithm 6) với seed ``xi``
    """
    if len(xi) != 32:
        raise ValueError("keygen_internal: xi must be 32 bytes")

    # Mở rộng seed: thêm domain separator (k, l) trước khi băm để tránh
    # va chạm seed giữa các bộ tham số khác nhau.
    expanded = HShake256(xi + bytes([params.k, params.l]), 128)
    rho = expanded[:32]
    rho_prime = expanded[32:96]
    K = expanded[96:128]

    A_hat = ExpandA(rho, params)
    s1, s2 = ExpandS(rho_prime, params)

    # Nhân A·s1 trong miền NTT rồi đưa về miền chuẩn để cộng s2.
    s1_hat = [NTT(p) for p in s1]
    As_hat = MatrixVectorNTT(A_hat, s1_hat)
    As = [INTT(p) for p in As_hat]
    t = VectorReduce(VectorAdd(As, s2))

    # Tách phần cao/thấp của t: chỉ t1 đi vào public key (giảm 13 bit/hệ số),
    # còn t0 nằm trong secret key để khử "noise" khi verify.
    t1, t0 = VectorPower2Round(t)

    pk = pkEncode(rho, t1, params)
    # tr = H(pk) được dùng làm "label" của public key trong các bước sau.
    tr = HShake256(pk, 64)
    sk = skEncode(rho, K, tr, s1, s2, t0, params)
    return pk, sk


def KeyGen(
    params: MLDSAParams, *, xi: Optional[bytes] = None
) -> Tuple[bytes, bytes]:
    """
    Sinh cặp khóa ML-DSA.
    """
    if xi is None:
        xi = os.urandom(32)
    return KeyGen_internal(xi, params)


__all__ = ["KeyGen", "KeyGen_internal"]
