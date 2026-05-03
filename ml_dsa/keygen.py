"""ML-DSA.KeyGen / KeyGen_internal (FIPS 204 §6.1, Algorithm 1 & 6)."""

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
    """KeyGen_internal (Algorithm 6) với seed `ξ` xác định.

    Trả về `(pk, sk)`.
    """
    if len(xi) != 32:
        raise ValueError("keygen_internal: xi must be 32 bytes")

    expanded = H_shake256(xi + bytes([params.k, params.l]), 128)
    rho = expanded[:32]
    rho_prime = expanded[32:96]
    K = expanded[96:128]

    A_hat = expand_a(rho, params)
    s1, s2 = expand_s(rho_prime, params)

    s1_hat = ntt_vec(s1)
    As_hat = ntt_matvec(A_hat, s1_hat)
    As = intt_vec(As_hat)
    t = vec_reduce(vec_add(As, s2))

    t1, t0 = vec_power2round(t)

    pk = pk_encode(rho, t1, params)
    tr = H_shake256(pk, 64)
    sk = sk_encode(rho, K, tr, s1, s2, t0, params)
    return pk, sk


def keygen(
    params: MLDSAParams, *, xi: Optional[bytes] = None
) -> Tuple[bytes, bytes]:
    """Sinh cặp khóa ML-DSA. Khi `xi` không truyền sẽ lấy từ `os.urandom`."""
    if xi is None:
        xi = os.urandom(32)
    return keygen_internal(xi, params)


__all__ = ["keygen", "keygen_internal"]
