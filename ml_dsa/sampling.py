"""Sampling và XOF cho ML-DSA (FIPS 204 §7.3, §C).

Bao gồm:
- `SHAKEStream`: bao bọc `hashlib.shake_*` để hỗ trợ "squeeze" tăng dần.
- `RejNTTPoly` / `ExpandA`.
- `RejBoundedPoly` / `ExpandS`.
- `ExpandMask`.
- `SampleInBall`.

Lưu ý: hệ số trả về luôn nằm trong `[0, q)`. Giá trị âm được biểu diễn
qua phép `mod q` (ví dụ -1 → q - 1).
"""

from __future__ import annotations

import hashlib
from typing import List, Optional, Tuple

from ml_dsa.encoding import bit_unpack, integer_to_bytes
from ml_dsa.params import MLDSAParams, N, Q
from ml_dsa.poly import Poly, Vec


class SHAKEStream:
    """Stream bytes từ SHAKE-XOF.

    `hashlib` không hỗ trợ squeeze tăng dần thực sự, nhưng `digest(n)` trên
    cùng một đối tượng XOF luôn trả về cùng tiền tố. Do đó ta giữ một
    buffer nội bộ và mở rộng khi cần thêm dữ liệu.
    """

    def __init__(self, hash_fn, data: bytes):
        self._hash_fn = hash_fn
        self._data = bytes(data)
        self._buffer = b""
        self._pos = 0

    def _ensure(self, end: int) -> None:
        if end > len(self._buffer):
            new_size = max(end, len(self._buffer) * 2, 168)
            self._buffer = self._hash_fn(self._data).digest(new_size)

    def read(self, n: int) -> bytes:
        end = self._pos + n
        self._ensure(end)
        out = self._buffer[self._pos : end]
        self._pos = end
        return out


def _shake256(data: bytes, length: int) -> bytes:
    return hashlib.shake_256(data).digest(length)


def _shake128(data: bytes, length: int) -> bytes:
    return hashlib.shake_128(data).digest(length)


def coeff_from_three_bytes(b0: int, b1: int, b2: int) -> Optional[int]:
    z = b0 + 256 * b1 + 65536 * (b2 & 0x7F)
    if z < Q:
        return z
    return None


def coeff_from_half_byte(b: int, eta: int) -> Optional[int]:
    if eta == 2 and b < 15:
        return 2 - (b % 5)
    if eta == 4 and b < 9:
        return 4 - b
    return None


def rej_ntt_poly(seed: bytes) -> Poly:
    stream = SHAKEStream(hashlib.shake_128, seed)
    out: Poly = []
    while len(out) < N:
        chunk = stream.read(3)
        z = coeff_from_three_bytes(chunk[0], chunk[1], chunk[2])
        if z is not None:
            out.append(z)
    return out


def rej_bounded_poly(seed: bytes, eta: int) -> Poly:
    """Trả về đa thức có hệ số trong `[-η, η]` (dạng balanced).

    Các thao tác hạ nguồn (NTT, vec_add, vec_reduce) đều thực hiện `% Q`
    nên giá trị âm vẫn an toàn. Việc giữ dạng balanced giúp `bit_pack`
    hoạt động đúng theo đặc tả Algorithm 16 (đầu vào ∈ [-η, η]).
    """
    stream = SHAKEStream(hashlib.shake_256, seed)
    out: Poly = []
    while len(out) < N:
        z = stream.read(1)[0]
        z0 = coeff_from_half_byte(z & 0x0F, eta)
        z1 = coeff_from_half_byte(z >> 4, eta)
        if z0 is not None:
            out.append(z0)
        if len(out) < N and z1 is not None:
            out.append(z1)
    return out


def expand_a(rho: bytes, params: MLDSAParams) -> List[List[Poly]]:
    if len(rho) != 32:
        raise ValueError("expand_a: rho must be 32 bytes")
    A: List[List[Poly]] = []
    for r in range(params.k):
        row: List[Poly] = []
        for s in range(params.l):
            seed = rho + bytes([s, r])
            row.append(rej_ntt_poly(seed))
        A.append(row)
    return A


def expand_s(rho: bytes, params: MLDSAParams) -> Tuple[Vec, Vec]:
    if len(rho) != 64:
        raise ValueError("expand_s: rho must be 64 bytes")
    s1: Vec = []
    for r in range(params.l):
        seed = rho + integer_to_bytes(r, 2)
        s1.append(rej_bounded_poly(seed, params.eta))
    s2: Vec = []
    for r in range(params.k):
        seed = rho + integer_to_bytes(r + params.l, 2)
        s2.append(rej_bounded_poly(seed, params.eta))
    return s1, s2


def expand_mask(rho_pp: bytes, mu_counter: int, params: MLDSAParams) -> Vec:
    if len(rho_pp) != 64:
        raise ValueError("expand_mask: rho'' must be 64 bytes")
    c = params.gamma1_bits
    y: Vec = []
    for r in range(params.l):
        n = integer_to_bytes(mu_counter + r, 2)
        v = _shake256(rho_pp + n, 32 * c)
        y.append(bit_unpack(v, params.gamma1 - 1, params.gamma1))
    return y


def sample_in_ball(rho: bytes, params: MLDSAParams) -> Poly:
    if len(rho) != params.c_tilde_bytes:
        raise ValueError(
            f"sample_in_ball: expected {params.c_tilde_bytes} bytes, got {len(rho)}"
        )
    tau = params.tau
    c: Poly = [0] * N
    stream = SHAKEStream(hashlib.shake_256, rho)
    sign_bytes = stream.read(8)
    sign_bits: List[int] = []
    for byte in sign_bytes:
        for k in range(8):
            sign_bits.append((byte >> k) & 1)
    for i in range(N - tau, N):
        while True:
            j = stream.read(1)[0]
            if j <= i:
                break
        c[i] = c[j]
        bit = sign_bits[i + tau - N]
        c[j] = 1 if bit == 0 else Q - 1
    return c


def H_shake256(data: bytes, length: int) -> bytes:
    return _shake256(data, length)


def G_shake128(data: bytes, length: int) -> bytes:
    return _shake128(data, length)


__all__ = [
    "SHAKEStream",
    "coeff_from_three_bytes",
    "coeff_from_half_byte",
    "rej_ntt_poly",
    "rej_bounded_poly",
    "expand_a",
    "expand_s",
    "expand_mask",
    "sample_in_ball",
    "H_shake256",
    "G_shake128",
]
