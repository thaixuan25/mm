"""Bit-level encoding/decoding theo FIPS 204 §8.

Các thuật toán:

- IntegerToBits / BitsToInteger / BitsToBytes / BytesToBits.
- BitPack / BitUnpack (Algorithm 16/17): mã hóa hệ số trong ``[-a, b]``.
- SimpleBitPack / SimpleBitUnpack (Algorithm 18/19): hệ số trong ``[0, b]``.
- HintBitPack / HintBitUnpack (Algorithm 20/21).
- pk/sk/sig encode-decode (Algorithm 22-27), w1Encode (Algorithm 28).

Chú ý: tất cả encoding dùng little-endian theo bit và byte (bit 0 ở
vị trí thấp nhất của byte). Mọi hàm decode đều kiểm tra độ dài đầu
vào, raise ``ValueError`` thay vì truncate hay padding ngầm.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ml_dsa.params import D, MLDSAParams, N, Q
from ml_dsa.poly import Poly, Vec, zero_poly


def _bitlen(x: int) -> int:
    """Số bit dương cần để biểu diễn ``x`` (tối thiểu 1).

    Khác với ``int.bit_length()`` ở chỗ trả về 1 cho ``x = 0`` để dùng
    được làm "số bit/hệ số" trong các hàm BitPack mà không gây ra
    chia cho 0.
    """
    return max(1, x.bit_length())


def integer_to_bits(x: int, alpha: int) -> List[int]:
    """Tách ``x`` thành ``alpha`` bit (LSB → MSB)."""
    return [(x >> i) & 1 for i in range(alpha)]


def bits_to_integer(bits: List[int]) -> int:
    """Hợp các bit (LSB → MSB) thành số nguyên."""
    out = 0
    for i, b in enumerate(bits):
        out |= (b & 1) << i
    return out


def bits_to_bytes(bits: List[int]) -> bytes:
    """Đóng gói danh sách bit thành byte (8 bit/byte, little-endian)."""
    if len(bits) % 8 != 0:
        raise ValueError("bit-string length must be multiple of 8")
    out = bytearray(len(bits) // 8)
    for i, b in enumerate(bits):
        out[i // 8] |= (b & 1) << (i % 8)
    return bytes(out)


def bytes_to_bits(data: bytes) -> List[int]:
    """Mở từng byte thành 8 bit theo thứ tự ngược (LSB trước)."""
    out: List[int] = [0] * (len(data) * 8)
    for i, byte in enumerate(data):
        for j in range(8):
            out[8 * i + j] = (byte >> j) & 1
    return out


def integer_to_bytes(x: int, alpha: int) -> bytes:
    """Đóng gói ``x`` thành ``alpha`` byte little-endian (FIPS 204 §6)."""
    return x.to_bytes(alpha, "little")


def bit_pack(w: Poly, a: int, b: int) -> bytes:
    """BitPack: pack đa thức có hệ số ``∈ [-a, b]`` (Algorithm 16).

    Mỗi hệ số ``w_i`` được dịch sang ``b - w_i`` (luôn ∈ ``[0, a + b]``)
    rồi đóng vào ``c = bitlen(a + b)`` bit.
    """
    c = _bitlen(a + b)
    bits: List[int] = [0] * (N * c)
    for i, wi in enumerate(w):
        v = (b - wi) % (1 << c)
        for j in range(c):
            bits[i * c + j] = (v >> j) & 1
    return bits_to_bytes(bits)


def bit_unpack(z: bytes, a: int, b: int) -> Poly:
    """BitUnpack: nghịch của ``bit_pack`` (Algorithm 17).

    Vì giá trị đã pack là ``b - w_i``, ta tái tạo hệ số bằng ``b - v``.
    """
    c = _bitlen(a + b)
    if len(z) != 32 * c:
        raise ValueError(f"bit_unpack: expected {32*c} bytes, got {len(z)}")
    bits = bytes_to_bits(z)
    w: Poly = [0] * N
    for i in range(N):
        v = 0
        for j in range(c):
            v |= bits[i * c + j] << j
        w[i] = b - v
    return w


def simple_bit_pack(w: Poly, b: int) -> bytes:
    """SimpleBitPack: pack hệ số ∈ ``[0, b]`` (Algorithm 18)."""
    c = _bitlen(b)
    bits: List[int] = [0] * (N * c)
    for i, wi in enumerate(w):
        for j in range(c):
            bits[i * c + j] = (wi >> j) & 1
    return bits_to_bytes(bits)


def simple_bit_unpack(z: bytes, b: int) -> Poly:
    """SimpleBitUnpack: nghịch của ``simple_bit_pack`` (Algorithm 19)."""
    c = _bitlen(b)
    if len(z) != 32 * c:
        raise ValueError(f"simple_bit_unpack: expected {32*c} bytes, got {len(z)}")
    bits = bytes_to_bits(z)
    w: Poly = [0] * N
    for i in range(N):
        v = 0
        for j in range(c):
            v |= bits[i * c + j] << j
        w[i] = v
    return w


def hint_bit_pack(h: Vec, params: MLDSAParams) -> bytes:
    """HintBitPack (Algorithm 20).

    Vì ``h`` là vector thưa, ta lưu các vị trí có bit 1 vào ``omega``
    byte đầu, theo sau là ``k`` byte chỉ ra "ranh giới" giữa các đa
    thức. Cách này cho phép truyền hint với ngân sách cố định
    (``omega + k``) và khôi phục tuần tự khi unpack.
    """
    omega = params.omega
    k = params.k
    out = bytearray(omega + k)
    index = 0
    for i in range(k):
        for j in range(N):
            if h[i][j] != 0:
                if index >= omega:
                    raise ValueError("HintBitPack: too many ones")
                out[index] = j
                index += 1
        out[omega + i] = index
    return bytes(out)


def hint_bit_unpack(y: bytes, params: MLDSAParams) -> Optional[Vec]:
    """HintBitUnpack (Algorithm 21).

    Trả về ``None`` khi format không hợp lệ thay vì raise — điều này
    khớp với hành vi "reject silently" của ``ML_DSA.Verify`` trên đầu
    vào sai dạng. Các kiểm tra:

    1. Độ dài chuỗi đúng ``omega + k`` byte.
    2. Mỗi ranh giới ``end`` không lùi và không vượt ``omega``.
    3. Trong cùng một đa thức, các vị trí phải tăng nghiêm ngặt.
    4. Phần ``omega`` byte còn lại sau ``index`` cuối phải toàn 0.
    """
    omega = params.omega
    k = params.k
    if len(y) != omega + k:
        return None
    h: Vec = [zero_poly() for _ in range(k)]
    index = 0
    for i in range(k):
        end = y[omega + i]
        if end < index or end > omega:
            return None
        first = index
        while index < end:
            if index > first and y[index - 1] >= y[index]:
                return None
            h[i][y[index]] = 1
            index += 1
    for j in range(index, omega):
        if y[j] != 0:
            return None
    return h


def pk_encode(rho: bytes, t1: Vec, params: MLDSAParams) -> bytes:
    """pkEncode (Algorithm 22): public key = ``rho || pack(t1)``."""
    if len(rho) != 32:
        raise ValueError("rho must be 32 bytes")
    bound = (1 << params.t1_bits) - 1
    chunks = [rho]
    for i in range(params.k):
        chunks.append(simple_bit_pack(t1[i], bound))
    return b"".join(chunks)


def pk_decode(pk: bytes, params: MLDSAParams) -> Tuple[bytes, Vec]:
    """pkDecode (Algorithm 23). Raise ValueError nếu sai độ dài."""
    if len(pk) != params.pk_bytes:
        raise ValueError(
            f"pk_decode: expected {params.pk_bytes} bytes, got {len(pk)}"
        )
    rho = pk[:32]
    bound = (1 << params.t1_bits) - 1
    t1: Vec = []
    chunk_size = 32 * params.t1_bits
    offset = 32
    for _ in range(params.k):
        t1.append(simple_bit_unpack(pk[offset : offset + chunk_size], bound))
        offset += chunk_size
    return rho, t1


def sk_encode(
    rho: bytes,
    K: bytes,
    tr: bytes,
    s1: Vec,
    s2: Vec,
    t0: Vec,
    params: MLDSAParams,
) -> bytes:
    """skEncode (Algorithm 24): kết hợp rho/K/tr cùng các vector bí mật.

    ``s1`` và ``s2`` được pack với biên ``[-η, η]``, còn ``t0`` được
    pack với biên ``[-(2^{d-1} - 1), 2^{d-1}]`` để khớp khoảng giá
    trị sau Power2Round.
    """
    if len(rho) != 32 or len(K) != 32 or len(tr) != 64:
        raise ValueError("sk_encode: invalid lengths for rho/K/tr")
    eta = params.eta
    chunks: List[bytes] = [rho, K, tr]
    for i in range(params.l):
        chunks.append(bit_pack(s1[i], eta, eta))
    for i in range(params.k):
        chunks.append(bit_pack(s2[i], eta, eta))
    a = (1 << (D - 1)) - 1
    b = 1 << (D - 1)
    for i in range(params.k):
        chunks.append(bit_pack(t0[i], a, b))
    return b"".join(chunks)


def sk_decode(sk: bytes, params: MLDSAParams) -> Tuple[bytes, bytes, bytes, Vec, Vec, Vec]:
    """skDecode (Algorithm 25). Raise ValueError nếu sai độ dài."""
    if len(sk) != params.sk_bytes:
        raise ValueError(
            f"sk_decode: expected {params.sk_bytes} bytes, got {len(sk)}"
        )
    eta = params.eta
    eta_bits = _bitlen(2 * eta)
    rho = sk[:32]
    K = sk[32:64]
    tr = sk[64:128]
    offset = 128
    s_chunk = 32 * eta_bits
    s1: Vec = []
    for _ in range(params.l):
        s1.append(bit_unpack(sk[offset : offset + s_chunk], eta, eta))
        offset += s_chunk
    s2: Vec = []
    for _ in range(params.k):
        s2.append(bit_unpack(sk[offset : offset + s_chunk], eta, eta))
        offset += s_chunk
    a = (1 << (D - 1)) - 1
    b = 1 << (D - 1)
    t0_chunk = 32 * D
    t0: Vec = []
    for _ in range(params.k):
        t0.append(bit_unpack(sk[offset : offset + t0_chunk], a, b))
        offset += t0_chunk
    return rho, K, tr, s1, s2, t0


def sig_encode(c_tilde: bytes, z: Vec, h: Vec, params: MLDSAParams) -> bytes:
    """sigEncode (Algorithm 26): chữ ký = ``c_tilde || pack(z) || pack(h)``."""
    if len(c_tilde) != params.c_tilde_bytes:
        raise ValueError("sig_encode: invalid c_tilde length")
    a = params.gamma1 - 1
    b = params.gamma1
    chunks: List[bytes] = [c_tilde]
    for i in range(params.l):
        chunks.append(bit_pack(z[i], a, b))
    chunks.append(hint_bit_pack(h, params))
    return b"".join(chunks)


def sig_decode(sig: bytes, params: MLDSAParams) -> Optional[Tuple[bytes, Vec, Vec]]:
    """sigDecode (Algorithm 27).

    Trả ``None`` khi sai độ dài hoặc hint không hợp lệ — verifier dựa
    vào dấu hiệu này để từ chối chữ ký mà không raise.
    """
    if len(sig) != params.sig_bytes:
        return None
    a = params.gamma1 - 1
    b = params.gamma1
    c_tilde = sig[: params.c_tilde_bytes]
    offset = params.c_tilde_bytes
    z_chunk = 32 * params.gamma1_bits
    z: Vec = []
    for _ in range(params.l):
        z.append(bit_unpack(sig[offset : offset + z_chunk], a, b))
        offset += z_chunk
    h = hint_bit_unpack(sig[offset:], params)
    if h is None:
        return None
    return c_tilde, z, h


def w1_encode(w1: Vec, params: MLDSAParams) -> bytes:
    """w1Encode (Algorithm 28): pack vector HighBits cho phép băm Fiat-Shamir.

    ``bound = (q - 1)/(2γ2) - 1`` chính là giá trị tối đa của HighBits,
    nên ``simple_bit_pack`` luôn an toàn.
    """
    bound = (Q - 1) // (2 * params.gamma2) - 1
    chunks: List[bytes] = []
    for i in range(params.k):
        chunks.append(simple_bit_pack(w1[i], bound))
    return b"".join(chunks)


__all__ = [
    "integer_to_bits",
    "bits_to_integer",
    "bits_to_bytes",
    "bytes_to_bits",
    "integer_to_bytes",
    "bit_pack",
    "bit_unpack",
    "simple_bit_pack",
    "simple_bit_unpack",
    "hint_bit_pack",
    "hint_bit_unpack",
    "pk_encode",
    "pk_decode",
    "sk_encode",
    "sk_decode",
    "sig_encode",
    "sig_decode",
    "w1_encode",
]
