import math
from typing import List, Optional, Tuple
# Hằng số chuẩn của ML-DSA (Mục 2.3)
from .params import Q as q

def bitlen(a: int) -> int:
    """Ký hiệu Mục 2.3: bitlen(a)"""
    return a.bit_length()

def IntegerToBits(x: int, alpha: int) -> List[int]:
    """Algorithm 9: IntegerToBits(x, alpha)"""
    y = [0] * alpha
    for i in range(alpha):
        y[i] = x % 2
        x = x // 2
    return y

def BitsToInteger(y: List[int], alpha: int) -> int:
    """Algorithm 10: BitsToInteger(y, alpha)"""
    x = 0
    # Tài liệu dùng chỉ số 1 đến alpha, ta dùng range(1, alpha + 1)
    for i in range(1, alpha + 1):
        x = 2 * x + y[alpha - i]
    return x

def IntegerToBytes(x: int, alpha: int) -> bytes:
    """Algorithm 11: IntegerToBytes(x, alpha)"""
    y = bytearray(alpha)
    for i in range(alpha):
        y[i] = x % 256
        x = x // 256
    return bytes(y)

def BitsToBytes(y: List[int]) -> bytes:
    """Algorithm 12: BitsToBytes(y)"""
    alpha = len(y)
    z = bytearray(math.ceil(alpha / 8))
    for i in range(alpha):
        # z[floor(i/8)] <- z[floor(i/8)] + y[i] * 2^(i mod 8)
        z[i // 8] += y[i] * (2 ** (i % 8))
    return bytes(z)

def BytesToBits(z: bytes) -> List[int]:
    """Algorithm 13: BytesToBits(z)"""
    alpha = len(z)
    y = [0] * (8 * alpha)
    for i in range(alpha):
        z_prime_i = z[i]
        for j in range(8):
            y[8 * i + j] = z_prime_i % 2
            z_prime_i = z_prime_i // 2
    return y

def CoeffFromThreeBytes(b0: int, b1: int, b2: int) -> int | None:
    """Algorithm 14: CoeffFromThreeBytes(b0, b1, b2)"""
    b2_prime = b2
    if b2_prime > 127:
        b2_prime -= 128
    z_val = (2**16) * b2_prime + (2**8) * b1 + b0
    if z_val < q:
        return z_val
    return None

def CoeffFromHalfByte(b: int, eta: int) -> int | None:
    """Algorithm 15: CoeffFromHalfByte(b)"""
    if eta == 2 and b < 15:
        return 2 - (b % 5)
    elif eta == 4 and b < 9:
        return 4 - b
    return None

__all__ = [
    "bitlen",
    "IntegerToBits",
    "BitsToInteger",
    "IntegerToBytes",
    "BitsToBytes",
    "BytesToBits",
    "CoeffFromThreeBytes",
    "CoeffFromHalfByte"
]
