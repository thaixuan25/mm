from __future__ import annotations
import math
from .params import MLDSAParams, Q, N, D, ZETA
from .poly import Poly
from typing import List, Optional, Tuple
from .conversions import (
    BitsToBytes,
    BitsToInteger as _BitsToInteger,
    BytesToBits,
    IntegerToBits,
    IntegerToBytes,
)

def BitsToInteger(y: List[int], alpha: Optional[int] = None) -> int:
    return _BitsToInteger(y, len(y) if alpha is None else alpha)

def bitlen(a: int) -> int:
    return max(1, a.bit_length())

def SimpleBitPack(w: Poly, b: int) -> bytes:
    """Algorithm 16"""
    c = bitlen(b)
    z: List[int] = []
    for i in range(256):
        z.extend(IntegerToBits(w[i], c))
    return BitsToBytes(z)

def BitPack(w: Poly, a: int, b: int) -> bytes:
    """Algorithm 17"""
    c = bitlen(a + b)
    z: List[int] = []
    for i in range(256):
        z.extend(IntegerToBits(b - w[i], c))
    return BitsToBytes(z)

def SimpleBitUnpack(v: bytes, b: int) -> Poly:
    """Algorithm 18"""
    c = bitlen(b)
    if len(v) != 32 * c:
        raise ValueError(f"SimpleBitUnpack: expected {32 * c} bytes, got {len(v)}")
    z = BytesToBits(v)
    w: Poly = []
    for i in range(256):
        w.append(BitsToInteger(z[c * i : c * i + c], c))
    return w

def BitUnpack(v: bytes, a: int, b: int) -> Poly:
    """Algorithm 19"""
    c = bitlen(a + b)
    if len(v) != 32 * c:
        raise ValueError(f"BitUnpack: expected {32 * c} bytes, got {len(v)}")
    z = BytesToBits(v)
    w: Poly = []
    for i in range(256):
        w.append(b - BitsToInteger(z[c * i : c * i + c], c))
    return w

def HintBitPack(h: List[Poly], params: MLDSAParams) -> bytes:
    """Algorithm 20"""
    omega = params.omega
    Index = 0
    y = bytearray(omega + len(h))
    for i in range(len(h)):
        for j in range(256):
            if h[i][j] != 0:
                y[Index] = j
                Index += 1
        y[omega + i] = Index
    return bytes(y)

def HintBitUnpack(y: bytes, params: MLDSAParams) -> List[Poly] | None:
    """Algorithm 21"""
    k = params.k
    omega = params.omega
    if len(y) != omega + k:
        return None
    Index = 0
    h: List[Poly] = [[0] * 256 for _ in range(k)]
    for i in range(k):
        if y[omega + i] < Index or y[omega + i] > omega:
            return None
        First = Index
        while Index < y[omega + i]:
            if Index > First:
                if y[Index - 1] >= y[Index]:
                    return None
            h[i][y[Index]] = 1
            Index += 1
    for i in range(Index, omega):
        if y[i] != 0:
            return None
    return h        
      
def pkEncode(rho: bytes, t1: List[Poly], params: MLDSAParams) -> bytes:
    """Algorithm 22"""
    pk = bytearray(rho)
    k = params.k
    for i in range(k):
        pk.extend(SimpleBitPack(t1[i], pow(2, bitlen(Q - 1) - D) - 1))
    return bytes(pk)

def pkDecode(pk: bytes, params: MLDSAParams) -> Tuple[bytes, List[Poly]]:
    """Algorithm 23"""
    if len(pk) != params.pk_bytes:
        raise ValueError(f"pkDecode: expected {params.pk_bytes} bytes, got {len(pk)}")
    k = params.k
    rho = pk[:32]
    t1: List[Poly] = []
    chunk_size = 32 * (bitlen(Q - 1) - D)
    for i in range(k):
        t1.append(SimpleBitUnpack(pk[32 + i * chunk_size:32 + (i + 1) * chunk_size], pow(2, bitlen(Q - 1) - D) - 1))
    return rho, t1

def skEncode(
        rho: bytes,
        K: bytes,
        tr: bytes,
        s1: List[Poly],
        s2: List[Poly],
        t0: List[Poly],
        params: MLDSAParams
) -> bytes:
    """Algorithm 24"""
    eta = params.eta
    sk = bytearray(rho + K + tr)
    l = len(s1)
    k = len(s2)
    for i in range(l):
        sk.extend(BitPack(s1[i], eta, eta))
    for i in range(k):
        sk.extend(BitPack(s2[i], eta, eta))
    for i in range(k):
        sk.extend(BitPack(t0[i], pow(2, D - 1) - 1, pow(2, D - 1)))
    return bytes(sk)

def skDecode(sk:bytes, params: MLDSAParams) -> Tuple[bytes, bytes, bytes, List[Poly], List[Poly], List[Poly]]:
    """Algorithm 25"""
    if len(sk) != params.sk_bytes:
        raise ValueError(f"skDecode: expected {params.sk_bytes} bytes, got {len(sk)}")
    eta = params.eta
    rho = sk[:32]
    K = sk[32:64]
    tr = sk[64:128]
    s1: List[Poly] = []
    s2: List[Poly] = []
    t0: List[Poly] = []
    chunk_size_s1 = 32 * bitlen(2 * eta)
    chunk_size_s2 = 32 * bitlen(2 * eta)
    chunk_size_t0 = 32 * D
    for i in range(params.l):
        s1.append(BitUnpack(sk[128 + i * chunk_size_s1:128 + (i + 1) * chunk_size_s1], eta, eta))
    for i in range(params.k):
        s2.append(BitUnpack(sk[128 + params.l * chunk_size_s1 + i * chunk_size_s2:128 + params.l * chunk_size_s1 + (i + 1) * chunk_size_s2], eta, eta))
    for i in range(params.k):
        t0.append(BitUnpack(sk[128 + params.l * chunk_size_s1 + params.k * chunk_size_s2 + i * chunk_size_t0:128 + params.l * chunk_size_s1 + params.k * chunk_size_s2 + (i + 1) * chunk_size_t0], pow(2, D - 1) - 1, pow(2, D - 1)))
    return rho, K, tr, s1, s2, t0

def sigEncode(c_tilde: bytes, z: List[Poly], h: List[Poly], params: MLDSAParams) -> bytes:
    """Algorithm 26"""
    gamma1 = params.gamma1
    sigma = bytearray(c_tilde)
    for i in range(params.l):
        sigma.extend(BitPack(z[i], gamma1 - 1, gamma1))
    sigma.extend(HintBitPack(h, params))
    return bytes(sigma)

def sigDecode(sigma: bytes, params: MLDSAParams) -> Tuple[bytes, List[Poly], List[Poly]] | None:
    """Algorithm 27"""
    if len(sigma) != params.sig_bytes:
        return None
    c_tilde = sigma[:params.lam // 4]
    gamma1 = params.gamma1
    z: List[Poly] = []
    chunk_size_z = 32 * (bitlen(gamma1 - 1) + 1)
    y = sigma[params.lam // 4 + 32 * params.l * (bitlen(gamma1 - 1) + 1) : params.lam // 4 + 32 * params.l * (bitlen(gamma1 - 1) + 1) + params.omega + params.k]
    for i in range(params.l):
        z.append(BitUnpack(sigma[params.lam // 4 + i * chunk_size_z : params.lam // 4 + (i + 1) * chunk_size_z], gamma1 - 1, gamma1))
    h = HintBitUnpack(y, params)
    if h is None:
        return None
    return c_tilde, z, h

def w1Encode(w1: List[Poly], params: MLDSAParams) -> bytes:
    """Algorithm 28"""
    w1_tilde = bytearray()
    b = (Q - 1) // (2 * params.gamma2) - 1
    for i in range(params.k):
        w1_tilde.extend(SimpleBitPack(w1[i], b))
    return bytes(w1_tilde)

__all__ = [
    "bitlen",
    "IntegerToBits",
    "BitsToInteger",
    "IntegerToBytes",
    "BitsToBytes",
    "BytesToBits",
    "BitPack",
    "BitUnpack",
    "SimpleBitPack",
    "SimpleBitUnpack",
    "HintBitPack",
    "HintBitUnpack",
    "pkEncode",
    "pkDecode",
    "skEncode",
    "skDecode",
    "sigEncode",
    "sigDecode",
    "w1Encode"
]
