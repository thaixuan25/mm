from typing import List
from .params import N, Q, ZETA
from .poly import Poly

def _BITREV8(n: int) -> int:
    """Algorithm 43"""
    return int('{:08b}'.format(n)[::-1], 2)

def _precompute_zetas() -> list[int]:
    zetas = [0] * 256
    for m in range(256):
        zetas[m] = pow(ZETA, _BITREV8(m), Q)
    return zetas

ZETAS = _precompute_zetas()

def NTT(w: Poly) -> Poly:
    """
    Algorithm 41
    """
    w = [x % Q for x in w]
    m = 0
    length = 128
    while length >= 1:
        start = 0
        while start < 256:
            m += 1
            z = ZETAS[m]
            for j in range(start, start + length):
                t = (z * w[j + length]) % Q
                
                w[j + length] = (w[j] - t) % Q
                
                w[j] = (w[j] + t) % Q
                
            start += 2 * length
        length //= 2
        
    return w

def INTT(w: Poly) -> Poly:
    """
    Algorithm 42
    """
    w = [x % Q for x in w]
    m = 256
    length = 1
    while length < 256:
        start = 0
        while start < 256:
            m -= 1
            z = (-ZETAS[m]) % Q
            for j in range(start, start + length):
                t = w[j]
                
                w[j] = (t + w[j + length]) % Q
                
                w[j + length] = (t - w[j + length]) % Q
                
                w[j + length] = (z * w[j + length]) % Q
                
            start += 2 * length
        length *= 2
    
    F = 8347681
    for j in range(256):
        w[j] = (w[j] * F) % Q
        
    return w

def AddNTT(a: Poly, b: Poly) -> Poly:
    """
    Algorithm 44
    """
    out: Poly = [0] * N
    for i in range(N):
        out[i] = (a[i] + b[i]) % Q
    return out

def MultiplyNTT(a: Poly, b: Poly) -> Poly:
    """
    Algorithm 45
    """
    out: Poly = [0] * N
    for i in range(N):
        out[i] = (a[i] * b[i]) % Q
    return out

def AddVectorNTT(a: List[Poly], b: List[Poly]) -> List[Poly]:
    """Algorithm 46"""
    out: List[Poly] = [0] * len(a)
    for i in range(len(a)):
        out[i] = AddNTT(a[i], b[i])
    return out

def ScalarVectorNTT(c: Poly, v: List[Poly]) -> List[Poly]:
    """Algorithm 47"""
    out: List[Poly] = [0] * len(v)
    for i in range(len(v)):
        out[i] = MultiplyNTT(c, v[i])
    return out

def MatrixVectorNTT(M: List[List[Poly]], v: List[Poly]) -> List[Poly]:
    """Algorithm 48"""
    out: List[Poly] = [[0] * N for _ in range(len(M))]
    for i in range(len(M)):
        for j in range(len(v)):
            out[i] = AddNTT(out[i], MultiplyNTT(M[i][j], v[j]))
    return out

__all__ = [
    "ZETAS",
    "NTT",
    "INTT",
    "AddNTT",
    "MultiplyNTT",
    "AddVectorNTT",
    "ScalarVectorNTT",
    "MatrixVectorNTT"
]
