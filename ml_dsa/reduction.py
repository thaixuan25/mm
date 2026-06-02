from __future__ import annotations
from .params import D as d
from .params import Q as q
from typing import List, Tuple
from .poly import Poly

def Power2Round(r: int) -> Tuple[int, int]:
    """Algorithm 35"""
    r_plus = r % q
    r0 = r_plus & ((1 << d) - 1)
    if r0 > (1 << (d - 1)):
        r0 -= (1 << d)
    r1 = (r_plus - r0) >> d
    return r1, r0

def Decompose(r: int, gamma2: int) -> Tuple[int, int]:
    """Algorithm 36"""
    r_plus = r % q
    r0 = r_plus % (2 * gamma2)
    if r0 > gamma2:
        r0 -= 2 * gamma2
    if r_plus - r0 == q - 1:
        r1 = 0
        r0 -= 1
    else: r1 = (r_plus - r0) // (2 * gamma2)
    return r1, r0

def HighBits(r: int, gamma2: int) -> int:
    """Algorithm 37"""
    return Decompose(r, gamma2)[0]

def LowBits(r: int, gamma2: int) -> int:
    """Algorithm 38"""
    return Decompose(r, gamma2)[1]

def MakeHint(z: int, r: int, gamma2: int) -> bool:
    """Algorithm 39"""
    r1 = HighBits(r, gamma2)
    v1 = HighBits(r + z, gamma2)
    if r1 == v1:
        return False
    else:        
        return True
    
def UseHint(h: bool, r: int, gamma2: int) -> int:
    """Algorithm 40"""
    m = (q - 1) // (2 * gamma2)
    [r1, r0] = Decompose(r, gamma2)
    if h == 1 and r0 > 0:
        return (r1 + 1) % m
    elif h == 1 and r0 <= 0:
        return (r1 - 1) % m
    return r1

def PolyPower2Round(a: Poly) -> Tuple[Poly, Poly]:
    out1: Poly = [0] * len(a)
    out0: Poly = [0] * len(a)
    for i, x in enumerate(a):
        [r1, r0] = Power2Round(x)
        out1[i] = r1
        out0[i] = r0
    return out1, out0

def PolyDecompose(a: Poly, gamma2: int) -> Tuple[Poly, Poly]:
    out1: Poly = [0] * len(a)
    out0: Poly = [0] * len(a)
    for i, x in enumerate(a):
        [r1, r0] = Decompose(x, gamma2)
        out1[i] = r1
        out0[i] = r0
    return out1, out0

def PolyHighBits(a: Poly, gamma2: int) -> Poly:
    return [HighBits(x, gamma2) for x in a]

def PolyLowBits(a: Poly, gamma2: int) -> Poly:
    return [LowBits(x, gamma2) for x in a]

def PolyMakeHint(z: Poly, r: Poly, gamma2: int) -> Poly:
    out: Poly = [False] * len(z)
    for i in range(len(z)):
        out[i] = MakeHint(z[i], r[i], gamma2)
    return out

def PolyUseHint(h: Poly, r: Poly, gamma2: int) -> Poly:
    out: Poly = [0] * len(r)
    for i in range(len(r)):
        out[i] = UseHint(h[i], r[i], gamma2)
    return out

def VectorPower2Round(v: List[Poly]) -> Tuple[List[Poly], List[Poly]]:
    v1: List[Poly] = []
    v0: List[Poly] = []
    for p in v:
        a1, a0 = PolyPower2Round(p)
        v1.append(a1)
        v0.append(a0)
    return v1, v0

def VectorDecompose(v: List[Poly], gamma2: int) -> Tuple[List[Poly], List[Poly]]:
    v1: List[Poly] = []
    v0: List[Poly] = []
    for p in v:
        a1, a0 = PolyDecompose(p, gamma2)
        v1.append(a1)
        v0.append(a0)
    return v1, v0

def VectorHighBits(v: List[Poly], gamma2: int) -> List[Poly]:
    return [PolyHighBits(p, gamma2) for p in v]

def VectorLowBits(v: List[Poly], gamma2: int) -> List[Poly]:
    return [PolyLowBits(p, gamma2) for p in v]

def VectorMakeHint(z: List[Poly], r: List[Poly], gamma2: int) -> List[Poly]:
    out: List[Poly] = []
    for i in range(len(z)):
        out.append(PolyMakeHint(z[i], r[i], gamma2))
    return out

def VectorUseHint(h: List[Poly], r: List[Poly], gamma2: int) -> List[Poly]:
    out: List[Poly] = []
    for i in range(len(r)):
        out.append(PolyUseHint(h[i], r[i], gamma2))
    return out

__all__ = [
    "Power2Round",
    "PolyDecompose",
    "PolyHighBits",
    "PolyLowBits",
    "MakeHint",
    "UseHint",
    "PolyPower2Round",
    "PolyDecompose",
    "PolyHighBits",
    "PolyLowBits",
    "PolyMakeHint",
    "PolyUseHint",
    "VectorPower2Round",
    "VectorDecompose",
    "VectorHighBits",
    "VectorLowBits",
    "VectorMakeHint",
    "VectorUseHint"
]
