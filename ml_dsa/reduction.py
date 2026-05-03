"""Decompose / Power2Round / HighBits / LowBits / MakeHint / UseHint.

Theo FIPS 204 §7.4 (Algorithms 35-40). Các hàm thao tác trên số nguyên
trong `[0, q)`. Tham số `γ2` được truyền tường minh để tránh phụ thuộc
ngầm vào bộ tham số.
"""

from __future__ import annotations

from typing import List, Tuple

from ml_dsa.field import mod_pm
from ml_dsa.params import D, Q
from ml_dsa.poly import Poly


def power2round(r: int) -> Tuple[int, int]:
    """`Power2Round_q(r)` → (r1, r0) với r0 ∈ (-2^{d-1}, 2^{d-1}]."""
    r = r % Q
    r0 = mod_pm(r, 1 << D)
    r1 = (r - r0) >> D
    return r1, r0


def decompose(r: int, gamma2: int) -> Tuple[int, int]:
    """`Decompose_q(r)` theo FIPS 204 Algorithm 36."""
    r = r % Q
    r0 = mod_pm(r, 2 * gamma2)
    if r - r0 == Q - 1:
        r1 = 0
        r0 -= 1
    else:
        r1 = (r - r0) // (2 * gamma2)
    return r1, r0


def high_bits(r: int, gamma2: int) -> int:
    return decompose(r, gamma2)[0]


def low_bits(r: int, gamma2: int) -> int:
    return decompose(r, gamma2)[1]


def make_hint(z: int, r: int, gamma2: int) -> int:
    r1 = high_bits(r, gamma2)
    v1 = high_bits((r + z) % Q, gamma2)
    return 0 if r1 == v1 else 1


def use_hint(h: int, r: int, gamma2: int) -> int:
    m = (Q - 1) // (2 * gamma2)
    r1, r0 = decompose(r, gamma2)
    if h == 0:
        return r1
    if r0 > 0:
        return (r1 + 1) % m
    return (r1 - 1) % m


def poly_power2round(a: Poly) -> Tuple[Poly, Poly]:
    out1: Poly = [0] * len(a)
    out0: Poly = [0] * len(a)
    for i, x in enumerate(a):
        r1, r0 = power2round(x)
        out1[i] = r1
        out0[i] = r0
    return out1, out0


def poly_decompose(a: Poly, gamma2: int) -> Tuple[Poly, Poly]:
    out1: Poly = [0] * len(a)
    out0: Poly = [0] * len(a)
    for i, x in enumerate(a):
        r1, r0 = decompose(x, gamma2)
        out1[i] = r1
        out0[i] = r0
    return out1, out0


def poly_high_bits(a: Poly, gamma2: int) -> Poly:
    return [high_bits(x, gamma2) for x in a]


def poly_low_bits(a: Poly, gamma2: int) -> Poly:
    return [low_bits(x, gamma2) for x in a]


def poly_make_hint(z: Poly, r: Poly, gamma2: int) -> Poly:
    return [make_hint(z[i], r[i], gamma2) for i in range(len(z))]


def poly_use_hint(h: Poly, r: Poly, gamma2: int) -> Poly:
    return [use_hint(h[i], r[i], gamma2) for i in range(len(r))]


def vec_power2round(v: List[Poly]) -> Tuple[List[Poly], List[Poly]]:
    v1: List[Poly] = []
    v0: List[Poly] = []
    for p in v:
        a1, a0 = poly_power2round(p)
        v1.append(a1)
        v0.append(a0)
    return v1, v0


def vec_decompose(v: List[Poly], gamma2: int) -> Tuple[List[Poly], List[Poly]]:
    v1: List[Poly] = []
    v0: List[Poly] = []
    for p in v:
        a1, a0 = poly_decompose(p, gamma2)
        v1.append(a1)
        v0.append(a0)
    return v1, v0


def vec_high_bits(v: List[Poly], gamma2: int) -> List[Poly]:
    return [poly_high_bits(p, gamma2) for p in v]


def vec_low_bits(v: List[Poly], gamma2: int) -> List[Poly]:
    return [poly_low_bits(p, gamma2) for p in v]


def vec_make_hint(z: List[Poly], r: List[Poly], gamma2: int) -> List[Poly]:
    return [poly_make_hint(z[i], r[i], gamma2) for i in range(len(z))]


def vec_use_hint(h: List[Poly], r: List[Poly], gamma2: int) -> List[Poly]:
    return [poly_use_hint(h[i], r[i], gamma2) for i in range(len(r))]


__all__ = [
    "power2round",
    "decompose",
    "high_bits",
    "low_bits",
    "make_hint",
    "use_hint",
    "poly_power2round",
    "poly_decompose",
    "poly_high_bits",
    "poly_low_bits",
    "poly_make_hint",
    "poly_use_hint",
    "vec_power2round",
    "vec_decompose",
    "vec_high_bits",
    "vec_low_bits",
    "vec_make_hint",
    "vec_use_hint",
]
