"""Polynomial arithmetic on R_q = Z_q[X] / (X^n + 1).

Đa thức biểu diễn bằng `list[int]` độ dài 256, các hệ số trong `[0, q)`.
Vector và ma trận dùng `list[Poly]` và `list[list[Poly]]`.

Các hàm trong module này thao tác ở miền chuẩn (không NTT). Nhân đa thức
mặc định dùng schoolbook để dễ kiểm thử; nhân theo NTT nằm ở `ntt.py`.
"""

from __future__ import annotations

from typing import List, Sequence

from ml_dsa.field import center_mod_q, infinity_norm, mod_q
from ml_dsa.params import N, Q

Poly = List[int]
Vec = List[Poly]
Mat = List[List[Poly]]


def zero_poly() -> Poly:
    return [0] * N


def zero_vec(length: int) -> Vec:
    return [zero_poly() for _ in range(length)]


def zero_mat(rows: int, cols: int) -> Mat:
    return [[zero_poly() for _ in range(cols)] for _ in range(rows)]


def poly_copy(a: Poly) -> Poly:
    return list(a)


def vec_copy(v: Vec) -> Vec:
    return [poly_copy(p) for p in v]


def poly_add(a: Poly, b: Poly) -> Poly:
    return [(a[i] + b[i]) % Q for i in range(N)]


def poly_sub(a: Poly, b: Poly) -> Poly:
    return [(a[i] - b[i]) % Q for i in range(N)]


def poly_neg(a: Poly) -> Poly:
    return [(-a[i]) % Q for i in range(N)]


def poly_scalar_mul(a: Poly, c: int) -> Poly:
    return [(a[i] * c) % Q for i in range(N)]


def poly_schoolbook_mul(a: Poly, b: Poly) -> Poly:
    """Nhân `a · b` trong `Z_q[X]/(X^n+1)` bằng schoolbook.

    Dùng để kiểm thử tính đúng của NTT (so khớp kết quả).
    """
    out = [0] * N
    for i in range(N):
        ai = a[i]
        if ai == 0:
            continue
        for j in range(N):
            bj = b[j]
            if bj == 0:
                continue
            k = i + j
            if k < N:
                out[k] = (out[k] + ai * bj) % Q
            else:
                out[k - N] = (out[k - N] - ai * bj) % Q
    return out


def vec_add(u: Vec, v: Vec) -> Vec:
    if len(u) != len(v):
        raise ValueError("vec lengths differ")
    return [poly_add(u[i], v[i]) for i in range(len(u))]


def vec_sub(u: Vec, v: Vec) -> Vec:
    if len(u) != len(v):
        raise ValueError("vec lengths differ")
    return [poly_sub(u[i], v[i]) for i in range(len(u))]


def vec_neg(v: Vec) -> Vec:
    return [poly_neg(p) for p in v]


def vec_scalar_mul(v: Vec, c: int) -> Vec:
    return [poly_scalar_mul(p, c) for p in v]


def vec_infinity_norm(v: Vec) -> int:
    return max((infinity_norm(p) for p in v), default=0)


def poly_reduce(a: Poly) -> Poly:
    return [mod_q(x) for x in a]


def vec_reduce(v: Vec) -> Vec:
    return [poly_reduce(p) for p in v]


def poly_to_centered(a: Poly) -> List[int]:
    return [center_mod_q(x) for x in a]


__all__ = [
    "Poly",
    "Vec",
    "Mat",
    "zero_poly",
    "zero_vec",
    "zero_mat",
    "poly_copy",
    "vec_copy",
    "poly_add",
    "poly_sub",
    "poly_neg",
    "poly_scalar_mul",
    "poly_schoolbook_mul",
    "vec_add",
    "vec_sub",
    "vec_neg",
    "vec_scalar_mul",
    "vec_infinity_norm",
    "poly_reduce",
    "vec_reduce",
    "poly_to_centered",
]
