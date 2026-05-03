"""Polynomial arithmetic on R_q = Z_q[X] / (X^n + 1).

Đa thức biểu diễn bằng ``list[int]`` độ dài 256, các hệ số trong
``[0, q)``. Vector và ma trận dùng ``list[Poly]`` và
``list[list[Poly]]``.

Các hàm trong module này thao tác ở miền chuẩn (không NTT). Nhân đa
thức mặc định dùng schoolbook để dễ kiểm thử; nhân theo NTT nằm ở
``ntt.py``.

Quy ước về cấu trúc dữ liệu:

- ``Poly``: ``list[int]`` đúng ``N = 256`` phần tử.
- ``Vec``: danh sách ``Poly``, độ dài tuỳ vector (``k`` hoặc ``l``).
- ``Mat``: danh sách hàng, mỗi hàng là một ``Vec``.
"""

from __future__ import annotations

from typing import List, Sequence

from ml_dsa.field import center_mod_q, infinity_norm, mod_q
from ml_dsa.params import N, Q

Poly = List[int]
Vec = List[Poly]
Mat = List[List[Poly]]


def zero_poly() -> Poly:
    """Trả về đa thức 0 (mọi hệ số bằng 0) trong R_q."""
    return [0] * N


def zero_vec(length: int) -> Vec:
    """Trả về vector ``length`` đa thức 0."""
    return [zero_poly() for _ in range(length)]


def zero_mat(rows: int, cols: int) -> Mat:
    """Trả về ma trận ``rows × cols`` chứa toàn đa thức 0."""
    return [[zero_poly() for _ in range(cols)] for _ in range(rows)]


def poly_copy(a: Poly) -> Poly:
    """Sao chép đa thức (deep copy ở mức list — int là immutable)."""
    return list(a)


def vec_copy(v: Vec) -> Vec:
    """Sao chép vector mỗi phần tử bằng ``poly_copy``."""
    return [poly_copy(p) for p in v]


def poly_add(a: Poly, b: Poly) -> Poly:
    """Cộng hai đa thức theo từng hệ số (mod q)."""
    return [(a[i] + b[i]) % Q for i in range(N)]


def poly_sub(a: Poly, b: Poly) -> Poly:
    """Trừ ``a - b`` theo từng hệ số (mod q)."""
    return [(a[i] - b[i]) % Q for i in range(N)]


def poly_neg(a: Poly) -> Poly:
    """Đổi dấu đa thức (giữ trong ``[0, q)``)."""
    return [(-a[i]) % Q for i in range(N)]


def poly_scalar_mul(a: Poly, c: int) -> Poly:
    """Nhân đa thức với một hằng số nguyên (mod q)."""
    return [(a[i] * c) % Q for i in range(N)]


def poly_schoolbook_mul(a: Poly, b: Poly) -> Poly:
    """Nhân ``a · b`` trong ``Z_q[X]/(X^n+1)`` bằng schoolbook.

    Dùng để kiểm thử tính đúng của NTT (so khớp kết quả). Khi tích
    rơi vào bậc ≥ N, ta áp ``X^N = -1`` để gập lại — đây là lý do
    nhánh ``out[k - N] -= ai * bj`` xuất hiện.
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
                # Bậc ≥ N: X^N = -1 → đảo dấu rồi gộp về vị trí (k - N).
                out[k - N] = (out[k - N] - ai * bj) % Q
    return out


def vec_add(u: Vec, v: Vec) -> Vec:
    """Cộng hai vector (cùng độ dài) theo từng đa thức."""
    if len(u) != len(v):
        raise ValueError("vec lengths differ")
    return [poly_add(u[i], v[i]) for i in range(len(u))]


def vec_sub(u: Vec, v: Vec) -> Vec:
    """Trừ ``u - v`` theo từng đa thức."""
    if len(u) != len(v):
        raise ValueError("vec lengths differ")
    return [poly_sub(u[i], v[i]) for i in range(len(u))]


def vec_neg(v: Vec) -> Vec:
    """Đổi dấu mọi đa thức trong vector."""
    return [poly_neg(p) for p in v]


def vec_scalar_mul(v: Vec, c: int) -> Vec:
    """Nhân từng đa thức của vector với hằng số ``c``."""
    return [poly_scalar_mul(p, c) for p in v]


def vec_infinity_norm(v: Vec) -> int:
    """``max ||v_i||_∞`` trên toàn vector (0 nếu vector rỗng)."""
    return max((infinity_norm(p) for p in v), default=0)


def poly_reduce(a: Poly) -> Poly:
    """Áp ``mod q`` lên từng hệ số để đưa về dạng chuẩn ``[0, q)``."""
    return [mod_q(x) for x in a]


def vec_reduce(v: Vec) -> Vec:
    """Áp ``poly_reduce`` lên từng đa thức của vector."""
    return [poly_reduce(p) for p in v]


def poly_to_centered(a: Poly) -> List[int]:
    """Chuyển từng hệ số về dải ``(-q/2, q/2]`` để chuẩn bị bit-pack."""
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
