from __future__ import annotations

from typing import List, Sequence

from .field import CenterModQ, InfinityNorm, ModQ
from .params import N, Q

Poly = List[int]
Vector = List[Poly]
Matrix = List[List[Poly]]


def ZeroPoly() -> Poly:
    """Trả về đa thức 0 (mọi hệ số bằng 0) trong R_q."""
    return [0] * N


def ZeroVector(length: int) -> Vector:
    """Trả về vector ``length`` đa thức 0."""
    return [ZeroPoly() for _ in range(length)]


def ZeroMatrix(rows: int, cols: int) -> Matrix:
    """Trả về ma trận ``rows × cols`` chứa toàn đa thức 0."""
    return [[ZeroPoly() for _ in range(cols)] for _ in range(rows)]


def PolyCopy(a: Poly) -> Poly:
    """Sao chép đa thức (deep copy ở mức list — int là immutable)."""
    return list(a)


def VectorCopy(v: Vector) -> Vector:
    """Sao chép vector mỗi phần tử bằng ``poly_copy``."""
    return [PolyCopy(p) for p in v]


def PolyAdd(a: Poly, b: Poly) -> Poly:
    """Cộng hai đa thức theo từng hệ số (mod q)."""
    return [(a[i] + b[i]) % Q for i in range(N)]


def PolySub(a: Poly, b: Poly) -> Poly:
    """Trừ ``a - b`` theo từng hệ số (mod q)."""
    return [(a[i] - b[i]) % Q for i in range(N)]


def PolyNeg(a: Poly) -> Poly:
    """Đổi dấu đa thức (giữ trong ``[0, q)``)."""
    return [(-a[i]) % Q for i in range(N)]


def PolyScalarMul(a: Poly, c: int) -> Poly:
    """Nhân đa thức với một hằng số nguyên (mod q)."""
    return [(a[i] * c) % Q for i in range(N)]


def PolySchoolbookMul(a: Poly, b: Poly) -> Poly:
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


def VectorAdd(u: Vector, v: Vector) -> Vector:
    """Cộng hai vector (cùng độ dài) theo từng đa thức."""
    if len(u) != len(v):
        raise ValueError("vec lengths differ")
    return [PolyAdd(u[i], v[i]) for i in range(len(u))]


def VectorSub(u: Vector, v: Vector) -> Vector:
    """Trừ ``u - v`` theo từng đa thức."""
    if len(u) != len(v):
        raise ValueError("vec lengths differ")
    return [PolySub(u[i], v[i]) for i in range(len(u))]


def VectorNeg(v: Vector) -> Vector:
    """Đổi dấu mọi đa thức trong vector."""
    return [PolyNeg(p) for p in v]


def VectorScalarMultiply(v: Vector, c: int) -> Vector:
    """Nhân từng đa thức của vector với hằng số ``c``."""
    return [PolyScalarMul(p, c) for p in v]


def VectorInfinityNorm(v: Vector) -> int:
    """``max ||v_i||_∞`` trên toàn vector (0 nếu vector rỗng)."""
    return max((InfinityNorm(p) for p in v), default=0)


def PolyReduce(a: Poly) -> Poly:
    """Áp ``mod q`` lên từng hệ số để đưa về dạng chuẩn ``[0, q)``."""
    return [ModQ(x) for x in a]


def VectorReduce(v: Vector) -> Vector:
    """Áp ``poly_reduce`` lên từng đa thức của vector."""
    return [PolyReduce(p) for p in v]


def PolyToCentered(a: Poly) -> List[int]:
    """Chuyển từng hệ số về dải ``(-q/2, q/2]`` để chuẩn bị bit-pack."""
    return [CenterModQ(x) for x in a]


__all__ = [
    "Poly",
    "Vector",
    "Matrix",
    "ZeroPoly",
    "ZeroVector",
    "ZeroMatrix",
    "PolyCopy",
    "VectorCopy",
    "PolyAdd",
    "PolySub",
    "PolyNeg",
    "PolyScalarMul",
    "PolySchoolbookMul",
    "VectorAdd",
    "VectorSub",
    "VectorNeg",
    "VectorScalarMultiply",
    "VectorInfinityNorm",
    "PolyReduce",
    "VectorReduce",
    "PolyToCentered",
]
