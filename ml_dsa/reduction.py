"""Decompose / Power2Round / HighBits / LowBits / MakeHint / UseHint.

Theo FIPS 204 §7.4 (Algorithms 35-40). Các hàm thao tác trên số nguyên
trong ``[0, q)``. Tham số ``γ2`` được truyền tường minh để tránh phụ
thuộc ngầm vào bộ tham số.

Ý nghĩa các phép toán:

- ``Power2Round(r) = (r1, r0)`` với ``r = r1·2^d + r0``, dùng để tách
  ``t`` thành ``(t1, t0)`` khi sinh khóa.
- ``Decompose(r) = (r1, r0)`` với ``r = r1·(2γ2) + r0`` (mod q), được
  dùng trong vòng lặp ký để tách ``w``.
- ``MakeHint`` / ``UseHint`` cho phép verifier tái lập ``HighBits``
  từ một xấp xỉ mà không cần ``t0``.
"""

from __future__ import annotations

from typing import List, Tuple

from ml_dsa.field import mod_pm
from ml_dsa.params import D, Q
from ml_dsa.poly import Poly


def power2round(r: int) -> Tuple[int, int]:
    """``Power2Round_q(r)`` → (r1, r0) với ``r0 ∈ (-2^{d-1}, 2^{d-1}]``.

    ``r`` được hiểu là số dư trong ``[0, q)``. Phần thấp ``r0`` lấy
    theo modulo đối xứng (``mod_pm``) để biên độ tối thiểu, còn ``r1``
    là phần được giữ lại trong public key.
    """
    r = r % Q
    r0 = mod_pm(r, 1 << D)
    r1 = (r - r0) >> D
    return r1, r0


def decompose(r: int, gamma2: int) -> Tuple[int, int]:
    """``Decompose_q(r)`` theo FIPS 204 Algorithm 36.

    Có một edge case khi ``r1 · (2γ2)`` đúng bằng ``q - 1``: ta ép
    ``r1 = 0`` để tránh trường hợp số dư vượt quá ``γ2`` sau khi cộng.
    """
    r = r % Q
    r0 = mod_pm(r, 2 * gamma2)
    if r - r0 == Q - 1:
        # Trường hợp biên: gập về 0 và bù 1 cho r0 để giữ ``r = r1·2γ2 + r0``.
        r1 = 0
        r0 -= 1
    else:
        r1 = (r - r0) // (2 * gamma2)
    return r1, r0


def high_bits(r: int, gamma2: int) -> int:
    """Phần ``r1`` của ``Decompose`` (HighBits theo FIPS 204)."""
    return decompose(r, gamma2)[0]


def low_bits(r: int, gamma2: int) -> int:
    """Phần ``r0`` của ``Decompose`` (LowBits theo FIPS 204)."""
    return decompose(r, gamma2)[1]


def make_hint(z: int, r: int, gamma2: int) -> int:
    """Trả về 1 khi cộng thêm ``z`` làm thay đổi ``HighBits`` của ``r``.

    Verifier dùng tổng ``r + z = w - cs2`` ở phía sign; nếu HighBits
    khớp giữa hai phía thì hint = 0, ngược lại = 1.
    """
    r1 = high_bits(r, gamma2)
    v1 = high_bits((r + z) % Q, gamma2)
    return 0 if r1 == v1 else 1


def use_hint(h: int, r: int, gamma2: int) -> int:
    """Tái lập ``HighBits(r + z)`` từ ``r`` và bit hint ``h``.

    Khi hint = 0 ta giữ nguyên ``r1``. Khi hint = 1, ``r1`` được
    điều chỉnh +1 hoặc -1 dựa vào dấu của ``r0`` (modulo m).
    """
    m = (Q - 1) // (2 * gamma2)
    r1, r0 = decompose(r, gamma2)
    if h == 0:
        return r1
    if r0 > 0:
        return (r1 + 1) % m
    return (r1 - 1) % m


def poly_power2round(a: Poly) -> Tuple[Poly, Poly]:
    """Áp ``power2round`` cho từng hệ số của đa thức.

    Trả về ``(a1, a0)`` cùng độ dài với ``a``. Hàm khởi tạo trước rồi
    gán theo chỉ số nhằm tránh chi phí build list bằng list comprehension
    với 256 phần tử lặp hai lần.
    """
    out1: Poly = [0] * len(a)
    out0: Poly = [0] * len(a)
    for i, x in enumerate(a):
        r1, r0 = power2round(x)
        out1[i] = r1
        out0[i] = r0
    return out1, out0


def poly_decompose(a: Poly, gamma2: int) -> Tuple[Poly, Poly]:
    """Áp ``decompose`` cho từng hệ số của đa thức."""
    out1: Poly = [0] * len(a)
    out0: Poly = [0] * len(a)
    for i, x in enumerate(a):
        r1, r0 = decompose(x, gamma2)
        out1[i] = r1
        out0[i] = r0
    return out1, out0


def poly_high_bits(a: Poly, gamma2: int) -> Poly:
    """HighBits cho mỗi hệ số."""
    return [high_bits(x, gamma2) for x in a]


def poly_low_bits(a: Poly, gamma2: int) -> Poly:
    """LowBits cho mỗi hệ số."""
    return [low_bits(x, gamma2) for x in a]


def poly_make_hint(z: Poly, r: Poly, gamma2: int) -> Poly:
    """MakeHint theo từng hệ số (z, r cùng độ dài)."""
    return [make_hint(z[i], r[i], gamma2) for i in range(len(z))]


def poly_use_hint(h: Poly, r: Poly, gamma2: int) -> Poly:
    """UseHint theo từng hệ số."""
    return [use_hint(h[i], r[i], gamma2) for i in range(len(r))]


def vec_power2round(v: List[Poly]) -> Tuple[List[Poly], List[Poly]]:
    """Áp ``poly_power2round`` cho từng đa thức trong vector."""
    v1: List[Poly] = []
    v0: List[Poly] = []
    for p in v:
        a1, a0 = poly_power2round(p)
        v1.append(a1)
        v0.append(a0)
    return v1, v0


def vec_decompose(v: List[Poly], gamma2: int) -> Tuple[List[Poly], List[Poly]]:
    """Áp ``poly_decompose`` cho từng đa thức trong vector."""
    v1: List[Poly] = []
    v0: List[Poly] = []
    for p in v:
        a1, a0 = poly_decompose(p, gamma2)
        v1.append(a1)
        v0.append(a0)
    return v1, v0


def vec_high_bits(v: List[Poly], gamma2: int) -> List[Poly]:
    """HighBits cho cả vector."""
    return [poly_high_bits(p, gamma2) for p in v]


def vec_low_bits(v: List[Poly], gamma2: int) -> List[Poly]:
    """LowBits cho cả vector."""
    return [poly_low_bits(p, gamma2) for p in v]


def vec_make_hint(z: List[Poly], r: List[Poly], gamma2: int) -> List[Poly]:
    """MakeHint cho từng cặp đa thức (z[i], r[i])."""
    return [poly_make_hint(z[i], r[i], gamma2) for i in range(len(z))]


def vec_use_hint(h: List[Poly], r: List[Poly], gamma2: int) -> List[Poly]:
    """UseHint cho cả vector."""
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
