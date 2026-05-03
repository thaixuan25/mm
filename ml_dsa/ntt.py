"""Number Theoretic Transform cho ``Z_q[X]/(X^n+1)``.

Cài đặt theo FIPS 204 §B.2 (Algorithm 41/42): in-place Cooley-Tukey ở
miền NTT, dùng bảng ``zetas`` được sinh từ ζ = 1753 và phép đảo bit
8 bit.

NTT(a) trả về vector các giá trị ``a(ζ^{2·BitRev_8(k)+1})`` cho
``k = 0..255``, nhưng được truy cập theo thứ tự bit-reversed nên ta
chỉ cần bảng zetas truyền thống.

Nhân đa thức trong NTT là phép nhân điểm-đến-điểm, do đó
``a · b = INTT(NTT(a) ∘ NTT(b))``. Đây là lý do toàn bộ quy trình ký
được thiết kế để ở miền NTT càng lâu càng tốt.
"""

from __future__ import annotations

from typing import List

from ml_dsa.params import N, Q, ZETA
from ml_dsa.poly import Poly


def _bitrev8(k: int) -> int:
    """Đảo bit của ``k`` trong 8 bit thấp (cho ``k ∈ [0, 256)``).

    Cách viết ``f"{k:08b}"[::-1]`` không hiệu quả nhưng rất rõ ý — bảng
    zetas chỉ tính một lần khi import nên không ảnh hưởng hiệu năng.
    """
    return int(f"{k:08b}"[::-1], 2)


def _build_zetas() -> List[int]:
    """Sinh bảng ``zetas[k] = ζ^{BitRev_8(k)} mod q`` (256 phần tử)."""
    return [pow(ZETA, _bitrev8(i), Q) for i in range(N)]


# Bảng ``zetas`` được tính một lần khi module load và share toàn module.
ZETAS: List[int] = _build_zetas()
# ``N^{-1} mod q`` dùng trong INTT để chia cuối cùng.
N_INV: int = pow(N, -1, Q)


def ntt(a: Poly) -> Poly:
    """Forward NTT (Algorithm 41), trả về list mới (không in-place).

    Thuật toán Cooley-Tukey: bắt đầu với ``length = N/2`` rồi chia đôi
    sau mỗi tầng cho tới khi ``length == 0``. Mỗi tầng quét toàn bộ
    đa thức theo "butterfly" ``(w[j], w[j+length]) → (w[j]+t, w[j]-t)``
    với ``t = z · w[j+length]``.
    """
    w = list(a)
    k = 0
    length = N // 2
    while length >= 1:
        start = 0
        while start < N:
            k += 1
            z = ZETAS[k]
            for j in range(start, start + length):
                t = (z * w[j + length]) % Q
                w[j + length] = (w[j] - t) % Q
                w[j] = (w[j] + t) % Q
            start += 2 * length
        length //= 2
    return w


def intt(a: Poly) -> Poly:
    """Inverse NTT (Algorithm 42), trả về list mới.

    Đối ngẫu của ``ntt``: bắt đầu với ``length = 1`` rồi nhân đôi.
    Cuối cùng nhân toàn bộ với ``N^{-1}`` để hoàn tất phép biến đổi
    nghịch (theo định nghĩa NTT có hệ số chuẩn hoá 1/N).
    """
    w = list(a)
    k = N
    length = 1
    while length < N:
        start = 0
        while start < N:
            k -= 1
            z = (-ZETAS[k]) % Q
            for j in range(start, start + length):
                t = w[j]
                w[j] = (t + w[j + length]) % Q
                w[j + length] = (t - w[j + length]) % Q
                w[j + length] = (z * w[j + length]) % Q
            start += 2 * length
        length *= 2
    return [(x * N_INV) % Q for x in w]


def ntt_pointwise(a: Poly, b: Poly) -> Poly:
    """Nhân điểm-đến-điểm trong miền NTT (Algorithm 45)."""
    return [(a[i] * b[i]) % Q for i in range(N)]


def ntt_vec(v):
    """Áp ``ntt`` lên từng đa thức của vector."""
    return [ntt(p) for p in v]


def intt_vec(v):
    """Áp ``intt`` lên từng đa thức của vector."""
    return [intt(p) for p in v]


def ntt_matvec(A_hat, s_hat):
    """``Â ∘ ŝ`` trong miền NTT.

    ``A_hat`` là ma trận ``k×l``, ``s_hat`` là vector ``l``. Kết quả
    là vector ``k`` với mỗi phần tử là tổng các tích pointwise theo cột.
    Vòng lặp tích luỹ thủ công thay vì dùng ``poly_add`` để tránh tạo
    list trung gian cho mỗi cột.
    """
    k = len(A_hat)
    l = len(s_hat)
    out = []
    for i in range(k):
        acc = [0] * N
        for j in range(l):
            prod = ntt_pointwise(A_hat[i][j], s_hat[j])
            for t in range(N):
                acc[t] = (acc[t] + prod[t]) % Q
        out.append(acc)
    return out


__all__ = [
    "ZETAS",
    "N_INV",
    "ntt",
    "intt",
    "ntt_pointwise",
    "ntt_vec",
    "intt_vec",
    "ntt_matvec",
]
