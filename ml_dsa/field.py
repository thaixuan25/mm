"""Modular arithmetic on F_q with q = 8380417.

ML-DSA hệ số luôn nằm trong khoảng `[0, q)` ở dạng chuẩn, nhưng nhiều
thuật toán (Decompose, MakeHint, ...) cần biểu diễn "balanced" trong
`(-q/2, q/2]`. Module này cung cấp tiện ích chuyển đổi và thu gọn.
"""

from __future__ import annotations

from ml_dsa.params import Q


def mod_q(x: int) -> int:
    """Chuẩn hóa về `[0, q)`."""
    return x % Q


def center_mod_q(x: int) -> int:
    """Chuẩn hóa về dạng balanced `(-q/2, q/2]`.

    Đây là biểu diễn được dùng trong các kiểm tra `||·||_∞` của FIPS 204.
    """
    r = x % Q
    if r > Q // 2:
        r -= Q
    return r


def mod_pm(x: int, alpha: int) -> int:
    """`x mod^± alpha` theo định nghĩa FIPS 204 §2.3.

    Trả về r ∈ (-α/2, α/2] khi α chẵn, hoặc r ∈ [-(α-1)/2, (α-1)/2] khi α lẻ,
    sao cho x ≡ r (mod α).
    """
    r = x % alpha
    if alpha % 2 == 0:
        if r > alpha // 2:
            r -= alpha
    else:
        if r > (alpha - 1) // 2:
            r -= alpha
    return r


def infinity_norm(coeffs) -> int:
    """`||v||_∞` trên biểu diễn balanced."""
    return max((abs(center_mod_q(c)) for c in coeffs), default=0)


__all__ = ["mod_q", "center_mod_q", "mod_pm", "infinity_norm"]
