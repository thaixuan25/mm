"""Modular arithmetic on F_q with q = 8380417.

ML-DSA hệ số luôn nằm trong khoảng ``[0, q)`` ở dạng chuẩn, nhưng nhiều
thuật toán (Decompose, MakeHint, ...) cần biểu diễn "balanced" trong
``(-q/2, q/2]``. Module này cung cấp tiện ích chuyển đổi và thu gọn.

Vì Python xử lý số nguyên độ dài tuỳ ý, các hàm dưới đây đơn giản hoá
phép modulo bằng toán tử ``%``; trong cài đặt tối ưu thực thụ ta sẽ
dùng Montgomery hoặc Barrett, nhưng tốc độ không phải mục tiêu của
phiên bản giáo dục này.
"""

from __future__ import annotations

from ml_dsa.params import Q


def mod_q(x: int) -> int:
    """Chuẩn hóa ``x`` về biểu diễn không âm ``[0, q)``.

    Đây là dạng được các hàm encode/NTT yêu cầu. Vì Python ``%`` luôn trả
    về cùng dấu với divisor, ta không cần xử lý trường hợp âm thêm nữa.
    """
    return x % Q


def center_mod_q(x: int) -> int:
    """Chuẩn hóa về dạng balanced ``(-q/2, q/2]``.

    Đây là biểu diễn được dùng trong các kiểm tra ``||·||_∞`` của
    FIPS 204 (vì norm ``∞`` chỉ có ý nghĩa trên dạng đối xứng quanh 0).
    """
    r = x % Q
    if r > Q // 2:
        r -= Q
    return r


def mod_pm(x: int, alpha: int) -> int:
    """``x mod^± alpha`` theo định nghĩa FIPS 204 §2.3.

    Trả về r ∈ (-α/2, α/2] khi α chẵn, hoặc r ∈ [-(α-1)/2, (α-1)/2]
    khi α lẻ, sao cho ``x ≡ r (mod α)``. Đây là phép modulo "đối xứng"
    được dùng trong Decompose và Power2Round.
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
    """``||v||_∞`` trên biểu diễn balanced.

    Hệ số được centered hoá trước khi lấy giá trị tuyệt đối, vì norm
    ``∞`` chỉ có nghĩa khi tính trên dải ``(-q/2, q/2]``. Hàm trả về
    0 khi ``coeffs`` rỗng (tránh ``ValueError`` từ ``max``).
    """
    return max((abs(center_mod_q(c)) for c in coeffs), default=0)


__all__ = ["mod_q", "center_mod_q", "mod_pm", "infinity_norm"]
