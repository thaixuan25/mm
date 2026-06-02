"""ML-DSA parameter sets (FIPS 204, Section 4 Table 1).

Mọi giá trị byte-size đều được tính lại bằng công thức trong FIPS 204
để tránh hard-code sai. Các hằng số toàn cục q, n, d, ζ là cố định cho
mọi mức an toàn.

Nếu có bộ tham số mới (ví dụ test vector), chỉ cần khai báo thêm một
``MLDSAParams(...)`` mới — các property byte-size sẽ tự suy ra đúng.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2

# Modulus chính của ML-DSA: q = 2^23 - 2^13 + 1, là số nguyên tố và
# tương thích NTT trên Z_q[X]/(X^256 + 1).
Q: int = 8380417
# Độ dài đa thức (cũng là số hệ số trong R_q).
N: int = 256
# Số bit dùng cho Power2Round: t = t1·2^d + t0.
D: int = 13
# Căn nguyên thuỷ bậc 512 của 1 trong F_q, dùng để dựng bảng zetas.
ZETA: int = 1753


def _bitlen(x: int) -> int:
    """Số bit cần để biểu diễn ``x`` (yêu cầu ``x > 0``).

    Khác với ``int.bit_length()`` ở chỗ raise rõ ràng khi ``x ≤ 0`` để
    bắt sớm các tham số sai (ví dụ ``a + b == 0`` trong bit-pack).
    """
    if x <= 0:
        raise ValueError("bitlen requires positive integer")
    return x.bit_length()


@dataclass(frozen=True)
class MLDSAParams:
    """Bộ tham số cho một mức an toàn ML-DSA.

    Các trường ý nghĩa:

    - ``k``, ``l``: kích thước ma trận ``A ∈ R_q^{k×l}`` (hàng × cột).
    - ``eta``: biên hệ số của ``s1, s2`` (∈ ``[-η, η]``).
    - ``tau``: số hệ số khác 0 trong challenge ``c`` (SampleInBall).
    - ``beta = tau · eta``: ngưỡng dùng trong rejection sampling.
    - ``gamma1``: biên cho mặt nạ ``y`` (cũng là biên cho ``z``).
    - ``gamma2``: biên dùng trong Decompose / HighBits / LowBits.
    - ``omega``: giới hạn số bit hint = 1 trên toàn vector.
    - ``lam``: tham số an toàn (bit). Quyết định độ dài ``c_tilde``.
    """

    name: str
    k: int
    l: int
    eta: int
    tau: int
    beta: int
    gamma1: int
    gamma2: int
    omega: int
    lam: int

    @property
    def q(self) -> int:
        """Modulus chung của ML-DSA (8380417)."""
        return Q

    @property
    def n(self) -> int:
        """Bậc đa thức (256) — cố định cho mọi bộ tham số."""
        return N

    @property
    def d(self) -> int:
        """Số bit Power2Round (13)."""
        return D

    @property
    def c_tilde_bytes(self) -> int:
        """Độ dài commitment ``c_tilde`` theo byte (lam/4)."""
        return self.lam // 4

    @property
    def t1_bits(self) -> int:
        """Số bit cần để pack mỗi hệ số của ``t1``."""
        return _bitlen(Q - 1) - D

    @property
    def t0_bits(self) -> int:
        """Số bit cần để pack mỗi hệ số của ``t0`` (chính là D)."""
        return D

    @property
    def gamma1_bits(self) -> int:
        """Số bit cần để pack mỗi hệ số của ``z`` (lệch bởi ``gamma1``)."""
        return _bitlen(self.gamma1 - 1) + 1

    @property
    def w1_bits(self) -> int:
        """Số bit cần để pack mỗi hệ số ``w1`` (output của HighBits)."""
        return _bitlen((Q - 1) // (2 * self.gamma2) - 1)

    @property
    def pk_bytes(self) -> int:
        """Tổng số byte sau ``pk_encode`` (32 cho rho + ``t1`` đã pack)."""
        return 32 + 32 * self.k * self.t1_bits

    @property
    def sk_bytes(self) -> int:
        """Tổng số byte sau ``sk_encode``.

        Bao gồm: rho (32) + K (32) + tr (64) + s1, s2 đã pack theo
        ``eta_bits`` + t0 đã pack theo ``D``.
        """
        eta_bits = _bitlen(2 * self.eta)
        return (
            32
            + 32
            + 64
            + 32 * ((self.k + self.l) * eta_bits + self.k * D)
        )

    @property
    def sig_bytes(self) -> int:
        """Tổng số byte sau ``sig_encode``.

        Gồm: c_tilde + z (theo ``gamma1_bits``) + hint header (omega+k).
        """
        return (
            self.c_tilde_bytes
            + 32 * self.l * self.gamma1_bits
            + self.omega
            + self.k
        )


# Mức an toàn ~Category 2: cân bằng nhất giữa kích thước và tốc độ.
ML_DSA_44_PARAMS = MLDSAParams(
    name="ML-DSA-44",
    k=4,
    l=4,
    eta=2,
    tau=39,
    beta=78,
    gamma1=2 ** 17,
    gamma2=(Q - 1) // 88,
    omega=80,
    lam=128,
)

# Mức an toàn ~Category 3: lựa chọn khuyến nghị cho ứng dụng chung.
ML_DSA_65_PARAMS = MLDSAParams(
    name="ML-DSA-65",
    k=6,
    l=5,
    eta=4,
    tau=49,
    beta=196,
    gamma1=2 ** 19,
    gamma2=(Q - 1) // 32,
    omega=55,
    lam=192,
)

# Mức an toàn ~Category 5: cấu hình bảo mật cao nhất.
ML_DSA_87_PARAMS = MLDSAParams(
    name="ML-DSA-87",
    k=8,
    l=7,
    eta=2,
    tau=60,
    beta=120,
    gamma1=2 ** 19,
    gamma2=(Q - 1) // 32,
    omega=75,
    lam=256,
)


PARAMS_BY_NAME = {
    p.name: p for p in (ML_DSA_44_PARAMS, ML_DSA_65_PARAMS, ML_DSA_87_PARAMS)
}


__all__ = [
    "Q",
    "N",
    "D",
    "ZETA",
    "MLDSAParams",
    "ML_DSA_44_PARAMS",
    "ML_DSA_65_PARAMS",
    "ML_DSA_87_PARAMS",
    "PARAMS_BY_NAME",
]
