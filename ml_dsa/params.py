"""ML-DSA parameter sets (FIPS 204, Section 4 Table 1).

Mọi giá trị byte-size đều được tính lại bằng công thức trong FIPS 204
để tránh hard-code sai. Các hằng số toàn cục q, n, d, ζ là cố định cho
mọi mức an toàn.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2

Q: int = 8380417
N: int = 256
D: int = 13
ZETA: int = 1753


def _bitlen(x: int) -> int:
    if x <= 0:
        raise ValueError("bitlen requires positive integer")
    return x.bit_length()


@dataclass(frozen=True)
class MLDSAParams:
    """Bộ tham số cho một mức an toàn ML-DSA."""

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
        return Q

    @property
    def n(self) -> int:
        return N

    @property
    def d(self) -> int:
        return D

    @property
    def c_tilde_bytes(self) -> int:
        return self.lam // 4

    @property
    def t1_bits(self) -> int:
        return _bitlen(Q - 1) - D

    @property
    def t0_bits(self) -> int:
        return D

    @property
    def gamma1_bits(self) -> int:
        return _bitlen(self.gamma1 - 1) + 1

    @property
    def w1_bits(self) -> int:
        return _bitlen((Q - 1) // (2 * self.gamma2) - 1)

    @property
    def pk_bytes(self) -> int:
        return 32 + 32 * self.k * self.t1_bits

    @property
    def sk_bytes(self) -> int:
        eta_bits = _bitlen(2 * self.eta)
        return (
            32
            + 32
            + 64
            + 32 * ((self.k + self.l) * eta_bits + self.k * D)
        )

    @property
    def sig_bytes(self) -> int:
        return (
            self.c_tilde_bytes
            + 32 * self.l * self.gamma1_bits
            + self.omega
            + self.k
        )


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
