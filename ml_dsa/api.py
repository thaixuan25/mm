"""High-level API tiện dùng cho ML-DSA-44 / 65 / 87."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

from ml_dsa.keygen import keygen as _keygen, keygen_internal as _keygen_internal
from ml_dsa.params import (
    ML_DSA_44_PARAMS,
    ML_DSA_65_PARAMS,
    ML_DSA_87_PARAMS,
    MLDSAParams,
)
from ml_dsa.sign import sign as _sign, sign_internal as _sign_internal
from ml_dsa.verify import verify as _verify, verify_internal as _verify_internal


@dataclass(frozen=True)
class ML_DSA:
    """Wrapper bám 1 bộ tham số ML-DSA cụ thể."""

    params: MLDSAParams

    @property
    def name(self) -> str:
        return self.params.name

    @property
    def pk_bytes(self) -> int:
        return self.params.pk_bytes

    @property
    def sk_bytes(self) -> int:
        return self.params.sk_bytes

    @property
    def sig_bytes(self) -> int:
        return self.params.sig_bytes

    def keygen(self, *, xi: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        return _keygen(self.params, xi=xi)

    def keygen_internal(self, xi: bytes) -> Tuple[bytes, bytes]:
        return _keygen_internal(xi, self.params)

    def sign(
        self,
        sk: bytes,
        message: bytes,
        *,
        ctx: bytes = b"",
        deterministic: bool = False,
        rnd: Optional[bytes] = None,
    ) -> bytes:
        return _sign(
            sk,
            message,
            self.params,
            ctx=ctx,
            deterministic=deterministic,
            rnd=rnd,
        )

    def sign_internal(self, sk: bytes, message_prime: bytes, rnd: bytes) -> bytes:
        return _sign_internal(sk, message_prime, rnd, self.params)

    def verify(
        self, pk: bytes, message: bytes, sig: bytes, *, ctx: bytes = b""
    ) -> bool:
        return _verify(pk, message, sig, self.params, ctx=ctx)

    def verify_internal(
        self, pk: bytes, message_prime: bytes, sig: bytes
    ) -> bool:
        return _verify_internal(pk, message_prime, sig, self.params)


ML_DSA_44 = ML_DSA(ML_DSA_44_PARAMS)
ML_DSA_65 = ML_DSA(ML_DSA_65_PARAMS)
ML_DSA_87 = ML_DSA(ML_DSA_87_PARAMS)


__all__ = ["ML_DSA", "ML_DSA_44", "ML_DSA_65", "ML_DSA_87"]
