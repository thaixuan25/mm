"""High-level API tiện dùng cho ML-DSA-44 / 65 / 87.

Module này gói gọn các hàm `keygen`, `sign`, `verify` ở mức thấp lại
thành một class `ML_DSA` gắn cứng vào một bộ tham số cụ thể, giúp người
dùng không phải truyền `params` ở mỗi lời gọi.

Ba instance hằng (`ML_DSA_44`, `ML_DSA_65`, `ML_DSA_87`) được expose
sẵn để dùng ngay cho từng mức an toàn theo FIPS 204.
"""

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
    """Wrapper bám 1 bộ tham số ML-DSA cụ thể.

    Đối tượng này không lưu trạng thái mật mã nào (mọi state đều nằm trong
    `sk` hoặc `pk`), nên có thể tái sử dụng cho nhiều phiên ký/xác thực
    song song một cách an toàn.
    """

    params: MLDSAParams

    @property
    def name(self) -> str:
        """Tên ngắn của bộ tham số (ví dụ ``"ML-DSA-65"``)."""
        return self.params.name

    @property
    def pk_bytes(self) -> int:
        """Kích thước public key (byte) sau khi `pk_encode`."""
        return self.params.pk_bytes

    @property
    def sk_bytes(self) -> int:
        """Kích thước secret key (byte) sau khi `sk_encode`."""
        return self.params.sk_bytes

    @property
    def sig_bytes(self) -> int:
        """Kích thước signature (byte) sau khi `sig_encode`."""
        return self.params.sig_bytes

    def keygen(self, *, xi: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """ML-DSA.KeyGen (Algorithm 1).

        Tham số ``xi`` (32 byte) cho phép truyền seed tất định để tái lập
        kết quả; nếu bỏ trống sẽ lấy entropy từ ``os.urandom``.
        """
        return _keygen(self.params, xi=xi)

    def keygen_internal(self, xi: bytes) -> Tuple[bytes, bytes]:
        """KeyGen_internal (Algorithm 6) — luôn xác định theo ``xi``.

        Dùng cho test KAT cần tái lập từng bước theo seed cố định.
        """
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
        """ML-DSA.Sign (Algorithm 2).

        - ``ctx``: chuỗi context tối đa 255 byte cho domain separation.
        - ``deterministic``: nếu True thì dùng rnd toàn 0 (chế độ xác định).
        - ``rnd``: ép giá trị 32 byte cụ thể, ưu tiên hơn ``deterministic``.
        """
        return _sign(
            sk,
            message,
            self.params,
            ctx=ctx,
            deterministic=deterministic,
            rnd=rnd,
        )

    def sign_internal(self, sk: bytes, message_prime: bytes, rnd: bytes) -> bytes:
        """Sign_internal (Algorithm 7).

        ``message_prime`` đã được tiền xử lý theo Algorithm 2 (gắn domain
        separator + context). Hàm này thường chỉ được gọi từ KAT/test.
        """
        return _sign_internal(sk, message_prime, rnd, self.params)

    def verify(
        self, pk: bytes, message: bytes, sig: bytes, *, ctx: bytes = b""
    ) -> bool:
        """ML-DSA.Verify (Algorithm 3).

        Trả về ``True`` khi chữ ký hợp lệ. Mọi sai lệch độ dài hoặc context
        > 255 byte đều bị từ chối ngay (không raise) để tránh side-channel.
        """
        return _verify(pk, message, sig, self.params, ctx=ctx)

    def verify_internal(
        self, pk: bytes, message_prime: bytes, sig: bytes
    ) -> bool:
        """Verify_internal (Algorithm 8) — phiên bản nhận ``message_prime``."""
        return _verify_internal(pk, message_prime, sig, self.params)


ML_DSA_44 = ML_DSA(ML_DSA_44_PARAMS)
ML_DSA_65 = ML_DSA(ML_DSA_65_PARAMS)
ML_DSA_87 = ML_DSA(ML_DSA_87_PARAMS)


__all__ = ["ML_DSA", "ML_DSA_44", "ML_DSA_65", "ML_DSA_87"]
