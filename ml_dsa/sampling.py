"""Sampling và XOF cho ML-DSA (FIPS 204 §7.3, §C).

Bao gồm:

- ``SHAKEStream``: bao bọc ``hashlib.shake_*`` để hỗ trợ "squeeze" tăng dần.
- ``RejNTTPoly`` / ``ExpandA``.
- ``RejBoundedPoly`` / ``ExpandS``.
- ``ExpandMask``.
- ``SampleInBall``.

Lưu ý: hệ số trả về luôn nằm trong ``[0, q)`` (trừ ``rej_bounded_poly``
trả về dạng balanced trong ``[-η, η]`` để phục vụ ``bit_pack``). Giá
trị âm trong miền chuẩn được biểu diễn qua phép ``mod q``
(ví dụ -1 → q - 1).
"""

from __future__ import annotations

import hashlib
from typing import List, Optional, Tuple

from ml_dsa.encoding import bit_unpack, integer_to_bytes
from ml_dsa.params import MLDSAParams, N, Q
from ml_dsa.poly import Poly, Vec


class SHAKEStream:
    """Stream bytes từ SHAKE-XOF.

    ``hashlib`` không hỗ trợ squeeze tăng dần thực sự, nhưng
    ``digest(n)`` trên cùng một đối tượng XOF luôn trả về cùng tiền tố.
    Do đó ta giữ một buffer nội bộ và mở rộng khi cần thêm dữ liệu.

    Cách mở rộng buffer dùng tăng theo cấp số nhân (``len * 2``) với
    sàn 168 byte (khớp với rate của SHAKE-128/256) để giảm số lần gọi
    lại ``digest`` mà vẫn không tốn bộ nhớ vô lý.
    """

    def __init__(self, hash_fn, data: bytes):
        self._hash_fn = hash_fn
        # Bản sao chống ngoài-thay-đổi sau khi stream được khởi tạo.
        self._data = bytes(data)
        self._buffer = b""
        self._pos = 0

    def _ensure(self, end: int) -> None:
        """Đảm bảo buffer đủ dài tới chỉ số ``end`` (re-digest nếu cần)."""
        if end > len(self._buffer):
            new_size = max(end, len(self._buffer) * 2, 168)
            self._buffer = self._hash_fn(self._data).digest(new_size)

    def read(self, n: int) -> bytes:
        """Đọc ``n`` byte tiếp theo từ stream và dịch con trỏ."""
        end = self._pos + n
        self._ensure(end)
        out = self._buffer[self._pos : end]
        self._pos = end
        return out


def _shake256(data: bytes, length: int) -> bytes:
    return hashlib.shake_256(data).digest(length)


def _shake128(data: bytes, length: int) -> bytes:
    return hashlib.shake_128(data).digest(length)


def coeff_from_three_bytes(b0: int, b1: int, b2: int) -> Optional[int]:
    """Diễn giải 3 byte thành hệ số trong ``[0, q)`` (Algorithm 14).

    Bit cao của ``b2`` bị mask để giới hạn ``z < 2^{23}``. Nếu kết quả
    rơi vào ``[q, 2^{23})`` thì trả về ``None`` để bên gọi lấy thử bộ
    byte khác (rejection sampling).
    """
    z = b0 + 256 * b1 + 65536 * (b2 & 0x7F)
    if z < Q:
        return z
    return None


def coeff_from_half_byte(b: int, eta: int) -> Optional[int]:
    """Diễn giải nibble thành hệ số trong ``[-η, η]`` (Algorithm 15).

    Đối với ``η = 2``: dùng các giá trị < 15, ánh xạ qua ``2 - (b mod 5)``
    để cho phân phối đều trên ``[-2, 2]``. Đối với ``η = 4``: dùng
    nibble < 9 và lấy ``4 - b``. Trường hợp khác trả ``None`` (reject).
    """
    if eta == 2 and b < 15:
        return 2 - (b % 5)
    if eta == 4 and b < 9:
        return 4 - b
    return None


def rej_ntt_poly(seed: bytes) -> Poly:
    """Sinh đa thức trong miền NTT bằng rejection sampling từ SHAKE-128.

    Lặp lại đọc 3 byte tới khi thu được ``N`` hệ số hợp lệ. Vì xác suất
    reject ≈ 1/64 mỗi mẫu nên độ dài stream cần thiết là vừa phải.
    """
    stream = SHAKEStream(hashlib.shake_128, seed)
    out: Poly = []
    while len(out) < N:
        chunk = stream.read(3)
        z = coeff_from_three_bytes(chunk[0], chunk[1], chunk[2])
        if z is not None:
            out.append(z)
    return out


def rej_bounded_poly(seed: bytes, eta: int) -> Poly:
    """Trả về đa thức có hệ số trong ``[-η, η]`` (dạng balanced).

    Các thao tác hạ nguồn (NTT, vec_add, vec_reduce) đều thực hiện
    ``% Q`` nên giá trị âm vẫn an toàn. Việc giữ dạng balanced giúp
    ``bit_pack`` hoạt động đúng theo đặc tả Algorithm 16
    (đầu vào ∈ [-η, η]).
    """
    stream = SHAKEStream(hashlib.shake_256, seed)
    out: Poly = []
    while len(out) < N:
        # Mỗi byte cho 2 nibble → tới 2 mẫu, giảm bớt số lần read.
        z = stream.read(1)[0]
        z0 = coeff_from_half_byte(z & 0x0F, eta)
        z1 = coeff_from_half_byte(z >> 4, eta)
        if z0 is not None:
            out.append(z0)
        if len(out) < N and z1 is not None:
            out.append(z1)
    return out


def expand_a(rho: bytes, params: MLDSAParams) -> List[List[Poly]]:
    """ExpandA (Algorithm 32): sinh ma trận ``A`` từ seed ``rho``.

    Mỗi ô ``A[r][s]`` được sinh bằng RejNTTPoly với seed
    ``rho || (s, r)``. Thứ tự byte ``(s, r)`` được giữ đúng đặc tả để
    tương thích với KAT chính thức.
    """
    if len(rho) != 32:
        raise ValueError("expand_a: rho must be 32 bytes")
    A: List[List[Poly]] = []
    for r in range(params.k):
        row: List[Poly] = []
        for s in range(params.l):
            seed = rho + bytes([s, r])
            row.append(rej_ntt_poly(seed))
        A.append(row)
    return A


def expand_s(rho: bytes, params: MLDSAParams) -> Tuple[Vec, Vec]:
    """ExpandS (Algorithm 33): sinh ``s1`` (l đa thức) và ``s2`` (k đa thức).

    Tham số ``rho`` ở đây là ``ρ'`` 64 byte. Mỗi đa thức được seed bằng
    ``ρ' || u`` với ``u`` là chỉ số 2-byte little-endian; ``s1`` chiếm
    chỉ số ``[0, l)`` và ``s2`` chiếm ``[l, l + k)``.
    """
    if len(rho) != 64:
        raise ValueError("expand_s: rho must be 64 bytes")
    s1: Vec = []
    for r in range(params.l):
        seed = rho + integer_to_bytes(r, 2)
        s1.append(rej_bounded_poly(seed, params.eta))
    s2: Vec = []
    for r in range(params.k):
        seed = rho + integer_to_bytes(r + params.l, 2)
        s2.append(rej_bounded_poly(seed, params.eta))
    return s1, s2


def expand_mask(rho_pp: bytes, mu_counter: int, params: MLDSAParams) -> Vec:
    """ExpandMask (Algorithm 34): sinh mặt nạ ``y`` cho vòng ký.

    Mỗi đa thức của ``y`` được tạo từ ``SHAKE-256(rho'' || κ)`` rồi
    ``BitUnpack`` thành các hệ số trong ``[-(γ1 - 1), γ1]``. ``κ``
    được tăng theo bước ``params.l`` ở mỗi lần reject để tránh tái sử
    dụng cùng seed.
    """
    if len(rho_pp) != 64:
        raise ValueError("expand_mask: rho'' must be 64 bytes")
    c = params.gamma1_bits
    y: Vec = []
    for r in range(params.l):
        n = integer_to_bytes(mu_counter + r, 2)
        v = _shake256(rho_pp + n, 32 * c)
        y.append(bit_unpack(v, params.gamma1 - 1, params.gamma1))
    return y


def sample_in_ball(rho: bytes, params: MLDSAParams) -> Poly:
    """SampleInBall (Algorithm 29): sinh challenge ``c`` với τ hệ số ±1.

    Thuật toán Fisher-Yates ngược: với ``i = N - τ .. N - 1`` ta sinh
    ngẫu nhiên ``j ≤ i``, hoán đổi ``c[i] ↔ c[j]`` và gán ``c[j] = ±1``
    theo bit dấu lấy từ 8 byte đầu của stream. Kết quả: đúng τ hệ số
    khác 0 phân bố đều, mỗi hệ số có dấu độc lập.
    """
    if len(rho) != params.c_tilde_bytes:
        raise ValueError(
            f"sample_in_ball: expected {params.c_tilde_bytes} bytes, got {len(rho)}"
        )
    tau = params.tau
    c: Poly = [0] * N
    stream = SHAKEStream(hashlib.shake_256, rho)
    sign_bytes = stream.read(8)
    sign_bits: List[int] = []
    for byte in sign_bytes:
        for k in range(8):
            sign_bits.append((byte >> k) & 1)
    for i in range(N - tau, N):
        # Lấy mẫu ``j`` đến khi ``j ≤ i`` (bỏ qua mẫu vượt giới hạn).
        while True:
            j = stream.read(1)[0]
            if j <= i:
                break
        c[i] = c[j]
        bit = sign_bits[i + tau - N]
        # bit 0 → +1, bit 1 → -1 (biểu diễn -1 trong ``[0, q)`` là q - 1).
        c[j] = 1 if bit == 0 else Q - 1
    return c


def H_shake256(data: bytes, length: int) -> bytes:
    """Hàm băm chính trong ML-DSA (SHAKE-256 squeeze ``length`` byte)."""
    return _shake256(data, length)


def G_shake128(data: bytes, length: int) -> bytes:
    """Hàm băm phụ trợ (SHAKE-128) — chủ yếu phục vụ test/util."""
    return _shake128(data, length)


__all__ = [
    "SHAKEStream",
    "coeff_from_three_bytes",
    "coeff_from_half_byte",
    "rej_ntt_poly",
    "rej_bounded_poly",
    "expand_a",
    "expand_s",
    "expand_mask",
    "sample_in_ball",
    "H_shake256",
    "G_shake128",
]
