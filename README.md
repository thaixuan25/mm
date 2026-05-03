# ML-DSA from scratch (Python)

Mô phỏng giáo dục thuật toán chữ ký số hậu lượng tử **ML-DSA** (Module Lattice-Based Digital Signature Algorithm) theo **FIPS 204**, viết hoàn toàn bằng Python thuần (không dùng thư viện PQC bên thứ ba). Ưu tiên đúng đặc tả và dễ đọc, không tối ưu hiệu năng.

> Cảnh báo: đây là code giáo dục/nghiên cứu. KHÔNG dùng cho production. Implementation Python pure không bảo vệ trước side-channel và rất chậm so với cài đặt C/Rust được kiểm chứng.

## Trạng thái

- Cài đặt đầy đủ ba mức an toàn `ML-DSA-44`, `ML-DSA-65`, `ML-DSA-87`.
- Khớp 100% với NIST ACVP test vectors:
  - **KeyGen**: 75/75 (25 mỗi mức).
  - **SigGen**: 90/90 (cả deterministic + hedged, cả 3 mức).
- 92 unit/property/regression tests trong `tests/` đều pass.

## Cấu trúc dự án

```
ml_dsa/
  params.py        # Tham số ML-DSA-44/65/87
  field.py         # Modular arithmetic mod q
  poly.py          # Đa thức trên R_q, vector/matrix
  ntt.py           # NTT/INTT (Algorithm 41/42)
  reduction.py     # Power2Round, Decompose, MakeHint, UseHint
  encoding.py      # BitPack, pkEncode, sigEncode, ...
  sampling.py      # SHAKE wrappers, ExpandA/S/Mask, SampleInBall
  keygen.py        # ML-DSA.KeyGen / KeyGen_internal
  sign.py          # ML-DSA.Sign / Sign_internal
  verify.py        # ML-DSA.Verify / Verify_internal
  api.py           # Wrapper tiện dùng

tests/             # pytest: math, NTT, reduction, encoding, sampling, pipeline, regression
scripts/
  demo.py          # Ví dụ ký/xác minh
  benchmark.py     # Đo thời gian keygen/sign/verify
  validate_kat.py  # Đối chiếu với NIST ACVP test vectors

SPEC.md            # Ánh xạ chi tiết "FIPS 204 thuật toán → Python hàm"
```

## Cài đặt

Yêu cầu Python ≥ 3.10. Chỉ cần `pytest` cho test:

```bash
pip install -r requirements.txt
```

## Sử dụng nhanh

```python
from ml_dsa import ML_DSA_65

pk, sk = ML_DSA_65.keygen()
sig = ML_DSA_65.sign(sk, b"hello", ctx=b"my-app")
assert ML_DSA_65.verify(pk, b"hello", sig, ctx=b"my-app")
```

Tùy chọn:

- `ML_DSA_65.keygen(xi=...)`: seed cố định 32 byte cho keygen deterministic.
- `ML_DSA_65.sign(sk, m, deterministic=True)`: dùng `rnd = 0…0` để chữ ký lặp lại được.
- `ML_DSA_65.sign(sk, m, rnd=os.urandom(32))`: hedged mode rõ ràng.
- `ctx`: chuỗi context ≤ 255 byte, dùng cho domain separation.

## Chạy test

```bash
python -m pytest tests/ -v
```

## Demo & benchmark

```bash
python scripts/demo.py
python scripts/benchmark.py -n 5
```

## Đối chiếu với NIST ACVP KAT

```bash
# Tải test vectors từ ACVP-Server (ví dụ keyGen):
# https://raw.githubusercontent.com/usnistgov/ACVP-Server/master/gen-val/json-files/ML-DSA-keyGen-FIPS204/internalProjection.json
python scripts/validate_kat.py path/to/keyGen.json
python scripts/validate_kat.py path/to/sigGen.json
```

## Tham chiếu thuật toán

Xem [SPEC.md](SPEC.md) để có bảng ánh xạ chi tiết giữa từng Algorithm trong FIPS 204 và hàm Python tương ứng. File này cũng liệt kê toàn bộ tham số và quy ước biểu diễn.

## Hạn chế đã biết

- Không tối ưu cho tốc độ (Python list of int, không SIMD/NumPy).
- Không có biện pháp constant-time / chống side-channel.
- Chưa cài chế độ `HashML-DSA` (pre-hash) — chỉ hỗ trợ chế độ `pure`.
