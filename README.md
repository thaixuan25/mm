# MI4100 - ML-DSA Demo

Repository này cài đặt và minh họa thuật toán chữ ký số **ML-DSA** theo FIPS
204. Project tập trung vào ba pipeline chính:

- `KeyGen`: sinh public key `pk` và secret key `sk`.
- `Sign`: dùng `sk` để ký message và tạo signature `sig`.
- `Verify`: dùng `pk`, message và `sig` để kiểm tra chữ ký.

Ngoài phần cài đặt thuật toán, repo có giao diện Streamlit để demo KeyGen,
Sign, Verify; bộ test tự động; script benchmark; script validate KAT; và tài
liệu thuyết trình/bảo vệ.

> Đây là project học thuật/demo. Bộ test hiện tại chứng minh tính tự nhất quán
> và một phần conformance khi có KAT vector, nhưng không đồng nghĩa với chứng
> nhận chính thức theo NIST.

## Tính Năng Chính

- Hỗ trợ 3 parameter set: `ML_DSA_44`, `ML_DSA_65`, `ML_DSA_87`.
- API cấp cao stateless qua class `ML_DSA`.
- Public API nhận/trả bytes cho `pk`, `sk`, `sig`.
- Hỗ trợ Pure ML-DSA và HashML-DSA qua tham số `pre_hash`.
- Hỗ trợ context `ctx` tối đa 255 byte để tách miền sử dụng.
- Hỗ trợ deterministic signing cho test/KAT và randomized signing cho demo thực tế.
- UI Streamlit cho thao tác sinh khóa, ký, xác minh và nhập/xuất hex.

## Cài Đặt Nhanh

Yêu cầu Python 3.10+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pytest
```

## Chạy Demo UI

```powershell
streamlit run app.py
```

Trong giao diện:

1. Chọn `ML-DSA-44`, `ML-DSA-65` hoặc `ML-DSA-87`.
2. Chọn mode `Pure` hoặc `HashML-DSA`.
3. Tạo `pk/sk` ở tab `KeyGen`.
4. Ký message ở tab `Sign`.
5. Kiểm tra chữ ký ở tab `Verify`.

## Dùng API Python

```python
from ml_dsa import ML_DSA_65

message = b"hello ML-DSA"

pk, sk = ML_DSA_65.keygen()
sig = ML_DSA_65.sign(sk, message, deterministic=True)

assert ML_DSA_65.verify(pk, message, sig) is True
assert ML_DSA_65.verify(pk, b"tampered", sig) is False
```

Ví dụ HashML-DSA với context:

```python
from ml_dsa import ML_DSA_65

message = b"message to be pre-hashed"
ctx = b"demo-context"

pk, sk = ML_DSA_65.keygen()
sig = ML_DSA_65.sign(
    sk,
    message,
    ctx=ctx,
    deterministic=True,
    pre_hash="sha256",
)

assert ML_DSA_65.verify(pk, message, sig, ctx=ctx, pre_hash="sha256") is True
```

## Cấu Trúc Repo

```text
.
├── app.py                         # Streamlit UI
├── ui_helpers.py                  # Parse/validate input cho UI
├── ml_dsa/
│   ├── api.py                     # ML_DSA wrapper và 3 public instances
│   ├── keygen.py                  # KeyGen / KeyGen_internal
│   ├── sign.py                    # Sign / Sign_internal
│   ├── verify.py                  # Verify / Verify_internal
│   ├── params.py                  # Bộ tham số ML-DSA
│   ├── prehash.py                 # Pure / HashML-DSA formatting
│   ├── sampling.py                # ExpandA, ExpandS, ExpandMask, SampleInBall
│   ├── ntt.py                     # NTT / INTT
│   ├── reduction.py               # Power2Round, HighBits, LowBits, Hint
│   └── encoding.py                # Encode/decode pk, sk, sig
├── test/                          # Unit, integration, regression tests
├── scripts/
│   ├── benchmark.py               # Benchmark keygen/sign/verify
│   └── validate_kat.py            # Validate ACVP KAT JSON
├── docs/                          # Tài liệu thuyết trình và Q&A
├── SPEC.md                        # Mapping FIPS 204 -> code
└── NIST.FIPS.204.pdf              # Chuẩn tham chiếu
```

## Kiểm Thử

Chạy toàn bộ test:

```powershell
python -m pytest
```

Các nhóm test chính:

- Pipeline end-to-end: `keygen -> sign -> verify`.
- Tamper cases: đổi message, signature, public key, context, pre-hash.
- Encode/decode round-trip cho `pk`, `sk`, `sig`, hint.
- NTT/INTT, sampling, reduction và regression hash.

## Benchmark

```powershell
python scripts\benchmark.py --iterations 5
python scripts\benchmark.py --scheme 65 --iterations 10
```

Script đo thời gian `keygen`, `sign`, `verify` cho từng parameter set.

## Validate KAT

Script nhận ACVP JSON cho `keyGen` hoặc `sigGen`:

```powershell
python scripts\validate_kat.py path\to\keyGen.json
python scripts\validate_kat.py path\to\sigGen.json --max 10
```

Hỗ trợ parameter set:

- `ML-DSA-44`
- `ML-DSA-65`
- `ML-DSA-87`

## Tài Liệu

- `SPEC.md`: ánh xạ thuật toán FIPS 204 sang module/hàm Python.
- `docs/ml_dsa_algorithm_flow_presentation.md`: thuyết trình luồng thuật toán.
- `docs/qa_pipeline_keygen_sign_verify_api.md`: Q&A bảo vệ pipeline và API.
- `docs/phan_vi_pipeline_keygen_sign_verify_api_slides.md`: kịch bản slide Phần VI.

## Ghi Chú Bảo Mật

Implementation này phục vụ mục đích học thuật, demo và kiểm thử. Nếu muốn dùng
trong môi trường production, cần đánh giá thêm về constant-time behavior,
side-channel, entropy, memory handling, KAT đầy đủ và review bảo mật độc lập.
