# ML-DSA Specification Mapping (FIPS 204)

Tài liệu này cố định đặc tả ML-DSA mà chúng ta cài đặt và ánh xạ trực tiếp các thuật toán/hàm trong FIPS 204 sang module Python tương ứng. Mục đích để mọi quyết định triển khai đều bám đặc tả, dễ kiểm thử và bảo trì.

## 1. Tham số toàn cục

- `q = 8380417 = 2^23 - 2^13 + 1` (modulo nguyên tố)
- `n = 256` (bậc đa thức)
- `d = 13` (số bit thấp loại bỏ khi Power2Round)
- `ζ = 1753` (căn nguyên thủy bậc 256 của đơn vị mod q)
- Ring `Rq = Zq[X] / (X^n + 1)`

Bộ tham số theo mức an toàn (FIPS 204 §4 Table 1):

| Tham số | ML-DSA-44 | ML-DSA-65 | ML-DSA-87 |
|---|---|---|---|
| `(k, l)` | `(4, 4)` | `(6, 5)` | `(8, 7)` |
| `η`     | 2 | 4 | 2 |
| `τ`     | 39 | 49 | 60 |
| `β = τ·η` | 78 | 196 | 120 |
| `γ1`    | `2^17` | `2^19` | `2^19` |
| `γ2`    | `(q-1)/88` | `(q-1)/32` | `(q-1)/32` |
| `ω`     | 80 | 55 | 75 |
| `λ`     | 128 | 192 | 256 |

Kích thước (byte): `pk`, `sk`, `sig` được tính từ các tham số trên — xem `params.py`.

## 2. Ánh xạ thuật toán → Python

| FIPS 204 | Python module:hàm |
|---|---|
| §7.1 KeyGen / KeyGen_internal | `ml_dsa/keygen.py:keygen`, `keygen_internal` |
| §7.2 Sign / Sign_internal | `ml_dsa/sign.py:sign`, `sign_internal` |
| §7.3 Verify / Verify_internal | `ml_dsa/verify.py:verify`, `verify_internal` |
| Algorithm 35 NTT | `ml_dsa/ntt.py:ntt` |
| Algorithm 36 NTT^{-1} | `ml_dsa/ntt.py:intt` |
| Algorithm 30 ExpandA | `ml_dsa/sampling.py:expand_a` |
| Algorithm 33 ExpandS | `ml_dsa/sampling.py:expand_s` |
| Algorithm 34 ExpandMask | `ml_dsa/sampling.py:expand_mask` |
| Algorithm 29 SampleInBall | `ml_dsa/sampling.py:sample_in_ball` |
| Algorithm 31 RejNTTPoly | `ml_dsa/sampling.py:rej_ntt_poly` |
| Algorithm 32 RejBoundedPoly | `ml_dsa/sampling.py:rej_bounded_poly` |
| Algorithm 35 Power2Round | `ml_dsa/reduction.py:power2round` |
| Algorithm 36 Decompose | `ml_dsa/reduction.py:decompose` |
| Algorithm 37 HighBits | `ml_dsa/reduction.py:high_bits` |
| Algorithm 38 LowBits | `ml_dsa/reduction.py:low_bits` |
| Algorithm 39 MakeHint | `ml_dsa/reduction.py:make_hint` |
| Algorithm 40 UseHint | `ml_dsa/reduction.py:use_hint` |
| Algorithm 16/17 BitPack/BitUnpack | `ml_dsa/encoding.py:bit_pack`, `bit_unpack` |
| Algorithm 18/19 SimpleBitPack/SimpleBitUnpack | `ml_dsa/encoding.py:simple_bit_pack`, `simple_bit_unpack` |
| Algorithm 20 HintBitPack | `ml_dsa/encoding.py:hint_bit_pack` |
| Algorithm 21 HintBitUnpack | `ml_dsa/encoding.py:hint_bit_unpack` |
| Algorithm 22 pkEncode | `ml_dsa/encoding.py:pk_encode` |
| Algorithm 23 pkDecode | `ml_dsa/encoding.py:pk_decode` |
| Algorithm 24 skEncode | `ml_dsa/encoding.py:sk_encode` |
| Algorithm 25 skDecode | `ml_dsa/encoding.py:sk_decode` |
| Algorithm 26 sigEncode | `ml_dsa/encoding.py:sig_encode` |
| Algorithm 27 sigDecode | `ml_dsa/encoding.py:sig_decode` |
| Algorithm 28 w1Encode | `ml_dsa/encoding.py:w1_encode` |

## 3. Hàm hash & XOF

- `H = SHAKE256` cho TR, μ, c̃, ρ′, K mở rộng.
- `G = SHAKE128` cho mở rộng ma trận A (ExpandA dùng ExpandA-XOF dựa trên SHAKE128).
- Cài đặt qua `hashlib.shake_128`, `hashlib.shake_256`.

## 4. Quy ước biểu diễn

- Hệ số đa thức lưu dạng số nguyên trong `[0, q)`.
- NTT/INTT thao tác in-place trên list 256 phần tử.
- Vector/ma trận biểu diễn bằng list các đa thức (list[int]).
- Định dạng byte little-endian theo đặc tả.

## 5. Quy tắc kiểm thử bám đặc tả

1. NTT đảo: `INTT(NTT(a)) == a` cho mọi đa thức ngẫu nhiên.
2. Tương đương phép nhân: `INTT(NTT(a) ∘ NTT(b)) == a · b mod (X^n+1)`.
3. Round-trip encode: `decode(encode(x)) == x` cho pk, sk, sig, hint.
4. Tự nhất quán: `verify(pk, m, sign(sk, m)) == True`.
5. Sửa chữa ký/thông điệp/khóa → verify trả về False.
6. Tính quyết định khi seed cố định (deterministic mode).
