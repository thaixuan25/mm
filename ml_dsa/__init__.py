"""ML-DSA (FIPS 204) educational implementation in pure Python.

Module này expose ba bộ tham số chuẩn hóa của FIPS 204 dưới dạng wrapper
tiện dụng:

- ``ML_DSA_44``: mức an toàn ~Category 2 (lam=128).
- ``ML_DSA_65``: mức an toàn ~Category 3 (lam=192).
- ``ML_DSA_87``: mức an toàn ~Category 5 (lam=256).

Mỗi wrapper cung cấp các phương thức `keygen / sign / verify`, đồng thời
giữ nguyên các biến nội bộ (`keygen_internal`, `sign_internal`,
`verify_internal`) để phục vụ việc kiểm thử KAT.

Example
-------
>>> from ml_dsa import ML_DSA_65
>>> pk, sk = ML_DSA_65.keygen()
>>> sig = ML_DSA_65.sign(sk, b"hello")
>>> ML_DSA_65.verify(pk, b"hello", sig)
True
"""

from ml_dsa.api import ML_DSA, ML_DSA_44, ML_DSA_65, ML_DSA_87

__all__ = ["ML_DSA", "ML_DSA_44", "ML_DSA_65", "ML_DSA_87"]
