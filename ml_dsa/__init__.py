"""ML-DSA (FIPS 204) educational implementation in pure Python.

Public API:
    from ml_dsa import ML_DSA_44, ML_DSA_65, ML_DSA_87
    pk, sk = ML_DSA_65.keygen()
    sig = ML_DSA_65.sign(sk, b"message")
    ok = ML_DSA_65.verify(pk, b"message", sig)
"""

from ml_dsa.api import ML_DSA, ML_DSA_44, ML_DSA_65, ML_DSA_87

__all__ = ["ML_DSA", "ML_DSA_44", "ML_DSA_65", "ML_DSA_87"]
