"""Demo ký và xác minh thông điệp với ML-DSA.

Chạy: `python scripts/demo.py`
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_dsa import ML_DSA_44, ML_DSA_65, ML_DSA_87
from ml_dsa.api import ML_DSA


def _hex_summary(blob: bytes, head: int = 8, tail: int = 4) -> str:
    if len(blob) <= head + tail:
        return blob.hex()
    return f"{blob[:head].hex()}…{blob[-tail:].hex()} ({len(blob)} bytes)"


def demo(scheme: ML_DSA, message: bytes, ctx: bytes = b"demo") -> None:
    print(f"\n=== {scheme.name} ===")
    pk, sk = scheme.keygen()
    print(f"  pk: {_hex_summary(pk)}")
    print(f"  sk: {_hex_summary(sk)}")

    sig = scheme.sign(sk, message, ctx=ctx)
    print(f"  message: {message!r}, ctx: {ctx!r}")
    print(f"  sig: {_hex_summary(sig)}")

    ok = scheme.verify(pk, message, sig, ctx=ctx)
    print(f"  verify(pk, m, sig, ctx) -> {ok}")

    tampered = bytearray(sig)
    tampered[scheme.params.c_tilde_bytes] ^= 0x01
    bad = scheme.verify(pk, message, bytes(tampered), ctx=ctx)
    print(f"  verify(pk, m, tampered_sig, ctx) -> {bad}")

    wrong_ctx = scheme.verify(pk, message, sig, ctx=b"other")
    print(f"  verify(pk, m, sig, ctx='other') -> {wrong_ctx}")


def main() -> None:
    msg = b"Post-quantum signatures with ML-DSA in pure Python."
    for scheme in (ML_DSA_44, ML_DSA_65, ML_DSA_87):
        demo(scheme, msg)


if __name__ == "__main__":
    main()
