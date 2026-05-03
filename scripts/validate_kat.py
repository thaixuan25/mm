"""Đối chiếu implementation với NIST ACVP KAT vectors cho ML-DSA.

Chạy:
    python scripts/validate_kat.py path/to/keyGen.json
    python scripts/validate_kat.py path/to/sigGen.json

Định dạng tương thích với ACVP-Server `internalProjection.json` cho
ML-DSA-keyGen-FIPS204 và ML-DSA-sigGen-FIPS204.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_dsa import ML_DSA_44, ML_DSA_65, ML_DSA_87
from ml_dsa.api import ML_DSA


SCHEMES = {
    "ML-DSA-44": ML_DSA_44,
    "ML-DSA-65": ML_DSA_65,
    "ML-DSA-87": ML_DSA_87,
}


def _get_scheme(name: str) -> ML_DSA:
    if name not in SCHEMES:
        raise ValueError(f"unknown scheme: {name}")
    return SCHEMES[name]


def _validate_keygen(data: dict, max_per_set: int) -> int:
    failed = 0
    total = 0
    for group in data["testGroups"]:
        scheme = _get_scheme(group["parameterSet"])
        print(f"\n[{scheme.name}] keyGen: {len(group['tests'])} tests; first {max_per_set}")
        for tc in group["tests"][:max_per_set]:
            seed = bytes.fromhex(tc["seed"])
            expected_pk = bytes.fromhex(tc["pk"])
            expected_sk = bytes.fromhex(tc["sk"])
            pk, sk = scheme.keygen(xi=seed)
            ok = pk == expected_pk and sk == expected_sk
            total += 1
            failed += 0 if ok else 1
            mark = "PASS" if ok else "FAIL"
            print(f"  tcId={tc['tcId']:>3} [{mark}]")
            if not ok:
                if pk != expected_pk:
                    print(f"    pk diff @ first byte: got={pk[:8].hex()} want={expected_pk[:8].hex()}")
                if sk != expected_sk:
                    print(f"    sk diff @ first byte: got={sk[:8].hex()} want={expected_sk[:8].hex()}")
    print(f"\nKeyGen: {total - failed}/{total} passed")
    return failed


def _validate_siggen(data: dict, max_per_set: int) -> int:
    failed = 0
    total = 0
    for group in data["testGroups"]:
        if group.get("signatureInterface") != "external":
            print(f"\n[{group.get('parameterSet')}] skip group (interface={group.get('signatureInterface')})")
            continue
        if group.get("preHash") != "pure":
            print(f"\n[{group.get('parameterSet')}] skip group (preHash={group.get('preHash')})")
            continue
        if group.get("externalMu") is True:
            print(f"\n[{group.get('parameterSet')}] skip group (externalMu=true)")
            continue
        scheme = _get_scheme(group["parameterSet"])
        deterministic = group.get("deterministic", True)
        print(
            f"\n[{scheme.name}] sigGen: {len(group['tests'])} tests; first {max_per_set} "
            f"(deterministic={deterministic})"
        )
        for tc in group["tests"][:max_per_set]:
            sk = bytes.fromhex(tc["sk"])
            msg = bytes.fromhex(tc["message"])
            ctx = bytes.fromhex(tc.get("context", ""))
            expected_sig = bytes.fromhex(tc["signature"])
            rnd = bytes.fromhex(tc.get("rnd", "00" * 32)) if not deterministic else b"\x00" * 32
            sig = scheme.sign(sk, msg, ctx=ctx, rnd=rnd)
            ok = sig == expected_sig
            total += 1
            failed += 0 if ok else 1
            mark = "PASS" if ok else "FAIL"
            print(f"  tcId={tc['tcId']:>3} [{mark}]")
            if not ok:
                print(f"    sig diff @ first byte: got={sig[:8].hex()} want={expected_sig[:8].hex()}")
    print(f"\nSigGen: {total - failed}/{total} passed")
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="đường dẫn JSON KAT")
    parser.add_argument(
        "--max", type=int, default=3, help="số test tối đa cho mỗi mức (mặc định 3)"
    )
    args = parser.parse_args()
    with open(args.path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mode = data.get("mode")
    if mode == "keyGen":
        failed = _validate_keygen(data, args.max)
    elif mode == "sigGen":
        failed = _validate_siggen(data, args.max)
    else:
        raise ValueError(f"unsupported mode: {mode}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
