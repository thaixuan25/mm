"""Benchmark đơn giản cho keygen/sign/verify trên cả 3 mức ML-DSA.

Chạy: `python scripts/benchmark.py [--iterations N]`
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from statistics import mean, stdev
from typing import Callable, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_dsa import ML_DSA_44, ML_DSA_65, ML_DSA_87
from ml_dsa.api import ML_DSA


def _time_call(fn: Callable[[], object], iterations: int) -> List[float]:
    samples: List[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return samples


def _format(samples: List[float]) -> str:
    if not samples:
        return "n/a"
    avg = mean(samples) * 1000
    sd = stdev(samples) * 1000 if len(samples) > 1 else 0.0
    return f"{avg:8.2f} ms ± {sd:6.2f}"


def benchmark(scheme: ML_DSA, iterations: int) -> None:
    print(f"\n=== {scheme.name} ===")
    print(f"  pk_bytes={scheme.pk_bytes}, sk_bytes={scheme.sk_bytes}, sig_bytes={scheme.sig_bytes}")

    keygen_times = _time_call(lambda: scheme.keygen(), iterations)
    print(f"  keygen   : {_format(keygen_times)}")

    pk, sk = scheme.keygen()
    msg = b"benchmark message - " + os.urandom(16)

    sign_times = _time_call(
        lambda: scheme.sign(sk, msg, deterministic=False), iterations
    )
    print(f"  sign     : {_format(sign_times)}")

    sig = scheme.sign(sk, msg, deterministic=False)
    verify_times = _time_call(lambda: scheme.verify(pk, msg, sig), iterations)
    print(f"  verify   : {_format(verify_times)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=5,
        help="Số lần lặp cho mỗi phép đo (mặc định 5).",
    )
    parser.add_argument(
        "--scheme",
        choices=["44", "65", "87", "all"],
        default="all",
        help="Mức ML-DSA (mặc định all).",
    )
    args = parser.parse_args()

    schemes = {
        "44": ML_DSA_44,
        "65": ML_DSA_65,
        "87": ML_DSA_87,
    }
    targets = (
        list(schemes.values()) if args.scheme == "all" else [schemes[args.scheme]]
    )
    for scheme in targets:
        benchmark(scheme, args.iterations)


if __name__ == "__main__":
    main()
