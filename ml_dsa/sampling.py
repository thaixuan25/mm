from __future__ import annotations

import hashlib
from typing import List, Optional, Tuple

from .params import MLDSAParams, N, Q
from .poly import Poly, Vector


class SHAKEStream:
    """XOF stream wrapper for hashlib.shake_128 and hashlib.shake_256."""

    def __init__(self, shake_func, seed: bytes):
        self.shake = shake_func(seed)
        self.buffer = b""
        self.offset = 0

    def read(self, num_bytes: int) -> bytes:
        needed = self.offset + num_bytes
        if len(self.buffer) < needed:
            chunk_size = max(needed * 2, 4096)
            self.buffer = self.shake.digest(chunk_size)

        result = self.buffer[self.offset : needed]
        self.offset += num_bytes
        return result


def CoeffFromThreeBytes(b0: int, b1: int, b2: int) -> Optional[int]:
    """Algorithm 14: CoeffFromThreeBytes."""
    c = ((b2 & 0x7F) << 16) | (b1 << 8) | b0
    if c < Q:
        return c
    return None


def CoeffFromHalfByte(b: int, eta: int) -> Optional[int]:
    """Algorithm 15: CoeffFromHalfByte."""
    if eta == 2 and b < 15:
        return 2 - (b % 5)
    if eta == 4 and b < 9:
        return 4 - b
    return None


def RejNttPoly(seed: bytes) -> Poly:
    """Algorithm 30: RejNTTPoly."""
    stream = SHAKEStream(hashlib.shake_128, seed)
    a: Poly = []

    while len(a) < N:
        buf = stream.read(3)
        c = CoeffFromThreeBytes(buf[0], buf[1], buf[2])
        if c is not None:
            a.append(c)
    return a


def RejBoundedPoly(seed: bytes, eta: int) -> Poly:
    """Algorithm 31: RejBoundedPoly."""
    stream = SHAKEStream(hashlib.shake_256, seed)
    a: Poly = []

    while len(a) < N:
        z = stream.read(1)[0]
        z0 = z & 0x0F
        z1 = z >> 4

        c0 = CoeffFromHalfByte(z0, eta)
        if c0 is not None:
            a.append(c0)

        if len(a) >= N:
            break

        c1 = CoeffFromHalfByte(z1, eta)
        if c1 is not None:
            a.append(c1)
    return a


def ExpandA(rho: bytes, params: MLDSAParams) -> List[List[Poly]]:
    """Algorithm 32: ExpandA."""
    if len(rho) != 32:
        raise ValueError("ExpandA: rho must be 32 bytes")

    A: List[List[Poly]] = []
    for r in range(params.k):
        row: List[Poly] = []
        for c in range(params.l):
            seed = rho + bytes([c, r])
            row.append(RejNttPoly(seed))
        A.append(row)
    return A


def ExpandS(rho_prime: bytes, params: MLDSAParams) -> Tuple[Vector, Vector]:
    """Algorithm 33: ExpandS."""
    if len(rho_prime) != 64:
        raise ValueError("ExpandS: rho_prime must be 64 bytes")

    s1: Vector = []
    for i in range(params.l):
        seed = rho_prime + i.to_bytes(2, byteorder="little")
        s1.append(RejBoundedPoly(seed, params.eta))

    s2: Vector = []
    for i in range(params.k):
        seed = rho_prime + (params.l + i).to_bytes(2, byteorder="little")
        s2.append(RejBoundedPoly(seed, params.eta))

    return s1, s2


def ExpandMask(rho_pp: bytes, mu_counter: int, params: MLDSAParams) -> Vector:
    """Algorithm 34: ExpandMask."""
    if len(rho_pp) != 64:
        raise ValueError("ExpandMask: rho'' must be 64 bytes")

    y: Vector = []
    for i in range(params.l):
        nonce = mu_counter + i
        seed = rho_pp + nonce.to_bytes(2, byteorder="little")

        if params.gamma1 == (1 << 17):
            stream = SHAKEStream(hashlib.shake_256, seed)
            data = stream.read(576)
            poly: Poly = []
            for chunk_idx in range(64):
                offset = chunk_idx * 9
                v = int.from_bytes(data[offset : offset + 9], byteorder="little")
                for shift in (0, 18, 36, 54):
                    z = (v >> shift) & 0x3FFFF
                    poly.append(params.gamma1 - z)
            y.append(poly)
        elif params.gamma1 == (1 << 19):
            stream = SHAKEStream(hashlib.shake_256, seed)
            data = stream.read(640)
            poly = []
            for chunk_idx in range(128):
                offset = chunk_idx * 5
                v = int.from_bytes(data[offset : offset + 5], byteorder="little")
                for shift in (0, 20):
                    z = (v >> shift) & 0xFFFFF
                    poly.append(params.gamma1 - z)
            y.append(poly)
        else:
            raise ValueError("gamma1 must be 2^17 or 2^19")
    return y


def SampleInBall(rho: bytes, params: MLDSAParams) -> Poly:
    """Algorithm 29: SampleInBall."""
    if len(rho) != params.c_tilde_bytes:
        raise ValueError(
            f"SampleInBall: expected {params.c_tilde_bytes} bytes, got {len(rho)}"
        )

    tau = params.tau
    c: Poly = [0] * N
    stream = SHAKEStream(hashlib.shake_256, rho)
    s = stream.read(8)

    h: List[int] = []
    for b in s:
        for bit in range(8):
            h.append((b >> bit) & 1)

    for i in range(N - tau, N):
        while True:
            j = stream.read(1)[0]
            if j <= i:
                break

        c[i] = c[j]
        c[j] = 1 - 2 * h[i + tau - N]

    return c


def HShake256(data: bytes, length: int) -> bytes:
    return hashlib.shake_256(data).digest(length)


def GShake128(data: bytes, length: int) -> bytes:
    return hashlib.shake_128(data).digest(length)


__all__ = [
    "SHAKEStream",
    "CoeffFromThreeBytes",
    "CoeffFromHalfByte",
    "RejNttPoly",
    "RejBoundedPoly",
    "ExpandA",
    "ExpandS",
    "ExpandMask",
    "SampleInBall",
    "HShake256",
    "GShake128",
]
