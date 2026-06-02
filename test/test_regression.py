"""Regression tests: bám hash deterministic của pk/sk/sig.

Mục đích: phát hiện sớm bất kỳ thay đổi ngẫu nhiên nào trong implementation
gây ra output khác đi. Các giá trị hash được "đóng băng" sau khi
implementation đã pass toàn bộ kiểm thử pipeline trên cả 3 mức.

Nếu refactor có chủ đích thay đổi output (ví dụ sửa bug), cần cập nhật
các fixture này một cách có ý thức.
"""

from __future__ import annotations

import hashlib

import pytest

from ml_dsa import ML_DSA_44, ML_DSA_65, ML_DSA_87
from ml_dsa.api import ML_DSA


FIXTURES = {
    "ML-DSA-44": {
        "pk_sha256": "d7e152ccde2ca935ab4a86b70dcf9f0a3a898bbe4892f99b19a4e850f7c716dd",
        "sk_sha256": "eae73ea1666d4d01404a972830c997eccd371b3babe535d37834936c000edcdb",
        "sig_sha256": "79a24dfd0b2608efb77949b9b23c0319951423d7e22b1d0b3fd6dd394b9548ac",
    },
    "ML-DSA-65": {
        "pk_sha256": "52c78d901978283cd8406e5c984fa1e76929339217171edc74fdd91de019c82b",
        "sk_sha256": "10534c2861e06ada3eac8268f102b81d1fe18a5a9bd80f5ca983f773dd9b940f",
        "sig_sha256": "b00ccc4f400c964cea2b62a80c38f3073caee397888d6c226a9ed2fe0d5bbac4",
    },
    "ML-DSA-87": {
        "pk_sha256": "0b7c848572e0148a41bfcaf55cf7f9badbc7879404e6b7590aad9e4ebc0a009c",
        "sk_sha256": "27b13ce8a25983c3a34a508b216ddf76fe223dc7c166b280bc12b1f18e0128bd",
        "sig_sha256": "cb91f42693d1dfc55ac78ce1017efa0273f84a10f782b7b41559a0f8953f1558",
    },
}

PARAMS = [
    pytest.param(ML_DSA_44, id="ML-DSA-44"),
    pytest.param(ML_DSA_65, id="ML-DSA-65"),
    pytest.param(ML_DSA_87, id="ML-DSA-87"),
]


@pytest.mark.parametrize("scheme", PARAMS)
def test_deterministic_outputs_match_frozen_hashes(scheme: ML_DSA):
    fixture = FIXTURES[scheme.name]
    xi = bytes([scheme.params.k]) * 32
    msg = b"regression-fixture"
    pk, sk = scheme.keygen(xi=xi)
    sig = scheme.sign(sk, msg, deterministic=True)
    assert hashlib.sha256(pk).hexdigest() == fixture["pk_sha256"]
    assert hashlib.sha256(sk).hexdigest() == fixture["sk_sha256"]
    assert hashlib.sha256(sig).hexdigest() == fixture["sig_sha256"]
    assert scheme.verify(pk, msg, sig) is True
