"""Parameter-set conformance tests for FIPS 204 Table 1/2 values."""

from __future__ import annotations

from ml_dsa.params import (
    D,
    N,
    PARAMS_BY_NAME,
    Q,
    ZETA,
)


EXPECTED = {
    "ML-DSA-44": {
        "k": 4,
        "l": 4,
        "eta": 2,
        "tau": 39,
        "beta": 78,
        "gamma1": 2**17,
        "gamma2": (Q - 1) // 88,
        "omega": 80,
        "lam": 128,
        "pk_bytes": 1312,
        "sk_bytes": 2560,
        "sig_bytes": 2420,
    },
    "ML-DSA-65": {
        "k": 6,
        "l": 5,
        "eta": 4,
        "tau": 49,
        "beta": 196,
        "gamma1": 2**19,
        "gamma2": (Q - 1) // 32,
        "omega": 55,
        "lam": 192,
        "pk_bytes": 1952,
        "sk_bytes": 4032,
        "sig_bytes": 3309,
    },
    "ML-DSA-87": {
        "k": 8,
        "l": 7,
        "eta": 2,
        "tau": 60,
        "beta": 120,
        "gamma1": 2**19,
        "gamma2": (Q - 1) // 32,
        "omega": 75,
        "lam": 256,
        "pk_bytes": 2592,
        "sk_bytes": 4896,
        "sig_bytes": 4627,
    },
}


def test_global_parameters_match_fips204_constants():
    assert Q == 8380417
    assert N == 256
    assert D == 13
    assert ZETA == 1753


def test_parameter_sets_match_declared_fips204_values():
    assert set(PARAMS_BY_NAME) == set(EXPECTED)
    for name, expected in EXPECTED.items():
        params = PARAMS_BY_NAME[name]
        for field, value in expected.items():
            assert getattr(params, field) == value
        assert params.beta == params.tau * params.eta
        assert params.c_tilde_bytes == params.lam // 4
