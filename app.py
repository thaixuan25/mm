from __future__ import annotations

import streamlit as st

from ml_dsa import ML_DSA_44, ML_DSA_65, ML_DSA_87
from ui_helpers import (
    MESSAGE_FORMATS,
    PRE_HASH_OPTIONS,
    InputValidationError,
    bytes_to_hex,
    parse_hex_bytes,
    parse_message,
    preview_bytes,
    resolve_pre_hash,
    validate_context,
)


SCHEMES = {
    ML_DSA_44.name: ML_DSA_44,
    ML_DSA_65.name: ML_DSA_65,
    ML_DSA_87.name: ML_DSA_87,
}


def init_state() -> None:
    defaults = {
        "pk": None,
        "sk": None,
        "sig": None,
        "message": b"",
        "message_label": "",
        "last_action": "",
        "verify_result": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def set_theme() -> None:
    st.set_page_config(
        page_title="ML-DSA Demo",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        :root {
            --accent: #0f766e;
            --accent-soft: #ccfbf1;
            --ink: #111827;
            --muted: #4b5563;
            --line: #d1d5db;
            --surface: #ffffff;
            --surface-soft: #f8fafc;
            --danger: #b91c1c;
            --success: #047857;
        }
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1280px;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        textarea, input {
            font-family: "Consolas", "Menlo", "SFMono-Regular", monospace;
        }
        .hero {
            border: 1px solid var(--line);
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border-radius: 8px;
            padding: 1.2rem 1.35rem;
            margin-bottom: 1rem;
        }
        .hero-title {
            color: var(--ink);
            font-size: 1.85rem;
            line-height: 1.2;
            font-weight: 700;
            margin: 0 0 .35rem 0;
        }
        .hero-copy {
            color: var(--muted);
            margin: 0;
            max-width: 76ch;
        }
        .status-box {
            border: 1px solid var(--line);
            border-left-width: 6px;
            border-radius: 8px;
            background: var(--surface);
            padding: .85rem 1rem;
            margin: .75rem 0;
        }
        .status-box strong {
            display: block;
            color: var(--ink);
            margin-bottom: .2rem;
        }
        .status-box span {
            color: var(--muted);
        }
        .status-valid {
            border-left-color: var(--success);
            background: #f0fdf4;
        }
        .status-invalid {
            border-left-color: var(--danger);
            background: #fef2f2;
        }
        .status-info {
            border-left-color: var(--accent);
            background: #f0fdfa;
        }
        .small-label {
            color: var(--muted);
            font-size: .86rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def status_box(title: str, body: str, state: str = "info") -> None:
    safe_title = title.replace("<", "&lt;").replace(">", "&gt;")
    safe_body = body.replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(
        f"""
        <div class="status-box status-{state}">
            <strong>{safe_title}</strong>
            <span>{safe_body}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hex_output(label: str, value: bytes | None, expected_len: int) -> None:
    if value is None:
        st.text_area(label, value="", height=116, disabled=True)
        st.caption(f"Expected length: {expected_len} bytes.")
        return
    st.text_area(label, value=bytes_to_hex(value), height=156, disabled=True)
    st.caption(f"Length: {len(value)} bytes. Expected: {expected_len} bytes.")


def render_sidebar():
    with st.sidebar:
        st.header("Configuration")
        scheme_name = st.selectbox("Parameter set", tuple(SCHEMES), index=0)
        scheme = SCHEMES[scheme_name]

        st.metric("Public key bytes", scheme.pk_bytes)
        st.metric("Secret key bytes", scheme.sk_bytes)
        st.metric("Signature bytes", scheme.sig_bytes)

        mode = st.radio("Signing mode", ("Pure", "HashML-DSA"), horizontal=False)
        pre_hash = st.selectbox("Pre-hash function", PRE_HASH_OPTIONS, index=0, disabled=mode == "Pure")
        ctx = st.text_input("Context", value="", max_chars=255)
        deterministic = st.checkbox("Deterministic signing", value=True)

        st.caption("The same configuration is used for signing and verification.")

    return scheme, mode, pre_hash, ctx, deterministic


def render_keygen_panel(scheme) -> None:
    with st.container(border=True):
        st.subheader("KeyGen")
        seed_mode = st.radio(
            "Seed source",
            ("Random seed", "Fixed 32-byte hex seed"),
            horizontal=True,
            key="keygen_seed_mode",
        )
        seed_hex = ""
        if seed_mode == "Fixed 32-byte hex seed":
            seed_hex = st.text_input(
                "Seed hex",
                value="00" * 32,
                help="Exactly 32 bytes, shown as 64 hex characters.",
            )

        if st.button("Generate key pair", type="primary", use_container_width=True):
            try:
                xi = None
                if seed_mode == "Fixed 32-byte hex seed":
                    xi = parse_hex_bytes(seed_hex, field_name="Seed", expected_len=32)
                pk, sk = scheme.keygen(xi=xi)
                st.session_state.pk = pk
                st.session_state.sk = sk
                st.session_state.sig = None
                st.session_state.verify_result = None
                st.session_state.last_action = f"Generated key pair for {scheme.name}."
                status_box("Key pair generated", st.session_state.last_action, "valid")
            except (InputValidationError, ValueError) as exc:
                status_box("KeyGen input error", str(exc), "invalid")

        cols = st.columns(2)
        with cols[0]:
            render_hex_output("Public key hex", st.session_state.pk, scheme.pk_bytes)
        with cols[1]:
            render_hex_output("Secret key hex", st.session_state.sk, scheme.sk_bytes)


def render_sign_panel(scheme, mode: str, selected_hash: str, ctx: str, deterministic: bool) -> None:
    with st.container(border=True):
        st.subheader("Sign")
        sk_source = st.radio(
            "Secret key source",
            ("Current generated secret key", "Paste secret key hex"),
            horizontal=True,
            key="sign_sk_source",
        )
        pasted_sk = ""
        if sk_source == "Paste secret key hex":
            pasted_sk = st.text_area("Secret key hex for signing", height=140)

        message_format = st.radio("Message input format", MESSAGE_FORMATS, horizontal=True, key="sign_msg_format")
        default_message = "The quick brown fox jumps over the lazy dog"
        message_value = st.text_area("Message to sign", value=default_message, height=130)

        if st.button("Sign message", type="primary", use_container_width=True):
            try:
                if sk_source == "Current generated secret key":
                    if st.session_state.sk is None:
                        raise InputValidationError("Generate a key pair first or paste a secret key.")
                    sk = st.session_state.sk
                else:
                    sk = parse_hex_bytes(
                        pasted_sk,
                        field_name="Secret key",
                        expected_len=scheme.sk_bytes,
                    )
                message = parse_message(message_value, message_format)
                ctx_bytes = validate_context(ctx)
                pre_hash = resolve_pre_hash(mode, selected_hash)
                sig = scheme.sign(
                    sk,
                    message,
                    ctx=ctx_bytes,
                    deterministic=deterministic,
                    pre_hash=pre_hash,
                )
                st.session_state.sig = sig
                st.session_state.message = message
                st.session_state.message_label = preview_bytes(message)
                st.session_state.verify_result = None
                status_box("Message signed", f"Signature created with {scheme.name}.", "valid")
            except (InputValidationError, ValueError) as exc:
                status_box("Sign input error", str(exc), "invalid")

        render_hex_output("Signature hex", st.session_state.sig, scheme.sig_bytes)


def render_verify_panel(scheme, mode: str, selected_hash: str, ctx: str) -> None:
    with st.container(border=True):
        st.subheader("Verify")
        pk_source = st.radio(
            "Public key source",
            ("Current generated public key", "Paste public key hex"),
            horizontal=True,
            key="verify_pk_source",
        )
        pasted_pk = ""
        if pk_source == "Paste public key hex":
            pasted_pk = st.text_area("Public key hex for verification", height=140)

        sig_source = st.radio(
            "Signature source",
            ("Current signature", "Paste signature hex"),
            horizontal=True,
            key="verify_sig_source",
        )
        pasted_sig = ""
        if sig_source == "Paste signature hex":
            pasted_sig = st.text_area("Signature hex for verification", height=150)

        msg_source = st.radio(
            "Message source",
            ("Last signed message", "Custom message"),
            horizontal=True,
            key="verify_msg_source",
        )
        verify_format = MESSAGE_FORMATS[0]
        verify_value = ""
        if msg_source == "Last signed message":
            st.caption(f"Last signed message preview: {st.session_state.message_label or '(empty)'}")
        else:
            verify_format = st.radio(
                "Verification message input format",
                MESSAGE_FORMATS,
                horizontal=True,
                key="verify_msg_format",
            )
            verify_value = st.text_area("Message to verify", height=130)

        if st.button("Verify signature", type="primary", use_container_width=True):
            try:
                if pk_source == "Current generated public key":
                    if st.session_state.pk is None:
                        raise InputValidationError("Generate a key pair first or paste a public key.")
                    pk = st.session_state.pk
                else:
                    pk = parse_hex_bytes(
                        pasted_pk,
                        field_name="Public key",
                        expected_len=scheme.pk_bytes,
                    )

                if sig_source == "Current signature":
                    if st.session_state.sig is None:
                        raise InputValidationError("Sign a message first or paste a signature.")
                    sig = st.session_state.sig
                else:
                    sig = parse_hex_bytes(
                        pasted_sig,
                        field_name="Signature",
                        expected_len=scheme.sig_bytes,
                    )

                if msg_source == "Last signed message":
                    verify_message = st.session_state.message
                else:
                    verify_message = parse_message(verify_value, verify_format)

                ctx_bytes = validate_context(ctx)
                pre_hash = resolve_pre_hash(mode, selected_hash)
                is_valid = scheme.verify(pk, verify_message, sig, ctx=ctx_bytes, pre_hash=pre_hash)
                st.session_state.verify_result = is_valid
            except (InputValidationError, ValueError) as exc:
                st.session_state.verify_result = None
                status_box("Verify input error", str(exc), "invalid")

        if st.session_state.verify_result is True:
            status_box("Valid", "The signature matches the selected public key, message, and mode.", "valid")
        elif st.session_state.verify_result is False:
            status_box("Invalid", "Verification failed for the selected public key, message, signature, or mode.", "invalid")
        else:
            status_box("Ready", "Run verification after generating or pasting the required values.", "info")


def main() -> None:
    init_state()
    set_theme()

    st.markdown(
        """
        <section class="hero">
            <h1 class="hero-title">ML-DSA Demo</h1>
            <p class="hero-copy">
                Generate keys, sign messages, and verify ML-DSA signatures locally with the Python implementation in this repository.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    scheme, mode, selected_hash, ctx, deterministic = render_sidebar()

    tabs = st.tabs(["KeyGen", "Sign", "Verify"])
    with tabs[0]:
        render_keygen_panel(scheme)
    with tabs[1]:
        render_sign_panel(scheme, mode, selected_hash, ctx, deterministic)
    with tabs[2]:
        render_verify_panel(scheme, mode, selected_hash, ctx)


if __name__ == "__main__":
    main()
