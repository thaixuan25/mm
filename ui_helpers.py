from __future__ import annotations

import re
import string
from typing import Final


PRE_HASH_OPTIONS: Final[tuple[str, ...]] = (
    "sha256",
    "sha512",
    "sha3_256",
    "shake128",
    "shake256",
)

MESSAGE_FORMATS: Final[tuple[str, ...]] = ("UTF-8 text", "Hex bytes")


class InputValidationError(ValueError):
    """Raised when UI input cannot be converted to the expected byte value."""


def bytes_to_hex(value: bytes, *, line_chars: int = 64) -> str:
    """Format bytes as wrapped lowercase hex for readable text areas."""
    hex_value = value.hex()
    if line_chars <= 0:
        return hex_value
    return "\n".join(
        hex_value[i : i + line_chars] for i in range(0, len(hex_value), line_chars)
    )


def parse_hex_bytes(
    value: str,
    *,
    field_name: str,
    expected_len: int | None = None,
    allow_empty: bool = False,
) -> bytes:
    """Parse UI hex text, ignoring whitespace and one leading 0x prefix."""
    cleaned = re.sub(r"\s+", "", value or "")
    if cleaned.lower().startswith("0x"):
        cleaned = cleaned[2:]

    if not cleaned:
        if allow_empty:
            return b""
        raise InputValidationError(f"{field_name} is required.")

    if len(cleaned) % 2 != 0:
        raise InputValidationError(f"{field_name} must contain an even number of hex characters.")

    invalid = [ch for ch in cleaned if ch not in string.hexdigits]
    if invalid:
        raise InputValidationError(f"{field_name} contains non-hex characters.")

    decoded = bytes.fromhex(cleaned)
    if expected_len is not None and len(decoded) != expected_len:
        raise InputValidationError(
            f"{field_name} must be {expected_len} bytes, got {len(decoded)} bytes."
        )
    return decoded


def parse_message(value: str, input_format: str) -> bytes:
    if input_format == "UTF-8 text":
        return (value or "").encode("utf-8")
    if input_format == "Hex bytes":
        return parse_hex_bytes(value, field_name="Message", allow_empty=True)
    raise InputValidationError(f"Unsupported message format: {input_format}.")


def validate_context(ctx: str) -> bytes:
    ctx_bytes = (ctx or "").encode("utf-8")
    if len(ctx_bytes) > 255:
        raise InputValidationError(f"Context must be at most 255 bytes, got {len(ctx_bytes)} bytes.")
    return ctx_bytes


def resolve_pre_hash(mode: str, selected_hash: str) -> str | None:
    if mode == "Pure":
        return None
    if mode != "HashML-DSA":
        raise InputValidationError(f"Unsupported signing mode: {mode}.")
    if selected_hash not in PRE_HASH_OPTIONS:
        raise InputValidationError(f"Unsupported pre-hash function: {selected_hash}.")
    return selected_hash


def preview_bytes(value: bytes, *, max_chars: int = 96) -> str:
    text = value.decode("utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."
