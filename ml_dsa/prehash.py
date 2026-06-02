import hashlib

# Định nghĩa mã định danh OID của các hàm băm theo DER (FIPS 204 Table 3)
PH_OIDS = {
    "sha256": bytes.fromhex("0609608648016503040201"),
    "sha512": bytes.fromhex("0609608648016503040203"),
    "sha3_256": bytes.fromhex("0609608648016503040208"),
    "shake128": bytes.fromhex("060960864801650304020b"),
    "shake256": bytes.fromhex("060960864801650304020c")
}

def NormalizePHName(ph_name: str) -> str:
    name = ph_name.lower().replace("-", "_")
    aliases = {
        "sha_256": "sha256",
        "sha2_256": "sha256",
        "sha_512": "sha512",
        "sha2_512": "sha512",
        "shake_128": "shake128",
        "shake_256": "shake256",
    }
    return aliases.get(name, name)

def ComputePH(ph_name: str, message: bytes) -> bytes:
    """
    Hỗ trợ tính giá trị băm PH(M) tương ứng với từng giải thuật.
    """
    name = NormalizePHName(ph_name)
    if name == "sha256":
        return hashlib.sha256(message).digest()
    elif name == "sha512":
        return hashlib.sha512(message).digest()
    elif name == "sha3_256":
        return hashlib.sha3_256(message).digest()
    elif name == "shake128":
        # Độ dài đầu ra mặc định cho PH shake128 là 32 bytes (FIPS 204 Table 3)
        return hashlib.shake_128(message).digest(32)
    elif name == "shake256":
        # Độ dài đầu ra mặc định cho PH shake256 là 64 bytes (FIPS 204 Table 3)
        return hashlib.shake_256(message).digest(64)
    else:
        raise ValueError(f"Hàm băm pre-hash không được hỗ trợ: {ph_name}")


def FormatMessagePrime(message: bytes, ctx: bytes = b"", *, pre_hash: str | None = None) -> bytes:
    """
    Định dạng cấu trúc bản tin M' phục vụ cho quá trình KeyGen / Sign / Verify.
    - Pure ML-DSA:  0x00 || len(ctx) || ctx || M
    - HashML-DSA:  0x01 || len(ctx) || ctx || OID(PH) || PH(M)
    """
    if len(ctx) > 255:
        raise ValueError("Độ dài ngữ cảnh (ctx) không được vượt quá 255 bytes.")
        
    ctx_len = bytes([len(ctx)])
    
    if pre_hash is None:
        # Chế độ Pure ML-DSA
        return b"\x00" + ctx_len + ctx + message
    else:
        # Chế độ HashML-DSA
        normalized_ph = NormalizePHName(pre_hash)
        if normalized_ph not in PH_OIDS:
            raise ValueError(f"Hàm băm pre-hash không tồn tại trong danh mục: {pre_hash}")
            
        oid = PH_OIDS[normalized_ph]
        ph_m = ComputePH(normalized_ph, message)
        return b"\x01" + ctx_len + ctx + oid + ph_m
