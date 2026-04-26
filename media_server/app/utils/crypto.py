from __future__ import annotations

import base64
import os
from dataclasses import dataclass

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


@dataclass(frozen=True)
class EncryptedPayload:
    nonce_b64: str
    ciphertext_b64: str
    tag_b64: str


def encrypt_bytes(plaintext: bytes, aes_key: bytes) -> EncryptedPayload:
    nonce = os.urandom(12)
    aesgcm = AESGCM(aes_key)
    encrypted = aesgcm.encrypt(nonce=nonce, data=plaintext, associated_data=None)
    return EncryptedPayload(
        nonce_b64=base64.b64encode(nonce).decode("utf-8"),
        ciphertext_b64=base64.b64encode(encrypted[:-16]).decode("utf-8"),
        tag_b64=base64.b64encode(encrypted[-16:]).decode("utf-8"),
    )


def decrypt_bytes(*, nonce_b64: str, ciphertext_b64: str, tag_b64: str, aes_key: bytes) -> bytes:
    nonce = base64.b64decode(nonce_b64, validate=True)
    ciphertext = base64.b64decode(ciphertext_b64, validate=True)
    tag = base64.b64decode(tag_b64, validate=True)
    aesgcm = AESGCM(aes_key)
    return aesgcm.decrypt(nonce=nonce, data=ciphertext + tag, associated_data=None)
