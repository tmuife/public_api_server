"""
crypto.py — AES-256-GCM 加密/解密工具
"""
import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from config import settings


def encrypt(data: bytes) -> bytes:
    """返回 IV(12字节) + 密文，可直接存储或传输"""
    iv = os.urandom(12)
    ct = AESGCM(settings.AES_KEY).encrypt(iv, data, None)
    return iv + ct


def decrypt(data: bytes) -> bytes:
    """接受 IV(12字节) + 密文，返回原始数据"""
    iv, ct = data[:12], data[12:]
    return AESGCM(settings.AES_KEY).decrypt(iv, ct, None)
