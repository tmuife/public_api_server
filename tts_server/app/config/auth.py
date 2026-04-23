from __future__ import annotations

import os
import re
from dataclasses import dataclass

_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "no", "n", "off"}

# Compact placeholder tokens: lowercased and non-alphanumeric removed.
_PLACEHOLDER_API_KEY_TOKENS = {
    "apikey",
    "apiaccesskey",
    "yourapikey",
    "yourtoken",
    "changeme",
    "replacewithsecureapikey",
    "replacewithyourapikey",
    "placeholder",
    "default",
}


@dataclass(frozen=True)
class AuthConfig:
    api_key: str
    auth_required: bool
    docs_public_in_dev: bool = False

    @classmethod
    def from_env(cls) -> "AuthConfig":
        return cls(
            api_key=str(os.getenv("API_KEY", "")).strip(),
            auth_required=_read_bool_env("AUTH_REQUIRED", default=True),
            docs_public_in_dev=_read_bool_env("DOCS_PUBLIC_IN_DEV", default=False),
        )


def validate_auth_config(auth_config: AuthConfig) -> None:
    if not auth_config.auth_required:
        return
    if _is_placeholder_like_api_key(auth_config.api_key):
        raise RuntimeError(
            "Invalid auth configuration: AUTH_REQUIRED=true requires a non-placeholder API_KEY. "
            "Set API_KEY to a strong secret token."
        )


def _read_bool_env(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    normalized = str(raw).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return bool(default)


def _is_placeholder_like_api_key(value: str) -> bool:
    normalized = str(value).strip()
    if not normalized:
        return True

    compact = re.sub(r"[^a-z0-9]", "", normalized.lower())
    if compact in _PLACEHOLDER_API_KEY_TOKENS:
        return True

    # Catch generic keys like "replace_me", "your-key-here", etc.
    if "replace" in compact or "placeholder" in compact:
        return True
    if compact.startswith("your") and compact.endswith("key"):
        return True
    return False
