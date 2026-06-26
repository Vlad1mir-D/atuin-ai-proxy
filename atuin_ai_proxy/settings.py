from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .diagnostics import DEFAULT_TRACE_PAYLOAD_BYTES


DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_CODEX_ISSUER = "https://auth.openai.com"
CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
VALID_BACKENDS = {"openai", "codex-token", "codex-oauth"}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


@dataclass(slots=True)
class Settings:
    backend: str = "openai"
    model: str | None = None
    atuin_proxy_token: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str = DEFAULT_OPENAI_BASE_URL
    codex_access_token: str | None = None
    codex_account_id: str | None = None
    codex_fedramp: bool = False
    codex_base_url: str = DEFAULT_CODEX_BASE_URL
    codex_home: str = "/data/codex"
    codex_auth_file: str | None = None
    codex_issuer: str = DEFAULT_CODEX_ISSUER
    codex_auth_api_base_url: str = DEFAULT_CODEX_ISSUER
    codex_client_id: str = CODEX_CLIENT_ID
    request_timeout_seconds: int = 120
    trace_payload_bytes: int = DEFAULT_TRACE_PAYLOAD_BYTES
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        self.backend = self.backend.strip().lower()
        if self.backend not in VALID_BACKENDS:
            valid = ", ".join(sorted(VALID_BACKENDS))
            raise ValueError(f"BACKEND must be one of: {valid}")

    @classmethod
    def from_env(cls) -> "Settings":
        codex_home = os.getenv("CODEX_HOME", "/data/codex")
        return cls(
            backend=os.getenv("BACKEND", "openai"),
            model=os.getenv("MODEL") or None,
            atuin_proxy_token=os.getenv("ATUIN_PROXY_TOKEN") or None,
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_base_url=os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
            codex_access_token=os.getenv("CODEX_ACCESS_TOKEN") or None,
            codex_account_id=os.getenv("CODEX_ACCOUNT_ID") or None,
            codex_fedramp=_env_bool("CODEX_FEDRAMP"),
            codex_base_url=os.getenv("CODEX_BASE_URL", DEFAULT_CODEX_BASE_URL),
            codex_home=codex_home,
            codex_auth_file=os.getenv("CODEX_AUTH_FILE") or None,
            codex_issuer=os.getenv("CODEX_ISSUER", DEFAULT_CODEX_ISSUER),
            codex_auth_api_base_url=os.getenv(
                "CODEX_AUTH_API_BASE_URL", DEFAULT_CODEX_ISSUER
            ),
            codex_client_id=os.getenv("CODEX_CLIENT_ID", CODEX_CLIENT_ID),
            request_timeout_seconds=_env_int("REQUEST_TIMEOUT_SECONDS", 120),
            trace_payload_bytes=_env_int(
                "TRACE_PAYLOAD_BYTES",
                DEFAULT_TRACE_PAYLOAD_BYTES,
            ),
            host=os.getenv("HOST", "127.0.0.1"),
            port=_env_int("PORT", 8000),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )

    @property
    def resolved_codex_auth_file(self) -> Path:
        if self.codex_auth_file:
            return Path(self.codex_auth_file).expanduser()
        return Path(self.codex_home).expanduser() / "auth.json"

    @property
    def backend_base_url(self) -> str:
        if self.backend == "openai":
            return self.openai_base_url
        return self.codex_base_url
