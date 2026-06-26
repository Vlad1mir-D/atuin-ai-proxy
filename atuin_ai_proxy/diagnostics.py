from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any


TRACE_LEVEL = 5
DEFAULT_TRACE_PAYLOAD_BYTES = 4096
REDACTED = "[REDACTED]"

_SENSITIVE_KEYS = {
    "authorization",
    "api_key",
    "access_token",
    "refresh_token",
    "id_token",
    "token",
    "openai_api_key",
    "codex_access_token",
    "account_id",
    "chatgpt_account_id",
    "codex_account_id",
}

_TOKEN_PATTERNS = [
    re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9._-]+"),
    re.compile(r"\bat-[A-Za-z0-9._-]+"),
    re.compile(r"\bacct_[A-Za-z0-9._-]+"),
]


def configure_trace_logging() -> None:
    logging.addLevelName(TRACE_LEVEL, "TRACE")


def log_level_value(name: str) -> int:
    normalized = name.strip().upper()
    if normalized == "TRACE":
        return TRACE_LEVEL
    return int(getattr(logging, normalized, logging.INFO))


def new_request_id() -> str:
    return "req_" + uuid.uuid4().hex[:12]


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            key_name = str(key).lower()
            if key_name in _SENSITIVE_KEYS:
                redacted[key] = REDACTED
            else:
                redacted[key] = redact(item)
        return redacted
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact(item) for item in value)
    return value


def sanitized_text_excerpt(text: str, limit: int = DEFAULT_TRACE_PAYLOAD_BYTES) -> str:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        sanitized = _redact_text(text)
    else:
        sanitized = json.dumps(
            redact(parsed),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return _truncate(sanitized, limit)


def sanitized_json_excerpt(value: Any, limit: int = DEFAULT_TRACE_PAYLOAD_BYTES) -> str:
    sanitized = json.dumps(
        redact(value),
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return _truncate(sanitized, limit)


def structured_error(
    code: str,
    message: str,
    request_id: str,
    *,
    details: str | None = None,
    upstream_status: int | None = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {
        "code": code,
        "message": message,
        "request_id": request_id,
    }
    if upstream_status is not None:
        error["upstream_status"] = upstream_status
    if details:
        error["details"] = details
    return {"error": error}


def _redact_text(text: str) -> str:
    sanitized = text
    for pattern in _TOKEN_PATTERNS:
        sanitized = pattern.sub(REDACTED, sanitized)
    return sanitized


def _truncate(text: str, limit: int) -> str:
    if len(text.encode("utf-8")) <= limit:
        return text
    encoded = text.encode("utf-8")[:limit]
    truncated = encoded.decode("utf-8", errors="ignore")
    return truncated + "...[truncated]"
