from __future__ import annotations

import json
import logging
import secrets
import time
import uuid
from collections.abc import Iterator
from copy import deepcopy
from http.client import HTTPConnection, HTTPSConnection
from threading import Lock
from typing import Any
from urllib.parse import urljoin, urlparse

from . import __version__
from .auth import AuthError, provider_for_settings
from .diagnostics import TRACE_LEVEL, redact, sanitized_json_excerpt, sanitized_text_excerpt
from .settings import Settings


logger = logging.getLogger(__name__)


CODEX_COMPATIBILITY_VERSION = "0.144.0"
RESPONSES_LITE_HEADER = "X-OpenAI-Internal-Codex-Responses-Lite"
RESPONSES_LITE_MODELS = frozenset(
    {
        "gpt-5.6-luna",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
    }
)
_learned_responses_lite_models: set[tuple[str, str]] = set()
_learned_responses_lite_models_lock = Lock()


class BackendHTTPError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"Backend returned HTTP {status}: {body}")
        self.status = status
        self.body = body


class ResponsesBackend:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.auth_provider = provider_for_settings(settings)

    def open_stream(
        self,
        body: dict[str, Any],
        request_id: str | None = None,
        api: str = "responses",
        session_id: str | None = None,
    ) -> tuple[int, dict[str, str], Iterator[tuple[str, dict[str, Any]]]]:
        try:
            return self._open_stream_with_compatibility(
                body,
                request_id,
                api,
                session_id,
            )
        except BackendHTTPError as exc:
            if exc.status == 401 and self.auth_provider.refresh():
                logger.info(
                    "Retrying backend request after credential refresh request_id=%s",
                    request_id,
                )
                return self._open_stream_with_compatibility(
                    body,
                    request_id,
                    api,
                    session_id,
                )
            raise

    def _open_stream_with_compatibility(
        self,
        body: dict[str, Any],
        request_id: str | None,
        api: str,
        session_id: str | None,
    ) -> tuple[int, dict[str, str], Iterator[tuple[str, dict[str, Any]]]]:
        model_key = self._responses_lite_model_key(body, api)
        use_responses_lite = bool(
            model_key and _is_known_responses_lite_model(model_key)
        )
        try:
            return self._open_stream_once(
                body,
                request_id,
                api,
                use_responses_lite=use_responses_lite,
                source_session_id=session_id,
            )
        except BackendHTTPError as exc:
            if use_responses_lite or not model_key or not _is_model_not_found(exc):
                raise

        logger.info(
            "Retrying Codex request with Responses Lite compatibility "
            "request_id=%s model=%s",
            request_id,
            model_key[1],
        )
        result = self._open_stream_once(
            body,
            request_id,
            api,
            use_responses_lite=True,
            source_session_id=session_id,
        )
        _remember_responses_lite_model(model_key)
        return result

    def _responses_lite_model_key(
        self,
        body: dict[str, Any],
        api: str,
    ) -> tuple[str, str] | None:
        model = body.get("model")
        if self.settings.backend == "openai" or api != "responses" or not model:
            return None
        return self.settings.backend_base_url, str(model)

    def _open_stream_once(
        self,
        body: dict[str, Any],
        request_id: str | None,
        api: str,
        *,
        use_responses_lite: bool = False,
        source_session_id: str | None = None,
    ) -> tuple[int, dict[str, str], Iterator[tuple[str, dict[str, Any]]]]:
        endpoint = "chat/completions" if api == "chat_completions" else "responses"
        url = urljoin(self.settings.backend_base_url.rstrip("/") + "/", endpoint)
        headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "User-Agent": f"atuin-ai-proxy/{__version__}",
        }
        try:
            headers.update(self.auth_provider.headers())
        except AuthError:
            raise

        request_body = body
        if use_responses_lite:
            responses_lite_session_id = _responses_lite_session_id(
                source_session_id,
            )
            request_body = _responses_lite_body(body, responses_lite_session_id)
            headers.update(
                {
                    "session-id": responses_lite_session_id,
                    "x-session-affinity": responses_lite_session_id,
                    "version": CODEX_COMPATIBILITY_VERSION,
                    RESPONSES_LITE_HEADER: "true",
                }
            )

        body_bytes = json.dumps(request_body, separators=(",", ":")).encode("utf-8")
        logger.debug(
            "Opening backend stream request_id=%s backend=%s path=%s body_bytes=%s tools=%s",
            request_id,
            self.settings.backend,
            _safe_url_path(url),
            len(body_bytes),
            _tool_names(request_body),
        )
        if logger.isEnabledFor(TRACE_LEVEL):
            logger.log(
                TRACE_LEVEL,
                "Backend request payload request_id=%s body=%s",
                request_id,
                sanitized_json_excerpt(
                    request_body,
                    self.settings.trace_payload_bytes,
                ),
            )
            logger.log(
                TRACE_LEVEL,
                "Backend request headers request_id=%s headers=%s",
                request_id,
                redact(headers),
            )
        response, connection = _post(
            url,
            headers,
            body_bytes,
            self.settings.request_timeout_seconds,
        )
        response_headers = {key.lower(): value for key, value in response.getheaders()}
        logger.debug(
            "Backend stream opened request_id=%s upstream_status=%s",
            request_id,
            response.status,
        )
        if not 200 <= response.status < 300:
            error_body = response.read().decode("utf-8", errors="replace")
            if logger.isEnabledFor(TRACE_LEVEL):
                logger.log(
                    TRACE_LEVEL,
                    "Backend error body request_id=%s upstream_status=%s body=%s",
                    request_id,
                    response.status,
                    sanitized_text_excerpt(
                        error_body,
                        self.settings.trace_payload_bytes,
                    ),
                )
            connection.close()
            raise BackendHTTPError(response.status, error_body)

        def events() -> Iterator[tuple[str, dict[str, Any]]]:
            try:
                yield from parse_sse_events(_response_lines(response))
            finally:
                connection.close()

        return response.status, response_headers, events()


def parse_sse_events(lines: Iterator[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    event_name = "message"
    data_lines: list[str] = []
    for line in lines:
        line = line.rstrip("\r\n")
        if not line:
            if data_lines:
                raw_data = "\n".join(data_lines)
                try:
                    data = json.loads(raw_data)
                except json.JSONDecodeError:
                    data = {"message": raw_data}
                yield event_name, data
            event_name = "message"
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if not separator:
            continue
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_name = value
        elif field == "data":
            data_lines.append(value)

    if data_lines:
        raw_data = "\n".join(data_lines)
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError:
            data = {"message": raw_data}
        yield event_name, data


def _post(
    url: str, headers: dict[str, str], body: bytes, timeout: int
) -> tuple[Any, HTTPConnection | HTTPSConnection]:
    parsed = urlparse(url)
    connection_cls = HTTPSConnection if parsed.scheme == "https" else HTTPConnection
    port = parsed.port
    host = parsed.hostname
    if not host:
        raise ValueError(f"Invalid backend URL: {url}")
    connection = connection_cls(host, port=port, timeout=timeout)
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    connection.request("POST", path, body=body, headers=headers)
    return connection.getresponse(), connection


def _response_lines(response: Any) -> Iterator[str]:
    while True:
        line = response.readline()
        if not line:
            break
        yield line.decode("utf-8", errors="replace")


def _safe_url_path(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path or "/"
    if parsed.query:
        return path + "?[query]"
    return path


def _tool_names(body: dict[str, Any]) -> list[str]:
    tools = body.get("tools")
    if not isinstance(tools, list):
        return []
    names = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("name"):
            names.append(str(tool["name"]))
            continue
        function = tool.get("function")
        if isinstance(function, dict) and function.get("name"):
            names.append(str(function["name"]))
    return names


def _is_model_not_found(exc: BackendHTTPError) -> bool:
    return exc.status == 404 and "model not found" in exc.body.lower()


def _is_known_responses_lite_model(model_key: tuple[str, str]) -> bool:
    if model_key[1] in RESPONSES_LITE_MODELS:
        return True
    with _learned_responses_lite_models_lock:
        return model_key in _learned_responses_lite_models


def _remember_responses_lite_model(model_key: tuple[str, str]) -> None:
    with _learned_responses_lite_models_lock:
        _learned_responses_lite_models.add(model_key)


def _responses_lite_session_id(
    source_session_id: str | None,
) -> str:
    if not source_session_id:
        return new_uuid7()

    canonical_uuid7 = canonical_uuid7_string(source_session_id)
    if canonical_uuid7:
        return canonical_uuid7
    return new_uuid7()


def _responses_lite_body(body: dict[str, Any], session_id: str) -> dict[str, Any]:
    request = deepcopy(body)
    input_items = request.get("input")
    tools = request.get("tools", [])
    instructions = request.get("instructions")
    if not isinstance(input_items, list):
        raise ValueError("Responses Lite requires an input array")
    if not isinstance(tools, list):
        raise ValueError("Responses Lite requires a tools array")
    if instructions is not None and not isinstance(instructions, str):
        raise ValueError("Responses Lite requires string instructions")

    developer_items: list[dict[str, Any]] = [
        {"type": "additional_tools", "role": "developer", "tools": tools}
    ]
    if instructions:
        developer_items.append(
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": instructions}],
            }
        )
    request["input"] = developer_items + input_items
    request.pop("tools", None)
    request.pop("instructions", None)
    request["tool_choice"] = "auto"
    request["parallel_tool_calls"] = False
    request["prompt_cache_key"] = session_id
    reasoning = request.get("reasoning")
    request["reasoning"] = {
        **(reasoning if isinstance(reasoning, dict) else {}),
        "context": "all_turns",
    }
    _strip_image_detail(request["input"])
    return request


def _strip_image_detail(value: Any) -> None:
    if isinstance(value, list):
        for item in value:
            _strip_image_detail(item)
        return
    if not isinstance(value, dict):
        return
    if value.get("type") == "input_image":
        value.pop("detail", None)
    for item in value.values():
        _strip_image_detail(item)


def canonical_uuid7_string(value: str) -> str | None:
    try:
        parsed = uuid.UUID(value)
    except ValueError:
        return None
    if parsed.version != 7:
        return None
    return str(parsed)


def new_uuid7() -> str:
    timestamp_ms = int(time.time() * 1000) & ((1 << 48) - 1)
    random_a = secrets.randbits(12)
    random_b = secrets.randbits(62)
    value = (
        (timestamp_ms << 80)
        | (0x7 << 76)
        | (random_a << 64)
        | (0b10 << 62)
        | random_b
    )
    return str(uuid.UUID(int=value))
