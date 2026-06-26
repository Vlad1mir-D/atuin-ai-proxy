from __future__ import annotations

import json
from collections.abc import Iterator
from http.client import HTTPConnection, HTTPSConnection
from typing import Any
from urllib.parse import urljoin, urlparse

from . import __version__
from .auth import AuthError, provider_for_settings
from .settings import Settings


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
        self, body: dict[str, Any]
    ) -> tuple[int, dict[str, str], Iterator[tuple[str, dict[str, Any]]]]:
        try:
            return self._open_stream_once(body)
        except BackendHTTPError as exc:
            if exc.status == 401 and self.auth_provider.refresh():
                return self._open_stream_once(body)
            raise

    def _open_stream_once(
        self, body: dict[str, Any]
    ) -> tuple[int, dict[str, str], Iterator[tuple[str, dict[str, Any]]]]:
        url = urljoin(self.settings.backend_base_url.rstrip("/") + "/", "responses")
        headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "User-Agent": f"atuin-ai-proxy/{__version__}",
        }
        try:
            headers.update(self.auth_provider.headers())
        except AuthError:
            raise

        body_bytes = json.dumps(body, separators=(",", ":")).encode("utf-8")
        response, connection = _post(url, headers, body_bytes, self.settings.request_timeout_seconds)
        response_headers = {key.lower(): value for key, value in response.getheaders()}
        if not 200 <= response.status < 300:
            error_body = response.read().decode("utf-8", errors="replace")
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
