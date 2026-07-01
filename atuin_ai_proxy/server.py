from __future__ import annotations

import json
import logging
import signal
import socket
import time
import uuid
from json import JSONDecodeError
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import urlparse

from .backend import BackendHTTPError, ResponsesBackend
from .auth import AuthError
from .diagnostics import (
    TRACE_LEVEL,
    configure_trace_logging,
    log_level_value,
    new_request_id,
    sanitized_json_excerpt,
    sanitized_text_excerpt,
    structured_error,
)
from .protocol import (
    build_chat_completions_request,
    build_responses_request,
    encode_sse_event,
    resolve_model,
    translate_chat_completions_events,
    translate_responses_events,
)
from .settings import Settings


BackendFactory = Callable[[Settings], Any]
logger = logging.getLogger(__name__)


def _raise_keyboard_interrupt(_signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt


class ProxyHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        settings: Settings,
        backend_factory: BackendFactory = ResponsesBackend,
    ) -> None:
        super().__init__(server_address, ProxyRequestHandler)
        self.settings = settings
        self.backend_factory = backend_factory


class ProxyRequestHandler(BaseHTTPRequestHandler):
    server: ProxyHTTPServer

    def do_GET(self) -> None:
        request_id = new_request_id()
        if urlparse(self.path).path == "/healthz":
            self._send_json(HTTPStatus.OK, {"ok": True}, request_id=request_id)
            return
        self._send_error(
            HTTPStatus.NOT_FOUND,
            "not_found",
            "not found",
            request_id,
        )

    def do_POST(self) -> None:
        request_id = new_request_id()
        started = time.monotonic()
        if urlparse(self.path).path != "/api/cli/chat":
            self._send_error(
                HTTPStatus.NOT_FOUND,
                "not_found",
                "not found",
                request_id,
                started=started,
            )
            return
        logger.info(
            "Chat request started request_id=%s client=%s",
            request_id,
            self.client_address[0],
        )
        if not self._is_authorized():
            self._send_error(
                HTTPStatus.UNAUTHORIZED,
                "unauthorized",
                "unauthorized",
                request_id,
                started=started,
            )
            return

        try:
            atuin_request = self._read_json_body()
            session_id = str(atuin_request.get("session_id") or uuid.uuid4())
            resolve_model(atuin_request, self.server.settings)
            backend = self.server.backend_factory(self.server.settings)
            upstream_api, translate_events, upstream_events = self._open_upstream_stream(
                backend,
                atuin_request,
                request_id,
            )
        except JSONDecodeError as exc:
            self._send_error(
                HTTPStatus.BAD_REQUEST,
                "invalid_json",
                f"Invalid JSON request body: {exc.msg}",
                request_id,
                started=started,
            )
            return
        except ValueError as exc:
            message = str(exc)
            code = "missing_model" if message.startswith("MODEL ") else "invalid_request"
            self._send_error(
                HTTPStatus.BAD_REQUEST,
                code,
                message,
                request_id,
                started=started,
            )
            return
        except AuthError as exc:
            self._send_error(
                HTTPStatus.BAD_GATEWAY,
                "auth_error",
                str(exc),
                request_id,
                started=started,
            )
            return
        except BackendHTTPError as exc:
            details = sanitized_text_excerpt(
                exc.body,
                self.server.settings.trace_payload_bytes,
            )
            self._send_error(
                HTTPStatus.BAD_GATEWAY,
                "upstream_http_error",
                f"{self.server.settings.backend} backend returned HTTP {exc.status}",
                request_id,
                details=details,
                upstream_status=exc.status,
                started=started,
            )
            return
        except (TimeoutError, socket.timeout) as exc:
            self._send_error(
                HTTPStatus.GATEWAY_TIMEOUT,
                "upstream_timeout",
                f"Backend request timed out: {exc}",
                request_id,
                started=started,
            )
            return
        except OSError as exc:
            self._send_error(
                HTTPStatus.BAD_GATEWAY,
                "upstream_network_error",
                f"Backend network error: {exc}",
                request_id,
                started=started,
            )
            return
        except Exception as exc:
            logger.exception("Failed to start chat stream request_id=%s", request_id)
            self._send_error(
                HTTPStatus.BAD_GATEWAY,
                "upstream_protocol_error",
                str(exc),
                request_id,
                started=started,
            )
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("x-atuin-ai-session-id", session_id)
        self.send_header("X-Request-ID", request_id)
        self.end_headers()
        try:
            logged_events = self._log_upstream_events(request_id, upstream_events)
            for chunk in translate_events(logged_events, session_id):
                self.wfile.write(chunk)
                self.wfile.flush()
        except Exception as exc:
            logger.exception("Streaming backend failed request_id=%s", request_id)
            self.wfile.write(
                encode_sse_event(
                    "error",
                    {
                        "code": "upstream_protocol_error",
                        "message": str(exc),
                        "request_id": request_id,
                    },
                )
            )
            self.wfile.flush()
        finally:
            logger.info(
                "Chat request finished request_id=%s status=%s elapsed_ms=%d",
                request_id,
                HTTPStatus.OK,
                self._elapsed_ms(started),
            )

    def log_message(self, format: str, *args: Any) -> None:
        logger.info("%s - %s", self.address_string(), format % args)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        body = self.rfile.read(length)
        if not body:
            return {}
        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _is_authorized(self) -> bool:
        expected = self.server.settings.atuin_proxy_token
        if not expected:
            return True
        return self.headers.get("Authorization") == f"Bearer {expected}"

    def _send_json(
        self,
        status: HTTPStatus,
        payload: dict[str, Any],
        *,
        request_id: str,
    ) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Request-ID", request_id)
        self.end_headers()
        self.wfile.write(body)

    def _send_error(
        self,
        status: HTTPStatus,
        code: str,
        message: str,
        request_id: str,
        *,
        details: str | None = None,
        upstream_status: int | None = None,
        started: float | None = None,
    ) -> None:
        payload = structured_error(
            code,
            message,
            request_id,
            details=details,
            upstream_status=upstream_status,
        )
        logger.error(
            "Request failed request_id=%s status=%s code=%s upstream_status=%s elapsed_ms=%s",
            request_id,
            status,
            code,
            upstream_status,
            self._elapsed_ms(started) if started is not None else None,
        )
        self._send_json(status, payload, request_id=request_id)

    def _open_upstream_stream(
        self,
        backend: Any,
        atuin_request: dict[str, Any],
        request_id: str,
    ) -> tuple[str, Callable[[Any, str], Any], Any]:
        candidates = self.server.settings.openai_api_candidates()
        for index, upstream_api in enumerate(candidates):
            self._log_request_summary(request_id, atuin_request, upstream_api)
            upstream_body, translate_events = self._build_upstream_request(
                atuin_request,
                request_id,
                upstream_api,
            )
            try:
                _status, _headers, upstream_events = backend.open_stream(
                    upstream_body,
                    request_id=request_id,
                    api=upstream_api,
                )
                return upstream_api, translate_events, upstream_events
            except BackendHTTPError as exc:
                if not self._should_fallback_to_next_api(exc, index, candidates):
                    raise
                next_api = candidates[index + 1]
                logger.info(
                    "Retrying backend request with fallback API request_id=%s "
                    "failed_api=%s fallback_api=%s upstream_status=%s",
                    request_id,
                    upstream_api,
                    next_api,
                    exc.status,
                )
        raise RuntimeError("No upstream API candidates configured")

    def _build_upstream_request(
        self,
        atuin_request: dict[str, Any],
        request_id: str,
        upstream_api: str,
    ) -> tuple[dict[str, Any], Callable[[Any, str], Any]]:
        if upstream_api == "chat_completions":
            return (
                build_chat_completions_request(
                    atuin_request,
                    self.server.settings,
                    request_id=request_id,
                ),
                translate_chat_completions_events,
            )
        return (
            build_responses_request(
                atuin_request,
                self.server.settings,
                request_id=request_id,
            ),
            translate_responses_events,
        )

    @staticmethod
    def _should_fallback_to_next_api(
        exc: BackendHTTPError,
        index: int,
        candidates: tuple[str, ...],
    ) -> bool:
        return index + 1 < len(candidates) and exc.status in {400, 404, 405, 422}

    def _log_request_summary(
        self,
        request_id: str,
        atuin_request: dict[str, Any],
        upstream_api: str,
    ) -> None:
        config = atuin_request.get("config") if isinstance(atuin_request.get("config"), dict) else {}
        model_source = "atuin" if config.get("model") else "proxy"
        logger.debug(
            "Chat request summary request_id=%s backend=%s upstream_api=%s model_source=%s invocation_id=%s has_session_id=%s",
            request_id,
            self.server.settings.backend,
            upstream_api,
            model_source,
            atuin_request.get("invocation_id"),
            bool(atuin_request.get("session_id")),
        )
        if logger.isEnabledFor(TRACE_LEVEL):
            logger.log(
                TRACE_LEVEL,
                "Atuin request payload request_id=%s body=%s",
                request_id,
                sanitized_json_excerpt(
                    atuin_request,
                    self.server.settings.trace_payload_bytes,
                ),
            )

    def _log_upstream_events(self, request_id: str, upstream_events: Any) -> Any:
        for event, data in upstream_events:
            logger.debug(
                "Upstream SSE event request_id=%s event=%s",
                request_id,
                event,
            )
            if logger.isEnabledFor(TRACE_LEVEL):
                logger.log(
                    TRACE_LEVEL,
                    "Upstream SSE payload request_id=%s event=%s data=%s",
                    request_id,
                    event,
                    sanitized_json_excerpt(
                        data,
                        self.server.settings.trace_payload_bytes,
                    ),
                )
            yield event, data

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return int((time.monotonic() - started) * 1000)


def run_server(settings: Settings) -> None:
    configure_trace_logging()
    logging.basicConfig(
        level=log_level_value(settings.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    server = ProxyHTTPServer((settings.host, settings.port), settings)
    logging.info("Listening on %s:%s", settings.host, settings.port)
    previous_sigterm_handler = signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down")
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
        server.server_close()
