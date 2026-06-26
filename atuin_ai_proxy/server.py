from __future__ import annotations

import json
import logging
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import urlparse

from .backend import BackendHTTPError, ResponsesBackend
from .protocol import build_responses_request, encode_sse_event, translate_responses_events
from .settings import Settings


BackendFactory = Callable[[Settings], Any]


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
        if urlparse(self.path).path == "/healthz":
            self._send_json(HTTPStatus.OK, {"ok": True})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/cli/chat":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        if not self._is_authorized():
            self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return

        try:
            atuin_request = self._read_json_body()
            session_id = str(atuin_request.get("session_id") or uuid.uuid4())
            responses_body = build_responses_request(atuin_request, self.server.settings)
            backend = self.server.backend_factory(self.server.settings)
            _status, _headers, upstream_events = backend.open_stream(responses_body)
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except BackendHTTPError as exc:
            self._send_json(
                HTTPStatus.BAD_GATEWAY,
                {"error": f"backend HTTP {exc.status}", "body": exc.body},
            )
            return
        except Exception as exc:
            logging.exception("Failed to start chat stream")
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("x-atuin-ai-session-id", session_id)
        self.end_headers()
        try:
            for chunk in translate_responses_events(upstream_events, session_id):
                self.wfile.write(chunk)
                self.wfile.flush()
        except Exception as exc:
            logging.exception("Streaming backend failed")
            self.wfile.write(encode_sse_event("error", {"message": str(exc)}))
            self.wfile.flush()

    def log_message(self, format: str, *args: Any) -> None:
        logging.info("%s - %s", self.address_string(), format % args)

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

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    server = ProxyHTTPServer((settings.host, settings.port), settings)
    logging.info("Listening on %s:%s", settings.host, settings.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down")
    finally:
        server.server_close()
