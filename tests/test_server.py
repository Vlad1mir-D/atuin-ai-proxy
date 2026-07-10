import json
import logging
import signal
import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

from atuin_ai_proxy.backend import BackendHTTPError
from atuin_ai_proxy.server import ProxyRequestHandler, run_server
from atuin_ai_proxy.settings import Settings


class FakeBackend:
    def open_stream(self, body, request_id=None, api="responses"):
        self.body = body
        self.api = api
        if api == "chat_completions":
            return 200, {}, iter(
                [
                    (
                        "message",
                        {
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": "hello"},
                                    "finish_reason": None,
                                }
                            ]
                        },
                    ),
                    ("message", {"message": "[DONE]"}),
                ]
            )
        return 200, {}, iter(
            [
                ("response.output_text.delta", {"delta": "hello"}),
                ("response.completed", {}),
            ]
        )


class ChatCompletionsFallbackBackend:
    def __init__(self) -> None:
        self.calls = []

    def open_stream(self, body, request_id=None, api="responses"):
        self.calls.append((api, body))
        if api == "chat_completions":
            raise BackendHTTPError(400, '{"detail":"chat completions rejected"}')
        return 200, {}, iter(
            [
                ("response.output_text.delta", {"delta": "fallback"}),
                ("response.completed", {}),
            ]
        )


class FailingHTTPBackend:
    def open_stream(self, body, request_id=None, api="responses"):
        raise BackendHTTPError(
            400,
            json.dumps(
                {
                    "error": {
                        "message": "invalid backend payload",
                        "access_token": "at-secret",
                        "account_id": "acct-secret",
                    }
                }
            ),
        )


class StreamingFailureBackend:
    def open_stream(self, body, request_id=None, api="responses"):
        def events():
            if api == "chat_completions":
                yield (
                    "message",
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "before failure"},
                                "finish_reason": None,
                            }
                        ]
                    },
                )
            else:
                yield ("response.output_text.delta", {"delta": "before failure"})
            raise RuntimeError("upstream stream broke")

        return 200, {}, events()


class ServerTests(unittest.TestCase):
    def setUp(self) -> None:
        # Expected failure-path logs should not look like test-runner failures.
        self.enterContext(
            patch.object(logging.getLogger("atuin_ai_proxy.server"), "disabled", True)
        )
        self.enterContext(patch("atuin_ai_proxy.server.logging.basicConfig"))
        self.enterContext(patch("atuin_ai_proxy.server.logging.info"))

    def test_healthz_returns_ok(self) -> None:
        backend = FakeBackend()
        response = handle_request(
            b"GET /healthz HTTP/1.1\r\nHost: localhost\r\n\r\n",
            Settings(backend="openai", model="gpt-test", openai_api_key="sk-test"),
            lambda _settings: backend,
        )

        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(response.body), {"ok": True})

    def test_chat_endpoint_streams_atuin_events(self) -> None:
        backend = FakeBackend()
        response = handle_request(
            _chat_request(token="proxy-token"),
            Settings(
                backend="openai",
                model="gpt-test",
                openai_api_key="sk-test",
                atuin_proxy_token="proxy-token",
            ),
            lambda _settings: backend,
        )

        body = response.body.decode()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.headers["content-type"], "text/event-stream")
        self.assertIn("x-atuin-ai-session-id", response.headers)
        self.assertIn('event: text\ndata: {"content":"hello"}\n\n', body)
        self.assertIn("event: done", body)
        self.assertEqual(backend.body["model"], "gpt-test")

    def test_chat_endpoint_auto_uses_chat_completions_first(self) -> None:
        backend = FakeBackend()
        response = handle_request(
            _chat_request(),
            Settings(
                backend="openai",
                model="gpt-test",
                openai_api_key="sk-test",
            ),
            lambda _settings: backend,
        )

        body = response.body.decode()
        self.assertEqual(response.status, 200)
        self.assertEqual(backend.api, "chat_completions")
        self.assertIn("messages", backend.body)
        self.assertNotIn("input", backend.body)
        self.assertIn('event: text\ndata: {"content":"hello"}\n\n', body)
        self.assertIn("event: done", body)

    def test_chat_endpoint_auto_falls_back_to_responses_after_chat_rejection(self) -> None:
        backend = ChatCompletionsFallbackBackend()
        response = handle_request(
            _chat_request(),
            Settings(
                backend="openai",
                model="gpt-test",
                openai_api_key="sk-test",
            ),
            lambda _settings: backend,
        )

        body = response.body.decode()
        self.assertEqual(response.status, 200)
        self.assertEqual([api for api, _body in backend.calls], ["chat_completions", "responses"])
        self.assertIn("messages", backend.calls[0][1])
        self.assertIn("input", backend.calls[1][1])
        self.assertIn('event: text\ndata: {"content":"fallback"}\n\n', body)
        self.assertIn("event: done", body)

    def test_chat_endpoint_rejects_wrong_proxy_token(self) -> None:
        response = handle_request(
            (
                b"POST /api/cli/chat HTTP/1.1\r\n"
                b"Host: localhost\r\n"
                b"Authorization: Bearer wrong\r\n"
                b"Content-Length: 2\r\n"
                b"\r\n{}"
            ),
            Settings(
                backend="openai",
                model="gpt-test",
                openai_api_key="sk-test",
                atuin_proxy_token="proxy-token",
            ),
            lambda _settings: FakeBackend(),
        )

        self.assertEqual(response.status, 401)
        self.assertIn("x-request-id", response.headers)
        payload = json.loads(response.body)
        self.assertEqual(payload["error"]["code"], "unauthorized")
        self.assertEqual(payload["error"]["request_id"], response.headers["x-request-id"])

    def test_chat_endpoint_returns_structured_missing_model_error(self) -> None:
        response = handle_request(
            _chat_request(),
            Settings(backend="openai", openai_api_key="sk-test"),
            lambda _settings: FakeBackend(),
        )

        self.assertEqual(response.status, 400)
        self.assertIn("x-request-id", response.headers)
        error = json.loads(response.body)["error"]
        self.assertEqual(error["code"], "missing_model")
        self.assertIn("MODEL must be configured", error["message"])
        self.assertEqual(error["request_id"], response.headers["x-request-id"])

    def test_chat_endpoint_returns_sanitized_upstream_http_error(self) -> None:
        response = handle_request(
            _chat_request(),
            Settings(backend="openai", model="gpt-test", openai_api_key="sk-test"),
            lambda _settings: FailingHTTPBackend(),
        )

        self.assertEqual(response.status, 502)
        self.assertIn("x-request-id", response.headers)
        error = json.loads(response.body)["error"]
        self.assertEqual(error["code"], "upstream_http_error")
        self.assertEqual(error["upstream_status"], 400)
        self.assertIn("invalid backend payload", error["details"])
        self.assertIn("[REDACTED]", error["details"])
        self.assertNotIn("at-secret", error["details"])
        self.assertNotIn("acct-secret", error["details"])

    def test_streaming_failure_emits_error_event_with_request_id(self) -> None:
        response = handle_request(
            _chat_request(),
            Settings(backend="openai", model="gpt-test", openai_api_key="sk-test"),
            lambda _settings: StreamingFailureBackend(),
        )

        body = response.body.decode()
        self.assertEqual(response.status, 200)
        self.assertIn("x-request-id", response.headers)
        self.assertIn('event: text\ndata: {"content":"before failure"}\n\n', body)
        self.assertIn('"code":"upstream_protocol_error"', body)
        self.assertIn(f'"request_id":"{response.headers["x-request-id"]}"', body)

    def test_run_server_treats_sigterm_as_graceful_shutdown(self) -> None:
        servers = []
        sigterm_handlers = []

        class FakeHTTPServer:
            def __init__(self, _address, _settings) -> None:
                self.closed = False
                servers.append(self)

            def serve_forever(self) -> None:
                sigterm_handlers[-1](signal.SIGTERM, None)

            def server_close(self) -> None:
                self.closed = True

        def fake_signal(signum, handler):
            if signum == signal.SIGTERM:
                sigterm_handlers.append(handler)
            return signal.SIG_DFL

        with (
            patch("atuin_ai_proxy.server.ProxyHTTPServer", FakeHTTPServer),
            patch("signal.signal", fake_signal),
        ):
            run_server(Settings(backend="openai", model="gpt-test"))

        self.assertTrue(servers[0].closed)
        self.assertEqual(sigterm_handlers[-1], signal.SIG_DFL)


class NonClosingBytesIO(BytesIO):
    def close(self) -> None:
        pass


class FakeSocket:
    def __init__(self, request: bytes) -> None:
        self.input = NonClosingBytesIO(request)
        self.output = NonClosingBytesIO()

    def makefile(self, mode: str, *args, **kwargs):
        if "r" in mode:
            return self.input
        return self.output

    def sendall(self, data: bytes) -> None:
        self.output.write(data)


class Response:
    def __init__(self, raw: bytes) -> None:
        head, _separator, body = raw.partition(b"\r\n\r\n")
        lines = head.decode().split("\r\n")
        self.status = int(lines[0].split()[1])
        self.headers = {}
        for line in lines[1:]:
            key, _separator, value = line.partition(":")
            self.headers[key.lower()] = value.strip()
        self.body = body


def _chat_request(*, token: str | None = None) -> bytes:
    payload = json.dumps({"messages": [], "config": {}, "invocation_id": "i"})
    authorization = f"Authorization: Bearer {token}\r\n" if token else ""
    return (
        "POST /api/cli/chat HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Content-Type: application/json\r\n"
        f"{authorization}"
        f"Content-Length: {len(payload.encode())}\r\n"
        "\r\n"
        f"{payload}"
    ).encode()


def handle_request(request: bytes, settings: Settings, backend_factory) -> Response:
    fake_socket = FakeSocket(request)
    server = SimpleNamespace(settings=settings, backend_factory=backend_factory)
    ProxyRequestHandler(fake_socket, ("127.0.0.1", 12345), server)
    return Response(fake_socket.output.getvalue())


if __name__ == "__main__":
    unittest.main()
