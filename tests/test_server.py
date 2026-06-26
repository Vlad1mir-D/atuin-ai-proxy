import json
import unittest
from io import BytesIO
from types import SimpleNamespace

from atuin_ai_proxy.server import ProxyRequestHandler
from atuin_ai_proxy.settings import Settings


class FakeBackend:
    def open_stream(self, body):
        self.body = body
        return 200, {}, iter(
            [
                (
                    "response.output_text.delta",
                    {"delta": "hello"},
                ),
                ("response.completed", {}),
            ]
        )


class ServerTests(unittest.TestCase):
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
        payload = json.dumps({"messages": [], "config": {}, "invocation_id": "i"})
        request = (
            "POST /api/cli/chat HTTP/1.1\r\n"
            "Host: localhost\r\n"
            "Content-Type: application/json\r\n"
            "Authorization: Bearer proxy-token\r\n"
            f"Content-Length: {len(payload.encode())}\r\n"
            "\r\n"
            f"{payload}"
        ).encode()
        response = handle_request(
            request,
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


def handle_request(request: bytes, settings: Settings, backend_factory) -> Response:
    fake_socket = FakeSocket(request)
    server = SimpleNamespace(settings=settings, backend_factory=backend_factory)
    ProxyRequestHandler(fake_socket, ("127.0.0.1", 12345), server)
    return Response(fake_socket.output.getvalue())


if __name__ == "__main__":
    unittest.main()
