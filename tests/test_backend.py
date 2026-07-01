import unittest
from unittest.mock import patch

from atuin_ai_proxy.backend import ResponsesBackend
from atuin_ai_proxy.settings import Settings


class BackendTests(unittest.TestCase):
    def test_open_stream_posts_chat_completions_to_chat_endpoint(self) -> None:
        response = FakeResponse([b"data: [DONE]\n", b"\n"])
        connection = FakeConnection()

        with patch(
            "atuin_ai_proxy.backend._post",
            return_value=(response, connection),
        ) as post:
            status, _headers, events = ResponsesBackend(
                Settings(
                    backend="openai",
                    model="chat-model",
                    openai_api_key="sk-test",
                    openai_base_url="https://example.test/v1",
                )
            ).open_stream({}, request_id="req-test", api="chat_completions")

        self.assertEqual(status, 200)
        self.assertEqual(post.call_args.args[0], "https://example.test/v1/chat/completions")
        self.assertEqual(list(events), [("message", {"message": "[DONE]"})])
        self.assertTrue(connection.closed)


class FakeResponse:
    status = 200

    def __init__(self, lines: list[bytes]) -> None:
        self.lines = lines

    def getheaders(self) -> list[tuple[str, str]]:
        return []

    def readline(self) -> bytes:
        if not self.lines:
            return b""
        return self.lines.pop(0)


class FakeConnection:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


if __name__ == "__main__":
    unittest.main()
