import json
import unittest
import uuid
from unittest.mock import patch

from atuin_ai_proxy import backend as backend_module
from atuin_ai_proxy.backend import ResponsesBackend
from atuin_ai_proxy.settings import Settings


class BackendTests(unittest.TestCase):
    def setUp(self) -> None:
        backend_module._learned_responses_lite_models.clear()

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

    def test_codex_model_not_found_negotiates_and_caches_responses_lite(self) -> None:
        model = "gpt-future-variant"
        original_body = {
            "model": model,
            "instructions": "Be concise.",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,test",
                            "detail": "high",
                        }
                    ],
                }
            ],
            "tools": [{"type": "function", "name": "noop"}],
            "stream": True,
        }
        settings = Settings(
            backend="codex-token",
            codex_access_token="token",
            codex_account_id="account",
            codex_base_url="https://example.test/backend-api/codex",
        )
        responses = [
            (
                FakeResponse(status=404, body=b'{"message":"Model not found"}'),
                FakeConnection(),
            ),
            (FakeResponse(), FakeConnection()),
            (FakeResponse(), FakeConnection()),
        ]

        with patch("atuin_ai_proxy.backend._post", side_effect=responses) as post:
            first = ResponsesBackend(settings).open_stream(original_body)
            second = ResponsesBackend(settings).open_stream(original_body)

        self.assertEqual(first[0], 200)
        self.assertEqual(second[0], 200)
        self.assertEqual(post.call_count, 3)
        self.assertNotIn(
            "X-OpenAI-Internal-Codex-Responses-Lite",
            post.call_args_list[0].args[1],
        )

        first_lite_headers = post.call_args_list[1].args[1]
        second_lite_headers = post.call_args_list[2].args[1]
        self.assertEqual(
            first_lite_headers["X-OpenAI-Internal-Codex-Responses-Lite"],
            "true",
        )
        self.assertEqual(first_lite_headers["version"], "0.144.0")
        self.assertEqual(uuid.UUID(first_lite_headers["session-id"]).version, 7)
        self.assertEqual(
            first_lite_headers["x-session-affinity"],
            first_lite_headers["session-id"],
        )
        self.assertEqual(
            second_lite_headers["X-OpenAI-Internal-Codex-Responses-Lite"],
            "true",
        )

        lite_body = json.loads(post.call_args_list[1].args[2])
        self.assertNotIn("tools", lite_body)
        self.assertNotIn("instructions", lite_body)
        self.assertEqual(lite_body["input"][0]["type"], "additional_tools")
        self.assertEqual(lite_body["input"][1]["role"], "developer")
        self.assertNotIn("detail", lite_body["input"][2]["content"][0])
        self.assertFalse(lite_body["parallel_tool_calls"])
        self.assertEqual(lite_body["reasoning"]["context"], "all_turns")
        self.assertEqual(
            lite_body["prompt_cache_key"],
            first_lite_headers["session-id"],
        )
        self.assertIn("tools", original_body)
        self.assertIn("detail", original_body["input"][0]["content"][0])

    def test_known_responses_lite_models_use_lite_on_first_request(self) -> None:
        settings = Settings(
            backend="codex-token",
            codex_access_token="token",
            codex_account_id="account",
            codex_base_url="https://example.test/backend-api/codex",
        )

        for model in ("gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"):
            with self.subTest(model=model):
                with patch(
                    "atuin_ai_proxy.backend._post",
                    return_value=(FakeResponse(), FakeConnection()),
                ) as post:
                    ResponsesBackend(settings).open_stream(
                        {"model": model, "input": [], "tools": []}
                    )

                self.assertEqual(post.call_count, 1)
                headers = post.call_args.args[1]
                posted_body = json.loads(post.call_args.args[2])
                self.assertEqual(
                    headers["X-OpenAI-Internal-Codex-Responses-Lite"],
                    "true",
                )
                self.assertEqual(posted_body["model"], model)
                self.assertEqual(posted_body["input"][0]["type"], "additional_tools")

    def test_unexpected_session_identity_is_regenerated(self) -> None:
        body = {"model": "gpt-5.6-luna", "input": [], "tools": []}
        settings = Settings(
            backend="codex-token",
            codex_access_token="token",
            codex_account_id="account",
            codex_base_url="https://example.test/backend-api/codex",
        )

        with patch(
            "atuin_ai_proxy.backend._post",
            side_effect=[
                (FakeResponse(), FakeConnection()),
                (FakeResponse(), FakeConnection()),
            ],
        ) as post:
            ResponsesBackend(settings).open_stream(body, session_id="atuin-session")
            ResponsesBackend(settings).open_stream(body, session_id="atuin-session")

        first_headers = post.call_args_list[0].args[1]
        second_headers = post.call_args_list[1].args[1]
        self.assertEqual(uuid.UUID(first_headers["session-id"]).version, 7)
        self.assertEqual(uuid.UUID(second_headers["session-id"]).version, 7)
        self.assertNotEqual(first_headers["session-id"], second_headers["session-id"])

    def test_uuid7_session_identity_is_reused_directly(self) -> None:
        body = {"model": "gpt-5.6-luna", "input": [], "tools": []}
        source_session_id = backend_module.new_uuid7()
        settings = Settings(
            backend="codex-token",
            codex_access_token="token",
            codex_account_id="account",
            codex_base_url="https://example.test/backend-api/codex",
        )

        with patch(
            "atuin_ai_proxy.backend._post",
            side_effect=[
                (FakeResponse(), FakeConnection()),
                (FakeResponse(), FakeConnection()),
            ],
        ) as post:
            ResponsesBackend(settings).open_stream(
                body,
                session_id=source_session_id,
            )
            ResponsesBackend(settings).open_stream(
                body,
                session_id=source_session_id,
            )

        first_headers = post.call_args_list[0].args[1]
        second_headers = post.call_args_list[1].args[1]
        self.assertEqual(first_headers["session-id"], source_session_id)
        self.assertEqual(second_headers["session-id"], source_session_id)

    def test_successful_codex_model_keeps_legacy_request_unchanged(self) -> None:
        body = {"model": "already-supported", "input": [], "tools": []}
        settings = Settings(
            backend="codex-token",
            codex_access_token="token",
            codex_account_id="account",
            codex_base_url="https://example.test/backend-api/codex",
        )

        with patch(
            "atuin_ai_proxy.backend._post",
            return_value=(FakeResponse(), FakeConnection()),
        ) as post:
            ResponsesBackend(settings).open_stream(body)

        headers = post.call_args.args[1]
        posted_body = json.loads(post.call_args.args[2])
        self.assertNotIn("X-OpenAI-Internal-Codex-Responses-Lite", headers)
        self.assertEqual(posted_body, body)

    def test_openai_backend_does_not_use_internal_responses_lite(self) -> None:
        body = {"model": "gpt-5.6-luna", "input": [], "tools": []}
        settings = Settings(
            backend="openai",
            openai_api_key="token",
            openai_base_url="https://example.test/v1",
            openai_api="responses",
        )

        with patch(
            "atuin_ai_proxy.backend._post",
            return_value=(FakeResponse(), FakeConnection()),
        ) as post:
            ResponsesBackend(settings).open_stream(body)

        headers = post.call_args.args[1]
        posted_body = json.loads(post.call_args.args[2])
        self.assertNotIn("X-OpenAI-Internal-Codex-Responses-Lite", headers)
        self.assertEqual(posted_body, body)


class FakeResponse:
    def __init__(
        self,
        lines: list[bytes] | None = None,
        *,
        status: int = 200,
        body: bytes = b"",
    ) -> None:
        self.lines = lines or []
        self.status = status
        self.body = body

    def getheaders(self) -> list[tuple[str, str]]:
        return []

    def readline(self) -> bytes:
        if not self.lines:
            return b""
        return self.lines.pop(0)

    def read(self) -> bytes:
        return self.body


class FakeConnection:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


if __name__ == "__main__":
    unittest.main()
