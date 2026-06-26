import json
import unittest

from atuin_ai_proxy.protocol import (
    build_responses_request,
    encode_sse_event,
    translate_responses_events,
)
from atuin_ai_proxy.settings import Settings


class ProtocolTests(unittest.TestCase):
    def test_encode_sse_event_uses_atuin_wire_format(self) -> None:
        self.assertEqual(
            encode_sse_event("text", {"content": "hello"}).decode(),
            'event: text\ndata: {"content":"hello"}\n\n',
        )

    def test_build_responses_request_converts_atuin_history_and_tools(self) -> None:
        settings = Settings(
            backend="openai",
            model="gpt-test",
            openai_api_key="sk-test",
        )
        atuin_request = {
            "messages": [
                {"role": "user", "content": "list files"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "read_file",
                            "input": {"file_path": "README.md"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": "hello",
                            "is_error": False,
                        }
                    ],
                },
            ],
            "context": {"shell": "bash", "pwd": "/tmp"},
            "config": {
                "capabilities": ["client_v1_read_file"],
                "model": None,
            },
            "invocation_id": "inv-1",
        }

        body = build_responses_request(atuin_request, settings)

        self.assertEqual(body["model"], "gpt-test")
        self.assertTrue(body["stream"])
        self.assertFalse(body["store"])
        self.assertEqual(body["input"][1]["role"], "user")
        self.assertEqual(body["input"][2]["type"], "function_call")
        self.assertEqual(body["input"][2]["call_id"], "call_1")
        self.assertEqual(body["input"][3]["type"], "function_call_output")
        tool_names = {tool["name"] for tool in body["tools"]}
        self.assertIn("suggest_command", tool_names)
        self.assertIn("read_file", tool_names)
        self.assertNotIn("write_file", tool_names)

    def test_build_responses_request_requires_model(self) -> None:
        settings = Settings(backend="openai", openai_api_key="sk-test")

        with self.assertRaisesRegex(ValueError, "MODEL"):
            build_responses_request({"messages": [], "config": {}}, settings)

    def test_translate_responses_events_maps_text_tool_and_done(self) -> None:
        upstream_events = [
            (
                "response.output_text.delta",
                {"delta": "ls"},
            ),
            (
                "response.output_item.done",
                {
                    "item": {
                        "type": "function_call",
                        "call_id": "call_2",
                        "name": "suggest_command",
                        "arguments": json.dumps({"command": "ls -la"}),
                    }
                },
            ),
            ("response.completed", {"response": {"id": "resp_1"}}),
        ]

        translated = [
            event.decode()
            for event in translate_responses_events(upstream_events, "session-1")
        ]

        self.assertEqual(translated[0], 'event: text\ndata: {"content":"ls"}\n\n')
        self.assertEqual(
            translated[1],
            'event: tool_call\ndata: {"id":"call_2","name":"suggest_command","input":{"command":"ls -la","danger":"low","confidence":"medium"}}\n\n',
        )
        self.assertEqual(
            translated[2], 'event: done\ndata: {"session_id":"session-1"}\n\n'
        )


if __name__ == "__main__":
    unittest.main()
