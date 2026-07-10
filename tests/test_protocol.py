import json
import unittest

from atuin_ai_proxy.protocol import (
    build_chat_completions_request,
    build_responses_request,
    encode_sse_event,
    translate_chat_completions_events,
    translate_responses_events,
)
from atuin_ai_proxy.settings import Settings


def _history_request() -> dict[str, object]:
    return {
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
        atuin_request = _history_request()

        body = build_responses_request(atuin_request, settings)

        self.assertEqual(body["model"], "gpt-test")
        self.assertTrue(body["stream"])
        self.assertFalse(body["store"])
        self.assertEqual(body["input"][1]["role"], "user")
        self.assertEqual(body["input"][2]["type"], "function_call")
        self.assertEqual(body["input"][2]["call_id"], "call_1")
        self.assertEqual(body["input"][3]["type"], "function_call_output")
        call_1_outputs = [
            item
            for item in body["input"]
            if item.get("type") == "function_call_output"
            and item.get("call_id") == "call_1"
        ]
        self.assertEqual(len(call_1_outputs), 1)
        self.assertEqual(call_1_outputs[0]["output"], "hello")
        tool_names = {tool["name"] for tool in body["tools"]}
        self.assertIn("suggest_command", tool_names)
        self.assertIn("read_file", tool_names)
        self.assertNotIn("write_file", tool_names)

    def test_build_responses_request_repairs_orphaned_suggest_command_history(self) -> None:
        settings = Settings(
            backend="openai",
            model="gpt-test",
            openai_api_key="sk-test",
        )
        atuin_request = {
            "messages": [
                {"role": "user", "content": "show me the disk usage command"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_suggest_1",
                            "name": "suggest_command",
                            "input": {
                                "command": "du -sh .",
                                "confidence": "high",
                                "danger": "low",
                            },
                        }
                    ],
                },
                {"role": "user", "content": "now show the largest files"},
            ],
            "config": {"model": None},
        }

        body = build_responses_request(atuin_request, settings)

        self.assertEqual(body["input"][2]["type"], "function_call")
        self.assertEqual(body["input"][2]["call_id"], "call_suggest_1")
        self.assertEqual(body["input"][3]["type"], "function_call_output")
        self.assertEqual(body["input"][3]["call_id"], "call_suggest_1")
        self.assertEqual(
            body["input"][3]["output"],
            "Atuin displayed this command suggestion to the user; no command execution "
            "output is available. Use current context or last_command if present to "
            "infer whether it was later run.",
        )
        self.assertEqual(body["input"][4]["role"], "user")
        self.assertFalse(_orphaned_function_call_ids(body["input"]))

    def test_build_responses_request_repairs_orphaned_client_tool_history(self) -> None:
        settings = Settings(
            backend="openai",
            model="gpt-test",
            openai_api_key="sk-test",
        )
        atuin_request = {
            "messages": [
                {"role": "user", "content": "read README"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_read_1",
                            "name": "read_file",
                            "input": {"file_path": "README.md"},
                        }
                    ],
                },
                {"role": "user", "content": "summarize it"},
            ],
            "config": {"capabilities": ["client_v1_read_file"], "model": None},
        }

        body = build_responses_request(atuin_request, settings)

        self.assertEqual(body["input"][3]["type"], "function_call_output")
        self.assertEqual(body["input"][3]["call_id"], "call_read_1")
        self.assertEqual(
            body["input"][3]["output"],
            "Atuin did not provide a result for this previous tool call; treat the "
            "result as unavailable.",
        )
        self.assertFalse(_orphaned_function_call_ids(body["input"]))

    def test_build_responses_request_replays_text_history_as_string_content(self) -> None:
        settings = Settings(
            backend="openai",
            model="gpt-test",
            openai_api_key="sk-test",
        )
        atuin_request = {
            "messages": [
                {"role": "user", "content": "what are you?"},
                {
                    "role": "assistant",
                    "content": "I'm Atuin AI, an assistant built into your shell.",
                },
                {"role": "user", "content": "What model are you based on?"},
            ],
            "config": {"model": None},
        }

        body = build_responses_request(atuin_request, settings)

        assistant_message = body["input"][2]
        self.assertEqual(assistant_message["role"], "assistant")
        self.assertEqual(
            assistant_message["content"],
            "I'm Atuin AI, an assistant built into your shell.",
        )

    def test_build_responses_request_replays_tool_history_with_item_ids(self) -> None:
        settings = Settings(
            backend="openai",
            model="gpt-test",
            openai_api_key="sk-test",
        )
        atuin_request = {
            "messages": [
                {"role": "user", "content": "What model are you based on?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Let me check the project configuration.",
                        },
                        {
                            "type": "tool_use",
                            "id": "call_find_model",
                            "name": "execute_shell_command",
                            "input": {
                                "command": "grep -ri \"model\" .",
                                "description": "Find files mentioning model",
                            },
                        },
                        {
                            "type": "tool_use",
                            "id": "call_list_files",
                            "name": "execute_shell_command",
                            "input": {
                                "command": "ls -la",
                                "description": "List project root files",
                            },
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_find_model",
                            "content": "Permission denied by the user",
                            "is_error": True,
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_list_files",
                            "content": "Exit code: 0\n\nstdout:\nREADME.md",
                            "is_error": False,
                        },
                    ],
                },
            ],
            "config": {
                "capabilities": ["client_v1_execute_shell_command"],
                "model": None,
            },
        }

        body = build_responses_request(atuin_request, settings)

        function_calls = [
            item for item in body["input"] if item.get("type") == "function_call"
        ]
        function_outputs = [
            item
            for item in body["input"]
            if item.get("type") == "function_call_output"
        ]
        self.assertEqual(
            [item.get("id") for item in function_calls],
            ["fc_call_find_model", "fc_call_list_files"],
        )
        self.assertEqual(
            [item.get("id") for item in function_outputs],
            ["fco_call_find_model", "fco_call_list_files"],
        )
        self.assertEqual(
            [item["call_id"] for item in function_outputs],
            ["call_find_model", "call_list_files"],
        )
        self.assertEqual(
            [item.get("status") for item in function_calls + function_outputs],
            ["completed", "completed", "completed", "completed"],
        )

    def test_build_responses_request_requires_model(self) -> None:
        settings = Settings(backend="openai", openai_api_key="sk-test")

        with self.assertRaisesRegex(ValueError, "MODEL"):
            build_responses_request({"messages": [], "config": {}}, settings)

    def test_build_chat_completions_request_converts_history_and_tools(self) -> None:
        settings = Settings(
            backend="openai",
            model="chat-model",
            openai_api_key="sk-test",
        )
        atuin_request = _history_request()

        body = build_chat_completions_request(atuin_request, settings)

        self.assertEqual(body["model"], "chat-model")
        self.assertTrue(body["stream"])
        self.assertNotIn("store", body)
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertEqual(body["messages"][1]["role"], "user")
        self.assertEqual(body["messages"][2], {"role": "user", "content": "list files"})
        assistant_message = body["messages"][3]
        self.assertEqual(assistant_message["role"], "assistant")
        self.assertIsNone(assistant_message["content"])
        self.assertEqual(
            assistant_message["tool_calls"],
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"file_path":"README.md"}',
                    },
                }
            ],
        )
        self.assertEqual(
            body["messages"][4],
            {"role": "tool", "tool_call_id": "call_1", "content": "hello"},
        )
        tool_names = {tool["function"]["name"] for tool in body["tools"]}
        self.assertIn("suggest_command", tool_names)
        self.assertIn("read_file", tool_names)
        self.assertNotIn("write_file", tool_names)

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

    def test_translate_responses_events_uses_type_from_message_events(self) -> None:
        upstream_events = [
            ("message", {"type": "response.created"}),
            (
                "message",
                {"type": "response.output_text.delta", "delta": "hello"},
            ),
            (
                "message",
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "function_call",
                        "call_id": "call_3",
                        "name": "suggest_command",
                        "arguments": json.dumps({"command": "pwd"}),
                    },
                },
            ),
            ("message", {"message": "[DONE]"}),
        ]

        translated = [
            event.decode()
            for event in translate_responses_events(upstream_events, "session-2")
        ]

        self.assertEqual(
            translated,
            [
                'event: status\ndata: {"state":"thinking"}\n\n',
                'event: text\ndata: {"content":"hello"}\n\n',
                'event: tool_call\ndata: {"id":"call_3","name":"suggest_command","input":{"command":"pwd","danger":"low","confidence":"medium"}}\n\n',
                'event: done\ndata: {"session_id":"session-2"}\n\n',
            ],
        )

    def test_translate_chat_completions_events_maps_text_tool_and_done(self) -> None:
        upstream_events = [
            (
                "message",
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "ls"},
                            "finish_reason": None,
                        }
                    ]
                },
            ),
            (
                "message",
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_2",
                                        "type": "function",
                                        "function": {
                                            "name": "suggest_command",
                                            "arguments": '{"command"',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                },
            ),
            (
                "message",
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": ':"ls -la"}'},
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            ),
            ("message", {"message": "[DONE]"}),
        ]

        translated = [
            event.decode()
            for event in translate_chat_completions_events(upstream_events, "session-1")
        ]

        self.assertEqual(translated[0], 'event: text\ndata: {"content":"ls"}\n\n')
        self.assertEqual(
            translated[1],
            'event: tool_call\ndata: {"id":"call_2","name":"suggest_command","input":{"command":"ls -la","danger":"low","confidence":"medium"}}\n\n',
        )
        self.assertEqual(
            translated[2], 'event: done\ndata: {"session_id":"session-1"}\n\n'
        )


def _orphaned_function_call_ids(input_items: list[dict[str, object]]) -> set[str]:
    call_ids = {
        str(item.get("call_id"))
        for item in input_items
        if item.get("type") == "function_call"
    }
    output_ids = {
        str(item.get("call_id"))
        for item in input_items
        if item.get("type") == "function_call_output"
    }
    return call_ids - output_ids


if __name__ == "__main__":
    unittest.main()
