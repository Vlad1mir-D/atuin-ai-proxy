from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from typing import Any


JSON = dict[str, Any]
logger = logging.getLogger(__name__)


CAPABILITY_TO_TOOL = {
    "client_v1_read_file": "read_file",
    "client_v1_edit_file": "edit_file",
    "client_v1_write_file": "write_file",
    "client_v1_execute_shell_command": "execute_shell_command",
    "client_v1_atuin_history": "atuin_history",
    "client_v1_atuin_output": "atuin_output",
    "client_v1_load_skill": "load_skill",
}


SYSTEM_INSTRUCTIONS = """You are powering Atuin AI inside a user's shell.
Answer concisely and prefer direct shell-oriented help.
When the user should run a command, call suggest_command with command, confidence, danger, and brief notes instead of only writing prose.
Only call client-side tools that are advertised in the request capabilities.
Treat file and shell tools as operations executed by the user's Atuin client, not by this proxy."""


def encode_sse_event(event: str, data: Any) -> bytes:
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    lines = [f"event: {event}"]
    lines.extend(f"data: {line}" for line in payload.splitlines() or [""])
    return ("\n".join(lines) + "\n\n").encode("utf-8")


def build_responses_request(
    atuin_request: JSON, settings: Any, request_id: str | None = None
) -> JSON:
    config = atuin_request.get("config") or {}
    model = resolve_model(atuin_request, settings)

    capabilities = set(config.get("capabilities") or [])
    messages = atuin_request.get("messages") or []
    completed_tool_use_ids = _tool_result_ids(messages)
    input_items = [_context_message(atuin_request)]
    for message in messages:
        input_items.extend(
            _convert_message(
                message,
                completed_tool_use_ids,
                request_id=request_id,
            )
        )

    return {
        "model": model,
        "instructions": SYSTEM_INSTRUCTIONS,
        "input": input_items,
        "tools": tool_definitions(capabilities),
        "stream": True,
        "store": False,
    }


def build_chat_completions_request(
    atuin_request: JSON, settings: Any, request_id: str | None = None
) -> JSON:
    config = atuin_request.get("config") or {}
    model = resolve_model(atuin_request, settings)

    capabilities = set(config.get("capabilities") or [])
    messages = atuin_request.get("messages") or []
    completed_tool_use_ids = _tool_result_ids(messages)
    chat_messages = [{"role": "system", "content": SYSTEM_INSTRUCTIONS}]
    chat_messages.append(_context_message(atuin_request))
    for message in messages:
        chat_messages.extend(
            _convert_chat_message(
                message,
                completed_tool_use_ids,
                request_id=request_id,
            )
        )

    return {
        "model": model,
        "messages": chat_messages,
        "tools": chat_tool_definitions(capabilities),
        "tool_choice": "auto",
        "stream": True,
    }


def resolve_model(atuin_request: JSON, settings: Any) -> str:
    config = atuin_request.get("config") or {}
    model = config.get("model") or settings.model
    if not model:
        raise ValueError("MODEL must be configured or supplied by Atuin config.model")
    return str(model)


def tool_definitions(capabilities: set[str]) -> list[JSON]:
    tools = [_suggest_command_tool()]
    for capability, name in CAPABILITY_TO_TOOL.items():
        if capability in capabilities:
            tools.append(_client_tool(name))
    return tools


def chat_tool_definitions(capabilities: set[str]) -> list[JSON]:
    tools = []
    for tool in tool_definitions(capabilities):
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            }
        )
    return tools


def translate_responses_events(
    upstream_events: Iterable[tuple[str, JSON]], session_id: str
) -> Iterator[bytes]:
    done_sent = False
    for event, data in upstream_events:
        event_type = _responses_event_type(event, data)
        if event_type == "response.output_text.delta":
            delta = data.get("delta") or data.get("text") or ""
            if delta:
                yield encode_sse_event("text", {"content": delta})
        elif event_type == "response.output_item.done":
            item = data.get("item") or {}
            if item.get("type") == "function_call":
                yield encode_sse_event("tool_call", _tool_call_from_item(item))
        elif event_type == "response.created":
            yield encode_sse_event("status", {"state": "thinking"})
        elif event_type in {"response.failed", "response.incomplete", "error"}:
            yield encode_sse_event("error", {"message": _error_message(data)})
            done_sent = True
        elif event_type == "response.completed":
            yield encode_sse_event("done", {"session_id": session_id})
            done_sent = True

    if not done_sent:
        yield encode_sse_event("done", {"session_id": session_id})


def translate_chat_completions_events(
    upstream_events: Iterable[tuple[str, JSON]], session_id: str
) -> Iterator[bytes]:
    done_sent = False
    pending_tool_calls: dict[tuple[int, int], JSON] = {}
    for event, data in upstream_events:
        if _chat_done_event(event, data):
            yield from _drain_chat_tool_calls(pending_tool_calls)
            yield encode_sse_event("done", {"session_id": session_id})
            done_sent = True
            continue

        if data.get("error"):
            yield encode_sse_event("error", {"message": _error_message(data)})
            done_sent = True
            continue

        choices = data.get("choices")
        if not isinstance(choices, list):
            continue

        for choice in choices:
            if not isinstance(choice, dict):
                continue
            choice_index = _chat_choice_index(choice)
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                delta = {}

            content = delta.get("content") or delta.get("refusal")
            if content:
                yield encode_sse_event("text", {"content": str(content)})

            _accumulate_chat_tool_calls(choice_index, delta, pending_tool_calls)
            if choice.get("finish_reason") in {"tool_calls", "function_call"}:
                yield from _drain_chat_tool_calls(
                    pending_tool_calls,
                    choice_index=choice_index,
                )

    if not done_sent:
        yield from _drain_chat_tool_calls(pending_tool_calls)
        yield encode_sse_event("done", {"session_id": session_id})


def _responses_event_type(event: str, data: JSON) -> str:
    if event != "message":
        return event
    data_type = data.get("type")
    if isinstance(data_type, str):
        return data_type
    return event


def _context_message(atuin_request: JSON) -> JSON:
    context = {
        "context": atuin_request.get("context") or {},
        "config": atuin_request.get("config") or {},
        "invocation_id": atuin_request.get("invocation_id"),
        "session_id": atuin_request.get("session_id"),
    }
    return _message("user", "Atuin client context:\n" + _compact_json(context))


def _convert_message(
    message: JSON,
    completed_tool_use_ids: set[str] | None = None,
    *,
    request_id: str | None = None,
) -> list[JSON]:
    completed_tool_use_ids = completed_tool_use_ids or set()
    role = message.get("role") or "user"
    content = message.get("content")
    if isinstance(content, str):
        return [_message(role, content)]
    if not isinstance(content, list):
        return [_message(role, _compact_json(content))]

    converted: list[JSON] = []
    text_parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            text_parts.append(str(block))
            continue

        block_type = block.get("type")
        if block_type in {"text", "input_text", "output_text"}:
            text_parts.append(str(block.get("text") or block.get("content") or ""))
        elif block_type == "tool_use":
            if text_parts:
                converted.append(_message(role, "\n".join(text_parts)))
                text_parts = []
            call_id = str(block.get("id") or "")
            name = str(block.get("name") or "")
            converted.append(
                {
                    "type": "function_call",
                    "id": _responses_item_id("fc", call_id),
                    "call_id": call_id,
                    "name": name,
                    "arguments": _compact_json(block.get("input") or {}),
                    "status": "completed",
                }
            )
            if call_id not in completed_tool_use_ids:
                converted.append(
                    {
                        "type": "function_call_output",
                        "id": _responses_item_id("fco", call_id),
                        "call_id": call_id,
                        "output": _synthetic_tool_result_output(name),
                        "status": "completed",
                    }
                )
                logger.debug(
                    "Repaired orphaned Atuin tool call history request_id=%s "
                    "tool_name=%s call_id=%s",
                    request_id,
                    name,
                    call_id,
                )
        elif block_type == "tool_result":
            if text_parts:
                converted.append(_message(role, "\n".join(text_parts)))
                text_parts = []
            converted.append(
                {
                    "type": "function_call_output",
                    "id": _responses_item_id(
                        "fco",
                        str(block.get("tool_use_id") or ""),
                    ),
                    "call_id": str(block.get("tool_use_id") or ""),
                    "output": _tool_result_output(block),
                    "status": "completed",
                }
            )
        else:
            text_parts.append(_compact_json(block))

    if text_parts:
        converted.append(_message(role, "\n".join(text_parts)))
    return converted


def _convert_chat_message(
    message: JSON,
    completed_tool_use_ids: set[str] | None = None,
    *,
    request_id: str | None = None,
) -> list[JSON]:
    completed_tool_use_ids = completed_tool_use_ids or set()
    role = str(message.get("role") or "user")
    content = message.get("content")
    if isinstance(content, str):
        return [_message(role, content)]
    if not isinstance(content, list):
        return [_message(role, _compact_json(content))]

    converted: list[JSON] = []
    text_parts: list[str] = []
    tool_calls: list[JSON] = []
    synthetic_outputs: list[JSON] = []
    for block in content:
        if not isinstance(block, dict):
            text_parts.append(str(block))
            continue

        block_type = block.get("type")
        if block_type in {"text", "input_text", "output_text"}:
            text_parts.append(str(block.get("text") or block.get("content") or ""))
        elif block_type == "tool_use":
            call_id = str(block.get("id") or "")
            name = str(block.get("name") or "")
            tool_calls.append(_chat_tool_call(call_id, name, block.get("input") or {}))
            if call_id not in completed_tool_use_ids:
                synthetic_outputs.append(
                    _chat_tool_result_message(
                        call_id,
                        _synthetic_tool_result_output(name),
                    )
                )
                logger.debug(
                    "Repaired orphaned Atuin tool call history request_id=%s "
                    "tool_name=%s call_id=%s",
                    request_id,
                    name,
                    call_id,
                )
        elif block_type == "tool_result":
            if text_parts:
                converted.append(_message(role, "\n".join(text_parts)))
                text_parts = []
            converted.append(
                _chat_tool_result_message(
                    str(block.get("tool_use_id") or ""),
                    _tool_result_output(block),
                )
            )
        else:
            text_parts.append(_compact_json(block))

    if tool_calls:
        converted.append(
            {
                "role": "assistant",
                "content": "\n".join(text_parts) if text_parts else None,
                "tool_calls": tool_calls,
            }
        )
        converted.extend(synthetic_outputs)
    elif text_parts:
        converted.append(_message(role, "\n".join(text_parts)))
    return converted


def _chat_tool_call(call_id: str, name: str, arguments: Any) -> JSON:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": _compact_json(arguments),
        },
    }


def _chat_tool_result_message(call_id: str, output: str) -> JSON:
    return {"role": "tool", "tool_call_id": call_id, "content": output}


def _responses_item_id(prefix: str, call_id: str) -> str:
    return f"{prefix}_{call_id}" if call_id else prefix


def _tool_result_ids(messages: list[JSON]) -> set[str]:
    tool_use_ids: set[str] = set()
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            tool_use_id = block.get("tool_use_id")
            if tool_use_id:
                tool_use_ids.add(str(tool_use_id))
    return tool_use_ids


def _message(role: str, text: str) -> JSON:
    return {"role": role, "content": text}


def _tool_result_output(block: JSON) -> str:
    if block.get("remote"):
        length = block.get("content_length")
        suffix = f", content_length={length}" if length is not None else ""
        return f"[Atuin remote tool result{suffix}]"
    content = block.get("content")
    return content if isinstance(content, str) else _compact_json(content)


def _synthetic_tool_result_output(name: str) -> str:
    if name == "suggest_command":
        return (
            "Atuin displayed this command suggestion to the user; no command execution "
            "output is available. Use current context or last_command if present to "
            "infer whether it was later run."
        )
    return (
        "Atuin did not provide a result for this previous tool call; treat the result "
        "as unavailable."
    )


def _tool_call_from_item(item: JSON) -> JSON:
    return _tool_call_payload(
        str(item.get("call_id") or item.get("id") or ""),
        str(item.get("name") or ""),
        item.get("arguments") or "{}",
    )


def _tool_call_payload(call_id: str, name: str, raw_arguments: Any) -> JSON:
    try:
        arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    except json.JSONDecodeError:
        arguments = {"raw_arguments": raw_arguments}
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}

    if name == "suggest_command":
        arguments.setdefault("danger", "low")
        arguments.setdefault("confidence", "medium")

    return {
        "id": call_id,
        "name": name,
        "input": arguments,
    }


def _chat_done_event(event: str, data: JSON) -> bool:
    return event == "done" or data.get("message") == "[DONE]"


def _chat_choice_index(choice: JSON) -> int:
    try:
        return int(choice.get("index") or 0)
    except (TypeError, ValueError):
        return 0


def _chat_tool_index(tool_call: JSON) -> int:
    try:
        return int(tool_call.get("index") or 0)
    except (TypeError, ValueError):
        return 0


def _accumulate_chat_tool_calls(
    choice_index: int,
    delta: JSON,
    pending_tool_calls: dict[tuple[int, int], JSON],
) -> None:
    tool_calls = delta.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            key = (choice_index, _chat_tool_index(tool_call))
            pending = pending_tool_calls.setdefault(
                key,
                {"id": "", "name": "", "arguments": ""},
            )
            if tool_call.get("id"):
                pending["id"] = str(tool_call["id"])
            function = tool_call.get("function") or {}
            if isinstance(function, dict):
                if function.get("name"):
                    pending["name"] = str(function["name"])
                if function.get("arguments"):
                    pending["arguments"] += str(function["arguments"])

    function_call = delta.get("function_call")
    if isinstance(function_call, dict):
        pending = pending_tool_calls.setdefault(
            (choice_index, 0),
            {"id": "", "name": "", "arguments": ""},
        )
        if function_call.get("name"):
            pending["name"] = str(function_call["name"])
        if function_call.get("arguments"):
            pending["arguments"] += str(function_call["arguments"])


def _drain_chat_tool_calls(
    pending_tool_calls: dict[tuple[int, int], JSON],
    *,
    choice_index: int | None = None,
) -> Iterator[bytes]:
    keys = sorted(pending_tool_calls)
    for key in keys:
        if choice_index is not None and key[0] != choice_index:
            continue
        pending = pending_tool_calls.pop(key)
        yield encode_sse_event(
            "tool_call",
            _tool_call_payload(
                str(pending.get("id") or ""),
                str(pending.get("name") or ""),
                pending.get("arguments") or "{}",
            ),
        )


def _error_message(data: JSON) -> str:
    error = data.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error)
    if error:
        return str(error)
    return str(data.get("message") or data.get("response", {}).get("status") or data)


def _compact_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _function_tool(name: str, description: str, parameters: JSON) -> JSON:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": parameters,
    }


def _object_schema(properties: JSON, required: list[str] | None = None) -> JSON:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def _suggest_command_tool() -> JSON:
    return _function_tool(
        "suggest_command",
        "Suggest a shell command for Atuin to present to the user.",
        _object_schema(
            {
                "command": {"type": "string"},
                "description": {"type": "string"},
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                "confidence_notes": {"type": "string"},
                "danger": {"type": "string", "enum": ["low", "medium", "high"]},
                "danger_notes": {"type": "string"},
            },
            ["command"],
        ),
    )


def _client_tool(name: str) -> JSON:
    schemas = {
        "read_file": _object_schema(
            {
                "file_path": {"type": "string"},
                "offset": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
            },
            ["file_path"],
        ),
        "edit_file": _object_schema(
            {
                "file_path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            ["file_path", "old_string", "new_string"],
        ),
        "write_file": _object_schema(
            {
                "file_path": {"type": "string"},
                "content": {"type": "string"},
                "overwrite": {"type": "boolean"},
            },
            ["file_path", "content"],
        ),
        "execute_shell_command": _object_schema(
            {
                "command": {"type": "string"},
                "dir": {"type": "string"},
                "shell": {"type": "string"},
                "timeout": {"type": "integer", "minimum": 1, "maximum": 600},
                "description": {"type": "string"},
            },
            ["command"],
        ),
        "atuin_history": _object_schema(
            {
                "filter_modes": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["global", "host", "session", "directory", "workspace"],
                    },
                },
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            ["filter_modes", "query"],
        ),
        "atuin_output": _object_schema(
            {
                "history_id": {"type": "string"},
                "ranges": {"type": "array", "items": {"type": "object"}},
            },
            ["history_id"],
        ),
        "load_skill": _object_schema({"name": {"type": "string"}}, ["name"]),
    }
    descriptions = {
        "read_file": "Read a file through the Atuin client.",
        "edit_file": "Edit a file through the Atuin client.",
        "write_file": "Write a file through the Atuin client.",
        "execute_shell_command": "Ask the Atuin client to execute a shell command.",
        "atuin_history": "Search the user's Atuin shell history.",
        "atuin_output": "Fetch captured command output from Atuin history.",
        "load_skill": "Load an Atuin skill by name.",
    }
    return _function_tool(name, descriptions[name], schemas[name])
