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
    model = config.get("model") or settings.model
    if not model:
        raise ValueError("MODEL must be configured or supplied by Atuin config.model")

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


def tool_definitions(capabilities: set[str]) -> list[JSON]:
    tools = [_suggest_command_tool()]
    for capability, name in CAPABILITY_TO_TOOL.items():
        if capability in capabilities:
            tools.append(_client_tool(name))
    return tools


def translate_responses_events(
    upstream_events: Iterable[tuple[str, JSON]], session_id: str
) -> Iterator[bytes]:
    done_sent = False
    for event, data in upstream_events:
        if event == "response.output_text.delta":
            delta = data.get("delta") or data.get("text") or ""
            if delta:
                yield encode_sse_event("text", {"content": delta})
        elif event == "response.output_item.done":
            item = data.get("item") or {}
            if item.get("type") == "function_call":
                yield encode_sse_event("tool_call", _tool_call_from_item(item))
        elif event == "response.created":
            yield encode_sse_event("status", {"state": "thinking"})
        elif event in {"response.failed", "response.incomplete", "error"}:
            yield encode_sse_event("error", {"message": _error_message(data)})
            done_sent = True
        elif event == "response.completed":
            yield encode_sse_event("done", {"session_id": session_id})
            done_sent = True

    if not done_sent:
        yield encode_sse_event("done", {"session_id": session_id})


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
                    "call_id": call_id,
                    "name": name,
                    "arguments": _compact_json(block.get("input") or {}),
                }
            )
            if call_id not in completed_tool_use_ids:
                converted.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": _synthetic_tool_result_output(name),
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
                    "call_id": str(block.get("tool_use_id") or ""),
                    "output": _tool_result_output(block),
                }
            )
        else:
            text_parts.append(_compact_json(block))

    if text_parts:
        converted.append(_message(role, "\n".join(text_parts)))
    return converted


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
    content_type = "input_text" if role != "assistant" else "output_text"
    return {
        "type": "message",
        "role": role,
        "content": [{"type": content_type, "text": text}],
    }


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
    raw_arguments = item.get("arguments") or "{}"
    try:
        arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    except json.JSONDecodeError:
        arguments = {"raw_arguments": raw_arguments}
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}

    name = str(item.get("name") or "")
    if name == "suggest_command":
        arguments.setdefault("danger", "low")
        arguments.setdefault("confidence", "medium")

    return {
        "id": str(item.get("call_id") or item.get("id") or ""),
        "name": name,
        "input": arguments,
    }


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
