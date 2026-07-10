# Atuin AI Proxy

Python 3 proxy that exposes the Atuin Hub AI endpoint expected by Atuin and forwards requests to OpenAI-compatible Chat Completions or Responses backends.

## Atuin config

Point Atuin at the proxy:

```toml
[ai]
enabled = true
endpoint = "http://localhost:8000"
api_token = "change-me"
```

Set `api_token` to the same value as `ATUIN_PROXY_TOKEN`. If `ATUIN_PROXY_TOKEN` is unset, the proxy accepts local requests without bearer auth.

## Run with Docker Compose

```sh
cp .env.example .env
# edit .env
docker compose up --build
```

Backends:

```sh
# OpenAI-compatible API
BACKEND=openai
OPENAI_API_KEY=sk-...
OPENAI_API=auto
MODEL=...

# Codex access token or personal access token
BACKEND=codex-token
CODEX_ACCESS_TOKEN=at-...
CODEX_ACCOUNT_ID=...
MODEL=...

# Codex OAuth auth.json
BACKEND=codex-oauth
CODEX_HOME=/data/codex
MODEL=...
```

Server binding:

```sh
HOST=0.0.0.0
PORT=8000
```

`OPENAI_API` applies only to `BACKEND=openai` and accepts `auto`, `responses`,
or `chat_completions`. The default `auto` mode tries Chat Completions first and
falls back to Responses when the Chat Completions request is rejected as
unsupported.

The `codex-token` and `codex-oauth` backends support only the Responses API. If
`OPENAI_API=chat_completions` is configured with either Codex backend, the proxy
logs a startup warning and uses the Responses API.

For Codex OAuth device login:

```sh
docker compose run --rm atuin-ai-proxy atuin-ai-proxy auth login --device-code
docker compose up
```

To use an existing Codex CLI auth file, mount it as `/data/codex/auth.json` or set `CODEX_AUTH_FILE` to the mounted path.
`CODEX_CLIENT_ID` can override the bundled Codex OAuth client id when needed.

## Local development

The implementation uses only the Python standard library at runtime.

```sh
python3 -m unittest discover -s tests
python3 -m atuin_ai_proxy serve
```

## Debugging

The proxy includes a request id in every HTTP response and in stream errors. When
Atuin reports `SSE request failed (...)`, copy the `request_id` from the JSON
body and search for it in the proxy logs.

Logging levels:

```sh
# Normal operational logs
LOG_LEVEL=INFO

# Startup warnings and errors only
LOG_LEVEL=WARNING

# Request/backend diagnostics: model source, upstream status, tool names, timings
LOG_LEVEL=DEBUG

# DEBUG plus sanitized request, backend, and SSE payload excerpts
LOG_LEVEL=TRACE
TRACE_PAYLOAD_BYTES=4096
```

For local runs, the `serve` command can override `LOG_LEVEL` with any level the
proxy emits: `TRACE`, `DEBUG`, `INFO`, `WARNING`, or `ERROR`.

```sh
python3 -m atuin_ai_proxy serve --log-level DEBUG
```

TRACE output is sanitized and bounded, but it can still include shell history,
prompts, paths, and command output. Use it only while diagnosing a problem.

Common failures:

- `400 missing_model`: set `MODEL=...` or configure Atuin to send a model.
- `400 invalid_json` / `400 invalid_request`: Atuin sent a body the proxy could
  not parse or convert.
- `401 unauthorized`: Atuin `[ai].api_token` does not match `ATUIN_PROXY_TOKEN`.
- `502 auth_error`: backend auth is missing or invalid.
- `502 upstream_http_error`: the selected backend rejected the converted
  upstream request; the error body includes a sanitized upstream excerpt.
- `504 upstream_timeout`: the backend did not respond before
  `REQUEST_TIMEOUT_SECONDS`.

## Protocol notes

The proxy accepts `POST /api/cli/chat`, returns `text/event-stream`, sets `x-atuin-ai-session-id`, and translates upstream stream events into Atuin stream events:

- `response.output_text.delta` -> `text`
- Chat Completions `choices[].delta.content` -> `text`
- completed `function_call` items -> `tool_call`
- completed Chat Completions `tool_calls` -> `tool_call`
- `response.completed` -> `done`
- Chat Completions `[DONE]` -> `done`
- upstream failures -> `error`

Client-side tools are exposed only when Atuin advertises the matching capability. The `suggest_command` tool is always exposed so models can send Atuin command suggestions.
