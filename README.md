# Atuin AI Proxy

Python 3 proxy that exposes the Atuin Hub AI endpoint expected by Atuin and forwards requests to OpenAI or Codex-compatible Responses backends.

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
# OpenAI Responses API
BACKEND=openai
OPENAI_API_KEY=sk-...
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

For Codex OAuth device login:

```sh
docker compose run --rm atuin-ai-proxy atuin-ai-proxy auth login --device-code
docker compose up
```

To use an existing Codex CLI auth file, mount it as `/data/codex/auth.json` or set `CODEX_AUTH_FILE` to the mounted path.

## Local development

The implementation uses only the Python standard library at runtime.

```sh
python3 -m unittest discover -s tests
python3 -m atuin_ai_proxy serve
```

## Protocol notes

The proxy accepts `POST /api/cli/chat`, returns `text/event-stream`, sets `x-atuin-ai-session-id`, and translates OpenAI/Codex Responses stream events into Atuin stream events:

- `response.output_text.delta` -> `text`
- completed `function_call` items -> `tool_call`
- `response.completed` -> `done`
- upstream failures -> `error`

Client-side tools are exposed only when Atuin advertises the matching capability. The `suggest_command` tool is always exposed so models can send Atuin command suggestions.
