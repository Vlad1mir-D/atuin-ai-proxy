FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CODEX_HOME=/data/codex \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

COPY pyproject.toml README.md ./
COPY atuin_ai_proxy ./atuin_ai_proxy

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["atuin-ai-proxy", "serve"]
