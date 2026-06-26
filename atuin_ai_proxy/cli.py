from __future__ import annotations

import argparse
import json
import sys

from .auth import AuthError, CodexAuthFileProvider, CodexDeviceAuthenticator
from .server import run_server
from .settings import Settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="atuin-ai-proxy")
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", help="Run the Atuin AI proxy")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument(
        "--debug",
        action="store_true",
        help="Shortcut for LOG_LEVEL=DEBUG",
    )

    auth = subparsers.add_parser("auth", help="Manage Codex OAuth credentials")
    auth_subparsers = auth.add_subparsers(dest="auth_command")
    login = auth_subparsers.add_parser("login", help="Login with Codex device auth")
    login.add_argument("--device-code", action="store_true")
    auth_subparsers.add_parser("status", help="Show Codex auth-file status")
    auth_subparsers.add_parser("logout", help="Remove Codex auth-file credentials")

    args = parser.parse_args(argv)
    settings = Settings.from_env()

    try:
        if args.command == "serve":
            if args.host:
                settings.host = args.host
            if args.port:
                settings.port = args.port
            if args.debug:
                settings.log_level = "DEBUG"
            run_server(settings)
            return 0
        if args.command == "auth":
            return _auth(args, settings)
        parser.print_help()
        return 2
    except AuthError as exc:
        print(f"auth error: {exc}", file=sys.stderr)
        return 1


def _auth(args: argparse.Namespace, settings: Settings) -> int:
    if args.auth_command == "login":
        if not args.device_code:
            raise AuthError("only --device-code login is supported")
        CodexDeviceAuthenticator(settings).login()
        return 0
    provider = CodexAuthFileProvider(settings)
    if args.auth_command == "status":
        print(json.dumps(provider.status(), indent=2, sort_keys=True))
        return 0
    if args.auth_command == "logout":
        provider.delete()
        print(f"Removed {provider.path}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
