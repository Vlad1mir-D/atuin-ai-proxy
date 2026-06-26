from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import stat
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

from .settings import Settings


class AuthError(RuntimeError):
    pass


@dataclass(slots=True)
class StaticBearerProvider:
    token: str
    account_id: str | None = None
    fedramp: bool = False

    def headers(self) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self.token}"}
        if self.account_id:
            headers["ChatGPT-Account-ID"] = self.account_id
        if self.fedramp:
            headers["X-OpenAI-Fedramp"] = "true"
        return headers

    def refresh(self) -> bool:
        return False


class CodexAccessTokenProvider(StaticBearerProvider):
    def __init__(self, settings: Settings) -> None:
        if not settings.codex_access_token:
            raise AuthError("CODEX_ACCESS_TOKEN is required for BACKEND=codex-token")
        super().__init__(
            settings.codex_access_token,
            settings.codex_account_id,
            settings.codex_fedramp,
        )
        self.settings = settings
        self._metadata_loaded = bool(settings.codex_account_id)

    def headers(self) -> dict[str, str]:
        if not self._metadata_loaded:
            self._load_metadata()
        return super().headers()

    def _load_metadata(self) -> None:
        url = urljoin(
            self.settings.codex_auth_api_base_url.rstrip("/") + "/",
            "api/accounts/v1/user-auth-credential/whoami",
        )
        request = Request(url, headers={"Authorization": f"Bearer {self.token}"})
        try:
            with urlopen(request, timeout=self.settings.request_timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - network dependent
            raise AuthError(
                "CODEX_ACCOUNT_ID is not set and Codex token metadata lookup failed"
            ) from exc

        self.account_id = payload.get("chatgpt_account_id") or payload.get("account_id")
        self.fedramp = bool(payload.get("chatgpt_account_is_fedramp"))
        self._metadata_loaded = True


class CodexAuthFileProvider:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.path = settings.resolved_codex_auth_file
        self._loaded: dict[str, Any] | None = None

    def headers(self) -> dict[str, str]:
        data = self._load()
        if self._should_refresh(data):
            self.refresh()
            data = self._load(force=True)

        token, account_id, fedramp = self._extract_credentials(data)
        headers = {"Authorization": f"Bearer {token}"}
        if account_id:
            headers["ChatGPT-Account-ID"] = account_id
        if fedramp:
            headers["X-OpenAI-Fedramp"] = "true"
        return headers

    def refresh(self) -> bool:
        data = self._load()
        tokens = data.get("tokens") if isinstance(data.get("tokens"), dict) else {}
        refresh_token = tokens.get("refresh_token")
        if not refresh_token:
            return False

        response = _json_request(
            urljoin(self.settings.codex_issuer.rstrip("/") + "/", "oauth/token"),
            {
                "client_id": self.settings.codex_client_id,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            },
            timeout=self.settings.request_timeout_seconds,
        )
        expires_in = int(response.get("expires_in") or 3600)
        tokens["access_token"] = response["access_token"]
        tokens["refresh_token"] = response.get("refresh_token", refresh_token)
        tokens["expires_at"] = int(time.time()) + expires_in
        if response.get("id_token"):
            tokens["id_token"] = response["id_token"]
        self._write(data)
        self._loaded = data
        return True

    def status(self) -> dict[str, Any]:
        data = self._load()
        token, account_id, fedramp = self._extract_credentials(data)
        return {
            "auth_file": str(self.path),
            "has_access_token": bool(token),
            "account_id": account_id,
            "fedramp": fedramp,
            "expires_at": (data.get("tokens") or {}).get("expires_at")
            if isinstance(data.get("tokens"), dict)
            else None,
        }

    def delete(self) -> None:
        self.path.unlink(missing_ok=True)
        self._loaded = None

    def _load(self, force: bool = False) -> dict[str, Any]:
        if self._loaded is not None and not force:
            return self._loaded
        if not self.path.exists():
            raise AuthError(
                f"Codex auth file not found at {self.path}. Run auth login or mount auth.json."
            )
        self._loaded = json.loads(self.path.read_text())
        return self._loaded

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        os.chmod(tmp_path, stat.S_IRUSR | stat.S_IWUSR)
        os.replace(tmp_path, self.path)

    def _should_refresh(self, data: dict[str, Any]) -> bool:
        tokens = data.get("tokens") if isinstance(data.get("tokens"), dict) else {}
        expires_at = tokens.get("expires_at")
        return bool(expires_at and int(expires_at) <= int(time.time()) + 60)

    def _extract_credentials(self, data: dict[str, Any]) -> tuple[str, str | None, bool]:
        pat = data.get("personal_access_token")
        if isinstance(pat, dict) and pat.get("access_token"):
            return (
                pat["access_token"],
                pat.get("account_id") or pat.get("chatgpt_account_id"),
                bool(pat.get("is_fedramp_account") or pat.get("chatgpt_account_is_fedramp")),
            )

        tokens = data.get("tokens")
        if isinstance(tokens, dict) and tokens.get("access_token"):
            return (
                tokens["access_token"],
                tokens.get("account_id") or tokens.get("chatgpt_account_id"),
                bool(tokens.get("is_fedramp_account") or tokens.get("chatgpt_account_is_fedramp")),
            )

        raise AuthError(f"No Codex access token found in {self.path}")


class CodexDeviceAuthenticator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.path = settings.resolved_codex_auth_file

    def login(self) -> None:
        verifier, _challenge = _pkce_pair()
        user_code = _json_request(
            urljoin(
                self.settings.codex_issuer.rstrip("/") + "/",
                "api/accounts/deviceauth/usercode",
            ),
            {"client_id": self.settings.codex_client_id},
            timeout=self.settings.request_timeout_seconds,
        )
        code = user_code["user_code"]
        device_auth_id = user_code["device_auth_id"]
        interval = int(user_code.get("interval") or 5)
        expires_in = int(user_code.get("expires_in") or 900)
        verification_url = (
            user_code.get("verification_uri")
            or user_code.get("verification_url")
            or urljoin(self.settings.codex_issuer.rstrip("/") + "/", "codex/device")
        )

        print(f"Open {verification_url} and enter code: {code}", flush=True)
        deadline = time.time() + expires_in
        authorization_code = None
        while time.time() < deadline:
            time.sleep(interval)
            try:
                poll = _json_request(
                    urljoin(
                        self.settings.codex_issuer.rstrip("/") + "/",
                        "api/accounts/deviceauth/token",
                    ),
                    {"device_auth_id": device_auth_id, "user_code": code},
                    timeout=self.settings.request_timeout_seconds,
                )
            except HTTPError as exc:
                if exc.code in {400, 401, 428}:
                    continue
                raise
            authorization_code = poll.get("authorization_code") or poll.get("code")
            if authorization_code:
                break

        if not authorization_code:
            raise AuthError("Timed out waiting for Codex device authorization")

        tokens = _form_request(
            urljoin(self.settings.codex_issuer.rstrip("/") + "/", "oauth/token"),
            {
                "grant_type": "authorization_code",
                "code": authorization_code,
                "redirect_uri": urljoin(
                    self.settings.codex_issuer.rstrip("/") + "/",
                    "deviceauth/callback",
                ),
                "client_id": self.settings.codex_client_id,
                "code_verifier": verifier,
            },
            timeout=self.settings.request_timeout_seconds,
        )
        expires_in = int(tokens.get("expires_in") or 3600)
        auth_json = {
            "auth_mode": "chatgpt",
            "tokens": {
                "id_token": tokens.get("id_token"),
                "access_token": tokens["access_token"],
                "refresh_token": tokens.get("refresh_token"),
                "account_id": tokens.get("account_id") or tokens.get("chatgpt_account_id"),
                "expires_at": int(time.time()) + expires_in,
            },
        }
        provider = CodexAuthFileProvider(self.settings)
        provider._write(auth_json)
        print(f"Saved Codex credentials to {self.path}", flush=True)


def provider_for_settings(settings: Settings) -> StaticBearerProvider | CodexAuthFileProvider:
    if settings.backend == "openai":
        if not settings.openai_api_key:
            raise AuthError("OPENAI_API_KEY is required for BACKEND=openai")
        return StaticBearerProvider(settings.openai_api_key)
    if settings.backend == "codex-token":
        return CodexAccessTokenProvider(settings)
    return CodexAuthFileProvider(settings)


def _json_request(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _form_request(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    body = urlencode(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _pkce_pair() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).decode().rstrip("=")
    return verifier, challenge
