import json
import tempfile
import unittest
from pathlib import Path

from atuin_ai_proxy.auth import CodexAuthFileProvider, StaticBearerProvider
from atuin_ai_proxy.settings import Settings


class AuthTests(unittest.TestCase):
    def test_static_bearer_provider_sets_expected_headers(self) -> None:
        provider = StaticBearerProvider(
            token="at-test",
            account_id="acct_123",
            fedramp=True,
        )

        self.assertEqual(
            provider.headers(),
            {
                "Authorization": "Bearer at-test",
                "ChatGPT-Account-ID": "acct_123",
                "X-OpenAI-Fedramp": "true",
            },
        )

    def test_codex_auth_file_provider_loads_codex_tokens_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            auth_file = Path(tmpdir) / "auth.json"
            auth_file.write_text(
                json.dumps(
                    {
                        "auth_mode": "chatgpt",
                        "tokens": {
                            "access_token": "access-test",
                            "refresh_token": "refresh-test",
                            "account_id": "acct_456",
                            "expires_at": 4_102_444_800,
                        },
                    }
                )
            )
            provider = CodexAuthFileProvider(
                Settings(
                    backend="codex-oauth",
                    model="gpt-test",
                    codex_auth_file=str(auth_file),
                )
            )

            self.assertEqual(
                provider.headers(),
                {
                    "Authorization": "Bearer access-test",
                    "ChatGPT-Account-ID": "acct_456",
                },
            )

    def test_codex_auth_file_provider_loads_personal_access_token_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            auth_file = Path(tmpdir) / "auth.json"
            auth_file.write_text(
                json.dumps(
                    {
                        "auth_mode": "chatgpt",
                        "personal_access_token": {
                            "access_token": "at-test",
                            "account_id": "acct_789",
                            "is_fedramp_account": True,
                        },
                    }
                )
            )
            provider = CodexAuthFileProvider(
                Settings(
                    backend="codex-oauth",
                    model="gpt-test",
                    codex_auth_file=str(auth_file),
                )
            )

            self.assertEqual(
                provider.headers(),
                {
                    "Authorization": "Bearer at-test",
                    "ChatGPT-Account-ID": "acct_789",
                    "X-OpenAI-Fedramp": "true",
                },
            )


if __name__ == "__main__":
    unittest.main()
