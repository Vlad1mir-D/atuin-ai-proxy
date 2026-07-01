import os
import unittest
from unittest.mock import patch

from atuin_ai_proxy.settings import Settings


class SettingsTests(unittest.TestCase):
    def test_openai_api_comes_from_environment(self) -> None:
        env = {
            "BACKEND": "openai",
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_API": "chat_completions",
        }

        with patch.dict(os.environ, env, clear=True):
            settings = Settings.from_env()

        self.assertEqual(settings.openai_api, "chat_completions")

    def test_openai_api_auto_tries_chat_completions_before_responses(self) -> None:
        settings = Settings(
            backend="openai",
            model="gpt-test",
            openai_api_key="sk-test",
        )

        self.assertEqual(
            settings.openai_api_candidates(),
            ("chat_completions", "responses"),
        )

    def test_openai_api_explicit_responses_skips_chat_completions(self) -> None:
        settings = Settings(
            backend="openai",
            model="gpt-test",
            openai_api_key="sk-test",
            openai_api="responses",
        )

        self.assertEqual(settings.openai_api_candidates(), ("responses",))

    def test_trace_payload_bytes_comes_from_environment(self) -> None:
        env = {
            "BACKEND": "openai",
            "OPENAI_API_KEY": "sk-test",
            "LOG_LEVEL": "TRACE",
            "TRACE_PAYLOAD_BYTES": "1234",
        }

        with patch.dict(os.environ, env, clear=True):
            settings = Settings.from_env()

        self.assertEqual(settings.log_level, "TRACE")
        self.assertEqual(settings.trace_payload_bytes, 1234)


if __name__ == "__main__":
    unittest.main()
