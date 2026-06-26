import unittest

from atuin_ai_proxy.diagnostics import redact


class DiagnosticsTests(unittest.TestCase):
    def test_redact_removes_nested_sensitive_values(self) -> None:
        value = {
            "Authorization": "Bearer sk-secret",
            "messages": [{"content": "keep this"}],
            "nested": {
                "access_token": "at-secret",
                "account_id": "acct-secret",
                "safe": "visible",
            },
            "tokens": ["safe", {"refresh_token": "refresh-secret"}],
        }

        redacted = redact(value)

        self.assertEqual(redacted["Authorization"], "[REDACTED]")
        self.assertEqual(redacted["messages"][0]["content"], "keep this")
        self.assertEqual(redacted["nested"]["access_token"], "[REDACTED]")
        self.assertEqual(redacted["nested"]["account_id"], "[REDACTED]")
        self.assertEqual(redacted["nested"]["safe"], "visible")
        self.assertEqual(redacted["tokens"][1]["refresh_token"], "[REDACTED]")


if __name__ == "__main__":
    unittest.main()
