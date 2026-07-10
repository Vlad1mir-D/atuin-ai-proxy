import os
import unittest
from contextlib import redirect_stderr
from io import StringIO
from unittest.mock import patch

from atuin_ai_proxy.cli import main


class CliTests(unittest.TestCase):
    def test_serve_accepts_each_supported_log_level(self) -> None:
        for log_level in ("TRACE", "DEBUG", "INFO", "ERROR"):
            with self.subTest(log_level=log_level):
                with (
                    patch.dict(os.environ, {}, clear=True),
                    patch("atuin_ai_proxy.cli.run_server") as run_server,
                ):
                    result = main(["serve", "--log-level", log_level.lower()])

                self.assertEqual(result, 0)
                settings = run_server.call_args.args[0]
                self.assertEqual(settings.log_level, log_level)

    def test_serve_log_level_overrides_environment(self) -> None:
        with (
            patch.dict(os.environ, {"LOG_LEVEL": "INFO"}, clear=True),
            patch("atuin_ai_proxy.cli.run_server") as run_server,
        ):
            result = main(["serve", "--log-level", "ERROR"])

        self.assertEqual(result, 0)
        settings = run_server.call_args.args[0]
        self.assertEqual(settings.log_level, "ERROR")

    def test_serve_rejects_removed_and_unsupported_log_level_flags(self) -> None:
        for arguments in (("--debug",), ("--log-level", "WARNING")):
            with self.subTest(arguments=arguments):
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
                    main(["serve", *arguments])

                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
