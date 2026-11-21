import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cli_transport


class CLITransportTests(unittest.TestCase):
    def test_non_request_command_delegated(self):
        with patch.object(cli_transport, "original", return_value="delegated") as original:
            self.assertEqual(cli_transport.run(["codex", "login", "status"]), "delegated")
            original.assert_called_once()

    def test_request_preserves_command_and_uses_owned_group(self):
        class Process:
            returncode = 0

            def communicate(self, **kwargs):
                self.kwargs = kwargs
                return "structured-output", "warnings"

        process = Process()
        command = ["codex", "exec", "--model", "gpt-6.1-sol", "-"]
        with patch.object(cli_transport.subprocess, "Popen", return_value=process) as launch:
            result = cli_transport.run(
                command, input="unchanged prompt", capture_output=True, text=True, timeout=180
            )
        self.assertEqual(launch.call_args.args[0], command)
        self.assertTrue(launch.call_args.kwargs["start_new_session"])
        self.assertEqual(process.kwargs, {"input": "unchanged prompt", "timeout": 600})
        self.assertEqual(result.stdout, "structured-output")
        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
