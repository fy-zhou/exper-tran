"""Historical blinded CLI protocol, with bounded transport/capture and owned process groups."""

import os
import runpy
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "experiments/2026-10-a100-precommercial-v7-targeted-50/src/run_codex_cli_pilot.py"
sys.path.insert(0, str(RUNNER.parent))
original = subprocess.run


def run(command, *args, **kwargs):
    if not (isinstance(command, list) and command[:2] == ["codex", "exec"]):
        return original(command, *args, **kwargs)
    payload = kwargs.get("input")
    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(input=payload, timeout=600)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
        folder = Path(command[command.index("--output-last-message") + 1]).parent
        (folder / "timeout_stdout.jsonl").write_text(stdout or "")
        (folder / "timeout_stderr.txt").write_text(stderr or "")
        raise subprocess.TimeoutExpired(command, 600, output=stdout, stderr=stderr) from None
    return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)


if __name__ == "__main__":
    subprocess.run = run
    sys.argv = [str(RUNNER), *sys.argv[1:]]
    runpy.run_path(str(RUNNER), run_name="__main__")
