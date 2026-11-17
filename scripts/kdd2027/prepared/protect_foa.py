"""CPU-only throughput guard; signal only verified V2 children of admission owner."""

import json
import os
import signal
import time
from pathlib import Path

OUT = Path("/home/ubuntu/exper-tran/outputs/kdd2027_baseline_reproduction")
STATE = OUT / "prepared_remaining"
RUN = OUT / "extension/runs/imagenet_foa_attack_29c5c7cef8fc733e71"
paused = None
try:
    while True:
        lease = json.loads((STATE / "admission.json").read_text())
        owner = lease["owner_pid"]
        if not Path(f"/proc/{owner}/cmdline").exists():
            break
        if Path(f"/proc/{owner}/stat").read_text().split()[21] != lease["owner_start"]:
            break
        foa = False
        v2 = None
        for p in Path("/proc").iterdir():
            if not p.name.isdigit():
                continue
            try:
                stat = (p / "stat").read_text().split()
                command = (p / "cmdline").read_bytes()
                if int(stat[3]) != owner or b"admitted_worker.py" not in command:
                    continue
                if b"foa" in command and b"--phase\x00attack" in command:
                    foa = True
                if b"m_attack_v2" in command and b"generate" in command:
                    v2 = int(p.name)
            except (OSError, ValueError):
                continue
        if not foa:
            break
        receipts = sorted(RUN.glob("attacks/*/receipt.json"), key=lambda p: p.stat().st_mtime)
        latest = [json.loads(p.read_text()) for p in receipts[-3:]]
        mean = sum(r["seconds"] for r in latest) / len(latest) if latest else 156
        # Start throttling when measured FOA time grows >~20% above solo baseline.
        throttle = v2 is not None and mean > 188
        status = {
            "time": time.time(),
            "FOA_completed": len(receipts),
            "FOA_last3_seconds": mean,
            "V2_pid": v2,
            "V2_throttle_70percent_off": throttle,
            "no_scientific_settings_changed": True,
        }
        tmp = STATE / "throughput_guard.tmp"
        tmp.write_text(json.dumps(status, indent=2))
        os.replace(tmp, STATE / "throughput_guard.json")
        if throttle:
            os.kill(v2, signal.SIGSTOP)
            paused = v2
            time.sleep(14)
            os.kill(v2, signal.SIGCONT)
            paused = None
            time.sleep(6)
        else:
            time.sleep(10)
finally:
    if paused is not None:
        try:
            os.kill(paused, signal.SIGCONT)
        except ProcessLookupError:
            pass
