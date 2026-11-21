"""Same-date frozen Uniform4 control, after baseline CLI queue; no attack tuning."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import gpt_baselines as G

from primary_ml_cka.baselines.protocol import atomic, digest, freeze, sha, valid_receipt


def main():
    while not (G.STATE / "complete.json").exists():
        blocked = G.STATE / "blocked.json"
        if blocked.exists():
            print("BLOCKED_PRECEDING_CLI_QUEUE; no new calls", flush=True)
            return 1
        time.sleep(30)
    data = json.loads((G.OUT / "audits/ours.json").read_text())
    condition = next(
        c for c in data["conditions"] if c["family"] == "qwen" and c["method"] == "uniform"
    )
    assert (
        condition["eligible"] == 150
        and condition["valid_attacks"] == 150
        and not condition["errors"]
    )
    raw = {r["image_id"]: Path(r["receipt"]) for r in condition["receipts"]}
    pairs = []
    rows = json.loads((G.OUT / "manifests/final150.json").read_text())["images"]
    for row in rows:
        original = raw[row["image_id"]]
        value = json.loads(original.read_text())
        if value["status"] != "COMPLETE" or value["target_access"] or value["steps"] != 50:
            raise RuntimeError("Invalid Ours provenance")
        if sha(value["clean"]) != row["clean_sha256"]:
            raise RuntimeError("Different clean cohort")
        imported = G.STATE / "ours_imported_receipts" / (digest(row["image_id"])[:18] + ".json")
        freeze(
            imported,
            {
                **value,
                "status": "ATTACK_FROZEN",
                "signature": digest([sha(original), "CLI_readonly_import"]),
                "original_receipt": str(original),
                "original_receipt_sha256": sha(original),
            },
        )
        if not valid_receipt(imported, json.loads(imported.read_text())["signature"]):
            raise RuntimeError("PNG validation failed")
        pairs.append((row, imported))
    freeze(
        G.STATE / "ours_control_protocol.json",
        {
            "method": "ours_uniform4",
            "condition": condition["config"],
            "N": 150,
            "reasoning_effort": "low",
            "model": "gpt-6.1-sol",
            "same_current_date_requests_as_baselines": True,
            "current_target_feedback_for_selection": False,
            "historical_GPT50_used_for_selection_disclosed": data["GPT50_used_for_selection"],
            "original_audit_sha256": sha(G.OUT / "audits/ours.json"),
        },
    )
    base = G.packet("ours_uniform4", pairs, 150)
    with (G.STATE / "ours_uniform4_150.log").open("a") as f:
        r = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "cli_transport.py"),
                "--model",
                "gpt-6.1-sol",
                "--images",
                "150",
                "--pack-dir",
                str(base),
                "--result-dir",
                str(G.STATE / "responses/gpt-6.1-sol"),
            ],
            stdout=f,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "V7_DATA_ROOT": str(G.OUT.parent / "a100_precommercial_v7_targeted50"),
            },
        )
    G.summarize("ours_uniform4", base, 150)
    atomic(
        G.STATE / "ours_control_done.json",
        {
            "status": "COMPLETE" if not r.returncode else "BLOCKED",
            "exit": r.returncode,
            "time": time.time(),
        },
    )
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
