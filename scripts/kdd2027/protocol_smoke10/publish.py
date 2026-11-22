"""CPU-only final integrity/export; no GPU scheduling or model calls."""

import argparse
import json
import subprocess
import time

from PIL import Image, ImageChops
from prepare import OUT, REPO
from report import report

from primary_ml_cka.baselines.protocol import atomic, digest, sha, valid_receipt


def publish(push=False):
    rows = report()
    checks, audits = [], []
    for path in sorted((OUT / "runs").glob("*/lock.json")):
        lock = json.loads(path.read_text())
        if any(sha(REPO / n) != h for n, h in lock["scientific_sources"].items()):
            continue
        for row in lock["manifest"]["images"]:
            receipt = path.parent / "attacks" / digest(row["image_id"])[:18] / "receipt.json"
            valid = valid_receipt(receipt, digest([lock, row]))
            linf = None
            if valid:
                r = json.loads(receipt.read_text())
                with Image.open(r["clean"]) as clean, Image.open(r["adv"]) as adv:
                    diff = ImageChops.difference(clean.convert("RGB"), adv.convert("RGB"))
                    linf = max(v[1] for v in diff.getextrema())
                valid = (
                    linf <= 16
                    and r["target_access"] is False
                    and r["steps"] == lock["config"]["steps"]
                )
            checks.append(
                {
                    "run": path.parent.name,
                    "image_id": row["image_id"],
                    "valid": valid,
                    "linf_uint8": linf,
                }
            )
        for file in sorted(path.parent.glob("audits/*.json")):
            audits.append(
                {"path": str(file), "sha256": sha(file), "audit": json.loads(file.read_text())}
            )
        file = path.parent / "bf16_proxies.json"
        if file.exists():
            audits.append(
                {"path": str(file), "sha256": sha(file), "audit": json.loads(file.read_text())}
            )
    state = json.loads((OUT / "state.json").read_text()) if (OUT / "state.json").exists() else {}
    integrity = {
        "state": state,
        "all_planned_raw_cells_complete": len(rows) == 24
        and all(r["returned"] == r["planned"] for r in rows),
        "all_present_attacks_valid": bool(checks) and all(r["valid"] for r in checks),
        "official_llava_scoring_complete": False,
        "official_llava_blocker": "GPT4 access not configured; no replacement judge",
        "scored_commercial_requests": 0,
        "attacks": checks,
    }
    destination = REPO / "docs/kdd2027/results/protocol_smoke10"
    atomic(destination / "integrity.json", integrity)
    atomic(destination / "model_audits.json", audits)
    atomic(destination / "state.json", state)
    access = OUT / "llava_official/access.json"
    if access.exists():
        atomic(destination / "official_llava_access.json", json.loads(access.read_text()))
    if push:
        relative = str(destination.relative_to(REPO))
        subprocess.run(["git", "add", "--", relative], cwd=REPO, check=True)
        dirty = subprocess.run(
            ["git", "diff", "--cached", "--quiet", "--", relative], cwd=REPO
        ).returncode
        if dirty:
            subprocess.run(
                [
                    "git",
                    "commit",
                    "-m",
                    "Report CrossVLAD and LLaVA smoke integrity and measured progress",
                    "--",
                    relative,
                ],
                cwd=REPO,
                check=True,
            )
        subprocess.run(
            [
                "git",
                "push",
                "git@github.com:pe0-p1e/exper-tran.git",
                "HEAD:experiment/kdd2027-baseline-reproduction",
            ],
            cwd=REPO,
            check=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--wait", action="store_true")
    p.add_argument("--push", action="store_true")
    args = p.parse_args()
    if args.wait:
        while True:
            file = OUT / "state.json"
            state = json.loads(file.read_text()) if file.exists() else {}
            if state.get("status") in [
                "RAW_COMPLETE_OFFICIAL_LLAVA_JUDGE_BLOCKED",
                "PARTIAL_RESUMABLE",
                "BLOCKED",
            ]:
                break
            time.sleep(30)
    publish(args.push)
