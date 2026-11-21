# ruff: noqa: E402
"""Reuse frozen PNGs and the verified subscription-CLI blinded runner; no API fallback."""

import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path("/home/ubuntu/exper-tran/outputs/kdd2027_baseline_reproduction")
STATE = OUT / "nov21/gpt61_low"
V7 = ROOT / "experiments/2026-10-a100-precommercial-v7-targeted-50"
sys.path.insert(0, str(ROOT / "src"))
from primary_ml_cka.baselines import campaign as legacy
from primary_ml_cka.baselines.protocol import atomic, digest, freeze, sha, valid_receipt


def generations():
    original = json.loads((OUT / "manifests/final150.json").read_text())["images"]
    methods = {}
    for name, budget in [("m_attack", 300), ("pgd", 50), ("mi_fgsm", 50)]:
        root = OUT / "runs" / legacy_run(budget)
        methods[name] = [
            (r, root / "attacks" / name / digest(r["image_id"])[:18] / "receipt.json")
            for r in original
        ]
    for name, root in [
        ("foa_attack", OUT / "extension/runs/imagenet_foa_attack_29c5c7cef8fc733e71"),
        ("anyattack", OUT / "prepared_remaining/generation_anyattack"),
    ]:
        methods[name] = [
            (r, root / "attacks" / digest(r["image_id"])[:18] / "receipt.json") for r in original
        ]
    return methods


def legacy_run(budget):
    return "final150_300_87488fbb38aae77f36" if budget == 300 else "final150_50_206015cb1202ed50f1"


def packet(method, pairs, n):
    # Balanced deterministic10-image connectivity cohort; no target-informed selection.
    if n == 10:
        selected = []
        for tid in ["T01", "T03", "T05", "T07", "T09"]:
            selected.extend([(r, p) for r, p in pairs if r["transition"] == tid][:2])
    else:
        selected = pairs
    assert len(selected) == n
    base = STATE / "packets" / f"{method}_{n}"
    prompt = legacy.canonical_prompt()
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    cases = []
    requests = []
    for row, receipt in selected:
        value = json.loads(receipt.read_text())
        if not valid_receipt(receipt, value["signature"]):
            raise RuntimeError("Invalid frozen attack " + str(receipt))
        ids = {}
        for kind, path in [("clean", value["clean"]), ("adversarial", value["adv"])]:
            image_hash = sha(path)
            rid = "r" + digest(["nov21_gpt61_low", image_hash, prompt_hash])[:26]
            ids[kind] = rid
            folder = base / "blinded" / rid
            folder.mkdir(parents=True, exist_ok=True)
            link = folder / "image.png"
            if not link.exists():
                link.symlink_to(Path(path).resolve())
            (folder / "prompt.txt").write_text(prompt)
            requests.append(
                {"request_id": rid, "image_sha256": image_hash, "prompt_sha256": prompt_hash}
            )
        cases.append(
            {
                "image_id": row["image_id"],
                "transition": row["transition"],
                "source_label": row["source"],
                "target_label": row["target"],
                "requests": ids,
            }
        )
    freeze(
        base / "private/manifest.json",
        {
            "cases": cases,
            "requests": requests,
            "method": method,
            "cohort": "fixed150" if n == 150 else "fixed150_balanced10_prefix_not_independent",
            "parameters_not_selected_from_GPT": True,
        },
    )
    order = [r["request_id"] for r in requests]
    random.Random(42).shuffle(order)
    freeze(base / "request_order.json", order)
    return base


def summarize(method, base, n):
    cases = json.loads((base / "private/manifest.json").read_text())["cases"]
    measurements = []
    for c in cases:
        paths = {
            k: STATE / "responses/gpt-6.1-sol" / rid / "result.json"
            for k, rid in c["requests"].items()
        }
        if not all(p.exists() for p in paths.values()):
            continue
        clean = json.loads(paths["clean"].read_text())
        adv = json.loads(paths["adversarial"].read_text())
        measurements.append(
            {
                **{k: c[k] for k in ["image_id", "transition", "source_label", "target_label"]},
                "clean_label": clean["parsed_label"],
                "adv_label": adv["parsed_label"],
                "hit": adv["parsed_label"] == c["target_label"],
                "request_ids": c["requests"],
            }
        )
    result = {
        "method": method,
        "model": "gpt-6.1-sol",
        "reasoning_effort": "low",
        "N": n,
        "completed_pairs": len(measurements),
        "hits": sum(r["hit"] for r in measurements),
        "ASR": sum(r["hit"] for r in measurements) / n if len(measurements) == n else None,
        "rows": measurements,
        "selection_feedback": False,
        "server_snapshot": "undisclosed",
    }
    atomic(STATE / "summaries" / f"{method}_{n}.json", result)
    print("SUMMARY", method, n, len(measurements), result["ASR"], flush=True)


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    auth = subprocess.run(["codex", "login", "status"], capture_output=True, text=True)
    if "Logged in using ChatGPT" not in auth.stdout + auth.stderr:
        raise RuntimeError("BLOCKED_AUTH; no paid fallback")
    catalog = json.loads((Path.home() / ".codex/models_cache.json").read_text())
    selected = next(m for m in catalog["models"] if m["slug"] == "gpt-6.1-sol")
    assert "image" in selected["input_modalities"]
    freeze(
        STATE / "protocol.json",
        {
            "model": "gpt-6.1-sol",
            "reasoning_effort": "low",
            "client": subprocess.check_output(["codex", "--version"], text=True).strip(),
            "methods": ["m_attack", "foa_attack", "pgd", "mi_fgsm", "anyattack"],
            "cohort": sha(OUT / "manifests/final150.json"),
            "stages": [10, 150],
            "no_hyperparameter_tuning": True,
            "no_paid_API": True,
            "runner_sha256": sha(V7 / "src/run_codex_cli_pilot.py"),
        },
    )
    methods = generations()
    for n in [10, 150]:
        for method in ["m_attack", "foa_attack", "pgd", "mi_fgsm", "anyattack"]:
            base = packet(method, methods[method], n)
            log = STATE / f"{method}_{n}.log"
            with log.open("a") as f:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).parent / "cli_transport.py"),
                        "--model",
                        "gpt-6.1-sol",
                        "--images",
                        str(n),
                        "--pack-dir",
                        str(base),
                        "--result-dir",
                        str(STATE / "responses/gpt-6.1-sol"),
                    ],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env={
                        **os.environ,
                        "V7_DATA_ROOT": str(OUT.parent / "a100_precommercial_v7_targeted50"),
                    },
                )
            summarize(method, base, n)
            if result.returncode:
                atomic(
                    STATE / "blocked.json",
                    {
                        "method": method,
                        "N": n,
                        "exit": result.returncode,
                        "time": time.time(),
                        "log": str(log),
                        "no_repeated_scored_request": True,
                    },
                )
                raise SystemExit("CLI blocked: inspect raw trace; no automatic auth/quota bypass")
    atomic(
        STATE / "complete.json",
        {"status": "COMPLETE", "methods": 5, "N_per_method": 150, "time": time.time()},
    )


if __name__ == "__main__":
    main()
