"""CPU-only live/final evidence aggregation; never launches inference."""

import argparse
import csv
import hashlib
import json
import subprocess
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path("/home/ubuntu/exper-tran/outputs/kdd2027_baseline_reproduction/nov21")
DEST = ROOT / "docs/kdd2027/results/nov21_multitask"
TARGETS = ["Qwen3.5-27B", "InternVL3.5-14B-HF", "Gemma 4 31B-it"]
NEW = ["attackvlm_mfii", "attackvlm_mfit", "ssa_cwa_clip"]
OLD = ["m_attack", "foa_attack", "pgd", "mi_fgsm", "anyattack"]


def read(p, default=None):
    return json.loads(p.read_text()) if p.exists() else default


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def collect():
    reports = []
    inventory = []
    current_sources = {
        p.name: sha(p)
        for p in [
            ROOT / "scripts/kdd2027/nov21/worker.py",
            ROOT / "scripts/kdd2027/nov21/methods.py",
        ]
    }
    for lock in sorted((OUT / "local").glob("*/lock.json")):
        d = read(lock)
        c = d["config"]
        run = lock.parent
        inventory.append(
            {
                "run": run.name,
                "method": c["method"],
                "dataset": c["dataset"],
                "N": c["N"],
                "engineering": c["engineering"],
                "steps": c["steps"],
                "frozen_images": len(list(run.glob("attacks/*/receipt.json"))),
                "evaluated_queries": len(list(run.glob("evaluations/*/*.json"))),
                "lock_sha256": sha(lock),
            }
        )
        if c["engineering"] or d["scientific_sources"] != current_sources:
            continue
        for row in read(run / "task_summary.json", []):
            reports.append({"run": run.name, **row})
    gpt = []
    responses = OUT / "gpt61_low/responses/gpt-6.1-sol"
    for path in sorted((OUT / "gpt61_low/packets").glob("*/private/manifest.json")):
        packet = read(path)
        pairs = []
        for case in packet["cases"]:
            paths = {k: responses / v / "result.json" for k, v in case["requests"].items()}
            if not all(p.exists() for p in paths.values()):
                continue
            clean = read(paths["clean"])
            adv = read(paths["adversarial"])
            pairs.append(
                {
                    "image_id": case["image_id"],
                    "hit": adv["parsed_label"] == case["target_label"],
                    "clean_invalid": clean["parsed_label"] is None,
                    "adv_invalid": adv["parsed_label"] is None,
                }
            )
        n = len(packet["cases"])
        hits = sum(p["hit"] for p in pairs)
        gpt.append(
            {
                "method": packet["method"],
                "N": n,
                "completed_pairs": len(pairs),
                "hits": hits,
                "ASR": hits / n if len(pairs) == n else None,
                "clean_parser_invalid_observed": sum(p["clean_invalid"] for p in pairs),
                "adv_parser_invalid_observed": sum(p["adv_invalid"] for p in pairs),
            }
        )
    cells = []
    for dataset, methods in [
        ("imagenet", NEW),
        ("coco_captioning", ["m_attack", "foa_attack", "pgd"]),
        ("llava_bench_coco", ["m_attack", "foa_attack", "pgd"]),
    ]:
        for method in methods:
            for target in TARGETS:
                found = [
                    r
                    for r in reports
                    if r["dataset"] == dataset and r["method"] == method and r["target"] == target
                ]
                row = found[-1] if found else None
                cells.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "target": target,
                        "status": row["status"] if row else "PENDING",
                        "N": row["planned_images"]
                        if row
                        else 150
                        if dataset == "imagenet"
                        else 10
                        if dataset == "coco_captioning"
                        else 3,
                        "ASR_percent": 100 * row["ASR"]
                        if row and row.get("ASR") is not None
                        else None,
                    }
                )
    counts = {s: sum(c["status"] == s for c in cells) for s in {c["status"] for c in cells}}
    status = {
        "time_utc": time.strftime("%FT%TZ", time.gmtime()),
        "planned_open_target_cells": 27,
        "counts": counts,
        "GPT_full150_complete": sum(
            r["N"] == 150 and r["completed_pairs"] == 150 and r["method"] in OLD for r in gpt
        ),
        "GPT_methods_planned": 5,
        "GPT_ours_current_date_control": read(OUT / "gpt61_low/ours_control_done.json", {}),
        "GPT_blocked": read(OUT / "gpt61_low/blocked.json"),
        "local_queue_finished": (OUT / "local_finished.json").exists(),
    }
    return status, cells, reports, gpt, inventory


def export():
    DEST.mkdir(parents=True, exist_ok=True)
    status, cells, reports, gpt, inventory = collect()
    (DEST / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    (DEST / "all_local_summaries.json").write_text(json.dumps(reports, indent=2) + "\n")
    (DEST / "GPT_summaries.json").write_text(json.dumps(gpt, indent=2) + "\n")
    write_csv(DEST / "completion_matrix.csv", cells)
    write_csv(DEST / "run_inventory.csv", inventory)
    text = [
        "# November21 targeted baseline and task-extension campaign",
        "",
        (
            f"Status at {status['time_utc']}: {status['counts']}. "
            "Missing results are N/A, never zeros."
        ),
        "",
        "ImageNet new methods: AttackVLM MF-ii and MF-it (two variants of one paper), "
        "and the official SSA-CWA optimizer adapted to a CLIP ten-class zero-shot classifier. "
        "Full150 source cohort and all three fixed large targets. Target ou"
        "tput never enters generation.",
        "",
        "COCO:10 fixed images, concise captioning, clean/adv CIDEr on this subset and "
        "target-caption alignment. LLaVA:3 fixed original benchmark images,"
        "9 original open questions; "
        "raw clean/adv answers and reference ROUGE-L diagnostic. These are "
        "initial task trials, not "
        "full benchmark results. ROUGE-L is not official VQA correctness; G"
        "PT4 judge remains pending.",
        "",
        "|Task|Method|Target|N|Status|ASR|",
        "|---|---|---|---:|---|---:|",
    ]
    for c in cells:
        value = f"{c['ASR_percent']:.2f}%" if c["ASR_percent"] is not None else "N/A"
        text.append(f"|{c['dataset']}|{c['method']}|{c['target']}|{c['N']}|{c['status']}|{value}|")
    text += [
        "",
        "## Caption task trial (raw pycocoevalcap CIDEr units, same10-image IDF set)",
        "",
        "|Method|Target|Clean CIDEr|Adv CIDEr|",
        "|---|---|---:|---:|",
    ]
    for r in reports:
        if r["dataset"] == "coco_captioning":
            text.append(
                f"|{r['method']}|{r['target']}|{r.get('clean_CIDEr_valid_subset',float('nan')):.4f}|{r.get('adv_CIDEr_valid_subset',float('nan')):.4f}|"
            )
    text += [
        "",
        "## Subscription CLI targeted classification",
        "",
        "Requested gpt-6.1-sol, low effort, same frozen150 images, existing ChatGPT login only; "
        "private server snapshot undisclosed. Balanced10 is a transport/sam"
        "ple check from the final "
        "cohort, not an independent development set. No attack parameter is selected using it. "
        "One unscored180s timeout left a verified empty answer file; its ev"
        "idence was archived once. "
        "600s transport capture/owned process-group cleanup now apply. Scor"
        "ed results are never retried. "
        "Auth/quota failures stop calls; no API/model fallback. A current-d"
        "ate frozen Uniform4 control "
        "runs only after the five-baseline queue, sharing clean answers und"
        "er identical request identities.",
        "",
        "|Method|N|Completed pairs|Hits|ASR|",
        "|---|---:|---:|---:|---:|",
    ]
    for r in gpt:
        value = f"{100*r['ASR']:.2f}%" if r.get("ASR") is not None else "N/A"
        text.append(f"|{r['method']}|{r['N']}|{r['completed_pairs']}|{r['hits']}|{value}|")
    text += [
        "",
        "All open model floating parameters are BF16/unquantized; model-load audits, "
        "frozen manifests, raw answers, request attempts, source locks and loss traces remain on "
        f"persistent root `{OUT}`. Weights are cached only on /ephemeral. Native step counts and "
        "forward/backward budgets differ; source reference counts, text/image guidance and "
        "surrogate identities are disclosed in the plan. This is not a full"
        "y compute/information-matched ranking.",
        "",
        "IAU and V2 stay paused. Old campaigns, Ours attacks, paper and ori"
        "ginal results are untouched. "
        "No LLaVA official score or full-dataset result is invented.",
        "",
    ]
    (DEST / "REPORT.md").write_text("\n".join(text))
    return status


def validate_final():
    import numpy as np
    from PIL import Image

    image_checks = []
    audit_checks = []
    for lockfile in sorted((OUT / "local").glob("*/lock.json")):
        lock = read(lockfile)
        rows = {r["image_id"]: r for r in lock["manifest"]["images"]}
        for receipt in lockfile.parent.glob("attacks/*/receipt.json"):
            value = read(receipt)
            row = rows[value["image_id"]]
            expected = hashlib.sha256(
                json.dumps(
                    [lock, row], sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode()
            ).hexdigest()
            hashes = all(
                Path(p).exists() and sha(Path(p)) == h for p, h in value["png_hashes"].items()
            )
            clean = np.asarray(Image.open(value["clean"]).convert("RGB"), dtype=np.int16)
            adv = np.asarray(Image.open(value["adv"]).convert("RGB"), dtype=np.int16)
            bound = int(np.abs(clean - adv).max())
            image_checks.append(
                {
                    "run": lockfile.parent.name,
                    "image_id": row["image_id"],
                    "pass": hashes
                    and value["signature"] == expected
                    and bound <= 16
                    and not value["target_access"]
                    and value["steps"] == lock["config"]["steps"],
                    "linf_uint8": bound,
                }
            )
        audit_paths = list(lockfile.parent.glob("audits/*.json")) + [
            lockfile.parent / "bf16_proxies.json"
        ]
        for path in audit_paths:
            if not path.exists():
                continue
            data = read(path)
            for a in data if isinstance(data, list) else [data]:
                quant = a.get("quantization", {})
                audit_checks.append(
                    {
                        "source": str(path),
                        "model": a.get("model"),
                        "pass": a.get("status") == "PASS"
                        and set(a.get("parameter_counts", {})) == {"torch.bfloat16"}
                        and not any(
                            quant.get(k)
                            for k in [
                                "is_loaded_in_4bit",
                                "is_loaded_in_8bit",
                                "is_quantized",
                                "quantized_modules",
                                "config_quantization",
                            ]
                        ),
                    }
                )
    passed = (
        bool(image_checks)
        and bool(audit_checks)
        and all(x["pass"] for x in image_checks + audit_checks)
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "PNG_checks": image_checks,
        "model_audits": audit_checks,
    }
    (DEST / "integrity.json").write_text(json.dumps(result, indent=2) + "\n")
    if not passed:
        raise RuntimeError("Integrity validation failed; no success publication")


def evidence():
    sources = [
        p
        for p in OUT.rglob("*")
        if p.is_file()
        and p.suffix in {".json", ".jsonl", ".csv", ".log", ".md", ".txt", ".yaml", ".yml"}
        and not p.is_symlink()
    ]
    index = [
        {"path": str(p.relative_to(OUT)), "sha256": sha(p), "bytes": p.stat().st_size}
        for p in sorted(sources)
    ]
    (DEST / "evidence_files.json").write_text(json.dumps(index, indent=2) + "\n")
    with tarfile.open(DEST / "structured_evidence.tar.gz", "w:gz") as t:
        for p in sorted(sources):
            t.add(p, arcname=str(p.relative_to(OUT)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--watch", action="store_true")
    p.add_argument("--push", action="store_true")
    a = p.parse_args()
    while True:
        state = export()
        print(json.dumps(state), flush=True)
        cli_done = (OUT / "gpt61_low/complete.json").exists() and (
            OUT / "gpt61_low/ours_control_done.json"
        ).exists()
        terminal = state["local_queue_finished"] and (cli_done or state["GPT_blocked"])
        if not a.watch or terminal:
            break
        time.sleep(60)
    if terminal:
        validate_final()
        evidence()
    if a.push:
        rel = str(DEST.relative_to(ROOT))
        subprocess.run(["git", "add", "--", rel], cwd=ROOT, check=True)
        if subprocess.run(["git", "diff", "--cached", "--quiet", "--", rel], cwd=ROOT).returncode:
            subprocess.run(
                [
                    "git",
                    "commit",
                    "-m",
                    "report November21 new baselines GPT and distinct task trials",
                    "--",
                    rel,
                ],
                cwd=ROOT,
                check=True,
            )
        subprocess.run(
            [
                "git",
                "push",
                "git@github.com:pe0-p1e/exper-tran.git",
                "HEAD:experiment/kdd2027-baseline-reproduction",
            ],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
