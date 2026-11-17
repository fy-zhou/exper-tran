"""CPU-only FOA completion watcher, scientific evidence exporter and optional git push."""

import argparse
import csv
import fcntl
import hashlib
import json
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("/home/ubuntu/exper-tran/outputs/kdd2027_baseline_reproduction")
FOA = OUT / "extension/runs/imagenet_foa_attack_29c5c7cef8fc733e71"
CO = OUT / "extension/runs/imagenet_co_attack_16f5fdb8880fdfa9e7"
DEST = ROOT / "docs/kdd2027/results/foa_stage_2026-11-17"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def finished():
    rows = read(FOA / "task_summary.json", [])
    return len(rows) == 3 and all(
        x["status"] == "COMPLETE" and x["planned_images"] == 150 and x["returned_queries"] == 150
        for x in rows
    )


def evaluation_integrity(run):
    receipts = {read(p)["image_id"]: p for p in run.glob("attacks/*/receipt.json")}
    folders = list((run / "evaluations").glob("*"))
    if len(receipts) != 150 or len(folders) != 3:
        return False
    for folder in folders:
        rows = [read(p) for p in folder.glob("*.json")]
        if len(rows) != 150 or {r["image_id"] for r in rows} != set(receipts):
            return False
        for r in rows:
            if r["receipt_sha256"] != sha(receipts[r["image_id"]]):
                return False
            if r["target_hit"] != (r["adv"].get("label") == r["target_label"]):
                return False
    return True


def precision_integrity():
    audits = []
    for run in [FOA, CO]:
        paths = list(run.glob("audits/*.json")) + [run / "bf16_proxies.json"]
        for path in paths:
            if not path.exists():
                continue
            values = read(path)
            for audit in values if isinstance(values, list) else [values]:
                counts = audit.get("parameter_counts", {})
                quant = audit.get("quantization", {})
                audits.append(
                    audit.get("status") == "PASS"
                    and set(counts) == {"torch.bfloat16"}
                    and not any(
                        quant.get(k)
                        for k in [
                            "is_loaded_in_4bit",
                            "is_loaded_in_8bit",
                            "is_quantized",
                            "quantized_modules",
                            "config_quantization",
                        ]
                    )
                )
    return bool(audits) and all(audits)


def export():
    import numpy as np
    from PIL import Image

    DEST.mkdir(parents=True, exist_ok=True)
    checks = []
    for run in [FOA, CO]:
        for f in sorted(run.glob("attacks/*/receipt.json")):
            r = read(f)
            hashes = all(
                Path(p).is_file() and sha(Path(p)) == h for p, h in r["png_hashes"].items()
            )
            a = np.asarray(Image.open(r["adv"]).convert("RGB"), dtype=np.int16)
            b = np.asarray(Image.open(r["clean"]).convert("RGB"), dtype=np.int16)
            linf = int(np.abs(a - b).max())
            checks.append(
                dict(
                    run=run.name,
                    image_id=r["image_id"],
                    hashes_pass=hashes,
                    linf_uint8=linf,
                    no_target_access=r.get("target_access") is False,
                    steps=r["steps"],
                    receipt_sha256=sha(f),
                )
            )
    integrity = all(
        x["hashes_pass"] and x["linf_uint8"] <= 16 and x["no_target_access"] for x in checks
    )
    summaries = {r.name: read(r / "task_summary.json", []) for r in [FOA, CO]}
    allrows = []
    for f in (OUT / "runs").glob("final*/results.json"):
        for r in read(f, []):
            allrows.append(dict(run=f.parent.name, **r))
    for _run, rows in summaries.items():
        for r in rows:
            allrows.append(dict(run=_run, **r))
    comparison = []
    for row in allrows:
        if row.get("transition", "ALL") != "ALL":
            continue
        if row["method"] == "foa_attack" and not row["run"].startswith("imagenet_"):
            continue
        score = row.get("ASR", row.get("target_ASR"))
        steps = row.get("steps", 300 if "final150_300" in row["run"] else 50)
        comparison.append(
            dict(
                method=row["method"],
                steps=steps,
                target=row["target"],
                hits=row.get("target_hits"),
                N=150,
                ASR_percent=100 * score
                if score is not None and row["status"] == "COMPLETE"
                else None,
                status=row["status"],
                track="supplemental" if row["method"] == "co_attack" else "primary",
                source=row["run"],
            )
        )
    ours = (
        OUT.parent / "a100_precommercial_v8_joint_selection24h/paper_closeout/reports/results.csv"
    )
    if ours.exists():
        shutil.copyfile(ours, DEST / "ours_preserved_results.csv")
        for row in csv.DictReader(ours.open()):
            if row["split"] != "final150" or row["model"] not in [
                "Qwen3.5-27B",
                "InternVL3.5-14B-HF",
                "Gemma 4 31B-it",
            ]:
                continue
            comparison.append(
                dict(
                    method="Ours_" + row["label"],
                    steps=50,
                    target=row["model"],
                    hits=int(row["hits"]),
                    N=int(row["N"]),
                    ASR_percent=100 * float(row["ASR"]),
                    status="COMPLETE",
                    track="primary",
                    source=str(ours),
                )
            )
    with (DEST / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "method",
                "steps",
                "target",
                "hits",
                "N",
                "ASR_percent",
                "status",
                "track",
                "source",
            ],
        )
        writer.writeheader()
        writer.writerows(comparison)
    (DEST / "all_results.json").write_text(json.dumps(allrows, indent=2) + "\n")
    (DEST / "integrity.json").write_text(json.dumps(checks, indent=2) + "\n")
    for run in [FOA, CO]:
        for name in ["lock.json", "task_summary.json", "bf16_proxies.json"]:
            if (run / name).exists():
                d = DEST / run.name
                d.mkdir(exist_ok=True)
                shutil.copyfile(run / name, d / name)
    sources = [
        p
        for p in OUT.rglob("*")
        if p.is_file()
        and p.suffix in {".json", ".jsonl", ".csv", ".log", ".txt", ".md", ".yaml", ".yml"}
        and "stage_reports" not in p.parts
        and "prepared_remaining" not in p.parts
    ]
    file_index = [
        dict(path=str(p.relative_to(OUT)), bytes=p.stat().st_size, sha256=sha(p))
        for p in sorted(sources)
    ]
    (DEST / "evidence_files.json").write_text(json.dumps(file_index, indent=2) + "\n")
    with tarfile.open(DEST / "structured_evidence.tar.gz", "w:gz") as t:
        for p in sorted(sources):
            t.add(p, arcname=str(p.relative_to(OUT)))
    # Snapshot all local output files including every new PNG; keep it on persistent root.
    backup = OUT.parent / "archives/kdd2027_baselines_foa_stage_2026-11-17.tar.gz"
    backup.parent.mkdir(exist_ok=True)
    with tarfile.open(backup, "w:gz") as t:
        t.add(
            OUT,
            arcname=OUT.name,
            filter=lambda member: None
            if "prepared_remaining" in Path(member.name).parts
            else member,
        )
    complete = (
        finished()
        and integrity
        and len(checks) == 300
        and evaluation_integrity(FOA)
        and evaluation_integrity(CO)
        and precision_integrity()
    )
    state = "COMPLETE_FOA_AND_CO" if complete else "PARTIAL_OR_INVALID"
    report = [
        "# ImageNet baseline stage results",
        "",
        f'UTC cutoff: {time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}',
        "",
        (
            f"Status: **{state}**. This completes only the requested ImageNet Co/FOA stage, "
            "not all seven methods or all three datasets."
        ),
        "",
        "|Method|Steps|Target|Hits / planned N|ASR|Status|",
        "|---|---:|---|---:|---:|---|",
    ]
    for _run, rows in summaries.items():
        for r in rows:
            score = f"{100*r['ASR']:.2f}%" if r.get("ASR") is not None else "N/A"
            report.append(
                f"|{r['method']}|{r['steps']}|{r['target']}|"
                f"{r.get('target_hits','?')}/{r['planned_images']}|{score}|{r['status']}|"
            )
    report += [
        "",
        (
            "Corrected FOA replaces old partial FOA in the main comparison only "
            "if all150 attacks and all450 target rows validate. Original results "
            "remain archived; revised and old attacks are never mixed."
        ),
        "",
        (
            "Co-Attack is supplemental image-only transfer from a joint "
            "untargeted retrieval method, excluded from the primary targeted "
            "comparison for threat-model mismatch, irrespective of its score. "
            "SGA is similarly excluded; neither is silently replaced."
        ),
        "",
        (
            f"PNG/receipt checks: {len(checks)}; integrity PASS={integrity}. "
            "Full raw replies, losses, configuration/source locks, revisions, audits, failures "
            "and previous exploratory records are in structured_evidence.tar.gz. "
            "No new GPU inference runs during reporting."
        ),
        "",
        (
            "IAU remains pilot-only (Qwen10/10, InternVL10/10, Gemma8/10); "
            "V2/AnyAttack have separate authorized runs, excluded here. "
            "Missing scores remain N/A. Ours "
            "and old published artifacts are unchanged."
        ),
        "",
        (
            f"Persistent live output: `{OUT}`. Full local image/checkpoint archive: `{backup}`. "
            "Canonical inputs in persistent V8 directories remain required. "
            "Model caches on /ephemeral are not backups."
        ),
        "",
        (
            "The broad63-cell queue is paused. This watcher launches no experiment. "
            "Separately authorized live V2/AnyAttack files are excluded from this stable archive."
        ),
        "",
    ]
    (DEST / "REPORT.md").write_text("\n".join(report))
    result = {
        "status": state,
        "integrity_pass": integrity,
        "FOA_full150_complete": complete,
        "bf16_audits_pass": precision_integrity(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "local_archive": str(backup),
        "local_archive_sha256": sha(backup),
        "local_archive_bytes": backup.stat().st_size,
        "structured_archive_sha256": sha(DEST / "structured_evidence.tar.gz"),
    }
    (DEST / "snapshot.json").write_text(json.dumps(result, indent=2) + "\n")
    local = OUT / "stage_reports/foa_stage_2026-11-17"
    local.mkdir(parents=True, exist_ok=True)
    for name in ["REPORT.md", "snapshot.json"]:
        shutil.copyfile(DEST / name, local / name)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wait", action="store_true")
    p.add_argument("--push", action="store_true")
    args = p.parse_args()
    owner = (OUT / "extension/foa_publication.lock").open("a")
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.wait:
        log = OUT / "extension/finish_foa_co.log"
        while not finished():
            text = log.read_text() if log.exists() else ""
            if "METHOD_FINISHED foa_attack" in text:
                break
            print(
                time.strftime("%FT%TZ", time.gmtime()),
                "WAITING_FOR_FOA_ATTACK_AND_THREE_TARGET_EVALUATIONS",
                flush=True,
            )
            time.sleep(30)
    elif not finished():
        raise SystemExit(
            "FOA still running: use --wait. No incomplete snapshot published accidentally."
        )
    if args.wait:
        time.sleep(3)  # Allow terminal runner status/log writes to settle.
    result = export()
    print(json.dumps(result, indent=2), flush=True)
    if args.push:
        # Scope staging to this report directory only. Never stage auth/caches or active source.
        rel = str(DEST.relative_to(ROOT))
        subprocess.run(["git", "add", "--", rel], cwd=ROOT, check=True)
        if subprocess.run(["git", "diff", "--cached", "--quiet", "--", rel], cwd=ROOT).returncode:
            subprocess.run(
                [
                    "git",
                    "commit",
                    "-m",
                    "report completed FOA stage with full structured evidence",
                    "--",
                    rel,
                ],
                cwd=ROOT,
                check=True,
            )
        for _attempt in range(3):
            pushed = subprocess.run(
                [
                    "git",
                    "push",
                    "git@github.com:pe0-p1e/exper-tran.git",
                    "HEAD:experiment/kdd2027-baseline-reproduction",
                ],
                cwd=ROOT,
            )
            if pushed.returncode == 0:
                break
            time.sleep(30)
        else:
            raise SystemExit(
                "UPLOAD_FAILED: local reports/commit preserved; no force push attempted"
            )
    return 0 if result["status"] == "COMPLETE_FOA_AND_CO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
