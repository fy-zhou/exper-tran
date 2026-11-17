"""Wait for requested FOA+AnyAttack finish, publish evidence, launch no experiments."""

import csv
import json
import subprocess
import tarfile
import time
from pathlib import Path

import publish_foa_stage as stage

ROOT = Path(__file__).resolve().parents[2]
OUT = stage.OUT
REMAIN = OUT / "prepared_remaining"
ANY = REMAIN / "evaluation_anyattack"
DEST = ROOT / "docs/kdd2027/results/foa_anyattack_final_2026-11-17"
TARGETS = ["Qwen3.5-27B", "InternVL3.5-14B-HF", "Gemma 4 31B-it"]


def read(p, default=None):
    return json.loads(p.read_text()) if p.exists() else default


def requested_complete():
    rows = read(ANY / "task_summary.json", [])
    return (
        stage.finished()
        and len(rows) == 3
        and all(
            r["status"] == "COMPLETE"
            and r["planned_images"] == 150
            and r["returned_queries"] == 150
            for r in rows
        )
    )


def export():
    DEST.mkdir(parents=True, exist_ok=True)
    summaries = read(ANY / "task_summary.json", []) + read(stage.FOA / "task_summary.json", [])
    integrity = stage.evaluation_integrity(stage.FOA) and stage.evaluation_integrity(ANY)
    # Validate all frozen PNG identities and decoded pixel budgets in the final requested scope.
    import numpy as np
    from PIL import Image

    pngs = []
    for run in [stage.FOA, ANY]:
        for path in sorted(run.glob("attacks/*/receipt.json")):
            receipt = read(path)
            hashes = all(
                Path(p).exists() and stage.sha(Path(p)) == h
                for p, h in receipt["png_hashes"].items()
            )
            a = np.asarray(Image.open(receipt["adv"]).convert("RGB"), dtype=np.int16)
            b = np.asarray(Image.open(receipt["clean"]).convert("RGB"), dtype=np.int16)
            maximum = int(np.abs(a - b).max())
            pngs.append(
                {
                    "run": run.name,
                    "image_id": receipt["image_id"],
                    "hashes_pass": hashes,
                    "linf_uint8": maximum,
                    "no_target_access": not receipt.get("target_access", True),
                }
            )
    precision = []
    paths = [stage.FOA / "bf16_proxies.json", REMAIN / "generation_anyattack/bf16_audit.json"]
    paths += list((stage.FOA / "audits").glob("*.json")) + list((ANY / "audits").glob("*.json"))
    for path in paths:
        values = read(path, [])
        for value in values if isinstance(values, list) else [values]:
            q = value.get("quantization", {})
            precision.append(
                {
                    "source": str(path),
                    "model": value.get("model"),
                    "pass": value.get("status") == "PASS"
                    and set(value.get("parameter_counts", {})) == {"torch.bfloat16"}
                    and not any(
                        q.get(k)
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
    complete = (
        requested_complete()
        and integrity
        and len(pngs) == 300
        and all(p["hashes_pass"] and p["linf_uint8"] <= 16 and p["no_target_access"] for p in pngs)
        and bool(precision)
        and all(p["pass"] for p in precision)
    )
    (DEST / "scope_summaries.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (DEST / "PNG_integrity.json").write_text(json.dumps(pngs, indent=2) + "\n")
    (DEST / "BF16_integrity.json").write_text(json.dumps(precision, indent=2) + "\n")
    stop = read(REMAIN / "user_stop_v2.json", {})
    (DEST / "V2_user_pause.json").write_text(json.dumps(stop, indent=2) + "\n")
    comparison = []
    for f in (OUT / "runs").glob("final*/results.json"):
        for r in read(f, []):
            if r["transition"] != "ALL" or r["status"] != "COMPLETE" or r["method"] == "foa_attack":
                continue
            comparison.append(
                {
                    "method": r["method"],
                    "steps": 300 if "final150_300" in f.parent.name else 50,
                    "target": r["target"],
                    "hits": r["target_hits"],
                    "N": 150,
                    "ASR_percent": 100 * r["target_ASR"],
                }
            )
    for r in summaries:
        comparison.append(
            {
                "method": r["method"],
                "steps": r["steps"],
                "target": r["target"],
                "hits": r["target_hits"],
                "N": 150,
                "ASR_percent": 100 * r["ASR"] if r.get("ASR") is not None else None,
            }
        )
    ours = (
        OUT.parent / "a100_precommercial_v8_joint_selection24h/paper_closeout/reports/results.csv"
    )
    for r in csv.DictReader(ours.open()):
        if r["split"] == "final150" and r["model"] in TARGETS:
            comparison.append(
                {
                    "method": "Ours_" + r["label"],
                    "steps": 50,
                    "target": r["model"],
                    "hits": int(r["hits"]),
                    "N": int(r["N"]),
                    "ASR_percent": 100 * float(r["ASR"]),
                }
            )
    with (DEST / "comparison_percent.csv").open("w") as f:
        writer = csv.DictWriter(
            f, fieldnames=["method", "steps", "target", "hits", "N", "ASR_percent"]
        )
        writer.writeheader()
        writer.writerows(comparison)
    sources = [
        p
        for p in OUT.rglob("*")
        if p.is_file()
        and p.suffix in {".json", ".jsonl", ".csv", ".log", ".md", ".yaml", ".yml", ".txt"}
        and "stage_reports" not in p.parts
    ]
    index = [
        {"path": str(p.relative_to(OUT)), "bytes": p.stat().st_size, "sha256": stage.sha(p)}
        for p in sorted(sources)
    ]
    (DEST / "evidence_files.json").write_text(json.dumps(index, indent=2) + "\n")
    with tarfile.open(DEST / "structured_evidence.tar.gz", "w:gz") as t:
        for p in sorted(sources):
            t.add(p, arcname=str(p.relative_to(OUT)))
    archive = OUT.parent / "archives/kdd2027_FOA_AnyAttack_final_2026-11-17.tar.gz"
    archive.parent.mkdir(exist_ok=True)
    with tarfile.open(archive, "w:gz") as t:
        t.add(OUT, arcname=OUT.name)
    report = [
        "# Completed requested FOA + AnyAttack stage",
        "",
        f'UTC: {time.strftime("%FT%TZ",time.gmtime())}',
        "",
        f'Status: **{"COMPLETE_REQUESTED_SCOPE" if complete else "PARTIAL_OR_INVALID"}**.',
        "",
        "The requested scope is corrected FOA150 and AnyAttack150, evaluated "
        "on three large targets. "
        "V2 was paused by the user after8 frozen images; IAU was not started. "
        "This is not completion of the historical63-cell campaign. No further"
        " experiment is launched.",
        "",
        "|Method|Steps|Qwen27B|InternVL14B|Gemma31B|",
        "|---|---:|---:|---:|---:|",
    ]
    groups = {}
    for r in comparison:
        groups.setdefault((r["method"], r["steps"]), {})[r["target"]] = r["ASR_percent"]
    for (name, steps), scores in groups.items():
        cells = [f"{scores[t]:.2f}%" if scores.get(t) is not None else "N/A" for t in TARGETS]
        report.append("|" + name + "|" + str(steps) + "|" + "|".join(cells) + "|")
    report += [
        "",
        "All complete rates use all150 fixed ImageNet images. PGD/MI use a di"
        "sclosed target-image cosine adaptation; "
        "AnyAttack is a released COCO-trained generator and has extra trainin"
        "g resources. Native300 and matched50 budgets "
        "are distinct. Reference-image and proxy counts differ; this is not a"
        " fully compute/information-matched ranking. "
        "The new FOA fully validated run replaces old partial FOA in the main"
        " table; old evidence is preserved. "
        "Co/SGA remain supplemental/mismatched, not main targeted comparators.",
        "",
        "V2 partial generation is preserved without an ASR; the raw controlle"
        "r may record its terminated worker as failed, "
        "but user_stop_v2.json establishes USER_PAUSED, not a scientific fail"
        "ure. No V2 resume or target evaluation was requested.",
        "",
        "structured_evidence.tar.gz contains full raw answers, precision audi"
        "ts, model revisions, configs/source locks, "
        "loss traces, manifests, failures and intermediate exploratory eviden"
        "ce, including V2 partial receipts. "
        "PNG_integrity.json and BF16_integrity.json contain validation checks"
        ". Missing outcomes are not zeros.",
        "",
        f"Full images/checkpoints remain on persistent root: `{OUT}`. "
        f"Full local archive: `{archive}`. "
        "Canonical inputs in earlier persistent V8 directories remain require"
        "d. Model weights on /ephemeral are disposable.",
        "",
    ]
    (DEST / "REPORT.md").write_text("\n".join(report))
    state = {
        "status": "COMPLETE_REQUESTED_SCOPE" if complete else "PARTIAL_OR_INVALID",
        "V2_status": "PAUSED_BY_USER",
        "IAU_started": False,
        "no_more_experiments": True,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "archive": str(archive),
        "archive_sha256": stage.sha(archive),
    }
    (DEST / "snapshot.json").write_text(json.dumps(state, indent=2) + "\n")
    local = OUT / "stage_reports/final_requested"
    local.mkdir(parents=True, exist_ok=True)
    for name in ["snapshot.json", "REPORT.md"]:
        (local / name).write_bytes((DEST / name).read_bytes())
    return state


def main():
    while not (REMAIN / "final_status.json").exists():
        print(
            time.strftime("%FT%TZ", time.gmtime()), "WAITING_FOR_FOA_AND_ANYATTACK_ONLY", flush=True
        )
        time.sleep(30)
    time.sleep(3)
    result = export()
    print(json.dumps(result, indent=2), flush=True)
    rel = str(DEST.relative_to(ROOT))
    subprocess.run(["git", "add", "--", rel], cwd=ROOT, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet", "--", rel], cwd=ROOT).returncode:
        subprocess.run(
            [
                "git",
                "commit",
                "-m",
                "archive final requested FOA and AnyAttack results; V2 user-paused",
                "--",
                rel,
            ],
            cwd=ROOT,
            check=True,
        )
    for _ in range(3):
        if (
            subprocess.run(
                [
                    "git",
                    "push",
                    "git@github.com:pe0-p1e/exper-tran.git",
                    "HEAD:experiment/kdd2027-baseline-reproduction",
                ],
                cwd=ROOT,
            ).returncode
            == 0
        ):
            break
        time.sleep(30)
    else:
        raise SystemExit("UPLOAD_FAILED; local results and commit preserved")
    return 0 if result["status"] == "COMPLETE_REQUESTED_SCOPE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
