"""Non-commercial task expansion; one existing GPU owner and immutable receipts."""

import argparse
import copy
import fcntl
import importlib.util
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = Path("/home/ubuntu/exper-tran/outputs/kdd2027_baseline_reproduction")
OUT = BASE / "nov21_dataset_expansion"
DEST = HERE.parents[2] / "docs/kdd2027/results/nov21_dataset_expansion"


def api():
    spec = importlib.util.spec_from_file_location("dataset_base_worker", HERE / "worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(w, method, dataset, n, steps):
    _, _, original, manifest = w.setup(method, dataset, n, False)
    lock = copy.deepcopy(original)
    config = lock["config"]
    config.update(steps=steps, study="non_gpt_task_expansion_v1")
    config["native_budget_or_adaptation"] = (
        "matched50_task_trial_not_native300" if steps == 50 else "native300_task_extension"
    )
    lock["expansion_source_sha256"] = w.sha(Path(__file__))
    run = OUT / "runs" / f"{dataset}_{method}_{n}_{steps}_{w.digest(lock)[:14]}"
    w.freeze(run / "lock.json", lock)
    w.freeze(run / "analysis_input/extension/manifests" / f"{dataset}.json", manifest)
    return run, config, lock, manifest


def compatible_lock(old, new):
    """N/study are bookkeeping; every scientific input/settings/source must agree."""
    a, b = copy.deepcopy(old), copy.deepcopy(new)
    for value in [a, b]:
        value.pop("expansion_source_sha256", None)
        value.pop("manifest", None)
        for key in ["N", "study"]:
            value["config"].pop(key, None)
    return a == b


def reuse(w, run, lock, manifest):
    rows = {r["image_id"]: r for r in manifest["images"]}
    imported = 0
    for oldfile in sorted((BASE / "nov21/local").glob("*/lock.json")):
        old = json.loads(oldfile.read_text())
        if not compatible_lock(old, lock):
            continue
        oldrun = oldfile.parent
        for row in old["manifest"]["images"]:
            if rows.get(row["image_id"]) != row:
                continue
            source = w.oldworker.receipt_path(oldrun, row)
            dest = w.oldworker.receipt_path(run, row)
            if not w.valid_receipt(source, w.oldworker.sig(old, row)):
                continue
            if not w.valid_receipt(dest, w.oldworker.sig(lock, row)):
                value = json.loads(source.read_text())
                value.update(
                    signature=w.oldworker.sig(lock, row),
                    reuse_receipt=str(source),
                    reuse_receipt_sha256=w.sha(source),
                )
                w.oldworker.atomic(dest, value)
                imported += 1
            for name, spec in w.oldworker.REG.items():
                for q in row["queries"]:
                    filename = w.digest([row["image_id"], q["id"]])[:18] + ".json"
                    source_eval = oldrun / "evaluations" / name / filename
                    if not w.oldworker.valid_evaluation(source_eval, oldrun, row, q, spec):
                        continue
                    dest_eval = run / "evaluations" / name / filename
                    if w.oldworker.valid_evaluation(dest_eval, run, row, q, spec):
                        continue
                    value = json.loads(source_eval.read_text())
                    value.update(
                        receipt_sha256=w.sha(dest),
                        reused_evaluation=str(source_eval),
                        reused_evaluation_sha256=w.sha(source_eval),
                    )
                    w.oldworker.atomic(dest_eval, value)
    print("REUSED_ATTACKS", imported, str(run), flush=True)


def phase(args):
    w = api()
    run, config, lock, manifest = prepare(w, args.method, args.dataset, args.N, args.steps)
    reuse(w, run, lock, manifest)
    if args.phase == "report":
        from metrics import summarize

        summarize(run, run / "analysis_input")
        return
    with w.legacy.gpu_owner(BASE):
        if args.phase == "attack":
            w.oldworker.attacks(config, run, lock, manifest)
        else:
            w.oldworker.evaluations(config, run, lock, manifest)


def report():
    OUT.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    summaries = []
    for path in sorted((OUT / "runs").glob("*/task_summary.json")):
        summaries.extend({"run": path.parent.name, **r} for r in json.loads(path.read_text()))
    (DEST / "results.json").write_text(json.dumps(summaries, indent=2) + "\n")
    state = json.loads((OUT / "state.json").read_text()) if (OUT / "state.json").exists() else {}
    (DEST / "state.json").write_text(json.dumps(state, indent=2) + "\n")
    lines = [
        "# Non-GPT COCO/LLaVA expansion",
        "",
        "Status: " + state.get("status", "PREPARED"),
        "",
        "COCO150 captioning and original LLaVA30/90 questions; all three fixed large targets.",
        "Matched50 trials are not native300 reproductions. "
        "No commercial calls or target-feedback tuning.",
        "LLaVA ROUGE-L is diagnostic only; official GPT judge remains pending.",
        "",
        "|Dataset|Method|Steps|Target|Returned queries|Status|Clean metric|Adv metric|",
        "|---|---|---:|---|---:|---|---:|---:|",
    ]
    for r in summaries:
        metric = "CIDEr_valid_subset" if r["dataset"] == "coco_captioning" else "ROUGE_L_proxy"
        lines.append(
            f'|{r["dataset"]}|{r["method"]}|{r["steps"]}|{r["target"]}|'
            f'{r["returned_queries"]}/{r["planned_queries"]}|{r["status"]}|'
            f'{r.get("clean_"+metric, "N/A")}|{r.get("adv_"+metric, "N/A")}|'
        )
    (DEST / "REPORT.md").write_text("\n".join(lines) + "\n")


def finalize(w):
    """Artifact integrity and raw-query completeness, separate from judge availability."""
    import numpy as np
    from PIL import Image

    checks = []
    for path in sorted((OUT / "runs").glob("*/lock.json")):
        lock = json.loads(path.read_text())
        run = path.parent
        for row in lock["manifest"]["images"]:
            receipt = w.oldworker.receipt_path(run, row)
            valid = w.valid_receipt(receipt, w.oldworker.sig(lock, row))
            bound = None
            if valid:
                data = json.loads(receipt.read_text())
                clean = np.asarray(Image.open(data["clean"]).convert("RGB"), dtype=np.int16)
                adv = np.asarray(Image.open(data["adv"]).convert("RGB"), dtype=np.int16)
                bound = int(np.abs(adv - clean).max())
                valid = bound <= 16 and data["steps"] == lock["config"]["steps"]
            checks.append(
                {
                    "run": run.name,
                    "image_id": row["image_id"],
                    "valid_attack": valid,
                    "linf_uint8": bound,
                }
            )
    results = json.loads((DEST / "results.json").read_text())
    core = [r for r in results if r["method"] in ["pgd", "m_attack", "foa_attack"]]
    complete = len(core) == 18 and all(r["raw_complete"] for r in core)
    audit = {
        "all_present_attack_artifacts_valid": all(r["valid_attack"] for r in checks),
        "core_18_raw_target_cells_complete": complete,
        "official_llava_judge": "NOT_CALLED_NON_GPT_CAMPAIGN",
        "images": checks,
    }
    (DEST / "integrity.json").write_text(json.dumps(audit, indent=2) + "\n")
    sources = [
        p
        for p in OUT.rglob("*")
        if p.is_file()
        and not p.is_symlink()
        and p.suffix in [".json", ".jsonl", ".csv", ".log", ".txt"]
    ]
    (DEST / "evidence_files.json").write_text(
        json.dumps(
            [{"path": str(p.relative_to(OUT)), "sha256": w.sha(p)} for p in sorted(sources)],
            indent=2,
        )
        + "\n"
    )
    with tarfile.open(DEST / "structured_evidence.tar.gz", "w:gz") as archive:
        for p in sorted(sources):
            archive.add(p, arcname=str(p.relative_to(OUT)))
    rel = str(DEST.relative_to(HERE.parents[2]))
    subprocess.run(["git", "add", "--", rel], cwd=HERE.parents[2], check=True)
    if subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", rel], cwd=HERE.parents[2]
    ).returncode:
        subprocess.run(
            ["git", "commit", "-m", "report non-GPT COCO LLaVA task expansion", "--", rel],
            cwd=HERE.parents[2],
            check=True,
        )
    subprocess.run(
        [
            "git",
            "push",
            "git@github.com:pe0-p1e/exper-tran.git",
            "HEAD:experiment/kdd2027-baseline-reproduction",
        ],
        cwd=HERE.parents[2],
        check=True,
    )


def campaign(args):
    w = api()
    OUT.mkdir(parents=True, exist_ok=True)
    statepath = OUT / "state.json"
    if statepath.exists():
        state = json.loads(statepath.read_text())
    else:
        state = {
            "status": "WAITING_EXISTING_GPU_OWNER",
            "start": time.time(),
            "deadline": time.time() + args.hours * 3600,
            "jobs": [],
            "commercial_calls": False,
            "planned_hours": args.hours,
        }

    def save():
        w.oldworker.atomic(statepath, state)
        report()

    save()
    jobs = [
        (m, d, n, 50, True)
        for d, n in [("coco_captioning", 150), ("llava_bench_coco", 30)]
        for m in ["pgd", "m_attack", "foa_attack"]
    ]
    # Scientifically useful additions only if the measured budget permits.
    jobs += [
        ("mi_fgsm", "coco_captioning", 150, 50, False),
        ("mi_fgsm", "llava_bench_coco", 30, 50, False),
    ]
    env = {
        **os.environ,
        "PYTHONPATH": str(HERE.parents[2] / "src"),
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "2",
        "PRIMARY_ML_CKA_STRICT_BF16": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    # Same owner as the active ImageNet queue. Never launch a competing controller.
    with (BASE / "reproduction_owner.lock").open("a") as owner:
        while True:
            try:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() >= state["deadline"]:
                    state["status"] = "PARTIAL_WINDOW_ENDED_WAITING_OWNER"
                    save()
                    return
                time.sleep(20)
        for method, dataset, n, steps, required in jobs:
            run, config, lock, manifest = prepare(w, method, dataset, n, steps)
            # Conservative admission estimate; do not shorten questions, steps or N.
            predicted = (55 if dataset == "coco_captioning" else 100) * 60
            if not required and time.time() + predicted + 600 > state["deadline"]:
                state["jobs"].append(
                    {"method": method, "dataset": dataset, "status": "QUEUED_BUDGET"}
                )
                save()
                continue
            if time.time() >= state["deadline"]:
                state["status"] = "PARTIAL_RESUMABLE_WINDOW_ENDED"
                save()
                finalize(w)
                return
            for step in ["attack", "evaluate", "report"]:
                state.update(
                    status="RUNNING",
                    current={
                        "method": method,
                        "dataset": dataset,
                        "N": n,
                        "steps": steps,
                        "phase": step,
                        "run": str(run),
                    },
                )
                save()
                cmd = [
                    sys.executable,
                    str(Path(__file__)),
                    "phase",
                    "--method",
                    method,
                    "--dataset",
                    dataset,
                    "--N",
                    str(n),
                    "--steps",
                    str(steps),
                    "--phase",
                    step,
                ]
                with (OUT / f"{dataset}_{method}_{steps}_{step}.log").open("a") as f:
                    child = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
                    while child.poll() is None:
                        state.update(heartbeat=time.time(), pid=child.pid)
                        save()
                        time.sleep(20)
                if child.returncode:
                    state["jobs"].append(
                        {
                            "method": method,
                            "dataset": dataset,
                            "phase": step,
                            "status": "FAILED",
                            "exit": child.returncode,
                        }
                    )
                    save()
                    break
            else:
                state["jobs"].append(
                    {"method": method, "dataset": dataset, "status": "ATTEMPTED_SEE_COMPLETENESS"}
                )
                save()
        state["status"] = "FINISHED_SEE_COMPLETENESS"
        state["finished"] = time.time()
        save()
        finalize(w)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "phase", "report"])
    parser.add_argument("--hours", type=float, default=10)
    parser.add_argument("--method", choices=["pgd", "mi_fgsm", "m_attack", "foa_attack"])
    parser.add_argument("--dataset", choices=["coco_captioning", "llava_bench_coco"])
    parser.add_argument("--N", type=int)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--phase", choices=["attack", "evaluate", "report"])
    args = parser.parse_args()
    if args.command == "phase":
        phase(args)
    elif args.command == "run":
        campaign(args)
    else:
        report()
