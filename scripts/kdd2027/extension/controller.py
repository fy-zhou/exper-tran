"""One owner for the 7x3x3 campaign; wait for the older owner, never steal GPU.

Eight hours is a MINIMUM requested work window, not a deadline, guarantee or a
reason to idle/repeat valid jobs. All genuine remaining jobs stay resumable.
"""

import argparse
import csv
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from metrics import summarize
from task_protocol import DATASETS, METHODS, TARGETS, build_manifests, condition
from worker import OUT, lock_job

from primary_ml_cka.baselines import campaign as legacy
from primary_ml_cka.baselines.protocol import atomic, freeze, sha

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
STATE = OUT / "extension"


def jobs():
    # Smoke every newly integrated core before its full-condition admission.
    smoke = [
        ("co_attack", "imagenet", "default"),
        ("sga", "imagenet", "default"),
        ("foa_attack", "imagenet", "default"),
        ("iau_foa", "imagenet", "default"),
        ("iau_foa", "imagenet", "family_disjoint4"),
        ("pgd", "coco_captioning", "default"),
        ("pgd", "llava_bench_coco", "default"),
    ]
    result = [dict(method=m, dataset=d, variant=v, smoke=True, pilot=False) for m, d, v in smoke]
    result += [
        dict(method="iau_foa", dataset="imagenet", variant=v, smoke=False, pilot=True)
        for v in ["default", "family_disjoint4"]
    ]
    # Reuse existing classification controls. Then cover both new tasks using
    # the cheaper controls before the expensive native300 clustered methods.
    for d in DATASETS:
        for m in METHODS:
            if m in ["iau_foa", "foa_attack"]:
                continue
            result.append(dict(method=m, dataset=d, variant="default", smoke=False, pilot=False))
    for d in DATASETS:
        result.append(
            dict(method="foa_attack", dataset=d, variant="default", smoke=False, pilot=False)
        )
    for d in DATASETS:
        for v in ["default", "family_disjoint4"]:
            result.append(dict(method="iau_foa", dataset=d, variant=v, smoke=False, pilot=False))
    return result


def status():
    plan_path = STATE / "queue.json"
    if not plan_path.exists():
        print("NOT_PREPARED")
        return
    plan = json.loads(plan_path.read_text())
    reports = []
    current_sources = {p.name: sha(p) for p in HERE.glob("*.py")}
    current_sources["legacy"] = legacy.source_hashes()
    for p in (STATE / "runs").glob("*/task_summary.json"):
        if json.loads((p.parent / "lock.json").read_text())["sources"] == current_sources:
            reports.extend(json.loads(p.read_text()))
    # Main table excludes all smoke/pilot rows, and includes all63 slots.
    rows = []
    for d in DATASETS:
        for m in METHODS:
            for target in TARGETS:
                matches = [
                    r
                    for r in reports
                    if r["method"] == m
                    and r["dataset"] == d
                    and r["target"] == target
                    and not r.get("smoke_only")
                    and not r.get("pilot")
                ]
                rows.append(
                    {
                        "dataset": d,
                        "method": m,
                        "target": target,
                        "status": matches[-1]["status"] if matches else "QUEUED",
                    }
                )
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "completion_matrix.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "method", "target", "status"])
        writer.writeheader()
        writer.writerows(rows)
    counts = {s: sum(r["status"] == s for r in rows) for s in {r["status"] for r in rows}}
    heartbeat = {
        "time": time.time(),
        "scope": "7_methods_x_3_tasks_x_3_targets",
        "conditions": 63,
        "counts": counts,
        "minimum_requested_hours": plan["minimum_hours"],
        "honest_judge_pending_not_complete": True,
    }
    atomic(STATE / "status.json", heartbeat)
    print(json.dumps(heartbeat, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["prepare", "start", "status", "report"])
    p.add_argument("--minimum-hours", type=float, default=8)
    args = p.parse_args()
    if args.mode in ["status", "report"]:
        status()
        return 0
    if args.minimum_hours < 8:
        p.error("This campaign requests at least an8hour work window")
    build_manifests(OUT, Path("/ephemeral/kdd2027/datasets"))
    plan = {
        "version": 1,
        "minimum_hours": args.minimum_hours,
        "run_until_queue_complete": True,
        "no_idle_to_fill_time": True,
        "datasets": DATASETS,
        "methods": METHODS,
        "targets": TARGETS,
        "conditions": 63,
        "jobs": jobs(),
        "commercial_judge_calls": False,
        "LLaVA_native_questions": 90,
        "COCO_frozen_images": 150,
        "ImageNet_frozen_images": 150,
    }
    freeze(STATE / "queue.json", plan)
    if args.mode == "prepare":
        status()
        return 0
    # Reuse the SAME outer owner and inner admission lock as existing jobs.
    print("WAITING_FOR_EXISTING_OWNER", flush=True)
    with (OUT / "reproduction_owner.lock").open("a") as outer:
        fcntl.flock(outer, fcntl.LOCK_EX)
        with (STATE / "controller.lock").open("a") as controller:
            fcntl.flock(controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
            window = STATE / "window.json"
            if not window.exists():
                freeze(
                    window,
                    {
                        "start": time.time(),
                        "minimum_hours": args.minimum_hours,
                        "stop_at_8h": False,
                        "finish_all_available_work": True,
                    },
                )
            failed_cores = set()
            for index, job in enumerate(plan["jobs"]):
                key = (job["method"], job["variant"])
                if key in failed_cores:
                    atomic(
                        STATE / "jobs" / f"{index:03d}.json",
                        {"job": job, "status": "BLOCKED_FAILED_SMOKE"},
                    )
                    continue
                config = condition(job["method"], job["dataset"], job["variant"], job["smoke"])
                config["pilot"] = job["pilot"]
                run, _, _ = lock_job(config)
                command = [
                    sys.executable,
                    str(HERE / "worker.py"),
                    "--method",
                    job["method"],
                    "--dataset",
                    job["dataset"],
                    "--variant",
                    job["variant"],
                ]
                if job["smoke"]:
                    command += ["--smoke"]
                if job["pilot"]:
                    command += ["--pilot"]
                log = STATE / "logs" / f"{index:03d}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                began = time.time()
                with log.open("a") as stream:
                    for phase in ["attack", "evaluate"]:
                        # Fresh evaluator cannot inherit resize monkeypatches,
                        # proxy weights or deterministic-kernel settings.
                        child = subprocess.Popen(
                            command + ["--phase", phase],
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
                        )
                        while child.poll() is None:
                            atomic(
                                STATE / "heartbeat.json",
                                {
                                    "time": time.time(),
                                    "job": job,
                                    "index": index,
                                    "phase": phase,
                                    "planned_jobs": len(plan["jobs"]),
                                    "pid": child.pid,
                                    "run": str(run),
                                    "seconds": time.time() - began,
                                    "minimum_hours": args.minimum_hours,
                                },
                            )
                            try:
                                child.wait(timeout=20)
                            except subprocess.TimeoutExpired:
                                pass
                        if child.returncode:
                            break
                if child.returncode and job["smoke"]:
                    failed_cores.add(key)
                try:
                    summarize(run, OUT)
                except Exception as e:
                    atomic(run / "metric_failure.json", {"error": repr(e)})
                atomic(
                    STATE / "jobs" / f"{index:03d}.json",
                    {
                        "job": job,
                        "exit": child.returncode,
                        "run": str(run),
                        "seconds": time.time() - began,
                        "status": "FINISHED_REQUIRES_REPORT_AUDIT"
                        if not child.returncode
                        else "FAILED",
                    },
                )
                status()
            status()
    final = json.loads((STATE / "status.json").read_text())
    return 0 if final["counts"].get("COMPLETE", 0) == 63 else 2


if __name__ == "__main__":
    raise SystemExit(main())
