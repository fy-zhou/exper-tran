"""Ours-first native-task pipeline, then fair baseline controls. No commercial calls."""

import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from prepare import BASE, OUT, REPO

from primary_ml_cka.baselines.protocol import atomic

HERE = Path(__file__).resolve().parent
ENV = {
    **os.environ,
    "PYTHONPATH": str(REPO / "src"),
    "PRIMARY_ML_CKA_STRICT_BF16": "1",
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "2",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


def execute(script, args, label, state):
    state.update(stage=label, heartbeat=time.time())
    atomic(OUT / "state.json", state)
    with (OUT / (label + ".log")).open("a") as f:
        child = subprocess.Popen(
            [sys.executable, str(HERE / script), *args], env=ENV, stdout=f, stderr=subprocess.STDOUT
        )
        while child.poll() is None:
            state.update(pid=child.pid, heartbeat=time.time())
            atomic(OUT / "state.json", state)
            time.sleep(15)
    if child.returncode:
        state.update(status="BLOCKED", failed_stage=label, exit_code=child.returncode)
        atomic(OUT / "state.json", state)
        return False
    return True


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    budget = json.loads((BASE / "nov21_dataset_expansion/state.json").read_text())
    state = {
        "status": "RUNNING",
        "started": time.time(),
        "deadline": budget["deadline"],
        "commercial_calls": False,
        "priority": "Ours native-task validity before baselines",
    }
    if not execute("prepare.py", [], "prepare_references", state):
        return 2
    with (BASE / "reproduction_owner.lock").open("a") as owner:
        state["stage"] = "WAITING_EXISTING_IMAGENET_OWNER"
        atomic(OUT / "state.json", state)
        fcntl.flock(owner, fcntl.LOCK_EX)
        if not execute("prepare.py", ["select"], "freeze_reference_banks", state):
            return 2
        for family in ["qwen", "internvl", "gemma4"]:
            if not execute(
                "worker.py",
                [
                    "attack",
                    "--dataset",
                    "coco_captioning",
                    "--family",
                    family,
                    "--N",
                    "2",
                    "--engineering",
                ],
                f"engineering_{family}",
                state,
            ):
                return 2
        # Engineering-only checks precede any full target run; no selection by target scores.
        for cohort in ["pilot", "full"]:
            for dataset, n in [
                ("coco_captioning", 10 if cohort == "pilot" else 150),
                ("llava_bench_coco", 3 if cohort == "pilot" else 30),
            ]:
                for family in ["qwen", "internvl", "gemma4"]:
                    if time.time() >= state["deadline"]:
                        state["status"] = "PARTIAL_RESUMABLE_WINDOW_ENDED"
                        atomic(OUT / "state.json", state)
                        return 2
                    args = ["--dataset", dataset, "--family", family, "--N", str(n)]
                    for phase in ["attack", "evaluate", "report"]:
                        label = f"{cohort}_{dataset}_{family}_{phase}"
                        if not execute("worker.py", [phase, *args], label, state):
                            return 2
                    # Resolve the immutable run without importing a same-name old worker.
                    import importlib.util

                    spec = importlib.util.spec_from_file_location(
                        "ours_task_worker", HERE / "worker.py"
                    )
                    worker = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(worker)
                    run, _, _, _ = worker.setup(dataset, family, n)
                    if dataset == "llava_bench_coco":
                        if not execute(
                            "judge.py", ["--run", str(run)], f"{cohort}_{family}_judge", state
                        ):
                            return 2
                    if not execute("report.py", [], f"{cohort}_{dataset}_{family}_quality", state):
                        return 2
        results = json.loads(
            (REPO / "docs/kdd2027/results/native_tasks_ours_first/results.json").read_text()
        )
        full = [r for r in results if r["N_images"] in [150, 30]]
        if len(full) != 6 or not all(r.get("primary_metric_valid") for r in full):
            state.update(status="PARTIAL_MEASUREMENT_FAILURE", baseline_queue="NOT_STARTED")
            atomic(OUT / "state.json", state)
            return 2
        state.update(status="OURS_RAW_TASKS_AND_ADAPTED_METRICS_COMPLETE", completed=time.time())
        atomic(OUT / "state.json", state)
        if not execute("asr.py", ["--ours-only"], "ours_binary_semantic_ASR", state):
            return 2
    # Same controller and locks, existing baseline phases, scoring after each condition.
    from prepare import expansion

    jobs = [
        (m, d, n, 50)
        for m in ["pgd", "m_attack", "foa_attack", "mi_fgsm"]
        for d, n in [("coco_captioning", 150), ("llava_bench_coco", 30)]
    ]
    jobs += [
        (m, d, n, 300)
        for m in ["m_attack", "foa_attack"]
        for d, n in [("coco_captioning", 150), ("llava_bench_coco", 30)]
    ]
    w = expansion.api()
    with (BASE / "reproduction_owner.lock").open("a") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX)
        for method, dataset, n, steps in jobs:
            if time.time() >= state["deadline"]:
                break
            estimate = (
                (280 if method == "m_attack" else 390) * 60
                if dataset == "coco_captioning"
                else 155 * 60
            )
            if steps == 300 and time.time() + estimate + 600 > state["deadline"]:
                continue
            run, _, _, _ = expansion.prepare(w, method, dataset, n, steps)
            for phase in ["attack", "evaluate", "report"]:
                args = [
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
                    phase,
                ]
                state.update(
                    stage="baseline_" + dataset + "_" + method + "_" + str(steps) + "_" + phase
                )
                atomic(OUT / "state.json", state)
                with (OUT / (state["stage"] + ".log")).open("a") as f:
                    code = subprocess.call(
                        [
                            sys.executable,
                            str(REPO / "scripts/kdd2027/nov21/dataset_expansion.py"),
                            *args,
                        ],
                        env=ENV,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                    )
                if code:
                    return code
            if dataset == "llava_bench_coco":
                if not execute(
                    "judge.py",
                    ["--run", str(run)],
                    "baseline_judge_" + method + "_" + str(steps),
                    state,
                ):
                    return 2
            if not execute(
                "report.py",
                [],
                "baseline_quality_" + method + "_" + dataset + "_" + str(steps),
                state,
            ):
                return 2
            if not execute(
                "asr.py", [], "semantic_ASR_" + method + "_" + dataset + "_" + str(steps), state
            ):
                return 2
    return 0


def publish():
    import importlib.util
    import tarfile

    spec = importlib.util.spec_from_file_location("native_final_report", HERE / "report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.report()
    destination = module.DEST
    if (OUT / "state.json").exists():
        (destination / "state.json").write_bytes((OUT / "state.json").read_bytes())
    with tarfile.open(destination / "structured_evidence.tar.gz", "w:gz") as archive:
        for path in OUT.rglob("*"):
            if (
                path.is_file()
                and path.suffix in [".json", ".jsonl", ".csv", ".log"]
                and path.name != "controller.log"
            ):
                archive.add(path, arcname=str(path.relative_to(OUT)))
    rel = str(destination.relative_to(REPO))
    subprocess.run(["git", "add", "--", rel], cwd=REPO, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet", "--", rel], cwd=REPO).returncode:
        subprocess.run(
            ["git", "commit", "-m", "report native task Ours-first campaign", "--", rel],
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
    try:
        code = main()
    finally:
        publish()
    raise SystemExit(code)
