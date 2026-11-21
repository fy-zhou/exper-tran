"""Bounded new baselines and initial distinct-task evaluations, no V2/IAU resume."""

import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path("/home/ubuntu/exper-tran/outputs/kdd2027_baseline_reproduction")
STATE = OUT / "nov21"
STATE.mkdir(exist_ok=True)
ENV = {
    **os.environ,
    "PYTHONPATH": str(HERE.parents[2] / "src"),
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "2",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PRIMARY_ML_CKA_STRICT_BF16": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


def main():
    jobs = []
    for method in ["attackvlm_mfii", "attackvlm_mfit", "ssa_cwa_clip"]:
        jobs.append(
            {"method": method, "dataset": "imagenet", "N": 2, "smoke": True, "phases": ["attack"]}
        )
    # Early task trials before new full150 classifications: these are different metrics/tasks.
    for dataset, n in [("coco_captioning", 10), ("llava_bench_coco", 3)]:
        for method in ["m_attack", "foa_attack", "pgd"]:
            jobs.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "N": n,
                    "smoke": False,
                    "phases": ["attack", "evaluate", "report"],
                }
            )
    for method in ["attackvlm_mfii", "attackvlm_mfit", "ssa_cwa_clip"]:
        jobs.append(
            {
                "method": method,
                "dataset": "imagenet",
                "N": 150,
                "smoke": False,
                "phases": ["attack", "evaluate", "report"],
            }
        )
    (STATE / "local_queue.json").write_text(json.dumps(jobs, indent=2) + "\n")
    failed = []
    with (OUT / "reproduction_owner.lock").open("a") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for i, job in enumerate(jobs):
            if job["method"] in failed:
                continue
            for phase in job["phases"]:
                log = STATE / f"job_{i:02d}_{phase}.log"
                cmd = [
                    sys.executable,
                    str(HERE / "worker.py"),
                    "--method",
                    job["method"],
                    "--dataset",
                    job["dataset"],
                    "--N",
                    str(job["N"]),
                    "--phase",
                    phase,
                ]
                if job["smoke"]:
                    cmd += ["--smoke"]
                print("START", i, job, phase, flush=True)
                with log.open("a") as f:
                    child = subprocess.Popen(cmd, env=ENV, stdout=f, stderr=subprocess.STDOUT)
                    while child.poll() is None:
                        (STATE / "local_heartbeat.json").write_text(
                            json.dumps(
                                {"time": time.time(), "job": job, "phase": phase, "pid": child.pid},
                                indent=2,
                            )
                            + "\n"
                        )
                        time.sleep(10)
                if child.returncode:
                    print("FAILED", i, phase, child.returncode, flush=True)
                    if job["smoke"]:
                        failed.append(job["method"])
                    break
            else:
                print("COMPLETE_JOB", i, flush=True)
    (STATE / "local_finished.json").write_text(
        json.dumps({"time": time.time(), "smoke_blocked": failed}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
