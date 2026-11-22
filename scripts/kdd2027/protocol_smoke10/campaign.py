"""Resumable small campaign; one existing admission owner, no extra GPU scheduler."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from prepare import BASE, OUT, REPO
from report import report

from primary_ml_cka.baselines.protocol import atomic

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", action="store_true")
    p.add_argument("--status", action="store_true")
    a = p.parse_args()
    if not a.start:
        print((OUT / "state.json").read_text() if (OUT / "state.json").exists() else "PREPARING")
        report()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO / "src"),
        "PRIMARY_ML_CKA_STRICT_BF16": "1",
        "OPENBLAS_NUM_THREADS": "2",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    statefile = OUT / "state.json"
    state = (
        json.loads(statefile.read_text())
        if statefile.exists()
        else {"started": time.time(), "done": [], "failed": []}
    )
    with (OUT / "controller.lock").open("a") as controller:
        fcntl.flock(controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Reuse CPU/raw artifacts before waiting for the active old GPU campaign.
        for family in ["qwen", "internvl", "gemma4"]:
            r = subprocess.run(
                [
                    sys.executable,
                    str(HERE / "worker.py"),
                    "reuse",
                    "--dataset",
                    "llava",
                    "--method",
                    "ours",
                    "--family",
                    family,
                ],
                env=env,
            )
            if r.returncode:
                raise RuntimeError("LLaVA reuse audit failed")
        report()
        with (BASE / "reproduction_owner.lock").open("a") as owner:
            while True:
                try:
                    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    state.update(status="WAITING_EXISTING_GPU_OWNER", heartbeat=time.time())
                    atomic(statefile, state)
                    time.sleep(15)
            # All Ours attacks first, then frozen post-attack evaluation. Final
            # targets never determine new candidates/settings or cohort contents.
            jobs = []
            for family in ["qwen", "internvl", "gemma4"]:
                jobs.append(("crossvlad", "ours", family, "attack"))
            for family in ["qwen", "internvl", "gemma4"]:
                jobs.append(("crossvlad", "ours", family, "evaluate"))
            for dataset in ["crossvlad", "llava"]:
                for method in ["m_attack", "foa_attack", "ssa_cwa_clip"]:
                    jobs.extend(
                        [(dataset, method, None, "attack"), (dataset, method, None, "evaluate")]
                    )
            for dataset, method, family, phase in jobs:
                identity = "_".join([dataset, method, family or "all", phase])
                if identity in state["done"]:
                    continue
                attack_identity = "_".join([dataset, method, family or "all", "attack"])
                if phase == "evaluate" and attack_identity not in state["done"]:
                    state["failed"].append({"stage": identity, "reason": "attack not complete"})
                    atomic(statefile, state)
                    continue
                args = [
                    sys.executable,
                    str(HERE / "worker.py"),
                    phase,
                    "--dataset",
                    dataset,
                    "--method",
                    method,
                ]
                if family:
                    args += ["--family", family]
                state.update(status="RUNNING", stage=identity, heartbeat=time.time())
                atomic(statefile, state)
                logfile = OUT / (identity + ".log")
                with logfile.open("a") as stream:
                    child = subprocess.Popen(args, env=env, stdout=stream, stderr=subprocess.STDOUT)
                    while child.poll() is None:
                        state.update(pid=child.pid, heartbeat=time.time())
                        atomic(statefile, state)
                        time.sleep(15)
                if child.returncode:
                    state["failed"].append(
                        {"stage": identity, "exit": child.returncode, "log": str(logfile)}
                    )
                else:
                    state["done"].append(identity)
                atomic(statefile, state)
                report()
            rows = report()
            ready = len(rows) == 24 and all(r["returned"] == r["planned"] for r in rows)
            state.update(
                status="RAW_COMPLETE_OFFICIAL_LLAVA_JUDGE_BLOCKED"
                if ready
                else "PARTIAL_RESUMABLE",
                finished=time.time(),
                heartbeat=time.time(),
            )
            atomic(statefile, state)


if __name__ == "__main__":
    main()
