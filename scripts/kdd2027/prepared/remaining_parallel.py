"""Single admission owner: resume immutable FOA + smoke/admit two new baselines.

IAU and mismatched SGA are never scheduled. FOA evaluation gets exclusive priority.
"""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path("/home/ubuntu/exper-tran/outputs/kdd2027_baseline_reproduction")
STATE = OUT / "prepared_remaining"
STATE.mkdir(exist_ok=True)
LEASE = STATE / "admission.json"
CHILDREN = []
ENV = {
    **os.environ,
    "PYTHONPATH": (
        f"/ephemeral/kdd2027/prepared-deps:{HERE.parents[2]}/src:" f"{HERE.parent}/extension:{HERE}"
    ),
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "TOKENIZERS_PARALLELISM": "false",
    "PRIMARY_ML_CKA_STRICT_BF16": "1",
}


def atomic(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def spawn(name, kind, args, cap):
    free = (
        float(
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                text=True,
            ).strip()
        )
        / 1024
    )
    if free < cap + 8:
        raise RuntimeError(f"Insufficient actual free VRAM for {name}: {free}GiB")
    stream = (STATE / f"{name}.log").open("a")
    proc = subprocess.Popen(
        [sys.executable, str(HERE / "admitted_worker.py"), kind, str(LEASE), str(cap), *args],
        stdout=stream,
        stderr=subprocess.STDOUT,
        env={
            **ENV,
            **(
                {
                    "OMP_NUM_THREADS": "4",
                    "MKL_NUM_THREADS": "4",
                    "PYTHONPATH": f"{HERE.parents[2]}/src:{HERE.parent}/extension:{HERE}",
                }
                if kind == "foa"
                else {}
            ),
        },
    )
    stream.close()
    CHILDREN.append(proc)
    print("START", name, proc.pid, "cap_GiB", cap, flush=True)
    return proc


def gen_args(method, smoke):
    split = "smoke" if smoke else "final150"
    result = [
        "--method",
        method,
        "--packet",
        str(STATE / f"{method}_{split}.json"),
        "--output",
        str(STATE / (f"smoke_{method}" if smoke else f"generation_{method}")),
        "--generate",
    ]
    if method == "m_attack_v2":
        result += ["--steps", "2" if smoke else "300"]
    else:
        result += [
            "--clip-weights",
            "/ephemeral/kdd2027/anyattack/ViT-B-32.pt",
            "--decoder-weights",
            "/ephemeral/kdd2027/hf-cache/hub/models--jiamingzz--anyattack/snapshots/4c7172a1ada67ab01e6606ddda4ccacb1362ea4a/checkpoints/coco_cos.pt",
            "--decoder-sha256",
            "bed1f25d46b4eaef0c3d4eb39c3853b2241991cc048b3b2fff21b7b8c38e0070",
        ]
    return result


def wait(proc, name):
    while proc.poll() is None:
        atomic(
            STATE / "heartbeat.json",
            {
                "time": time.time(),
                "waiting": name,
                "pid": proc.pid,
                "owner": os.getpid(),
                "IAU_disabled": True,
            },
        )
        time.sleep(10)
    print("EXIT", name, proc.returncode, flush=True)
    return proc.returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--takeover-owner", type=int, required=True)
    args = p.parse_args()
    # Only transfer our exact previously user-authorized shell/FOA child pair.
    owner = args.takeover_owner
    if Path(f"/proc/{owner}/cmdline").exists():
        command = Path(f"/proc/{owner}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        assert "bash scripts/kdd2027/finish_foa_co.sh" in command
        os.kill(owner, signal.SIGSTOP)
        children = []
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                stat = (proc / "stat").read_text().split()
                cmd = (proc / "cmdline").read_bytes()
                if int(stat[3]) == owner and b"extension/worker.py" in cmd and b"foa_attack" in cmd:
                    children.append(int(proc.name))
            except (OSError, ValueError):
                pass
        for child in children:
            os.kill(child, signal.SIGTERM)
        os.kill(owner, signal.SIGTERM)
        os.kill(owner, signal.SIGCONT)
        atomic(
            STATE / "handoff.json",
            {
                "old_owner": owner,
                "old_children": children,
                "new_owner": os.getpid(),
                "time": time.time(),
                "science_files_modified": False,
                "note": (
                    "Completed atomic receipts reused. "
                    "At most interrupted current image recomputed."
                ),
            },
        )
        for _ in range(20):
            if not any(
                Path(f"/proc/{x}/cmdline").exists() and Path(f"/proc/{x}/cmdline").read_bytes()
                for x in children
            ):
                break
            time.sleep(1)
    with (
        (OUT / "reproduction_owner.lock").open("a") as outer,
        (OUT / "gpu_owner.lock").open("a") as gpu,
    ):
        for attempt in range(60):
            try:
                fcntl.flock(outer, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if attempt == 59:
                    raise
                time.sleep(1)
        atomic(
            LEASE,
            {
                "owner_pid": os.getpid(),
                "owner_start": Path(f"/proc/{os.getpid()}/stat").read_text().split()[21],
            },
        )
        foa = spawn("foa_attack", "foa", ["--phase", "attack"], 8)
        pending = []
        failed = []
        # Serial initializations, at most FOA+one new generator concurrently.
        time.sleep(8)
        for method in ["anyattack", "m_attack_v2"]:
            if method == "m_attack_v2" and (STATE / "user_stop_v2.json").exists():
                print("USER_PAUSED m_attack_v2; do not resume", flush=True)
                continue
            while not (STATE / f"{method}_final150.json").exists():
                time.sleep(5)
            smoke = spawn("smoke_" + method, "generate", gen_args(method, True), 12)
            if wait(smoke, "smoke_" + method):
                failed.append(method)
                continue
            job = spawn("generation_" + method, "generate", gen_args(method, False), 16)
            if method == "anyattack":
                if wait(job, "generation_" + method):
                    failed.append(method)
                else:
                    pending.append(method)
            else:
                while foa.poll() is None and job.poll() is None:
                    atomic(
                        STATE / "heartbeat.json",
                        {
                            "time": time.time(),
                            "running": ["foa_attack", method],
                            "pids": [foa.pid, job.pid],
                            "IAU_disabled": True,
                        },
                    )
                    time.sleep(15)
                if job.poll() is None:
                    job.terminate()
                    wait(job, "paused_for_FOA_evaluation")
                    pending.append("resume_" + method)
                elif job.returncode:
                    failed.append(method)
                else:
                    pending.append(method)
        foa_code = wait(foa, "foa_attack")
        if foa_code == 0:
            evaluation = spawn("foa_evaluate", "foa", ["--phase", "evaluate"], 68)
            foa_code = wait(evaluation, "foa_evaluate")
            if not foa_code:
                subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import sys;sys.path.insert(0,'scripts/kdd2027/extension');"
                            "from worker import *;from metrics import summarize;"
                            "c=condition('foa_attack','imagenet','default',False);"
                            "c['pilot']=False;r,_,_=lock_job(c);summarize(r,OUT)"
                        ),
                    ],
                    env=ENV,
                    check=True,
                )
        with (OUT / "extension/finish_foa_co.log").open("a") as f:
            f.write(
                f'{time.strftime("%FT%TZ",time.gmtime())} METHOD_FINISHED '
                f'foa_attack failure={foa_code}\n'
            )
        for method in pending:
            if "m_attack_v2" in method and (STATE / "user_stop_v2.json").exists():
                continue
            if method.startswith("resume_"):
                method = method.removeprefix("resume_")
                proc = spawn("generation_" + method, "generate", gen_args(method, False), 16)
                if wait(proc, "generation_" + method):
                    failed.append(method)
                    continue
            proc = spawn("evaluate_" + method, "evaluate", [method], 68)
            if wait(proc, "evaluate_" + method):
                failed.append(method)
        atomic(
            STATE / "final_status.json",
            {
                "time": time.time(),
                "FOA_exit": foa_code,
                "failed": failed,
                "IAU_started": False,
                "completed_without_failures": not failed and foa_code == 0,
            },
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        for child in CHILDREN:
            if child.poll() is None:
                child.terminate()
