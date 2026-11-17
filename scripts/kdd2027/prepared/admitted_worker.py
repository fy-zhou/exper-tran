"""Execution child of the one verified GPU owner; cannot run standalone."""

import contextlib
import fcntl
import json
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path("/home/ubuntu/exper-tran/outputs/kdd2027_baseline_reproduction")
kind, leasefile, cap = sys.argv[1:4]
args = sys.argv[4:]
lease = json.loads(Path(leasefile).read_text())
parent = os.getppid()
assert (
    parent == lease["owner_pid"]
    and Path(f"/proc/{parent}/stat").read_text().split()[21] == lease["owner_start"]
)
assert "remaining_parallel.py" in Path(f"/proc/{parent}/cmdline").read_bytes().decode().replace(
    "\0", " "
)
# Parent holds both historical exclusive admission locks. No second GPU owner exists.
import torch  # noqa: E402

from primary_ml_cka.baselines import campaign as legacy  # noqa: E402


@contextlib.contextmanager
def admitted(output):
    free, total = torch.cuda.mem_get_info()
    if free < (float(cap) + 8) * 2**30:
        raise RuntimeError("Admission free-memory reserve no longer available")
    torch.cuda.set_per_process_memory_fraction(float(cap) * 2**30 / total)
    yield


legacy.gpu_owner = admitted
# Prepared entrypoint also requests outer ownership; it is delegated by our direct
# parent. Only that exact inode/path lock is delegated, all other locks stay real.
original = fcntl.flock


def delegated(fd, operation):
    number = fd if isinstance(fd, int) else fd.fileno()
    target = os.readlink(f"/proc/self/fd/{number}")
    if target == str(OUT / "reproduction_owner.lock"):
        assert os.getppid() == parent
        return
    return original(fd, operation)


fcntl.flock = delegated
if kind == "foa":
    sys.argv = ["worker.py", "--method", "foa_attack", "--dataset", "imagenet"] + args
    runpy.run_path(str(HERE.parent / "extension/worker.py"), run_name="__main__")
elif kind == "generate":
    sys.argv = ["prepare_or_generate.py"] + args
    runpy.run_path(str(HERE / "prepare_or_generate.py"), run_name="__main__")
elif kind == "evaluate":
    sys.path.insert(0, str(HERE.parent / "extension"))
    import worker
    from metrics import summarize

    from primary_ml_cka.baselines.protocol import atomic, digest, freeze, sha, valid_receipt

    method = args[0]
    source = OUT / "prepared_remaining" / f"generation_{method}"
    packet = json.loads((OUT / "prepared_remaining" / f"{method}_final150.json").read_text())
    oldlock = json.loads((source / "method_lock.json").read_text())
    manifest = json.loads((OUT / "extension/manifests/imagenet.json").read_text())
    config = {
        "method": method,
        "dataset": "imagenet",
        "variant": "default",
        "smoke_only": False,
        "pilot": False,
        "steps": oldlock["steps"],
    }
    lock = {
        "config": config,
        "generation_lock_sha256": sha(source / "method_lock.json"),
        "targets": worker.REG,
        "adapter_sha256": sha(__file__),
    }
    dest = OUT / "prepared_remaining" / f"evaluation_{method}"
    freeze(dest / "lock.json", lock)
    for row, p in zip(manifest["images"], packet["images"], strict=True):
        assert row["image_id"] == p["image_id"]
        original_receipt = source / "attacks" / digest(p["image_id"])[:18] / "receipt.json"
        if not valid_receipt(original_receipt, digest([oldlock, p])):
            raise RuntimeError("Missing/mismatched frozen PNG; target access forbidden")
        value = json.loads(original_receipt.read_text())
        atomic(
            worker.receipt_path(dest, row),
            {
                **value,
                "signature": worker.sig(lock, row),
                "generation_receipt": str(original_receipt),
                "generation_receipt_sha256": sha(original_receipt),
            },
        )
    with admitted(OUT):
        worker.evaluations(config, dest, lock, manifest)
    summarize(dest, OUT)
else:
    raise RuntimeError("Unknown admitted job")
