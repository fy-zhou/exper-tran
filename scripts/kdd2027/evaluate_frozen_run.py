"""Evaluate each frozen method independently; retain full planned denominators.

Operational recovery only: does not modify attack sources, locks, cohorts or PNGs.
A partial method is evaluated only after EVERY original image has a frozen output
or a recorded failed attempt. Failures remain missing outcomes, never exclusions.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from primary_ml_cka.baselines.campaign import (
    DEFAULT_OUT,
    evaluate,
    gpu_owner,
    report,
    signature,
    source_hashes,
)
from primary_ml_cka.baselines.protocol import atomic, digest, freeze, sha, valid_receipt


def build_plan(run, output):
    lock = json.loads((run / "method_lock.json").read_text())
    if source_hashes() != lock["sources"]:
        raise RuntimeError("Original source lock differs; no implicit source migration")
    manifest_path = output / "manifests" / (lock["split"] + ".json")
    if sha(manifest_path) != lock["manifest_sha256"]:
        raise RuntimeError("Original cohort changed")
    manifest = json.loads(manifest_path.read_text())
    groups = defaultdict(list)
    statuses = []
    for method in lock["methods"]:
        valid, failed, pending = [], [], []
        for row in manifest["images"]:
            dest = run / "attacks" / method / digest(row["image_id"])[:18]
            if valid_receipt(dest / "receipt.json", signature(lock, row, method)):
                valid.append(row)
            elif (dest / "failure.json").exists():
                failure = json.loads((dest / "failure.json").read_text())
                if failure.get("image_id") != row["image_id"] or failure.get("method") != method:
                    raise RuntimeError("Failure provenance mismatch")
                failed.append(row["image_id"])
            else:
                pending.append(row["image_id"])
        if valid and not pending:
            groups[tuple(r["image_id"] for r in valid)].append(method)
        statuses.append(
            {
                "method": method,
                "planned_N": len(manifest["images"]),
                "valid_pngs": len(valid),
                "failed_ids": failed,
                "pending_ids": pending,
                "eligible_for_evaluation": bool(valid and not pending),
            }
        )
    return lock, manifest, groups, statuses


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cache-dir", type=Path, default=Path("/ephemeral/kdd2027/hf-cache/hub"))
    args = p.parse_args()
    lock, manifest, groups, statuses = build_plan(args.run, args.output)
    plan = {
        "original_lock_sha256": sha(args.run / "method_lock.json"),
        "original_manifest_sha256": lock["manifest_sha256"],
        "recovery_source_sha256": sha(__file__),
        "methods": statuses,
        "subset_rule": "valid frozen PNGs only after all original images attempted; "
        "all failures retained in planned_N; no prediction gating",
    }
    plan_path = args.run / "evaluation_plans" / (digest(plan)[:18] + ".json")
    freeze(plan_path, plan)
    all_args = SimpleNamespace(methods=lock["methods"], split=lock["split"], steps=lock["steps"])
    try:
        with gpu_owner(args.output):
            for ids, methods in groups.items():
                wanted = set(ids)
                subset = {
                    **manifest,
                    "images": [r for r in manifest["images"] if r["image_id"] in wanted],
                }
                operational = SimpleNamespace(
                    methods=methods, output=args.output, cache_dir=args.cache_dir
                )
                print("EVALUATE", methods, f'{len(ids)}/{manifest["eligible"]} frozen', flush=True)
                evaluate(operational, subset, args.run, lock)
                report(all_args, manifest, args.run, lock)
    finally:
        report(all_args, manifest, args.run, lock)
        atomic(
            args.run / "evaluation_recovery_status.json",
            {"time": time.time(), "plan": str(plan_path), "methods": statuses},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
