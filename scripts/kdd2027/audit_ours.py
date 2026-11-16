"""Read-only audit of the historical published artifact chain, without importing Ours."""

import argparse
import json
from pathlib import Path

from primary_ml_cka.baselines.protocol import atomic, digest, sha


def audit(root):
    lock_path = root / "paper_closeout/final_method_lock.json"
    lock = json.loads(lock_path.read_text())
    cohort_path = root / "manifests/final150.json"
    cohort = json.loads(cohort_path.read_text())["images"]
    records = []
    for family, methods in lock["families"].items():
        for method, entry in methods.items():
            config = entry["condition"]
            receipts, errors = [], []
            for row in cohort:
                p = root / "attacks" / config["id"] / digest(row["image_id"])[:18] / "state.json"
                if not p.is_file():
                    errors.append({"image_id": row["image_id"], "error": "missing receipt"})
                    continue
                r = json.loads(p.read_text())
                good = (
                    r.get("status") == "COMPLETE"
                    and r.get("steps") == config["steps"]
                    and not r.get("target_access", True)
                    and r.get("linf_png", 1) <= 16 / 255 + 1e-7
                    and all(Path(x).is_file() and sha(x) == h for x, h in r["png_hashes"].items())
                )
                if not good:
                    errors.append(
                        {"image_id": row["image_id"], "error": "invalid PNG/protocol receipt"}
                    )
                else:
                    receipts.append(
                        {
                            "image_id": r["image_id"],
                            "receipt": str(p),
                            "sha256": sha(p),
                            "steps": r["steps"],
                        }
                    )
            records.append(
                {
                    "family": family,
                    "method": method,
                    "config": config,
                    "eligible": len(cohort),
                    "valid_attacks": len(receipts),
                    "errors": errors,
                    "receipts": receipts,
                }
            )
    return {
        "status": "PASS" if all(not r["errors"] for r in records) else "PARTIAL",
        "scope": "published_V8_artifacts_not_an_assertion_about_unsupplied_manuscript",
        "method_lock": str(lock_path),
        "method_lock_sha256": sha(lock_path),
        "cohort_sha256": sha(cohort_path),
        "N": len(cohort),
        "directions": sorted({(r["transition"], r["source"], r["target"]) for r in cohort}),
        "published_steps": sorted({r["config"]["steps"] for r in records}),
        "GPT50_used_for_selection": lock["GPT50_used_for_selection"],
        "final150_feedback_for_selection": lock["final150_feedback_for_selection"],
        "conditions": records,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=Path,
        default=Path("/home/ubuntu/exper-tran/outputs/" "a100_precommercial_v8_joint_selection24h"),
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = audit(args.root)
    atomic(args.output, result)
    print(
        result["status"],
        "conditions",
        len(result["conditions"]),
        "steps",
        result["published_steps"],
    )
    raise SystemExit(0 if result["status"] == "PASS" else 2)
