# ruff: noqa: E402
"""Isolated resumable new controls and small caption/VQA task trials."""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "extension"))
import methods
import torch
import worker as oldworker

from primary_ml_cka.baselines import campaign as legacy
from primary_ml_cka.baselines.protocol import atomic, digest, freeze, sha, valid_receipt

OUT = Path("/home/ubuntu/exper-tran/outputs/kdd2027_baseline_reproduction")


def setup(method, dataset, n, smoke):
    source = OUT / "extension/manifests" / f"{dataset}.json"
    manifest = json.loads(source.read_text())
    if smoke and dataset == "imagenet":
        rows = json.loads((OUT / "manifests/smoke.json").read_text())["images"][:2]
        rows = [
            {
                **r,
                "queries": [{"id": r["image_id"], "kind": "classification"}],
                "captions": ["A photo of a " + methods.CLASSES[r["source"] - 1] + "."],
            }
            for r in rows
        ]
    else:
        rows = manifest["images"][:n]
    manifest = {
        **manifest,
        "images": rows,
        "N_images": len(rows),
        "subset_note": "non_test_engineering"
        if smoke
        else (
            "full_original150"
            if dataset == "imagenet" and n == 150
            else "predeclared_exploratory_fixed_prefix"
        ),
    }
    steps = (
        2
        if smoke
        else (300 if method.startswith("attackvlm_") else 10 if method == "ssa_cwa_clip" else 50)
    )
    config = {
        "method": method,
        "dataset": dataset,
        "variant": "default",
        "steps": steps,
        "epsilon": 16 / 255,
        "seed": 42,
        "canvas": 224,
        "parameters": "bfloat16_unquantized",
        "primary_goal": "targeted_classification_CE"
        if method == "ssa_cwa_clip"
        else "targeted_feature_matching",
        "smoke_only": False,
        "pilot": False,
        "engineering": smoke,
        "N": len(rows),
        "native_budget_or_adaptation": "native300_MF_or_native10_CWA"
        if method.startswith("attackvlm") or method == "ssa_cwa_clip"
        else "matched50_task_trial_not_native300",
    }
    if method == "ssa_cwa_clip" and dataset != "imagenet":
        raise ValueError("SSA CLIP classifier is only defined for ImageNet")
    lock = {
        "config": config,
        "manifest": manifest,
        "source_manifest_sha256": sha(source),
        "scientific_sources": {p.name: sha(p) for p in [Path(__file__), HERE / "methods.py"]},
        "legacy_scientific_sources": legacy.source_hashes(),
        "targets": oldworker.REG,
        "upstream_pins": {"attackvlm": methods.ATTACKVLM_PIN, "ssa_cwa": methods.CWA_PIN},
    }
    run = OUT / "nov21/local" / f"{dataset}_{method}_{len(rows)}_{digest(lock)[:14]}"
    freeze(run / "lock.json", lock)
    freeze(run / "analysis_input/extension/manifests" / f"{dataset}.json", manifest)
    return run, config, lock, manifest


def attack(run, config, lock, manifest):
    if config["method"] in ["m_attack", "foa_attack", "pgd", "mi_fgsm"]:
        oldworker.attacks(config, run, lock, manifest)
        return
    bundle, description = methods.make(config["method"], config["steps"])
    features = bundle[0] if config["method"] == "ssa_cwa_clip" else bundle
    atomic(run / "bf16_proxies.json", [m.audit for m in features])
    atomic(run / "algorithm.json", description)
    for row in manifest["images"]:
        dest = oldworker.receipt_path(run, row)
        signature = oldworker.sig(lock, row)
        if valid_receipt(dest, signature):
            continue
        if (
            sha(row["clean"]) != row["clean_sha256"]
            or sha(row["target_reference"]) != row["target_reference_sha256"]
        ):
            raise RuntimeError("Frozen input changed")
        seed = int(digest([42, row["image_id"], config["method"]])[:8], 16)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        clean, reference = (
            legacy.load_image(row["clean"]),
            legacy.load_image(row["target_reference"]),
        )
        began = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        adv, trace = methods.generate(
            config["method"], bundle, clean, reference, row, config["steps"]
        )
        if not torch.isfinite(adv).all() or (adv - clean).abs().max() > 16 / 255 + 1e-7:
            raise RuntimeError("Invalid perturbation")
        legacy.save_png(dest.parent / "adv.png", adv)
        linf = float((legacy.load_image(dest.parent / "adv.png") - clean).abs().max())
        if linf > 16 / 255 + 1e-7:
            raise RuntimeError("Invalid decoded PNG bounds")
        atomic(dest.parent / "optimization.json", trace)
        atomic(
            dest,
            {
                "status": "ATTACK_FROZEN",
                "signature": signature,
                "image_id": row["image_id"],
                "clean": row["clean"],
                "adv": str(dest.parent / "adv.png"),
                "png_hashes": {
                    row["clean"]: sha(row["clean"]),
                    str(dest.parent / "adv.png"): sha(dest.parent / "adv.png"),
                },
                "linf_png": linf,
                "steps": config["steps"],
                "seed": seed,
                "target_access": False,
                "seconds": time.monotonic() - began,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "forward_calls": trace.get("model_forward_calls"),
            },
        )
        print(
            "FROZEN",
            config["method"],
            row["image_id"],
            round(time.monotonic() - began, 2),
            flush=True,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--method",
        required=True,
        choices=[
            "attackvlm_mfii",
            "attackvlm_mfit",
            "ssa_cwa_clip",
            "m_attack",
            "foa_attack",
            "pgd",
            "mi_fgsm",
        ],
    )
    p.add_argument(
        "--dataset", choices=["imagenet", "coco_captioning", "llava_bench_coco"], required=True
    )
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--phase", choices=["attack", "evaluate", "report"], required=True)
    a = p.parse_args()
    run, config, lock, manifest = setup(a.method, a.dataset, a.N, a.smoke)
    print("RUN", run, flush=True)
    if a.phase == "report":
        from metrics import summarize

        summarize(run, run / "analysis_input")
        return
    with legacy.gpu_owner(OUT):
        if a.phase == "attack":
            attack(run, config, lock, manifest)
        else:
            oldworker.evaluations(config, run, lock, manifest)


if __name__ == "__main__":
    main()
