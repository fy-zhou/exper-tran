"""Resumable baseline campaign; target access occurs only in evaluate()."""

from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .protocol import (
    PROXIES,
    UPSTREAMS,
    atomic,
    digest,
    freeze,
    make_manifest,
    sha,
    valid_receipt,
    validate_method,
)

REPO = Path(__file__).resolve().parents[3]
MAIN = Path("/home/ubuntu/exper-tran")
DEFAULT_OUT = MAIN / "outputs/kdd2027_baseline_reproduction"
V6 = REPO / "experiments/2026-10-a100-precommercial-v6"
TARGETS = ["Qwen3.5-27B", "InternVL3.5-14B-HF", "Gemma 4 31B-it"]


def source_hashes():
    files = sorted(Path(__file__).parent.glob("*.py"))
    files += [
        REPO / "src/primary_ml_cka/models/common/precision.py",
        REPO / "src/primary_ml_cka/models/backends/target_transformers_generation.py",
        REPO / "src/primary_ml_cka/models/targets/generation.py",
        REPO / "src/primary_ml_cka/prompts/parser.py",
        REPO / "configs/baselines/kdd2027_campaign.yaml",
    ]
    return {str(p.relative_to(REPO)): sha(p) for p in files}


def target_registry():
    bank = json.loads((V6 / "model_revisions.json").read_text())
    return {name: bank[name] for name in TARGETS}


def snapshots(cache_dir, specs):
    results = {}
    for name, spec in specs.items():
        try:
            p = (
                Path(cache_dir)
                / ("models--" + spec["repo_id"].replace("/", "--"))
                / "snapshots"
                / spec["revision"]
            )
            if not (p / "config.json").is_file():
                raise FileNotFoundError("Missing pinned snapshot/config")
            weights = list(p.glob("*.safetensors")) + list(p.glob("pytorch_model*.bin"))
            if not weights or any(not w.exists() for w in weights):
                raise FileNotFoundError("Missing model weight files")
            # If sharded, every indexed shard must exist (a directory alone is not proof).
            for index in p.glob("*.index.json"):
                for shard in set(json.loads(index.read_text()).get("weight_map", {}).values()):
                    if not (p / shard).is_file():
                        raise FileNotFoundError(f"Missing shard: {shard}")
            results[name] = {"status": "AVAILABLE", "path": str(p)}
        except Exception as e:
            results[name] = {"status": "BLOCKED_MISSING_CHECKPOINT", "error": str(e)}
    return results


def prepare(args):
    manifest = make_manifest(args.ours_root, args.data_root, args.split)
    p = args.output / "manifests" / f"{args.split}.json"
    freeze(p, manifest)
    lock = {
        "version": 1,
        "sources": source_hashes(),
        "manifest_sha256": sha(p),
        "models": PROXIES,
        "target_models": target_registry(),
        "reference_policy": manifest["reference_selection"],
        "steps": args.steps,
        "epsilon": 16 / 255,
        "alpha": 1 / 255,
        "seed": 42,
        "methods": args.methods,
        "split": args.split,
        "proxy_order": args.proxies,
        "parameter_dtype": "bfloat16",
        "no_quantization": True,
        "baseline_specific_objective_not_ours": True,
        "upstream_pins": UPSTREAMS,
        "target_feedback": False,
    }
    ident = digest(lock)[:18]
    run = args.output / "runs" / f"{args.split}_{args.steps}_{ident}"
    run.mkdir(parents=True, exist_ok=True)
    freeze(run / "method_lock.json", lock)
    return manifest, run, lock


def preflight(args, manifest, run):
    import torch

    status = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "branch": subprocess.check_output(
            ["git", "-C", str(REPO), "branch", "--show-current"], text=True
        ).strip(),
        "commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "cuda_available": torch.cuda.is_available(),
        "torch": torch.__version__,
        "eligible": manifest["eligible"],
        "excluded": 0,
        "proxies": snapshots(args.cache_dir, {n: PROXIES[n] for n in args.proxies}),
        "targets": snapshots(args.cache_dir, target_registry()),
        "persistent_output_root": str(args.output),
        "upstreams": {},
    }
    for name, spec in UPSTREAMS.items():
        p = args.upstream_root / name
        try:
            actual = subprocess.check_output(
                ["git", "-C", str(p), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            status["upstreams"][name] = {
                "pin": spec["revision"],
                "actual": actual,
                "status": "PASS" if actual == spec["revision"] else "BLOCKED",
            }
        except subprocess.CalledProcessError:
            status["upstreams"][name] = {"status": "BLOCKED_NOT_CLONED"}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        status["gpu"] = {
            "name": torch.cuda.get_device_name(),
            "free_bytes": free,
            "total_bytes": total,
            "bf16": torch.cuda.is_bf16_supported(),
        }
    status["attack_ready"] = (
        status["cuda_available"]
        and status["gpu"]["bf16"]
        and all(x["status"] == "AVAILABLE" for x in status["proxies"].values())
    )
    status["evaluation_ready"] = all(x["status"] == "AVAILABLE" for x in status["targets"].values())
    atomic(run / "preflight.json", status)
    print(json.dumps(status, indent=2), flush=True)
    return status


@contextlib.contextmanager
def gpu_owner(output):
    import torch

    output.mkdir(parents=True, exist_ok=True)
    with (output / "gpu_owner.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # External jobs are not terminated or competed with. One baseline owner only.
        free, total = torch.cuda.mem_get_info()
        if total - free > 2 * 2**30:
            raise RuntimeError("GPU already occupied; retry after existing workload finishes")
        torch.cuda.set_per_process_memory_fraction(min(0.90, 72 * 2**30 / total))
        yield


def load_image(path):
    import numpy as np
    import torch
    from PIL import Image

    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB")).copy()
    if array.shape != (224, 224, 3):
        raise ValueError(f"Changed canonical canvas: {path}")
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255


def save_png(path, image):
    import numpy as np
    from PIL import Image

    pixels = image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    array = np.rint(np.clip(pixels, 0, 1) * 255).astype("uint8")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as stream:
        Image.fromarray(array).save(stream, format="PNG")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def signature(lock, row, method):
    return digest([lock, row, method])


def attack(args, manifest, run, lock):
    import torch

    from .clip_features import ClipFeatures
    from .feature_attacks import run_feature_attack

    models = [ClipFeatures(n, args.cache_dir) for n in args.proxies]
    atomic(run / "bf16_proxies.json", [m.audit for m in models])
    failures = []
    for method in args.methods:
        validate_method(method)
        for ix, row in enumerate(manifest["images"]):
            dest = run / "attacks" / method / digest(row["image_id"])[:18]
            sig = signature(lock, row, method)
            if valid_receipt(dest / "receipt.json", sig):
                print(f'[{method}] {ix + 1}/{len(manifest["images"])} REUSED', flush=True)
                continue
            began = time.monotonic()
            try:
                # Recheck immutable inputs at execution, not just manifest creation.
                for p, h in [
                    (row["clean"], row["clean_sha256"]),
                    (row["target_reference"], row["target_reference_sha256"]),
                ]:
                    if sha(p) != h:
                        raise RuntimeError("Frozen input changed")
                clean, reference = load_image(row["clean"]), load_image(row["target_reference"])
                torch.cuda.reset_peak_memory_stats()
                counts = [m.forwards for m in models]
                seed = int(digest([42, row["image_id"], method])[:8], 16)
                adv, trace = run_feature_attack(
                    method,
                    models,
                    clean,
                    reference,
                    steps=args.steps,
                    seed=seed,
                    upstream_root=args.upstream_root,
                )
                # Both tensor AND saved/decoded PNG must satisfy the original budget.
                if (adv - clean).abs().max() > 16 / 255 + 1e-7:
                    raise RuntimeError("Tensor exceeds epsilon")
                dest.mkdir(parents=True, exist_ok=True)
                save_png(dest / "adv.png", adv)
                linf = float((load_image(dest / "adv.png") - clean).abs().max())
                if linf > 16 / 255 + 1e-7:
                    raise RuntimeError("Saved PNG exceeds epsilon")
                atomic(dest / "optimization.json", trace)
                receipt = {
                    "status": "ATTACK_FROZEN",
                    "signature": sig,
                    "image_id": row["image_id"],
                    "transition": row["transition"],
                    "source": row["source"],
                    "target": row["target"],
                    "clean": row["clean"],
                    "adv": str(dest / "adv.png"),
                    "png_hashes": {
                        row["clean"]: row["clean_sha256"],
                        str(dest / "adv.png"): sha(dest / "adv.png"),
                    },
                    "linf_png": linf,
                    "target_access": False,
                    "seed": seed,
                    "steps": args.steps,
                    "fidelity": trace["fidelity"],
                    "forward_calls": sum(
                        m.forwards - n for m, n in zip(models, counts, strict=True)
                    ),
                    "backward_calls": trace["backward_calls"],
                    "seconds": time.monotonic() - began,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                }
                atomic(dest / "receipt.json", receipt)
                print(
                    f'[{method}] {ix + 1}/{len(manifest["images"])} FROZEN '
                    f'{receipt["seconds"]:.1f}s eps={linf:.5f}',
                    flush=True,
                )
            except Exception as e:
                failure = {
                    "method": method,
                    "image_id": row["image_id"],
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                    "status": "FAILED_NOT_EXCLUDED",
                }
                atomic(dest / "failure.json", failure)
                failures.append(failure)
                print(json.dumps(failure), flush=True)
                # Keep independent cells, but never mask a broken method as success.
                if isinstance(e, torch.OutOfMemoryError):
                    raise
    del models
    torch.cuda.empty_cache()
    atomic(run / "attack_failures.json", failures)
    return not failures


def canonical_prompt():
    import yaml

    p = REPO / "experiments/2026-09-a100-final-v5/src/common.py"
    spec = importlib.util.spec_from_file_location("kdd_canonical_prompt", p)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.classification_prompt(
        yaml.safe_load((V6 / "config/runner_template.yaml").read_text())
    )


def evaluate(args, manifest, run, lock):
    """Separate phase: all method PNGs must be frozen before any target is loaded."""
    import torch

    from primary_ml_cka.models.backends.target_transformers_generation import (
        load_target_for_generation,
    )
    from primary_ml_cka.models.backends.transformers_backend import load_processor
    from primary_ml_cka.models.targets.generation import TransformersTargetGenerator

    from .clip_features import audit_model

    for method in args.methods:
        for row in manifest["images"]:
            p = run / "attacks" / method / digest(row["image_id"])[:18] / "receipt.json"
            if not valid_receipt(p, signature(lock, row, method)):
                raise RuntimeError(f"Target access blocked: missing/invalid frozen attack {p}")
    prompt = canonical_prompt()
    available = snapshots(args.cache_dir, target_registry())
    blocked = []
    for name, spec in target_registry().items():
        if available[name]["status"] != "AVAILABLE":
            blocked.append({"target": name, **available[name]})
            continue
        snapshot = Path(available[name]["path"])
        processor = load_processor(snapshot)
        model = load_target_for_generation(snapshot, torch.device("cuda"), precision="bf16")
        atomic(run / "audits" / (digest(name)[:12] + ".json"), audit_model(model, spec, "target"))
        generator = TransformersTargetGenerator(model, processor)
        for method in args.methods:
            for row in manifest["images"]:
                receipt_path = (
                    run / "attacks" / method / digest(row["image_id"])[:18] / "receipt.json"
                )
                receipt = json.loads(receipt_path.read_text())
                sig = digest([receipt["signature"], receipt["png_hashes"], spec, prompt])
                dest = (
                    run / "evaluations" / name / method / (digest(row["image_id"])[:18] + ".json")
                )
                if dest.exists() and json.loads(dest.read_text()).get("signature") == sig:
                    continue
                try:
                    ck = digest([row["clean_sha256"], spec, prompt])
                    cp = args.output / "cache/clean_predictions" / (ck + ".json")
                    with torch.inference_mode():
                        if cp.exists():
                            clean = json.loads(cp.read_text())
                        else:
                            response = generator.generate_label(Path(row["clean"]), prompt)
                            clean = {
                                "label": response.parsed_label,
                                "raw": response.raw_output,
                                "parser_status": response.parser_status,
                            }
                            atomic(cp, clean)
                        response = generator.generate_label(Path(receipt["adv"]), prompt)
                    atomic(
                        dest,
                        {
                            "signature": sig,
                            "image_id": row["image_id"],
                            "transition": row["transition"],
                            "model": spec,
                            "target_label": row["target"],
                            "source_label": row["source"],
                            "clean": clean,
                            "adv_label": response.parsed_label,
                            "adv_raw": response.raw_output,
                            "parser_status": response.parser_status,
                            "target_hit": response.parsed_label == row["target"],
                            "clean_correct": clean["label"] == row["source"],
                        },
                    )
                except Exception as e:
                    blocked.append(
                        {
                            "target": name,
                            "method": method,
                            "image_id": row["image_id"],
                            "error": repr(e),
                        }
                    )
        del generator, model, processor
        torch.cuda.empty_cache()
    atomic(run / "evaluation_blockers.json", blocked)
    return not blocked


def report(args, manifest, run, lock):
    rows = []
    for method in args.methods:
        frozen = sum(
            valid_receipt(
                run / "attacks" / method / digest(r["image_id"])[:18] / "receipt.json",
                signature(lock, r, method),
            )
            for r in manifest["images"]
        )
        for target in TARGETS:
            data = [
                json.loads(p.read_text())
                for p in (run / "evaluations" / target / method).glob("*.json")
            ]
            for tid in ["ALL", "T01", "T03", "T05", "T07", "T09"]:
                selected = [r for r in data if tid == "ALL" or r["transition"] == tid]
                N = sum(tid == "ALL" or r["transition"] == tid for r in manifest["images"])
                if not N:
                    continue
                hits = sum(r["target_hit"] for r in selected)
                clean_correct = [r for r in selected if r["clean_correct"]]
                complete = len(selected) == N
                interval = None
                if complete:
                    z = 1.95996398454
                    mean = (hits / N + z * z / (2 * N)) / (1 + z * z / N)
                    width = (
                        z
                        * ((hits / N * (1 - hits / N) / N + z * z / (4 * N * N)) ** 0.5)
                        / (1 + z * z / N)
                    )
                    interval = (mean - width, mean + width)
                rows.append(
                    {
                        "method": method,
                        "target": target,
                        "transition": tid,
                        "eligible": N,
                        "excluded": 0,
                        "attacks_frozen_total": frozen,
                        "evaluated": len(selected),
                        "target_hits": hits,
                        "target_ASR": hits / N if complete else None,
                        "wilson_low": interval[0] if interval else None,
                        "wilson_high": interval[1] if interval else None,
                        "clean_correct_N": len(clean_correct),
                        "clean_correct_target_hits": sum(r["target_hit"] for r in clean_correct),
                        "clean_correct_ASR": sum(r["target_hit"] for r in clean_correct)
                        / len(clean_correct)
                        if complete and clean_correct
                        else None,
                        "parser_invalid": sum(r["adv_label"] is None for r in selected),
                        "status": "COMPLETE" if complete else "PARTIAL" if selected else "BLOCKED",
                    }
                )
    atomic(run / "results.json", rows)
    with (run / "results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    complete = all(r["status"] == "COMPLETE" for r in rows)
    (run / "REPORT.md").write_text(
        f'# Baseline reproduction: {"COMPLETE" if complete else "PARTIAL / BLOCKED"}\n\n'
        f'Split `{args.split}`, eligible N={manifest["eligible"]}, steps={args.steps}. '
        'Missing evaluations are N/A, never zero ASR. '
        'PGD/MI use an explicitly adapted image-only cosine objective. '
        'M/FOA use pinned official cores with disclosed task/precision/API adaptations.\n\n'
        'See method_lock.json, bf16_proxies.json, attack receipts, optimization traces, '
        'evaluation_blockers.json and results.csv. Formal test scores require all planned images; '
        'smoke/pilot results are not final results. No commercial calls are made.\n'
    )
    print(f'Report: {run / "REPORT.md"}', flush=True)
    return complete


def download(args, targets=False):
    from huggingface_hub import snapshot_download

    specs = target_registry() if targets else PROXIES
    for name, spec in specs.items():
        print("Downloading pinned checkpoint:", name, flush=True)
        snapshot_download(
            spec["repo_id"],
            revision=spec["revision"],
            cache_dir=str(args.cache_dir),
            allow_patterns=["*.json", "*.jinja", "*.txt", "*.model", "*.safetensors"]
            if targets
            else [
                "config.json",
                "preprocessor_config.json",
                "tokenizer*",
                "vocab.json",
                "merges.txt",
                "special_tokens_map.json",
                "model.safetensors"
                if spec["repo_id"].startswith("laion/")
                else "pytorch_model.bin",
            ],
            max_workers=2,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage",
        choices=[
            "prepare",
            "preflight",
            "download",
            "download-targets",
            "smoke",
            "pilot",
            "attacks",
            "evaluate",
            "report",
            "run-all",
        ],
    )
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--ours-root", type=Path, default=MAIN / "outputs/a100_precommercial_v8_joint_selection24h"
    )
    p.add_argument(
        "--data-root", type=Path, default=MAIN / "outputs/a100_precommercial_v7_targeted50"
    )
    p.add_argument("--cache-dir", type=Path, default=Path("/ephemeral/kdd2027/hf-cache/hub"))
    p.add_argument("--upstream-root", type=Path, default=Path("/home/ubuntu/baseline-upstreams"))
    p.add_argument(
        "--split", choices=["smoke", "screen", "explore", "confirm", "final150"], default="final150"
    )
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--methods", nargs="+", default=["pgd", "mi_fgsm", "m_attack", "foa_attack"])
    p.add_argument("--proxies", nargs="+", choices=list(PROXIES), default=list(PROXIES))
    args = p.parse_args()
    if args.steps <= 0:
        p.error("steps must be positive")
    for method in args.methods:
        validate_method(method)
    os.environ["PRIMARY_ML_CKA_STRICT_BF16"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.stage in {"download", "download-targets"}:
        download(args, targets=args.stage == "download-targets")
        return 0
    if args.stage == "smoke":
        args.split, args.steps = "smoke", 2
    elif args.stage == "pilot":
        args.split = "screen"
    manifest, run, lock = prepare(args)
    if args.stage == "prepare":
        print(run)
        return 0
    if args.stage == "report":
        return 0 if report(args, manifest, run, lock) else 2
    state = preflight(args, manifest, run)
    if args.stage == "preflight":
        return 0 if state["attack_ready"] and state["evaluation_ready"] else 2
    if not state["attack_ready"]:
        report(args, manifest, run, lock)
        return 2
    ok = True
    try:
        with gpu_owner(args.output):
            if args.stage in {"smoke", "pilot", "attacks", "run-all"}:
                ok = attack(args, manifest, run, lock)
            if args.stage in {"evaluate", "run-all"} and ok:
                ok = evaluate(args, manifest, run, lock)
    except Exception as e:
        atomic(run / "stage_failure.json", {"error": repr(e), "traceback": traceback.format_exc()})
        traceback.print_exc()
        ok = False
    complete = report(args, manifest, run, lock)
    # Attack-only modes truthfully report only the attack phase as finished.
    return 0 if ok and (complete or args.stage in {"smoke", "pilot", "attacks"}) else 2


if __name__ == "__main__":
    raise SystemExit(main())
