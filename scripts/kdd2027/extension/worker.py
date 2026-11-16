"""One frozen method/dataset job. No target inference during image optimization."""

import argparse
import gc
import json
import os
import time
import traceback
from pathlib import Path

import native_methods as native
import torch
from task_protocol import DEBUG_INPUTS, TARGETS, condition, eval_rows, identity

from primary_ml_cka.baselines import campaign as legacy
from primary_ml_cka.baselines.clip_features import ClipFeatures, audit_model
from primary_ml_cka.baselines.feature_attacks import run_feature_attack
from primary_ml_cka.baselines.protocol import atomic, digest, freeze, sha, valid_receipt

OUT = Path("/home/ubuntu/exper-tran/outputs/kdd2027_baseline_reproduction")
CACHE = Path("/ephemeral/kdd2027/hf-cache/hub")
UPSTREAM = Path("/home/ubuntu/baseline-upstreams")
REG = legacy.target_registry()
LEGACY = {50: "final150_50_206015cb1202ed50f1", 300: "final150_300_87488fbb38aae77f36"}


def lock_job(config):
    manifest_path = OUT / "extension/manifests" / (config["dataset"] + ".json")
    manifest = json.loads(manifest_path.read_text())
    sources = {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")}
    sources["legacy"] = legacy.source_hashes()
    lock = {
        "config": config,
        "sources": sources,
        "manifest_sha256": sha(manifest_path),
        "engineering_manifest_sha256": sha(DEBUG_INPUTS)
        if config["smoke_only"] or config.get("pilot")
        else None,
        "models": {**native.PROXIES, **native.EXTRAS},
        "targets": REG,
    }
    run = (
        OUT
        / "extension/runs"
        / (config["dataset"] + "_" + config["method"] + "_" + identity(config, sources))
    )
    freeze(run / "lock.json", lock)
    return run, lock, manifest


def rows_for(config, manifest):
    return eval_rows(config, manifest)


def sig(lock, row):
    return digest([lock, row])


def receipt_path(run, row):
    return run / "attacks" / digest(row["image_id"])[:18] / "receipt.json"


def import_legacy(config, run, lock, row):
    if config["method"] == "foa_attack":
        return False  # Different, explicitly pinned KMeans dependency.
    if (
        config["steps"] not in LEGACY
        or config["dataset"] != "imagenet"
        or config["method"]
        not in [
            "pgd",
            "mi_fgsm",
            "m_attack",
            "foa_attack",
        ]
    ):
        return False
    oldroot = OUT / "runs" / LEGACY[config["steps"]]
    p = oldroot / "attacks" / config["method"] / digest(row["image_id"])[:18] / "receipt.json"
    if not p.exists():
        return False
    old = json.loads(p.read_text())
    if not valid_receipt(p, old["signature"]):
        return False
    oldlock = json.loads((oldroot / "method_lock.json").read_text())
    original_rows = json.loads((OUT / "manifests/final150.json").read_text())["images"]
    original = next(r for r in original_rows if r["image_id"] == row["image_id"])
    if old["signature"] != legacy.signature(oldlock, original, config["method"]):
        raise RuntimeError("Legacy attack signature mismatch")
    if legacy.source_hashes() != oldlock["sources"] or old["steps"] != config["steps"]:
        raise RuntimeError("Legacy source/budget changed")
    for key in ["clean_sha256", "target_reference_sha256"]:
        if original[key] != row[key]:
            raise RuntimeError("Legacy source/reference mismatch")
    atomic(
        receipt_path(run, row),
        {
            **old,
            "signature": sig(lock, row),
            "reuse_receipt": str(p),
            "reuse_receipt_sha256": sha(p),
        },
    )
    return True


def make_bundle(config):
    method = config["method"]
    if method in ["iau_foa", "co_attack", "sga"]:
        names = ["B16", "B32", "Laion", "DINO"] if config["variant"] == "family_disjoint4" else None
        return native.make_bundle(method, CACHE, UPSTREAM, names)
    if method == "foa_attack":
        import primary_ml_cka.baselines.feature_attacks as feature_core

        feature_core.bounded_kmeans = native.pinned_kmeans(UPSTREAM)
    models = [ClipFeatures(n, CACHE) for n in ["B16", "B32", "Laion"]]
    return type("Bundle", (), {"models": models, "audits": [m.audit for m in models]})()


def attacks(config, run, lock, manifest):
    rows = rows_for(config, manifest)
    pending = []
    for row in rows:
        p = receipt_path(run, row)
        if valid_receipt(p, sig(lock, row)) or import_legacy(config, run, lock, row):
            continue
        if not (p.parent / "failure.json").exists():
            pending.append(row)
    if not pending:
        return
    bundle = make_bundle(config)
    atomic(run / "bf16_proxies.json", bundle.audits)
    aux_counts = [0]
    if hasattr(bundle, "mlm"):

        def count_mlm(module, inputs, output):
            aux_counts[0] += 1

        bundle.mlm.register_forward_hook(count_mlm)
    for index, row in enumerate(pending):
        dest = receipt_path(run, row).parent
        began = time.monotonic()
        try:
            for path, h in [
                (row["clean"], row["clean_sha256"]),
                (row["target_reference"], row["target_reference_sha256"]),
            ]:
                if sha(path) != h:
                    raise RuntimeError("Frozen input changed")
            clean, reference = (
                legacy.load_image(row["clean"]),
                legacy.load_image(row["target_reference"]),
            )
            seed = int(digest([42, row["image_id"], config["method"]])[:8], 16)
            torch.cuda.reset_peak_memory_stats()
            counts = [m.forwards for m in bundle.models]
            text_before = getattr(getattr(bundle, "net", None), "text_forwards", 0)
            mlm_before = aux_counts[0]
            if config["method"] in ["iau_foa", "co_attack", "sga"]:
                adv, trace = native.run(
                    bundle, clean, reference, row["captions"], steps=config["steps"], seed=seed
                )
            else:
                adv, trace = run_feature_attack(
                    config["method"],
                    bundle.models,
                    clean,
                    reference,
                    steps=config["steps"],
                    seed=seed,
                    upstream_root=UPSTREAM,
                )
            if config["method"] == "foa_attack":
                trace["kmeans_revision"] = native.KMEANS_PIN
                trace["foa_kmeans_api_repair"] = False
            if not torch.isfinite(adv).all() or (adv - clean).abs().max() > 16 / 255 + 1e-7:
                raise RuntimeError("Nonfinite/budget-invalid adversarial tensor")
            legacy.save_png(dest / "adv.png", adv)
            linf = float((legacy.load_image(dest / "adv.png") - clean).abs().max())
            if linf > 16 / 255 + 1e-7:
                raise RuntimeError("Saved PNG violates epsilon")
            atomic(dest / "optimization.json", trace)
            atomic(
                receipt_path(run, row),
                {
                    "status": "ATTACK_FROZEN",
                    "signature": sig(lock, row),
                    "image_id": row["image_id"],
                    "clean": row["clean"],
                    "adv": str(dest / "adv.png"),
                    "png_hashes": {
                        row["clean"]: row["clean_sha256"],
                        str(dest / "adv.png"): sha(dest / "adv.png"),
                    },
                    "linf_png": linf,
                    "steps": config["steps"],
                    "seed": seed,
                    "target_access": False,
                    "text_encoder_forward_calls": getattr(
                        getattr(bundle, "net", None), "text_forwards", 0
                    )
                    - text_before,
                    "MLM_full_forward_calls": aux_counts[0] - mlm_before,
                    "forward_counter_scope": (
                        "visual_model_invocations; " "text/MLM separately counted"
                    ),
                    "forward_calls": sum(
                        m.forwards - c for m, c in zip(bundle.models, counts, strict=True)
                    ),
                    "seconds": time.monotonic() - began,
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "config": config,
                },
            )
            print(
                config["dataset"],
                config["method"],
                f"{index+1}/{len(pending)}",
                round(time.monotonic() - began, 1),
                "seconds",
                flush=True,
            )
        except Exception as e:
            atomic(
                dest / "failure.json",
                {
                    "image_id": row["image_id"],
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                    "config": config,
                },
            )
            print("FAILED", row["image_id"], repr(e), flush=True)
            if isinstance(e, torch.OutOfMemoryError):
                raise
    del bundle
    gc.collect()
    torch.cuda.empty_cache()


def allowed_targets(config):
    if config["method"] != "iau_foa":
        return TARGETS
    names = ["B16", "B32", "Laion", "DINO"]
    if config["variant"] != "family_disjoint4":
        names += ["InternVL1"]
        targets = [t for t in TARGETS if REG[t]["family"] != "internvl"]
    else:
        targets = [t for t in TARGETS if REG[t]["family"] == "internvl"]
    for t in targets:
        native.family_legal(names, REG[t]["family"])
    return targets


def generate(model, processor, path, prompt, limit):
    from PIL import Image

    from primary_ml_cka.prompts.chat_templates import classification_messages, render_chat_template

    rendered = render_chat_template(
        processor, classification_messages(prompt).prompt_only, add_generation_prompt=True
    )
    with Image.open(path) as image:
        image = image.convert("RGB")
    images = [[image]] if processor.__class__.__name__ == "Gemma4Processor" else [image]
    inputs = processor(text=[rendered], images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.inference_mode():
        tokens = model.generate(**inputs, do_sample=False, max_new_tokens=limit, use_cache=True)
    new = tokens[:, inputs["input_ids"].shape[1] :]
    return {
        "text": processor.batch_decode(new, skip_special_tokens=True)[0],
        "new_tokens": int(new.shape[1]),
        "max_new_tokens": limit,
        "possibly_truncated": new.shape[1] >= limit,
    }


def import_legacy_evaluation(config, run, lock, row, q, name):
    receipt = json.loads(receipt_path(run, row).read_text())
    if "reuse_receipt" not in receipt or q["kind"] != "classification":
        return False
    original_path = Path(receipt["reuse_receipt"])
    original = json.loads(original_path.read_text())
    oldrun = original_path.parents[3]
    path = (
        oldrun / "evaluations" / name / config["method"] / (digest(row["image_id"])[:18] + ".json")
    )
    if not path.exists():
        return False
    data = json.loads(path.read_text())
    expected = digest(
        [original["signature"], original["png_hashes"], REG[name], legacy.canonical_prompt()]
    )
    if data["signature"] != expected or data["image_id"] != row["image_id"]:
        raise RuntimeError("Reused target evaluation signature differs")
    atomic(
        run / "evaluations" / name / (digest([row["image_id"], q["id"]])[:18] + ".json"),
        {
            "image_id": row["image_id"],
            "query": q,
            "model": REG[name],
            "prompt": legacy.canonical_prompt(),
            "clean": {"text": data["clean"]["raw"], "label": data["clean"]["label"]},
            "adv": {"text": data["adv_raw"], "label": data["adv_label"]},
            "target_hit": data["target_hit"],
            "receipt_sha256": sha(receipt_path(run, row)),
            "reused_evaluation": str(path),
            "reused_evaluation_sha256": sha(path),
            "original_question_unmodified": True,
        },
    )
    return True


def valid_evaluation(path, run, row, q, spec):
    try:
        value = json.loads(path.read_text())
        return (
            value["image_id"] == row["image_id"]
            and value["query"] == q
            and value["model"] == spec
            and value["receipt_sha256"] == sha(receipt_path(run, row))
            and isinstance(value["clean"]["text"], str)
            and isinstance(value["adv"]["text"], str)
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def evaluations(config, run, lock, manifest):
    from primary_ml_cka.models.backends.target_transformers_generation import (
        load_target_for_generation,
    )
    from primary_ml_cka.models.backends.transformers_backend import load_processor
    from primary_ml_cka.models.targets.generation import TransformersTargetGenerator

    rows = rows_for(config, manifest)
    valid = [r for r in rows if valid_receipt(receipt_path(run, r), sig(lock, r))]
    if any(
        not valid_receipt(receipt_path(run, r), sig(lock, r))
        and not (receipt_path(run, r).parent / "failure.json").exists()
        for r in rows
    ):
        raise RuntimeError("Target evaluation before all images frozen/attempted")
    freeze(
        run / "evaluation_plan.json",
        {
            "valid_ids": [r["image_id"] for r in valid],
            "planned_N": len(rows),
            "missing_ids": [r["image_id"] for r in rows if r not in valid],
            "denominator_not_shrunk": True,
            "targets": allowed_targets(config),
        },
    )
    if not valid:
        return
    for name in allowed_targets(config):
        for row in valid:
            for q in row["queries"]:
                import_legacy_evaluation(config, run, lock, row, q, name)
        spec = REG[name]
        pending = [
            (r, q)
            for r in valid
            for q in r["queries"]
            if not valid_evaluation(
                run / "evaluations" / name / (digest([r["image_id"], q["id"]])[:18] + ".json"),
                run,
                r,
                q,
                spec,
            )
        ]
        if not pending:
            continue
        snapshot = (
            CACHE
            / ("models--" + spec["repo_id"].replace("/", "--"))
            / "snapshots"
            / spec["revision"]
        )
        processor = load_processor(snapshot)
        model = load_target_for_generation(snapshot, torch.device("cuda"), precision="bf16")
        atomic(run / "audits" / (digest(name)[:12] + ".json"), audit_model(model, spec, "target"))
        label_gen = TransformersTargetGenerator(model, processor)
        for row, q in pending:
            receipt = json.loads(receipt_path(run, row).read_text())
            prompt = legacy.canonical_prompt() if q["kind"] == "classification" else q["text"]
            limit = (
                4 if q["kind"] == "classification" else 128 if q["kind"] == "captioning" else 1024
            )
            key = digest([spec, prompt, row["clean_sha256"], limit])
            cleanpath = OUT / "extension/clean_predictions" / (key + ".json")
            if cleanpath.exists():
                clean = json.loads(cleanpath.read_text())
            else:
                if q["kind"] == "classification":
                    with torch.inference_mode():
                        response = label_gen.generate_label(Path(row["clean"]), prompt)
                    clean = {"text": response.raw_output, "label": response.parsed_label}
                else:
                    clean = generate(model, processor, row["clean"], prompt, limit)
                atomic(cleanpath, clean)
            if q["kind"] == "classification":
                with torch.inference_mode():
                    response = label_gen.generate_label(Path(receipt["adv"]), prompt)
                adv = {"text": response.raw_output, "label": response.parsed_label}
            else:
                adv = generate(model, processor, receipt["adv"], prompt, limit)
            dest = run / "evaluations" / name / (digest([row["image_id"], q["id"]])[:18] + ".json")
            if dest.exists():
                dest.rename(dest.with_suffix(".json.corrupt"))
            atomic(
                dest,
                {
                    "image_id": row["image_id"],
                    "query": q,
                    "prompt": prompt,
                    "clean": clean,
                    "adv": adv,
                    "receipt_sha256": sha(receipt_path(run, row)),
                    "model": spec,
                    "target_hit": adv.get("label") == row.get("target")
                    if q["kind"] == "classification"
                    else None,
                    "source_label": row.get("source"),
                    "target_label": row.get("target"),
                    "valid_untargeted_prediction_change": adv.get("label") is not None
                    and clean.get("label") is not None
                    and adv["label"] != clean["label"]
                    if q["kind"] == "classification"
                    else None,
                    "original_question_unmodified": True,
                },
            )
            print("EVAL", name, config["method"], q["id"], flush=True)
        del label_gen, model, processor
        gc.collect()
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--variant", default="default")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--phase", choices=["all", "attack", "evaluate"], default="all")
    args = p.parse_args()
    os.environ["PRIMARY_ML_CKA_STRICT_BF16"] = "1"
    config = condition(args.method, args.dataset, args.variant, args.smoke)
    config["pilot"] = args.pilot
    run, lock, manifest = lock_job(config)
    print("RUN", run, flush=True)
    with legacy.gpu_owner(OUT):
        if args.phase in ["all", "attack"]:
            attacks(config, run, lock, manifest)
        if args.smoke and any(
            not valid_receipt(receipt_path(run, r), sig(lock, r))
            for r in rows_for(config, manifest)
        ):
            raise RuntimeError("GPU smoke incomplete; do not admit full method")
        if args.phase in ["all", "evaluate"]:
            evaluations(config, run, lock, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
