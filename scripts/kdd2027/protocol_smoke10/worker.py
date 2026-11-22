# ruff: noqa: E402, E501
"""Isolated scientific adapters, sharing the existing exclusive GPU owner lock."""

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
import zipfile
from pathlib import Path

from prepare import BASE, OUT, RAW, REPO

from primary_ml_cka.baselines.protocol import atomic, digest, freeze, sha, valid_receipt

NATIVE = REPO / "scripts/kdd2027/native_tasks"
sys.path.insert(0, str(NATIVE))
# prepare here is this study's module. Load the old worker with its original
# preparation module without registering either as the shared top-level worker.
spec = importlib.util.spec_from_file_location("native_prepare_locked", NATIVE / "prepare.py")
np = importlib.util.module_from_spec(spec)
spec.loader.exec_module(np)
current_prepare = sys.modules["prepare"]
sys.modules["prepare"] = np
spec = importlib.util.spec_from_file_location("native_worker_locked", NATIVE / "worker.py")
nw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nw)
sys.modules["prepare"] = current_prepare
w = np.expansion.api()


def setup(method, dataset, family):
    if dataset == "crossvlad":
        data = json.loads((OUT / "crossvlad_manifest.json").read_text())
    else:
        original = np.manifests()["llava_bench_coco"]
        data = {
            **original,
            "images": original["images"][:10],
            "N_images": 10,
            "N_questions": 30,
            "subset_note": "fixed original manifest first10 images, all3 questions",
        }
    cfg = {
        "method": method,
        "dataset": data["dataset"],
        "variant": family or "default",
        "N": 10,
        "seed": 42,
        "epsilon": 16 / 255,
        "alpha": 1 / 255,
        "steps": 50 if method == "ours" else 10 if method == "ssa_cwa_clip" else 300,
        "parameter_dtype": "bfloat16",
        "smoke_only": False,
        "pilot": False,
        "reference_count": 48 if method == "ours" else 0 if method == "ssa_cwa_clip" else 1,
        "pull": 0.5,
        "push": 0.5,
        "DI": method == "ours",
        "CLS": 0,
        "patch": 0,
        "momentum": 1,
        "engineering": False,
        "budget_note": "Ours50; M/FOA native300; SSA-CWA native10 with20 spectrum samples; NOT compute matched",
    }
    sources = {
        str(p.relative_to(REPO)): sha(p)
        for p in [
            Path(__file__),
            Path(__file__).with_name("prepare.py"),
            NATIVE / "worker.py",
            NATIVE / "prepare.py",
            REPO / "scripts/kdd2027/nov21/methods.py",
            REPO / "scripts/kdd2027/extension/worker.py",
        ]
    }
    lock = {
        "config": cfg,
        "manifest": data,
        "scientific_sources": sources,
        "targets": w.oldworker.REG,
        "legacy_sources": w.legacy.source_hashes(),
        "reference_banks_sha256": sha(OUT / "reference_banks.json")
        if dataset == "crossvlad"
        else sha(np.OUT / "reference_banks.json"),
    }
    if method == "ours":
        lock["models"] = json.loads((OUT / "proxy_revisions.json").read_text())
        lock["target"] = nw.FAMILIES[family]
        if any(lock["models"][n]["family"] == family for n in nw.POOLS[family]):
            raise ValueError("Target-family leakage")
    run = OUT / "runs" / f'{dataset}_{method}_{family or "all"}_{digest(lock)[:14]}'
    freeze(run / "lock.json", lock)
    return run, cfg, lock, data


def reuse_llava_ours(run, lock, data):
    """Reuse only exact original inputs, sources, model/loss/banks; no relabeling."""
    family = lock["config"]["variant"]
    candidates = []
    for file in (np.OUT / "runs").glob("llava_bench_coco_" + family + "_30_*/lock.json"):
        old = json.loads(file.read_text())
        if (
            old["config"]["steps"] != 50
            or old["config"]["pull"] != 0.5
            or old["config"]["push"] != 0.5
        ):
            continue
        if old["sources"].get(str((NATIVE / "worker.py").relative_to(REPO))) != sha(
            NATIVE / "worker.py"
        ):
            continue
        if old["reference_banks_sha256"] != lock["reference_banks_sha256"]:
            continue
        if old["models"] != lock["models"]:
            continue
        oldrows = {r["image_id"]: r for r in old["manifest"]["images"]}
        if all(oldrows.get(r["image_id"]) == r for r in data["images"]):
            candidates.append((file.parent, old))
    if not candidates:
        raise RuntimeError(
            "No exact frozen LLaVA Ours reuse; do not silently regenerate a different method"
        )
    oldrun, old = sorted(candidates, key=lambda x: str(x[0]))[0]
    target = nw.FAMILIES[family]
    for row in data["images"]:
        source = w.oldworker.receipt_path(oldrun, row)
        if not valid_receipt(source, digest([old, row])):
            raise RuntimeError("Invalid original Ours PNG")
        value = json.loads(source.read_text())
        value.update(
            signature=digest([lock, row]),
            reused_receipt=str(source),
            reused_receipt_sha256=sha(source),
        )
        dest = w.oldworker.receipt_path(run, row)
        freeze(dest, value)
        for q in row["queries"]:
            name = digest([row["image_id"], q["id"]])[:18] + ".json"
            src = oldrun / "evaluations" / target / name
            if not w.oldworker.valid_evaluation(src, oldrun, row, q, w.oldworker.REG[target]):
                raise RuntimeError("Invalid original Ours evaluation")
            result = json.loads(src.read_text())
            result.update(
                receipt_sha256=sha(dest),
                reused_evaluation=str(src),
                reused_evaluation_sha256=sha(src),
            )
            freeze(run / "evaluations" / target / name, result)
    print("REUSED LLaVA Ours", family, "10 PNGs /30 answer pairs", flush=True)


def coco_categories():
    with zipfile.ZipFile(RAW / "annotations_trainval2017.zip") as z:
        anno = json.loads(z.read("annotations/instances_val2017.json"))
    return [r["name"] for r in sorted(anno["categories"], key=lambda r: r["id"])], anno


def ssa_attack(run, cfg, lock, data):
    import torch

    sys.path.insert(0, str(REPO / "scripts/kdd2027/nov21"))
    import methods

    classes, anno = coco_categories()
    # Same native SSA-CWA optimizer and CLIP zero-shot head; change only the
    # explicitly disclosed task vocabulary from ImageNet10 to COCO80.
    methods.CLASSES = classes
    bundle, provenance = methods.make("ssa_cwa_clip", cfg["steps"])
    atomic(
        run / "algorithm.json",
        {
            **provenance,
            "fidelity": "official SSA-CWA core / COCO80 CLIP classification adaptation",
            "classes": classes,
            "no_reference_images": True,
        },
    )
    atomic(run / "bf16_proxies.json", [m.audit for m in bundle[0]])
    target_labels = {}
    for a in anno["annotations"]:
        item = target_labels.get(a["image_id"])
        if not a.get("iscrowd") and (item is None or a["area"] > item["area"]):
            target_labels[a["image_id"]] = a
    cats = {r["id"]: r["name"] for r in anno["categories"]}
    for row in data["images"]:
        dest = w.oldworker.receipt_path(run, row)
        if valid_receipt(dest, digest([lock, row])):
            continue
        category = row.get("target_category")
        if category is None:
            category = cats[target_labels[int(Path(row["target_reference"]).stem)]["category_id"]]
        start = time.monotonic()
        clean = w.legacy.load_image(row["clean"])
        seed = int(digest([42, row["image_id"], cfg["method"]])[:8], 16)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
        adv, trace = methods.generate(
            "ssa_cwa_clip",
            bundle,
            clean,
            None,
            {"target": classes.index(category) + 1},
            cfg["steps"],
        )
        if not torch.isfinite(adv).all() or (adv - clean).abs().max() > 16 / 255 + 1e-7:
            raise ValueError("Invalid SSA-CWA perturbation")
        w.legacy.save_png(dest.parent / "adv.png", adv)
        linf = float((w.legacy.load_image(dest.parent / "adv.png") - clean).abs().max())
        if linf > 16 / 255 + 1e-7:
            raise ValueError("PNG bound")
        atomic(dest.parent / "optimization.json", trace)
        atomic(
            dest,
            {
                "status": "ATTACK_FROZEN",
                "signature": digest([lock, row]),
                "image_id": row["image_id"],
                "clean": row["clean"],
                "adv": str(dest.parent / "adv.png"),
                "png_hashes": {
                    row["clean"]: sha(row["clean"]),
                    str(dest.parent / "adv.png"): sha(dest.parent / "adv.png"),
                },
                "linf_png": linf,
                "steps": cfg["steps"],
                "target_category": category,
                "target_access": False,
                "seconds": time.monotonic() - start,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "seed": seed,
            },
        )
        print("FROZEN SSA-CWA", row["image_id"], flush=True)
    del bundle
    gc.collect()
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["attack", "evaluate", "reuse"])
    p.add_argument("--dataset", choices=["crossvlad", "llava"], required=True)
    p.add_argument(
        "--method", choices=["ours", "m_attack", "foa_attack", "ssa_cwa_clip"], required=True
    )
    p.add_argument("--family", choices=nw.FAMILIES)
    a = p.parse_args()
    if a.method == "ours" and not a.family:
        p.error("Ours requires a legal target family")
    os.environ.update(PRIMARY_ML_CKA_STRICT_BF16="1", HF_HOME=str(np.CACHE.parent))
    run, cfg, lock, data = setup(a.method, a.dataset, a.family)
    print("RUN", run, flush=True)
    if a.dataset == "llava" and a.method == "ours":
        reuse_llava_ours(run, lock, data)
        return
    if a.phase == "reuse":
        return
    with w.legacy.gpu_owner(BASE):
        if a.phase == "attack":
            if a.method == "ours":
                nw.OUT = OUT
                sys.path.insert(0, str(nw.V7 / "src"))
                nw.attack(run, cfg, lock, data)
            elif a.method == "ssa_cwa_clip":
                ssa_attack(run, cfg, lock, data)
            else:
                w.oldworker.attacks(cfg, run, lock, data)
        else:
            if a.method == "ours":
                w.oldworker.allowed_targets = lambda c: [nw.FAMILIES[a.family]]
            w.oldworker.evaluations(cfg, run, lock, data)
        if not all(
            valid_receipt(w.oldworker.receipt_path(run, row), digest([lock, row]))
            for row in data["images"]
        ):
            raise RuntimeError("Not all10 attacks have validated frozen receipts")
        if a.phase == "evaluate":
            targets = [lock["target"]] if a.method == "ours" else list(lock["targets"])
            for target in targets:
                for row in data["images"]:
                    for q in row["queries"]:
                        name = digest([row["image_id"], q["id"]])[:18] + ".json"
                        if not w.oldworker.valid_evaluation(
                            run / "evaluations" / target / name,
                            run,
                            row,
                            q,
                            lock["targets"][target],
                        ):
                            raise RuntimeError("Incomplete/invalid target evaluations")


if __name__ == "__main__":
    main()
