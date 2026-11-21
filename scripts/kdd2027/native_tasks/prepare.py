# ruff: noqa: E402, E741
"""Frozen label-free reference neighborhoods for native caption/VQA tasks."""

import concurrent.futures
import json
import os
import random
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "scripts/kdd2027/nov21"))
import dataset_expansion as expansion

from primary_ml_cka.baselines.protocol import atomic, freeze, sha

BASE = expansion.BASE
OUT = BASE / "native_tasks_ours_first"
CACHE = Path("/ephemeral/kdd2027/hf-cache/hub")
V7 = REPO / "experiments/2026-10-a100-precommercial-v7-targeted-50"
NAMES = ["CLIP ViT-L/14", "Qwen3.5-2B", "InternVL3.5-2B-HF", "Gemma 4 E2B-it"]


def manifests():
    return {
        ds: json.loads((BASE / "extension/manifests" / f"{ds}.json").read_text())
        for ds in ["coco_captioning", "llava_bench_coco"]
    }


def fetch(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        partial = dest.with_suffix(dest.suffix + ".part")
        subprocess.run(
            [
                "curl",
                "-fLsS",
                "--retry",
                "3",
                "--connect-timeout",
                "30",
                "--max-time",
                "600",
                "-o",
                str(partial),
                url,
            ],
            check=True,
        )
        os.replace(partial, dest)


def prepare():
    from huggingface_hub import snapshot_download

    registry = json.loads(
        (V7.parent / "2026-10-a100-precommercial-v6/model_revisions.json").read_text()
    )
    for name in NAMES:
        spec = registry[name]
        snapshot_download(
            spec["repo_id"],
            revision=spec["revision"],
            cache_dir=str(CACHE),
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.jinja",
                "*.model",
                "*.safetensors",
                "pytorch_model.bin",
            ],
            max_workers=2,
        )
    freeze(OUT / "proxy_revisions.json", {n: registry[n] for n in NAMES})
    data = manifests()
    rows = [r for m in data.values() for r in m["images"]]
    excluded = {int(r["image_id"].split("/")[-1]) for r in rows}
    pool = {}
    for r in rows:
        p = Path(r["target_reference"])
        identity = int(p.stem)
        assert identity not in excluded
        pool[identity] = {"image_id": identity, "path": str(p), "sha256": sha(p)}
    archive = Path("/ephemeral/kdd2027/datasets/annotations_trainval2017.zip")
    fetch("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", archive)
    with zipfile.ZipFile(archive) as z:
        captions = json.loads(z.read("annotations/captions_val2017.json"))
    candidates = sorted(
        [r for r in captions["images"] if r["id"] not in excluded | set(pool)],
        key=lambda r: r["id"],
    )
    random.Random(42).shuffle(candidates)
    chosen = candidates[: 512 - len(pool)]

    def one(r):
        from PIL import Image

        raw = Path("/ephemeral/kdd2027/datasets/ours_refs") / r["file_name"]
        dest = OUT / "reference_images" / Path(r["file_name"]).with_suffix(".png")
        fetch("http://images.cocodataset.org/val2017/" + r["file_name"], raw)
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(raw) as im:
                im.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC).save(dest)
        return r["id"], {"image_id": r["id"], "path": str(dest), "sha256": sha(dest)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        for identity, r in executor.map(one, chosen):
            pool[identity] = r
    assert len(pool) == 512 and not set(pool) & excluded
    freeze(
        OUT / "reference_pool.json",
        {
            "seed": 42,
            "N": 512,
            "excluded_attack_ids": sorted(excluded),
            "images": [pool[k] for k in sorted(pool)],
            "annotation_sha256": sha(archive),
        },
    )
    atomic(
        OUT / "preparation.json",
        {
            "status": "PASS",
            "references_disjoint_from_all_test_images": True,
            "reference_count": 512,
            "target_feedback": False,
        },
    )


def choose_banks(source, target, pool, k=48):
    """Target-first disjoint neighborhoods, all scores surrogate-only."""
    import numpy as np

    target_ids = np.argsort(-(pool @ target), kind="stable")[:k]
    banned = set(target_ids.tolist())
    source_ids = [
        int(i) for i in np.argsort(-(pool @ source), kind="stable") if int(i) not in banned
    ][:k]
    if len(source_ids) != k or len(target_ids) != k:
        raise ValueError("Reference pool too small")
    return source_ids, target_ids.tolist()


def select():
    import numpy as np
    import torch

    from primary_ml_cka.baselines import campaign as legacy
    from primary_ml_cka.baselines.clip_features import ClipFeatures

    pool = json.loads((OUT / "reference_pool.json").read_text())
    destination = OUT / "reference_banks.json"
    if destination.exists():
        return
    model = ClipFeatures("B32", CACHE)
    atomic(OUT / "selector_audit.json", model.audit)
    vectors = []
    with torch.inference_mode():
        for r in pool["images"]:
            assert sha(r["path"]) == r["sha256"]
            vectors.append(model(legacy.load_image(r["path"]) * 255).cpu().numpy()[0])
        vectors = np.stack(vectors)
        banks = {}
        for manifest in manifests().values():
            for r in manifest["images"]:
                a = model(legacy.load_image(r["clean"]) * 255).cpu().numpy()[0]
                b = model(legacy.load_image(r["target_reference"]) * 255).cpu().numpy()[0]
                source, target = choose_banks(a, b, vectors)
                banks[r["image_id"]] = {
                    "source_indices": source,
                    "target_indices": target,
                    "clean_sha256": r["clean_sha256"],
                    "target_reference_sha256": r["target_reference_sha256"],
                }
    freeze(
        destination,
        {
            "version": "label_free_neighborhood_PP48_v1",
            "k": 48,
            "selector": model.spec,
            "pool_sha256": sha(OUT / "reference_pool.json"),
            "selection": "fixed CLIP-B32 nearest neighbors, target first; disjoint banks",
            "target_model_feedback": False,
            "banks": banks,
        },
    )


if __name__ == "__main__":
    os.environ["PRIMARY_ML_CKA_STRICT_BF16"] = "1"
    if len(sys.argv) > 1 and sys.argv[1] == "select":
        w = expansion.api()
        with w.legacy.gpu_owner(BASE):
            select()
    else:
        prepare()
