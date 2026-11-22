# ruff: noqa: E501
"""CPU-only, official CrossVLAD labels; five predeclared directions x two images.

Reference sampling uses COCO annotations only, never a model prediction. Existing
studies are read-only. Raw downloads are disposable; manifests/PNGs are persistent.
"""

import concurrent.futures
import io
import json
import random
import re
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image

from primary_ml_cka.baselines.protocol import atomic, freeze, sha

REPO = Path(__file__).resolve().parents[3]
BASE = Path("/home/ubuntu/exper-tran/outputs/kdd2027_baseline_reproduction")
OUT = BASE / "protocol_smoke10_v2"
RAW = Path("/ephemeral/kdd2027/datasets")
PAIRS = [
    ("bicycle", "motorcycle"),
    ("car", "bus"),
    ("cat", "dog"),
    ("chair", "couch"),
    ("tv", "laptop"),
]
CRAFT_REV = "0f457d73b049d62111d39c69d893f2f7d10574b9"
LLAVA_REV = "c121f0432da27facab705978f83c4ada465e46fd"
ALIASES = {
    "bicycle": ["bicycle", "bicycles", "bike", "bikes"],
    "car": ["car", "cars"],
    "cat": ["cat", "cats", "kitten", "kittens"],
    "chair": ["chair", "chairs"],
    "tv": ["tv", "television", "monitor", "screen"],
}


def label_eligible(row):
    mentions = sum(
        any(
            re.search(r"\b" + word + r"\b", caption.lower())
            for word in ALIASES[row["source_object_category"]]
        )
        for caption in row["captions"]
    )
    areas = [
        b[2] * b[3] / (row["image_width"] * row["image_height"]) for b in row["source_object_boxes"]
    ]
    return mentions >= 3 and any(0.1 <= area <= 0.5 for area in areas)


def fetch(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        part = dest.with_suffix(dest.suffix + ".partial")
        subprocess.run(
            ["curl", "-fLsS", "--retry", "3", "--max-time", "180", "-o", str(part), url], check=True
        )
        part.replace(dest)
    return dest


def canvas(data, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        with Image.open(data) as image:
            image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC).save(dest)
    return {"path": str(dest), "sha256": sha(dest)}


def sample_labels(labels):
    chosen = []
    for source, target in PAIRS:
        candidates = sorted(
            [
                r
                for r in labels
                if (r["source_object_category"], r["target_object_category"]) == (source, target)
                and label_eligible(r)
            ],
            key=lambda r: r["image_id"],
        )
        if len(candidates) < 2:
            raise ValueError(f"Missing official direction {source}->{target}")
        random.Random(f"42:{source}:{target}").shuffle(candidates)
        chosen.extend(candidates[:2])
    if len({r["image_id"] for r in chosen}) != 10:
        raise ValueError("Duplicate source IDs")
    return chosen


def prepare():
    archive = RAW / "crossvlad/dataset.zip.part"
    if not zipfile.is_zipfile(archive):
        raise RuntimeError("Official CrossVLAD archive unavailable; no synthesized replacement")
    with zipfile.ZipFile(archive) as z:
        labels = [json.loads(z.read(n)) for n in sorted(z.namelist()) if n.endswith(".json")]
        source_ids = {r["image_id"] for r in labels}
        gt = defaultdict(set)
        with zipfile.ZipFile(RAW / "annotations_trainval2017.zip") as annotations:
            for split in ["val2017", "train2017"]:
                annotation = json.loads(annotations.read(f"annotations/instances_{split}.json"))
                classes = {r["id"]: r["name"] for r in annotation["categories"]}
                for a in annotation["annotations"]:
                    if a["image_id"] in source_ids:
                        gt[a["image_id"]].add(classes[a["category_id"]])
                del annotation
        gt_eligible = [
            r
            for r in labels
            if r["source_object_category"] in gt[r["image_id"]]
            and r["target_object_category"] not in gt[r["image_id"]]
        ]
        freeze(
            OUT / "released_annotation_audit.json",
            {
                "released_samples": len(labels),
                "released_direction_count": len(
                    {(r["source_object_category"], r["target_object_category"]) for r in labels}
                ),
                "sampling_policy": "revalidate published >=3/5 caption mentions and10-50pct box area BEFORE target output",
                "aliases": ALIASES,
                "GT_verified_source_present_target_absent": True,
                "pair_counts": [
                    {
                        "source": s,
                        "target": t,
                        "released": sum(
                            (r["source_object_category"], r["target_object_category"]) == (s, t)
                            for r in labels
                        ),
                        "eligible": sum(
                            (r["source_object_category"], r["target_object_category"]) == (s, t)
                            and label_eligible(r)
                            for r in labels
                        ),
                    }
                    for s, t in PAIRS
                ],
                "superseded_pre_gpu_manifest": str(
                    BASE / "protocol_smoke10/crossvlad_manifest.json"
                ),
            },
        )
        selected = sample_labels(gt_eligible)
        freeze(OUT / "selected_official_labels.json", selected)
        excluded = {r["image_id"] for r in labels}
        for name in ["coco_captioning", "llava_bench_coco"]:
            prior = json.loads((BASE / "extension/manifests" / f"{name}.json").read_text())
            excluded.update(int(r["image_id"].split("/")[-1]) for r in prior["images"])
        clean = {}
        for r in selected:
            clean[r["image_id"]] = canvas(
                io.BytesIO(z.read(f"CrossVLAD/images/{r['image_id']:012d}.jpg")),
                OUT / "images/clean" / f"{r['image_id']:012d}.png",
            )
    # val2017 provides an independent image pool and avoids downloading train2017.
    # Fail explicitly if a chosen category has fewer than 48 eligible references.
    with zipfile.ZipFile(RAW / "annotations_trainval2017.zip") as z:
        annotations = json.loads(z.read("annotations/instances_val2017.json"))
        captions = json.loads(z.read("annotations/captions_val2017.json"))
    classes = {r["id"]: r["name"] for r in annotations["categories"]}
    objects = defaultdict(set)
    for a in annotations["annotations"]:
        if not a.get("iscrowd"):
            objects[a["image_id"]].add(classes[a["category_id"]])
    human = defaultdict(list)
    for a in captions["annotations"]:
        human[a["image_id"]].append(a["caption"])
    banks, pool = {}, {}
    for source, target in PAIRS:
        banks[source + "->" + target] = {}
        for role, positive, negative in [("source", source, target), ("target", target, source)]:
            candidates = sorted(
                i
                for i, cs in objects.items()
                if i not in excluded and positive in cs and negative not in cs
            )
            random.Random(f"42:{positive}:not:{negative}").shuffle(candidates)
            if len(candidates) < 48:
                raise RuntimeError(f"Insufficient class bank {positive}: {len(candidates)}/48")
            ids = candidates[:48]
            banks[source + "->" + target][role] = ids
            for i in ids:
                pool[i] = {"image_id": i, "class_labels": sorted(objects[i]), "captions": human[i]}

    def one(item):
        i, meta = item
        original = fetch(
            f"http://images.cocodataset.org/val2017/{i:012d}.jpg",
            RAW / "crossvlad/references" / f"{i:012d}.jpg",
        )
        return {**meta, **canvas(original, OUT / "images/references" / f"{i:012d}.png")}

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        refs = sorted(executor.map(one, sorted(pool.items())), key=lambda r: r["image_id"])
    index = {r["image_id"]: n for n, r in enumerate(refs)}
    assignments, rows = {}, []
    for r in selected:
        source, target = r["source_object_category"], r["target_object_category"]
        bank = banks[source + "->" + target]
        identity = f"crossvlad/{r['image_id']}"
        assignments[identity] = {
            role + "_indices": [index[i] for i in ids] for role, ids in bank.items()
        }
        reference = refs[index[bank["target"][0]]]
        rows.append(
            {
                "image_id": identity,
                "coco_id": r["image_id"],
                "source_category": source,
                "target_category": target,
                "clean": clean[r["image_id"]]["path"],
                "clean_sha256": clean[r["image_id"]]["sha256"],
                "target_reference": reference["path"],
                "target_reference_sha256": reference["sha256"],
                "captions": r["captions"],
                "target_caption": r["caption_label"],
                "official_label": r,
                "queries": [{"id": identity, "kind": "captioning", "text": "Describe this image."}],
            }
        )
    freeze(
        OUT / "reference_pool.json",
        {
            "images": refs,
            "N": len(refs),
            "selection": "COCO val2017 GT-positive, GT-negative excluded; seeded random48",
            "excluded_attack_ids": sorted(excluded),
            "annotation_sha256": sha(RAW / "annotations_trainval2017.zip"),
        },
    )
    freeze(
        OUT / "reference_banks.json",
        {
            "k": 48,
            "banks": assignments,
            "pool_sha256": sha(OUT / "reference_pool.json"),
            "target_feedback": False,
        },
    )
    freeze(
        OUT / "crossvlad_manifest.json",
        {
            "dataset": "crossvlad_captioning",
            "N_images": 10,
            "directions": [list(pair) for pair in PAIRS],
            "images_per_direction": 2,
            "seed": 42,
            "official_archive_sha256": sha(archive),
            "official_code_revision": CRAFT_REV,
            "input_adaptation": "224x224 canvas; natural caption prompt for chat MLLMs, not Florence task token",
            "score": "official lowercase substring: target present AND source absent",
            "images": rows,
        },
    )
    original = REPO / "experiments/2026-10-a100-precommercial-v6/model_revisions.json"
    reg = json.loads(original.read_text())
    names = ["CLIP ViT-L/14", "Qwen3.5-2B", "InternVL3.5-2B-HF", "Gemma 4 E2B-it"]
    freeze(OUT / "proxy_revisions.json", {n: reg[n] for n in names})
    for relative in [
        "llava/eval/eval_gpt_review.py",
        "llava/eval/eval_gpt_review_bench.py",
        "llava/eval/table/rule.json",
        "llava/eval/summarize_gpt_review.py",
    ]:
        fetch(
            f"https://raw.githubusercontent.com/haotian-liu/LLaVA/{LLAVA_REV}/{relative}",
            OUT / "upstream/llava" / relative,
        )
    atomic(
        OUT / "preparation.json",
        {
            "status": "PASS",
            "crossvlad_images": 10,
            "references": len(refs),
            "class_bank_size": 48,
            "target_feedback": False,
        },
    )
    print("PREPARED CrossVLAD5x2", len(refs), "reference images", flush=True)


if __name__ == "__main__":
    prepare()
