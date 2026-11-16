"""Task-specific frozen cohorts: classification, captioning, official open VQA."""

import json
import random
from pathlib import Path

from PIL import Image

from primary_ml_cka.baselines.protocol import digest, freeze, sha

METHODS = ["pgd", "mi_fgsm", "co_attack", "sga", "m_attack", "foa_attack", "iau_foa"]
DATASETS = ["imagenet", "coco_captioning", "llava_bench_coco"]
TARGETS = ["Qwen3.5-27B", "InternVL3.5-14B-HF", "Gemma 4 31B-it"]
STEPS = {
    "pgd": 50,
    "mi_fgsm": 50,
    "co_attack": 50,
    "sga": 50,
    "m_attack": 300,
    "foa_attack": 300,
    "iau_foa": 300,
}


def canonical(path, destination):
    destination = destination.with_suffix(".png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with Image.open(path) as image:
            image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC).save(destination)
    return {
        "path": str(destination),
        "sha256": sha(destination),
        "original_path": str(path),
        "original_sha256": sha(path),
    }


def build_manifests(output, cache):
    manifests = {}
    original = json.loads((output / "manifests/final150.json").read_text())
    image_rows = []
    for row in original["images"]:
        image_rows.append(
            {
                **row,
                "captions": [
                    "A photo of a "
                    + row.get(
                        "class_name",
                        [
                            "goldfish",
                            "monarch butterfly",
                            "pineapple",
                            "acoustic guitar",
                            "laptop",
                            "espresso",
                            "volcano",
                            "rocking chair",
                            "soccer ball",
                            "school bus",
                        ][row["source"] - 1],
                    )
                    + "."
                ],
                "queries": [{"id": row["image_id"], "kind": "classification"}],
            }
        )
    manifests["imagenet"] = {
        "dataset": "imagenet",
        "task": "targeted_classification",
        "images": image_rows,
        "N_images": 150,
        "provenance": sha(output / "manifests/final150.json"),
    }
    sources = output / "datasets"
    coco_meta = json.loads((sources / "coco_captioning/source_manifest.json").read_text())
    llava = json.loads((sources / "llava_bench_coco/source_manifest.json").read_text())
    captions = json.loads((cache / "coco2017/annotations/captions_val2017.json").read_text())
    banks = {}
    for annotation in captions["annotations"]:
        banks.setdefault(annotation["image_id"], []).append(annotation["caption"])
    excluded = {r["image_id"] for r in llava["images"]}
    pool = sorted([r for r in captions["images"] if r["id"] not in excluded], key=lambda x: x["id"])
    random.Random(42).shuffle(pool)
    # Image-level disjoint attack/target banks; independent of all model outputs.
    clean, references = pool[:150], pool[150:330]
    canvas = output / "datasets/canonical224"
    rows = []
    for i, r in enumerate(clean):
        target = references[i]
        c = canonical(
            Path(coco_meta["image_root"]) / r["file_name"], canvas / "coco" / r["file_name"]
        )
        t = canonical(
            Path(coco_meta["image_root"]) / target["file_name"],
            canvas / "targets" / target["file_name"],
        )
        rows.append(
            {
                "image_id": "coco2017/" + str(r["id"]),
                "clean": c["path"],
                "clean_sha256": c["sha256"],
                "original": c,
                "target_original": t,
                "target_reference": t["path"],
                "target_reference_sha256": t["sha256"],
                "captions": banks[r["id"]],
                "target_captions": banks[target["id"]],
                "queries": [
                    {
                        "id": str(r["id"]),
                        "kind": "captioning",
                        "text": "Provide a concise, single-sentence caption describing this image.",
                    }
                ],
            }
        )
    manifests["coco_captioning"] = {
        "dataset": "coco_captioning",
        "task": "image_captioning",
        "N_images": 150,
        "images": rows,
        "caption_metric": "COCO_CIDEr_clean_vs_adv",
        "target_alignment_metric": "CIDEr_against_target_captions_not_ASR",
        "sampling_seed": 42,
        "source_annotations_sha256": coco_meta["annotation_sha256"],
        "N_source_pool": 5000,
        "LLaVA_image_ID_overlap": 0,
    }
    answer_path = sources / "llava_bench_coco/official/reference_answers.jsonl"
    answers = {
        str(x["question_id"]): x["text"]
        for x in (json.loads(s) for s in answer_path.read_text().splitlines() if s.strip())
    }
    rows = []
    for i, r in enumerate(llava["images"]):
        c = canonical(Path(r["image"]), canvas / "llava" / Path(r["image"]).name)
        # Separate held-out COCO target images, not another question's test image.
        target = references[150 + i]
        t = canonical(
            Path(coco_meta["image_root"]) / target["file_name"],
            canvas / "targets" / target["file_name"],
        )
        rows.append(
            {
                "image_id": "llava_coco/" + str(r["image_id"]),
                "clean": c["path"],
                "clean_sha256": c["sha256"],
                "original": c,
                "target_reference": t["path"],
                "target_reference_sha256": t["sha256"],
                "captions": [answers[str(x["question_id"])] for x in r["questions"]],
                "queries": [
                    {
                        "id": str(x["question_id"]),
                        "text": x["text"],
                        "kind": "open_vqa",
                        "category": x["category"],
                        "reference_answer": answers[str(x["question_id"])],
                    }
                    for x in r["questions"]
                ],
            }
        )
    manifests["llava_bench_coco"] = {
        "dataset": "llava_bench_coco",
        "task": "official_open_VQA",
        "N_images": 30,
        "N_questions": 90,
        "images": rows,
        "official_source_sha256": sha(sources / "llava_bench_coco/source_manifest.json"),
        "judge_status": "OFFICIAL_GPT4_JUDGE_NOT_CONFIGURED",
        "input_resolution_adaptation": "224_canvas_clean_and_adv_paired",
        "no_relabel_as_ten_class_ASR": True,
    }
    # Engineering cohorts are disjoint from every formal clean/reference set.
    debug = {}
    names = [
        "goldfish",
        "monarch butterfly",
        "pineapple",
        "acoustic guitar",
        "laptop",
        "espresso",
        "volcano",
        "rocking chair",
        "soccer ball",
        "school bus",
    ]
    debug["imagenet"] = {}
    for mode, filename in [("smoke", "smoke.json"), ("pilot", "screen.json")]:
        data = json.loads((output / "manifests" / filename).read_text())
        debug["imagenet"][mode] = [
            {
                **r,
                "captions": ["A photo of a " + names[r["source"] - 1] + "."],
                "queries": [{"id": r["image_id"], "kind": "classification"}],
            }
            for r in data["images"]
        ]
    dev = []
    for i, r in enumerate(pool[330:332]):
        target = pool[332 + i]
        c = canonical(
            Path(coco_meta["image_root"]) / r["file_name"], canvas / "debug" / r["file_name"]
        )
        t = canonical(
            Path(coco_meta["image_root"]) / target["file_name"],
            canvas / "debug_targets" / target["file_name"],
        )
        dev.append(
            {
                "image_id": "engineering_coco/" + str(r["id"]),
                "clean": c["path"],
                "clean_sha256": c["sha256"],
                "target_reference": t["path"],
                "target_reference_sha256": t["sha256"],
                "captions": banks[r["id"]],
                "target_captions": banks[target["id"]],
                "queries": [
                    {
                        "id": "debug:" + str(r["id"]),
                        "kind": "captioning",
                        "text": "Provide a concise, single-sentence caption describing this image.",
                    }
                ],
            }
        )
    debug["coco_captioning"] = {"smoke": dev}
    debug["llava_bench_coco"] = {
        "smoke": [
            {
                **r,
                "queries": [
                    {
                        "id": "debug:" + r["image_id"],
                        "kind": "open_vqa",
                        "category": "engineering_debug",
                        "reference_answer": r["captions"][0],
                        "text": "What objects are visible in this image?",
                    }
                ],
            }
            for r in dev
        ]
    }
    formal_clean = {r["clean_sha256"] for m in manifests.values() for r in m["images"]}
    formal_reference = {
        r["target_reference_sha256"] for m in manifests.values() for r in m["images"]
    }
    for groups in debug.values():
        for rows in groups.values():
            if any(r["clean_sha256"] in formal_clean | formal_reference for r in rows):
                raise ValueError("Engineering/formal or reference leakage")
    freeze(output / "extension/manifests/debug_inputs.json", debug)
    for name, manifest in manifests.items():
        ids = [r["image_id"] for r in manifest["images"]]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate image IDs")
        if {r["clean_sha256"] for r in manifest["images"]} & {
            r["target_reference_sha256"] for r in manifest["images"]
        }:
            raise ValueError("Clean/reference image leakage")
        freeze(output / "extension/manifests" / f"{name}.json", manifest)
    return manifests


DEBUG_INPUTS = Path(
    "/home/ubuntu/exper-tran/outputs/kdd2027_baseline_reproduction/extension/manifests/debug_inputs.json"
)


def eval_rows(config, manifest):
    if config.get("smoke_only") or config.get("pilot"):
        debug = json.loads(DEBUG_INPUTS.read_text())
        mode = "smoke" if config.get("smoke_only") else "pilot"
        if mode not in debug[config["dataset"]]:
            raise ValueError("No independent engineering cohort configured")
        return debug[config["dataset"]][mode]
    return manifest["images"]


def condition(method, dataset, variant="default", smoke=False):
    if variant not in ["default", "family_disjoint4"] or (
        variant != "default" and method != "iau_foa"
    ):
        raise ValueError("Unknown or inapplicable proxy variant")
    if method not in METHODS or dataset not in DATASETS:
        raise ValueError("Unknown condition, no fallback")
    return {
        "method": method,
        "dataset": dataset,
        "variant": variant,
        "steps": 2 if smoke else STEPS[method],
        "epsilon": 16 / 255,
        "seed": 42,
        "canvas": 224,
        "parameters": "bfloat16_unquantized",
        "primary_goal": "untargeted_retrieval_origin"
        if method in ["co_attack", "sga"]
        else "targeted_feature_matching",
        "Co_SGA_budget_adaptation": "eps16/255_steps50" if method in ["co_attack", "sga"] else None,
        "target_input": "frozen_image_with_original_task_prompt",
        "kmeans_dependency": "f7f36bd1cb4e3a761d73d584866d0a9c6b4d2805"
        if method in ["foa_attack", "iau_foa"]
        else None,
        "no_commercial_calls": True,
        "smoke_only": smoke,
    }


def identity(config, sources):
    return digest([config, sources])[:18]
