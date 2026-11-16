"""Task-specific summaries. Never relabel caption/VQA metrics as classification ASR."""

import json
import re
from pathlib import Path

from task_protocol import eval_rows

from primary_ml_cka.baselines.protocol import atomic


def rouge_l(reference, text):
    a = re.findall(r"\w+", reference.lower())
    b = re.findall(r"\w+", text.lower())
    prev = [0] * (len(b) + 1)
    for token in a:
        current = [0]
        for j, other in enumerate(b):
            current.append(prev[j] + 1 if token == other else max(prev[j + 1], current[-1]))
        prev = current
    return 2 * prev[-1] / (len(a) + len(b)) if a or b else 1.0


def wilson(hits, n):
    z = 1.95996398454
    center = (hits / n + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((hits / n * (1 - hits / n) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return [center - half, center + half]


def summarize(run, output):
    lock = json.loads((run / "lock.json").read_text())
    config = lock["config"]
    manifest = json.loads(
        (output / "extension/manifests" / (config["dataset"] + ".json")).read_text()
    )
    planned = eval_rows(config, manifest)
    summaries = []
    for target in (run / "evaluations").glob("*"):
        records = [json.loads(p.read_text()) for p in target.glob("*.json")]
        expected = sum(len(r["queries"]) for r in planned)
        complete = len(records) == expected
        result = {
            "smoke_only": config["smoke_only"],
            "pilot": config.get("pilot", False),
            "method": config["method"],
            "dataset": config["dataset"],
            "variant": config["variant"],
            "target": target.name,
            "planned_images": len(planned),
            "planned_queries": expected,
            "returned_queries": len(records),
            "raw_complete": complete,
            "steps": config["steps"],
            "status": "COMPLETE" if complete else "PARTIAL",
            "no_commercial_calls": True,
        }
        if config["dataset"] == "imagenet":
            hits = sum(r["target_hit"] for r in records)
            result.update(
                target_hits=hits,
                ASR=hits / expected if complete else None,
                Wilson95=wilson(hits, expected) if complete else None,
            )
            image_map = {r["image_id"]: r for r in planned}
            result["directions"] = {}
            for tid in ["T01", "T03", "T05", "T07", "T09"]:
                selected = [r for r in records if image_map[r["image_id"]]["transition"] == tid]
                n = sum(r["transition"] == tid for r in planned)
                h = sum(r["target_hit"] for r in selected)
                if n:
                    result["directions"][tid] = {
                        "hits": h,
                        "planned_N": n,
                        "returned": len(selected),
                        "ASR": h / n if len(selected) == n else None,
                    }
            result["valid_untargeted_prediction_changes"] = sum(
                r["adv"].get("label") is not None
                and r["clean"].get("label") is not None
                and r["adv"]["label"] != r["clean"]["label"]
                for r in records
            )
            result["method_original_goal"] = config["primary_goal"]
        elif config["dataset"] == "coco_captioning" and records:
            from pycocoevalcap.cider.cider import Cider
            from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

            rows = {r["image_id"]: r for r in planned}
            tokenizer = PTBTokenizer()
            gt = {
                r["image_id"]: [{"caption": c} for c in rows[r["image_id"]]["captions"]]
                for r in records
            }
            target_gt = {
                r["image_id"]: [{"caption": c} for c in rows[r["image_id"]]["target_captions"]]
                for r in records
            }
            gt = tokenizer.tokenize(gt)
            target_gt = tokenizer.tokenize(target_gt)
            for mode in ["clean", "adv"]:
                pred = tokenizer.tokenize(
                    {r["image_id"]: [{"caption": r[mode]["text"]}] for r in records}
                )
                result[mode + "_CIDEr_valid_subset"] = float(Cider().compute_score(gt, pred)[0])
                result[mode + "_target_CIDEr_valid_subset"] = float(
                    Cider().compute_score(target_gt, pred)[0]
                )
            result["metric_scope"] = "fixed_COCO150_subset_or_explicitly_partial_not_full5k"
        elif records:
            scores = {
                mode: [rouge_l(r["query"]["reference_answer"], r[mode]["text"]) for r in records]
                for mode in ["clean", "adv"]
            }
            result.update({mode + "_ROUGE_L_proxy": sum(v) / len(v) for mode, v in scores.items()})
            result["official_LLaVA_GPT4_judge_score"] = None
            result["status"] = "RAW_COMPLETE_OFFICIAL_JUDGE_PENDING" if complete else "PARTIAL"
            result["ROUGE_L_is_not_official_VQA_correctness"] = True
        summaries.append(result)
    atomic(run / "task_summary.json", summaries)
    return summaries


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    summarize(a.run, a.output)
