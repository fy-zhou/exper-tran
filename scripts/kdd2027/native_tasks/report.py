# ruff: noqa: E501
"""Native task quality with image-clustered uncertainty; no invented attack ASR."""

import csv
import json

from prepare import OUT, REPO

from primary_ml_cka.baselines.protocol import atomic

DEST = REPO / "docs/kdd2027/results/native_tasks_ours_first"


def interval(values):
    import numpy as np

    a = np.asarray(values, float)
    if len(a) == 0:
        return None
    rng = np.random.default_rng(42)
    means = np.array([a[rng.integers(0, len(a), len(a))].mean() for _ in range(1000)])
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def report():
    DEST.mkdir(parents=True, exist_ok=True)
    results = []
    per_image = []
    for lockpath in sorted((OUT / "runs").glob("*/lock.json")):
        run = lockpath.parent
        lock = json.loads(lockpath.read_text())
        cfg = lock["config"]
        if cfg.get("engineering"):
            continue
        rows = {r["image_id"]: r for r in lock["manifest"]["images"]}
        target = lock["target"]
        records = [json.loads(p.read_text()) for p in (run / "evaluations" / target).glob("*.json")]
        expected = sum(len(r["queries"]) for r in rows.values())
        summary = {
            "run": run.name,
            "dataset": cfg["dataset"],
            "target": target,
            "N_images": len(rows),
            "planned_queries": expected,
            "raw_returned": len(records),
            "raw_complete": len(records) == expected,
            "protocol_adaptation": "label_free_disjoint48_reference_neighborhoods",
            "official_llava_score": None,
        }
        if cfg["dataset"] == "coco_captioning" and records:
            from pycocoevalcap.cider.cider import Cider
            from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

            tokenizer = PTBTokenizer()
            gt = tokenizer.tokenize(
                {
                    r["image_id"]: [{"caption": c} for c in rows[r["image_id"]]["captions"]]
                    for r in records
                }
            )
            scores = {}
            for mode in ["clean", "adv"]:
                pred = tokenizer.tokenize(
                    {r["image_id"]: [{"caption": r[mode]["text"]}] for r in records}
                )
                mean, individual = Cider().compute_score(gt, pred)
                summary[mode + "_CIDEr_observed"] = float(mean)
                scores[mode] = [float(x) for x in individual]
            differences = []
            for i, r in enumerate(records):
                d = scores["clean"][i] - scores["adv"][i]
                differences.append(d)
                per_image.append(
                    {
                        "run": run.name,
                        "image_id": r["image_id"],
                        "metric": "CIDEr",
                        "clean": scores["clean"][i],
                        "adv": scores["adv"][i],
                        "quality_drop": d,
                    }
                )
            summary["image_bootstrap_drop95_observed"] = interval(differences)
            summary["primary_metric_valid"] = len(records) == expected
            summary["CIDEr_IDF_population"] = len(records)
        if cfg["dataset"] == "llava_bench_coco":
            judges = [
                json.loads(p.read_text()) for p in (run / "open_judge" / target).glob("*.json")
            ]
            valid = [r for r in judges if r["status"] == "VALID"]
            summary["judge_valid"] = len(valid)
            summary["judge_invalid"] = len(judges) - len(valid)
            summary["primary_metric_valid"] = len(valid) == expected
            if valid:
                grouped = {}
                for r in valid:
                    s = r["scores"]
                    d = s["clean"] - s["adv"]
                    grouped.setdefault(r["image_id"], []).append(d)
                    per_image.append(
                        {
                            "run": run.name,
                            "image_id": r["image_id"],
                            "question_id": r["question_id"],
                            "metric": "cross_family_open_judge_0to10",
                            "clean": s["clean"],
                            "adv": s["adv"],
                            "quality_drop": d,
                        }
                    )
                summary["clean_open_judge_observed"] = sum(
                    r["scores"]["clean"] for r in valid
                ) / len(valid)
                summary["adv_open_judge_observed"] = sum(r["scores"]["adv"] for r in valid) / len(
                    valid
                )
                summary["image_cluster_bootstrap_drop95_observed"] = interval(
                    [sum(v) / len(v) for v in grouped.values()]
                )
                summary["judge"] = valid[0]["judge"]
                summary["official_score"] = False
        results.append(summary)
    atomic(DEST / "results.json", results)
    from prepare import BASE

    baselines = []
    for root in [BASE / "nov21/local", BASE / "nov21_dataset_expansion/runs"]:
        for path in root.glob("*/open_judge/*"):
            records = [json.loads(p.read_text()) for p in path.glob("*.json")]
            valid = [r for r in records if r["status"] == "VALID"]
            if not records:
                continue
            baselines.append(
                {
                    "run": str(path.parent.parent),
                    "target": path.name,
                    "valid": len(valid),
                    "invalid": len(records) - len(valid),
                    "official_score": False,
                    "clean_observed": sum(r["scores"]["clean"] for r in valid) / len(valid)
                    if valid
                    else None,
                    "adv_observed": sum(r["scores"]["adv"] for r in valid) / len(valid)
                    if valid
                    else None,
                }
            )
    atomic(DEST / "baseline_open_judge.json", baselines)

    keys = list(dict.fromkeys(k for r in per_image for k in r))
    if keys:
        with (DEST / "per_image_metrics.csv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
            w.writeheader()
            w.writerows(per_image)
    lines = [
        "# Ours first: native COCO captioning and LLaVA open QA",
        "",
        "Ours uses a declared label-free48-reference-neighborhood adapter. It is not the original ImageNet class-bank protocol.",
        "All model parameters BF16, target-family-disjoint proxies,50 steps, no target feedback in generation.",
        "Positive quality drop means degradation. No classification ASR is inferred. Missing/invalid scores are not zero.",
        "LLaVA open-weight cross-family paired judge is adapted, blinded and explicitly not official GPT4 scoring.",
        "CIDEr depends on the evaluated reference population; compare methods only at matched N/IDs.",
        "",
        "|Dataset|Target|Images|Returned/planned|Primary metric valid|Clean observed|Adv observed|",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for r in results:
        clean = r.get("clean_CIDEr_observed", r.get("clean_open_judge_observed", "N/A"))
        adv = r.get("adv_CIDEr_observed", r.get("adv_open_judge_observed", "N/A"))
        lines.append(
            f"|{r['dataset']}|{r['target']}|{r['N_images']}|{r['raw_returned']}/{r['planned_queries']}|{r.get('primary_metric_valid',False)}|{clean}|{adv}|"
        )
    (DEST / "REPORT.md").write_text("\n".join(lines) + "\n")
    return results


if __name__ == "__main__":
    report()
