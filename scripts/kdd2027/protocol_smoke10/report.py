# ruff: noqa: E501
"""Official CrossVLAD caption criterion and unmodified LLaVA COCO scoring packets.

No invented judge/model fallback. Official historical COCO reviewer uses gpt-4,
reference first, original rule.json and a first-line pair of 1..10 scores.
"""

import csv
import json
import os

from prepare import LLAVA_REV, OUT, REPO

from primary_ml_cka.baselines.protocol import atomic, digest, freeze, sha, valid_receipt


def cross_hit(text, source, target):
    text = text.lower()
    return target.lower() in text and source.lower() not in text


def official_packet(question, reference, answer, rules):
    rule = rules.get(question["category"], rules["default"])
    role = rule["role"]
    return (
        f'[Question]\n{question["text"]}\n\n'
        f'[{role} 1]\n{reference}\n\n[End of {role} 1]\n\n'
        f'[{role} 2]\n{answer}\n\n[End of {role} 2]\n\n'
        f'[System]\n{rule["prompt"]}\n\n'
    )


def official_relative(pairs):
    if not pairs or any(not (1 <= a <= 10 and 1 <= b <= 10) for a, b in pairs):
        raise ValueError("Missing/invalid official review")
    return 100 * sum(b for a, b in pairs) / sum(a for a, b in pairs)


def report():
    rows, image_rows, requests = [], [], {}
    official = OUT / "upstream/llava"
    rule_file = official / "llava/eval/table/rule.json"
    rules = json.loads(rule_file.read_text()) if rule_file.exists() else None
    for path in sorted((OUT / "runs").glob("*/lock.json")):
        lock = json.loads(path.read_text())
        if any(
            sha(REPO / name) != checksum for name, checksum in lock["scientific_sources"].items()
        ):
            continue
        data, cfg = lock["manifest"], lock["config"]
        run = path.parent
        targets = [lock["target"]] if cfg["method"] == "ours" else list(lock["targets"])
        for target in targets:
            results = []
            for row in data["images"]:
                receipt = run / "attacks" / digest(row["image_id"])[:18] / "receipt.json"
                if not valid_receipt(receipt, digest([lock, row])):
                    continue
                for q in row["queries"]:
                    filename = digest([row["image_id"], q["id"]])[:18] + ".json"
                    p = run / "evaluations" / target / filename
                    if not p.exists():
                        continue
                    value = json.loads(p.read_text())
                    if (
                        value["receipt_sha256"] != sha(receipt)
                        or value["model"] != lock["targets"][target]
                    ):
                        raise RuntimeError("Evaluation provenance mismatch")
                    rec = {
                        "run": str(run),
                        "method": cfg["method"],
                        "target": target,
                        "image_id": row["image_id"],
                        "question_id": q["id"],
                        "evaluation_sha256": sha(p),
                        "clean": value["clean"]["text"],
                        "adv": value["adv"]["text"],
                        "steps": cfg["steps"],
                    }
                    if data["dataset"] == "crossvlad_captioning":
                        rec.update(
                            direction=row["source_category"] + "->" + row["target_category"],
                            adv_hit=cross_hit(
                                rec["adv"], row["source_category"], row["target_category"]
                            ),
                            clean_hit=cross_hit(
                                rec["clean"], row["source_category"], row["target_category"]
                            ),
                        )
                    elif rules is not None:
                        rec["category"] = q["category"]
                        for mode in ["clean", "adv"]:
                            packet = official_packet(q, q["reference_answer"], rec[mode], rules)
                            request = {
                                "model": "gpt-4",
                                "temperature": 0.2,
                                "max_tokens": 1024,
                                "messages": [
                                    {
                                        "role": "system",
                                        "content": "You are a helpful and precise assistant for checking the quality of the answer.",
                                    },
                                    {"role": "user", "content": packet},
                                ],
                                "official_revision": LLAVA_REV,
                                "rule_sha256": sha(rule_file),
                                "scoring_script_sha256": sha(
                                    official / "llava/eval/eval_gpt_review.py"
                                ),
                            }
                            key = digest(request)
                            requests[key] = request
                            freeze(OUT / "llava_official/requests" / f"{key}.json", request)
                            rec[mode + "_judge_request"] = key
                    results.append(rec)
                    image_rows.append(rec)
            planned = sum(len(r["queries"]) for r in data["images"])
            summary = {
                "dataset": data["dataset"],
                "method": cfg["method"],
                "target": target,
                "run": str(run),
                "steps": cfg["steps"],
                "returned": len(results),
                "planned": planned,
                "status": "RAW_COMPLETE" if len(results) == planned else "PARTIAL",
                "target_hits": None,
                "ASR": None,
                "official_clean_score": None,
                "official_adv_score": None,
            }
            if data["dataset"] == "crossvlad_captioning":
                summary["target_hits"] = sum(r["adv_hit"] for r in results)
                summary["clean_target_hits"] = sum(r["clean_hit"] for r in results)
                summary["ASR"] = (
                    summary["target_hits"] / planned if len(results) == planned else None
                )
                summary["directions"] = {
                    s + "->" + t: {
                        "hits": sum(
                            r["adv_hit"] for r in results if r["direction"] == s + "->" + t
                        ),
                        "returned": sum(r["direction"] == s + "->" + t for r in results),
                        "N": 2,
                    }
                    for s, t in data["directions"]
                }
            else:
                summary["status"] = (
                    "BLOCKED_OFFICIAL_GPT4_ACCESS"
                    if len(results) == planned
                    else "PARTIAL_RAW_OFFICIAL_JUDGE_BLOCKED"
                )
            rows.append(summary)
    atomic(OUT / "results.json", rows)
    atomic(OUT / "per_image.json", image_rows)
    atomic(
        OUT / "llava_official/access.json",
        {
            "status": "BLOCKED_NOT_CALLED",
            "OPENAI_API_KEY_present": bool(os.environ.get("OPENAI_API_KEY")),
            "subscription_cli_not_official_gpt4": True,
            "paid_fallback": False,
            "requests_prepared": len(requests),
            "scores_fabricated": False,
            "historical_COCO_reviewer_model": "gpt-4",
            "historical_in_the_wild_reviewer_model": "gpt-4-0314",
            "note": "No credentials/snapshot verification or approved paid route; only packets prepared.",
        },
    )
    dest = REPO / "docs/kdd2027/results/protocol_smoke10"
    dest.mkdir(parents=True, exist_ok=True)
    atomic(dest / "results.json", rows)
    atomic(dest / "per_image.json", image_rows)
    if rows:
        with (dest / "results.csv").open("w") as f:
            fields = [k for k in rows[0] if k != "directions"]
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# CrossVLAD5x2 and original LLaVA10 smoke",
        "",
        "Exploratory smoke, not a final benchmark. CrossVLAD uses official labels and caption criterion.",
        "Natural Describe-this-image prompt and224canvas are disclosed chat-MLLM/input adaptations.",
        "Five directions x2 identical sources; all3 large target families; Ours uses48GT references/class.",
        "M/FOA use one fixed exemplar from the same target class bank; SSA uses all80COCO class text prompts.",
        "These have different reference/text/compute resources; do not describe this as fully matched.",
        "Ours50steps; native-budget M/FOA300; SSA-CWA10outer steps/20spectrum samples.",
        "CrossVLAD official substring criterion retained, including its linguistic limitations.",
        "LLaVA10images/30original questions: historical COCO eval_gpt_review.py / rule.json / GPT4 reference.",
        "Original LLaVA Ours frozen outputs reused after receipt/source/bank checks.",
        "Official GPT4 access BLOCKED; no open judge or newer CLI is substituted. No ASR for LLaVA.",
        "",
        "|Task|Method|Target|Steps|Returned|Hits/N or official score|Status|",
        "|---|---|---|---:|---:|---|---|",
    ]
    for r in rows:
        score = (
            f"{r['target_hits']}/{r['planned']} ({r['ASR']:.1%})" if r["ASR"] is not None else "N/A"
        )
        lines.append(
            f"|{r['dataset']}|{r['method']}|{r['target']}|{r['steps']}|{r['returned']}/{r['planned']}|{score}|{r['status']}|"
        )
    (dest / "REPORT.md").write_text("\n".join(lines) + "\n")
    return rows


if __name__ == "__main__":
    report()
