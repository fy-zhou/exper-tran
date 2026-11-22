"""Scientific guardrails: cohort, actual reference isolation, official scoring."""

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import OUT, PAIRS, sample_labels
from report import cross_hit, official_packet, official_relative


def test_five_by_two_not_random_ten():
    labels = [
        {
            "image_id": i * 10 + j,
            "source_object_category": s,
            "target_object_category": t,
            "captions": [s] * 5,
            "image_width": 100,
            "image_height": 100,
            "source_object_boxes": [[0, 0, 50, 50]],
        }
        for i, (s, t) in enumerate(PAIRS)
        for j in range(5)
    ]
    a = sample_labels(labels)
    assert a == sample_labels(list(reversed(labels)))
    assert Counter(
        (r["source_object_category"], r["target_object_category"]) for r in a
    ) == dict.fromkeys(PAIRS, 2)
    with pytest.raises(ValueError):
        sample_labels(labels[:5])


def test_actual_reference_banks_disjoint():
    data = json.loads((OUT / "crossvlad_manifest.json").read_text())
    pool = json.loads((OUT / "reference_pool.json").read_text())
    banks = json.loads((OUT / "reference_banks.json").read_text())["banks"]
    assert len(data["images"]) == 10
    assert len({r["coco_id"] for r in data["images"]}) == 10
    assert not {r["coco_id"] for r in data["images"]} & {r["image_id"] for r in pool["images"]}
    for row in data["images"]:
        bank = banks[row["image_id"]]
        assert not set(bank["source_indices"]) & set(bank["target_indices"])
        for role, positive, negative in [
            ("source", row["source_category"], row["target_category"]),
            ("target", row["target_category"], row["source_category"]),
        ]:
            ids = bank[role + "_indices"]
            assert len(set(ids)) == 48
            for i in ids:
                assert positive in pool["images"][i]["class_labels"]
                assert negative not in pool["images"][i]["class_labels"]
                assert Path(pool["images"][i]["path"]).exists()


def test_official_caption_criterion_and_its_limitations():
    assert cross_hit("A DOG resting.", "cat", "dog")
    assert not cross_hit("A cat beside a dog.", "cat", "dog")
    assert not cross_hit("A cat resting.", "cat", "dog")
    # Exact official substring behavior, not a secretly improved parser.
    assert not cross_hit("A dog on vacation.", "cat", "dog")


def test_official_ratio_of_means_not_mean_of_ratios():
    assert official_relative([(10, 5), (5, 5)]) == pytest.approx(100 * 10 / 15)
    with pytest.raises(ValueError):
        official_relative([])
    with pytest.raises(ValueError):
        official_relative([(0, 10)])


def test_official_reference_first_and_question_unchanged():
    rules = {"default": {"role": "Assistant", "prompt": "RULE"}}
    q = {"category": "conv", "text": "What animal is this?"}
    packet = official_packet(q, "REF", "ADV", rules)
    assert (
        packet == "[Question]\nWhat animal is this?\n\n[Assistant 1]\nREF\n\n"
        "[End of Assistant 1]\n\n[Assistant 2]\nADV\n\n"
        "[End of Assistant 2]\n\n[System]\nRULE\n\n"
    )
