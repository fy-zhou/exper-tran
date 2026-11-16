"""Cross-task protocol, native dependency and legal proxy set invariants."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parents[2] / "scripts/kdd2027/extension"
sys.path.insert(0, str(HERE))
import native_methods as native  # noqa: E402
import task_protocol as protocol  # noqa: E402
import worker  # noqa: E402
from metrics import rouge_l  # noqa: E402


def test_matrix_has_exactly63_distinct_method_task_target_cells():
    cells = {
        (m, d, t) for m in protocol.METHODS for d in protocol.DATASETS for t in protocol.TARGETS
    }
    assert len(cells) == 63


def test_entire_target_family_exclusion():
    with pytest.raises(ValueError, match="collision"):
        native.family_legal(["B16", "InternVL1", "DINO"], "internvl")
    assert native.family_legal(["B16", "InternVL1", "DINO"], "qwen")
    assert native.family_legal(["B16", "B32", "Laion", "DINO"], "internvl")


def test_native5_and_family_adaptation_cover_exact_target_sets():
    five = protocol.condition("iau_foa", "imagenet")
    four = protocol.condition("iau_foa", "imagenet", "family_disjoint4")
    assert set(worker.allowed_targets(five)) == {"Qwen3.5-27B", "Gemma 4 31B-it"}
    assert worker.allowed_targets(four) == ["InternVL3.5-14B-HF"]


def test_unknown_variants_and_methods_fail_closed():
    for args in [("fake", "imagenet"), ("sga", "wrong"), ("sga", "imagenet", "family_disjoint4")]:
        with pytest.raises(ValueError):
            protocol.condition(*args)


def test_original_vqa_questions_are_not_expanded_into150_images():
    manifest = {
        "images": [
            {"image_id": str(i), "queries": [{"id": 3 * i + j} for j in range(3)]}
            for i in range(30)
        ]
    }
    config = protocol.condition("pgd", "llava_bench_coco")
    rows = protocol.eval_rows(config, manifest)
    assert len(rows) == 30 and sum(len(r["queries"]) for r in rows) == 90


def test_pilot_direction_balancing():
    manifest = {
        "images": [
            {"transition": t, "image_id": f"{t}-{i}"}
            for t in ["T01", "T03", "T05", "T07", "T09"]
            for i in range(30)
        ]
    }
    config = {**protocol.condition("iau_foa", "imagenet"), "pilot": True}
    result = protocol.eval_rows(config, manifest)
    assert len(result) == 10
    assert all(
        sum(r["transition"] == t for r in result) == 2 for t in ["T01", "T03", "T05", "T07", "T09"]
    )


def test_rouge_is_only_reference_overlap_proxy():
    assert rouge_l("a cat sits", "a cat sits") == 1
    assert rouge_l("a cat sits", "fish swims") == 0


def test_official_kmeans_pin_handles_empty_clusters():
    root = Path("/home/ubuntu/baseline-upstreams")
    if not (root / "kmeans_pinned").exists():
        pytest.skip("Pinned external dependency unavailable")
    kmeans = native.pinned_kmeans(root)
    with torch.device("cpu"):
        x = torch.ones(20, 3, requires_grad=True)
        np.random.seed(42)
        torch.manual_seed(42)
        _, centers = kmeans(x, 10, iter_limit=100, tqdm_flag=False)
        assert torch.isfinite(centers).all()
        gradient = torch.autograd.grad(centers.sum(), x)[0]
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_new_foa_never_reuses_incompatible_old_dependency():
    config = protocol.condition("foa_attack", "imagenet")
    assert not worker.import_legacy(config, None, None, {})


def test_smoke_budget_never_imports_full_old_attack():
    config = protocol.condition("pgd", "imagenet", smoke=True)
    assert not worker.import_legacy(config, None, None, {})


def test_original_question_and_auxiliary_answer_separation():
    path = worker.OUT / "extension/manifests/llava_bench_coco.json"
    if not path.exists():
        pytest.skip("Prepared source cohort unavailable")
    import json

    data = json.loads(path.read_text())
    for row in data["images"]:
        assert len(row["queries"]) == 3
        assert all(
            "reference_answer" in q and q["text"] != q["reference_answer"] for q in row["queries"]
        )
