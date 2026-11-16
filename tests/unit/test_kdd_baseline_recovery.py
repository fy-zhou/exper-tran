"""A failed baseline must never block an independent completed method's evaluation."""

import importlib.util
from pathlib import Path

from primary_ml_cka.baselines.campaign import signature
from primary_ml_cka.baselines.protocol import atomic, digest, sha


def module():
    p = Path(__file__).resolve().parents[2] / "scripts/kdd2027/evaluate_frozen_run.py"
    s = importlib.util.spec_from_file_location("baseline_recovery_test", p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def setup(tmp_path, monkeypatch):
    m = module()
    monkeypatch.setattr(m, "source_hashes", lambda: {"f": "unchanged"})
    rows = [{"image_id": "a"}, {"image_id": "b"}]
    manifest_path = tmp_path / "manifests/final150.json"
    atomic(manifest_path, {"eligible": 2, "images": rows})
    lock = {
        "sources": {"f": "unchanged"},
        "split": "final150",
        "manifest_sha256": sha(manifest_path),
        "methods": ["pgd", "foa_attack"],
    }
    run = tmp_path / "run"
    atomic(run / "method_lock.json", lock)
    png = tmp_path / "image.png"
    png.write_bytes(b"checkpoint-test-pixels")
    for method, cases in [("pgd", rows), ("foa_attack", rows[:1])]:
        for r in cases:
            atomic(
                run / "attacks" / method / digest(r["image_id"])[:18] / "receipt.json",
                {
                    "signature": signature(lock, r, method),
                    "status": "ATTACK_FROZEN",
                    "target_access": False,
                    "linf_png": 0.01,
                    "png_hashes": {str(png): sha(png)},
                },
            )
    return m, run, lock


def test_failed_method_does_not_block_complete_method(tmp_path, monkeypatch):
    m, run, _ = setup(tmp_path, monkeypatch)
    atomic(
        run / "attacks/foa_attack" / digest("b")[:18] / "failure.json",
        {"method": "foa_attack", "image_id": "b", "error": "empty cluster"},
    )
    _, manifest, groups, statuses = m.build_plan(run, tmp_path)
    assert groups[("a", "b")] == ["pgd"]
    assert groups[("a",)] == ["foa_attack"]
    assert manifest["eligible"] == 2
    assert statuses[1]["planned_N"] == 2 and statuses[1]["failed_ids"] == ["b"]


def test_unattempted_method_waits_without_blocking_completed(tmp_path, monkeypatch):
    m, run, _ = setup(tmp_path, monkeypatch)
    _, _, groups, statuses = m.build_plan(run, tmp_path)
    assert list(groups.values()) == [["pgd"]]
    assert not statuses[1]["eligible_for_evaluation"]
    assert statuses[1]["pending_ids"] == ["b"]


def test_changed_scientific_sources_rejected(tmp_path, monkeypatch):
    import pytest

    m, run, _ = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(m, "source_hashes", lambda: {"f": "changed"})
    with pytest.raises(RuntimeError, match="source lock"):
        m.build_plan(run, tmp_path)
