import importlib.util
from pathlib import Path
from types import SimpleNamespace


def queue_module():
    source = Path(__file__).resolve().parents[1] / "local_queue.py"
    spec = importlib.util.spec_from_file_location("nov21_test_queue", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker(valid_attack=True, valid_eval=True):
    return SimpleNamespace(
        valid_receipt=lambda *args: valid_attack,
        digest=lambda value: "key",
        oldworker=SimpleNamespace(
            receipt_path=lambda run, row: run / "receipt.json",
            sig=lambda lock, row: "signature",
            allowed_targets=lambda config: ["target"],
            REG={"target": {}},
            valid_evaluation=lambda *args: valid_eval,
        ),
    )


def test_completed_attack_and_evaluation_are_reused_without_inference(tmp_path):
    reusable = queue_module().reusable_phase
    manifest = {"images": [{"image_id": "a", "queries": [{"id": 1}]}]}
    assert reusable(worker(), tmp_path, {}, {}, manifest, "attack")
    assert reusable(worker(), tmp_path, {}, {}, manifest, "evaluate")
    assert not reusable(worker(), tmp_path, {}, {}, manifest, "report")


def test_incomplete_or_invalid_artifacts_are_never_skipped(tmp_path):
    reusable = queue_module().reusable_phase
    manifest = {"images": [{"image_id": "a", "queries": [{"id": 1}]}]}
    assert not reusable(worker(valid_attack=False), tmp_path, {}, {}, manifest, "attack")
    assert not reusable(worker(valid_eval=False), tmp_path, {}, {}, manifest, "evaluate")
    assert not reusable(worker(), tmp_path, {}, {}, {"images": []}, "attack")
