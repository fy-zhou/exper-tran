import copy
import importlib.util
from pathlib import Path


def module():
    path = Path(__file__).resolve().parents[1] / "dataset_expansion.py"
    spec = importlib.util.spec_from_file_location("dataset_expansion_test", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def lock():
    return {
        "config": {"N": 10, "steps": 50, "parameters": "bfloat16_unquantized"},
        "manifest": {"images": []},
        "scientific_sources": {"worker.py": "sha"},
        "source_manifest_sha256": "manifest_sha",
        "targets": {"target": "revision"},
    }


def test_only_bookkeeping_differences_allow_reuse():
    a = lock()
    b = copy.deepcopy(a)
    b["config"].update(N=150, study="new_study")
    b["manifest"]["images"] = ["new images"]
    b["expansion_source_sha256"] = "new wrapper"
    assert module().compatible_lock(a, b)


def test_changed_steps_precision_registry_or_sources_forbid_reuse():
    for key, value in [("steps", 300), ("parameters", "float16")]:
        a = lock()
        b = copy.deepcopy(a)
        b["config"][key] = value
        assert not module().compatible_lock(a, b)
    a = lock()
    b = copy.deepcopy(a)
    b["scientific_sources"]["worker.py"] = "changed"
    assert not module().compatible_lock(a, b)
    b = copy.deepcopy(a)
    b["targets"]["target"] = "changed_revision"
    assert not module().compatible_lock(a, b)
