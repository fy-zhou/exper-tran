# ruff: noqa: E402
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from prepare import choose_banks


def load(name):
    spec = importlib.util.spec_from_file_location("native_test_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_neighborhoods_disjoint_and_deterministic():
    rng = np.random.default_rng(42)
    pool = rng.normal(size=(512, 8))
    pool /= np.linalg.norm(pool, axis=1, keepdims=True)
    a, b = choose_banks(pool[0], pool[1], pool)
    assert len(a) == len(b) == 48
    assert not set(a) & set(b)
    assert (a, b) == choose_banks(pool[0], pool[1], pool)
    assert b[0] == 1


def test_judge_parser_rejects_missing_out_of_range_or_bool():
    parse = load("judge").parse
    assert parse('answer {"score_A":8,"score_B":3,"reason":"factual error"}')["score_B"] == 3
    for text in [
        "no JSON",
        '{"score_A":true,"score_B":3}',
        '{"score_A":11,"score_B":3}',
        '{"score_A":1}',
    ]:
        assert parse(text) is None


def test_proxy_pools_exclude_all_target_families():
    module = load("worker")
    families = {
        "CLIP ViT-L/14": "clip",
        "Qwen3.5-2B": "qwen",
        "InternVL3.5-2B-HF": "internvl",
        "Gemma 4 E2B-it": "gemma4",
    }
    for target, pool in module.POOLS.items():
        assert len(pool) == 3
        assert target not in {families[n] for n in pool}
    judges = load("judge").JUDGES
    assert all(k != v for k, v in judges.items())


def test_image_cluster_interval_preserves_zero_difference():
    assert load("report").interval([0.0] * 10) == [0.0, 0.0]
