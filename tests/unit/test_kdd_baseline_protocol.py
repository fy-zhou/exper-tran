"""Scientific invariants, upstream core equivalence and restart corruption tests."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from primary_ml_cka.baselines.feature_attacks import (
    TargetImageCosine,
    bounded_kmeans,
    load_upstream,
    run_feature_attack,
)
from primary_ml_cka.baselines.iterative import AttackConfig, attack_image
from primary_ml_cka.baselines.protocol import atomic, freeze, sha, valid_receipt, validate_method


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), 0, -1])
def test_step_size_must_be_finite_positive(bad):
    with pytest.raises(ValueError):
        AttackConfig("pgd", 2, 0.1, bad)


@pytest.mark.parametrize("bad", [1.5, True, 0, -1])
def test_integer_steps(bad):
    with pytest.raises(ValueError):
        AttackConfig("pgd", bad, 0.1, 0.01)


@pytest.mark.parametrize("name", ["sga", "co_attack", "iau_foa", "fake"])
def test_unimplemented_never_falls_back(name):
    with pytest.raises(ValueError):
        validate_method(name)


class TinyFeatures(torch.nn.Module):
    def forward(self, raw255):
        x = raw255.float().flatten(1)[:, :12] / 255
        return torch.nn.functional.normalize(x, dim=1)


def test_target_cosine_is_explicit_not_pull_push():
    model = TinyFeatures()
    x = torch.linspace(0.2, 0.8, 12).reshape(1, 3, 2, 2).requires_grad_()
    reference = x.detach().flip(-1)
    objective = TargetImageCosine([model], reference)
    expected = 1 - (model(x * 255) * model(reference * 255)).sum()
    assert objective(x).item() == pytest.approx(expected.item())
    assert torch.autograd.grad(objective(x), x)[0].abs().sum() > 0
    adv = attack_image(x, objective, AttackConfig("pgd", 5, 0.1, 0.01))
    assert objective(adv) < objective(x)


def test_freeze_and_corrupt_resume(tmp_path):
    image = tmp_path / "adv.png"
    image.write_bytes(b"original")
    r = {
        "signature": "s",
        "status": "ATTACK_FROZEN",
        "target_access": False,
        "linf_png": 0.01,
        "png_hashes": {str(image): sha(image)},
    }
    receipt = tmp_path / "receipt.json"
    atomic(receipt, r)
    assert valid_receipt(receipt, "s")
    assert not valid_receipt(receipt, "different-science")
    image.write_bytes(b"corrupted")
    assert not valid_receipt(receipt, "s")
    freeze(tmp_path / "lock.json", {"seed": 42})
    freeze(tmp_path / "lock.json", {"seed": 42})
    with pytest.raises(RuntimeError):
        freeze(tmp_path / "lock.json", {"seed": 43})
    assert json.loads((tmp_path / "lock.json").read_text())["seed"] == 42


def test_kmeans_centroids_keep_input_gradient():
    np.random.seed(42)
    x = torch.tensor([[0.0, 0.0], [0.01, 0.02], [3.0, 3.0], [3.01, 3.02]], requires_grad=True)
    _, centers = bounded_kmeans(x, 2)
    assert centers.shape == (2, 2)
    grad = torch.autograd.grad(centers.square().sum(), x)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_kmeans_nonconvergence_is_failure():
    with pytest.raises(RuntimeError, match="converge"):
        bounded_kmeans(torch.rand(8, 4), 2, iter_limit=0)


def test_upstream_m_cosine_value_and_gradient():
    root = Path("/home/ubuntu/baseline-upstreams")
    if not (root / "m_attack/.git").exists():
        pytest.skip("Optional pinned upstream checkout unavailable")
    module, _ = load_upstream("m_attack", root)
    models = [TinyFeatures(), TinyFeatures()]
    target = torch.rand(1, 3, 2, 2)
    image = torch.rand_like(target, requires_grad=True)
    loss = module.EnsembleFeatureLoss(models)
    loss.set_ground_truth(target * 255)
    official = loss(module.EnsembleFeatureExtractor(models)(image * 255))
    own = 1 - TargetImageCosine(models, target)(image)
    torch.testing.assert_close(official, own)
    torch.testing.assert_close(
        torch.autograd.grad(official, image)[0], torch.autograd.grad(own, image)[0]
    )


def test_foa_sinkhorn_matches_source_cosine_singleton():
    root = Path("/home/ubuntu/baseline-upstreams")
    if not (root / "foa_attack/.git").exists():
        pytest.skip("Optional pinned upstream checkout unavailable")
    module, _ = load_upstream("foa_attack", root)
    objective = module.EnsembleFeatureLoss_OT_foa_attack([TinyFeatures()])
    a = torch.tensor([[1.0, 2.0, 3.0]])
    b = torch.tensor([[3.0, 2.0, 1.0]], requires_grad=True)
    score = objective.OT(a, b)
    expected = torch.nn.functional.cosine_similarity(a, b).sum()
    torch.testing.assert_close(score, expected)
    assert torch.isfinite(torch.autograd.grad(score, b)[0]).all()


def test_vector_loss_rejected():
    with pytest.raises(ValueError, match="scalar"):
        attack_image(
            torch.full((1, 3, 2, 2), 0.5), lambda x: x.flatten(), AttackConfig("pgd", 1, 0.1, 0.01)
        )


def test_nan_gradient_rejected():
    def invalid(x):
        return (x - x.detach()).sqrt().sum()

    with pytest.raises(ValueError, match="gradient"):
        attack_image(torch.full((1, 3, 2, 2), 0.5), invalid, AttackConfig("pgd", 1, 0.1, 0.01))


def test_target_evaluation_module_not_imported_by_attack():
    import inspect

    from primary_ml_cka.baselines import feature_attacks

    assert "load_target_for_generation" not in inspect.getsource(feature_attacks)


def test_family_limit_and_canvas_are_not_silent():
    with pytest.raises(ValueError, match="canonical224"):
        run_feature_attack(
            "pgd",
            [],
            torch.zeros(1, 3, 2, 2),
            torch.zeros(1, 3, 2, 2),
            steps=1,
            seed=42,
            upstream_root="unused",
        )


def test_m_attack_loop_matches_official_entrypoint():
    import ast
    from types import SimpleNamespace

    from torchvision import transforms as T

    root = Path("/home/ubuntu/baseline-upstreams")
    source = root / "m_attack/generate_adversarial_samples.py"
    if not source.exists():
        pytest.skip("Pinned upstream unavailable")
    module, _ = load_upstream("m_attack", root)
    tree = ast.parse(source.read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "fgsm_attack"
    )
    # Execute ONLY the reviewed pure attack function, never upstream main/API/logins.
    body = ast.parse("from __future__ import annotations").body + [function]
    namespace = {
        "torch": torch,
        "tqdm": lambda x, **kwargs: x,
        "log_metrics": lambda *args, **kwargs: None,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), "exec"), namespace)
    models = [TinyFeatures(), TinyFeatures()]
    clean, target = torch.rand(1, 3, 224, 224), torch.rand(1, 3, 224, 224)
    cfg = SimpleNamespace(
        optim=SimpleNamespace(steps=3, epsilon=16, alpha=1),
        model=SimpleNamespace(use_source_crop=True),
    )
    torch.manual_seed(987)
    expected = namespace["fgsm_attack"](
        cfg,
        module.EnsembleFeatureExtractor(models),
        module.EnsembleFeatureLoss(models),
        T.RandomResizedCrop(224, scale=(0.5, 0.9)),
        T.RandomResizedCrop(224, scale=(0.5, 0.9)),
        0,
        clean * 255,
        target * 255,
    )
    actual, _ = run_feature_attack(
        "m_attack", models, clean, target, steps=3, seed=987, upstream_root=root
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_partial_evaluation_never_becomes_zero_asr(tmp_path):
    from types import SimpleNamespace

    from primary_ml_cka.baselines.campaign import report

    args = SimpleNamespace(methods=["pgd"], split="smoke", steps=2)
    manifest = {
        "eligible": 2,
        "images": [{"image_id": "a", "transition": "T01"}, {"image_id": "b", "transition": "T01"}],
    }
    assert not report(args, manifest, tmp_path, {})
    rows = json.loads((tmp_path / "results.json").read_text())
    assert all(r["target_ASR"] is None and r["status"] == "BLOCKED" for r in rows)
    assert all(r["eligible"] == 2 and r["evaluated"] == 0 for r in rows)


def test_evaluation_cannot_start_before_freeze(tmp_path):
    from types import SimpleNamespace

    from primary_ml_cka.baselines.campaign import evaluate

    args = SimpleNamespace(methods=["pgd"])
    manifest = {"images": [{"image_id": "missing"}]}
    with pytest.raises(RuntimeError, match="Target access blocked"):
        evaluate(args, manifest, tmp_path, {})


def test_kmeans_api_repair_matches_dependency_centers_and_gradients():
    from kmeans_pytorch import kmeans

    data = torch.tensor([[0.0, 0.0], [0.01, 0.02], [3.0, 3.0], [3.01, 3.02]])
    original = data.clone().requires_grad_()
    repaired = data.clone().requires_grad_()
    np.random.seed(42)
    labels_a, centers_a = kmeans(original, 2)
    np.random.seed(42)
    labels_b, centers_b = bounded_kmeans(repaired, 2)
    torch.testing.assert_close(labels_a, labels_b)
    torch.testing.assert_close(centers_a, centers_b)
    torch.testing.assert_close(
        torch.autograd.grad(centers_a.square().sum(), original)[0],
        torch.autograd.grad(centers_b.square().sum(), repaired)[0],
    )
