import pytest
import torch

from primary_ml_cka.baselines import AttackConfig, attack_image


@pytest.mark.parametrize("method", ["pgd", "mi_fgsm"])
def test_targeted_quadratic_descent_and_projection(method):
    x = torch.full((2, 3, 8, 8), 0.5)
    cfg = AttackConfig(method=method, steps=10, epsilon=16 / 255, step_size=2 / 255)
    def objective(z):
        return (z - 0.9).square().mean()
    y = attack_image(x, objective, cfg)
    assert y.shape == x.shape
    assert torch.all(y >= 0) and torch.all(y <= 1)
    assert torch.max((y - x).abs()) <= cfg.epsilon + 1e-6
    assert objective(y) < objective(x)


def test_seeded_random_start():
    x = torch.full((1, 3, 4, 4), 0.5)
    cfg = AttackConfig("pgd", 1, 8 / 255, 1 / 255, random_start=True)
    a = attack_image(x, lambda z: (z * 0).sum(), cfg, seed=123)
    b = attack_image(x, lambda z: (z * 0).sum(), cfg, seed=123)
    c = attack_image(x, lambda z: (z * 0).sum(), cfg, seed=124)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


@pytest.mark.parametrize("value", [0, -1])
def test_invalid_steps(value):
    with pytest.raises(ValueError):
        AttackConfig("pgd", value, 8 / 255, 1 / 255)


def test_reject_unknown_method():
    with pytest.raises(ValueError):
        AttackConfig("foa_attack", 5, 8 / 255, 1 / 255)


def test_no_gradient_fails_explicitly():
    cfg = AttackConfig("pgd", 2, 8 / 255, 1 / 255)
    with pytest.raises(RuntimeError):
        attack_image(torch.ones(1, 3, 2, 2) * 0.5, lambda _: torch.tensor(1.0), cfg)
