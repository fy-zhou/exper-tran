"""Explicit targeted controls and pinned official feature-alignment cores.

M/FOA use raw [0,255] delta sign ascent as upstream, NOT our MI optimizer.
No target evaluator, text generator or Pull/Push objective is imported here.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms as T

from .iterative import AttackConfig, attack_image
from .protocol import UPSTREAMS, sha, validate_method


def load_upstream(name, root):
    root = Path(root) / name
    pin = UPSTREAMS[name]["revision"]
    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        actual != pin
        or subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip()
    ):
        raise RuntimeError(f"Upstream must be clean at {pin}: {root}")
    source = root / "surrogates/FeatureExtractors/Base.py"
    spec = importlib.util.spec_from_file_location("kdd_upstream_" + name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, {"revision": pin, "file_sha256": sha(source)}


def bounded_kmeans(X, num_clusters, distance="euclidean", tol=1e-4, device="cpu", iter_limit=100):
    """API repair for official FOA's unsupported iter_limit argument.

    Same NumPy-random initialization and Euclidean Lloyd updates as dependency0.3.
    Cluster assignments detached; centroid means remain differentiable. Empty
    clusters and nonconvergence FAIL, never drop a local loss or return NaNs.
    """
    if distance != "euclidean" or not 1 < num_clusters <= X.shape[0]:
        raise ValueError("Invalid kmeans settings")
    X = X.float().to(device)
    centers = X[np.random.choice(len(X), num_clusters, replace=False)]
    for _ in range(iter_limit):
        with torch.no_grad():
            assignments = ((X[:, None] - centers[None, :]) ** 2).sum(-1).argmin(1)
        members = [X[assignments == i] for i in range(num_clusters)]
        if any(len(m) == 0 for m in members):
            raise RuntimeError("FOA kmeans empty cluster; retain failed attempt")
        previous = centers
        centers = torch.stack([m.mean(0) for m in members])
        shift = (centers.detach() - previous.detach()).square().sum(1).sqrt().sum()
        if shift.square() < tol:
            return assignments.cpu(), centers.cpu()
    raise RuntimeError("FOA kmeans did not converge within original 100-iteration request")


class TargetImageCosine:
    """Adapted PGD/MI objective: mean projected CLIP cosine to ONE target image.

    No source repulsion, class prototypes, multi-layer loss, DI or local crops.
    This is an explicitly named task adaptation, not a paper-faithful CE attack.
    """

    def __init__(self, models, reference):
        self.models = models
        with torch.no_grad():
            self.targets = [m(reference * 255).detach() for m in models]
        self.trace = []

    def __call__(self, image):
        loss = torch.stack(
            [
                1 - (m(image * 255) * t).sum(1).mean()
                for m, t in zip(self.models, self.targets, strict=True)
            ]
        ).mean()
        self.trace.append(float(loss.detach()))
        return loss


def run_feature_attack(
    method, models, clean, reference, *, steps, seed, upstream_root, clusters=10
):
    validate_method(method)
    if clean.shape != reference.shape or clean.shape != (1, 3, 224, 224):
        raise ValueError("Baseline runner requires one canonical224 RGB image")
    torch.manual_seed(seed)
    np.random.seed(seed % 2**32)
    if method in {"pgd", "mi_fgsm"}:
        objective = TargetImageCosine(models, reference)
        cfg = AttackConfig(method, steps, 16 / 255, 1 / 255, random_start=(method == "pgd"))
        adv = attack_image(clean, objective, cfg, seed=seed)
        return adv, {
            "loss_trace": objective.trace,
            "fidelity": "adapted_image_only",
            "objective": "mean_projected_CLIP_target_image_cosine",
            "backward_calls": steps,
            "upstream": None,
        }
    module, provenance = load_upstream(method, upstream_root)
    if method == "m_attack":
        extractor = module.EnsembleFeatureExtractor(models)
        objective = module.EnsembleFeatureLoss(models)
    else:
        # Official generate_adversarial_samples_foa_attack.py uses fixed K=10.
        # Never import FOAttack.py: it includes commercial API calls/target retries.
        module.kmeans = bounded_kmeans
        extractor = module.EnsembleFeatureExtractor_ot(models, cluster_number=clusters)
        objective = module.EnsembleFeatureLoss_OT_foa_attack(models, cluster_number=clusters)
    crop = T.RandomResizedCrop(224, scale=(0.5, 0.9))
    original = clean * 255
    target = reference * 255
    delta = torch.zeros_like(original)
    trace = []
    for step in range(steps):
        delta = delta.detach().requires_grad_(True)
        with torch.no_grad():
            objective.set_ground_truth(crop(target))
        x = original + delta
        # Official global diagnostic is also called BEFORE the local loss.
        # In FOA it updates dynamic ensemble history and MUST NOT be omitted.
        with torch.no_grad():
            f = extractor(x)
            global_score = objective(*f) if method == "foa_attack" else objective(f)
        f = extractor(crop(x))
        score = objective(*f) if method == "foa_attack" else objective(f)
        if score is None or score.ndim or not torch.isfinite(score):
            raise RuntimeError(f"Nonfinite upstream objective at step {step}")
        gradient = torch.autograd.grad(score, delta)[0]
        if not torch.isfinite(gradient).all() or gradient.abs().sum() == 0:
            raise RuntimeError(f"Invalid upstream input gradient at step {step}")
        trace.append(
            {
                "step": step,
                "global_score": float(global_score),
                "local_score": float(score.detach()),
                "gradient_l1": float(gradient.abs().sum()),
            }
        )
        delta = (delta + gradient.sign()).clamp(-16, 16)
    adv = ((original + delta) / 255).clamp(0, 1).detach()
    return adv, {
        "loss_trace": trace,
        "fidelity": "upstream_core_task_adapted",
        "objective": "local_to_local_projected_cosine"
        if method == "m_attack"
        else "global_CLS_plus_cluster_OT_dynamic_ensemble",
        "backward_calls": steps,
        "upstream": provenance,
        "foa_kmeans_api_repair": method == "foa_attack",
        "clusters": clusters,
    }
