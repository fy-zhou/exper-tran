# ruff: noqa: E402
"""Task-matched additional controls; native optimizer semantics, explicit adapters."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "prepared"))
from reproduction import official_module

from primary_ml_cka.baselines.clip_features import ClipFeatures

ATTACKVLM_PIN = "bdc5c7d34340bdf266e7dbcc3f5583d9621f263a"
CWA_PIN = "dbac7e7e32844f440876b57e6c0d3bbb95040a80"
CLASSES = [
    "goldfish",
    "monarch butterfly",
    "pineapple",
    "acoustic guitar",
    "laptop",
    "espresso",
    "volcano",
    "rocking chair",
    "soccer ball",
    "school bus",
]
CACHE = Path("/ephemeral/kdd2027/hf-cache/hub")


def verified(root, pin):
    import subprocess

    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "diff", "--stat"], text=True)
    if actual != pin or dirty:
        raise RuntimeError("Upstream source pin/cleanliness mismatch")


def text_features(model, text):
    from transformers import CLIPTokenizer

    tokens = CLIPTokenizer.from_pretrained(
        model.spec["repo_id"],
        revision=model.spec["revision"],
        cache_dir=str(CACHE),
        local_files_only=True,
    )(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        result = model.model.text_model(**{k: v.cuda() for k, v in tokens.items()})
        return F.normalize(model.model.text_projection(result.pooler_output).float(), dim=-1)


class ZeroShot(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features
        self.register_buffer(
            "texts", text_features(features, ["A photo of a " + c + "." for c in CLASSES])
        )
        self.forwards = 0

    def forward(self, image):
        self.forwards += 1
        # SSA's transformed values intentionally need not lie inside [0,1].
        # Native CLIP normalization without the image-pair adapter's input clamp.
        from torchvision import transforms as T

        x = T.CenterCrop(224)(
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC, antialias=True)(image)
        )
        mean = x.new_tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        std = x.new_tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        result = self.features.model.vision_model(
            pixel_values=((x - mean) / std).to(torch.bfloat16)
        )
        z = F.normalize(self.features.model.visual_projection(result.pooler_output).float(), dim=-1)
        return self.features.model.logit_scale.float().exp() * (z @ self.texts.T)


def make(method, steps):
    if method.startswith("attackvlm_"):
        verified(Path("/home/ubuntu/baseline-upstreams/attackvlm"), ATTACKVLM_PIN)
        models = [ClipFeatures("B32", CACHE)]
        return models, {
            "upstream": ATTACKVLM_PIN,
            "variant": method,
            "steps": steps,
            "alpha": 1 / 255,
            "epsilon": 16 / 255,
            "random_start": False,
            "fidelity": (
                "MF-ii official loop / MF-it paper objective adapter; "
                "HF BF16 CLIP parameter execution"
            ),
        }
    if method == "ssa_cwa_clip":
        import typing

        verified(Path("/home/ubuntu/baseline-upstreams/common_weakness"), CWA_PIN)
        features = [ClipFeatures(k, CACHE) for k in ["B16", "B32", "Laion"]]
        models = [ZeroShot(m) for m in features]
        root = Path("/home/ubuntu/baseline-upstreams/common_weakness")
        parent = official_module(
            root / "attacks/AdversarialInput/AdversarialInputBase.py",
            allowed_imports=("torch", "abc", "typing", "math"),
        )
        module = official_module(
            root / "attacks/AdversarialInput/SpectrumSimulationAttack.py",
            classes={"dct", "idct", "dct_2d", "idct_2d", "SSA_CommonWeakness"},
            allowed_imports=("torch", "numpy", "typing"),
            injected={
                "AdversarialInputAttacker": parent.AdversarialInputAttacker,
                "clamp": lambda x, min_value=0, max_value=1: torch.clamp(
                    x, min=min_value, max=max_value
                ),
                "List": list,
                "Callable": typing.Callable,
                "nn": nn,
            },
        )
        attacker = module.SSA_CommonWeakness(
            models, total_step=steps, targeted_attack=True, epsilon=16 / 255, random_start=False
        )
        return (features, models, attacker), {
            "upstream": CWA_PIN,
            "steps": steps,
            "fidelity": "official SSA-CWA core with CLIP ten-class zero-shot classifier adaptation",
            "spectrum_samples": 20,
            "rho": 0.5,
            "sigma": 16,
            "inner_step_size": 250,
            "outer_ksi": 16 / 255 / 5,
            "mu": 1,
            "targeted": True,
            "random_start": False,
        }
    raise ValueError(method)


def generate(method, bundle, clean, reference, row, steps):
    if method == "ssa_cwa_clip":
        _, models, attacker = bundle
        before = sum(m.forwards for m in models)
        adv = attacker(clean.clone(), torch.tensor([row["target"] - 1], device=clean.device))
        return adv.detach(), {"model_forward_calls": sum(m.forwards for m in models) - before}
    model = bundle[0]
    if method == "attackvlm_mfii":
        with torch.no_grad():
            target = model(reference * 255)
    elif method == "attackvlm_mfit":
        target = text_features(model, ["A photo of a " + CLASSES[row["target"] - 1] + "."])
    else:
        raise ValueError(method)
    # Exact source MF-ii optimizer in raw pixel units: zero start, cosine ascent,
    # alpha1, projection16, no momentum/DI/crop ensemble. MF-it changes target feature.
    original = clean * 255
    delta = torch.zeros_like(original, requires_grad=True)
    losses = []
    before = model.forwards
    for _ in range(steps):
        similarity = (model(original + delta) * target).sum(-1).mean()
        grad = torch.autograd.grad(similarity, delta)[0]
        if not torch.isfinite(grad).all() or grad.abs().sum() == 0:
            raise RuntimeError("Invalid gradient")
        losses.append(float(similarity.detach()))
        delta = (delta.detach() + grad.sign()).clamp(-16, 16).requires_grad_()
    return ((original + delta) / 255).clamp(0, 1).detach(), {
        "cosine_similarity_trace": losses,
        "model_forward_calls": model.forwards - before,
    }
