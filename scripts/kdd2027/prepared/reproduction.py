"""Prepared-only adapters. No imports here load weights, use CUDA or access a target.

Reviewed upstream computational classes are executed from pinned clean checkouts;
API entrypoints and their config/logging side effects are never imported.
"""

import ast
import hashlib
import subprocess
import types
from pathlib import Path

PINS = {
    "m_attack_v2": "8faca6d95e1caa190d676575489ab81bb31e4d03",
    "anyattack": "91b209120c61261a51a45ce217dae7f1daac081d",
}
BACKBONES = {
    "clip_b16": ("openai/clip-vit-base-patch16", "57c216476eefef5ab752ec549e440a49ae4ae5f3"),
    "clip_b32": ("openai/clip-vit-base-patch32", "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"),
    "clip_laion_b32": (
        "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        "1a25a446712ba5ee05982a381eed697ef9b435cf",
    ),
    "clip_laion_g14": (
        "laion/CLIP-ViT-G-14-laion2B-s12B-b42K",
        "4b0305adc6802b2632e11cbe6606a9bdd43d35c9",
    ),
}
OPENAI_B32_SHA = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_source(root, method):
    root = Path(root).resolve()
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    if head != PINS[method] or dirty:
        raise RuntimeError(f"Unpinned or modified {method} source: {head}")
    return root


def official_module(path, *, classes=None, allowed_imports=(), injected=None):
    """Keep exact selected class/function AST bodies; remove unrelated entrypoints/imports."""
    tree = ast.parse(Path(path).read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef):
            if classes is None or node.name in classes:
                nodes.append(node)
        elif isinstance(node, ast.Import | ast.ImportFrom):
            names = (
                [x.name for x in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            if all(n.split(".")[0] in allowed_imports for n in names):
                nodes.append(node)
    module = types.ModuleType("_reviewed_" + Path(path).stem)
    module.__dict__.update(injected or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


def finite_bounded(clean, adv, epsilon=16 / 255):
    import torch

    if adv.shape != clean.shape or not torch.isfinite(adv).all():
        raise RuntimeError("Invalid attack output")
    if adv.min() < 0 or adv.max() > 1 or (adv - clean).abs().max() > epsilon + 1e-7:
        raise RuntimeError("Attack output violates pixel bounds")


def v2_core(root):
    import torch

    root = verify_source(root, "m_attack_v2")
    # Only dependencies used by the exact core classes; no wandb/API/config import.
    base = official_module(
        root / "attack/base.py",
        allowed_imports=("typing", "einops", "torch", "tqdm"),
        injected={"MainConfig": object, "log_metrics": lambda *a, **kw: None},
    )
    loss = official_module(
        root / "surrogates/loss.py",
        allowed_imports=("typing", "einops", "torch"),
        injected={"BaseFeatureExtractor": torch.nn.Module},
    )
    return base, loss


class V2Runner:
    """Pinned native multipass/weighted loss; no paper-only patch-momentum invented."""

    def __init__(self, upstream, extractors, *, steps=300, multi_pass=10):
        from omegaconf import OmegaConf
        from torchvision import transforms as T

        base, loss = v2_core(upstream)
        if steps not in (2, 50, 300) or multi_pass != 10:
            raise ValueError("Only named smoke/matched50/native300, native K10")
        self.steps, self.multi_pass = steps, multi_pass
        cfg = OmegaConf.create(
            {
                "optim": {
                    "steps": steps,
                    "alpha": 0.005,
                    "epsilon": 16,
                    "optimizer": "adam",
                    "momentum": 0.9,
                }
            }
        )
        self.loss = loss.EnsWeightedMultiAlignmentLoss(extractors, beta=0.3)
        # Exact source/target transforms in official single/parallel generator.
        self.attack = base.PGDAttackMutiPass(
            cfg=cfg,
            ensemble_loss=self.loss,
            source_crop=[T.RandomResizedCrop(224, scale=(0.5, 1.0)) for _ in range(3)],
            target_crop=T.Compose(
                [
                    T.RandomResizedCrop(224, scale=(0.95, 1.0)),
                    T.RandomHorizontalFlip(),
                    T.RandomRotation(10),
                ]
            ),
            change_iters=[40, 260],
        )

    def __call__(self, clean, targets):
        import torch

        if clean.shape != (1, 3, 224, 224) or targets.shape != (3, 3, 224, 224):
            raise ValueError("Require one source and primary+two distinct retrieved target images")
        if not torch.isfinite(clean).all() or not torch.isfinite(targets).all():
            raise ValueError("Nonfinite input")
        # Disable default FP16 autocast/scaler. Models/inputs are explicitly BF16/FP32.
        adv = self.attack.attack(
            0,
            clean,
            targets,
            target_num=3,
            amp=False,
            log_wandb=False,
            disable_tqdm=True,
            multi_pass_num=self.multi_pass,
        )
        finite_bounded(clean, adv)
        return adv.detach()


def load_v2_extractors(cache, device="cuda"):
    import torch
    from torch import nn
    from torchvision import transforms as T
    from transformers import CLIPModel

    from primary_ml_cka.baselines.clip_features import audit_model

    class Extractor(nn.Module):
        def __init__(self, name, spec):
            super().__init__()
            repo, revision = spec
            self.model = (
                CLIPModel.from_pretrained(
                    repo,
                    revision=revision,
                    cache_dir=str(cache),
                    local_files_only=True,
                    dtype=torch.bfloat16,
                    attn_implementation="eager",
                )
                .vision_model.to(device=device, dtype=torch.bfloat16)
                .eval()
            )
            self.model.requires_grad_(False)
            self.audit = audit_model(
                self.model, {"repo_id": repo, "revision": revision}, "v2_proxy"
            )
            self.normalizer = T.Compose(
                [
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.Normalize(
                        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
                    ),
                ]
            )
            self.forwards = 0

        def forward(self, image, return_dict=False):
            self.forwards += 1
            result = self.model(
                pixel_values=self.normalizer(image).to(torch.bfloat16),
                output_hidden_states=True,
                return_dict=True,
            )
            return result if return_dict else result.pooler_output.float()

    return [Extractor(k, v) for k, v in BACKBONES.items()]


def load_anyattack(upstream, clip_weights, decoder_weights, decoder_sha256, device="cuda"):
    import torch

    from primary_ml_cka.baselines.clip_features import audit_model

    root = verify_source(upstream, "anyattack")
    if checksum(clip_weights) != OPENAI_B32_SHA:
        raise RuntimeError("Require official OpenAI B32 TorchScript checkpoint checksum")
    if len(decoder_sha256) != 64 or checksum(decoder_weights) != decoder_sha256:
        raise RuntimeError("Decoder checkpoint checksum mismatch")
    clip_core = official_module(
        root / "models/clip/model.py", allowed_imports=("collections", "typing", "numpy", "torch")
    )
    state = torch.jit.load(str(clip_weights), map_location="cpu").state_dict()
    encoder = clip_core.build_model(dict(state)).to(device=device, dtype=torch.bfloat16).eval()
    decoder_core = official_module(
        root / "models/model.py",
        classes={"get_group_norm", "EfficientAttention", "ResBlock", "UpBlock", "Decoder"},
        allowed_imports=("os", "torch", "torchvision"),
    )
    decoder = decoder_core.Decoder(embed_dim=512)
    checkpoint = torch.load(decoder_weights, map_location="cpu", weights_only=True)
    state = checkpoint["decoder_state_dict"]
    state = {k.removeprefix("module."): v for k, v in state.items()}
    decoder.load_state_dict(state, strict=True)
    decoder = decoder.to(device=device, dtype=torch.bfloat16).eval()
    encoder.requires_grad_(False)
    decoder.requires_grad_(False)
    # Upstream decoder explicitly casts activation float before fc; bridge only its
    # input to parameter dtype. No model parameter is converted to FP32.
    decoder.fc.register_forward_pre_hook(lambda module, args: (args[0].to(torch.bfloat16),))

    # OpenAI custom LayerNorm calls super on float input. Accumulate normalization
    # in FP32 using temporary casts while stored parameters remain strictly BF16.
    def layernorm(self, x):
        return torch.nn.functional.layer_norm(
            x.float(), self.normalized_shape, self.weight.float(), self.bias.float(), self.eps
        ).to(x.dtype)

    for m in encoder.modules():
        if isinstance(m, clip_core.LayerNorm):
            m.forward = types.MethodType(layernorm, m)
    audits = [
        audit_model(
            encoder, {"repo_id": "OpenAI/ViT-B-32", "revision": OPENAI_B32_SHA}, "anyattack_encoder"
        ),
        audit_model(
            decoder,
            {"repo_id": "AnyAttack/decoder", "revision": decoder_sha256},
            "anyattack_generator",
        ),
    ]
    return encoder, decoder, audits


def anyattack_generate(encoder, decoder, clean, target):
    import torch

    mean = target.new_tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = target.new_tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    with torch.inference_mode():
        embedding = encoder.encode_image(((target - mean) / std).to(torch.bfloat16))
        noise = decoder(embedding).float().clamp(-16 / 255, 16 / 255)
        adv = (clean + noise).clamp(0, 1)
    finite_bounded(clean, adv)
    return adv
