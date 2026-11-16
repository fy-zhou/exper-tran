"""Pinned upstream CLIP adapters. Parameters BF16; images/statistics FP32.

Preprocessing follows M-Attack/FOA source, WITHOUT Ours' STE quantization.
"""

from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms as T

from primary_ml_cka.models.common.precision import assert_bf16_unquantized, quantization_audit

from .protocol import PROXIES


def audit_model(model, spec, role):
    assert_bf16_unquantized(model, model_name=spec["repo_id"])
    counts, sizes, devices, components = defaultdict(int), defaultdict(int), set(), {}
    for name, parameter in model.named_parameters():
        counts[str(parameter.dtype)] += parameter.numel()
        sizes[str(parameter.dtype)] += parameter.numel() * parameter.element_size()
        devices.add(str(parameter.device))
        component = name.split(".")[0]
        components.setdefault(component, set()).add(str(parameter.dtype))
    return {
        "model": spec,
        "role": role,
        "parameter_counts": dict(counts),
        "parameter_bytes": dict(sizes),
        "total_parameters": sum(counts.values()),
        "components": {k: sorted(v) for k, v in components.items()},
        "devices": sorted(devices),
        "quantization": quantization_audit(model),
        "allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
        "reserved": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
        "status": "PASS",
    }


class ClipFeatures(nn.Module):
    def __init__(self, name, cache_dir, device="cuda"):
        super().__init__()
        from transformers import CLIPModel

        self.spec = PROXIES[name]
        self.model = (
            CLIPModel.from_pretrained(
                self.spec["repo_id"],
                revision=self.spec["revision"],
                cache_dir=str(cache_dir),
                local_files_only=True,
                dtype=torch.bfloat16,
                attn_implementation="eager",
            )
            .to(device=device, dtype=torch.bfloat16)
            .eval()
        )
        self.model.requires_grad_(False)
        self.audit = audit_model(self.model, self.spec, "baseline_proxy")
        self.forwards = 0
        self.resize = T.Resize(224, interpolation=T.InterpolationMode.BICUBIC, antialias=True)
        self.crop = T.CenterCrop(224)

    def preprocess(self, raw255):
        # Match upstream order: resize -> clamp/255 -> center crop -> normalization.
        x = self.crop(self.resize(raw255).clamp(0, 255) / 255)
        mean = x.new_tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        std = x.new_tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        return ((x - mean) / std).to(torch.bfloat16)

    def forward(self, raw255):
        self.forwards += 1
        output = self.model.vision_model(pixel_values=self.preprocess(raw255))
        features = self.model.visual_projection(output.pooler_output).float()
        return F.normalize(features, dim=1)

    def global_local_features(self, raw255):
        self.forwards += 1
        features = self.model.vision_model(pixel_values=self.preprocess(raw255)).last_hidden_state
        # The official FOA implementation normalizes LOCAL tokens across dim=1
        # (token axis). Preserve this unusual choice; do not silently "correct" it.
        return F.normalize(features[:, 0, :].float(), dim=1), F.normalize(
            features[:, 1:, :].float(), dim=1
        )
