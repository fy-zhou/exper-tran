"""Pinned paper cores for IAU-FOA, Co-Attack, SGA in isolated child processes.

Generation preserves joint surrogate text/image steps for Co/SGA and saves the
adversarial text. Image-only transfer to the MLLM is explicitly a different
input threat model from the papers' joint retrieval evaluation.
"""

import importlib
import importlib.util
import logging
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms as T

from primary_ml_cka.baselines.clip_features import ClipFeatures, audit_model
from primary_ml_cka.baselines.feature_attacks import load_upstream
from primary_ml_cka.baselines.protocol import PROXIES, UPSTREAMS, sha

EXTRAS = {
    "DINO": {
        "repo_id": "facebook/dinov2-base",
        "revision": "f9e44c814b77203eaa57a6bdbbd535f21ede1415",
        "family": "dinov2",
    },
    "InternVL1": {
        "repo_id": "OpenGVLab/InternVL3-1B-hf",
        "revision": "014c0583a0d4bedf29fbe2dbff4f865eb998e171",
        "family": "internvl",
    },
    "BERT": {
        "repo_id": "google-bert/bert-base-uncased",
        "revision": "86b5e0934494bd15c9632b12f734a8a67f723594",
        "family": "bert",
    },
}


KMEANS_PIN = "f7f36bd1cb4e3a761d73d584866d0a9c6b4d2805"


def pinned_kmeans(root):
    repo = Path(root) / "kmeans_pinned"
    actual = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        actual != KMEANS_PIN
        or subprocess.check_output(
            ["git", "-C", str(repo), "diff", "HEAD", "--name-only"], text=True
        ).strip()
    ):
        raise RuntimeError("IAU KMeans dependency pin mismatch")
    package = repo / "kmeans_pytorch"
    spec = importlib.util.spec_from_file_location(
        "kdd_kmeans_pinned", package / "__init__.py", submodule_search_locations=[str(package)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.kmeans


def family_legal(proxy_names, target_family):
    if any(EXTRAS.get(n, {"family": "clip"})["family"] == target_family for n in proxy_names):
        raise ValueError(f"Target-family proxy collision: {target_family}")
    return True


def file_module(method, relative, root):
    path = Path(root) / method
    actual = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        actual != UPSTREAMS[method]["revision"]
        or subprocess.check_output(
            ["git", "-C", str(path), "diff", "HEAD", "--name-only"], text=True
        ).strip()
    ):
        raise RuntimeError("Upstream source differs from frozen revision")
    source = path / relative
    spec = importlib.util.spec_from_file_location("kdd_" + method + source.stem, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExtraVision(nn.Module):
    def __init__(self, name, cache):
        super().__init__()
        from transformers import AutoModel, AutoModelForImageTextToText

        spec = EXTRAS[name]
        cls = AutoModelForImageTextToText if name == "InternVL1" else AutoModel
        self.model = (
            cls.from_pretrained(
                spec["repo_id"],
                revision=spec["revision"],
                cache_dir=str(cache),
                local_files_only=True,
                dtype=torch.bfloat16,
                attn_implementation="eager",
            )
            .to("cuda", torch.bfloat16)
            .eval()
        )
        self.model.requires_grad_(False)
        self.audit = audit_model(self.model, spec, "baseline_proxy")
        self.name, self.forwards = name, 0
        size = 448 if name == "InternVL1" else 224
        self.normalize = T.Compose(
            [
                T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                T.Lambda(lambda x: x.clamp(0, 255) / 255),
                T.Normalize(
                    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
                ),
            ]
        )

    def global_local_features(self, x):
        self.forwards += 1
        x = self.normalize(x).to(torch.bfloat16)
        if self.name == "InternVL1":
            features = self.model.model.vision_tower(pixel_values=x).last_hidden_state
        else:
            features = self.model(pixel_values=x).last_hidden_state
        # Preserve authors' token-axis normalization (not our spatial mean).
        return F.normalize(features[:, 0].float(), dim=1), F.normalize(
            features[:, 1:].float(), dim=1
        )


class ClipTextImage(nn.Module):
    """Official CLIP retrieval adapter: decode BERT IDs, then CLIP-tokenize."""

    def __init__(self, clip, bert_tokenizer, clip_tokenizer, method):
        super().__init__()
        self.clip = clip
        self.bert_tokenizer = bert_tokenizer
        self.clip_tokenizer = clip_tokenizer
        self.method = method
        self.text_forwards = 0

    def inference_image(self, normalized):
        self.clip.forwards += 1
        output = self.clip.model.vision_model(pixel_values=normalized.to(torch.bfloat16))
        raw = self.clip.model.visual_projection(output.pooler_output).float()
        norm = F.normalize(raw, dim=-1)
        return {"image_embed": norm if self.method == "co_attack" else raw, "image_feat": norm}

    def inference_text(self, inputs):
        texts = [
            self.bert_tokenizer.decode(t)
            .replace("[PAD]", "")
            .replace("[CLS]", "")
            .replace("[SEP]", "")
            .strip()
            for t in inputs.input_ids
        ]
        encoded = self.clip_tokenizer(
            texts, padding="max_length", truncation=True, max_length=77, return_tensors="pt"
        ).to("cuda")
        self.text_forwards += 1
        output = self.clip.model.text_model(**encoded)
        raw = self.clip.model.text_projection(output.pooler_output).float()
        norm = F.normalize(raw, dim=-1)
        return {"text_embed": norm if self.method == "co_attack" else raw, "text_feat": norm}


def make_bundle(method, cache, upstream, proxy_names=None):
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    if method == "iau_foa":
        root = Path(upstream) / method
        module, provenance = load_upstream(method, upstream)
        module.kmeans = pinned_kmeans(upstream)
        package_name = "kdd_native_iau"
        spec = importlib.util.spec_from_file_location(
            package_name,
            root / "iau_foa/__init__.py",
            submodule_search_locations=[str(root / "iau_foa")],
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        spec.loader.exec_module(package)
        attack_module = importlib.import_module(package_name + ".attack")
        resize = importlib.import_module(package_name + ".resize")
        resize.patch_resize()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        names = proxy_names or ["B16", "B32", "Laion", "InternVL1", "DINO"]
        models = [ClipFeatures(n, cache) if n in PROXIES else ExtraVision(n, cache) for n in names]
        extractor = module.EnsembleFeatureExtractor_ot(models, cluster_number=10)
        loss = module.EnsembleFeatureLoss_OT_foa_attack(
            models,
            cluster_number=10,
            local_weight=2.0,
            uot_reg_m=0.2,
            uot_normalize=True,
            uot_adaptive=True,
            uot_adaptive_gamma=1.0,
            uot_adaptive_norm=True,
            eps=0.1,
        )
        return SimpleNamespace(
            method=method,
            models=models,
            extractor=extractor,
            loss=loss,
            attack=attack_module.attack,
            audits=[m.audit for m in models],
            provenance={
                **provenance,
                "attack_sha256": sha(root / "iau_foa/attack.py"),
                "resize_sha256": sha(root / "iau_foa/resize.py"),
            },
        )
    clip = ClipFeatures("B16", cache)
    spec = EXTRAS["BERT"]
    tokenizer = AutoTokenizer.from_pretrained(
        spec["repo_id"], revision=spec["revision"], cache_dir=str(cache), local_files_only=True
    )
    mlm = (
        AutoModelForMaskedLM.from_pretrained(
            spec["repo_id"],
            revision=spec["revision"],
            cache_dir=str(cache),
            local_files_only=True,
            dtype=torch.bfloat16,
        )
        .to("cuda", torch.bfloat16)
        .eval()
        .requires_grad_(False)
    )
    clip_spec = PROXIES["B16"]
    ct = AutoTokenizer.from_pretrained(
        clip_spec["repo_id"],
        revision=clip_spec["revision"],
        cache_dir=str(cache),
        local_files_only=True,
    )
    net = ClipTextImage(clip, tokenizer, ct, method)
    normalize = T.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )
    if method == "sga":
        code = file_module(method, "attacker.py", upstream)
        text = code.TextAttacker(
            mlm,
            tokenizer,
            cls=False,
            max_length=77,
            number_perturbation=1,
            topk=10,
            threshold_pred_score=0.3,
        )
        image = code.ImageAttacker(normalize, eps=16 / 255, steps=50, step_size=1 / 255)
        attack = code.SGAttacker(net, image, text)
    elif method == "co_attack":
        text_code = file_module(method, "attack/bert_attack.py", upstream)
        image_code = file_module(method, "attack/imageAttack.py", upstream)
        code = file_module(method, "attack/multimodalAttack.py", upstream)
        text = text_code.BertAttack(mlm, tokenizer, cls=False)
        image = image_code.ImageAttacker(16 / 255, preprocess=normalize, bounding=(0, 1), cls=False)
        attack = code.MultiModalAttacker(net, image, text, tokenizer, cls=False)
    else:
        raise ValueError("No fallback for unsupported native method")
    return SimpleNamespace(
        method=method,
        models=[clip],
        mlm=mlm,
        net=net,
        attack=attack,
        image=image,
        audits=[clip.audit, audit_model(mlm, spec, "MLM_auxiliary")],
    )


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append({"level": record.levelname, "message": record.getMessage()})


def run(bundle, clean, reference, captions, *, steps, seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if bundle.method == "iau_foa":
        config = SimpleNamespace(
            model=SimpleNamespace(input_res=224, crop_scale=[0.5, 0.9]),
            attack=SimpleNamespace(steps=steps, alpha=1.0, epsilon=16),
            via=SimpleNamespace(num_scales=5, wb_spread=0.5),
        )
        logger, capture = logging.getLogger("iau_foa"), Capture()
        logger.addHandler(capture)
        logger.setLevel(logging.INFO)
        try:
            image = bundle.attack(
                config,
                bundle.extractor,
                bundle.loss,
                clean * 255,
                reference * 255,
                tag="frozen-image",
            )
        finally:
            logger.removeHandler(capture)
        return image.detach(), {
            "native_log": capture.records,
            "skipped_copy_warnings": sum("skip copy" in r["message"] for r in capture.records),
            "skipped_step_warnings": sum("skip step" in r["message"] for r in capture.records),
        }
    if not captions:
        raise ValueError("Co/SGA need dataset-grounded source text; never fabricate captions")
    if bundle.method == "sga":
        bundle.image.steps = steps
        image, texts = bundle.attack.attack(
            clean,
            captions,
            [0] * len(captions),
            device="cuda",
            max_length=77,
            scales=[0.5, 0.75, 1.25, 1.5],
        )
    else:
        image, texts = bundle.attack.run_before_fusion(
            clean, captions[:1], adv=4, num_iters=steps, max_length=77, alpha=3.0
        )
    return image.detach(), {
        "adversarial_texts": texts,
        "target_input": "image_only_transfer_of_joint_generated_attack",
        "original_goal": "untargeted_joint_image_text_retrieval",
        "not_a_faithful_targeted_ten_class_attack": True,
    }
