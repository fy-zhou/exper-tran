# ruff: noqa: E402, E741
"""Uniform4 PP with label-free disjoint48-reference neighborhoods; native tasks."""

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

from prepare import BASE, CACHE, OUT, REPO, V7, expansion, manifests

from primary_ml_cka.baselines.protocol import atomic, digest, freeze, sha, valid_receipt

FAMILIES = {"qwen": "Qwen3.5-27B", "internvl": "InternVL3.5-14B-HF", "gemma4": "Gemma 4 31B-it"}
POOLS = {
    "qwen": ["CLIP ViT-L/14", "InternVL3.5-2B-HF", "Gemma 4 E2B-it"],
    "internvl": ["CLIP ViT-L/14", "Qwen3.5-2B", "Gemma 4 E2B-it"],
    "gemma4": ["CLIP ViT-L/14", "Qwen3.5-2B", "InternVL3.5-2B-HF"],
}


def setup(dataset, family, n, engineering=False):
    data = manifests()[dataset]
    data = {**data, "images": data["images"][:n], "N_images": n}
    config = {
        "method": "ours_uniform4_neighborhood48",
        "dataset": dataset,
        "variant": family,
        "steps": 2 if engineering else 50,
        "engineering": engineering,
        "N": n,
        "seed": 42,
        "epsilon": 16 / 255,
        "alpha": 1 / 255,
        "canvas": 224,
        "pull": 0.5,
        "push": 0.5,
        "DI": True,
        "CLS": 0,
        "patch": 0,
        "momentum": 1,
        "parameter_dtype": "bfloat16",
        "smoke_only": False,
        "pilot": False,
        "reference_count": 48,
        "primary_goal": "native_task_quality_and_representation_transfer",
    }
    sources = {
        str(p.relative_to(REPO)): sha(p)
        for p in [
            Path(__file__),
            Path(__file__).with_name("prepare.py"),
            V7 / "src/vision_features.py",
            V7 / "src/attack_optimizers.py",
            V7 / "stage12_joint24h/device_residency.py",
        ]
    }
    lock = {
        "config": config,
        "manifest": data,
        "target": FAMILIES[family],
        "models": json.loads((OUT / "proxy_revisions.json").read_text()),
        "sources": sources,
        "reference_banks_sha256": sha(OUT / "reference_banks.json"),
        "reference_pool_sha256": sha(OUT / "reference_pool.json"),
    }
    for name in POOLS[family]:
        if lock["models"][name]["family"] == family:
            raise ValueError("Target family leakage")
    run = OUT / "runs" / f"{dataset}_{family}_{n}_{digest(lock)[:14]}"
    freeze(run / "lock.json", lock)
    freeze(run / "analysis_input/extension/manifests" / f"{dataset}.json", data)
    return run, config, lock, data


def init_proxy(name, root):
    import torch

    sys.path.insert(0, str(V7 / "src"))
    import vision_features

    w = expansion.api()

    def audit(model, **kwargs):
        spec = json.loads((OUT / "proxy_revisions.json").read_text())[name]
        result = w.oldworker.audit_model(model, spec, "native_task_proxy")
        atomic(root / "audits" / f"proxy_{digest(name)[:12]}.json", result)
        return result

    vision_features.audit_loaded_model = audit
    m = vision_features.VisionProxy(name, [0.25, 0.5, 0.75, 1.0], patch_k=1)
    spec = importlib.util.spec_from_file_location(
        "native_residency", V7 / "stage12_joint24h/device_residency.py"
    )
    residency = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(residency)
    residency.install(m)
    m.to("cuda")
    m.model.requires_grad_(False)
    atomic(
        root / "audits" / f"resident_{digest(name)[:12]}.json",
        w.oldworker.audit_model(m.model, m.spec, "native_task_proxy_resident"),
    )
    assert all(p.dtype == torch.bfloat16 for p in m.model.parameters() if p.is_floating_point())
    return m


def reference_features(m):
    import numpy as np
    import torch

    w = expansion.api()
    pool = json.loads((OUT / "reference_pool.json").read_text())
    key = digest([m.spec, m.layers, sha(OUT / "reference_pool.json"), "native_spatial_l2"])
    path = OUT / "cache" / f"{key}.npz"
    receipt = path.with_suffix(".json")
    if path.exists() and receipt.exists():
        if sha(path) != json.loads(receipt.read_text())["sha256"]:
            raise ValueError("Cache hash mismatch")
        with np.load(path) as d:
            return {l: torch.from_numpy(d[f"z_{l}"]).cuda() for l in m.layers}
    features = {l: [] for l in m.layers}
    with torch.inference_mode():
        for r in pool["images"]:
            if sha(r["path"]) != r["sha256"]:
                raise ValueError("Changed reference")
            values = m.features(w.legacy.load_image(r["path"]))
            for l in m.layers:
                features[l].append(values[l][0].cpu().numpy()[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **{f"z_{l}": np.stack(z) for l, z in features.items()})
    os.replace(temp, path)
    atomic(
        receipt,
        {
            "sha256": sha(path),
            "model": m.spec,
            "layers": m.layers,
            "pool": sha(OUT / "reference_pool.json"),
        },
    )
    return {l: torch.from_numpy(np.stack(z)).cuda() for l, z in features.items()}


def import_prefix(w, run, lock, data):
    if lock["config"]["N"] in [3, 10]:
        return
    for oldpath in (OUT / "runs").glob("*/lock.json"):
        old = json.loads(oldpath.read_text())
        a = dict(old["config"])
        b = dict(lock["config"])
        a.pop("N")
        b.pop("N")
        if (
            a != b
            or old["sources"] != lock["sources"]
            or old["reference_banks_sha256"] != lock["reference_banks_sha256"]
        ):
            continue
        oldrows = {r["image_id"]: r for r in old["manifest"]["images"]}
        for row in data["images"]:
            if oldrows.get(row["image_id"]) != row:
                continue
            src = w.oldworker.receipt_path(oldpath.parent, row)
            dest = w.oldworker.receipt_path(run, row)
            if not valid_receipt(src, digest([old, row])) or valid_receipt(
                dest, digest([lock, row])
            ):
                continue
            value = json.loads(src.read_text())
            value.update(
                signature=digest([lock, row]), reuse_receipt=str(src), reuse_receipt_sha256=sha(src)
            )
            atomic(dest, value)


def attack(run, config, lock, data):
    import torch
    import torch.nn.functional as F
    from attack_optimizers import differentiable_resize_pad, normalize_pixel_gradients

    w = expansion.api()
    import_prefix(w, run, lock, data)
    pending = [
        r
        for r in data["images"]
        if not valid_receipt(w.oldworker.receipt_path(run, r), digest([lock, r]))
    ]
    if not pending:
        return
    models = [init_proxy(name, run) for name in POOLS[config["variant"]]]
    refs = {m.name: reference_features(m) for m in models}
    banks = json.loads((OUT / "reference_banks.json").read_text())["banks"]
    try:
        for row in pending:
            if sha(row["clean"]) != row["clean_sha256"]:
                raise ValueError("Changed input")
            start = time.monotonic()
            clean = w.legacy.load_image(row["clean"])
            bank = banks[row["image_id"]]
            centers = {
                m.name: {
                    l: [
                        F.normalize(refs[m.name][l][bank[k]].float().mean(0), dim=0)
                        for k in ["source_indices", "target_indices"]
                    ]
                    for l in m.layers
                }
                for m in models
            }
            rng = torch.Generator(device="cuda").manual_seed(
                int(digest([42, row["image_id"], "start"])[:15], 16)
            )
            di = torch.Generator().manual_seed(int(digest([42, row["image_id"], "DI"])[:15], 16))
            adv = (
                (clean + torch.empty_like(clean).uniform_(-16 / 255, 16 / 255, generator=rng))
                .clamp(0, 1)
                .detach()
            )
            mom = torch.zeros_like(adv)
            trace = []
            torch.cuda.reset_peak_memory_stats()
            for step in range(config["steps"]):
                adv.requires_grad_(True)
                state = di.get_state()
                grads = []
                losses = []
                for m in models:
                    di.set_state(state)
                    f = m.features(differentiable_resize_pad(adv, generator=di))
                    losses_l = []
                    for l in m.layers:
                        s, t = centers[m.name][l]
                        z = f[l][0].float()
                        losses_l.append(0.5 * (1 - z @ t) + 0.5 * (1 + z @ s))
                    loss = torch.stack(losses_l).mean()
                    g = torch.autograd.grad(loss, adv)[0].detach().float()
                    if not torch.isfinite(g).all() or not g.abs().sum():
                        raise ValueError("Invalid gradient")
                    grads.append(g)
                    losses.append(float(loss.detach()))
                fused = normalize_pixel_gradients(torch.stack(grads)).mean(0)
                mom += fused
                adv = (
                    torch.maximum(
                        torch.minimum(adv - mom.sign() / 255, clean + 16 / 255), clean - 16 / 255
                    )
                    .clamp(0, 1)
                    .detach()
                )
                trace.append({"step": step + 1, "losses": losses})
            dest = w.oldworker.receipt_path(run, row)
            w.legacy.save_png(dest.parent / "adv.png", adv)
            linf = float((w.legacy.load_image(dest.parent / "adv.png") - clean).abs().max())
            if linf > 16 / 255 + 1e-7:
                raise ValueError("PNG bound failed")
            atomic(dest.parent / "optimization.json", trace)
            atomic(
                dest,
                {
                    "status": "ATTACK_FROZEN",
                    "signature": digest([lock, row]),
                    "image_id": row["image_id"],
                    "clean": row["clean"],
                    "adv": str(dest.parent / "adv.png"),
                    "png_hashes": {
                        row["clean"]: row["clean_sha256"],
                        str(dest.parent / "adv.png"): sha(dest.parent / "adv.png"),
                    },
                    "steps": config["steps"],
                    "seed": 42,
                    "target_access": False,
                    "linf_png": linf,
                    "seconds": time.monotonic() - start,
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "proxy_names": POOLS[config["variant"]],
                    "reference_banks_sha256": lock["reference_banks_sha256"],
                },
            )
            print(
                "FROZEN",
                config["variant"],
                row["image_id"],
                round(time.monotonic() - start, 2),
                flush=True,
            )
    finally:
        for m in models:
            m.close()
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["attack", "evaluate", "report"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--family", required=True, choices=FAMILIES)
    parser.add_argument("--N", required=True, type=int)
    parser.add_argument("--engineering", action="store_true")
    args = parser.parse_args()
    os.environ.update(HF_HOME=str(CACHE.parent), PRIMARY_ML_CKA_STRICT_BF16="1")
    sys.path.insert(0, str(V7 / "src"))
    w = expansion.api()
    run, config, lock, data = setup(args.dataset, args.family, args.N, args.engineering)
    w.oldworker.allowed_targets = lambda c: [FAMILIES[args.family]]
    if args.phase == "report":
        from metrics import summarize

        summarize(run, run / "analysis_input")
        return
    with w.legacy.gpu_owner(BASE):
        if args.phase == "attack":
            attack(run, config, lock, data)
        else:
            w.oldworker.evaluations(config, run, lock, data)


if __name__ == "__main__":
    main()
