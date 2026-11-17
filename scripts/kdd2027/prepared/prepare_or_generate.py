"""Explicitly opt-in generation; default is CPU/static preparation only."""

import argparse
import importlib.util
import json
import os
import random
import time
from pathlib import Path

from reproduction import (
    BACKBONES,
    PINS,
    V2Runner,
    anyattack_generate,
    checksum,
    load_anyattack,
    load_v2_extractors,
    verify_source,
)


def validate_packet(packet, method):
    rows = packet["images"]
    if not rows or len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Empty or duplicate image IDs")
    clean_hashes = {checksum(r["clean"]) for r in rows}
    expected = 3 if method == "m_attack_v2" else 1
    for r in rows:
        if checksum(r["clean"]) != r["clean_sha256"]:
            raise ValueError("Clean checksum mismatch")
        refs = r["target_references"]
        if len(refs) != expected:
            raise ValueError(f"{method} requires exactly {expected} target references")
        hashes = [checksum(t["path"]) for t in refs]
        if len(set(hashes)) != len(hashes) or clean_hashes.intersection(hashes):
            raise ValueError("References duplicated or overlap attack cohort")
        if hashes != [t["sha256"] for t in refs]:
            raise ValueError("Reference checksum mismatch")
        if method == "m_attack_v2" and not r.get("retrieval_provenance"):
            raise ValueError("Missing source-only retrieval provenance")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=list(PINS), required=True)
    p.add_argument("--upstream-root", type=Path, default=Path("/home/ubuntu/baseline-upstreams"))
    p.add_argument("--packet", type=Path)
    p.add_argument("--cache", type=Path, default=Path("/ephemeral/kdd2027/hf-cache/hub"))
    p.add_argument("--output", type=Path)
    p.add_argument(
        "--generate",
        action="store_true",
        help="Explicit future opt-in; not used during preparation",
    )
    p.add_argument("--steps", type=int, choices=[2, 50, 300], default=300)
    p.add_argument("--clip-weights", type=Path)
    p.add_argument("--decoder-weights", type=Path)
    p.add_argument("--decoder-sha256")
    args = p.parse_args()
    root = verify_source(args.upstream_root / args.method, args.method)
    packet = json.loads(args.packet.read_text()) if args.packet else None
    rows = validate_packet(packet, args.method) if packet else []
    info = {
        "mode": "PREPARED_NOT_GPU_VALIDATED",
        "method": args.method,
        "upstream_revision": PINS[args.method],
        "rows": len(rows),
        "backbones": BACKBONES if args.method == "m_attack_v2" else "official CLIP-B32+decoder",
        "weights_downloaded": False,
        "inference_executed": False,
        "dependencies": {
            x: importlib.util.find_spec(x) is not None
            for x in ["torch", "torchvision", "omegaconf", "einops", "transformers"]
        },
    }
    if not args.generate:
        print(json.dumps(info, indent=2))
        return
    if not packet or not args.output:
        p.error("--generate requires --packet and --output")
    if args.method == "anyattack" and not all(
        [args.clip_weights, args.decoder_weights, args.decoder_sha256]
    ):
        p.error("AnyAttack requires explicit local weight paths and decoder SHA256")
    # No inference, model initialization or CUDA query occurs before explicit opt-in.
    import numpy as np
    import torch
    from PIL import Image

    from primary_ml_cka.baselines import campaign as legacy
    from primary_ml_cka.baselines.protocol import atomic, digest, freeze, valid_receipt

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no automatic CPU fallback")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "method": args.method,
        "upstream": PINS[args.method],
        "steps": args.steps if args.method == "m_attack_v2" else 0,
        "seed": 42,
        "epsilon": 16 / 255,
        "packet_sha256": checksum(args.packet),
        "precision": "bf16_unquantized",
        "decoder_sha256": args.decoder_sha256,
        "clip_weights_sha256": checksum(args.clip_weights) if args.clip_weights else None,
        "adapter_sources": {
            f.name: checksum(f) for f in [Path(__file__), Path(__file__).parent / "reproduction.py"]
        },
        "status": "UNVALIDATED_IMPLEMENTATION_GENERATION_ONLY",
    }
    freeze(args.output / "method_lock.json", config)
    owner = Path("/home/ubuntu/exper-tran/outputs/kdd2027_baseline_reproduction")
    import fcntl

    with (owner / "reproduction_owner.lock").open("a") as admission:
        fcntl.flock(admission, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with legacy.gpu_owner(owner):
            if args.method == "m_attack_v2":
                models = load_v2_extractors(args.cache)
                runner = V2Runner(root, models, steps=args.steps)
                audits = [m.audit for m in models]
            else:
                encoder, decoder, audits = load_anyattack(
                    root, args.clip_weights, args.decoder_weights, args.decoder_sha256
                )
            atomic(args.output / "bf16_audit.json", audits)

            def image(path):
                a = np.asarray(Image.open(path).convert("RGB"))
                if a.shape != (224, 224, 3):
                    raise ValueError("Require canonical224 input; no silent resizing")
                return torch.from_numpy(a.copy()).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255

            for row in rows:
                seed = int(digest([42, row["image_id"], args.method])[:8], 16)
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                dest = args.output / "attacks" / digest(row["image_id"])[:18]
                signature = digest([config, row])
                if valid_receipt(dest / "receipt.json", signature):
                    continue
                began = time.monotonic()
                torch.cuda.reset_peak_memory_stats()
                clean = image(row["clean"])
                targets = torch.cat([image(t["path"]) for t in row["target_references"]])
                if args.method == "m_attack_v2":
                    adv = runner(clean, targets)
                else:
                    adv = anyattack_generate(encoder, decoder, clean, targets)
                a = (adv[0].detach().cpu().permute(1, 2, 0).numpy() * 255).round().astype("uint8")
                b = np.asarray(Image.open(row["clean"]).convert("RGB"), dtype=np.int16)
                linf = int(np.abs(a.astype(np.int16) - b).max())
                if linf > 16:
                    raise RuntimeError("Saved PNG violates perturbation budget")
                dest.mkdir(parents=True, exist_ok=True)
                tmp = dest / "adv.tmp.png"
                Image.fromarray(a).save(tmp)
                os.replace(tmp, dest / "adv.png")
                atomic(
                    dest / "receipt.json",
                    {
                        "status": "ATTACK_FROZEN",
                        "signature": signature,
                        "image_id": row["image_id"],
                        "clean": row["clean"],
                        "adv": str(dest / "adv.png"),
                        "png_hashes": {
                            row["clean"]: checksum(row["clean"]),
                            str(dest / "adv.png"): checksum(dest / "adv.png"),
                        },
                        "linf_png": linf / 255,
                        "target_access": False,
                        "seed": seed,
                        "steps": config["steps"],
                        "seconds": time.monotonic() - began,
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        "reference_hashes": [t["sha256"] for t in row["target_references"]],
                        "visual_forward_calls": sum(m.forwards for m in models)
                        if args.method == "m_attack_v2"
                        else None,
                        "forward_counter_scope": "cumulative model forwards within process",
                    },
                )
                print("FROZEN", row["image_id"], flush=True)


if __name__ == "__main__":
    main()
