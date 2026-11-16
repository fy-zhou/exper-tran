"""Immutable, image-level baseline provenance; no model imports or inference."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

PROXIES = {
    "B16": {
        "repo_id": "openai/clip-vit-base-patch16",
        "revision": "57c216476eefef5ab752ec549e440a49ae4ae5f3",
    },
    "B32": {
        "repo_id": "openai/clip-vit-base-patch32",
        "revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
    },
    "Laion": {
        "repo_id": "laion/CLIP-ViT-G-14-laion2B-s12B-b42K",
        "revision": "4b0305adc6802b2632e11cbe6606a9bdd43d35c9",
    },
}
UPSTREAMS = {
    "m_attack": {
        "repo": "VILA-Lab/M-Attack",
        "revision": "9e0cc64a22dff799fa84c12a4e51934d1e849e29",
    },
    "foa_attack": {
        "repo": "jiaxiaojunQAQ/FOA-Attack",
        "revision": "6aa9ad463a1e812b7f77ff2e5c4b1712e7c78c4b",
    },
    "co_attack": {
        "repo": "adversarial-for-goodness/Co-Attack",
        "revision": "96f2c0ebf743f600860ed7aa78c14b38c8883521",
    },
    "sga": {"repo": "Zoky-2020/SGA", "revision": "f49e6926cd13148e49bdbc2c3e537660e096230c"},
    "iau_foa": {
        "repo": "jiaxiaojunQAQ/IAU-FOA",
        "revision": "749f349da249bd933ace4d33776ee59249b8d2a3",
    },
}
IMPLEMENTED = {"pgd", "mi_fgsm", "m_attack", "foa_attack"}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def freeze(path, value):
    if Path(path).exists():
        if json.loads(Path(path).read_text()) != value:
            raise RuntimeError(f"Frozen provenance differs: {path}")
    else:
        atomic(path, value)


def validate_method(method):
    if method not in IMPLEMENTED:
        raise ValueError(f"{method}: not implemented/compatible; never substitute another method")


def make_manifest(ours_root, data_root, split):
    """Freeze exactly the old cohort; choose one reference by ID, never target feedback."""
    ours_root, data_root = Path(ours_root), Path(data_root)
    source = ours_root / "manifests" / f"{split}.json"
    rows = json.loads(source.read_text())["images"]
    refs = []
    for label in range(1, 11):
        p = data_root / "manifests/class_references" / f"class_references_{label:02d}.jsonl"
        bank = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        if len(bank) != 48 or any(r["human_label"] != label for r in bank):
            raise ValueError(f"Invalid 48-reference bank: {p}")
        refs.extend(bank)
    ids = [r["image_id"] for r in rows]
    ref_ids = [r["image_id"] for r in refs]
    if len(set(ids)) != len(ids) or len(set(ref_ids)) != 480 or set(ids) & set(ref_ids):
        raise ValueError("Duplicate images or reference/attack leakage")
    targets = {
        label: min((r for r in refs if r["human_label"] == label), key=lambda r: r["image_id"])
        for label in range(1, 11)
    }
    frozen = []
    for r in rows:
        clean = Path(r["path"])
        target = data_root / "canonical_images" / targets[r["target"]]["relative_path"]
        frozen.append(
            {
                **r,
                "clean": str(clean),
                "clean_sha256": sha(clean),
                "target_reference": str(target),
                "target_reference_sha256": sha(target),
                "target_reference_id": targets[r["target"]]["image_id"],
            }
        )
    # Verify pixel-level duplicates, not just filenames.
    from PIL import Image

    def pixels(p):
        with Image.open(p) as image:
            return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()

    reference_pixels = {pixels(data_root / "canonical_images" / r["relative_path"]) for r in refs}
    if any(pixels(r["clean"]) in reference_pixels for r in frozen):
        raise ValueError("Decoded reference/attack pixels overlap")
    return {
        "version": 1,
        "split": split,
        "source_sha256": sha(source),
        "images": frozen,
        "reference_count": 480,
        "reference_selection": "minimum_image_id_per_class",
        "eligible": len(rows),
        "excluded": 0,
        "exclusion_reasons": [],
        "eligibility": "ImageNet_ground_truth_only_no_prediction_gate",
    }


def valid_receipt(path, signature):
    path = Path(path)
    if not path.exists():
        return False
    try:
        r = json.loads(path.read_text())
        return (
            r.get("signature") == signature
            and r.get("status") == "ATTACK_FROZEN"
            and not r.get("target_access", True)
            and r.get("linf_png", 1) <= 16 / 255 + 1e-7
            and all(Path(p).is_file() and sha(p) == h for p, h in r["png_hashes"].items())
        )
    except (KeyError, ValueError, OSError):
        return False
