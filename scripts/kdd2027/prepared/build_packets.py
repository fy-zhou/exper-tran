"""Source-only CPU CLIP retrieval from the existing disjoint class banks."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from reproduction import checksum

from primary_ml_cka.baselines.clip_features import ClipFeatures

OUT = Path("/home/ubuntu/exper-tran/outputs/kdd2027_baseline_reproduction")
DEST = OUT / "prepared_remaining"
DEST.mkdir(exist_ok=True)
torch.set_num_threads(4)
model = ClipFeatures("B32", "/ephemeral/kdd2027/hf-cache/hub", device="cpu")
rankings = {}
for split in ["smoke", "final150"]:
    data = json.loads((OUT / "manifests" / f"{split}.json").read_text())["images"]
    packets = {"anyattack": [], "m_attack_v2": []}
    for row in data:
        primary = Path(row["target_reference"])
        if str(primary) not in rankings:
            candidates = sorted(primary.parent.glob("*.png"))

            def tensor(p):
                return (
                    torch.from_numpy(np.asarray(Image.open(p).convert("RGB")).copy())
                    .permute(2, 0, 1)
                    .float()
                )

            with torch.inference_mode():
                feats = torch.cat(
                    [
                        model(torch.stack([tensor(p) for p in candidates[i : i + 8]]))
                        for i in range(0, len(candidates), 8)
                    ]
                )
            index = candidates.index(primary)
            score = feats @ feats[index]
            order = score.argsort(descending=True).tolist()
            rankings[str(primary)] = [str(primary)] + [
                str(candidates[i]) for i in order if i != index
            ][:2]
        refs = [{"path": p, "sha256": checksum(p)} for p in rankings[str(primary)]]
        entry = {
            "image_id": row["image_id"],
            "clean": row["clean"],
            "clean_sha256": row["clean_sha256"],
            "target_references": refs,
            "retrieval_provenance": {
                "method": "CLIP-B32_projected_cosine",
                "revision": model.spec["revision"],
                "pool": "existing_disjoint_target_class_48_bank",
                "task_adaptation": "class_bank_retrieval_instead_of_external_COCO",
                "no_target_feedback": True,
            },
        }
        packets["m_attack_v2"].append(entry)
        packets["anyattack"].append({**entry, "target_references": refs[:1]})
    for method, rows in packets.items():
        path = DEST / f"{method}_{split}.json"
        value = {"split": split, "images": rows, "retrieval_audit": model.audit}
        if path.exists() and json.loads(path.read_text()) != value:
            raise RuntimeError("Existing packet differs")
        path.write_text(json.dumps(value, indent=2) + "\n")
        print(path, flush=True)
