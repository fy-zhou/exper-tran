"""Official COCO captioning and LLaVA-Bench(COCO) source preparation, CPU-only.

Raw reproducible sources live on ephemeral; frozen manifests and exact benchmark
images/questions on root disk. No model queries or class-screening selection.
"""

import argparse
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

from PIL import Image

from primary_ml_cka.baselines.protocol import atomic, freeze, sha

LLAVA_REVISION = "c121f0432da27facab705978f83c4ada465e46fd"
BASE = f"https://raw.githubusercontent.com/haotian-liu/LLaVA/{LLAVA_REVISION}/"
FILES = {
    "questions.jsonl": "playground/data/coco2014_val_qa_eval/qa90_questions.jsonl",
    "reference_answers.jsonl": "playground/data/coco2014_val_qa_eval/qa90_gpt4_answer.jsonl",
    "source_qa.jsonl": "playground/data/coco2014_val_gpt4_qa_30x3.jsonl",
    "official_benchmark.md": "docs/LLaVA_Bench.md",
}


def fetch(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        partial = dest.with_suffix(dest.suffix + ".part")
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--retry",
                "3",
                "--retry-delay",
                "3",
                "--continue-at",
                "-",
                "--output",
                str(partial),
                url,
            ],
            check=True,
        )
        partial.rename(dest)
    return {"url": url, "path": str(dest), "bytes": dest.stat().st_size, "sha256": sha(dest)}


def extract(archive, root):
    root = root.resolve()
    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            path = (root / info.filename).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Unsafe archive member")
            if not info.is_dir() and (not path.is_file() or path.stat().st_size != info.file_size):
                source.extract(info, root)


def prepare(cache, output):
    sources, failures = {}, []
    for filename in ["val2017.zip", "annotations_trainval2017.zip"]:
        category = "zips" if filename == "val2017.zip" else "annotations"
        # This is the official URL, published by the COCO project. Its HTTPS host
        # currently has a certificate-name mismatch; don't disable TLS checks.
        url = f"http://images.cocodataset.org/{category}/{filename}"
        try:
            sources[filename] = fetch(url, cache / filename)
            extract(cache / filename, cache / "coco2017")
        except Exception as e:
            failures.append({"source": url, "error": repr(e)})
    for name, path in FILES.items():
        try:
            sources[name] = fetch(BASE + path, output / "llava_bench_coco/official" / name)
        except Exception as e:
            failures.append({"source": BASE + path, "error": repr(e)})
    questions_path = output / "llava_bench_coco/official/questions.jsonl"
    bench = []
    if questions_path.is_file():
        questions = [json.loads(s) for s in questions_path.read_text().splitlines() if s.strip()]
        image_names = sorted({r["image"] for r in questions})
        if len(questions) != 90 or len(image_names) != 30:
            raise RuntimeError("Official LLaVA-Bench(COCO) unexpectedly changed")
        for name in image_names:
            image_id = int(Path(name).stem)
            canonical_name = f"COCO_val2014_{image_id:012d}.jpg"
            dest = output / "llava_bench_coco/images" / name
            try:
                receipt = fetch(
                    f"http://images.cocodataset.org/val2014/{canonical_name}",
                    cache / "llava_images" / canonical_name,
                )
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    shutil.copyfile(receipt["path"], dest)
                with Image.open(dest) as image:
                    image.verify()
                bench.append(
                    {
                        "image_id": image_id,
                        "image": str(dest),
                        "sha256": sha(dest),
                        "source": receipt["url"],
                        "questions": [r for r in questions if r["image"] == name],
                    }
                )
            except Exception as e:
                failures.append({"image": name, "error": repr(e)})
        if len(bench) == 30:
            freeze(
                output / "llava_bench_coco/source_manifest.json",
                {
                    "dataset": "LLaVA-Bench(COCO)",
                    "revision": LLAVA_REVISION,
                    "task": "official_open_ended_VQA",
                    "N_images": 30,
                    "N_questions": 90,
                    "images": bench,
                    "not_150_repeated_images": True,
                    "official_judge": "not_called_no_commercial_route_authorized",
                    "status": "SOURCE_PREPARED_NOT_AN_ATTACK_RESULT",
                },
            )
    annotation = cache / "coco2017/annotations/captions_val2017.json"
    if annotation.exists() and (cache / "coco2017/val2017").is_dir():
        data = json.loads(annotation.read_text())
        missing = [
            r["file_name"]
            for r in data["images"]
            if not (cache / "coco2017/val2017" / r["file_name"]).is_file()
        ]
        if len(data["images"]) != 5000 or missing:
            raise RuntimeError(f"COCO val2017 incomplete: {len(missing)} missing")
        freeze(
            output / "coco_captioning/source_manifest.json",
            {
                "dataset": "MS COCO2017",
                "split": "val2017",
                "task": "image_captioning",
                "N_source_images": 5000,
                "annotation_sha256": sha(annotation),
                "image_root": str(cache / "coco2017/val2017"),
                "source_archives": {k: v for k, v in sources.items() if k.endswith(".zip")},
                "sampling": "not_selected_yet_no_target_feedback",
                "status": "SOURCE_PREPARED_NOT_AN_ATTACK_RESULT",
            },
        )
    result = {
        "sources": sources,
        "failures": failures,
        "llava_images_downloaded": len(bench),
        "status": "PASS" if not failures else "PARTIAL",
    }
    atomic(output / "dataset_preparation.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, default=Path("/ephemeral/kdd2027/datasets"))
    p.add_argument(
        "--output",
        type=Path,
        default=Path("/home/ubuntu/exper-tran/outputs/" "kdd2027_baseline_reproduction/datasets"),
    )
    args = p.parse_args()
    result = prepare(args.cache, args.output)
    print(result["status"], "LLaVA images", result["llava_images_downloaded"], flush=True)
    raise SystemExit(0 if result["status"] == "PASS" else 2)
