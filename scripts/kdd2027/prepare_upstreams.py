"""Fetch exact upstream revisions; never run upstream entrypoints or install their stacks."""

import argparse
import subprocess
from pathlib import Path

from primary_ml_cka.baselines.protocol import UPSTREAMS

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/home/ubuntu/baseline-upstreams"))
    args = p.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    for name, spec in UPSTREAMS.items():
        dest = args.root / name
        if not dest.exists():
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    f'https://github.com/{spec["repo"]}.git',
                    str(dest),
                ],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(dest), "checkout", "--detach", spec["revision"]], check=True
            )
        actual = subprocess.check_output(
            ["git", "-C", str(dest), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != spec["revision"]:
            raise RuntimeError(f"Existing upstream revision differs; preserved: {dest}")
        print(name, actual)
