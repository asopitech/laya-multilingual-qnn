from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from laya_snapdragon.build import build


ROOT = Path(__file__).resolve().parent


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def stage_checkpoint(checkpoint: Path, models: Path) -> None:
    target = models / "laya"
    target.mkdir(parents=True, exist_ok=True)
    for filename in ("model.safetensors", "rl_agent_config.json", "training_report.json"):
        source = checkpoint / filename
        if source.exists():
            link_or_copy(source, target / filename)
    for directory in ("encoder", "tokenizer"):
        shutil.copytree(checkpoint / directory, target / directory, dirs_exist_ok=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Export a fine-tuned Laya checkpoint and compile static Qualcomm QNN HTP buckets."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT.parent / "models" / "laya-multilingual-ja-rlcd",
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=ROOT.parent / "models" / "laya-multilingual-ja-rlcd-npu",
    )
    parser.add_argument("--seq", type=int, nargs="+", default=[128])
    parser.add_argument("--markers", type=int, default=8)
    args = parser.parse_args()

    stage_checkpoint(args.checkpoint, args.models)
    build(args.models, seqs=args.seq, markers=args.markers, npu=True, from_torch=True)


if __name__ == "__main__":
    main()
