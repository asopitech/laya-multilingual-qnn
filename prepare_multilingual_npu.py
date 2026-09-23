from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import onnx
from huggingface_hub import hf_hub_download
from onnxruntime.tools.onnx_model_utils import fix_output_shapes, make_dim_param_fixed

from laya_snapdragon.build import npu_rewrite
from laya_snapdragon.runtime import npu_session


ROOT = Path(__file__).resolve().parent
REPO_ID = "mizchi/laya-multilingual-onnx"
REVISION = "d9d003d543e63d6d3375c21d44624136bd1e0bad"
SOURCE_DEFAULT = ROOT / "models" / "laya-multilingual-npu" / "hub"
MODELS_DEFAULT = ROOT / "models" / "laya-multilingual-npu" / "runtime"
FILES = (
    "model.onnx",
    "rl_agent_config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
)


def download_checkpoint(source: Path) -> None:
    source.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        path = Path(hf_hub_download(REPO_ID, filename, revision=REVISION, local_dir=source))
        print(f"downloaded {filename} ({path.stat().st_size / 1024**2:.1f} MiB)", flush=True)


def link_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def prepare(source: Path, models: Path, seqs: list[int], markers: int) -> None:
    checkpoint = models / "laya"
    onnx_dir = models / "onnx"
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "tokenizer").mkdir(parents=True, exist_ok=True)
    onnx_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(source / "rl_agent_config.json", checkpoint / "rl_agent_config.json")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy2(source / "tokenizer" / name, checkpoint / "tokenizer" / name)
    link_or_copy(source / "model.onnx", onnx_dir / "laya_fp32.onnx")

    for seq in seqs:
        fixed = onnx_dir / f"laya_s{seq}_m{markers}.onnx"
        context = onnx_dir / f"laya_s{seq}_m{markers}_qnn_ctx.onnx"
        if not fixed.exists():
            print(f"bucket multilingual {seq} tokens x {markers} options", flush=True)
            model = onnx.load(str(source / "model.onnx"))
            for dim, value in (("batch", 1), ("sequence", seq), ("markers", markers)):
                make_dim_param_fixed(model.graph, dim, value)
            fix_output_shapes(model)
            model = npu_rewrite(model)
            onnx.save(model, str(fixed), save_as_external_data=True, location=fixed.name + ".data")
        if not context.exists():
            print(f"compile multilingual {seq} x {markers} for QNN HTP", flush=True)
            npu_session(fixed, context=context)
    print("multilingual NPU models ready", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the multilingual Laya QNN model buckets")
    parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--models", type=Path, default=MODELS_DEFAULT)
    parser.add_argument("--seq", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--markers", type=int, default=8)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    if not args.skip_download:
        download_checkpoint(args.source)
    prepare(args.source, args.models, args.seq, args.markers)


if __name__ == "__main__":
    main()

