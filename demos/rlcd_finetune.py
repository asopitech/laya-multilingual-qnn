from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from laya.common import QTYPES, build_model, build_sequence, proper_reward


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT.parent / "models" / "laya-multilingual-trainable"
DEFAULT_OUTPUT = ROOT.parent / "models" / "laya-multilingual-ja-rlcd"


@dataclass
class TrainItem:
    dataset: str
    uid: str
    ids: list[int]
    markers: list[int]
    qtype: int
    target: list[float]
    label: int


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def one_hot(index: int, size: int) -> list[float]:
    return [float(i == index) for i in range(size)]


def make_item(
    tok: Any,
    dataset: str,
    uid: str,
    state: str,
    instructions: str,
    criteria: dict[str, str],
    gold_label: str,
    max_len: int,
    head_max_len: int,
) -> TrainItem:
    labels = list(criteria)
    ids, markers = build_sequence(
        tok,
        state,
        {"t": "choice", "ins": instructions, "crit": criteria},
        max_len=max_len,
        head_max_len=head_max_len,
    )
    if len(markers) != len(labels):
        raise ValueError(f"{dataset}:{uid}: expected {len(labels)} markers, got {len(markers)}")
    label = labels.index(gold_label)
    return TrainItem(dataset, uid, ids, markers, QTYPES["choice"], one_hot(label, len(labels)), label)


def jnli_item(tok: Any, row: dict[str, Any], max_len: int, head_max_len: int) -> TrainItem:
    criteria = {
        "entailment": "前提が正しければ仮説も正しいと判断できる",
        "neutral": "前提だけでは仮説が正しいとも誤りとも判断できない",
        "contradiction": "前提と仮説が矛盾している",
    }
    return make_item(
        tok,
        "jglue-jnli",
        str(row["sentence_pair_id"]),
        f"前提: {row['sentence1']}\n仮説: {row['sentence2']}",
        "前提と仮説の意味関係を判定してください。",
        criteria,
        str(row["label"]),
        max_len,
        head_max_len,
    )


def jcqa_item(tok: Any, row: dict[str, Any], max_len: int, head_max_len: int) -> TrainItem:
    labels = "abcde"
    criteria = {labels[i]: str(row[f"choice{i}"]) for i in range(5)}
    return make_item(
        tok,
        "jglue-jcommonsenseqa",
        str(row["q_id"]),
        "一般的な日本語の常識に基づいて判断してください。",
        str(row["question"]),
        criteria,
        labels[int(row["label"])],
        max_len,
        head_max_len,
    )


def sample_rows(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count >= len(rows):
        return rows
    return random.Random(seed).sample(rows, count)


def load_japanese_items(
    tok: Any,
    data_dir: Path,
    split: str,
    per_dataset: int,
    max_len: int,
    head_max_len: int,
    seed: int,
) -> list[TrainItem]:
    jnli = sample_rows(read_json_lines(data_dir / f"jnli_{split}.jsonl"), per_dataset, seed)
    jcqa = sample_rows(read_json_lines(data_dir / f"jcqa_{split}.jsonl"), per_dataset, seed + 1)
    items = [jnli_item(tok, row, max_len, head_max_len) for row in jnli]
    items.extend(jcqa_item(tok, row, max_len, head_max_len) for row in jcqa)
    return items


def collate(items: list[TrainItem], pad_id: int) -> dict[str, torch.Tensor]:
    n = len(items)
    length = max(len(item.ids) for item in items)
    options = max(len(item.markers) for item in items)
    ids = torch.full((n, length), pad_id, dtype=torch.long)
    attention = torch.zeros((n, length), dtype=torch.long)
    marker_pos = torch.zeros((n, options), dtype=torch.long)
    marker_mask = torch.zeros((n, options), dtype=torch.bool)
    target = torch.zeros((n, options), dtype=torch.float32)
    for i, item in enumerate(items):
        ids[i, : len(item.ids)] = torch.tensor(item.ids)
        attention[i, : len(item.ids)] = 1
        marker_pos[i, : len(item.markers)] = torch.tensor(item.markers)
        marker_mask[i, : len(item.markers)] = True
        target[i, : len(item.target)] = torch.tensor(item.target)
    return {
        "input_ids": ids,
        "attention_mask": attention,
        "marker_pos": marker_pos,
        "marker_mask": marker_mask,
        "qtype": torch.tensor([item.qtype for item in items]),
        "target": target,
        "label": torch.tensor([item.label for item in items]),
    }


def batches(items: list[TrainItem], batch_size: int) -> Iterable[list[TrainItem]]:
    for offset in range(0, len(items), batch_size):
        yield items[offset : offset + batch_size]


def parameter_group(name: str) -> str:
    if name.startswith("encoder."):
        return "encoder"
    if name.startswith("act_head."):
        return "act_head"
    return "decision_head"


def parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    result = {"encoder": 0, "decision_head": 0, "act_head": 0}
    for name, parameter in model.named_parameters():
        result[parameter_group(name)] += parameter.numel()
    return result


def parameter_fingerprints(model: torch.nn.Module) -> dict[str, str]:
    hashes = {name: hashlib.sha256() for name in ("encoder", "decision_head", "act_head")}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            digest = hashes[parameter_group(name)]
            digest.update(name.encode("utf-8"))
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return {name: digest.hexdigest() for name, digest in hashes.items()}


def gradient_norms(model: torch.nn.Module) -> dict[str, float]:
    squared = {"encoder": 0.0, "decision_head": 0.0, "act_head": 0.0}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            squared[parameter_group(name)] += float(parameter.grad.detach().float().square().sum())
    return {name: math.sqrt(value) for name, value in squared.items()}


def rlcd_loss(
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    group_size: int,
    sigma: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = batch["marker_mask"]
    target = batch["target"]
    k = mask.sum(-1, keepdim=True).float()
    noise = torch.randn((group_size,) + logits.shape, device=logits.device) * sigma * mask
    noise = (noise - noise.sum(-1, keepdim=True) / k) * mask
    sampled_logits = logits.detach().unsqueeze(0) + noise
    distributions = torch.softmax(sampled_logits.masked_fill(~mask, -1e4), -1)
    with torch.no_grad():
        reward = proper_reward(
            distributions,
            target.unsqueeze(0),
            batch["qtype"],
            mask,
            w_sph=0.75,
            w_rps=1.0,
        )
        advantage = reward - reward.mean(0, keepdim=True)
        advantage = advantage / (advantage.std() + 1e-6)
    log_probability = -(
        ((sampled_logits - logits.unsqueeze(0)) ** 2) * mask
    ).sum(-1) / (2 * sigma**2)
    loss_rl = -(advantage * log_probability).mean()
    loss_ce = -(
        target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)
    ).sum(-1).mean()
    loss = loss_rl + loss_ce
    return loss, {
        "loss": float(loss.detach()),
        "loss_rl": float(loss_rl.detach()),
        "loss_ce": float(loss_ce.detach()),
        "reward": float(reward.mean()),
    }


@torch.no_grad()
def evaluate(model: torch.nn.Module, items: list[TrainItem], pad_id: int, batch_size: int) -> dict[str, Any]:
    model.eval()
    correct = 0
    nll = 0.0
    by_dataset: dict[str, list[int]] = {}
    for chunk in batches(items, batch_size):
        batch = collate(chunk, pad_id)
        logits, _ = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["marker_pos"],
            batch["marker_mask"],
            batch["qtype"],
        )
        probabilities = torch.softmax(logits.masked_fill(~batch["marker_mask"], -1e4), -1)
        predicted = probabilities.argmax(-1)
        for i, item in enumerate(chunk):
            hit = int(predicted[i] == batch["label"][i])
            correct += hit
            nll -= math.log(max(1e-12, float(probabilities[i, item.label])))
            totals = by_dataset.setdefault(item.dataset, [0, 0])
            totals[0] += hit
            totals[1] += 1
    count = len(items)
    return {
        "count": count,
        "accuracy": correct / count,
        "nll": nll / count,
        "accuracy_by_dataset": {
            name: values[0] / values[1] for name, values in sorted(by_dataset.items())
        },
    }


def save_checkpoint(
    model: torch.nn.Module,
    tok: Any,
    cfg: dict[str, Any],
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().half().contiguous().cpu() for name, value in model.state_dict().items()}
    save_file(state, output_dir / "model.safetensors")
    model.encoder.config.save_pretrained(output_dir / "encoder")
    tok.save_pretrained(output_dir / "tokenizer")
    cfg = dict(cfg)
    cfg["fine_tuned"] = True
    cfg["model_name"] = "laya-multilingual-ja-rlcd"
    cfg["temperature"] = [1.0, 1.0, 1.0]
    cfg["temperature_by_options"] = {}
    cfg["local_rlcd"] = metadata["training"]
    training = dict(cfg.get("training", {}))
    training["fine_tuned_from_checkpoint"] = True
    training["local_updates"] = metadata["training"]["steps"]
    cfg["training"] = training
    (output_dir / "rl_agent_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "training_report.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a local RLCD fine-tuning smoke run that updates both encoder and decision head."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-dir", type=Path, default=ROOT / ".demo-data")
    parser.add_argument("--train-per-dataset", type=int, default=8)
    parser.add_argument("--eval-per-dataset", type=int, default=20)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--head-max-len", type=int, default=96)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--encoder-lr", type=float, default=2.5e-5)
    parser.add_argument("--head-lr", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be at least 1")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))

    started = time.time()
    cfg = json.loads((args.model_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(args.model_dir / "tokenizer")
    train_items = load_japanese_items(
        tok, args.data_dir, "train", args.train_per_dataset, args.max_len, args.head_max_len, args.seed
    )
    eval_items = load_japanese_items(
        tok, args.data_dir, "valid", args.eval_per_dataset, args.max_len, args.head_max_len, args.seed + 100
    )

    print(f"loading trainable checkpoint: {args.model_dir}", flush=True)
    model = build_model(cfg, str(args.model_dir / "encoder"))
    model.load_state_dict(load_file(args.model_dir / "model.safetensors"), strict=True)
    model.encoder.config.reference_compile = False
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.train()

    counts = parameter_counts(model)
    before_fingerprints = parameter_fingerprints(model)
    before_eval = evaluate(model, eval_items, tok.pad_token_id, args.eval_batch_size)
    print(f"before: {json.dumps(before_eval, ensure_ascii=False)}", flush=True)

    encoder_parameters = [p for name, p in model.named_parameters() if name.startswith("encoder.")]
    head_parameters = [p for name, p in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": args.encoder_lr},
            {"params": head_parameters, "lr": args.head_lr},
        ],
        weight_decay=0.01,
        foreach=False,
    )

    order = list(train_items)
    random.Random(args.seed).shuffle(order)
    history: list[dict[str, Any]] = []
    cursor = 0
    for step in range(args.steps):
        if cursor + args.batch_size > len(order):
            random.Random(args.seed + step + 1).shuffle(order)
            cursor = 0
        chunk = order[cursor : cursor + args.batch_size]
        cursor += args.batch_size
        batch = collate(chunk, tok.pad_token_id)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, act_logits = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["marker_pos"],
            batch["marker_mask"],
            batch["qtype"],
        )
        loss, metrics = rlcd_loss(logits.float(), batch, args.group_size, args.sigma)
        loss = loss + 0.0 * act_logits.sum()
        loss.backward()
        norms = gradient_norms(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        row = {
            "step": step + 1,
            "datasets": [item.dataset for item in chunk],
            **metrics,
            "gradient_l2": norms,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    del optimizer
    after_fingerprints = parameter_fingerprints(model)
    after_eval = evaluate(model, eval_items, tok.pad_token_id, args.eval_batch_size)
    updated = {
        name: before_fingerprints[name] != after_fingerprints[name]
        for name in before_fingerprints
    }
    if not updated["encoder"] or not updated["decision_head"]:
        raise RuntimeError(f"required parameter groups were not both updated: {updated}")

    metadata = {
        "status": "completed",
        "base_model": str(args.model_dir.resolve()),
        "output_model": str(args.output_dir.resolve()),
        "device": "cpu",
        "deployment_target": "Qualcomm QNN HTP/NPU",
        "parameter_counts": counts,
        "fingerprints_before": before_fingerprints,
        "fingerprints_after": after_fingerprints,
        "updated": updated,
        "evaluation_before": before_eval,
        "evaluation_after": after_eval,
        "history": history,
        "training": {
            "algorithm": "RLCD with proper-scoring reward and cross-entropy guidance",
            "datasets": ["JGLUE/JNLI", "JGLUE/JCommonsenseQA"],
            "train_items": len(train_items),
            "eval_items": len(eval_items),
            "steps": args.steps,
            "batch_size": args.batch_size,
            "group_size": args.group_size,
            "sigma": args.sigma,
            "encoder_lr": args.encoder_lr,
            "head_lr": args.head_lr,
            "max_len": args.max_len,
            "head_max_len": args.head_max_len,
            "seed": args.seed,
            "elapsed_seconds": time.time() - started,
            "act_head_note": "No action labels were supplied; its zero gradient is connected only for graph completeness.",
        },
    }
    save_checkpoint(model, tok, cfg, args.output_dir, metadata)
    print(json.dumps({"updated": updated, "output": str(args.output_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
