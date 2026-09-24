from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import re
import statistics
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_URLS = {
    "lawqa": "https://raw.githubusercontent.com/digital-go-jp/lawqa_jp/main/data/selection.json",
    "jnli_train": "https://raw.githubusercontent.com/yahoojapan/JGLUE/main/datasets/jnli-v1.3/train-v1.3.json",
    "jnli_valid": "https://raw.githubusercontent.com/yahoojapan/JGLUE/main/datasets/jnli-v1.3/valid-v1.3.json",
    "jcqa_train": "https://raw.githubusercontent.com/yahoojapan/JGLUE/main/datasets/jcommonsenseqa-v1.3/train-v1.3.json",
    "jcqa_valid": "https://raw.githubusercontent.com/yahoojapan/JGLUE/main/datasets/jcommonsenseqa-v1.3/valid-v1.3.json",
}
TYPED_WORKFLOWS = (
    "agent_trace_observability",
    "customer_service",
    "invoice_processing",
    "security_incidents",
)


@dataclass
class DecisionCase:
    dataset: str
    uid: str
    state: Any
    question: str
    kind: str
    criteria: Any
    gold_label: str
    gold_probs: dict[str, float] | None = None
    context: str = ""
    can_strip_criteria: bool = False

    @property
    def labels(self) -> list[str]:
        if self.kind == "choice":
            return list(self.criteria)
        if self.kind == "score":
            return [str(i) for i in range(len(self.criteria))]
        return ["false", "true"]


@dataclass
class Prediction:
    probabilities: dict[str, float]
    elapsed_ms: float
    npu_calls: int
    buckets: list[str]


def fetch(url: str, target: Path, offline: bool = False) -> Path:
    if target.exists():
        return target
    if offline:
        raise FileNotFoundError(f"offline cache miss: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "laya-multilingual-qnn-demo/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    target.write_bytes(payload)
    return target


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def sample_cases(cases: list[DecisionCase], limit: int, seed: int) -> list[DecisionCase]:
    if len(cases) <= limit:
        return cases
    return random.Random(seed).sample(cases, limit)


def parse_law_options(value: str) -> dict[str, str]:
    options: dict[str, str] = {}
    current: str | None = None
    for raw in value.splitlines():
        line = raw.strip()
        match = re.match(r"^([a-h])(?:[.)、:]|\s)+(.+)$", line, flags=re.IGNORECASE)
        if match:
            current = match.group(1).lower()
            options[current] = match.group(2).strip()
        elif current and line:
            options[current] += " " + line
    if len(options) < 2:
        raise ValueError(f"could not parse options: {value[:120]!r}")
    return options


def load_lawqa(cache: Path, offline: bool) -> list[DecisionCase]:
    path = fetch(DATA_URLS["lawqa"], cache / "lawqa-selection.json", offline)
    rows = json.loads(path.read_text(encoding="utf-8-sig"))["samples"]
    return [
        DecisionCase(
            dataset="lawqa",
            uid=str(row.get("回答オーダーマップ番号", row["ファイル名"])),
            state="追加の法令根拠は与えられていません。",
            question=row["問題文"],
            kind="choice",
            criteria=parse_law_options(row["選択肢"]),
            gold_label=str(row["output"]).strip().lower(),
            context=row["コンテキスト"],
        )
        for row in rows
    ]


def load_jnli(cache: Path, split: str, offline: bool) -> list[DecisionCase]:
    key = f"jnli_{split}"
    path = fetch(DATA_URLS[key], cache / f"{key}.jsonl", offline)
    criteria = {
        "entailment": "前提が正しければ仮説も正しいと判断できる",
        "neutral": "前提だけでは仮説が正しいとも誤りとも判断できない",
        "contradiction": "前提と仮説が矛盾している",
    }
    return [
        DecisionCase(
            dataset="jglue-jnli",
            uid=str(row["sentence_pair_id"]),
            state=f"前提: {row['sentence1']}\n仮説: {row['sentence2']}",
            question="前提と仮説の意味関係を、含意・中立・矛盾から判定してください。",
            kind="choice",
            criteria=criteria,
            gold_label=row["label"],
            can_strip_criteria=True,
        )
        for row in read_json_lines(path)
    ]


def load_jcommonsenseqa(cache: Path, split: str, offline: bool) -> list[DecisionCase]:
    key = f"jcqa_{split}"
    path = fetch(DATA_URLS[key], cache / f"{key}.jsonl", offline)
    labels = "abcde"
    cases = []
    for row in read_json_lines(path):
        criteria = {labels[i]: row[f"choice{i}"] for i in range(5)}
        cases.append(
            DecisionCase(
                dataset="jglue-jcommonsenseqa",
                uid=str(row["q_id"]),
                state="一般的な日本語の常識に基づいて判断してください。",
                question=row["question"],
                kind="choice",
                criteria=criteria,
                gold_label=labels[int(row["label"])],
            )
        )
    return cases


def typed_rows(cache: Path, workflow: str, split: str, length: int, offline: bool) -> list[dict[str, Any]]:
    target = cache / f"typed-{workflow}-{split}-{length}.json"
    query = urllib.parse.urlencode(
        {
            "dataset": "LocalLLaMA/typed-decisions",
            "config": workflow,
            "split": split,
            "offset": 0,
            "length": length,
        }
    )
    fetch(f"https://datasets-server.huggingface.co/rows?{query}", target, offline)
    payload = json.loads(target.read_text(encoding="utf-8"))
    return [item["row"] for item in payload["rows"]]


def load_typed(cache: Path, split: str, decision_limit: int, offline: bool) -> list[DecisionCase]:
    rows_per_workflow = max(1, math.ceil(decision_limit / (5 * len(TYPED_WORKFLOWS))))
    cases: list[DecisionCase] = []
    for workflow in TYPED_WORKFLOWS:
        for row in typed_rows(cache, workflow, split, rows_per_workflow, offline):
            state = json.loads(row["state"])
            questions = json.loads(row["questions"])
            gold = json.loads(row["gold"])
            for qid, qdef in questions.items():
                answer = gold[qid]
                cases.append(
                    DecisionCase(
                        dataset=f"typed-{workflow}",
                        uid=f"{row['id']}:{qid}",
                        state=state,
                        question=qdef["instructions"],
                        kind=qdef["type"],
                        criteria=qdef.get("criteria"),
                        gold_label=str(answer["label"]),
                        gold_probs={str(k): float(v) for k, v in answer["probabilities"].items()},
                        can_strip_criteria=True,
                    )
                )
    return cases[:decision_limit]


def character_ngrams(text: str, n: int = 2) -> set[str]:
    normalized = re.sub(r"[\W_]+", "", text.lower(), flags=re.UNICODE)
    if len(normalized) <= n:
        return {normalized} if normalized else set()
    return {normalized[i : i + n] for i in range(len(normalized) - n + 1)}


def context_chunks(context: str, max_chars: int = 120) -> list[str]:
    units = [
        re.sub(r"^#+\s*", "", part.strip())
        for part in re.split(r"(?<=[。！？])|\n(?=#+\s*)|\n(?=[-・])", context)
        if part.strip()
    ]
    chunks: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + len(unit) + 1 > max_chars:
            chunks.append(current)
            current = ""
        if len(unit) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(unit[i : i + max_chars] for i in range(0, len(unit), max_chars))
        else:
            current = f"{current}\n{unit}".strip()
    if current:
        chunks.append(current)
    return chunks or [context[:max_chars]]


def retrieve_context(case: DecisionCase, top_k: int = 3) -> list[str]:
    query = case.question + " " + " ".join(str(value) for value in case.criteria.values())
    query_terms = character_ngrams(query)
    ranked = []
    for index, chunk in enumerate(context_chunks(case.context)):
        terms = character_ngrams(chunk)
        overlap = len(query_terms & terms)
        score = overlap / math.sqrt(max(1, len(terms)))
        ranked.append((score, -index, chunk))
    ranked.sort(reverse=True)
    return [chunk for _, _, chunk in ranked[:top_k]]


def question_definition(case: DecisionCase, rich_criteria: bool, order: list[str] | None) -> dict[str, Any]:
    if case.kind == "choice":
        labels = order or case.labels
        if rich_criteria or not case.can_strip_criteria:
            criteria = {label: case.criteria[label] for label in labels}
        else:
            criteria = {label: None for label in labels}
    elif case.kind == "score":
        criteria = case.criteria if rich_criteria else [f"level {i}" for i in range(len(case.criteria))]
    else:
        criteria = case.criteria if rich_criteria else None
    return {"type": case.kind, "instructions": case.question, "criteria": criteria}


def probabilities_from_answer(case: DecisionCase, answer: dict[str, Any]) -> dict[str, float]:
    if case.kind in ("choice", "score"):
        return {str(k): float(v) for k, v in answer["probabilities"].items()}
    true_p = float(answer["noul"])
    return {"false": 1.0 - true_p, "true": true_p}


def predict_once(
    agent: Any,
    case: DecisionCase,
    state: Any,
    rich_criteria: bool = True,
    order: list[str] | None = None,
) -> Prediction:
    questions = {"decision": question_definition(case, rich_criteria, order)}
    prepared, _ = agent.prepare(state, questions)
    route = agent.route(prepared[0])
    if route is None:
        raise RuntimeError(f"{case.uid} would leave the NPU path")
    started = time.perf_counter()
    result = agent.predict(state, questions)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return Prediction(
        probabilities=probabilities_from_answer(case, result["answers"]["decision"]),
        elapsed_ms=elapsed_ms,
        npu_calls=1,
        buckets=[f"{route[0]}x{route[1]}"],
    )


def permutations(labels: list[str], count: int) -> list[list[str]]:
    count = max(1, min(count, len(labels)))
    return [labels[offset:] + labels[:offset] for offset in range(count)]


def fuse_predictions(predictions: Iterable[Prediction], labels: list[str]) -> Prediction:
    items = list(predictions)
    probabilities = {
        label: float(np.mean([item.probabilities[label] for item in items])) for label in labels
    }
    total = sum(probabilities.values())
    probabilities = {label: value / total for label, value in probabilities.items()}
    return Prediction(
        probabilities=probabilities,
        elapsed_ms=sum(item.elapsed_ms for item in items),
        npu_calls=sum(item.npu_calls for item in items),
        buckets=[bucket for item in items for bucket in item.buckets],
    )


def predict_ensemble(
    agent: Any,
    case: DecisionCase,
    states: list[Any],
    rich_criteria: bool = True,
    order_runs: int = 1,
) -> Prediction:
    orders: list[list[str] | None] = [None]
    if case.kind == "choice" and order_runs > 1:
        orders = permutations(case.labels, order_runs)
    return fuse_predictions(
        (
            predict_once(agent, case, state, rich_criteria=rich_criteria, order=order)
            for state in states
            for order in orders
        ),
        case.labels,
    )


def predict_option_verification(agent: Any, case: DecisionCase, states: list[Any]) -> Prediction:
    """Turn one N-way choice into N independent evidence checks on QNN HTP."""
    predictions: list[Prediction] = []
    scores: dict[str, list[float]] = {label: [] for label in case.labels}
    for state in states:
        for label, option in case.criteria.items():
            verification = DecisionCase(
                dataset=case.dataset,
                uid=f"{case.uid}:{label}",
                state=state,
                question=f"問題: {case.question}\n候補: {option}\n法令根拠から、この候補が正しいか判定してください。",
                kind="noul",
                criteria={
                    "false": "根拠と矛盾する、または根拠から正しいと確認できない",
                    "true": "根拠からこの候補が正しいと確認できる",
                },
                gold_label="true",
            )
            prediction = predict_once(agent, verification, state, rich_criteria=True)
            predictions.append(prediction)
            scores[label].append(prediction.probabilities["true"])
    probabilities = {label: max(1e-9, float(np.mean(values))) for label, values in scores.items()}
    total = sum(probabilities.values())
    probabilities = {label: value / total for label, value in probabilities.items()}
    return Prediction(
        probabilities=probabilities,
        elapsed_ms=sum(item.elapsed_ms for item in predictions),
        npu_calls=sum(item.npu_calls for item in predictions),
        buckets=[bucket for item in predictions for bucket in item.buckets],
    )


def scale_probabilities(probabilities: dict[str, float], temperature: float) -> dict[str, float]:
    labels = list(probabilities)
    values = np.array([max(1e-12, probabilities[label]) for label in labels], dtype=np.float64)
    logits = np.log(values) / max(1e-4, temperature)
    scaled = np.exp(logits - logits.max())
    scaled /= scaled.sum()
    return {label: float(value) for label, value in zip(labels, scaled)}


def target_distribution(case: DecisionCase) -> dict[str, float]:
    if case.gold_probs:
        values = {label: float(case.gold_probs.get(label, 0.0)) for label in case.labels}
        total = sum(values.values())
        if total > 0:
            return {label: value / total for label, value in values.items()}
    return {label: float(label == case.gold_label) for label in case.labels}


def fit_temperature(cases: list[DecisionCase], predictions: list[Prediction]) -> float:
    best_temperature, best_loss = 1.0, float("inf")
    for temperature in np.geomspace(0.25, 4.0, 81):
        loss = 0.0
        for case, prediction in zip(cases, predictions):
            scaled = scale_probabilities(prediction.probabilities, float(temperature))
            target = target_distribution(case)
            loss -= sum(target[label] * math.log(max(1e-12, scaled[label])) for label in case.labels)
        if loss < best_loss:
            best_temperature, best_loss = float(temperature), loss
    return best_temperature


def metrics(cases: list[DecisionCase], predictions: list[Prediction], temperature: float = 1.0) -> dict[str, Any]:
    correct = nll = brier = kl = 0.0
    confidence_bins: list[list[tuple[float, float]]] = [[] for _ in range(10)]
    elapsed = [prediction.elapsed_ms for prediction in predictions]
    calls = sum(prediction.npu_calls for prediction in predictions)
    bucket_counts: dict[str, int] = {}
    for case, prediction in zip(cases, predictions):
        probs = scale_probabilities(prediction.probabilities, temperature)
        predicted = max(probs, key=probs.get)
        is_correct = float(predicted == case.gold_label)
        correct += is_correct
        nll -= math.log(max(1e-12, probs[case.gold_label]))
        target = target_distribution(case)
        brier += sum((probs[label] - target[label]) ** 2 for label in case.labels)
        kl += sum(
            target[label] * math.log(max(1e-12, target[label]) / max(1e-12, probs[label]))
            for label in case.labels
            if target[label] > 0
        )
        confidence = probs[predicted]
        confidence_bins[min(9, int(confidence * 10))].append((confidence, is_correct))
        for bucket in prediction.buckets:
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    count = max(1, len(cases))
    ece = sum(
        len(bucket) / count
        * abs(statistics.mean(value[0] for value in bucket) - statistics.mean(value[1] for value in bucket))
        for bucket in confidence_bins
        if bucket
    )
    total_ms = sum(elapsed)
    return {
        "cases": len(cases),
        "accuracy": correct / count,
        "nll": nll / count,
        "brier": brier / count,
        "kl_to_gold": kl / count,
        "ece": ece,
        "p50_case_ms": statistics.median(elapsed) if elapsed else 0.0,
        "p95_case_ms": float(np.percentile(elapsed, 95)) if elapsed else 0.0,
        "npu_calls": calls,
        "npu_calls_per_second": calls / (total_ms / 1000.0) if total_ms else 0.0,
        "buckets": bucket_counts,
        "temperature": temperature,
    }


def run_predictions(
    agent: Any,
    cases: list[DecisionCase],
    state_fn: Callable[[DecisionCase], list[Any]],
    rich_criteria: bool,
    order_runs: int,
) -> list[Prediction]:
    return [
        predict_ensemble(
            agent,
            case,
            state_fn(case),
            rich_criteria=rich_criteria,
            order_runs=order_runs,
        )
        for case in cases
    ]


def add_method(
    suite: dict[str, Any],
    name: str,
    cases: list[DecisionCase],
    predictions: list[Prediction],
    temperature: float = 1.0,
) -> None:
    suite["methods"][name] = metrics(cases, predictions, temperature)


def run_typed_suite(agent: Any, cache: Path, limit: int, calibration: int, offline: bool, order_runs: int) -> dict[str, Any]:
    evaluation = load_typed(cache, "test", limit, offline)
    calibration_cases = load_typed(cache, "train", calibration, offline)
    suite: dict[str, Any] = {"description": "Typed Decisions on QNN HTP", "methods": {}}

    baseline = run_predictions(agent, evaluation, lambda case: [case.state], False, 1)
    rich = run_predictions(agent, evaluation, lambda case: [case.state], True, 1)
    ensemble = run_predictions(agent, evaluation, lambda case: [case.state], True, order_runs)
    calibration_predictions = run_predictions(agent, calibration_cases, lambda case: [case.state], True, order_runs)
    temperature = fit_temperature(calibration_cases, calibration_predictions)

    add_method(suite, "labels_only", evaluation, baseline)
    add_method(suite, "described_criteria", evaluation, rich)
    add_method(suite, "criteria_order_ensemble", evaluation, ensemble)
    add_method(suite, "ensemble_temperature_calibrated", evaluation, ensemble, temperature)
    return suite


def run_jglue_suite(agent: Any, cache: Path, limit: int, calibration: int, offline: bool, order_runs: int, seed: int) -> dict[str, Any]:
    evaluation = sample_cases(load_jnli(cache, "valid", offline), limit // 2, seed)
    evaluation += sample_cases(load_jcommonsenseqa(cache, "valid", offline), limit - len(evaluation), seed + 1)
    calibration_cases = sample_cases(load_jnli(cache, "train", offline), calibration // 2, seed + 2)
    calibration_cases += sample_cases(
        load_jcommonsenseqa(cache, "train", offline), calibration - len(calibration_cases), seed + 3
    )
    suite: dict[str, Any] = {"description": "JGLUE Japanese decisions on QNN HTP", "methods": {}}

    direct = run_predictions(agent, evaluation, lambda case: [case.state], True, 1)
    ensemble = run_predictions(agent, evaluation, lambda case: [case.state], True, order_runs)
    calibration_predictions = run_predictions(agent, calibration_cases, lambda case: [case.state], True, order_runs)
    temperature = fit_temperature(calibration_cases, calibration_predictions)

    add_method(suite, "direct", evaluation, direct)
    add_method(suite, "option_order_ensemble", evaluation, ensemble)
    add_method(suite, "ensemble_temperature_calibrated", evaluation, ensemble, temperature)
    return suite


def run_lawqa_suite(agent: Any, cache: Path, limit: int, calibration: int, offline: bool, order_runs: int, seed: int) -> dict[str, Any]:
    all_cases = load_lawqa(cache, offline)
    shuffled = all_cases[:]
    random.Random(seed).shuffle(shuffled)
    calibration_cases = shuffled[:calibration]
    evaluation = shuffled[calibration : calibration + limit]
    suite: dict[str, Any] = {"description": "Japanese law QA context methods on QNN HTP", "methods": {}}

    closed = run_predictions(agent, evaluation, lambda case: [case.state], True, 1)
    head = run_predictions(agent, evaluation, lambda case: [case.context], True, 1)
    top1 = run_predictions(agent, evaluation, lambda case: retrieve_context(case, 1), True, 1)
    fused = run_predictions(agent, evaluation, lambda case: retrieve_context(case, 3), True, 1)
    ensemble = run_predictions(agent, evaluation, lambda case: retrieve_context(case, 3), True, order_runs)
    verification = [predict_option_verification(agent, case, retrieve_context(case, 1)) for case in evaluation]
    verification_calibration = [
        predict_option_verification(agent, case, retrieve_context(case, 1)) for case in calibration_cases
    ]
    calibration_predictions = verification_calibration
    temperature = fit_temperature(calibration_cases, calibration_predictions)

    add_method(suite, "closed_book", evaluation, closed)
    add_method(suite, "context_head_truncation", evaluation, head)
    add_method(suite, "retrieved_top1", evaluation, top1)
    add_method(suite, "retrieved_top3_fusion", evaluation, fused)
    add_method(suite, "retrieved_top3_order_ensemble", evaluation, ensemble)
    add_method(suite, "optionwise_noul_verification", evaluation, verification)
    add_method(suite, "verification_temperature_calibrated", evaluation, verification, temperature)
    return suite


def find_models() -> Path:
    candidates = (
        ROOT / "models" / "laya-multilingual-npu" / "runtime",
        ROOT.parent / "models" / "laya-multilingual-npu" / "runtime",
    )
    for candidate in candidates:
        if (candidate / "onnx").exists():
            return candidate
    return candidates[0]


def load_npu_agent(models: Path, max_tokens: int, head_tokens: int) -> Any:
    source = ROOT / "laya-snapdragon"
    sys.path.insert(0, str(source))
    import laya_snapdragon
    import onnx

    context = next((models / "onnx").glob("laya_s*_m*_qnn_ctx.onnx"), None)
    if context is None:
        raise RuntimeError("no QNN HTP context model was found")
    output_names = [item.name for item in onnx.load(str(context), load_external_data=False).graph.output]
    action_output = "act_logits" if "act_logits" in output_names else "act"
    laya_snapdragon.OUTPUTS = ["logits", action_output]
    agent = laya_snapdragon.Agent(models, device="npu")
    agent.cfg["max_len"] = max_tokens
    agent.cfg["head_max_len"] = head_tokens
    if not agent.buckets:
        raise RuntimeError("no QNN HTP buckets were loaded")
    if max_tokens > max(bucket[0] for bucket in agent.buckets):
        raise ValueError("max_tokens exceeds the largest compiled NPU bucket")
    return agent


def print_summary(results: dict[str, Any]) -> None:
    print("\nNPU accuracy demo results")
    print("suite       method                                acc     nll   brier     ece   p50 ms  calls")
    for suite_name, suite in results["suites"].items():
        for method, value in suite["methods"].items():
            print(
                f"{suite_name:11s} {method:36s} "
                f"{value['accuracy']:6.3f} {value['nll']:7.3f} {value['brier']:7.3f} "
                f"{value['ece']:7.3f} {value['p50_case_ms']:8.2f} {value['npu_calls']:6d}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NPU-first accuracy improvements for Laya")
    parser.add_argument("--models", type=Path, default=find_models())
    parser.add_argument("--cache", type=Path, default=ROOT / ".demo-data")
    parser.add_argument("--output", type=Path, default=ROOT / "demo-results" / "npu-accuracy.json")
    parser.add_argument("--datasets", nargs="+", default=["typed", "jglue", "lawqa"], choices=["typed", "jglue", "lawqa"])
    parser.add_argument("--limit", type=int, default=40, help="evaluation decisions per suite")
    parser.add_argument("--calibration", type=int, default=20, help="held-out decisions used only for temperature fitting")
    parser.add_argument("--order-runs", type=int, default=4, help="choice-order rotations in an ensemble")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--head-tokens", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    if args.limit < 1 or args.calibration < 1:
        parser.error("--limit and --calibration must be positive")
    agent = load_npu_agent(args.models.resolve(), args.max_tokens, args.head_tokens)

    # Warm the HTP path before timing the benchmark methods.
    warm = DecisionCase("warmup", "warmup", "短い確認", "障害ですか", "noul", None, "false")
    predict_once(agent, warm, warm.state)

    results: dict[str, Any] = {
        "system": {
            "machine": platform.machine(),
            "backend": "QNNExecutionProvider/HTP",
            "device_mode": "npu",
            "cpu_model_fallback": False,
            "models": str(args.models.resolve()),
            "compiled_buckets": [f"{seq}x{markers}" for seq, markers, _ in agent.buckets],
            "max_tokens": args.max_tokens,
            "head_tokens": args.head_tokens,
        },
        "configuration": {
            "limit": args.limit,
            "calibration": args.calibration,
            "order_runs": args.order_runs,
            "seed": args.seed,
        },
        "suites": {},
    }
    runners = {
        "typed": lambda: run_typed_suite(agent, args.cache, args.limit, args.calibration, args.offline, args.order_runs),
        "jglue": lambda: run_jglue_suite(agent, args.cache, args.limit, args.calibration, args.offline, args.order_runs, args.seed),
        "lawqa": lambda: run_lawqa_suite(agent, args.cache, args.limit, args.calibration, args.offline, args.order_runs, args.seed),
    }
    for name in args.datasets:
        print(f"running {name} on QNN HTP...", flush=True)
        results["suites"][name] = runners[name]()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(results)
    print(f"\nfull results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
