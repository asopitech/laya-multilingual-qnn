# NPU accuracy improvement demos

`npu_accuracy_demo.py` compares inference-time accuracy improvements with the
Laya encoder and decision head on the Qualcomm QNN HTP path. It never creates
an `Agent(device="cpu")` and fails if a decision does not fit a compiled NPU
bucket. The existing QNN session may leave its small `GatherND` operation on
the CPU EP, but it never falls back to the dynamic FP32 CPU model.

The demo downloads and caches public evaluation data under `.demo-data/`:

- Typed Decisions: criteria descriptions, option-order ensembles, calibration
- JGLUE JNLI and JCommonsenseQA: Japanese three-way and five-way decisions
- Digital Agency LawQA: closed-book, context truncation, lightweight retrieval,
  multi-chunk probability fusion, option-order ensembles, per-option `noul`
  verification, and calibration

The default prompt budget is 256 tokens with a 160-token question/option head,
matching the checked-in setup instructions for the 128/256 x 8 QNN contexts.
CPU results are intentionally outside this demo's main table.

Run with the native ARM64 Python environment used to prepare the model:

```powershell
& ..\.runtime\python311-arm64-embed\python.exe `
  .\demos\npu_accuracy_demo.py `
  --models ..\models\laya-multilingual-npu\runtime `
  --limit 40 `
  --calibration 20
```

After the first run, repeat without network access:

```powershell
& ..\.runtime\python311-arm64-embed\python.exe `
  .\demos\npu_accuracy_demo.py --offline
```

The JSON output is written to `demo-results/npu-accuracy.json`. Accuracy is
reported with NLL, multiclass Brier score, ECE, KL to soft gold labels when the
dataset provides them, NPU latency, QNN bucket counts, and NPU call throughput.

Temperature calibration uses a disjoint training/calibration split and does
not change top-1 accuracy. Its purpose is to improve probability quality.
Option-order and multi-chunk ensembles may improve accuracy, but deliberately
trade extra NPU calls and latency for robustness.

## Measured exploratory result

The following result was measured on the tested Snapdragon X Elite with 40
evaluation decisions and 20 separate calibration decisions per suite. Every
forward pass used `Agent(device="npu")`; the result metadata reports
`cpu_model_fallback: false`.

| Suite | Baseline | Compared method | Accuracy | Median per case |
|---|---:|---|---:|---:|
| Typed Decisions | 0.325 | temperature-calibrated order ensemble | 0.300 | 52.42 ms |
| JGLUE | 0.525 | option-order ensemble | 0.650 | 58.67 ms |
| LawQA | 0.250 | per-option `noul` verification | 0.500 | 207.73 ms |

For Typed Decisions, calibration did not improve top-1 accuracy, but reduced
Brier score from 0.495 to 0.261 and NLL from 1.892 to 1.243. Simple criteria
descriptions and blindly prepending legal context were not improvements in this
run. The LawQA gain came from decomposing one four-way choice into four NPU
evidence checks. These are small exploratory samples, not final benchmark
claims; increase `--limit` before comparing model releases.

Dataset text is not committed to this repository. Review the upstream licenses
before redistributing cached data: Typed Decisions is Apache-2.0, JGLUE is
CC BY-SA 4.0 with component-specific terms, and LawQA uses Japan's Public Data
License 1.0.

## RLCD fine-tuning

`rlcd_finetune.py` performs an actual weight update of both the multilingual
encoder and the typed decision head. It follows the upstream RLCD recipe:
zero-mean logit exploration, proper-scoring reward, a group-mean baseline, and
cross-entropy guidance. QNN HTP is an inference target, so training runs in
PyTorch and the resulting checkpoint is exported and compiled for NPU use in a
separate step.

```powershell
& ..\.runtime\python311-arm64-embed\python.exe demos\rlcd_finetune.py `
  --model-dir ..\models\laya-multilingual-trainable `
  --output-dir ..\models\laya-multilingual-ja-rlcd `
  --steps 4
```

The output `training_report.json` records independent SHA-256 fingerprints for
the encoder, decision head, and action head before and after training. The run
fails unless both the encoder and decision head changed.
