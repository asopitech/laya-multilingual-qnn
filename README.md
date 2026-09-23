# Laya Multilingual QNN

Run Laya's multilingual decision model on the Qualcomm Hexagon NPU on Windows
on ARM using ONNX Runtime QNN. The model returns typed `choice`, `score`, and
`noul` decisions without generating text.

This repository adapts the QNN bucket preparation and serving path from
[`piffie/laya-snapdragon`](https://github.com/piffie/laya-snapdragon) to the
multilingual FP16 ONNX checkpoint. The checkpoint, tokenizer, and compiled QNN
contexts are downloaded or generated locally and are not committed here.

## Tested system

- Lenovo Yoga, Snapdragon X Elite X1E78100 (12 cores), 32 GB RAM
- Windows 11 ARM64, native ARM64 Python 3.11
- ONNX Runtime QNN 1.24.4
- Laya multilingual ONNX revision `d9d003d543e63d6d3375c21d44624136bd1e0bad`
- QNN sequence buckets: 128 and 256 tokens, 8 option slots

## Setup

Use native ARM64 Python 3.11. `platform.machine()` should report `ARM64`.

```powershell
git clone --recurse-submodules https://github.com/asopitech/laya-multilingual-qnn.git
cd laya-multilingual-qnn
py -V:3.11-arm64 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download the pinned ONNX checkpoint and compile the QNN contexts. The first
setup downloads about 617 MB and compiles two fixed-shape models; allow a few
minutes for compilation.

```powershell
python .\prepare_multilingual_npu.py
```

By default, this creates `models\laya-multilingual-npu\runtime` and compiles
the 128- and 256-token buckets. Use `--seq 128` to build only the short-input
bucket, or pass another list such as `--seq 128 256 512` if you want to test a
longer bucket. Long buckets can spill heavily from NPU memory and may be slower.

## Run the service

```powershell
.\start.ps1
```

The service listens on `http://127.0.0.1:8788` by default. Test it with the
included Japanese request:

```powershell
Invoke-RestMethod http://127.0.0.1:8788/health
$env:LOCAL_JEV_URL = "http://127.0.0.1:8788"
node .\example-client.mjs
```

The JSON request shape is `{"state": ..., "questions": ...}`. Each question
uses Laya's standard typed schema. The response includes option probabilities
and `usage.output_tokens: 0`.

## Benchmark

With the service running, use a second terminal:

```powershell
python .\benchmark.py --runs 15 --warmups 3 --concurrency 16
```

Measured on the tested Snapdragon X Elite, after warm-up:

| Questions per request | p50 | p95 | Throughput |
|---:|---:|---:|---:|
| 1 | 16.93 ms | 17.18 ms | 59.07 questions/s |
| 5 | 83.16 ms | 85.17 ms | 60.13 questions/s |
| 10 | 165.99 ms | 167.82 ms | 60.24 questions/s |
| 50 | 827.78 ms | 834.98 ms | 60.40 questions/s |

Sixteen concurrent one-question requests completed in 523 ms (30.58 requests/s).
The compiled QNN graph has batch size 1, and the server serializes access to its
shared session. Put related decisions in one request for the best throughput.

The same Japanese routing and outage request selected the same answers on CPU
and NPU. Routing probability was 0.9686 on CPU and 0.9682 on NPU; outage
probability was 0.8430 on both. The NPU's 128-token context compiled without
VTCM spill; the 256-token context reported about 224 MB spill traffic.

For comparison, the [upstream Laya model card](https://huggingface.co/convaiinnovations/laya/blob/main/README.md)
reports 32.8 ms for one question and 337 ms for 50 questions on a Tesla T4.
These are different devices and should not be treated as a controlled hardware
comparison.

## Limits

- Each question is routed independently through a batch-1 QNN context.
- The default buckets support up to 256 input tokens and 8 options per question.
- Requests outside the compiled buckets fail in this NPU-only service. Build a
  larger bucket or use the CPU fallback implementation.
- QNN uses FP16 execution. Small probability differences from CPU FP32 are
  expected; validate thresholds on your own workload.
- The reported probabilities have not been recalibrated on application data.

## License

This adapter is provided under Apache-2.0. The Laya weights retain their
upstream license. The QNN runtime is included as a pinned Apache-2.0 submodule;
see [NOTICE](NOTICE) and `laya-snapdragon/NOTICE`.

