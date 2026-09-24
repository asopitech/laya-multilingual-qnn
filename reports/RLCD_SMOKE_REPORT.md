# Laya multilingual 日本語RLCD更新・NPU再展開 実測レポート

実施日: 2026-09-24
対象PC: Lenovo YOGA-COPILOT / Windows ARM64 / Snapdragon X Elite / Qualcomm HTP NPU / RAM 32GB

## 結論

encoderとdecision headの両方を更新するRLCD学習を、実際に4 optimizer step実行した。更新済みcheckpointをPyTorchからONNXへexportし、128x8および256x8のQNN HTPコンテキストへcompileしたうえで、NPU上の精度と速度を再測定した。

ただし、これは学習パイプラインを実証するsmoke runであり、精度改善版ではない。40件のPyTorch validationはaccuracy `0.425 -> 0.375`、NPU上のJGLUE option-order ensembleは`0.650 -> 0.550`となった。更新済みcheckpointを本番候補として採用してはならない。

## このPCの学習上の制約

**このPCだけでは、322M parameterのLayaに対する実用的な規模の強化学習・RLCD fine-tuningをまともに実行できない。** このPCにはNVIDIA GPUとCUDAがなく、利用できるAI acceleratorはSnapdragon X EliteのQualcomm HTP/NPUだけである。現在のQNN HTP経路はONNX Runtimeによる推論専用であり、PyTorchのautograd、backward、optimizer stepをNPU上では実行できない。

ローカルのPyTorchは`2.14.0+cpu`で、`USE_CUDA=OFF`のWindows ARM64 buildである。したがって、重みを更新する処理はすべてCPUで行われる。RAMも32GBのため、322Mモデルのparameter、gradient、AdamW optimizer state、activationを保持すると、大きなbatchや長いsequenceを使いにくい。

今回の最小設定でも、batch size 1、sequence 128、わずか4 optimizer stepに87.4秒、平均約21.9秒/stepを要した。学習sampleは16 decisionだけであり、複数epoch、大規模dataset、複数seed、hyperparameter探索を含む本格学習へそのまま拡大できる速度ではない。この4-step実行は以下だけを確認するsmoke testである。

- encoderとdecision headへ勾配が到達すること
- optimizerが両方の重みを実際に変更すること
- 更新済みcheckpointをONNX/QNNへ変換できること
- 更新済みモデルをNPUで推論・評価できること

精度向上を目的とする本学習は、NVIDIA CUDA GPUを持つ別PCまたはcloud GPUで実行し、完成したcheckpointだけをこのPCへ戻してONNX export、QNN compile、NPU評価する必要がある。このPCの主な役割は、データ準備、短い学習コードの動作確認、QNN変換、NPU推論、精度・速度評価である。

## 実行した更新

公式fine-tuning notebookと同じRLCDの中心部分を使用した。

1. decision logitsへzero-mean Gaussian noiseを加え、4個の探索分布を生成
2. proper scoring ruleでrewardを計算
3. group mean baselineでadvantageを計算
4. policy-gradient lossとsoft cross-entropy guidanceを合算
5. encoderとheadを異なるlearning rateでAdamW更新

| 項目 | 値 |
|---|---:|
| encoder parameters | 306,939,648 |
| decision head parameters | 14,770,945 |
| action head parameters | 198,402 |
| encoder learning rate | 2.5e-5 |
| head learning rate | 1.0e-4 |
| optimizer steps | 4 |
| batch size | 1 |
| exploration group size | 4 |
| sigma | 0.2 |
| max sequence | 128 |
| 学習decision | JNLI 8 + JCommonsenseQA 8 |
| validation decision | JNLI 20 + JCommonsenseQA 20 |
| seed | 20260924 |
| 学習時間 | 87.4秒 |

学習backendはPyTorch CPUである。QNN HTPは学習backendではなく推論先なので、重み更新後の成果物をNPUへ再展開して評価した。CPU精度を最終成果として扱っておらず、このCPU学習を実用的な本学習とも位置づけていない。

## 両方を更新した証拠

全parameter byte列からグループ別SHA-256を計算した。encoderとdecision headはどちらも学習前後で変化した。各stepでも両グループに非ゼロ勾配を確認した。

| parameter group | before SHA-256 | after SHA-256 | updated |
|---|---|---|---:|
| encoder | `79d7cfa13a24...` | `b6197fc30731...` | yes |
| decision head | `56e3fa214d48...` | `6793219a658a...` | yes |
| action head | `f76b2bdfc5c...` | `849737bd2da4...` | weight decay only |

| step | encoder gradient L2 | decision head gradient L2 | action head gradient L2 |
|---:|---:|---:|---:|
| 1 | 428.19 | 18.19 | 0.00 |
| 2 | 1,221.09 | 289.44 | 0.00 |
| 3 | 216.49 | 47.20 | 0.00 |
| 4 | 2,007.61 | 170.91 | 0.00 |

action headにはhuman escalationの正解labelを与えていない。AdamWのweight decayでbyte列は変化したが、学習したとは評価しない。この実験で明示的に学習した対象はencoderとtyped decision headである。

## PyTorch validation

| model | overall | JNLI | JCommonsenseQA | NLL |
|---|---:|---:|---:|---:|
| base | 0.425 | 0.700 | 0.150 | 1.584 |
| RLCD smoke | 0.375 | 0.550 | 0.200 | 3.120 |

4 decisionだけを更新した時点でJNLIの性能と確率品質が悪化した。大きなencoderを少数sample、batch size 1で更新したことによる過学習と不安定化が主因と考えられる。

## ONNX・QNN HTP展開

更新済みcheckpointを次の順に展開した。

```text
PyTorch safetensors
  -> dynamic FP32 ONNX
  -> fixed 128x8 / 256x8 ONNX
  -> QNN HTP FP16 context
```

NPU経路に収まった組み込みverification 3ケース9判定は、すべてCPU ONNXと同じtop-1だった。NPUとCPU ONNXの最大確率差は`3.4e-3`、CPU ONNXとPyTorchの最大差は`0.0`だった。685 tokenの長文ケースは256 bucketを超えるため、この一致件数には含めていない。

## NPU再評価

前回と同じseed、各suite 40 evaluation decisions、20 calibration decisionsで測定した。全model forwardは更新済みQNN HTP bucketを使用し、動的CPU model fallbackは無効である。小さな`GatherND`だけは既存runtime設計どおりCPU EPで実行される。

| suite / method | base accuracy | RLCD smoke accuracy | delta | smoke P50 |
|---|---:|---:|---:|---:|
| Typed labels only | 0.325 | 0.350 | +0.025 | 52.64 ms |
| Typed order ensemble | 0.300 | 0.325 | +0.025 | 53.32 ms |
| JGLUE direct | 0.525 | 0.475 | -0.050 | 16.75 ms |
| JGLUE order ensemble | 0.650 | 0.550 | -0.100 | 59.17 ms |
| LawQA closed book | 0.250 | 0.300 | +0.050 | 52.67 ms |
| LawQA optionwise noul | 0.500 | 0.475 | -0.025 | 210.49 ms |

レイテンシは元checkpointと同等だが、主要なJGLUE指標が悪化した。これは「full-weight RLCD -> ONNX -> QNN HTP -> NPU評価」の工程が成立することを示す結果であり、精度向上を示す結果ではない。

## 成果物

Git管理対象:

- `demos/rlcd_finetune.py`: RLCD学習と更新検証
- `prepare_finetuned_npu.py`: PyTorch checkpointのONNX/QNN展開
- `reports/RLCD_SMOKE_TRAINING.json`: 学習生ログ
- `reports/RLCD_SMOKE_NPU.json`: NPU精度・速度生ログ
- `tests/test_rlcd_finetune.py`: RLCD lossとparameter groupingのテスト

ローカル生成物（サイズが大きいためGit対象外）:

- 学習可能な公式checkpoint: `C:\Users\asopi\workspaces\ai-works\models\laya-multilingual-trainable`
- 更新済みPyTorch checkpoint: `C:\Users\asopi\workspaces\ai-works\models\laya-multilingual-ja-rlcd`
- 更新済みNPU runtime: `C:\Users\asopi\workspaces\ai-works\models\laya-multilingual-ja-rlcd-npu`

## 次の本学習

次回はsmoke設定を性能実験として拡大する。最低限、数千decision、balanced sampler、batch/gradient accumulation、validationによるearly stopping、encoder freeze baseline、head-only baseline、CE-only baselineを用意する。act headを学習するときは、誤自動実行コストとhuman escalationコストを持つ専用labelが別途必要である。

性能採用条件は、元checkpointのJGLUE ensemble `0.650`とLawQA optionwise `0.500`を同じholdoutで上回り、PyTorch/ONNX/QNN top-1 parity 99%以上を維持することとする。
