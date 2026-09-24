# Laya Multilingual QNN NPU精度改善 実測レポート

**実測日:** 2026-09-24
**対象:** `asopitech/laya-multilingual-qnn`
**目的:** 公開データセットを使い、Laya multilingualの精度改善手法をQualcomm NPU実行のまま比較する。

## 結論

推論時の工夫だけでも、タスクに応じて精度を改善できた。

- JGLUEでは選択肢順アンサンブルによりAccuracyが`0.525`から`0.650`へ上昇した。
- 法令QAでは4択を4個の`noul`根拠判定へ分解し、Accuracyが`0.250`から`0.500`へ上昇した。
- Typed DecisionsではAccuracyは改善しなかったが、温度校正によりBrier scoreが`0.495`から`0.261`、NLLが`1.892`から`1.243`へ低下した。
- 法令コンテキストを単純に前置するだけでは`0.250`から`0.200`へ悪化した。コンテキスト追加と精度向上は同義ではない。

今回の実験は40判定ずつの探索的評価であり、最終ベンチマークではない。一方で、すべて同一端末、同一QNNモデル、固定seed、CPUモデルfallbackなしで比較しているため、次の方式選定には使える。

## 実行環境

| 項目 | 値 |
|---|---|
| CPU/SoC | Snapdragon X Elite X1E78100 |
| OS | Windows 11 ARM64 |
| Python | native ARM64 Python 3.11 |
| 推論backend | ONNX Runtime QNN Execution Provider / HTP |
| ONNX Runtime QNN | 1.24.4 |
| モデル | Laya multilingual ONNX |
| モデルrevision | `d9d003d543e63d6d3375c21d44624136bd1e0bad` |
| QNNバケット | 128 x 8、256 x 8、batch 1 |
| 評価入力上限 | 256 tokens |
| question/option head | 160 tokens |
| 動的FP32 CPUモデルfallback | 無効 |
| seed | `20260924` |

実験は`Agent(device="npu")`を使用し、QNNバケットに収まらない入力は失敗させる。既存runtimeの設計上、小さな`GatherND`演算にはCPU EPを許可しているが、encoderとdecision headを動的FP32 CPUモデルへ切り替えるfallbackは使用していない。

## データセット

### Typed Decisions

4 workflowの`choice`、`score`、`noul`を使用した。評価40判定、校正20判定。soft gold確率を持つため、Accuracy以外にKL、Brier、NLLも評価した。

### JGLUE

JNLIとJCommonsenseQAを20件ずつ使用した。評価40判定、学習splitから校正20判定。JNLIは3択、JCommonsenseQAは5択で、どちらも8候補制限内に収まる。

### デジタル庁 法令QA

公開140問からseed固定で校正20問、評価40問を分離した。各問は法令コンテキストと4選択肢を持つ。closed-book、単純コンテキスト、文字bigram検索、複数チャンク融合、候補別根拠判定を比較した。

## 比較方法

| 方法 | 内容 |
|---|---|
| `labels_only` | Typed Decisionsの候補説明を外し、labelだけで判定 |
| `described_criteria` | 候補ごとの説明を含める |
| `option_order_ensemble` | 選択肢を回転させ、元labelへ戻して確率平均 |
| `temperature_calibrated` | 独立calibration splitで温度をfit |
| `closed_book` | 法令コンテキストなしの4択 |
| `context_head_truncation` | 法令コンテキストを先頭から256-token枠へ格納 |
| `retrieved_top1` | 問題・選択肢との文字bigram重複が最大の短いチャンクを使用 |
| `retrieved_top3_fusion` | 上位3チャンクを別々にNPU判定して確率平均 |
| `retrieved_top3_order_ensemble` | 上位3チャンクと4通りの選択肢順を組み合わせる |
| `optionwise_noul_verification` | 各候補について「法令根拠から正しいか」を個別二値判定 |

温度校正はargmaxを変えないため、Accuracy改善ではなく確率品質改善の手法である。

## 全実測値

### Typed Decisions

| 方法 | Accuracy | NLL | Brier | ECE | KL | P50/case | P95/case | NPU calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| labels only | 0.325 | 1.651 | 0.373 | 0.338 | 0.900 | 52.03 ms | 52.97 ms | 40 |
| described criteria | 0.300 | 1.957 | 0.535 | 0.416 | 1.250 | 52.13 ms | 53.08 ms | 40 |
| criteria order ensemble | 0.300 | 1.892 | 0.495 | 0.392 | 1.160 | 52.42 ms | 209.00 ms | 76 |
| ensemble + temperature | 0.300 | 1.243 | 0.261 | 0.155 | 0.470 | 52.42 ms | 209.00 ms | 76 |

`choice`だけを複数回判定し、`score`と`noul`は順序を固定するため、40ケースに対するensemble呼び出しは76回である。fit温度は上限候補の`4.0`となり、未校正出力が強く過信していることを示す。ただし、この小さいcalibration splitでは温度値自体を確定値として扱えない。

### JGLUE

| 方法 | Accuracy | NLL | Brier | ECE | P50/case | P95/case | NPU calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct | 0.525 | 1.325 | 0.610 | 0.187 | 16.61 ms | 16.83 ms | 40 |
| option order ensemble | 0.650 | 1.299 | 0.583 | 0.226 | 58.67 ms | 67.10 ms | 140 |
| ensemble + temperature | 0.650 | 1.125 | 0.588 | 0.228 | 58.67 ms | 67.10 ms | 140 |

Accuracyは12.5ポイント改善した。温度校正はNLLを改善したが、このsampleではBrierとECEをわずかに悪化させた。校正は指標を一律に改善する保証がなく、より大きい独立校正セットが必要である。

### 法令QA

| 方法 | Accuracy | NLL | Brier | ECE | P50/case | P95/case | NPU calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| closed book | 0.250 | 1.594 | 0.845 | 0.254 | 51.85 ms | 52.20 ms | 40 |
| context head truncation | 0.200 | 1.588 | 0.857 | 0.304 | 52.21 ms | 52.88 ms | 40 |
| retrieved top 1 | 0.225 | 1.744 | 0.900 | 0.316 | 51.87 ms | 52.46 ms | 40 |
| retrieved top 3 fusion | 0.225 | 1.591 | 0.840 | 0.262 | 155.88 ms | 157.01 ms | 120 |
| top 3 + order ensemble | 0.250 | 1.517 | 0.810 | 0.200 | 623.39 ms | 626.66 ms | 480 |
| option-wise `noul` verification | 0.500 | 1.347 | 0.734 | 0.230 | 207.73 ms | 210.88 ms | 160 |
| verification + temperature | 0.500 | 1.369 | 0.742 | 0.242 | 207.73 ms | 210.88 ms | 160 |

最も有効だったのは候補別`noul`検証である。4択を一度に解かせるのではなく、各候補を独立した根拠照合へ変換するとAccuracyが倍増した。代償として1問4回のNPU呼び出しが必要になる。単純な検索・チャンク平均は改善せず、12回呼び出すtop 3 + order ensembleもAccuracyはclosed-bookと同じだった。

## NPU性能上の解釈

- JGLUE directは全40件が128 x 8バケットで、約60 NPU calls/sだった。
- Typed Decisionsは多くが256 x 8で、約19 NPU calls/sだった。
- 法令QAの候補別検証は159/160 callsが256 x 8で、約19.3 NPU calls/sだった。
- Accuracyを上げるensemble/decompositionは1回のNPU速度を変えず、呼び出し数にほぼ比例してcase latencyを増やす。
- このため、低リスク判定はdirect、高リスク判定だけ候補別検証へ送る二段構成が実用的である。

## 強化学習について

### このPC固有の制約

このPCにはNVIDIA GPU/CUDAがなく、AI acceleratorはQualcomm HTP/NPUだけである。QNN HTPは本構成ではONNX推論専用で、PyTorchのautograd、backward、optimizerを実行できない。ローカルのPyTorchもCPU-only buildであるため、322M parameterのencoderとdecision headを更新する処理はCPUへ載る。

実際に行った最小smoke runでも、batch size 1、sequence 128、4 optimizer stepに87.4秒を要した。これは更新・変換経路の確認には使えるが、数千から数万decision、複数epoch、複数seedを必要とするまともなRLCD fine-tuningには不十分である。本学習はCUDA GPUを持つ別PCまたはcloudで行い、このPCは完成checkpointのQNN変換とNPU実測に使う。

### 現状

Laya multilingualの現在の重み自体はRLCD（Reinforcement Learning for Calibrated Decisions）で学習されている。ローカル設定には`training.updates=15987`、`epochs_completed=4`、`hours=4.97`、`act_costs.escalate=0.5`、`cost_wrong_act=3.0`が記録されている。

RLCDではモデルが候補確率分布をpolicyとして出力し、logitへzero-mean Gaussian noiseを加えて探索する。rewardは以下を組み合わせたstrictly proper scoring ruleである。

1. 正解確率のlog score
2. 予測分布とtarget分布のspherical score
3. `score`質問用のranked probability score

更新はgroup-mean baselineを使うREINFORCEで、上流はGRPO-styleと説明している。複数ターン軌跡にはTD(λ=1.0)を使う。通常の生成LLMに対するPPO/GRPOとは異なり、文章生成policyではなく小さな離散確率分布を直接最適化する。

### 今回実施した範囲

今回のデモはRL学習を行っていない。実施したのは固定済みRLCD checkpointに対する次の処理である。

- NPU推論時のprompt/criteria変更
- 選択肢順ensemble
- context retrievalとmulti-pass融合
- 問題の候補別`noul`分解
- held-out temperature calibration

したがって、法令QAの`0.250 → 0.500`は重み更新による改善ではない。

### 2026-09-24追試: encoderとdecision headの実更新

上記の固定checkpoint実験とは別に、encoderとdecision headをともに更新する4-step RLCD smoke runを実施し、更新済み重みをONNX/QNN HTPへ再展開した。両parameter groupのSHA-256変化と非ゼロ勾配を確認済みである。一方、NPU上のJGLUE option-order ensembleは`0.650 -> 0.550`へ低下したため、このcheckpointは改善版として採用しない。設定、全実測値、成果物は[RLCD_SMOKE_REPORT.md](RLCD_SMOKE_REPORT.md)を参照。

### NPU版でRLCDを行う手順

QNN HTPはこのモデルの学習backendではないため、学習はCUDA GPU等のPyTorch環境で行い、完成した重みをNPUへ展開する。

```text
日本語のstate/questions/soft targetを作成
  -> GPU上でRLCD fine-tuning
  -> 独立validationでtemperature fitting
  -> PyTorch checkpointをONNX export
  -> 128/256 x 8の固定shapeへ変換
  -> QNN contextを再コンパイル
  -> PyTorch/ONNX/QNNの確率parity確認
  -> 本レポートと同じNPU評価を再実行
```

上流の公式notebookは2 x T4で約30,000 questions、4 epochs、4から5時間を目安としている。Typed Decisionsでは上流報告上、baseの約0.35からfine-tuned checkpointの0.766まで改善している。ただしこれは英語の4 workflowに特化した値であり、日本語法令QAへそのまま一般化しない。

### 推奨RLCD実験

| 実験 | 学習target | 目的 |
|---|---|---|
| CE/SFT baseline | one-hot label | RLCDとの差を分離する基準 |
| RLCD hard target | one-hot分布 | Accuracyと校正を同時最適化 |
| RLCD soft target | 複数annotatorまたはteacherの確率分布 | 曖昧さを保持して過信を抑える |
| cost-sensitive act | 誤自動実行コストとhuman escalationコスト | selective automationを学習 |
| on-policy correction | 実運用で誤った・人へ戻されたケース | 分布ずれを補正 |

本用途では、公開データだけでなく実際の日本語業務判断を最低数千decision収集する必要がある。train/calibration/testを分離し、同一文書や同一事件がsplitをまたがないようgroup splitする。法令QA140問だけでencoder全体をRLCD fine-tuningすると過学習の可能性が高い。

`action.act_probability`は現行上流model cardで有用な信号ではないと報告されているため、現checkpointの値をhuman escalation gateへ直接使うべきではない。まず通常のanswer confidenceまたは明示的な`choice`/`noul`でgateを作り、act headを再学習した後にcoverage-risk curve、誤自動実行率、human escalation率で評価する。

### 次の合格条件案

- JGLUE Accuracyが現在の0.650を上回る。
- 法令QAの候補別検証Accuracyが現在の0.500を上回る。
- 業務固有testでECE 0.10以下、かつBrier/NLLがbaseより改善する。
- PyTorchとQNN FP16でargmax一致率99%以上を維持する。
- 1回判定のNPU P50を現在の128/256 bucket水準から大きく悪化させない。
- confidence gate採用時は、coverage別Accuracyと誤自動実行率を必ず併記する。

## 制約

- 各suite 40判定の小規模評価で、信頼区間を狭くする件数ではない。
- JGLUEはJNLIとJCommonsenseQAの集計値であり、データセット別の値を分離していない。
- 公開データがmmBERTの事前学習や上流評価に含まれた可能性を完全には排除できない。
- 法令QAの文字bigram検索はNPU入力圧縮の簡易baselineであり、本格的なembedding retrieverではない。
- calibration 20件は小さく、温度`4.0`や`2.928`をproduction値として固定できない。

## 再現方法

初回は公開データを取得する。

```powershell
& ..\.runtime\python311-arm64-embed\python.exe `
  .\demos\npu_accuracy_demo.py `
  --models ..\models\laya-multilingual-npu\runtime `
  --limit 40 `
  --calibration 20 `
  --order-runs 4 `
  --output demo-results\npu-accuracy-40.json
```

データ取得後は`--offline`で再実行できる。実装は`demos/npu_accuracy_demo.py`、補助ロジックのテストは`tests/test_npu_accuracy_demo.py`にある。

## 参照資料

- [Laya公式READMEとRLCD説明](https://github.com/NandhaKishorM/laya)
- [Laya公式Typed Decisions fine-tuning notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)
- [Typed Decisions dataset](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
- [JGLUE](https://github.com/yahoojapan/JGLUE)
- [デジタル庁 法令QA](https://github.com/digital-go-jp/lawqa_jp)
