# Laya Multilingual QNN Research Report

**調査日:** 2026-09-23  
**対象:** `asopitech/laya-multilingual-qnn` のソースと、Snapdragon X Elite上で記録された測定値  
**目的:** 多言語Layaの日本語入力をWindows ARM64上のQualcomm NPUで実行する方式、性能、品質上の制約を整理する。

## 要約

本実装は、Laya多言語版のONNXモデルをONNX Runtime QNN Execution Provider経由でSnapdragonのHexagon HTP/NPUに載せるローカル推論アダプターである。Layaは自由文を生成するLLMではなく、与えた状態に対して`choice`、`score`、`noul`（二値確率）を返す非自己回帰型の判定モデルである。モデルの追加学習や日本語専用化を行ったものではない。

リポジトリに記録されたSnapdragon X Elite測定では、短い入力のサービスP50は1問16.93 ms、5問83.16 ms、10問165.99 ms、50問827.78 msで、およそ59–60問/秒に相当する。これはローカルREADME記載の実測値であり、この調査では対象端末上で再測定していない。16並列のHTTP要求はサーバー内で直列化され、全体523 ms、30.58要求/秒だった。

日本語について示されている品質比較は、ルーティング判定と障害判定の2問でCPU実行とNPU実行の回答が一致した例に限られる。多言語の代表性ある日本語正解データで精度、校正誤差、長文性能を測った結果ではない。したがって現段階の結論は「Snapdragon X Eliteで実行可能で、記録値上は約60問/秒の判定スループットが得られた」であり、「日本語業務で正確」「確率が業務上校正済み」とまでは結論できない。

## 対象モデルと処理方式

上流モデルカードによると、Laya multilingualはmmBERT-baseを基盤とする約322Mパラメーターの多言語判定モデルで、100超の言語を対象にし、標準コンテキスト長は1024トークンとされる。候補ラベルを入力時に与え、生成文ではなく候補ごとの確率を返す。[Laya multilingual model card](https://huggingface.co/convaiinnovations/laya-multilingual)

本リポジトリの準備スクリプトは、Hugging Faceの`mizchi/laya-multilingual-onnx`をコミット`d9d003d543e63d6d3375c21d44624136bd1e0bad`に固定して取得する。設定、トークナイザー、ONNX本体をローカルへ保存し、入力系列長を128/256、候補marker数を8、batchを1に固定したONNXグラフを作ってQNN HTP用コンテキストをコンパイルする。可変長のままHTPへ渡すのではなく、入力長に合う静的バケットを選ぶ構成である。

実行時はRust実装の`tokenizers`でトークン化し、Laya互換のプロンプトと候補markerを構築する。既存QNN実装由来のグラフ書き換えで`IsNaN`を`Not(Equal(x,x))`へ置換し、erf形式GELUをONNX RuntimeのfusionでGelu演算へまとめる。QNNセッションはHTPのburst設定で作られ、`GatherND`等の非対応部分にはCPU EP fallbackを許す。QNN EPはSnapdragon上のONNXモデルをQualcomm AI Runtime経由で実行するONNX Runtimeの公式経路で、コンテキストの事前コンパイルと再利用もサポートする。[ONNX Runtime QNN EP documentation](https://github.com/onnx/onnxruntime-qnn/blob/main/docs/execution_providers/QNN-ExecutionProvider.md)

HTTPサーバーは`127.0.0.1:8788`にバインドし、`/decision`と`/v1/systemone`を受け付ける。これはローカル専用で、TypeSafe Jev APIの実装・互換サーバーではない。リポジトリのREADMEとコードを参照: [source repository](https://github.com/asopitech/laya-multilingual-qnn)。

## 性能結果

下表はリポジトリのREADMEに記録されているSnapdragon X Elite測定値。P50/P95はサーバー側の処理時間で、HTTP応答のJSON直列化・転送時間は含まない。問数ごとのスループットは`問数 / P50`から算出される。

| 1要求あたりの問数 | P50 | 報告P95 | 算出スループット |
|---:|---:|---:|---:|
| 1 | 16.93 ms | 17.18 ms | 59.07問/秒 |
| 5 | 83.16 ms | 85.17 ms | 60.13問/秒 |
| 10 | 165.99 ms | 167.82 ms | 60.24問/秒 |
| 50 | 827.78 ms | 834.98 ms | 60.40問/秒 |

問数を増やしても約60問/秒で推移し、1問あたりの推論時間はほぼ線形に積み上がっている。これは複数質問を同時batch推論しているという意味ではない。`Agent.forward`は質問ごとにbatch-1のNPUセッションを呼び出す。サービスは共有セッションをロックで保護するため、16個の同時HTTP要求も順番に実行され、集計では30.58要求/秒だった。

### 測定上の注意

- `benchmark.py`の既定値はwarm-up 3回、計測15回である。実装のP95関数はソート後の`int(n × 0.95)`番目を選ぶため、n=15では最大値をP95として表示する。標準的な補間P95とは異なる。
- 問/秒は計測全体の処理数を経過時間で割ったaggregate throughputではなく、P50時間から算出した参考値である。再現ベンチでは実経過時間あたりの完了問数も併記するのがよい。
- READMEはPC機種、OS、Python、ORT、モデルrevision、バケットを示すが、電源モード、温度、バックグラウンド負荷、各試行の生ログは同梱していない。よって測定の独立再現性には限界がある。
- 上流Laya model cardのTesla T4値（1問32.8 ms、50問337 ms）は異なるGPU・runtime・計測条件の値で、Snapdragonとの直接的な優劣比較には使えない。[Laya benchmark section](https://huggingface.co/convaiinnovations/laya#benchmarks)

## 品質・確率の解釈

リポジトリ記録の日本語例では、CPUとNPUでroutingとoutageの選択結果が一致し、routing確率は0.9686対0.9682、outage確率はどちらも0.8430（表示精度）だった。これは同一モデルのFP32 CPUとHTP実行の差を見る小さな回帰テストとして有用だが、回答が正解かを独立ラベルで検証したものではない。

さらに上流の現行多言語モデルカードは、チェックポイントが未校正で確率が過信寄りになりうること、typed-decision系タスクのゼロショット成績が低い例、`score`が弱い例、低リソース言語の性能課題を明記している。これらは現行カードの結果であり、固定済みONNX revisionと完全に同一の測定とは限らないが、少なくとも確率値をそのまま業務上の信頼度として扱わない理由になる。[Model card limitations and benchmarks](https://huggingface.co/convaiinnovations/laya-multilingual#limitations)

したがって、運用前には対象業務の日本語データで、以下を別々に測る必要がある。

- 選択精度、macro-F1、クラス別再現率（`choice`）
- 二値判定のprecision/recall、Brier score、ECE（`noul`）
- 順序尺度のMAEおよび隣接ラベル誤り率（`score`）
- FP16 NPUとFP32 CPUの回答一致率・確率差
- 長さ、選択肢数、文体、否定、曖昧さ別の失敗率

確率閾値を使う場合は、学習・評価の混同を避けた保留データで校正し、モデルrevision、ONNX/QNN runtime、校正パラメーターを一緒に記録する。

## 適用範囲と制約

- 既定のNPUバケットは系列長128/256、1問あたり8候補まで。超過した入力はサービスで`device="npu"`を指定しているためエラーになる。ライブラリの`device="auto"`にはCPU fallbackがあるが、HTTPサービスはそれを使わない。
- 256-tokenバケットは実機コンパイルログ上、約224 MBのVTCM spill trafficが記録されている。128-tokenではspillなしとREADMEに記載される。長い入力ほど常に速いとは限らず、より大きなバケットを追加する場合は再測定が必要。
- 生成済みONNX固定形状ファイルとQNN context binaryはリポジトリに含まず、各端末で生成する。QNN contextは対象SoC/HTP/QAIRT環境との互換性を確認する必要があり、異なるPCへそのまま配布できる前提ではない。
- リポジトリは`onnxruntime-qnn==1.24.4`を指定している。現行公式QNN EPドキュメントはv2.6.0/ORT 1.27.0を案内し、v2未満を非推奨と記載する。現行版へ上げるべきかは、同じモデルで回答差、コンパイル互換性、レイテンシを測ってから判断する。[Current QNN EP versions](https://github.com/onnx/onnxruntime-qnn/blob/main/docs/execution_providers/QNN-ExecutionProvider.md#qnn-execution-provider-version-requirements)
- サーバーの初期版は認証機構を持たず、既定bind先をloopbackに限定している。公開ネットワークへbind変更する用途のAPIサーバーではない。

## 推奨する次の検証

1. 日本語の業務別ホールドアウトを作り、回答精度と確率校正を評価する。特に`noul`/`score`の閾値をオフラインで決める。
2. ベンチを最低50–100計測回に増やし、ウォームアップ、温度・電源条件、平均/中央値/P95/P99、実経過時間あたり問数、モデル初期化時間を記録する。
3. 128/256を基準に、512以上のバケットを追加した際の実測レイテンシ、spill、常駐メモリ、電力を比較する。
4. 固定モデルrevisionでQNN EP 1.24.4と現行2.xを同一端末で比較する。回答一致率だけでなくcold-startとcontext再利用後を分ける。
5. 異なるSnapdragon世代・HTPで動作確認し、端末ごとのcontext再生成が必要か明記する。

## 根拠資料

- 対象実装・固定revision・ベンチコード: [asopitech/laya-multilingual-qnn](https://github.com/asopitech/laya-multilingual-qnn)
- 多言語モデル構成・既知の限界: [Convai Innovations Laya multilingual model card](https://huggingface.co/convaiinnovations/laya-multilingual)
- この実装が取得するONNX export: [mizchi/laya-multilingual-onnx](https://huggingface.co/mizchi/laya-multilingual-onnx)
- QNN EP、Windows ARM64、HTP、context cache: [ONNX Runtime QNN EP documentation](https://github.com/onnx/onnxruntime-qnn/blob/main/docs/execution_providers/QNN-ExecutionProvider.md)
- 上流LayaのT4測定・モデルベンチ: [Convai Innovations Laya model card](https://huggingface.co/convaiinnovations/laya)

## 再現性に関する注記

このレポート作成時点では、本実行環境に`py` Python Launcherがなく、Snapdragon上でのNPU実行・測定は再実行していない。数値はリポジトリREADMEに記載された実測結果として引用し、モデルカードおよびONNX Runtime公式資料で仕様・制約を照合した。レポート内の性能結論は独立した再測定ではない。
