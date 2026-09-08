# llama-bench-tuner 拡張計画

作成日: 2026-08-30

## 目的

`llama-bench-tuner` は engine benchmark、capability probe、parser、telemetry/result記録、mechanical MTP-spec benchmarkを担当する。P40候補inventoryと実験方針は `p40-llm-lab` に置く。

全体方針は、運用記録を次の申し送りメモに集約する。

`/home/masaya/projects/kennel-system-laboratory/docs/ops/2026-08-30_llama-bench-tuner_latest-local-llm-evaluation-handover.md`

## リポジトリの責務

| リポジトリ | 担当 |
| --- | --- |
| `llama-bench-tuner` | Engine benchmark、capability probe、行保持parser、telemetry/result schema、mechanical MTP-spec benchmark、Grid/Optuna、可視化 |
| `p40-llm-lab` | P40候補inventory、KEEP/No-Go方針、context/KV/MTP/MoE実験方針 |
| `llm-warroom` | Coding Agent workload harness（#52/#53） |
| `kennel-system-laboratory` | ホスト状態、実施記録、運用変更、結果へのリンク |
| `llama.cpp` / `llama.cpp-laurent` | backend本体、実験的実装、ビルド成果物 |

モデル重み、秘密情報、巨大なrawログは `llama-bench-tuner` のGit管理対象にしない。

## 現状確認

- `tune.py` は `ngl`、`batch`、`flash-attn` のGrid探索が中心。
- `optuna_tune.py` は同じ基本軸の探索を行う。
- 現行結果は主に prefill/decode tok/s と実行条件をCSVへ保存する。
- 共通のTTFT、context depth、KV cache、MTP/speculation、MoE offload、GPU/RAM peak schemaは未整備。
- P40候補選定と実験方針は本リポジトリの対象外である。
- `kennel-system-laboratory` は実装本体ではなく、運用・ホスト状態の記録場所である。

## 実装状況

- Phase 1はappend-only共通項目、行保持parser、command builder、status taxonomy、read-only capability probeを提供する。
- 既存のGrid checkpoint/resumeとmetadata試作は保持し、既存列や可視化入力を置換しない。
- pp/tg/depth、KV、MTP/speculation、MoE offload、GPU telemetryは未実装。

## 段階的な実装

### Phase 1: 互換性/capability基盤

対象候補:

- `src/llama_bench_tuner/schema.py`
- `src/llama_bench_tuner/parsing.py`
- `src/llama_bench_tuner/command_builder.py`
- `src/llama_bench_tuner/status.py`
- `src/llama_bench_tuner/capabilities.py`
- `src/llama_bench_tuner/tune.py`
- `src/llama_bench_tuner/optuna_tune.py`

実施内容:

- requested offload/placement とruntimeで観測された値を別項目にするappend-only schemaを定義する。
- 既存CSVの列とCLIを維持し、新しい列は追加扱いにする。
- native llama-bench CSVの各行を保持しながら従来のpp/tg集約を維持する。
- Grid/Optunaのcommand lineと可視化入力の互換性を保つ。
- `llama-bench --help` に根拠のあるoptionだけをprobeし、success/failed/OOM/unsupported/timeout/skippedを分類する。

### Phase 2: ベンチマーク軸の拡張

- schema検証済みの外部ファイルからmodel profileを読み込む。P40候補inventoryやKEEP/No-Go方針はここに埋め込まない。
- pp/tg を独立した測定ケースにする。
- context depth sweep を設定ファイルで表現する。
- batch/ubatch、KV cache、flash-attn、MTP/speculation、MoE offloadを任意軸として扱う。
- 未対応CLIオプションは推測して実行せず、capability checkで明示的にスキップする。
- GridとOptunaの既存出力を壊さず、条件を併記しないtok/sを比較対象にしない。

### Phase 3: Engine telemetry統合

benchmarkのengine telemetryはここで記録する。P40候補選定と実験方針は `p40-llm-lab` 側の責務である。

- GPU0 OCuLink / GPU1 USB4の分離測定。
- GPU温度、core/memory clock、power、utilization、throttle reasonの時系列。
- 10〜20分以上の連続decodeと時点別tok/s。
- 1 GPU = 1 worker の並列処理と必要時のみdual-GPU split。

共通schemaの意味は合わせるが、Pythonパッケージ間の直接依存は作らない。

### 対象外: coding-agent workload

固定repo・issue・tool set・test commandを使うagent評価は `llm-warroom` #52/#53 と `p40-llm-lab` #9 の担当である。`llama-bench-tuner`には実装しない。

## 受け入れ条件

- 既存の `llama-tune`、`llama-tune-optuna`、可視化CLIが従来の引数で動く。
- 既存サンプルのCSV列・可視化を壊さない。
- 各新規ジョブに条件、実行環境、結果、エラー、経過時間が紐付く。
- 5分超のジョブは進捗表示と単一checkpointを持つ。
- 実機変更やサービス再起動は専用の運用手順とロールバックを伴う。

## 実装順序

1. Phase 1の互換性/capability基盤と既存CLI回帰確認。
2. pp/tg/depthの設定表現とrunner。
3. KV・MTP/speculation・MoEのcapability-awareな拡張。
4. `p40-llm-lab` との結果項目の意味を合わせ、inventory/方針は取り込まない。
5. 必要時のみ外部のcoding-agent harness interfaceを連携する。
