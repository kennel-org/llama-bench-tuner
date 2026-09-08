# llama-bench-tuner 更新フィードバックメモ

作成日: 2026-08-27
対象: Codex による `llama-bench-tuner` 改修
目的: llama.cpp の更新、新しい MoE / long-context / speculative decoding 対応モデルの登場を踏まえ、ベンチマーク対象モデル・測定パラメータ・評価指標を更新する。

---

## 1. 今回の改修方針

従来の `tok/s` 単独比較から、以下を分離して計測できる構成へ更新したい。

- Prefill throughput
- Decode throughput
- TTFT
- context depth に対する性能低下
- KV cache 量子化の影響
- MTP / speculative decoding の有無
- MoE の GPU / CPU / hybrid offload 差
- VRAM 使用量
- System RAM 使用量
- 長時間 agent workload での安定性

重要:

> 「何 tok/s 出たか」だけでは比較しない。
> prompt token 数、生成 token 数、context depth、batch、KV cache、speculation/MTP、MoE offload 条件を必ず記録する。

また、通常の `llama-bench` micro benchmark と、実際の coding-agent workload は分けて扱う。

---

# 2. 主な検証マシン

## x1ai

- Ubuntu 22.04
- RAM 64GB
- Tesla P40 24GB x2
- GPU0: OCuLink
- GPU1: USB4
- Ollama + llama.cpp

P40 x2 を単純な tensor parallel 用 48GB GPU として扱うことを主目的にはしない。

優先順位:

1. GPU0 / GPU1 個別 benchmark
2. OCuLink vs USB4 差
3. 1 GPU = 1 worker の並列 aggregate throughput
4. 必要に応じて dual-GPU split / tensor parallel

特に MoE / CPU offload では PCIe 接続差が結果に直接出る可能性があるため、
GPU0 / GPU1 の結果を混ぜない。

---

## Precision 3680

- RTX 4000 Ada 20GB
- RAM 96GB ECC
- Windows 11
- WSL2 Ubuntu 22.04
- CUDA 12.x 系
- NVMe 4TB

役割:

- Ada 世代での llama.cpp / EXL3 / FreeToken 比較
- 20GB VRAM + 96GB RAM を使った hybrid offload
- Qwen3.8-Flash-Next 等の「VRAMを超えるモデル」の実験
- coding-agent workload の主評価機

---

# 3. 基本 benchmark を pp/tg 分離へ統一

最低限以下を固定測定する。

## Prefill

候補:

- pp512
- pp4096
- pp16384
- pp32768

可能なモデルでは:

- pp65536
- pp131072

長文脈モデルでは context window だけでなく、
実際の prefill throughput と TTFT を見る。

## Decode

候補:

- tg128
- tg512
- tg1024

decode は以下の context depth でも測る。

- depth 0
- depth 4K
- depth 16K
- depth 32K
- depth 64K
- depth 128K
- 最大実用 context 付近

例:

```text
tg128 @ d=0
tg128 @ d=4096
tg128 @ d=16384
tg128 @ d=65536
```

これにより「empty context では速いが long context で急落する」モデルを分離する。

---

# 4. 必須メタデータ

各 benchmark 結果に最低限以下を保存する。

```text
model
quantization
model_file_size
backend / llama.cpp commit
GPU
GPU connection
CPU
RAM size
CUDA version
context size
context depth
prompt tokens
generated tokens
batch size (-b)
ubatch size (-ub)
KV cache type
flash-attn on/off
MTP on/off
speculative decoding on/off
draft model / draft method
MoE mode
n-cpu-moe / cmoe / ncmoe 等
VRAM peak
RAM peak
prefill tok/s
decode tok/s
TTFT
wall time
errors / crash / OOM
```

`tok/s` を表示する場合、条件を併記しない結果は原則比較対象外にする。

---

# 5. KV cache 比較

P40 長文脈ベンチで以前検討していた比較を継続する。

対象:

- f16
- q8_0
- q4_0

測定項目:

- VRAM 使用量
- usable context length
- prefill
- decode
- context depth による劣化
- 生成品質に明らかな破綻がないか

特に 24GB / 20GB GPU では、KV cache quantization が実用 context 上限に直結する。

---

# 6. batch / ubatch tuning

最近の Qwen3.8-Flash-Next 実測では、

```text
-b 4096 -ub 4096
```

により prefill が大幅改善した報告がある。

したがって `llama-bench-tuner` では固定値だけでなく、
モデル / GPU ごとの batch sweep を持たせたい。

候補:

```text
-b 512   -ub 512
-b 1024  -ub 1024
-b 2048  -ub 2048
-b 4096  -ub 4096
```

必要なら asymmetric も試す。

目的:

- Prefill 最大化
- VRAM OOM 境界確認
- TTFT 改善
- decode への副作用確認

---

# 7. MTP / speculative decoding を独立軸へ

今後は MTP / speculative decoding を「おまけ」と扱わない。

同一モデルで必ず、

```text
baseline
MTP
fixed speculation
adaptive speculation
```

を分けて計測できるようにする。

重要:

モデルによって最適方式が異なる。

過去の観測では Ornith 35B-A3B 系で fixed MTP が強い例、
Qwen 系で adaptive speculation が大きく効く例がある。

coding workload では既存コード / ファイル名 / shell command の再利用が多いため、
一般会話より speculative hit rate が高くなる可能性がある。

したがって micro benchmark と coding benchmark の双方で測る。

---

# 8. MoE offload / hybrid benchmark

今後の最重要項目。

従来:

> モデルが VRAM に入るかどうか

だけを見るのではなく、

> GPU VRAM + RAM + PCIe + CPU + backend

全体を見る。

比較対象:

```text
full GPU
GPU expert cache
CPU MoE
CPU/GPU hybrid
partial expert offload
```

llama.cpp 系で利用できる場合:

- `-cmoe`
- `-ncmoe N`

FreeToken / EXL3 等では相当する mode を記録。

測定:

- decode tok/s
- prefill tok/s
- RAM 使用量
- VRAM 使用量
- PCIe 接続差
- CPU utilization
- wall time

特に x1ai では、

```text
P40 GPU0 = OCuLink
P40 GPU1 = USB4
```

なので、同一モデル・同一設定で比較する価値がある。

---

# 9. 現在の重点モデル

## A. Qwen3.6-35B-A3B

役割:

- 35B-A3B MoE baseline
- P40 24GB の sweet spot 候補
- llama.cpp / Ollama / FreeToken 比較基準

対象 quant:

- Q4_K_M
- Q3_K_XL / Q3 系
- MTP 対応版があれば別測定

見る項目:

- P40 1枚での decode
- MTP 有無
- context depth
- OCuLink / USB4 差
- dual GPU の必要性

---

## B. Ornith-1.5-35B-A3B

優先度高。

候補:

- GGUF
- EXL3 2.75bpw
- NVFP4 / FreeToken 対応版

EXL3 2.75bpw は約17.5GBで、
RTX 4000 Ada 20GBに非常に適合する。

比較したいもの:

```text
llama.cpp + GGUF + MTP
ExLlamaV3 + EXL3 + CPU/GPU hybrid
FreeToken + NVFP4
```

同一ベースモデルで backend / quantization / offload 戦略の違いを見る。

---

## C. Granite 4.2 30B

dense control として追加候補。

- 30B dense
- agent / SWE / terminal 系強化
- IBM 公式 GGUF がある
- Q4_K_M 約17.7GB級

役割:

> 35B-A3B MoE と 30B dense の「実仕事完遂率」比較。

tok/s では不利でも、
agent loop の成功率が高い可能性がある。

---

# 10. 無理があっても試す大型モデル

重要:

> RAM / VRAM 的に厳しくても、起動可能性があるなら Qwen Flash / GLM Flash を試験対象から外さない。

成功 / 失敗自体をデータとして残す。

---

## A. Qwen3.8-Flash-Next

最優先大型試験候補。

主要特徴:

- main model 約125B
- N-gram embedding 約51B
- active 約6B/token
- native long context
- MoE
- host-memory N-gram residency を想定
- MTP / speculative decoding
- QSA / GDN 等の新構造

既に llama.cpp の実験 branch で、

- 24GB RTX 4090
- 約110GB DDR4 RAM
- UD-Q4_K_XL
- 80K～250K context
- full CPU MoE / hybrid offload

で約20～22 tok/s decode の報告がある。

重要な試験:

### Precision 3680

```text
RTX 4000 Ada 20GB
RAM 96GB
```

まず context を抑える。

候補:

```text
-c 32768
-c 65536
-c 131072
```

`-cmoe` を優先して VRAM を attention / KV 用に残す。

batch:

```text
-b 1024 -ub 1024
-b 2048 -ub 2048
-b 4096 -ub 4096
```

計測:

- 起動可否
- RAM peak
- VRAM peak
- prefill
- decode
- TTFT
- context 最大値
- OOM / swap 発生
- server stability

可能なら N-gram mmap / NVMe residency も後で評価する。

### x1ai P40

かなり無理があるが、将来対応 backend で試験価値はある。

まず Precision を優先。

---

## B. GLM-5.3-Flash

試験対象に追加。

主要特徴:

- 約320B total
- 約18B active
- coding / tool / terminal agent が強い
- Qwen Flash-Next より大幅に重い

現行ハードでは実用性が低い可能性が高いが、

> 「動かないと決めつけず、最低bit GGUF / CPU offload / split で起動可能性を確認する」

方針。

確認したいもの:

- llama.cpp 対応状況
- GGUF quant の種類
- Q2 / IQ 系最低bit
- required RAM
- CPU offload
- P40 x2 split
- Precision 96GB での最低構成

起動できなかった場合も、

```text
required RAM
OOM point
load time
VRAM usage
unsupported op
backend limitation
```

を保存する。

目的は「現在のハードでどこが壁なのか」を定量化すること。

---

# 11. gpt-oss-120b

巨大 MoE control として維持。

用途:

- FreeToken / P40 技術実証
- llama.cpp / FreeToken backend 比較
- RAM offload の確認

P40 + FreeToken で約4.7 tok/s の実動報告があるが、
日常 coding worker の速度基準としては遅い。

35B-A3B と比較して、
「巨大モデルにする価値がどこにあるか」を見る。

---

# 12. x1ai P40向け重点試験

FreeToken Pascal 対応が main に入るまでは、
llama.cpp / Ollama を安定系とする。

FreeToken 正式対応後:

1. Qwen3.6-35B-A3B
2. Ornith-1.5-35B-A3B
3. KAT-Coder-V2.5
4. gpt-oss-120b

目標:

```text
10 tok/s      = 最低ライン
15-20 tok/s   = coding worker として実用候補
20 tok/s超    = 積極評価
```

ただし tok/s だけで採否を決めない。

---

# 13. 1 GPU = 1 worker も benchmark する

x1ai の P40 x2 は非対称接続。

Tensor Parallel だけでなく、

```text
P40 #0 → worker A
P40 #1 → worker B
```

で independent requests を走らせる。

測定:

- aggregate tok/s
- per-worker tok/s
- latency
- RAM contention
- CPU contention
- PCIe contention

Coding Agent の「複数 issue 並列処理」では TP より有利な可能性がある。

---

# 14. Coding Agent benchmark を追加

HumanEval 的な単発コード生成だけでは評価しない。

固定 repo / fixed issue を使用して、

```text
issue理解
  ↓
repo探索
  ↓
ファイル編集
  ↓
test
  ↓
failure解析
  ↓
修正
  ↓
再test
  ↓
commit-ready diff
```

まで実行する。

固定条件:

- 同じ repo
- 同じ issue
- 同じ tool set
- 同じ max turns
- 同じ context budget
- 同じ test command

記録:

- task success
- test pass
- tool-call error
- infinite loop
- cloud escalation
- total tokens
- wall time
- prefill
- TTFT
- decode
- final diff quality

最重要:

> 「一番速いモデル」ではなく「実際に仕事を終わらせるモデル」を選ぶ。

---

# 15. micro benchmark と agent benchmark を混ぜない

結果表示を最低でも2系統に分ける。

## Engine Benchmark

例:

```text
pp512
pp4096
pp16384
tg128@d0
tg128@d16k
tg128@d64k
```

目的:

- llama.cpp 更新による性能差
- CUDA kernel 差
- quant 差
- GPU差

## Workload Benchmark

例:

```text
RepoFix-01
RepoFix-02
ShellDebug-01
LongContextRepo-01
```

目的:

- coding agent 成功率
- MTP / speculation の実効利益
- long context での実用速度
- total wall time

---

# 16. 結果フォーマット案

CSV / JSONL では以下のような列を持たせたい。

```text
timestamp
host
gpu
gpu_connection
llama_cpp_commit
model
quant
backend
context
depth
prompt_tokens
generated_tokens
batch
ubatch
kv_type
mtp
speculation
moe_mode
pp_tps
tg_tps
ttft_ms
wall_time_s
vram_peak_mb
ram_peak_mb
success
error
notes
```

agent benchmark では追加:

```text
repo
issue
tool_calls
tests_run
tests_passed
loop_count
cloud_escalation
diff_lines
final_status
```

---

# 17. llama.cpp 更新追従

`llama-bench-tuner` は llama.cpp の特定バージョン前提にしない。

最低限、

```text
llama.cpp commit hash
build options
CUDA version
CMAKE_CUDA_ARCHITECTURES
```

を記録する。

新しいモデル対応が PR branch にしかない場合は、

```text
upstream main
PR branch
```

を区別する。

Qwen3.8-Flash-Next のような experimental branch も試験可能にする。

---

# 18. 今後の優先実装順

Codex には以下の順で改修してほしい。

## Phase 1

- 現状コード確認
- llama.cpp CLI option の現行仕様追従
- pp / tg / depth 分離
- commit / environment metadata 保存
- CSV / JSONL schema 更新

## Phase 2

- batch / ubatch sweep
- KV cache sweep
- MTP / speculation flag
- MoE offload mode
- VRAM / RAM monitor

## Phase 3

- Qwen3.6 / Ornith / Granite 4.2 model profiles
- Qwen3.8-Flash-Next experimental profile
- GLM-5.3-Flash experimental / failure-profile

## Phase 4

- x1ai dual-worker aggregate benchmark
- Coding Agent workload benchmark
- result comparison / regression detection

---

# 19. モデル優先順位

現時点:

## 実用比較

1. Qwen3.6-35B-A3B
2. Ornith-1.5-35B-A3B
3. Granite 4.2 30B
4. KAT-Coder-V2.5

## 大型・無理やり試験

1. Qwen3.8-Flash-Next
2. GLM-5.3-Flash
3. gpt-oss-120b

Qwen3.8-Flash-Next は実用域まで到達する可能性があるため、
「無理やり試験」から通常評価へ昇格する可能性が高い。

GLM-5.3-Flash は RAM 容量上かなり厳しくても、
最低bit量子化 / CPU offload / split でどこまで行けるか確認する。

---

# 20. Codexへの最終要求

既存の `llama-bench-tuner` の思想や動作を壊さず、
現在の llama.cpp と新しい MoE / long-context モデルへ拡張する。

特に次の点を優先する。

1. `tok/s` を条件なしで比較しない
2. Prefill / Decode / TTFT を分離
3. context depth を測る
4. MTP / speculation を別条件として測る
5. MoE offload / hybrid を比較可能にする
6. GPU VRAM だけでなく RAM / PCIe / CPU を記録する
7. P40 の OCuLink / USB4 差を残す
8. 1 GPU = 1 worker の aggregate throughput を測る
9. Qwen3.8-Flash-Next を多少無理でも試す
10. GLM-5.3-Flash も「無理」と決めつけず起動可能性を試す
11. micro benchmark と real coding workload を分ける
12. 最終評価は「仕事を完遂できるか」で行う

以上を前提に、まず既存 repository を調査し、
現在の実装との差分と必要変更点を整理してから改修を開始してほしい。
