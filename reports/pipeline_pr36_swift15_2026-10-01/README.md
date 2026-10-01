# Sample pipeline run — pr36 / RTX 4000 Ada / Swift-1.5-Qwen3.8-27B IQ4_XS (2026-10-01)

Produced by **one** `llama-tune-pipeline all` invocation (no `--resume`, no manual intervention) with
stock llama.cpp `0b5be7e4a` on WSL2. The runtime code was this branch at commit `0d9e51b`
(before the `--retry-aborts` feature), so the one intermittent `CUDA error: unknown error` abort in the
Optuna d32768 study shows up as a `runtime_abort` row in `failure_summary.json` instead of being retried.

```bash
BENCH_HOST=pr36-wsl BENCH_GPU="RTX 4000 Ada 20GB" BENCH_GPU_CONNECTION="internal PCIe (WSL2)" \
uv run llama-tune-pipeline all \
  --llama-bench ~/src/llama.cpp/build/bin/llama-bench \
  --model ~/models/Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf --name pr36-swift15-clean \
  --gpu-index 0 --min-free-vram-gib 17 --per-run-timeout 900 \
  --depths 0,4k,8k,16k,32k,64k --kv f16,q8_0,q4_0 \
  --grid-space grid_space.json --grid-depths 8k,32k --optuna-depths 32k,64k --trials 8 --seed 0 \
  --validate-depths 8k,32k,64k --validate-reps 3 --top-k 4 --fast-depth 8k --balanced-depth 32k
# grid_space.json: {"batch":[1024,2048,4096],"ubatch":[256,512,1024],"flash_attn":[0,1]}
```

Wall clock ≈ 1.9 h of benchmark time summed from the stage outputs (capacity 18 runs ≈ 11 min, grid 72 runs ≈ 41 min,
optuna 2x8 trials ≈ 23 min of measured wall time, validation 48 runs ≈ 38 min); the shell-level `time` of the whole invocation was not recorded.

| file | content |
|---|---|
| `capacity.json/.csv` | per-KV capacity vs practical-candidate boundary, status counts, per-depth Pareto |
| `grid.json` | axis effects, failure boundaries (VRAM cliff), promising set, narrowed `next_space` |
| `optuna/d*.json` | one multi-objective study per depth (tg↑ pp↑ TTFT↓ VRAM↓ wall↓), Pareto trials |
| `validation.json` | top-4 candidates x depth x 3 repeats: median/IQR/CV, stability, practical flag |
| `profiles/*.json` | FAST / BALANCED / LONG-CONTEXT with the exact `llama-server` command |
| `pareto/pareto_d*.json/.csv` | per-depth non-dominated set over everything measured |
| `failure_summary.json` | counts and examples of every non-success status |

Raw per-run CSVs / stderr logs and the Optuna SQLite DB are not included. TTFT is an estimate from pp
(`ttft_kind=estimated_from_pp`). Throughput is `llama-bench` at KV depth, not an end-to-end server run.
