## llama-bench-tuner

Tools for scripting llama.cpp's `llama-bench` runs, exploring parameter grids or Optuna-driven searches, and generating quick visual summaries.

---

## Features

- **Grid tuning (`llama-tune`)** – iterate over `ngl`, `batch`, and `flash-attn` combinations, save each raw CSV/stdout, parse decode & prefill token-per-second metrics, and emit a timestamped summary table.
- **Optuna tuning (`llama-tune-optuna`)** – sample hyperparameters with Optuna, resume-able via SQLite storage, and capture top trial metadata plus the full trial ledger.
- **Visualization helpers** – convert summaries into ranked tables and plots suited for quick inspection of decode throughput.

---

## Requirements

- Python >= 3.10
- `llama-bench` binary from [llama.cpp](https://github.com/ggerganov/llama.cpp)
- GGUF model compatible with the chosen benchmark parameters
- Optional: GPU driver/toolchain needed by your llama.cpp build

We recommend installing dependencies via [uv](https://github.com/astral-sh/uv) (plain `pip` remains compatible, but instructions below assume uv).

---

## Setup

```bash
# install uv if you don't have it yet
curl -LsSf https://astral.sh/uv/install.sh | sh

# inside repo root
uv venv
source .venv/bin/activate
uv pip install -e .
```

The package installs three console scripts:

| Script | Entry point | Purpose |
| --- | --- | --- |
| `llama-tune` | `llama_bench_tuner.tune:main` | Grid search driver |
| `llama-tune-viz` | `llama_bench_tuner.viz:main` | Visualize grid summaries |
| `llama-tune-optuna` | `llama_bench_tuner.optuna_tune:main` | Optuna search driver |

All generated artifacts default to the `outfile/` directory (raw CSVs, summaries, plots) and `tmp/` for stderr logs; ensure those paths are writable.

---

## Usage

### 1. Grid tuning (`llama-tune`)

```bash
llama-tune \
  --llama-bench /path/to/llama-bench \
  --model /path/to/model.gguf \
  --threads 14 \
  --prompt 2048 \
  --ngen 256 \
  --ngl 16 20 24 28 \
  --batch 8 12 \
  --flash-attn 0 1
```

Example (from the GPT-OSS-120B debugging run):

```bash
llama-tune \
  --llama-bench /path/to/llama.cpp/build/bin/llama-bench \
  --model /path/to/llama.cpp/models/gpt-oss-120b/gpt-oss-120b-mxfp4-00001-of-00003.gguf \
  --threads 14 --prompt 2048 --ngen 256 \
  --ngl 16 20 24 28 \
  --batch 8 12 \
  --flash-attn 0 1
```

Typical GPT-OSS-120B search range
---------------------------------

For a single-node, 2×RTX 6000 Ada host used in our experiments, we sweep a slightly wider region to capture the memory/throughput trade-offs of 120B-sized models:

```bash
llama-tune \
  --llama-bench /path/to/llama.cpp/build/bin/llama-bench \
  --model /path/to/llama.cpp/models/gpt-oss-120b/gpt-oss-120b-mxfp4-00001-of-00003.gguf \
  --threads 14 --prompt 2048 --ngen 256 \
  --ngl 20 24 28 32 \
  --batch 6 8 10 12 \
  --flash-attn 0 1 \
  --ub-ratio 2.0 \
  --nkvo 0 \
  --split-mode layer
```

You can also externalize grid search spaces via `--space-file`. Example for running GPT-OSS-120B on a single RTX 4000 Ada (20 GB):

```jsonc
// infile/grid_space_rtx4000_gptoss120b.json
{
  "ngl": [12, 14, 16, 18, 20, 22, 24, 26, 28],
  "batch": [8, 12, 16],
  "flash_attn": [0, 1]
}
```

```bash
llama-tune \
  --llama-bench ... \
  --model ... \
  --threads 14 --prompt 2048 --ngen 256 \
  --space-file infile/grid_space_rtx4000_gptoss120b.json \
  --ub-ratio 2.0 --nkvo 0 --split-mode layer
```

The ranges above assume GPU offloading (e.g. `--split-mode layer`, `--nkvo 0`) to keep VRAM use below ~92 GB. Running fully on-GPU will consume more VRAM, so adjust `--ngl`/`--batch` accordingly. If you have extra headroom, extend `--ngl` to 34–36; if resources are tighter, narrow the batch list.

> Note: the GPT-OSS-120B MXFP4 model ships as three GGUF shards; point `--model` to the first shard (00001-of-00003) and keep the other files in the same directory so llama.cpp can stream them.


Outputs:

- `outfile/bench_*` – raw CSV dumps from llama-bench runs
- `tmp/bench_*.stderr.txt` – captured stderr per run (if non-empty)
- `outfile/summary_YYYYMMDD_HHMMSS.csv` – consolidated table with decode/prefill tok/s and flags

The CLI prints the best-performing configuration by decode tok/s (fallback to prefill tok/s for tie-breaking).

### 2. Optuna tuning (`llama-tune-optuna`)

```bash
llama-tune-optuna \
  --llama-bench /path/to/llama-bench \
  --model /path/to/model.gguf \
  --ngl-min 16 --ngl-max 28 \
  --batch-min 8 --batch-max 16 \
  --flash-attn 0 1 \
  --n-trials 30 \
  --storage sqlite:///outfile/optuna_study.db
```

Key artifacts:

- `outfile/optuna_best.json` – best trial value/params/metadata
- `outfile/optuna_trials.csv` – full trial history, including user attrs for later analysis

Resume tuning by reusing the same `--storage` and optionally `--study-name`.

### 3. Visualization (`llama-tune-viz` & `viz_optuna`)

Grid summaries:

```bash
llama-tune-viz --summary outfile/summary_20240101_120000.csv --outdir reports
```

Optuna summaries:

```bash
python -m llama_bench_tuner.viz_optuna \
  --trials outfile/optuna_trials.csv \
  --best outfile/optuna_best.json \
  --outdir reports
```

Both commands emit ranking CSVs and PNG plots (decode vs. `ngl`, trial progression, scatter maps) into the chosen output directory.

---

## Tips

- Use `--ub-ratio` to derive micro-batch (`ub`) automatically from batch size (`batch / ratio`).
- Enable/disable `--nkvo` or `--split-mode` to test offloading and weight split strategies.
- Configure `--timeout-per-trial` when runs can hang; failed trials are scored as zero, so Optuna will deprioritize them.
- Rich logging highlights each run and reports whether parse/metrics succeeded; inspect `tmp/` stderr logs when `ok=False`.

---

## Development

For linting or tests, extend this section as workflows grow. Contributions welcome via issues or PRs.

## License

Released under the [MIT License](./LICENSE). Copyright (c) 2025 [@kennel_org](https://x.com/kennel_org).
