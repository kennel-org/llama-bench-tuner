# llama-bench-tuner Extension Plan

Date: 2026-08-30

## Purpose

Extend `llama-bench-tuner` for engine benchmarks, capability probes, parsers, telemetry/result recording, and mechanical MTP-spec benchmarks. P40 candidate inventory and experiment policy belong to `p40-llm-lab`.

Operational notes remain in:

`/home/masaya/projects/kennel-system-laboratory/docs/ops/2026-08-30_llama-bench-tuner_latest-local-llm-evaluation-handover.md`

## Repository boundaries

| Repository | Responsibility |
| --- | --- |
| `llama-bench-tuner` | Engine benchmark, capability probe, parser, telemetry/result schema, mechanical MTP-spec benchmark, Grid/Optuna, visualization |
| `p40-llm-lab` | P40 candidate inventory, KEEP/No-Go policy, context/KV/MTP/MoE experiment policy |
| `llm-warroom` | Coding-agent workload harnesses (#52/#53) |
| `kennel-system-laboratory` | Host state, operational records, changes, and links to results |
| `llama.cpp` / `llama.cpp-laurent` | Backend implementation, experimental changes, and builds |

Model weights, secrets, and large raw logs remain outside the Git-tracked core repositories.

## Current state

- `tune.py` mainly searches `ngl`, `batch`, and `flash-attn`.
- `optuna_tune.py` searches the same basic axes.
- Current output records mainly prefill/decode tok/s and tuning parameters.
- A common schema for TTFT, context depth, KV cache, MTP/speculation, MoE offload, and GPU/RAM peaks is not yet implemented.
- P40 candidate selection and experimental policy are external to this repository.
- `kennel-system-laboratory` is the operational record repository, not the implementation repository.

## Implementation status

- Phase 1 provides append-only common fields, a row-preserving parser, command builder, status taxonomy, and a read-only capability probe.
- Existing Grid checkpoint/resume and metadata experiments are retained; no legacy output columns or visualization inputs are replaced.
- pp/tg/depth, KV, MTP/speculation, MoE offload, and GPU telemetry are not implemented yet.

## Phased implementation

### Phase 1: Compatibility and capability foundation

Candidate files:

- `src/llama_bench_tuner/schema.py`
- `src/llama_bench_tuner/parsing.py`
- `src/llama_bench_tuner/command_builder.py`
- `src/llama_bench_tuner/status.py`
- `src/llama_bench_tuner/capabilities.py`
- `src/llama_bench_tuner/tune.py`
- `src/llama_bench_tuner/optuna_tune.py`

Deliverables:

- Define append-only fields with requested offload/placement distinct from runtime-observed values.
- Preserve existing CSV columns and CLI arguments; add new fields compatibly.
- Preserve each native llama-bench CSV row while retaining the legacy pp/tg aggregate.
- Keep Grid and Optuna command lines and visualization inputs compatible.
- Probe only options evidenced by `llama-bench --help`; classify success, failure, OOM, unsupported, timeout, and skip outcomes.

### Phase 2: Benchmark-axis expansion

- Load model profiles from external schema-validated files; do not embed P40 candidate inventory or KEEP/No-Go policy here.
- Represent pp/tg as independent cases.
- Represent context-depth sweeps in configuration files.
- Support batch/ubatch, KV cache, flash attention, MTP/speculation, and MoE offload as optional axes.
- Do not guess unsupported CLI options; use capability checks and record explicit skips.
- Preserve existing Grid and Optuna output and reject comparisons without their conditions.

### Phase 3: Engine telemetry integration

Record engine telemetry for benchmarks here. `p40-llm-lab` owns candidate selection and experiment policy:

- Separate GPU0 OCuLink and GPU1 USB4 results.
- Time-series GPU temperature, core/memory clocks, power, utilization, and throttle reason.
- 10–20 minute continuous decode with time-point tok/s.
- Prefer one worker per GPU; use dual-GPU split only when required.

Align schema semantics without creating a direct Python package dependency.

### Out of scope: Coding-agent workload

Fixed-repository, fixed-issue, fixed-tool-set, and fixed-test-command agent evaluation belongs to `llm-warroom` #52/#53 and `p40-llm-lab` #9. Do not implement it in this repository.

## Acceptance criteria

- Existing `llama-tune`, `llama-tune-optuna`, and visualization commands still work with their current arguments.
- Existing sample CSVs and plots remain usable.
- Every new run links conditions, environment, results, errors, and elapsed time.
- Jobs expected to exceed five minutes print progress and use one checkpoint file.
- Hardware changes and service restarts have dedicated procedures and rollback commands.

## Order of work

1. Implement Phase 1 compatibility/capability foundation and run regression checks.
2. Add pp/tg/depth case representation and runner support.
3. Add capability-aware KV, MTP/speculation, and MoE support.
4. Align result-field semantics with `p40-llm-lab` without importing its inventory or policy.
5. Coordinate only the external coding-agent harness interface, if needed.
