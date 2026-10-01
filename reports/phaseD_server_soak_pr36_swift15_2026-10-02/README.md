# Phase D sample — `server` + `soak` stages on pr36 (RTX 4000 Ada, WSL2), 2026-10-02

Input: the profiles of `reports/pipeline_pr36_swift15_2026-10-01/` (Swift-1.5-Qwen3.8-27B IQ4_XS, stock llama.cpp
`b10703-0b5be7e4a`). Commands:

```bash
llama-tune-pipeline server --name pr36-swift15-phaseD ... --server-sizes 4k,16k --server-reps 2 --spec-compare
llama-tune-pipeline soak   --name pr36-swift15-phaseD ... --soak-profile balanced --soak-minutes 5 --soak-window 20
```

* `server_results.json` — measured TTFT / pp / tg per prompt size (unique prompts, `cache_prompt:false`) and the
  `--spec-type draft-mtp` comparison. Result: the plain ukisai GGUF already carries an MTP head; tg ≈ 1.9x
  (19.2 → 36.5 tok/s at 4K, 18.4 → 35.6 at 16K), draft acceptance ≈ 57 %, pp −6…−9 %, TTFT +5…+6 %.
  **The measurement prompt asks the model to count (very predictable), so acceptance/speed-up are an upper-bound style
  figure for this synthetic workload, not a promise for chat or coding traffic.**
* `soak_balanced.json` — 327 s of continuous decode: tg 18.6 → 18.6 tok/s, max 79 °C, SM clock ≥ 1875 MHz, throttle reason
  `sw_power_cap` only (not thermal) → verdict `pass`. Five minutes is a smoke test; use `--soak-minutes 15..20` for a thermal claim.
* `profiles/*.json` — the original profiles annotated with `measured_server`, `mtp_speculation`, `soak_server` (and
  `commands.llama_server_mtp` where every size gained >= 10 %).

Note: clock/throttle aggregation in this run used the version before the "ignore idle samples" fix; the soak windows
(utilisation-bound) are unaffected, the run-level `sm_clock_min_mhz` is.
