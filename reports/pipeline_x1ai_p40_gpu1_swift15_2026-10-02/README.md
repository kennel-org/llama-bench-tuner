# x1ai / Tesla P40 (GPU1, USB4 eGPU dock) — Swift-1.5-Qwen3.8-27B IQ4_XS, 2026-10-02

llama.cpp `b11146-7fe450e19` (v0.5.0, CUDA 12.8, sm_61), driver 580.159.03, one P40 (23040 MiB), GPU1 only.
Produced by `llama-tune-pipeline` (stages capacity, grid, optuna, validate, profile, server, soak). **Not a clean single
invocation** — read the deviations:

* The plan was cut for the P40 (one 128K run takes ~21 min): Grid only at 8K over `f16,q8_0` with
  `batch {1024,2048} x ubatch {256,512} x FA {0,1}`; one Optuna study (32K, 6 trials); validation top-2 candidates at
  8K/32K/64K/128K with 3 repeats (**2 repeats at 128K**, `--validate-reps-deep 2`).
* **Practical criteria were relaxed for this slow card and are recorded in each profile** (`evidence.criteria`):
  `--min-tg 6 --pp-ratio 0.2` instead of 10 / 0.3. Under the default criteria q8_0 stops being practical at 64K
  (tg 9.9 < 10) and the LONG profile would be 32K.
* The pipeline was restarted twice with `--resume` (reduced plan; telemetry idle-sample fix). One Grid row
  (`f16 8K b1024/ub256 FA0`) and the first capacity row (`f16 d0`) were measured before that fix, so their
  `sm_clock_min_mhz` (544 MHz) is an idle reading, not a slowdown.
* GPU0 (OCuLink) could not be measured: it reported an **uncorrectable ECC error** (double-bit, device memory) right after
  the first run and became unavailable to CUDA (see the issue comments). `outfile/pipeline/x1ai-p40-gpu0-swift15.INVALID-ecc-fault`
  was quarantined; its `oom` rows are the ECC failure misclassified by the pre-fix classifier.
* The capacity sweep ran with the earlier code (before `--retry-aborts`/fault handling), single-process-per-point as designed.

Highlights: f16 KV is fastest on the P40 (q4_0 KV is much slower: tg 8.75 vs 11.3 tok/s at 32K — no fast quantised-KV
flash-attention path on Pascal); f16 128K OOMs, q8_0 128K completes (19.4 GB) but needs ~21 min per run; MTP gives 1.5–1.65x;
a 15-minute soak showed no thermal throttling (tg 11.93 → 12.03 tok/s, <= 63 °C, SM clock held at 1531 MHz).
