"""Classify llama-bench outcomes without losing the legacy success signal."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BenchStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    OOM = "oom"
    UNSUPPORTED = "unsupported"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"
    # Added for the capacity/pipeline workflow (append-only; legacy runners never
    # need to produce these, and `ok` semantics are unchanged).
    RUNTIME_ABORT = "runtime_abort"
    SLOWDOWN = "slowdown"
    GPU_BUSY = "gpu_busy"
    SKIPPED_AFTER_FAIL = "skipped_after_fail"


@dataclass(frozen=True)
class BenchOutcome:
    status: BenchStatus
    ok: bool
    error: str


_OOM_MARKERS = (
    "out of memory",
    "cuda error: out of memory",
    "cuda_error_out_of_memory",
    "failed to allocate",
    "cannot allocate memory",
    # llama-bench without -v swallows ggml's "CUDA error: out of memory" text; the backtrace
    # still names the pool allocator that failed (mangled and demangled spellings).
    "ggml_cuda_pool_vmm5alloc",
    "ggml_cuda_pool_leg5alloc",
    "ggml_cuda_pool_vmm::alloc",
    "ggml_cuda_pool_leg::alloc",
)
_UNSUPPORTED_MARKERS = (
    "unknown model architecture",
    "unsupported",
    "not supported",
    "invalid parameter for argument",
    "unrecognized option",
)
# Backend runtime faults that are *not* memory exhaustion: HIP/ROCm hardware
# exceptions, generic ggml backend aborts, driver faults. They are reported
# separately so a ROCm abort is never mistaken for an OOM boundary.
_RUNTIME_ABORT_MARKERS = (
    "hw exception",
    "rocm error",
    "hip error",
    "hiperror",
    "cuda error",
    "ggml_abort",
    "ggml_cuda_error",
    "core dumped",
    "segmentation fault",
    "illegal memory access",
    "device-side assert",
)
_ABORT_RETURN_CODES = (134, -6, 139, -11)

# A GPU hardware fault (uncorrectable ECC, GPU fell off the bus) makes every later CUDA call fail,
# often with allocator-looking messages ("cudaMalloc failed: uncorrectable ECC error encountered").
# It must be checked before the OOM markers and is never retried.
HARDWARE_FAULT_PREFIX = "GPU hardware fault"
_HARDWARE_FAULT_MARKERS = (
    "uncorrectable ecc",
    "ecc error encountered",
    "fallen off the bus",
    "gpu is lost",
)

# With ``-v`` llama.cpp prints its whole load log; benign lines in it can contain words such as
# "unsupported". Only the end of stderr (where a fatal error and its backtrace land) is searched.
_STDERR_TAIL_LINES = 120

# Statuses that mean "this configuration failed to run" (as opposed to unsupported/skipped).
FAILURE_STATUSES = frozenset({"oom", "runtime_abort", "timeout", "failed"})


def _tail(text: str, lines: int = _STDERR_TAIL_LINES) -> str:
    return "\n".join(text.splitlines()[-lines:])


def classify_bench_outcome(
    *,
    returncode: int | None,
    decode_tps: float | None,
    stdout: str = "",
    stderr: str = "",
    timed_out: bool = False,
    skip_reason: str = "",
) -> BenchOutcome:
    """Classify one completed or skipped execution.

    ``ok`` preserves the legacy success contract used by existing Grid/Optuna
    consumers (a parsed positive decode metric), unaffected by ``returncode``.
    ``status`` is the stricter canonical taxonomy: a non-zero exit always
    yields ``failed``/``oom``/``unsupported`` even when a decode metric was
    still parsed, so a crashed-but-partially-printed run is never reported as
    ``success``.
    """

    ok = (decode_tps or 0.0) > 0.0
    # stdout is the CSV table (no diagnostics); diagnostics live in the tail of stderr.
    material = f"{_tail(stdout, 20)}\n{_tail(stderr)}".lower()

    if skip_reason:
        return BenchOutcome(BenchStatus.SKIPPED, False, skip_reason)
    if timed_out:
        return BenchOutcome(BenchStatus.TIMEOUT, False, "llama-bench exceeded timeout")
    if any(m in material for m in _HARDWARE_FAULT_MARKERS):
        return BenchOutcome(BenchStatus.RUNTIME_ABORT, False,
                            f"{HARDWARE_FAULT_PREFIX} (ECC/bus error reported by the CUDA driver); "
                            "the GPU needs a reset before it can be benchmarked again")

    if returncode not in (None, 0):
        if any(marker in material for marker in _OOM_MARKERS):
            return BenchOutcome(BenchStatus.OOM, ok, "llama-bench reported out of memory")
        if any(marker in material for marker in _UNSUPPORTED_MARKERS):
            return BenchOutcome(BenchStatus.UNSUPPORTED, ok, "llama-bench reported unsupported configuration")
        if returncode in _ABORT_RETURN_CODES or any(m in material for m in _RUNTIME_ABORT_MARKERS):
            return BenchOutcome(
                BenchStatus.RUNTIME_ABORT, ok,
                f"llama-bench aborted at runtime (exit code {returncode}); not classified as OOM",
            )
        return BenchOutcome(BenchStatus.FAILED, ok, f"llama-bench exited with code {returncode}")

    if ok:
        return BenchOutcome(BenchStatus.SUCCESS, True, "")
    if any(marker in material for marker in _OOM_MARKERS):
        return BenchOutcome(BenchStatus.OOM, False, "llama-bench reported out of memory")
    if any(marker in material for marker in _UNSUPPORTED_MARKERS):
        return BenchOutcome(BenchStatus.UNSUPPORTED, False, "llama-bench reported unsupported configuration")
    return BenchOutcome(BenchStatus.FAILED, False, "decode metric was not parsed")
