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
)
_UNSUPPORTED_MARKERS = (
    "unknown model architecture",
    "unsupported",
    "not supported",
    "invalid parameter for argument",
    "unrecognized option",
)


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
    material = f"{stdout}\n{stderr}".lower()

    if skip_reason:
        return BenchOutcome(BenchStatus.SKIPPED, False, skip_reason)
    if timed_out:
        return BenchOutcome(BenchStatus.TIMEOUT, False, "llama-bench exceeded timeout")

    if returncode not in (None, 0):
        if any(marker in material for marker in _OOM_MARKERS):
            return BenchOutcome(BenchStatus.OOM, ok, "llama-bench reported out of memory")
        if any(marker in material for marker in _UNSUPPORTED_MARKERS):
            return BenchOutcome(BenchStatus.UNSUPPORTED, ok, "llama-bench reported unsupported configuration")
        return BenchOutcome(BenchStatus.FAILED, ok, f"llama-bench exited with code {returncode}")

    if ok:
        return BenchOutcome(BenchStatus.SUCCESS, True, "")
    if any(marker in material for marker in _OOM_MARKERS):
        return BenchOutcome(BenchStatus.OOM, False, "llama-bench reported out of memory")
    if any(marker in material for marker in _UNSUPPORTED_MARKERS):
        return BenchOutcome(BenchStatus.UNSUPPORTED, False, "llama-bench reported unsupported configuration")
    return BenchOutcome(BenchStatus.FAILED, False, "decode metric was not parsed")
