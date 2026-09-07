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

    Existing runners use a parsed positive decode metric as their success
    contract. Preserve that contract first; status enriches failed cases only.
    """

    if skip_reason:
        return BenchOutcome(BenchStatus.SKIPPED, False, skip_reason)
    if timed_out:
        return BenchOutcome(BenchStatus.TIMEOUT, False, "llama-bench exceeded timeout")
    if (decode_tps or 0.0) > 0.0:
        return BenchOutcome(BenchStatus.SUCCESS, True, "")

    material = f"{stdout}\n{stderr}".lower()
    if any(marker in material for marker in _OOM_MARKERS):
        return BenchOutcome(BenchStatus.OOM, False, "llama-bench reported out of memory")
    if any(marker in material for marker in _UNSUPPORTED_MARKERS):
        return BenchOutcome(BenchStatus.UNSUPPORTED, False, "llama-bench reported unsupported configuration")
    if returncode not in (None, 0):
        return BenchOutcome(BenchStatus.FAILED, False, f"llama-bench exited with code {returncode}")
    return BenchOutcome(BenchStatus.FAILED, False, "decode metric was not parsed")
