"""Wait for / skip busy GPUs. Never signals, stops or otherwise touches other processes."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .telemetry import GpuBackend


@dataclass
class GateResult:
    ok: bool
    gpu_index: Optional[int]
    free_mib: Optional[int]
    waited_s: float
    reason: str
    active_processes: list[str] = field(default_factory=list)


def _candidates(backend: GpuBackend, gpu_index: Optional[int]) -> list[int]:
    if gpu_index is not None:
        return [gpu_index]
    return [gpu.index for gpu in backend.list_gpus()]


def check_gpu(backend: Optional[GpuBackend], *, min_free_mib: Optional[int],
              gpu_index: Optional[int] = None) -> GateResult:
    """One-shot check. With ``gpu_index=None`` the GPU with the most free memory that
    satisfies the threshold is chosen (the choice is reported, never assumed)."""

    if backend is None:
        return GateResult(True, gpu_index, None, 0.0, "no_gpu_backend: cannot gate")
    if not min_free_mib:
        return GateResult(True, gpu_index, None, 0.0, "no_threshold")
    best: tuple[int, int] | None = None
    seen: dict[int, Optional[int]] = {}
    for index in _candidates(backend, gpu_index):
        free = backend.free_mib(index)
        seen[index] = free
        if free is not None and free >= min_free_mib and (best is None or free > best[1]):
            best = (index, free)
    if best is not None:
        return GateResult(True, best[0], best[1], 0.0, "free_vram_ok")
    # Record who is occupying the GPU(s) for diagnosis; this is read-only.
    procs: list[str] = []
    for index in seen:
        procs.extend(f"gpu{index}: {p}" for p in backend.compute_processes(index))
    detail = ", ".join(f"gpu{i}={'?' if f is None else f}MiB" for i, f in seen.items())
    return GateResult(False, gpu_index, max((f for f in seen.values() if f is not None), default=None),
                      0.0, f"free VRAM below {min_free_mib} MiB ({detail})", procs)


def wait_for_gpu(backend: Optional[GpuBackend], *, min_free_mib: Optional[int],
                 gpu_index: Optional[int] = None, wait: bool = False, timeout_s: float = 0.0,
                 poll_s: float = 60.0, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> GateResult:
    """Check once; if busy and ``wait`` is set, poll until free or ``timeout_s`` elapses."""

    start = clock()
    result = check_gpu(backend, min_free_mib=min_free_mib, gpu_index=gpu_index)
    while not result.ok and wait and (clock() - start) < timeout_s:
        sleep(min(poll_s, max(0.0, timeout_s - (clock() - start))))
        result = check_gpu(backend, min_free_mib=min_free_mib, gpu_index=gpu_index)
    result.waited_s = clock() - start
    if not result.ok and wait:
        result.reason += f"; waited {result.waited_s:.0f}s (timeout {timeout_s:.0f}s)"
    return result
