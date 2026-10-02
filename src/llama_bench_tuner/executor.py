"""Run one benchmark subprocess with a timeout and resource monitoring."""

from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Optional, Sequence

from .telemetry import GpuBackend, PeakMonitor, Peaks


@dataclass
class ExecResult:
    returncode: Optional[int]
    stdout: str
    stderr: str
    timed_out: bool
    elapsed_s: float
    start_iso: str
    end_iso: str
    peaks: Peaks


def _text(value: object) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def gpu_env(backend: Optional[GpuBackend], gpu_index: Optional[int]) -> dict[str, str]:
    """Environment restricting the child to one device. Only set for a known backend."""

    env = dict(os.environ)
    if gpu_index is not None and backend is not None:
        key = "CUDA_VISIBLE_DEVICES" if backend.name == "nvidia-smi" else "HIP_VISIBLE_DEVICES"
        env[key] = str(gpu_index)
    return env


def _kill_tree(proc: "subprocess.Popen[str]") -> None:
    """Kill the benchmark and anything it spawned (its own process group on POSIX).

    Killing only the direct child can leave a grandchild holding the stdout pipe open,
    which would make the follow-up ``communicate()`` block far past the timeout."""

    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)
        else:  # pragma: no cover - non-POSIX
            proc.kill()
    except (ProcessLookupError, PermissionError):
        proc.kill()


def run_monitored(cmd: Sequence[str], *, timeout: Optional[float], backend: Optional[GpuBackend],
                  gpu_index: Optional[int], env: Optional[Mapping[str, str]] = None,
                  poll_interval: float = 1.0) -> ExecResult:
    start = datetime.now(timezone.utc)
    proc = subprocess.Popen(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            env=dict(env) if env is not None else None,
                            start_new_session=hasattr(os, "killpg"))
    monitor = PeakMonitor(backend, gpu_index, proc.pid, poll_interval)
    monitor.start()
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        _kill_tree(proc)
        stdout, stderr = proc.communicate()
        stdout = _text(exc.stdout) or _text(stdout)
        stderr = (_text(exc.stderr) or _text(stderr)) + "\n[timeout] benchmark exceeded timeout"
    except KeyboardInterrupt:
        _kill_tree(proc)
        proc.communicate()
        raise
    finally:
        peaks = monitor.stop()
    end = datetime.now(timezone.utc)
    return ExecResult(
        returncode=None if timed_out else proc.returncode,
        stdout=_text(stdout), stderr=_text(stderr), timed_out=timed_out,
        elapsed_s=(end - start).total_seconds(),
        start_iso=start.isoformat(timespec="seconds"), end_iso=end.isoformat(timespec="seconds"),
        peaks=peaks,
    )
