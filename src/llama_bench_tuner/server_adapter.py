"""Server-based measurements: real TTFT, speculative decoding (MTP) comparison, long soak.

``llama-bench`` has no TTFT and no speculation option, so these run the real ``llama-server`` from a
profile's exact command and talk to it over its OpenAI-compatible API. Results are labelled
``benchmark_kind=server`` / ``ttft_kind=measured``.

Prompts are unique per request (random words, unique prefix) and sent with ``cache_prompt: false``,
because a reused prefix would let the server skip prefill and understate TTFT.
"""

from __future__ import annotations

import json
import os
import random
import re
import shlex
import signal
import socket
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from .executor import gpu_env
from .stage_common import median_iqr
from .telemetry import GpuBackend, PeakMonitor, Peaks

_WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa "
          "quebec romeo sierra tango uniform victor whiskey xray yankee zulu").split()

# nvidia-smi clocks_event_reasons bits that mean the GPU was slowed down for protection.
THERMAL_THROTTLE_BITS = 0x8 | 0x20 | 0x40 | 0x80  # hw slowdown, sw thermal, hw thermal, hw power brake
THROTTLE_NAMES = {0x1: "gpu_idle", 0x2: "applications_clocks_setting", 0x4: "sw_power_cap", 0x8: "hw_slowdown",
                  0x10: "sync_boost", 0x20: "sw_thermal_slowdown", 0x40: "hw_thermal_slowdown",
                  0x80: "hw_power_brake_slowdown"}


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def decode_throttle(mask_hex: Optional[str]) -> list[str]:
    if not mask_hex:
        return []
    mask = int(mask_hex, 16)
    return [name for bit, name in THROTTLE_NAMES.items() if mask & bit]


def _http(url: str, body: Optional[dict] = None, timeout: float = 30.0) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


@dataclass
class ServerSession:
    """A running llama-server (one process, one GPU) with resource monitoring."""

    argv: list[str]
    backend: Optional[GpuBackend]
    gpu_index: Optional[int]
    log_path: Path
    startup_timeout: float = 900.0
    record_series: bool = False
    poll_interval: float = 1.0
    env_override: Optional[dict] = None
    """Extra environment for the server process (e.g. CUDA_VISIBLE_DEVICES=0,1 for a multi-GPU run)."""
    port: int = 0
    proc: Optional[subprocess.Popen] = None
    monitor: Optional[PeakMonitor] = None
    _log: Any = None

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> "ServerSession":
        self.port = free_port()
        argv = [*self.argv, "--host", "127.0.0.1", "--port", str(self.port)]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("w")
        self.proc = subprocess.Popen(argv, stdout=self._log, stderr=subprocess.STDOUT,
                                     env={**gpu_env(self.backend, self.gpu_index), **(self.env_override or {})}, start_new_session=True)
        self.monitor = PeakMonitor(self.backend, self.gpu_index, self.proc.pid, self.poll_interval,
                                   record_series=self.record_series)
        self.monitor.start()
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                self.stop()
                raise RuntimeError(f"llama-server exited during startup (rc={self.proc.returncode}); see {self.log_path}")
            try:
                if _http(self.base + "/health", timeout=5).get("status") == "ok":
                    return self
            except (urllib.error.URLError, OSError, ValueError):
                pass
            time.sleep(1.0)
        self.stop()
        raise RuntimeError(f"llama-server not ready within {self.startup_timeout:.0f}s; see {self.log_path}")

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def stop(self) -> Peaks:
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=30)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        peaks = self.monitor.stop() if self.monitor else Peaks()
        if self._log:
            self._log.close()
            self._log = None
        return peaks

    def props(self) -> dict[str, Any]:
        try:
            return _http(self.base + "/props")
        except (urllib.error.URLError, OSError, ValueError):
            return {}

    def tokenize(self, text: str) -> int:
        out = _http(self.base + "/tokenize", {"content": text}, timeout=60)
        return len(out.get("tokens", []))


def unique_prompt(session: ServerSession, target_tokens: int, seed: int,
                  instruction: str = "\n\nIgnore the text above. Count from 1 to 400, separated by spaces, "
                                     "with no other text.") -> tuple[str, int]:
    """Random text of about ``target_tokens`` tokens, calibrated with the server's own tokenizer."""

    rnd = random.Random(seed)
    words = max(8, int(target_tokens * 0.6))
    text, n = "", 0
    for _ in range(3):  # converge on the target using the real tokenizer
        text = f"[{seed:08x}] " + " ".join(rnd.choice(_WORDS) for _ in range(words)) + instruction
        n = session.tokenize(text)
        if abs(n - target_tokens) <= max(16, target_tokens // 50):
            break
        words = max(8, int(words * target_tokens / max(1, n)))
    return text, n


def stream_request(session: ServerSession, prompt: str, max_tokens: int, *, timeout: float = 3600.0) -> dict[str, Any]:
    """One streamed chat completion; returns timing, usage, server timings and per-chunk arrival times."""

    body = {"model": "x", "stream": True, "max_tokens": max_tokens, "temperature": 0, "cache_prompt": False,
            "stream_options": {"include_usage": True}, "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(session.base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft: Optional[float] = None
    stamps: list[float] = []
    usage = timings = None
    reasoning = answer = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            chunk = json.loads(line[5:])
            usage = chunk.get("usage") or usage
            timings = chunk.get("timings") or timings
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                rc, c = delta.get("reasoning_content") or "", delta.get("content") or ""
                if rc or c:
                    now = time.perf_counter() - t0
                    ttft = now if ttft is None else ttft
                    stamps.append(now)
                    reasoning += len(rc)
                    answer += len(c)
    total = time.perf_counter() - t0
    return {"ttft_s": ttft, "total_s": total, "usage": usage, "timings": timings, "stamps": stamps,
            "reasoning_chars": reasoning, "answer_chars": answer}


def _timing(r: dict[str, Any], key: str) -> Optional[float]:
    value = (r.get("timings") or {}).get(key)
    return float(value) if value not in (None, "") else None


def measure_sizes(session: ServerSession, sizes: Sequence[int], reps: int, max_tokens: int,
                  seed: int = 1) -> list[dict[str, Any]]:
    """TTFT / prefill / decode at each prompt size, ``reps`` unique prompts each."""

    results = []
    for size in sizes:
        runs = []
        for rep in range(reps):
            prompt, n_tokens = unique_prompt(session, size, seed + size * 31 + rep)
            r = stream_request(session, prompt, max_tokens)
            drafted, accepted = _timing(r, "draft_n"), _timing(r, "draft_n_accepted")
            runs.append({
                "prompt_tokens": (r["usage"] or {}).get("prompt_tokens", n_tokens),
                "completion_tokens": (r["usage"] or {}).get("completion_tokens"),
                "ttft_s": r["ttft_s"], "total_s": r["total_s"],
                "pp_tps": _timing(r, "prompt_per_second"), "tg_tps": _timing(r, "predicted_per_second"),
                "draft_n": drafted, "draft_n_accepted": accepted,
                "draft_acceptance": (accepted / drafted) if drafted else None,
            })
        ok = [x for x in runs if x["ttft_s"] is not None]
        results.append({
            "target_tokens": size, "reps": len(runs), "n_ok": len(ok), "runs": runs,
            "ttft_s": median_iqr([x["ttft_s"] for x in ok]),
            "pp_tps": median_iqr([x["pp_tps"] for x in ok if x["pp_tps"]]),
            "tg_tps": median_iqr([x["tg_tps"] for x in ok if x["tg_tps"]]),
            "total_s": median_iqr([x["total_s"] for x in ok]),
            "draft_acceptance": median_iqr([x["draft_acceptance"] for x in ok if x["draft_acceptance"] is not None]),
        })
    return results


def window_rates(stamps: Sequence[float], window_s: float) -> list[tuple[float, float, float]]:
    """(start, end, tok/s) per ``window_s`` window from per-token arrival stamps (seconds)."""

    if not stamps:
        return []
    out = []
    start, end_all = stamps[0], stamps[-1]
    t = start
    while t + window_s <= end_all:  # full windows only: a short trailing window would read as a slowdown
        n = sum(1 for s in stamps if t <= s < t + window_s)
        out.append((t, t + window_s, n / window_s))
        t += window_s
    if not out and end_all > start:  # run shorter than one window: report the whole span once
        out.append((start, end_all, len(stamps) / (end_all - start)))
    return out


def soak(session: ServerSession, duration_s: float, window_s: float, prompt_tokens: int = 512,
         max_tokens: int = 512) -> dict[str, Any]:
    """Keep the server decoding for ``duration_s`` and report tg per window with GPU telemetry.

    The thermal verdict is explicit and conservative: it passes only if no thermal/power-brake
    throttle bit was ever set, the SM clock never dropped by more than 10 % from its early value
    and decode speed in the last windows stayed within 10 % of the first windows."""

    assert session.monitor is not None
    t_start = time.monotonic()
    token_stamps: list[float] = []
    requests = 0
    while time.monotonic() - t_start < duration_s:
        prompt, _ = unique_prompt(session, prompt_tokens, seed=requests + 1000)
        offset = time.monotonic() - t_start
        r = stream_request(session, prompt, max_tokens)
        token_stamps.extend(offset + s for s in r["stamps"])
        requests += 1
    windows = []
    series = session.monitor.series
    for w0, w1, rate in window_rates(token_stamps, window_s):
        inside = [s for s in series if w0 <= s["t"] < w1]

        def agg(key: str, fn):
            vals = [s[key] for s in inside if s.get(key) is not None]
            return fn(vals) if vals else None
        windows.append({"t0": round(w0, 1), "t1": round(w1, 1), "tg_tps": round(rate, 2),
                        "temp_c": agg("temp_c", max), "sm_clock_mhz": agg("sm_clock_mhz", statistics.median),
                        "sm_clock_min_mhz": agg("sm_clock_mhz", min),
                        "power_w": agg("power_w", max), "throttle_mask": agg("throttle_mask", max)})
    rates = [w["tg_tps"] for w in windows]
    k = max(1, len(rates) // 5)
    first = statistics.median(rates[:k]) if rates else None
    last = statistics.median(rates[-k:]) if rates else None
    clocks = [w["sm_clock_mhz"] for w in windows if w["sm_clock_mhz"]]
    clock_drop = (1 - min(clocks) / statistics.median(clocks[:k])) if len(clocks) >= 2 else None
    mask = 0
    for s in series:
        mask |= int(s.get("throttle_mask") or 0)
    drift = (last / first - 1) if first and last else None
    thermal = bool(mask & THERMAL_THROTTLE_BITS)
    verdict = {
        "pass": bool(windows) and not thermal and (drift is None or drift >= -0.10)
                and (clock_drop is None or clock_drop <= 0.10),
        "criteria": "no thermal/power-brake throttle bit, SM clock drop <= 10 %, tg drift >= -10 %",
        "thermal_throttle_seen": thermal, "tg_drift": round(drift, 4) if drift is not None else None,
        "sm_clock_drop": round(clock_drop, 4) if clock_drop is not None else None,
        "throttle_reasons": decode_throttle(hex(mask)) if mask else [],
        "telemetry_available": bool(series),
    }
    return {"duration_s": round(time.monotonic() - t_start, 1), "requests": requests, "window_s": window_s,
            "tg_first_tps": first, "tg_last_tps": last, "temp_max_c": max((w["temp_c"] for w in windows if w["temp_c"]), default=None),
            "windows": windows, "verdict": verdict}


def spec_args(server: Path, spec_type: str, draft_n_max: int) -> Optional[list[str]]:
    """``--spec-type`` arguments, or ``None`` if this llama-server's ``--help`` does not list the type."""

    try:
        text = subprocess.run([str(server), "--help"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    if "--spec-type" not in text or spec_type not in text:
        return None
    return ["--spec-type", spec_type, "--spec-draft-n-max", str(draft_n_max)]


def command_with(profile_command: str, extra: Sequence[str]) -> list[str]:
    return [*shlex.split(profile_command), *extra]


def server_row_fields(result: dict[str, Any], peaks: Peaks) -> dict[str, Any]:
    """Pipeline-schema values for one measured size (median over reps)."""

    return {
        "pp_tps": result["pp_tps"]["median"], "tg_tps": result["tg_tps"]["median"],
        "ttft_ms": (result["ttft_s"]["median"] * 1000) if result["ttft_s"]["median"] is not None else None,
        "ttft_kind": "measured", "wall_time_s": result["total_s"]["median"],
        "vram_peak_mb": peaks.vram_peak_mib, "ram_peak_mb": peaks.ram_peak_mib,
        "draft_acceptance": result["draft_acceptance"]["median"],
    }
