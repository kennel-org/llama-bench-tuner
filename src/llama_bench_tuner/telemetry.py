"""Best-effort GPU/RAM telemetry. Missing data stays ``None``; nothing is guessed.

Backends are small adapters so the same capacity/pipeline code runs on NVIDIA
(native Linux and WSL), AMD (sysfs, including APU unified memory) and hosts with
no GPU interface at all.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

_WSL_NVIDIA_SMI = "/usr/lib/wsl/lib/nvidia-smi"


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    memory_total_mib: Optional[int]
    driver_version: Optional[str] = None
    backend: str = ""


@dataclass
class Peaks:
    vram_baseline_mib: Optional[int] = None
    vram_peak_mib: Optional[int] = None
    ram_peak_mib: Optional[int] = None
    ram_peak_source: Optional[str] = None
    temp_max_c: Optional[float] = None
    sm_clock_min_mhz: Optional[float] = None
    power_max_w: Optional[float] = None
    throttle_reasons: Optional[str] = None
    samples: int = 0
    status: str = "none"


class GpuBackend(Protocol):
    name: str

    def list_gpus(self) -> list[GpuInfo]: ...

    def used_mib(self, index: int) -> Optional[int]: ...

    def free_mib(self, index: int) -> Optional[int]: ...

    def sample(self, index: int) -> dict[str, Optional[float]]: ...

    def compute_processes(self, index: int) -> list[str]: ...


def _to_float(text: str) -> Optional[float]:
    text = text.strip().split(" ")[0]
    try:
        return float(text)
    except ValueError:
        return None  # "[N/A]", "Not Supported", ""


def _run(cmd: list[str], timeout: float = 10.0) -> Optional[str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout if proc.returncode == 0 else None


class NvidiaSmiBackend:
    name = "nvidia-smi"

    def __init__(self, binary: str):
        self.binary = binary
        self._throttle_field: Optional[str] = None

    def _query(self, index: int, fields: str) -> Optional[list[str]]:
        out = _run([self.binary, "-i", str(index), f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
        if out is None or not out.strip():
            return None
        return [part.strip() for part in out.strip().splitlines()[0].split(",")]

    def list_gpus(self) -> list[GpuInfo]:
        out = _run([self.binary, "--query-gpu=index,name,memory.total,driver_version",
                    "--format=csv,noheader,nounits"])
        gpus: list[GpuInfo] = []
        for line in (out or "").strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            total = _to_float(parts[2])
            gpus.append(GpuInfo(int(parts[0]), parts[1], int(total) if total else None,
                                parts[3] or None, self.name))
        return gpus

    def used_mib(self, index: int) -> Optional[int]:
        row = self._query(index, "memory.used")
        value = _to_float(row[0]) if row else None
        return int(value) if value is not None else None

    def free_mib(self, index: int) -> Optional[int]:
        row = self._query(index, "memory.free")
        value = _to_float(row[0]) if row else None
        return int(value) if value is not None else None

    def _throttle_query_field(self, index: int) -> Optional[str]:
        if self._throttle_field is None:
            self._throttle_field = ""
            for candidate in ("clocks_event_reasons.active", "clocks_throttle_reasons.active"):
                if self._query(index, candidate) is not None:
                    self._throttle_field = candidate
                    break
        return self._throttle_field or None

    def sample(self, index: int) -> dict[str, Optional[float]]:
        throttle = self._throttle_query_field(index)
        fields = "memory.used,temperature.gpu,clocks.sm,power.draw" + (f",{throttle}" if throttle else "")
        row = self._query(index, fields)
        if row is None:
            return {}
        row = row + [""] * (4 - len(row))  # tolerate short/odd rows; missing -> None
        values: dict[str, Optional[float]] = {
            "used_mib": _to_float(row[0]), "temp_c": _to_float(row[1]),
            "sm_clock_mhz": _to_float(row[2]), "power_w": _to_float(row[3]),
        }
        if throttle and len(row) > 4:
            try:
                values["throttle_mask"] = float(int(row[4], 16))
            except ValueError:
                values["throttle_mask"] = None
        return values

    def compute_processes(self, index: int) -> list[str]:
        out = _run([self.binary, "-i", str(index), "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader"])
        return [line.strip() for line in (out or "").splitlines() if line.strip()]


class AmdSysfsBackend:
    """amdgpu sysfs counters. APU pools are reported as VRAM+GTT (unified memory)."""

    name = "amdgpu-sysfs"

    def __init__(self, drm_root: Path = Path("/sys/class/drm")):
        self.root = drm_root

    def _devices(self) -> list[Path]:
        devices = []
        for card in sorted(self.root.glob("card[0-9]*")):
            dev = card / "device"
            if (dev / "mem_info_vram_total").exists():
                devices.append(dev)
        return devices

    @staticmethod
    def _read_int(path: Path) -> Optional[int]:
        try:
            return int(path.read_text().strip())
        except (OSError, ValueError):
            return None

    def _pool(self, dev: Path) -> tuple[Optional[int], Optional[int]]:
        vram_t = self._read_int(dev / "mem_info_vram_total") or 0
        gtt_t = self._read_int(dev / "mem_info_gtt_total") or 0
        vram_u = self._read_int(dev / "mem_info_vram_used")
        gtt_u = self._read_int(dev / "mem_info_gtt_used")
        if vram_u is None and gtt_u is None:
            return None, None
        return (vram_t + gtt_t) // (1 << 20), ((vram_u or 0) + (gtt_u or 0)) // (1 << 20)

    def list_gpus(self) -> list[GpuInfo]:
        gpus = []
        for index, dev in enumerate(self._devices()):
            total, _ = self._pool(dev)
            name = ""
            try:
                name = (dev / "product_name").read_text().strip()
            except OSError:
                pass
            gpus.append(GpuInfo(index, name or "AMD GPU (sysfs)", total, None, self.name))
        return gpus

    def used_mib(self, index: int) -> Optional[int]:
        devices = self._devices()
        return self._pool(devices[index])[1] if index < len(devices) else None

    def free_mib(self, index: int) -> Optional[int]:
        devices = self._devices()
        if index >= len(devices):
            return None
        total, used = self._pool(devices[index])
        return None if total is None or used is None else total - used

    def sample(self, index: int) -> dict[str, Optional[float]]:
        used = self.used_mib(index)
        return {"used_mib": float(used) if used is not None else None}

    def compute_processes(self, index: int) -> list[str]:
        return []  # sysfs exposes no per-process list; never guessed


def detect_backend(preferred: Optional[str] = None) -> Optional[GpuBackend]:
    """Pick a telemetry backend. ``BENCH_NVIDIA_SMI`` overrides the binary path."""

    if preferred in (None, "auto", "nvidia"):
        binary = os.environ.get("BENCH_NVIDIA_SMI") or shutil.which("nvidia-smi")
        if not binary and Path(_WSL_NVIDIA_SMI).exists():
            binary = _WSL_NVIDIA_SMI
        if binary:
            return NvidiaSmiBackend(binary)
    if preferred in (None, "auto", "amd"):
        amd = AmdSysfsBackend()
        if amd._devices():
            return amd
    return None


def read_rss_peak_mib(pid: int) -> Optional[int]:
    """Peak resident set of a live Linux process (VmHWM); ``None`` where /proc is absent."""

    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


@dataclass
class PeakMonitor:
    """Samples a GPU (and the child's RSS) while a benchmark process runs."""

    backend: Optional[GpuBackend]
    gpu_index: Optional[int]
    pid: int
    interval: float = 1.0
    peaks: Peaks = field(default_factory=Peaks)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: Optional[threading.Thread] = None

    def _poll(self) -> None:
        peaks = self.peaks
        index = self.gpu_index if self.gpu_index is not None else 0
        while True:
            if self.backend is not None:
                sample = self.backend.sample(index)
                used = sample.get("used_mib")
                if used is not None:
                    peaks.vram_peak_mib = max(peaks.vram_peak_mib or 0, int(used))
                temp = sample.get("temp_c")
                if temp is not None:
                    peaks.temp_max_c = max(peaks.temp_max_c or temp, temp)
                clock = sample.get("sm_clock_mhz")
                if clock is not None:
                    peaks.sm_clock_min_mhz = min(peaks.sm_clock_min_mhz or clock, clock)
                power = sample.get("power_w")
                if power is not None:
                    peaks.power_max_w = max(peaks.power_max_w or power, power)
                mask = sample.get("throttle_mask")
                if mask:
                    merged = int(peaks.throttle_reasons or "0x0", 16) | int(mask)
                    peaks.throttle_reasons = hex(merged)
                if sample:
                    peaks.samples += 1
            rss = read_rss_peak_mib(self.pid)
            if rss is not None:
                peaks.ram_peak_mib = max(peaks.ram_peak_mib or 0, rss)
                peaks.ram_peak_source = "proc_vmhwm"
            if self._stop.wait(self.interval):
                return

    def start(self) -> None:
        if self.backend is not None:
            self.peaks.vram_baseline_mib = self.backend.used_mib(
                self.gpu_index if self.gpu_index is not None else 0
            )
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> Peaks:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self.backend is None and self.peaks.ram_peak_mib is None:
            self.peaks.status = "none"
        elif self.peaks.samples == 0 and self.backend is not None:
            self.peaks.status = "gpu_unavailable"
        elif self.backend is None:
            self.peaks.status = "ram_only"
        else:
            self.peaks.status = "ok" if self.peaks.ram_peak_mib is not None else "gpu_only"
        return self.peaks
