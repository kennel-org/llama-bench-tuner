"""Pieces shared by the pipeline stages (grid, optuna, validation, profile)."""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from .capabilities import LlamaBenchCapabilities, probe_llama_bench
from .executor import ExecResult, run_monitored
from .gpu_gate import GateResult
from .measure import Measurer
from .schema import PIPELINE_RESULT_FIELDS, PIPELINE_SCHEMA_VERSION
from .telemetry import GpuBackend, GpuInfo, detect_backend
from .tune import _atomic_write_text, _file_identity, _load_json, benchmark_fingerprint


@dataclass
class Context:
    """Host/runtime facts fixed for a whole pipeline run."""

    llama_bench: Path
    model: Path
    caps: LlamaBenchCapabilities
    backend: Optional[GpuBackend]
    gpu: Optional[GpuInfo]
    gpu_index: Optional[int]
    timeout: float = 1500.0
    runner: Callable[..., ExecResult] = run_monitored
    gate: Optional[Callable[[], GateResult]] = None

    def measurer(self, stage_dir: Path, log_root: Path) -> Measurer:
        raw = stage_dir / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        log_root.mkdir(parents=True, exist_ok=True)
        return Measurer(llama_bench=self.llama_bench, model=self.model, caps=self.caps, backend=self.backend,
                        gpu=self.gpu, gpu_index=self.gpu_index, raw_dir=raw, log_dir=log_root,
                        timeout=self.timeout, runner=self.runner, gate=self.gate)

    def identity(self) -> dict[str, Any]:
        return {
            "llama_bench": str(self.llama_bench.resolve()),
            "llama_bench_identity": _file_identity(self.llama_bench, hash_content=True),
            "model": str(self.model.resolve()),
            "model_identity": _file_identity(self.model, hash_content=False),
            "gpu_index": self.gpu_index,
        }


def build_context(llama_bench: Path, model: Path, *, gpu_index: Optional[int], timeout: float,
                  backend: Optional[GpuBackend] = None, detect: bool = True,
                  runner: Callable[..., ExecResult] = run_monitored) -> Context:
    """Probe the binary and pick the single GPU to measure (never guessed on multi-GPU hosts)."""

    if backend is None and detect:
        backend = detect_backend()
    gpus = backend.list_gpus() if backend else []
    if gpu_index is None and len(gpus) > 1:
        raise SystemExit(
            f"[FATAL] {len(gpus)} GPUs detected; pass --gpu-index so single-GPU results are never mixed "
            "with multi-GPU ones."
        )
    if gpu_index is None and gpus:
        gpu_index = gpus[0].index
    gpu = next((g for g in gpus if g.index == gpu_index), None)
    return Context(llama_bench=llama_bench, model=model, caps=probe_llama_bench(llama_bench), backend=backend,
                   gpu=gpu, gpu_index=gpu_index, timeout=timeout, runner=runner)


class Checkpoint:
    """One checkpoint file per stage (overwritten), keyed by a fingerprint of the definition."""

    def __init__(self, path: Path, definition: dict[str, Any], resume: bool, label: str):
        self.path = path
        self.fingerprint = benchmark_fingerprint(definition)
        self.definition = definition
        self.rows: list[dict[str, Any]] = []
        self.done: list[str] = []
        self.base_elapsed = 0.0
        self.started = datetime.now(timezone.utc)
        if resume and path.exists():
            state = _load_json(path, f"{label} checkpoint")
            if state.get("benchmark_fingerprint") != self.fingerprint:
                raise SystemExit(f"[FATAL] Cannot resume {label}: settings differ from the checkpoint; "
                                 "use a new run directory.")
            self.rows = state.get("results", [])
            self.done = state.get("completed_items", [])
            self.base_elapsed = float(state.get("elapsed_seconds", 0.0))
            print(f"RESUME {label}: {len(self.done)} items from {path}")
        elif path.exists() and not resume:
            raise SystemExit(f"[FATAL] {path} exists; use --resume or a new run directory.")

    def elapsed(self) -> float:
        return self.base_elapsed + (datetime.now(timezone.utc) - self.started).total_seconds()

    def add(self, key: str, row: dict[str, Any], total: int, label: str) -> None:
        self.rows.append(row)
        self.done.append(key)
        elapsed = self.elapsed()
        eta = elapsed / len(self.done) * max(0, total - len(self.done))
        print(f"[{len(self.done)}/{total}] elapsed={elapsed:.0f}s eta={eta:.0f}s {label} {key} "
              f"status={row.get('status')} pp={row.get('pp_tps')} tg={row.get('tg_tps')} "
              f"vram_peak={row.get('vram_peak_mb')}MiB", flush=True)
        self.save()

    def save(self) -> None:
        elapsed = self.elapsed()
        _atomic_write_text(self.path, json.dumps({
            "pipeline_schema_version": PIPELINE_SCHEMA_VERSION, "benchmark_fingerprint": self.fingerprint,
            "completed_index": len(self.done), "completed_items": self.done, "results": self.rows,
            "elapsed_seconds": elapsed, "wall_clock": str(timedelta(seconds=int(elapsed))),
        }, ensure_ascii=False, indent=2) + "\n")


def write_rows_csv(path: Path, rows: Iterable[dict[str, Any]], extra_fields: Sequence[str] = ()) -> None:
    fields = PIPELINE_RESULT_FIELDS + [f for f in extra_fields if f not in PIPELINE_RESULT_FIELDS]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")


def read_json(path: Path, label: str) -> dict[str, Any]:
    return _load_json(path, label)


def median_iqr(values: Sequence[float]) -> dict[str, Optional[float]]:
    """Median, quartile spread and coefficient of variation; ``None`` where undefined."""

    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {"n": 0, "median": None, "min": None, "max": None, "iqr": None, "cv": None}
    median = statistics.median(vals)
    if len(vals) >= 4:
        q = statistics.quantiles(vals, n=4, method="inclusive")
        iqr = q[2] - q[0]
    elif len(vals) >= 2:
        iqr = vals[-1] - vals[0]  # range stands in for IQR with <4 samples
    else:
        iqr = None
    cv = (statistics.pstdev(vals) / statistics.mean(vals)) if len(vals) >= 2 and statistics.mean(vals) else None
    return {"n": len(vals), "median": median, "min": vals[0], "max": vals[-1],
            "iqr": iqr, "cv": round(cv, 4) if cv is not None else None}


def load_capacity(path: Path) -> dict[str, Any]:
    data = read_json(path, "capacity.json")
    if data.get("stage") != "capacity":
        raise SystemExit(f"[FATAL] {path} is not a capacity.json")
    return data


def model_quant_from_name(model: Path) -> Optional[str]:
    """Best-effort quant label from the file name (recorded with its source, never authoritative)."""

    import re
    match = re.search(r"((?:IQ|Q)\d(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4|NVFP4)", model.name, re.IGNORECASE)
    return match.group(1).upper() if match else None
