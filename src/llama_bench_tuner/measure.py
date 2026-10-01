"""Measure one benchmark point (``BenchPoint``) with llama-bench and return a pipeline row.

Shared by every pipeline stage (capacity, grid, optuna, validation) so command
construction, failure classification, telemetry and result naming live in one place.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional

from .capabilities import LlamaBenchCapabilities
from .command_builder import LlamaBenchCommand, build_llama_bench_command
from .executor import ExecResult, gpu_env, run_monitored
from .parsing import extract_metrics_by_depth, parse_bench_csv
from .schema import environment_metadata, pipeline_row
from .status import BenchStatus, classify_bench_outcome
from .gpu_gate import GateResult
from .telemetry import GpuBackend, GpuInfo


class GpuBusyError(RuntimeError):
    """The GPU never became free; the stage stops cleanly and can be resumed."""

    def __init__(self, gate: GateResult):
        super().__init__(gate.reason)
        self.gate = gate


@dataclass(frozen=True)
class BenchPoint:
    """Every tunable axis of one measurement. ``None`` = leave llama-bench's default."""

    kv: str = "f16"
    depth: int = 0
    ngl: int = 99
    batch: int = 2048
    ubatch: int = 512
    flash_attn: int = 1
    n_cpu_moe: Optional[int] = None
    split_mode: Optional[str] = None
    nkvo: Optional[int] = None
    prompt: int = 512
    ngen: int = 64
    reps: int = 1
    threads: Optional[int] = None

    def key(self) -> str:
        parts = [f"kv={self.kv}", f"d={self.depth}", f"ngl={self.ngl}", f"b={self.batch}",
                 f"ub={self.ubatch}", f"fa={self.flash_attn}"]
        if self.n_cpu_moe is not None:
            parts.append(f"ncmoe={self.n_cpu_moe}")
        if self.split_mode:
            parts.append(f"sm={self.split_mode}")
        if self.nkvo is not None:
            parts.append(f"nkvo={self.nkvo}")
        parts += [f"p={self.prompt}", f"n={self.ngen}"]
        return ",".join(parts)

    def config_key(self) -> str:
        """Identity of the *configuration* (everything except depth / measurement length)."""

        return self.replace(depth=0, prompt=0, ngen=0, reps=1).key()

    def replace(self, **changes: Any) -> "BenchPoint":
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def point_command(llama_bench: Path, model: Path, point: BenchPoint) -> list[str]:
    spec = LlamaBenchCommand(
        llama_bench=llama_bench, model=model, threads=point.threads, ngl=point.ngl, batch=point.batch,
        ubatch=point.ubatch, prompt=point.prompt, ngen=point.ngen, mmap=None, flash_attn=point.flash_attn,
        nkvo=point.nkvo, split_mode=point.split_mode, depth=point.depth, cache_type_k=point.kv,
        cache_type_v=point.kv, repetitions=point.reps, n_cpu_moe=point.n_cpu_moe,
    )
    return build_llama_bench_command(spec)


def unsupported_reason(caps: LlamaBenchCapabilities, point: BenchPoint) -> str:
    """Why a point must not be executed, from the ``--help`` probe. Empty = run it.

    A failed probe never blocks execution: absence of evidence is not evidence of absence."""

    if caps.error == "":
        if point.depth > 0 and caps.flags.get("n_depth") is False:
            return "llama-bench --help shows no -d/--n-depth"
        if point.kv != "f16" and (caps.flags.get("cache_type_k") is False or caps.flags.get("cache_type_v") is False):
            return "llama-bench --help shows no -ctk/-ctv"
        if point.n_cpu_moe is not None and caps.flags.get("n_cpu_moe") is False:
            return "llama-bench --help shows no -ncmoe/--n-cpu-moe"
        if point.split_mode and caps.flags.get("split_mode") is False:
            return "llama-bench --help shows no -sm/--split-mode"
    if point.kv != "f16" and point.flash_attn == 0:
        return "quantized KV cache requires flash attention (llama.cpp constraint)"
    if point.ubatch > point.batch:
        return "ubatch larger than batch is not a valid llama.cpp configuration"
    return ""


@dataclass
class Measurer:
    """Everything fixed for a run: binary, model, host/GPU, output dirs, executor."""

    llama_bench: Path
    model: Path
    caps: LlamaBenchCapabilities
    backend: Optional[GpuBackend]
    gpu: Optional[GpuInfo]
    gpu_index: Optional[int]
    raw_dir: Path
    log_dir: Path
    timeout: float = 1500.0
    runner: Callable[..., ExecResult] = run_monitored
    gate: Optional[Callable[[], GateResult]] = None

    def base_row(self, stage: str, point: BenchPoint) -> dict[str, Any]:
        env = environment_metadata()
        return pipeline_row(
            stage=stage, benchmark_kind="llama-bench", case_key=point.key(),
            host=env["host"], gpu=env["gpu"] or (self.gpu.name if self.gpu else ""),
            gpu_connection=env["gpu_connection"], platform=env["platform"], python=env["python"],
            backend="llama.cpp/llama-bench", llama_bench=str(self.llama_bench), model=str(self.model),
            model_size_bytes=self.model.stat().st_size if self.model.exists() else None,
            context=point.depth + point.prompt + point.ngen, context_depth=point.depth,
            prompt_tokens=point.prompt, generated_tokens=point.ngen, batch=point.batch,
            ubatch=point.ubatch, kv_type=point.kv, kv_type_k=point.kv, kv_type_v=point.kv,
            flash_attn=point.flash_attn, ngl=point.ngl, n_cpu_moe=point.n_cpu_moe,
            gpu_index=self.gpu_index, driver_version=self.gpu.driver_version if self.gpu else None,
            vram_total_mb=self.gpu.memory_total_mib if self.gpu else None, attempts=1,
        )

    def measure(self, stage: str, point: BenchPoint, tag: str = "") -> dict[str, Any]:
        row = self.base_row(stage, point)
        reason = unsupported_reason(self.caps, point)
        if reason:
            row.update(status=BenchStatus.UNSUPPORTED.value, success=False, capacity_ok=False,
                       practical_candidate=False, error=reason, skip_reason=reason,
                       practical_reason="unsupported", telemetry_status="not_run")
            return row

        if self.gate is not None:
            verdict = self.gate()
            if not verdict.ok:
                raise GpuBusyError(verdict)
        cmd = point_command(self.llama_bench, self.model, point)
        result = self.runner(cmd, timeout=self.timeout, backend=self.backend, gpu_index=self.gpu_index,
                             env=gpu_env(self.backend, self.gpu_index))
        slug = "".join(c if c.isalnum() or c in "-_=" else "_" for c in point.key())
        name = f"{stage}_{tag + '_' if tag else ''}{slug}"
        csv_path = self.raw_dir / f"{name}.csv"
        err_path = self.log_dir / f"{name}.stderr.txt"
        csv_path.write_text(result.stdout)
        if result.stderr.strip():
            err_path.write_text(result.stderr)

        rows = parse_bench_csv(result.stdout.splitlines())
        metrics = extract_metrics_by_depth(rows).get(point.depth)
        pp = metrics.pp_tps if metrics else None
        tg = metrics.tg_tps if metrics else None
        outcome = classify_bench_outcome(returncode=result.returncode, decode_tps=tg, stdout=result.stdout,
                                         stderr=result.stderr, timed_out=result.timed_out)
        peaks = result.peaks
        completed = outcome.status is BenchStatus.SUCCESS
        row.update(
            status=outcome.status.value, success=completed, capacity_ok=completed,
            error="" if completed else outcome.error, returncode=result.returncode, pp_tps=pp, tg_tps=tg,
            wall_time_s=round(result.elapsed_s, 3), start=result.start_iso, end=result.end_iso,
            vram_baseline_mb=peaks.vram_baseline_mib, vram_peak_mb=peaks.vram_peak_mib,
            ram_peak_mb=peaks.ram_peak_mib, ram_peak_source=peaks.ram_peak_source,
            temp_max_c=peaks.temp_max_c, sm_clock_min_mhz=peaks.sm_clock_min_mhz,
            power_max_w=peaks.power_max_w, throttle_reasons=peaks.throttle_reasons,
            telemetry_status=peaks.status,
            runtime_commit=next((r.values.get("build_commit") for r in rows if r.values.get("build_commit")), None),
            csv=csv_path.relative_to(self.raw_dir.parent).as_posix(),
            stderr=err_path.relative_to(self.log_dir.parent).as_posix() if err_path.exists() else "",
        )
        return row


def estimate_ttft_s(depth: int, pp_d0: Optional[float], pp_depth: Optional[float]) -> Optional[float]:
    """TTFT estimate for a ``depth``-token prompt from pp at depth 0 and at ``depth``.

    ``llama-bench`` reports no time-to-first-token. Prefill rate falls as the context
    fills, so the time to ingest ``depth`` tokens is approximated by the trapezoid of the
    reciprocal rates: ``depth * (1/pp0 + 1/ppD) / 2``. This is an *estimate* and is
    always labelled ``ttft_kind=estimated_from_pp``; real TTFT needs a server run."""

    if not pp_d0 or not pp_depth or pp_d0 <= 0 or pp_depth <= 0:
        return None
    return round(depth * (1.0 / pp_d0 + 1.0 / pp_depth) / 2.0, 3)
