"""Phase 0 — capacity / compatibility sweep (``llama-tune-capacity``).

Finds, per KV-cache type, how deep a context a model/runtime/host can run and how
deep it stays *practical*. One ``llama-bench`` process per (kv, depth) so a single
abort/OOM/timeout costs only that point. Outputs a normalized CSV, ``capacity.json``
(boundaries + survivors for the Grid stage) and a one-file checkpoint for resume.

Nothing here guesses hardware: GPU/RAM numbers are ``None`` unless measured, flags
absent from ``--help`` are recorded as ``unsupported`` instead of being executed, and
other processes on the GPU are never touched (see :mod:`gpu_gate`).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from .capabilities import LlamaBenchCapabilities, probe_llama_bench
from .executor import ExecResult, run_monitored
from .gpu_gate import wait_for_gpu
from .measure import BenchPoint, Measurer, point_command
from .pareto import pareto_by_group
from .schema import PIPELINE_RESULT_FIELDS, PIPELINE_SCHEMA_VERSION
from .status import BenchStatus
from .telemetry import GpuBackend, GpuInfo, detect_backend
from .tune import _atomic_write_text, _file_identity, _load_json, benchmark_fingerprint

DEFAULT_DEPTHS = (0, 4096, 8192, 16384, 32768, 65536, 131072)
EXTENDED_DEPTHS = (262144, 393216, 524288)
DEFAULT_KV = ("f16", "q8_0", "q4_0")
_FAILURE_STATUSES = {
    BenchStatus.OOM.value, BenchStatus.RUNTIME_ABORT.value, BenchStatus.TIMEOUT.value,
    BenchStatus.FAILED.value,
}


# ----------------------------------------------------------------------------- config
@dataclass(frozen=True)
class PracticalCriteria:
    """What "practical" means. All thresholds are explicit and recorded in the output."""

    min_tg_tps: float = 10.0
    tg_ratio: float = 0.5          # tg(depth) / tg(d0, same KV) must stay above this
    pp_ratio: float = 0.3
    max_wall_s: Optional[float] = None
    vram_margin_mib: int = 512     # peak VRAM must leave at least this much headroom
    slowdown_tg_ratio: float = 0.25  # below this a completed run is reported as ``slowdown``
    slowdown_pp_ratio: float = 0.10


@dataclass(frozen=True)
class CapacityConfig:
    llama_bench: Path
    model: Path
    depths: tuple[int, ...] = DEFAULT_DEPTHS
    kv_types: tuple[str, ...] = DEFAULT_KV
    ngl: int = 99
    batch: int = 2048
    ubatch: int = 512
    flash_attn: int = 1
    prompt: int = 512
    ngen: int = 64
    repetitions: int = 1
    threads: Optional[int] = None
    per_run_timeout: float = 1500.0
    gpu_index: Optional[int] = None
    min_free_vram_mib: Optional[int] = None
    wait_for_gpu: bool = False
    gpu_wait_timeout: float = 0.0
    gpu_poll_s: float = 60.0
    stop_after_fail: bool = True
    criteria: PracticalCriteria = PracticalCriteria()


# ------------------------------------------------------------------------ pure helpers
_SUFFIX = {"k": 1024, "m": 1024 * 1024}


def parse_depth(token: str) -> int:
    """``4096``, ``4k``, ``128K``, ``1m`` -> int."""

    match = re.fullmatch(r"(\d+)([kKmM]?)", token.strip())
    if not match:
        raise ValueError(f"invalid depth {token!r}")
    return int(match.group(1)) * _SUFFIX.get(match.group(2).lower(), 1)


def parse_int_list(values: Sequence[str]) -> tuple[int, ...]:
    tokens = [t for v in values for t in v.split(",") if t.strip()]
    return tuple(parse_depth(t) for t in tokens)


def parse_str_list(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(t.strip() for v in values for t in v.split(",") if t.strip())


def case_key(kv: str, depth: int) -> str:
    return f"kv={kv},d={depth}"


def evaluate_practical(*, completed: bool, tg: Optional[float], pp: Optional[float],
                       tg0: Optional[float], pp0: Optional[float], wall_s: float,
                       vram_peak: Optional[int], vram_total: Optional[int],
                       crit: PracticalCriteria) -> tuple[bool, str]:
    """Return (practical_candidate, reason). A candidate still needs repeated validation."""

    if not completed:
        return False, "run did not complete"
    if tg is None or tg <= 0:
        return False, "no tg measurement"
    problems: list[str] = []
    if tg < crit.min_tg_tps:
        problems.append(f"tg {tg:.1f} < min {crit.min_tg_tps:g} tok/s")
    if tg0 and tg / tg0 < crit.tg_ratio:
        problems.append(f"tg ratio {tg / tg0:.2f} < {crit.tg_ratio:g}")
    if pp0 and pp is not None and pp / pp0 < crit.pp_ratio:
        problems.append(f"pp ratio {pp / pp0:.2f} < {crit.pp_ratio:g}")
    if crit.max_wall_s is not None and wall_s > crit.max_wall_s:
        problems.append(f"wall {wall_s:.0f}s > {crit.max_wall_s:g}s")
    if vram_peak is not None and vram_total is not None and vram_total - vram_peak < crit.vram_margin_mib:
        problems.append(f"VRAM headroom {vram_total - vram_peak} MiB < {crit.vram_margin_mib} MiB")
    return (not problems), "; ".join(problems)


def _ratio(value: Optional[float], base: Optional[float]) -> Optional[float]:
    return round(value / base, 4) if value is not None and base else None


# ---------------------------------------------------------------------------- one case
def _point(cfg: CapacityConfig, kv: str, depth: int) -> BenchPoint:
    return BenchPoint(kv=kv, depth=depth, ngl=cfg.ngl, batch=cfg.batch, ubatch=cfg.ubatch,
                      flash_attn=cfg.flash_attn, prompt=cfg.prompt, ngen=cfg.ngen,
                      reps=cfg.repetitions, threads=cfg.threads)


def _command(cfg: CapacityConfig, kv: str, depth: int) -> list[str]:
    return point_command(cfg.llama_bench, cfg.model, _point(cfg, kv, depth))


def run_case(cfg: CapacityConfig, kv: str, depth: int, *, measurer: Measurer,
             baselines: dict[str, dict[str, float]]) -> dict[str, Any]:
    row = measurer.measure("capacity", _point(cfg, kv, depth))
    if row["status"] == BenchStatus.UNSUPPORTED.value:
        return row

    status = row["status"]
    completed = status == BenchStatus.SUCCESS.value
    pp, tg = row["pp_tps"], row["tg_tps"]
    base = baselines.get(kv) or baselines.get("f16") or {}
    tg0, pp0 = base.get("tg"), base.get("pp")
    if depth == 0 and completed:
        baselines[kv] = {"tg": tg, "pp": pp}
        tg0, pp0 = tg, pp
    pp_ratio, tg_ratio = _ratio(pp, pp0), _ratio(tg, tg0)
    crit = cfg.criteria
    if completed and ((tg_ratio is not None and tg_ratio < crit.slowdown_tg_ratio)
                      or (pp_ratio is not None and pp_ratio < crit.slowdown_pp_ratio)):
        status = BenchStatus.SLOWDOWN.value
    practical, practical_reason = evaluate_practical(
        completed=completed and status == BenchStatus.SUCCESS.value, tg=tg, pp=pp, tg0=tg0, pp0=pp0,
        wall_s=row["wall_time_s"] or 0.0, vram_peak=row["vram_peak_mb"], vram_total=row["vram_total_mb"],
        crit=crit)
    row.update(status=status, practical_candidate=practical, practical_reason=practical_reason,
               pp_ratio_vs_d0=pp_ratio, tg_ratio_vs_d0=tg_ratio)
    return row


# ------------------------------------------------------------------------ aggregation
def summarize_boundaries(rows: list[dict[str, Any]], cfg: CapacityConfig) -> dict[str, Any]:
    """Capacity boundary (deepest completed) vs practical-candidate boundary, per KV type."""

    per_kv: dict[str, Any] = {}
    for kv in cfg.kv_types:
        kv_rows = sorted((r for r in rows if r["kv_type"] == kv), key=lambda r: r["context_depth"])
        ok = [r for r in kv_rows if r["capacity_ok"]]
        practical = [r for r in kv_rows if r["practical_candidate"]]
        failures = [r for r in kv_rows if r["status"] in _FAILURE_STATUSES]
        per_kv[kv] = {
            "capacity_max_depth": max((r["context_depth"] for r in ok), default=None),
            "practical_candidate_max_depth": max((r["context_depth"] for r in practical), default=None),
            "first_failure": ({"depth": failures[0]["context_depth"], "status": failures[0]["status"],
                               "error": failures[0]["error"]} if failures else None),
            "unsupported": all(r["status"] == BenchStatus.UNSUPPORTED.value for r in kv_rows) if kv_rows else None,
            "note": "practical_candidate needs repeated validation (stage 'validation') before it is final",
        }
    return per_kv


def pareto_tables(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Per-depth non-dominated KV configs over pp↑, tg↑, peak VRAM↓ (completed runs only)."""

    done = [r for r in rows if r["capacity_ok"]]
    fronts = pareto_by_group(done, "context_depth", maximize=["pp_tps", "tg_tps"], minimize=["vram_peak_mb"])
    return {
        str(depth): [{k: r[k] for k in ("kv_type", "pp_tps", "tg_tps", "vram_peak_mb", "ram_peak_mb", "status")}
                     for r in front]
        for depth, front in sorted(fronts.items())
    }


def write_outputs(run_root: Path, rows: list[dict[str, Any]], cfg: CapacityConfig, meta: dict[str, Any]) -> None:
    csv_path = run_root / "capacity.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PIPELINE_RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    boundaries = summarize_boundaries(rows, cfg)
    summary = {
        **meta,
        "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
        "stage": "capacity",
        "criteria": asdict(cfg.criteria),
        "per_kv": boundaries,
        "survivors": [{"kv": r["kv_type"], "depth": r["context_depth"]} for r in rows if r["practical_candidate"]],
        "capacity_survivors": [{"kv": r["kv_type"], "depth": r["context_depth"]} for r in rows if r["capacity_ok"]],
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "pareto_by_depth": pareto_tables(rows),
    }
    _atomic_write_text(run_root / "capacity.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


# ------------------------------------------------------------------------------- driver
def _fingerprint(cfg: CapacityConfig) -> tuple[dict[str, Any], str]:
    definition = {
        "stage": "capacity",
        "llama_bench": str(cfg.llama_bench.resolve()),
        "llama_bench_identity": _file_identity(cfg.llama_bench, hash_content=True),
        "model": str(cfg.model.resolve()),
        "model_identity": _file_identity(cfg.model, hash_content=False),
        "depths": list(cfg.depths), "kv_types": list(cfg.kv_types), "ngl": cfg.ngl, "batch": cfg.batch,
        "ubatch": cfg.ubatch, "flash_attn": cfg.flash_attn, "prompt": cfg.prompt, "ngen": cfg.ngen,
        "repetitions": cfg.repetitions, "threads": cfg.threads, "gpu_index": cfg.gpu_index,
        "per_run_timeout": cfg.per_run_timeout, "criteria": asdict(cfg.criteria),
    }
    return definition, benchmark_fingerprint(definition)


def resolve_gpu(cfg: CapacityConfig, backend: Optional[GpuBackend]) -> tuple[Optional[GpuInfo], Optional[int], str]:
    """Pick the single GPU to measure. Multi-GPU hosts must be explicit (single vs dual are never mixed)."""

    if backend is None:
        return None, cfg.gpu_index, "no GPU telemetry backend; running unpinned without VRAM numbers"
    gpus = backend.list_gpus()
    if cfg.gpu_index is not None:
        match = next((g for g in gpus if g.index == cfg.gpu_index), None)
        return match, cfg.gpu_index, ""
    if len(gpus) <= 1:
        return (gpus[0] if gpus else None), (gpus[0].index if gpus else None), ""
    gate = wait_for_gpu(backend, min_free_mib=cfg.min_free_vram_mib, wait=cfg.wait_for_gpu,
                        timeout_s=cfg.gpu_wait_timeout, poll_s=cfg.gpu_poll_s)
    if cfg.min_free_vram_mib and gate.ok and gate.gpu_index is not None:
        return next((g for g in gpus if g.index == gate.gpu_index), None), gate.gpu_index, ""
    raise SystemExit(
        f"[FATAL] {len(gpus)} GPUs detected; pass --gpu-index (single-GPU baseline) or "
        "--min-free-vram-gib to select one. Dual-GPU runs are not part of this sweep."
    )


def run_capacity(cfg: CapacityConfig, run_root: Path, log_root: Path, *, backend: Optional[GpuBackend],
                 resume: bool = False, runner: Callable[..., ExecResult] = run_monitored,
                 sleep: Callable[[float], None] | None = None) -> tuple[list[dict[str, Any]], int]:
    """Execute the sweep. Returns (rows, exit_code); exit 3 = stopped because the GPU was busy."""

    raw_dir = run_root / "raw"
    for path in (run_root, raw_dir, log_root):
        path.mkdir(parents=True, exist_ok=True)
    caps = probe_llama_bench(cfg.llama_bench)
    definition, fingerprint = _fingerprint(cfg)
    checkpoint = run_root / "checkpoint.json"
    metadata = run_root / "run_metadata.json"
    rows: list[dict[str, Any]] = []
    done: list[str] = []
    base_elapsed = 0.0
    if resume and checkpoint.exists():
        state = _load_json(checkpoint, "checkpoint")
        if state.get("benchmark_fingerprint") != fingerprint:
            raise SystemExit("[FATAL] Cannot resume: capacity fingerprint differs; use a new --run-dir.")
        rows, done = state.get("results", []), state.get("completed_items", [])
        base_elapsed = float(state.get("elapsed_seconds", 0.0))
        print(f"RESUME {len(done)} cases from {checkpoint}")

    gpu, gpu_index, gpu_note = resolve_gpu(cfg, backend)
    if gpu_note:
        print(f"[WARN] {gpu_note}")
    started = datetime.now(timezone.utc)
    meta = {
        "started": started.isoformat(timespec="seconds"), "benchmark_fingerprint": fingerprint,
        "benchmark_definition": definition, "capabilities": caps.to_dict(),
        "gpu": asdict(gpu) if gpu else None, "gpu_index": gpu_index, "gpu_note": gpu_note or None,
        "runtime": {"llama_bench_help_sha256": caps.help_sha256},
    }
    if not (resume and metadata.exists()):
        _atomic_write_text(metadata, json.dumps(meta, ensure_ascii=False, indent=2) + "\n")

    measurer = Measurer(llama_bench=cfg.llama_bench, model=cfg.model, caps=caps, backend=backend, gpu=gpu,
                        gpu_index=gpu_index, raw_dir=raw_dir, log_dir=log_root, timeout=cfg.per_run_timeout,
                        runner=runner)
    cases = [(kv, depth) for kv in cfg.kv_types for depth in sorted(cfg.depths)]
    baselines: dict[str, dict[str, float]] = {}
    for r in rows:  # rebuild d0 baselines after a resume
        if r["context_depth"] == 0 and r["capacity_ok"]:
            baselines[r["kv_type"]] = {"tg": r["tg_tps"], "pp": r["pp_tps"]}
    failed_kv: set[str] = {r["kv_type"] for r in rows if r["status"] in _FAILURE_STATUSES} if cfg.stop_after_fail else set()
    exit_code = 0

    for index, (kv, depth) in enumerate(cases, 1):
        key = case_key(kv, depth)
        if key in done:
            continue
        if kv in failed_kv:
            row = measurer.base_row("capacity", _point(cfg, kv, depth))
            row.update(status=BenchStatus.SKIPPED_AFTER_FAIL.value, success=False, capacity_ok=False,
                       practical_candidate=False, skip_reason="shallower depth already failed for this KV type",
                       error="", practical_reason="skipped_after_fail", telemetry_status="not_run")
        else:
            gate = wait_for_gpu(backend, min_free_mib=cfg.min_free_vram_mib, gpu_index=gpu_index,
                                wait=cfg.wait_for_gpu, timeout_s=cfg.gpu_wait_timeout, poll_s=cfg.gpu_poll_s,
                                **({"sleep": sleep} if sleep else {}))
            if not gate.ok:
                row = measurer.base_row("capacity", _point(cfg, kv, depth))
                row.update(status=BenchStatus.GPU_BUSY.value, success=False, capacity_ok=False,
                           practical_candidate=False, error=gate.reason, telemetry_status="not_run",
                           skip_reason="; ".join(gate.active_processes) or None)
                rows.append(row)  # recorded but NOT checkpointed as done: a resume retries it
                print(f"GPU_BUSY {key}: {gate.reason}")
                exit_code = 3
                break
            print(f"RUN {key}: {shlex.join(_command(cfg, kv, depth))}", flush=True)
            row = run_case(cfg, kv, depth, measurer=measurer, baselines=baselines)
            if cfg.stop_after_fail and row["status"] in _FAILURE_STATUSES:
                failed_kv.add(kv)
        rows.append(row)
        done.append(key)
        elapsed = base_elapsed + (datetime.now(timezone.utc) - started).total_seconds()
        eta = elapsed / len(done) * (len(cases) - len(done)) if done else 0.0
        tg = row["tg_tps"]
        print(f"[{len(done)}/{len(cases)}] elapsed={elapsed:.0f}s eta={eta:.0f}s {key} status={row['status']} "
              f"pp={row['pp_tps']} tg={tg} vram_peak={row['vram_peak_mb']}MiB", flush=True)
        _atomic_write_text(checkpoint, json.dumps({
            "pipeline_schema_version": PIPELINE_SCHEMA_VERSION, "benchmark_fingerprint": fingerprint,
            "completed_index": len(done), "completed_items": done,
            "results": [r for r in rows if r["status"] != BenchStatus.GPU_BUSY.value],
            "elapsed_seconds": elapsed, "wall_clock": str(timedelta(seconds=int(elapsed))),
        }, ensure_ascii=False, indent=2) + "\n")

    elapsed = base_elapsed + (datetime.now(timezone.utc) - started).total_seconds()
    meta.update(elapsed_seconds=elapsed, wall_clock=str(timedelta(seconds=int(elapsed))), exit_code=exit_code)
    write_outputs(run_root, rows, cfg, meta)
    return rows, exit_code


# --------------------------------------------------------------------------------- CLI
def add_capacity_options(p: argparse.ArgumentParser) -> None:
    """Options shared by ``llama-tune-capacity`` and ``llama-tune-pipeline``."""

    p.add_argument("--depths", nargs="+", default=[",".join(map(str, DEFAULT_DEPTHS))],
                   help="context depths; accepts 4096 or 4k/128K/1m. Extended: 262k 384k 512k")
    p.add_argument("--kv", nargs="+", default=[",".join(DEFAULT_KV)], help="KV cache types (K and V set together)")
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--ubatch", type=int, default=512)
    p.add_argument("--flash-attn", type=int, default=1, choices=[0, 1])
    p.add_argument("--prompt", type=int, default=512)
    p.add_argument("--ngen", type=int, default=64)
    p.add_argument("--reps", type=int, default=1, help="llama-bench -r (repeats inside one process)")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--per-run-timeout", type=float, default=1500.0, help="seconds per benchmark process")
    p.add_argument("--gpu-index", type=int, default=None, help="measure exactly this GPU (required on multi-GPU hosts)")
    p.add_argument("--min-free-vram-gib", type=float, default=None, help="require this much free VRAM before each run")
    p.add_argument("--wait-for-gpu", action="store_true", help="poll until the GPU is free instead of stopping")
    p.add_argument("--gpu-wait-timeout", type=float, default=0.0, help="seconds to wait with --wait-for-gpu")
    p.add_argument("--gpu-poll", type=float, default=60.0)
    p.add_argument("--no-stop-after-fail", action="store_true", help="also try deeper depths after a failure")
    p.add_argument("--min-tg", type=float, default=PracticalCriteria.min_tg_tps)
    p.add_argument("--tg-ratio", type=float, default=PracticalCriteria.tg_ratio)
    p.add_argument("--pp-ratio", type=float, default=PracticalCriteria.pp_ratio)
    p.add_argument("--max-wall", type=float, default=None)
    p.add_argument("--vram-margin-mib", type=int, default=PracticalCriteria.vram_margin_mib)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Capacity / compatibility sweep: context depth x KV type, one process per point")
    p.add_argument("--llama-bench", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    add_capacity_options(p)
    p.add_argument("--out-dir", type=Path, default=Path("outfile"))
    p.add_argument("--tmp-dir", type=Path, default=Path("tmp"))
    p.add_argument("--run-dir", type=Path, default=None, help="stable run directory (required with --resume)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    return p.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> CapacityConfig:
    return CapacityConfig(
        llama_bench=args.llama_bench, model=args.model, depths=tuple(sorted(set(parse_int_list(args.depths)))),
        kv_types=parse_str_list(args.kv), ngl=args.ngl, batch=args.batch, ubatch=args.ubatch,
        flash_attn=args.flash_attn, prompt=args.prompt, ngen=args.ngen, repetitions=args.reps,
        threads=args.threads, per_run_timeout=args.per_run_timeout, gpu_index=args.gpu_index,
        min_free_vram_mib=int(args.min_free_vram_gib * 1024) if args.min_free_vram_gib else None,
        wait_for_gpu=args.wait_for_gpu, gpu_wait_timeout=args.gpu_wait_timeout, gpu_poll_s=args.gpu_poll,
        stop_after_fail=not args.no_stop_after_fail,
        criteria=PracticalCriteria(min_tg_tps=args.min_tg, tg_ratio=args.tg_ratio, pp_ratio=args.pp_ratio,
                                   max_wall_s=args.max_wall, vram_margin_mib=args.vram_margin_mib),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    cfg = config_from_args(args)
    if not cfg.llama_bench.exists():
        raise SystemExit(f"[FATAL] llama-bench not found: {cfg.llama_bench}")
    if not cfg.model.exists():
        raise SystemExit(f"[FATAL] model not found: {cfg.model}")
    if args.resume and args.run_dir is None:
        raise SystemExit("[FATAL] --resume requires --run-dir")
    if args.dry_run:
        for kv in cfg.kv_types:
            for depth in cfg.depths:
                print(shlex.join(_command(cfg, kv, depth)))
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.run_dir or (args.out_dir / "capacity" / stamp)
    if run_root.exists() and any(run_root.iterdir()) and not args.resume:
        raise SystemExit(f"[FATAL] Run directory is not empty; use --resume: {run_root}")
    log_root = args.tmp_dir / "capacity" / run_root.name
    backend = detect_backend()
    rows, code = run_capacity(cfg, run_root, log_root, backend=backend, resume=args.resume)
    print(f"capacity.json: {run_root / 'capacity.json'}")
    print(f"capacity.csv:  {run_root / 'capacity.csv'}")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
