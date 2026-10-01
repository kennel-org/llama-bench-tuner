"""``llama-tune-pipeline`` — capacity -> grid -> optuna -> validate -> profile.

Each stage can run on its own (``llama-tune-pipeline grid ...``) or all together (``all``).
Every stage keeps one overwritten checkpoint; with ``--resume`` finished stages are skipped and
interrupted ones continue. Stage outputs live under ``<out-dir>/pipeline/<name>/``::

    capacity/    capacity.json, capacity.csv, checkpoint.json, raw/
    grid/        grid.json, grid.csv, checkpoint.json, raw/
    optuna/d<N>/ optuna.json, optuna_rows.csv, study.db, trial_rows.jsonl, raw/
    validation/  validation.json, validation.csv, checkpoint.json, raw/
    profiles/    <host>-gpu<N>-<model>-{fast,balanced,long}.json
    pareto/      pareto_d<N>.json / .csv
    failure_summary.json
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from .capacity import (CapacityConfig, PracticalCriteria, add_capacity_options, config_from_args, parse_depth, parse_int_list,
                       parse_str_list, run_capacity)
from .gpu_gate import wait_for_gpu
from .grid_stage import load_space, run_grid
from .measure import BenchPoint, GpuBusyError, GpuFaultError
from .optuna_mo import DEFAULT_OBJECTIVES, PrunePolicy, refs_from_grid, run_study
from .profiles import write_profiles
from .stage_common import Context, build_context, load_capacity, read_json
from .telemetry import detect_backend
from .validation import run_validation

STAGES = ("capacity", "grid", "optuna", "validate", "profile")
OPTIONAL_STAGES = ("server", "soak")  # need a real llama-server; run explicitly, not part of ``all``


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llama-tune-pipeline", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=[*STAGES, *OPTIONAL_STAGES, "all"])
    p.add_argument("--llama-bench", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--name", default=None, help="run name (default: timestamp); the run root is <out-dir>/pipeline/<name>")
    p.add_argument("--out-dir", type=Path, default=Path("outfile"))
    p.add_argument("--tmp-dir", type=Path, default=Path("tmp"))
    p.add_argument("--resume", action="store_true", help="skip finished stages, continue interrupted ones")
    add_capacity_options(p)
    g = p.add_argument_group("grid")
    g.add_argument("--grid-space", type=Path, default=None, help="JSON {axis: [values]} (ngl,batch,ubatch,flash_attn,n_cpu_moe,...)")
    g.add_argument("--grid-depths", nargs="+", default=["8k,32k,64k,128k"], help="depths the Grid may use (must survive capacity)")
    g.add_argument("--grid-kv", nargs="+", default=None)
    g.add_argument("--include-capacity-only", action="store_true", help="also grid over completed-but-not-practical points")
    g.add_argument("--max-cases", type=int, default=240)
    o = p.add_argument_group("optuna")
    o.add_argument("--optuna-depths", nargs="+", default=["32k"], help="one study (Pareto set) per depth")
    o.add_argument("--optuna-space", type=Path, default=None, help="override the narrowed space from grid.json")
    o.add_argument("--trials", type=int, default=24)
    o.add_argument("--mode", default="pareto", help="pareto | score:fast | score:balanced | score:long")
    o.add_argument("--objectives", nargs="+", default=list(DEFAULT_OBJECTIVES))
    o.add_argument("--seed", type=int, default=0)
    o.add_argument("--population", type=int, default=None)
    o.add_argument("--prune-min-completed", type=int, default=PrunePolicy.min_completed)
    o.add_argument("--prune-tg-ratio", type=float, default=PrunePolicy.tg_ratio)
    v = p.add_argument_group("validation / profiles")
    v.add_argument("--validate-depths", nargs="+", default=["8k,32k,64k,128k"])
    v.add_argument("--validate-reps", type=int, default=3)
    v.add_argument("--validate-reps-deep", type=int, default=None,
                   help="repeats for depths >= 128K (default: same as --validate-reps); the used value is recorded")
    v.add_argument("--top-k", type=int, default=4)
    v.add_argument("--cv-limit", type=float, default=0.05)
    v.add_argument("--soak-ngen", type=int, default=0, help="extra long decode at the deepest practical depth (0 = off)")
    v.add_argument("--fast-depth", type=parse_depth, default=8192, help="e.g. 8192 or 8k")
    v.add_argument("--balanced-depth", type=parse_depth, default=32768, help="e.g. 32768 or 32k")
    sv = p.add_argument_group("server / soak (stages `server`, `soak`; need llama-server beside llama-bench)")
    sv.add_argument("--server-sizes", nargs="+", default=["4k,16k,32k"], help="prompt sizes for measured TTFT")
    sv.add_argument("--server-reps", type=int, default=3)
    sv.add_argument("--server-max-tokens", type=int, default=128)
    sv.add_argument("--server-profiles", nargs="+", default=None, choices=["fast", "balanced", "long"])
    sv.add_argument("--spec-compare", action="store_true", help="also run each profile with speculative decoding and compare")
    sv.add_argument("--spec-type", default="draft-mtp")
    sv.add_argument("--spec-draft-n-max", type=int, default=3)
    sv.add_argument("--server-startup-timeout", type=float, default=900.0)
    sv.add_argument("--soak-profile", default="balanced", choices=["fast", "balanced", "long"])
    sv.add_argument("--soak-minutes", type=float, default=15.0)
    sv.add_argument("--soak-window", type=float, default=30.0, help="seconds per decode-speed window")
    return p


def _context(args: argparse.Namespace) -> Context:
    backend = detect_backend()
    ctx = build_context(args.llama_bench, args.model, gpu_index=args.gpu_index, timeout=args.per_run_timeout,
                        retry_aborts=args.retry_aborts, backend=backend, detect=False)
    threshold = int(args.min_free_vram_gib * 1024) if args.min_free_vram_gib else None
    if threshold and backend is not None:
        ctx.gate = lambda: wait_for_gpu(backend, min_free_mib=threshold, gpu_index=ctx.gpu_index,
                                        wait=args.wait_for_gpu, timeout_s=args.gpu_wait_timeout,
                                        poll_s=args.gpu_poll)
    return ctx


def _base(args: argparse.Namespace) -> BenchPoint:
    return BenchPoint(ngl=args.ngl, batch=args.batch, ubatch=args.ubatch, flash_attn=args.flash_attn,
                      prompt=args.prompt, ngen=args.ngen, reps=args.reps, threads=args.threads,
                      n_cpu_moe=args.n_cpu_moe)


def _guard(out: Path, marker: str, resume: bool, label: str) -> bool:
    """True when the stage is already complete and should be skipped."""

    if (out / marker).exists():
        if resume:
            print(f"SKIP {label}: {out / marker} already exists (--resume)")
            return True
        raise SystemExit(f"[FATAL] {out / marker} exists; use --resume or a new --name.")
    return False


def stage_capacity(args, ctx: Context, root: Path) -> None:
    out = root / "capacity"
    if _guard(out, "capacity.json", args.resume, "capacity"):
        return
    if (out / "checkpoint.json").exists() and not args.resume:
        raise SystemExit(f"[FATAL] {out / 'checkpoint.json'} exists (interrupted run); use --resume or a new --name.")
    cfg = config_from_args(args)
    _, code = run_capacity(cfg, out, args.tmp_dir / "pipeline" / root.name / "capacity",
                           backend=ctx.backend, resume=args.resume, caps=ctx.caps)
    if code:
        raise SystemExit(code)


def stage_grid(args, ctx: Context, root: Path) -> None:
    out = root / "grid"
    if _guard(out, "grid.json", args.resume, "grid"):
        return
    run_grid(ctx, root / "capacity" / "capacity.json", out, args.tmp_dir / "pipeline" / root.name / "grid",
             space=load_space(args.grid_space), depths=parse_int_list(args.grid_depths),
             kv_filter=parse_str_list(args.grid_kv) if args.grid_kv else None,
             include_capacity_only=args.include_capacity_only, base=_base(args), max_cases=args.max_cases,
             resume=args.resume)


def stage_optuna(args, ctx: Context, root: Path) -> None:
    capacity = load_capacity(root / "capacity" / "capacity.json")
    grid_path = root / "grid" / "grid.json"
    space = load_space(args.optuna_space) if args.optuna_space else (
        read_json(grid_path, "grid.json").get("next_space", {}) if grid_path.exists() else {})
    policy = PrunePolicy(min_completed=args.prune_min_completed, tg_ratio=args.prune_tg_ratio)
    for depth in parse_int_list(args.optuna_depths):
        usable = sorted({s["kv"] for s in capacity["survivors"] if s["depth"] == depth})
        if not usable:
            print(f"SKIP optuna d={depth}: no KV type is a practical candidate at this depth")
            continue
        study_space = {**space, "kv": [k for k in (space.get("kv") or usable) if k in usable] or usable}
        out = root / "optuna" / f"d{depth}"
        if _guard(out, "optuna.json", args.resume, f"optuna d={depth}"):
            continue
        if (out / "study.db").exists() and not args.resume:
            raise SystemExit(f"[FATAL] {out / 'study.db'} exists; use --resume or a new --name.")
        measurer = ctx.measurer(out, args.tmp_dir / "pipeline" / root.name / f"optuna_d{depth}")
        summary = run_study(ctx, measurer, out, space=study_space, base=_base(args), depth=depth,
                            n_trials=args.trials, mode=args.mode, objectives=args.objectives, policy=policy,
                            seed=args.seed, population=args.population,
                            refs=refs_from_grid(grid_path, depth), study_name=f"{root.name}_d{depth}")
        print(f"optuna d={depth}: {summary['trial_states']} pareto/best={len(summary['best'])}")


def stage_validate(args, ctx: Context, root: Path) -> None:
    out = root / "validation"
    if _guard(out, "validation.json", args.resume, "validation"):
        return
    crit = config_from_args(args).criteria
    run_validation(ctx, root, root / "capacity" / "capacity.json", out,
                   args.tmp_dir / "pipeline" / root.name / "validation", depths=parse_int_list(args.validate_depths),
                   reps=args.validate_reps, top_k=args.top_k, criteria=crit, cv_limit=args.cv_limit,
                   soak_ngen=args.soak_ngen, base=_base(args), resume=args.resume,
                   reps_deep=args.validate_reps_deep)


def stage_profile(args, ctx: Context, root: Path) -> None:
    result = write_profiles(ctx, root, root / "validation" / "validation.json", fast_depth=args.fast_depth,
                            balanced_depth=args.balanced_depth)
    for name, path in result["profiles"].items():
        print(f"profile {name}: {path}")
    print(f"pareto depths: {result['pareto_depths']}")


def stage_server(args, ctx: Context, root: Path) -> None:
    from .server_stage import run_server_stage
    report = run_server_stage(
        ctx, root, root / "server", args.tmp_dir / "pipeline" / root.name / "server",
        sizes=parse_int_list(args.server_sizes), reps=args.server_reps, max_tokens=args.server_max_tokens,
        spec_compare=args.spec_compare, spec_type=args.spec_type, spec_draft_n_max=args.spec_draft_n_max,
        startup_timeout=args.server_startup_timeout, only=args.server_profiles)
    for name, entry in report["profiles"].items():
        spec = entry["speculation"].get("status")
        print(f"server {name}: sizes={entry['sizes']} speculation={spec}")


def stage_soak(args, ctx: Context, root: Path) -> None:
    from .server_stage import run_soak_stage
    result = run_soak_stage(ctx, root, root / "soak", args.tmp_dir / "pipeline" / root.name / "soak",
                            profile_name=args.soak_profile, minutes=args.soak_minutes, window_s=args.soak_window,
                            startup_timeout=args.server_startup_timeout)
    print(f"soak {args.soak_profile}: verdict={result['verdict']}")


_RUNNERS = {"server": stage_server, "soak": stage_soak,
            "capacity": stage_capacity, "grid": stage_grid, "optuna": stage_optuna, "validate": stage_validate,
            "profile": stage_profile}


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.llama_bench.exists():
        raise SystemExit(f"[FATAL] llama-bench not found: {args.llama_bench}")
    if not args.model.exists():
        raise SystemExit(f"[FATAL] model not found: {args.model}")
    name = args.name or datetime.now().strftime("%Y%m%d_%H%M%S")
    root = args.out_dir / "pipeline" / name
    root.mkdir(parents=True, exist_ok=True)
    ctx = _context(args)
    stages = STAGES if args.stage == "all" else (args.stage,)
    try:
        for stage in stages:
            print(f"=== stage: {stage} ===", flush=True)
            _RUNNERS[stage](args, ctx, root)
    except GpuFaultError as exc:
        print(f"GPU_FAULT: {exc}", file=sys.stderr)
        raise SystemExit(4)
    except GpuBusyError as exc:
        print(f"GPU_BUSY: {exc}; progress is checkpointed — rerun with --resume", file=sys.stderr)
        raise SystemExit(3)
    print(f"run root: {root}")


def profile_main() -> None:
    """Entry point ``llama-tune-profile`` = ``llama-tune-pipeline profile``."""

    main(["profile", *sys.argv[1:]])


if __name__ == "__main__":
    main()
