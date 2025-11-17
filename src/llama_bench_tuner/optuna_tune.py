# src/llama_bench_tuner/optuna_tune.py
import argparse
import csv
import json
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import optuna

from .parsing import extract_tps_from_csv


@dataclass
class BenchArgs:
    llama_bench: Path
    model: Path
    threads: int
    prompt: int
    ngen: int
    mmap: int
    nkvo: Optional[int]
    split_mode: Optional[str]
    ub_ratio: float
    flash_attn: Tuple[int, ...]
    ngl_min: int
    ngl_max: int
    batch_min: int
    batch_max: int
    n_trials: int
    pruner: str
    seed: Optional[int]
    timeout_per_trial: Optional[int]
    storage: Optional[str]         # e.g., "sqlite:///outfile/optuna_study.db"
    study_name: str
    out_dir: Path
    tmp_dir: Path
    best: Optional[Path]


def parse_args() -> BenchArgs:
    p = argparse.ArgumentParser(description="Optuna tuner for llama.cpp (llama-bench)")
    p.add_argument("--llama-bench", type=Path, required=True, help="Path to llama-bench binary")
    p.add_argument("--model", type=Path, required=True, help="Path to GGUF model")
    p.add_argument("--threads", type=int, default=14)
    p.add_argument("--prompt", type=int, default=2048)
    p.add_argument("--ngen", type=int, default=256)
    p.add_argument("--mmap", type=int, default=1, help="-mmp 1/0")

    p.add_argument("--nkvo", type=int, default=None, help="--no-kv-offload (0/1); None to skip")
    p.add_argument("--split-mode", type=str, default=None, choices=[None,"none","layer","row"])
    p.add_argument("--ub-ratio", type=float, default=2.0, help="ubatch = max(1, int(batch/ratio))")
    p.add_argument(
        "--flash-attn",
        nargs="*",
        default=["0", "1"],
        help="candidate FA values, e.g. 0 1",
    )

    # Default ranges target a slightly wider search than the previous 16-28 / 8-12 grid
    p.add_argument("--ngl-min", type=int, default=14)
    p.add_argument("--ngl-max", type=int, default=32)
    p.add_argument("--batch-min", type=int, default=6)
    p.add_argument("--batch-max", type=int, default=16)

    p.add_argument("--n-trials", type=int, default=16)
    p.add_argument("--pruner", type=str, default="median", choices=["none","median","asha"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--timeout-per-trial", type=int, default=None, help="seconds; kill trial if exceeded")

    p.add_argument("--storage", type=str, default="sqlite:///outfile/optuna_study.db",
                   help="Optuna storage URL; set to none to disable persistent storage")
    p.add_argument("--study-name", type=str, default="llama_bench_optuna")

    p.add_argument("--out-dir", type=Path, default=Path("outfile"))
    p.add_argument("--tmp-dir", type=Path, default=Path("tmp"))
    p.add_argument("--best", type=Path, default=None, help="Existing best JSON for comparison/copy")
    args = p.parse_args()

    fa_tuple = tuple(int(x) for x in args.flash_attn)
    storage = None if args.storage in (None,"none","") else args.storage

    return BenchArgs(
        llama_bench=args.llama_bench,
        model=args.model,
        threads=args.threads,
        prompt=args.prompt,
        ngen=args.ngen,
        mmap=args.mmap,
        nkvo=args.nkvo,
        split_mode=args.split_mode,
        ub_ratio=args.ub_ratio,
        flash_attn=fa_tuple,
        ngl_min=args.ngl_min,
        ngl_max=args.ngl_max,
        batch_min=args.batch_min,
        batch_max=args.batch_max,
        n_trials=args.n_trials,
        pruner=args.pruner,
        seed=args.seed,
        timeout_per_trial=args.timeout_per_trial,
        storage=storage,
        study_name=args.study_name,
        out_dir=args.out_dir,
        tmp_dir=args.tmp_dir,
        best=args.best,
    )


def ensure_dirs(*paths: Path):
    for d in paths:
        d.mkdir(parents=True, exist_ok=True)


def _slugify(name: str) -> str:
    slug = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name).strip("_")
    return slug or "run"


def run_llama_bench(
    args: BenchArgs,
    ngl: int,
    b: int,
    fa: int,
    trial_tag: str,
    raw_dir: Path,
    log_dir: Path,
) -> Tuple[bool, float, float, str, str]:
    """Run one llama-bench and return (ok, decode_tps, prefill_tps, csv_rel, stderr_rel)."""
    ub = max(1, int(round(b / args.ub_ratio)))
    tag = f"optuna_{trial_tag}_ngl{ngl}_p{args.prompt}_n{args.ngen}_b{b}ub{ub}_fa{fa}_mmp{args.mmap}"
    csv_path = raw_dir / f"bench_{tag}.csv"
    err_path = log_dir / f"bench_{tag}.stderr.txt"

    cmd = [
        str(args.llama_bench),
        "-m", str(args.model),
        "-t", str(args.threads),
        "-ngl", str(ngl),
        "-b", str(b),
        "-ub", str(ub),
        "-p", str(args.prompt),
        "-n", str(args.ngen),
        "-mmp", str(args.mmap),
        "-o", "csv",
    ]
    if args.nkvo is not None:
        cmd += ["-nkvo", str(args.nkvo)]
    if args.split_mode:
        cmd += ["-sm", args.split_mode]
    if fa is not None:
        cmd += ["-fa", str(fa)]

    print(f"[RUN] {shlex.join(cmd)}")
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=args.timeout_per_trial if args.timeout_per_trial else None
        )
        stdout, stderr = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        stderr_raw = exc.stderr or ""
        if isinstance(stderr_raw, bytes):
            stderr_raw = stderr_raw.decode("utf-8", errors="replace")
        stderr = stderr_raw + "\n[timeout] llama-bench exceeded timeout"
        if stdout:
            csv_path.write_text(stdout)
        err_path.write_text(stderr.strip() or "timeout with no stderr")
        csv_rel = (
            csv_path.relative_to(raw_dir.parent if raw_dir.parent != raw_dir else raw_dir).as_posix()
            if csv_path.exists()
            else ""
        )
        err_rel = err_path.relative_to(log_dir.parent if log_dir.parent != log_dir else log_dir).as_posix()
        return False, 0.0, 0.0, csv_rel, err_rel

    csv_path.write_text(stdout)
    if stderr.strip():
        err_path.write_text(stderr)

    prefill_tps, decode_tps = extract_tps_from_csv(stdout.splitlines())

    ok = (decode_tps or 0.0) > 0.0
    csv_rel = csv_path.relative_to(raw_dir.parent if raw_dir.parent != raw_dir else raw_dir).as_posix()
    err_rel = (
        err_path.relative_to(log_dir.parent if log_dir.parent != log_dir else log_dir).as_posix()
        if err_path.exists()
        else ""
    )
    return ok, (decode_tps or 0.0), (prefill_tps or 0.0), csv_rel, err_rel


def build_pruner(name: str):
    if name == "none":
        return optuna.pruners.NopPruner()
    if name == "asha":
        # ASHA works well when you can compute intermediate values; we only have final tps,
        # but it still can early-stop by step if we add reports later.
        return optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=3)
    # default: median
    return optuna.pruners.MedianPruner(n_warmup_steps=3)


def objective(
    trial: optuna.trial.Trial,
    args: BenchArgs,
    raw_dir: Path,
    log_dir: Path,
) -> float:
    # Search space
    ngl = trial.suggest_int("ngl", args.ngl_min, args.ngl_max, step=2)   # even steps to stay near 16/20/24/28
    b   = trial.suggest_int("batch", args.batch_min, args.batch_max, step=4)  # 8,12...
    fa  = trial.suggest_categorical("fa", list(args.flash_attn)) if args.flash_attn else 0

    trial_tag = f"trial{trial.number:04d}"
    ok, decode_tps, prefill_tps, csv_name, stderr_name = run_llama_bench(args, ngl, b, fa, trial_tag, raw_dir, log_dir)

    # Attach trial user attrs for later inspection
    trial.set_user_attr("csv", csv_name)
    trial.set_user_attr("stderr", stderr_name)
    trial.set_user_attr("prefill_tps", prefill_tps)

    if not ok:
        # Penalize failed run
        return 0.0

    # Report final value (higher is better)
    return float(decode_tps)


def main():
    args = parse_args()
    if not args.llama_bench.exists():
        raise SystemExit(f"[FATAL] llama-bench not found: {args.llama_bench}")
    if not args.model.exists():
        raise SystemExit(f"[FATAL] model not found: {args.model}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    study_slug = _slugify(args.study_name)
    run_root = args.out_dir / "optuna" / f"{timestamp}_{study_slug}"
    raw_dir = run_root / "raw"
    log_root = args.tmp_dir / "optuna" / f"{timestamp}_{study_slug}"

    ensure_dirs(args.out_dir, args.tmp_dir, run_root, raw_dir, log_root)

    # Create study
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = build_pruner(args.pruner)
    study_kwargs = dict(direction="maximize", sampler=sampler, pruner=pruner)

    if args.storage:
        study = optuna.create_study(study_name=args.study_name, storage=args.storage, load_if_exists=True, **study_kwargs)
    else:
        study = optuna.create_study(**study_kwargs)

    print(f"[INFO] Storage: {args.storage or '(in-memory)'} / Study: {study.study_name}")
    print(f"[INFO] Search space: ngl=[{args.ngl_min},{args.ngl_max}], batch=[{args.batch_min},{args.batch_max}], fa={args.flash_attn}")
    print(f"[INFO] Outputs: {run_root}")

    study.optimize(lambda t: objective(t, args, raw_dir, log_root), n_trials=args.n_trials, show_progress_bar=True)

    # Save best result
    best = study.best_trial
    best_payload = {
        "value": best.value,
        "params": best.params,
        "user_attrs": best.user_attrs,
        "datetime": datetime.now().isoformat(timespec="seconds"),
    }
    best_json = run_root / "optuna_best.json"
    best_json.write_text(json.dumps(best_payload, indent=2))

    best_info = best_payload
    if args.best and args.best.exists():
        try:
            best_info = json.loads(args.best.read_text())
        except Exception:
            best_info = best_payload

    (run_root / "optuna_best_copy.json").write_text(json.dumps(best_info, indent=2))

    # Persist all trials to CSV (nice to have)
    trials_csv = run_root / "optuna_trials.csv"
    with trials_csv.open("w", newline="") as f:
        cols = ["number","value","state","ngl","batch","fa","prefill_tps","csv","stderr"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for t in study.trials:
            row = {
                "number": t.number,
                "value": t.value,
                "state": str(t.state),
                "ngl": t.params.get("ngl"),
                "batch": t.params.get("batch"),
                "fa": t.params.get("fa"),
                "prefill_tps": t.user_attrs.get("prefill_tps"),
                "csv": t.user_attrs.get("csv"),
                "stderr": t.user_attrs.get("stderr"),
            }
            w.writerow(row)

    print(f"[OK] Best saved to: {best_json}")
    print(f"[OK] Trials saved to: {trials_csv}")
    print(f"[OK] You can resume with the same --storage and --study-name.")


if __name__ == "__main__":
    main()
