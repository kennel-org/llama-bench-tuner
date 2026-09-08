# src/llama_bench_tuner/tune.py
import argparse
import csv
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta
from rich import print

from .command_builder import (
    LlamaBenchCommand,
    build_llama_bench_command,
    native_reported_offload_json,
    native_reported_placement_json,
    requested_offload_json,
    requested_placement_json,
)
from .parsing import extract_tps_from_rows, parse_bench_csv
from .schema import SCHEMA_VERSION, csv_fields, enrich_result
from .status import BenchStatus, classify_bench_outcome

def parse_args():
    p = argparse.ArgumentParser(description="Grid tuner for llama-bench")
    p.add_argument("--llama-bench", type=Path, required=True, help="Path to llama-bench binary")
    p.add_argument("--model", type=Path, required=True, help="Path to GGUF model file")
    p.add_argument("--threads", type=int, default=14)
    p.add_argument("--prompt", type=int, default=2048)
    p.add_argument("--ngen", type=int, default=256)
    p.add_argument("--mmap", type=int, default=1)
    p.add_argument("--flash-attn", nargs="*", default=["0"], help="List of 0/1 values")
    p.add_argument("--ngl", nargs="+", default=None, help="List of ngl values")
    p.add_argument("--batch", nargs="+", default=None, help="List of batch sizes")
    p.add_argument("--ub-ratio", type=float, default=2.0, help="ubatch = max(1, int(batch/ratio))")
    p.add_argument("--nkvo", type=int, default=None, help="--no-kv-offload (0/1); None to skip")
    p.add_argument("--split-mode", type=str, default=None, choices=[None, "none", "layer", "row"])
    p.add_argument("--in-dir", type=Path, default=Path("infile"))
    p.add_argument("--out-dir", type=Path, default=Path("outfile"))
    p.add_argument("--tmp-dir", type=Path, default=Path("tmp"))
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Use a stable run directory; required with --resume",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume completed cases from checkpoint.json in --run-dir",
    )
    p.add_argument(
        "--space-file",
        type=Path,
        default=None,
        help="Path to JSON file defining ngl/batch/flash_attn arrays",
    )
    p.add_argument(
        "--allow-wsl-unsafe",
        action="store_true",
        help="Run configs that are known to crash 20GB-class GPUs under WSL",
    )
    return p.parse_args()


def load_space_from_file(space_path: Path) -> dict:
    try:
        data = json.loads(space_path.read_text())
    except Exception as exc:
        raise SystemExit(f"[FATAL] Failed to read space file {space_path}: {exc}")
    if not isinstance(data, dict):
        raise SystemExit(f"[FATAL] Space file {space_path} must contain a JSON object")
    return data


def apply_space_file(args):
    if args.space_file is None:
        return args
    space = load_space_from_file(args.space_file)

    def _get_list(key, current):
        values = space.get(key, current)
        if values is None:
            return None
        if not isinstance(values, list):
            raise SystemExit(f"[FATAL] Space file key '{key}' must be a list")
        return [str(v) for v in values]

    args.ngl = _get_list("ngl", args.ngl)
    args.batch = _get_list("batch", args.batch)
    args.flash_attn = _get_list("flash_attn", args.flash_attn)
    if not args.ngl or not args.batch or not args.flash_attn:
        raise SystemExit("[FATAL] Space file must provide ngl, batch, and flash_attn arrays")
    return args


def running_in_wsl() -> bool:
    return "WSL_DISTRO_NAME" in os.environ or Path("/proc/sys/fs/binfmt_misc/WSLInterop").exists()


def skip_due_to_wsl_limits(args, ngl: int, b: int, fa: int) -> bool:
    if args.allow_wsl_unsafe or not running_in_wsl():
        return False
    if ngl < 32:
        return False
    if fa == 1 and b >= 8:
        return True
    if fa == 0 and b >= 12:
        return True
    return False

def ensure_dirs(*paths: Path):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def case_key(ngl: int, b: int, fa: int) -> str:
    return f"ngl={ngl},b={b},fa={fa}"


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path, *, hash_content: bool) -> dict:
    """Return a lightweight identity for a path used in the fingerprint.

    A resolved path alone does not detect a binary rebuild or a model
    replacement at the same path, so a checkpoint could silently mix results
    from different builds/weights. Binaries are small, so their full content
    is hashed; models can be tens of GB, so only cheap stat-based identity
    (size + mtime_ns) is used for them.
    """

    try:
        info = path.stat()
    except OSError:
        return {"exists": False}
    identity: dict[str, object] = {
        "exists": True,
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
    }
    if hash_content:
        identity["sha256"] = _sha256_file(path)
    return identity


def benchmark_definition(args, cases: list[tuple[int, int, int]]) -> dict:
    """Return every setting that makes a Grid result incomparable on resume."""

    return {
        "llama_bench": str(args.llama_bench.resolve()),
        "llama_bench_identity": _file_identity(args.llama_bench, hash_content=True),
        "model": str(args.model.resolve()),
        "model_identity": _file_identity(args.model, hash_content=False),
        "threads": args.threads,
        "prompt": args.prompt,
        "ngen": args.ngen,
        "mmap": args.mmap,
        "ub_ratio": args.ub_ratio,
        "nkvo": args.nkvo,
        "split_mode": args.split_mode,
        "allow_wsl_unsafe": args.allow_wsl_unsafe,
        "cases": cases,
    }


def benchmark_fingerprint(definition: dict) -> str:
    payload = json.dumps(definition, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_write_text(path: Path, content: str) -> None:
    """Replace a checkpoint atomically, preserving the old file on interruption."""

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def write_checkpoint(path: Path, *, completed_items: list[str], rows: list[dict],
                     started: datetime, benchmark_fingerprint_value: str,
                     base_elapsed: float = 0.0) -> None:
    elapsed = base_elapsed + (datetime.now(timezone.utc) - started).total_seconds()
    content = json.dumps({
        "schema_version": SCHEMA_VERSION,
        "benchmark_fingerprint": benchmark_fingerprint_value,
        "completed_index": len(completed_items),
        "completed_items": completed_items,
        "results": rows,
        "elapsed_seconds": elapsed,
        "wall_clock": str(timedelta(seconds=int(elapsed))),
    }, ensure_ascii=False, indent=2) + "\n"
    _atomic_write_text(path, content)


def _load_json(path: Path, label: str) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"[FATAL] Cannot resume: invalid {label} at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"[FATAL] Cannot resume: {label} at {path} is not a JSON object")
    return data


def _validate_resume_fingerprint(source: str, state: dict, expected: str) -> None:
    actual = state.get("benchmark_fingerprint")
    if actual != expected:
        raise SystemExit(
            f"[FATAL] Cannot resume: {source} benchmark fingerprint differs from this invocation. "
            "Use a new --run-dir for changed benchmark settings."
        )

def run_once(args, ngl:int, b:int, fa:int, raw_dir:Path, log_dir:Path):
    """Run llama-bench once and return decode/prefill tps. Saves raw CSV and STDERR."""
    ub = max(1, int(round(b / args.ub_ratio)))
    tag = f"ngl{ngl}_p{args.prompt}_n{args.ngen}_b{b}ub{ub}_fa{fa}_mmp{args.mmap}"
    csv_path = raw_dir / f"bench_{tag}.csv"
    err_path = log_dir / f"bench_{tag}.stderr.txt"

    spec = LlamaBenchCommand(
        llama_bench=args.llama_bench, model=args.model, threads=args.threads,
        ngl=ngl, batch=b, ubatch=ub, prompt=args.prompt, ngen=args.ngen,
        mmap=args.mmap, flash_attn=fa, nkvo=args.nkvo,
        split_mode=args.split_mode, verbose=True,
    )
    cmd = build_llama_bench_command(spec)

    print(f"[cyan]RUN[/cyan] {' '.join(cmd)}")
    start_dt = datetime.now(timezone.utc)
    start_iso = start_dt.isoformat(timespec="seconds")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except KeyboardInterrupt:
        raise SystemExit("Interrupted")
    end_dt = datetime.now(timezone.utc)
    end_iso = end_dt.isoformat(timespec="seconds")
    elapsed = (end_dt - start_dt).total_seconds()
    stdout, stderr = proc.stdout, proc.stderr
    csv_path.write_text(stdout)
    if stderr.strip():
        err_path.write_text(stderr)

    bench_rows = parse_bench_csv(stdout.splitlines())
    prefill_tps, decode_tps = extract_tps_from_rows(bench_rows)
    outcome = classify_bench_outcome(
        returncode=proc.returncode, decode_tps=decode_tps, stdout=stdout, stderr=stderr,
    )

    csv_rel = csv_path.relative_to(raw_dir.parent if raw_dir.parent != raw_dir else raw_dir).as_posix()
    err_rel = err_path.relative_to(log_dir.parent if log_dir.parent != log_dir else log_dir).as_posix() if err_path.exists() else ""
    return enrich_result({
        "ok": outcome.ok,
        "status": outcome.status.value,
        "ngl": ngl,
        "b": b,
        "ub": ub,
        "fa": fa,
        "decode_tps": decode_tps or 0.0,
        "prefill_tps": prefill_tps or 0.0,
        "csv": csv_rel,
        "stderr": err_rel,
        "start": start_iso,
        "end": end_iso,
        "elapsed_sec": round(elapsed, 3),
        "error": outcome.error,
    }, model=args.model, llama_bench=args.llama_bench, prompt=args.prompt, ngen=args.ngen,
       requested_offload=requested_offload_json(spec),
       requested_placement=requested_placement_json(spec),
       native_reported_offload=native_reported_offload_json(bench_rows),
       native_reported_placement=native_reported_placement_json(bench_rows))

def main():
    args = apply_space_file(parse_args())
    if args.ngl is None or args.batch is None:
        raise SystemExit("[FATAL] --ngl and --batch must be provided via CLI or --space-file")
    if args.resume and args.run_dir is None:
        raise SystemExit("[FATAL] --resume requires --run-dir")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_root = args.run_dir or (args.out_dir / "grid" / timestamp)
    raw_dir = run_root / "raw"
    summary_path = run_root / f"summary_{run_root.name}.csv"
    checkpoint_path = run_root / "checkpoint.json"
    metadata_path = run_root / "run_metadata.json"

    log_root = args.tmp_dir / "grid" / run_root.name

    if run_root.exists() and any(run_root.iterdir()) and not args.resume and args.run_dir:
        raise SystemExit(f"[FATAL] Run directory is not empty; use --resume: {run_root}")
    ensure_dirs(args.in_dir, run_root, raw_dir, log_root)

    cases = [
        (ngl, b, fa)
        for ngl in [int(x) for x in args.ngl]
        for b in [int(x) for x in args.batch]
        for fa in [int(x) for x in args.flash_attn]
    ]
    definition = benchmark_definition(args, cases)
    fingerprint = benchmark_fingerprint(definition)
    started = datetime.now(timezone.utc)
    base_elapsed = 0.0
    rows: list[dict] = []
    completed_items: list[str] = []
    if args.resume:
        if checkpoint_path.exists() and not metadata_path.exists():
            raise SystemExit(
                "[FATAL] Cannot resume: checkpoint exists without run_metadata.json; "
                "use a new --run-dir."
            )
        if metadata_path.exists():
            _validate_resume_fingerprint(
                "run metadata", _load_json(metadata_path, "run metadata"), fingerprint,
            )
        if checkpoint_path.exists():
            state = _load_json(checkpoint_path, "checkpoint")
            _validate_resume_fingerprint("checkpoint", state, fingerprint)
            rows = state.get("results", [])
            completed_items = state.get("completed_items", [])
            base_elapsed = float(state.get("elapsed_seconds", 0.0))
            print(f"[cyan]RESUME[/cyan] {len(completed_items)}/{len(cases)} cases from {checkpoint_path}")
        else:
            print(f"[cyan]RESUME[/cyan] no checkpoint found; starting a new run in {run_root}")

    if not (args.resume and metadata_path.exists()):
        metadata_path.write_text(json.dumps({
            "schema_version": SCHEMA_VERSION,
            "runner": "llama-tune",
            "started": started.isoformat(timespec="seconds"),
            "args": {key: str(value) for key, value in vars(args).items()},
            "benchmark_definition": definition,
            "benchmark_fingerprint": fingerprint,
        }, ensure_ascii=False, indent=2) + "\n")

    total = len(cases)
    for index, (ngl, b, fa) in enumerate(cases, 1):
        key = case_key(ngl, b, fa)
        if key in completed_items:
            continue
        if skip_due_to_wsl_limits(args, ngl, b, fa):
            res = enrich_result({
                "ok": False, "status": BenchStatus.SKIPPED.value, "ngl": ngl, "b": b,
                "ub": max(1, int(round(b / args.ub_ratio))), "fa": fa,
                "decode_tps": 0.0, "prefill_tps": 0.0,
                "csv": "", "stderr": "", "start": "", "end": "",
                "elapsed_sec": 0.0, "skip_reason": "known WSL VRAM safety limit",
                "error": "skipped by WSL VRAM safety limit",
            }, model=args.model, llama_bench=args.llama_bench,
               prompt=args.prompt, ngen=args.ngen,
               requested_offload=requested_offload_json(LlamaBenchCommand(
                   args.llama_bench, args.model, args.threads, ngl, b,
                   max(1, int(round(b / args.ub_ratio))), args.prompt, args.ngen,
                   args.mmap, fa, args.nkvo, args.split_mode,
               )),
               requested_placement=requested_placement_json(LlamaBenchCommand(
                   args.llama_bench, args.model, args.threads, ngl, b,
                   max(1, int(round(b / args.ub_ratio))), args.prompt, args.ngen,
                   args.mmap, fa, args.nkvo, args.split_mode,
               )))
            print(f"[yellow]SKIP[/yellow] {key}: exceeds ~20GB VRAM on WSL")
        else:
            res = run_once(args, ngl, b, fa, raw_dir, log_root)
            print(f"[green]OK={res['ok']}[/green] decode={res['decode_tps']:.2f} prefill={res['prefill_tps']:.2f} "
                  f"(ngl={ngl}, b={b}, ub={res['ub']}, fa={fa})")
        rows.append(res)
        completed_items.append(key)
        elapsed = base_elapsed + (datetime.now(timezone.utc) - started).total_seconds()
        eta = elapsed / len(completed_items) * (total - len(completed_items)) if completed_items else 0.0
        print(f"[{index}/{total}] elapsed={elapsed:.0f}s eta={eta:.0f}s item={key}", flush=True)
        write_checkpoint(
            checkpoint_path,
            completed_items=completed_items,
            rows=rows,
            started=started,
            benchmark_fingerprint_value=fingerprint,
            base_elapsed=base_elapsed,
        )

    # Save summary
    fields = csv_fields(["ok","ngl","b","ub","fa","decode_tps","prefill_tps","csv","stderr","start","end","elapsed_sec"])
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    elapsed = base_elapsed + (datetime.now(timezone.utc) - started).total_seconds()
    (run_root / "run_result.json").write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "runner": "llama-tune",
        "benchmark_fingerprint": fingerprint,
        "results": rows,
        "result_count": len(rows),
        "elapsed_seconds": elapsed,
        "wall_clock": str(timedelta(seconds=int(elapsed))),
    }, ensure_ascii=False, indent=2) + "\n")

    # Print best row
    goods = [r for r in rows if r["ok"]]
    if goods:
        best = sorted(goods, key=lambda r: (-r["decode_tps"], -r["prefill_tps"]))[0]
        print("\n[bold magenta]=== BEST CONFIG ===[/bold magenta]")
        print(best)
        print(f"[bold]Summary saved:[/bold] {summary_path}")
        print(f"[bold]Run elapsed:[/bold] {elapsed:.0f}s ({timedelta(seconds=int(elapsed))})")
    else:
        print("\n[red]No successful rows. Check stderr logs under tmp/[/red]")

if __name__ == "__main__":
    main()
