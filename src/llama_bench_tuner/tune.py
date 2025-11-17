# src/llama_bench_tuner/tune.py
import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from rich import print

from .parsing import extract_tps_from_csv

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

def run_once(args, ngl:int, b:int, fa:int, raw_dir:Path, log_dir:Path):
    """Run llama-bench once and return decode/prefill tps. Saves raw CSV and STDERR."""
    ub = max(1, int(round(b / args.ub_ratio)))
    tag = f"ngl{ngl}_p{args.prompt}_n{args.ngen}_b{b}ub{ub}_fa{fa}_mmp{args.mmap}"
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
        "-v",
    ]
    if fa is not None:
        cmd += ["-fa", str(fa)]
    if args.nkvo is not None:
        cmd += ["-nkvo", str(args.nkvo)]
    if args.split_mode:
        cmd += ["-sm", args.split_mode]

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

    prefill_tps, decode_tps = extract_tps_from_csv(stdout.splitlines())

    csv_rel = csv_path.relative_to(raw_dir.parent if raw_dir.parent != raw_dir else raw_dir).as_posix()
    err_rel = err_path.relative_to(log_dir.parent if log_dir.parent != log_dir else log_dir).as_posix() if err_path.exists() else ""
    ok = (decode_tps or 0.0) > 0.0
    return {
        "ok": ok,
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
    }

def main():
    args = apply_space_file(parse_args())
    if args.ngl is None or args.batch is None:
        raise SystemExit("[FATAL] --ngl and --batch must be provided via CLI or --space-file")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    run_root = args.out_dir / "grid" / timestamp
    raw_dir = run_root / "raw"
    summary_path = run_root / f"summary_{timestamp}.csv"

    log_root = args.tmp_dir / "grid" / timestamp

    ensure_dirs(args.in_dir, run_root, raw_dir, log_root)

    fields = ["ok","ngl","b","ub","fa","decode_tps","prefill_tps","csv","stderr","start","end","elapsed_sec"]
    rows = []

    for ngl in [int(x) for x in args.ngl]:
        for b in [int(x) for x in args.batch]:
            for fa in [int(x) for x in args.flash_attn]:
                if skip_due_to_wsl_limits(args, ngl, b, fa):
                    print(
                        f"[yellow]SKIP[/yellow] ngl={ngl}, b={b}, fa={fa}: exceeds ~20GB VRAM on WSL. "
                        "Use --allow-wsl-unsafe to force."
                    )
                    continue
                res = run_once(args, ngl, b, fa, raw_dir, log_root)
                rows.append(res)
                print(f"[green]OK={res['ok']}[/green] decode={res['decode_tps']:.2f} prefill={res['prefill_tps']:.2f} "
                      f"(ngl={ngl}, b={b}, ub={res['ub']}, fa={fa})")

    # Save summary
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # Print best row
    goods = [r for r in rows if r["ok"]]
    if goods:
        best = sorted(goods, key=lambda r: (-r["decode_tps"], -r["prefill_tps"]))[0]
        print("\n[bold magenta]=== BEST CONFIG ===[/bold magenta]")
        print(best)
        print(f"[bold]Summary saved:[/bold] {summary_path}")
    else:
        print("\n[red]No successful rows. Check stderr logs under tmp/[/red]")

if __name__ == "__main__":
    main()

