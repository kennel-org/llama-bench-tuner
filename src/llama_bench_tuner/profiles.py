"""Final stage: turn validated results into FAST / BALANCED / LONG-CONTEXT profiles + Pareto tables."""

from __future__ import annotations

import re
import shlex
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .measure import BenchPoint, point_command
from .optuna_mo import SCORE_WEIGHTS, use_case_score
from .pareto import pareto_by_group
from .schema import PIPELINE_SCHEMA_VERSION, environment_metadata
from .stage_common import Context, model_quant_from_name, read_json, write_json, write_rows_csv

PROFILES = ("fast", "balanced", "long")


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-").lower()


def server_fa_args(server: Path, flash_attn: int) -> list[str]:
    """Flash-attention flag in the syntax the *server* binary accepts (probed from its --help)."""

    try:
        text = subprocess.run([str(server), "--help"], capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.TimeoutExpired):
        text = ""
    if re.search(r"--flash-attn\s*\[?<?on\|off\|auto", text):
        return ["-fa", "on" if flash_attn else "off"]
    if "--flash-attn" in text:
        return ["-fa"] if flash_attn else []
    return ["-fa", "on" if flash_attn else "off"]  # unprobed: documented in the profile notes


def server_command(llama_bench: Path, model: Path, point: BenchPoint, context: int) -> tuple[list[str], bool]:
    """Exact ``llama-server`` command. Returns (argv, server_binary_found)."""

    server = llama_bench.with_name("llama-server")
    found = server.exists()
    cmd = [str(server) if found else "llama-server", "-m", str(model), "-ngl", str(point.ngl), "-c", str(context),
           "-ctk", point.kv, "-ctv", point.kv, "-b", str(point.batch), "-ub", str(point.ubatch), "-np", "1"]
    cmd += server_fa_args(server, point.flash_attn) if found else ["-fa", "on" if point.flash_attn else "off"]
    if point.n_cpu_moe is not None:
        cmd += ["--n-cpu-moe", str(point.n_cpu_moe)]
    if point.split_mode:
        cmd += ["-sm", point.split_mode]
    cmd += ["--jinja"]
    return cmd, found


def _entries(validation: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten validation.json into one entry per (candidate, depth>0)."""

    out = []
    for cand in validation["candidates"]:
        for depth, s in cand["depths"].items():
            if int(depth) == 0 or s["n_ok"] == 0:
                continue
            out.append({"cand": cand, "depth": int(depth), "s": s})
    return out


def _entry_row(e: dict[str, Any], total: Optional[int]) -> dict[str, Any]:
    s = e["s"]
    return {"tg_tps": s["tg"]["median"], "pp_tps": s["pp"]["median"], "ttft_est_s": s["ttft_est_s"],
            "vram_peak_mb": s["vram_peak_mb"], "vram_total_mb": total}


def select_profiles(validation: dict[str, Any], *, fast_depth: int, balanced_depth: int,
                    vram_total: Optional[int]) -> dict[str, Any]:
    """Pick one validated entry per profile. Only ``practical`` entries are eligible; each choice
    records why, and a profile with no eligible entry is reported as ``None`` with the reason."""

    entries = [e for e in _entries(validation) if e["s"]["practical"]]
    picks: dict[str, Any] = {}

    def best(profile: str, pool: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not pool:
            return None
        refs = {"tg": max(_entry_row(e, vram_total)["tg_tps"] for e in pool),
                "pp": max(_entry_row(e, vram_total)["pp_tps"] for e in pool)}
        ttfts = [_entry_row(e, vram_total)["ttft_est_s"] for e in pool if _entry_row(e, vram_total)["ttft_est_s"]]
        if ttfts:
            refs["ttft"] = max(ttfts)
        return max(pool, key=lambda e: use_case_score(profile, _entry_row(e, vram_total), refs))

    def nearest(depth: int, below: bool) -> Optional[int]:
        depths = sorted({e["depth"] for e in entries})
        cands = [d for d in depths if (d <= depth if below else d >= depth)]
        return (max(cands) if below else min(cands)) if cands else None

    fd = nearest(fast_depth, True) or (min((e["depth"] for e in entries), default=None))
    bd = nearest(balanced_depth, True) or nearest(balanced_depth, False)
    ld = max((e["depth"] for e in entries), default=None)
    for profile, depth in (("fast", fd), ("balanced", bd), ("long", ld)):
        pool = [e for e in entries if e["depth"] == depth] if depth else []
        chosen = best(profile, pool)
        picks[profile] = {"entry": chosen, "depth": depth, "requested_depth": {
            "fast": fast_depth, "balanced": balanced_depth, "long": "deepest practical"}[profile],
            "n_eligible": len(pool)}
    return picks


def build_profile(ctx: Context, name: str, pick: dict[str, Any], validation: dict[str, Any], paths: dict[str, str],
                  vram_total: Optional[int]) -> dict[str, Any]:
    env = environment_metadata()
    if pick["entry"] is None:
        return {"profile": name, "available": False, "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
                "reason": "no practical validated configuration at the required depth",
                "requested_depth": pick["requested_depth"]}
    e = pick["entry"]
    cfg = e["cand"]["config"]
    point = BenchPoint(**cfg).replace(depth=e["depth"])
    s = e["s"]
    server_cmd, server_found = server_command(ctx.llama_bench, ctx.model, point, e["depth"])
    quant = model_quant_from_name(ctx.model)
    st = ctx.model.stat() if ctx.model.exists() else None
    return {
        "profile": name, "available": True, "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": env["host"], "gpu": env["gpu"] or (ctx.gpu.name if ctx.gpu else None),
        "gpu_index": ctx.gpu_index, "gpu_connection": env["gpu_connection"] or None, "platform": env["platform"],
        "runtime": {"llama_bench": str(ctx.llama_bench), "driver_version": ctx.gpu.driver_version if ctx.gpu else None,
                    "llama_cpp_commit": ",".join(validation.get("runtime_commits") or []) or None,
                    "help_sha256": ctx.caps.help_sha256},
        "model": {"path": str(ctx.model), "size_bytes": st.st_size if st else None,
                  "mtime_ns": st.st_mtime_ns if st else None, "quant": quant,
                  "quant_source": "filename" if quant else None},
        "context": e["depth"], "context_note": "validated with `context` tokens already in the KV cache plus a 512-token "
                                               "prompt and 64 generated tokens (n_ctx = context + 576)",
        "kv_type_k": point.kv, "kv_type_v": point.kv, "batch": point.batch, "ubatch": point.ubatch,
        "flash_attn": point.flash_attn, "ngl": point.ngl, "n_cpu_moe": point.n_cpu_moe,
        "split_mode": point.split_mode, "mtp_speculation": None,
        "expected": {"pp_tps": s["pp"]["median"], "tg_tps": s["tg"]["median"], "tg_cv": s["tg"]["cv"],
                     "ttft_est_s": s["ttft_est_s"], "ttft_kind": "estimated_from_pp" if s["ttft_est_s"] else None,
                     "vram_peak_mb": s["vram_peak_mb"], "vram_total_mb": vram_total, "ram_peak_mb": s["ram_peak_mb"],
                     "validated_reps": s["n_ok"], "temp_max_c": s["temp_max_c"],
                     "sm_clock_min_mhz": s["sm_clock_min_mhz"], "throttle_reasons": s["throttle_reasons"]},
        "soak": e["cand"].get("soak"),
        "commands": {"llama_server": shlex.join(server_cmd), "llama_server_binary_found": server_found,
                     "llama_bench_reproduce": shlex.join(point_command(ctx.llama_bench, ctx.model, point.replace(reps=3)))},
        "notes": ["llama-server flags are taken from the validated llama-bench configuration; the server binary's "
                  "--help is probed only for the flash-attention syntax.",
                  "MTP / speculative decoding is not part of this profile (llama-bench has no such option)."],
        "evidence": {**paths, "criteria": validation["criteria"], "reps": validation["reps"],
                     "selection": f"best {name} score among practical entries at depth {e['depth']} "
                                  f"({pick['n_eligible']} eligible); requested {pick['requested_depth']}"},
    }


def write_pareto(out_dir: Path, run_root: Path, validation: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Per-depth Pareto tables over every measurement we have (pp↑ tg↑ ttft↓ vram↓).

    Rows from grid / optuna / validation are labelled by ``source``; rows with an unmeasured metric
    never dominate and are only listed when nothing dominates them."""

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    import csv as _csv

    def add_csv(path: Path, source: str, final_only: bool = False) -> None:
        if not path.exists():
            return
        with path.open(newline="") as handle:
            for r in _csv.DictReader(handle):
                if r.get("capacity_ok") != "True":
                    continue
                if final_only and r.get("step") not in ("final",):
                    continue
                rows.append({"source": source, "depth": int(r["context_depth"]), "kv": r["kv_type"], "ngl": r["ngl"],
                             "batch": r["batch"], "ubatch": r["ubatch"], "flash_attn": r["flash_attn"],
                             "pp_tps": r["pp_tps"], "tg_tps": r["tg_tps"], "ttft_ms": r["ttft_ms"],
                             "vram_peak_mb": r["vram_peak_mb"]})
    add_csv(run_root / "grid" / "grid.csv", "grid")
    for opt in sorted((run_root / "optuna").glob("*/optuna_rows.csv")):
        add_csv(opt, "optuna", final_only=True)
    if validation:
        for cand in validation["candidates"]:
            for depth, s in cand["depths"].items():
                if int(depth) and s["n_ok"]:
                    cfg = cand["config"]
                    rows.append({"source": "validation(median)", "depth": int(depth), "kv": cfg["kv"], "ngl": cfg["ngl"],
                                 "batch": cfg["batch"], "ubatch": cfg["ubatch"], "flash_attn": cfg["flash_attn"],
                                 "pp_tps": s["pp"]["median"], "tg_tps": s["tg"]["median"],
                                 "ttft_ms": (s["ttft_est_s"] * 1000 if s["ttft_est_s"] else None),
                                 "vram_peak_mb": s["vram_peak_mb"]})
    summary: dict[str, Any] = {}
    have_vram = any(r["vram_peak_mb"] not in (None, "") for r in rows)
    minimize = ["ttft_ms"] + (["vram_peak_mb"] if have_vram else [])
    for depth, front in sorted(pareto_by_group(rows, "depth", maximize=["pp_tps", "tg_tps"], minimize=minimize).items()):
        if depth == 0:
            continue
        write_json(out_dir / f"pareto_d{depth}.json", {"depth": depth, "maximize": ["pp_tps", "tg_tps"],
                                                       "minimize": minimize, "front": front})
        with (out_dir / f"pareto_d{depth}.csv").open("w", newline="") as handle:
            writer = _csv.DictWriter(handle, fieldnames=list(front[0]) if front else ["depth"])
            writer.writeheader()
            writer.writerows(front)
        summary[str(depth)] = len(front)
    return summary


def failure_summary(run_root: Path) -> dict[str, Any]:
    """Counts and examples of every non-success status across stages."""

    import csv as _csv
    counts: dict[str, Counter] = {}
    examples: list[dict[str, Any]] = []
    for stage, rel in (("capacity", "capacity/capacity.csv"), ("grid", "grid/grid.csv"),
                       ("optuna", None), ("validation", "validation/validation.csv")):
        paths = [run_root / rel] if rel else sorted((run_root / "optuna").glob("*/optuna_rows.csv"))
        for path in paths:
            if not path.exists():
                continue
            with path.open(newline="") as handle:
                for r in _csv.DictReader(handle):
                    counts.setdefault(stage, Counter())[r["status"]] += 1
                    if r["status"] not in ("success", "skipped_after_fail") and len(examples) < 50:
                        examples.append({"stage": stage, "case": r["case_key"], "status": r["status"],
                                         "error": r["error"], "csv": r["csv"], "stderr": r["stderr"]})
    return {"status_counts": {k: dict(v) for k, v in counts.items()}, "examples": examples}


def write_profiles(ctx: Context, run_root: Path, validation_path: Path, *, fast_depth: int,
                   balanced_depth: int) -> dict[str, Any]:
    validation = read_json(validation_path, "validation.json")
    vram_total = ctx.gpu.memory_total_mib if ctx.gpu else None
    picks = select_profiles(validation, fast_depth=fast_depth, balanced_depth=balanced_depth, vram_total=vram_total)
    out = run_root / "profiles"
    out.mkdir(parents=True, exist_ok=True)
    env = environment_metadata()
    stem = _slug(f"{env['host']}-gpu{ctx.gpu_index if ctx.gpu_index is not None else 'x'}-{ctx.model.stem}")
    paths = {"validation": str(validation_path), "capacity": str(run_root / "capacity" / "capacity.json")}
    written: dict[str, str] = {}
    for name in PROFILES:
        profile = build_profile(ctx, name, picks[name], validation, paths, vram_total)
        target = out / f"{stem}-{name}.json"
        write_json(target, profile)
        written[name] = str(target)
    pareto = write_pareto(run_root / "pareto", run_root, validation)
    write_json(run_root / "failure_summary.json", failure_summary(run_root))
    return {"profiles": written, "pareto_depths": pareto}
