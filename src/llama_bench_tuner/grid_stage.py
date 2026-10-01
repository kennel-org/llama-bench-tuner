"""Phase 1 — coarse Grid restricted to what the capacity sweep says can run.

The Grid exists to *show* discontinuities (VRAM cliffs, OOM boundaries, FlashAttention
and KV-type effects, offload edges), not to pick a winner: its output narrows the
Optuna search space (``next_space``) and records per-axis effects.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

from .measure import BenchPoint
from .pareto import pareto_by_group
from .stage_common import Checkpoint, Context, load_capacity, median_iqr, read_json, write_json, write_rows_csv
from .status import BenchStatus

_FAIL = {BenchStatus.OOM.value, BenchStatus.RUNTIME_ABORT.value, BenchStatus.TIMEOUT.value, BenchStatus.FAILED.value}

AXES = ("ngl", "batch", "ubatch", "flash_attn", "n_cpu_moe", "split_mode", "nkvo")


def load_space(path: Optional[Path]) -> dict[str, list[Any]]:
    """Optional JSON ``{axis: [values...]}``. Unknown keys are rejected so typos cannot be silently ignored."""

    if path is None:
        return {}
    data = read_json(path, "grid space")
    allowed = set(AXES) | {"kv", "depths"}
    unknown = set(data) - allowed
    if unknown:
        raise SystemExit(f"[FATAL] Unknown grid-space key(s) {sorted(unknown)}; allowed: {sorted(allowed)}")
    for key, value in data.items():
        if not isinstance(value, list) or not value:
            raise SystemExit(f"[FATAL] grid-space key '{key}' must be a non-empty list")
    return data


def default_space(model_is_moe: bool = False) -> dict[str, list[Any]]:
    return {"ngl": [99], "batch": [512, 1024, 2048, 4096], "ubatch": [256, 512, 1024], "flash_attn": [0, 1]}


def targets_from_capacity(capacity: dict[str, Any], depths: Optional[Sequence[int]],
                          kv_filter: Optional[Sequence[str]], include_capacity_only: bool) -> list[tuple[str, int]]:
    """(kv, depth) pairs the Grid may use: practical candidates (or merely-completed with the flag)."""

    key = "capacity_survivors" if include_capacity_only else "survivors"
    pairs = [(s["kv"], int(s["depth"])) for s in capacity.get(key, [])]
    if depths:
        pairs = [(kv, d) for kv, d in pairs if d in set(depths)]
    if kv_filter:
        pairs = [(kv, d) for kv, d in pairs if kv in set(kv_filter)]
    return sorted(set(pairs), key=lambda p: (p[0], p[1]))


def build_cases(space: dict[str, list[Any]], targets: list[tuple[str, int]], base: BenchPoint,
                max_cases: Optional[int]) -> list[BenchPoint]:
    """Cross product of config axes x capacity targets, shallow-to-deep within each configuration so a
    failure at one depth can skip the deeper ones for that configuration."""

    axes = {name: space.get(name) or [getattr(base, name)] for name in AXES}
    configs = []
    for combo in itertools.product(*(axes[a] for a in AXES)):
        cfg = dict(zip(AXES, combo))
        if cfg["ubatch"] is not None and cfg["batch"] is not None and cfg["ubatch"] > cfg["batch"]:
            continue  # not a valid llama.cpp configuration; not worth a process
        configs.append(cfg)
    cases: list[BenchPoint] = []
    kvs = sorted({kv for kv, _ in targets})
    for cfg in configs:
        for kv in kvs:
            for depth in sorted(d for k, d in targets if k == kv):
                cases.append(base.replace(kv=kv, depth=depth, **cfg))
    if max_cases is not None and len(cases) > max_cases:
        raise SystemExit(f"[FATAL] Grid has {len(cases)} cases (> --max-cases {max_cases}). "
                         "Shrink the space or raise the limit; the Grid is meant to be coarse.")
    return cases


def axis_effects(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per axis value: how many runs completed/failed and the best pp/tg — makes FA, KV, offload and
    VRAM-cliff effects visible without picking a winner."""

    columns = {"ngl": "ngl", "batch": "batch", "ubatch": "ubatch", "flash_attn": "flash_attn",
               "kv": "kv_type", "n_cpu_moe": "n_cpu_moe"}
    out: dict[str, Any] = {}
    for axis, column in columns.items():
        buckets: dict[str, dict[str, Any]] = defaultdict(lambda: {"ok": 0, "fail": 0, "tg": [], "pp": []})
        for r in rows:
            value = r.get(column)
            if value is None:
                continue
            b = buckets[str(value)]
            if r["capacity_ok"]:
                b["ok"] += 1
                if r["tg_tps"] is not None:
                    b["tg"].append(r["tg_tps"])
                if r["pp_tps"] is not None:
                    b["pp"].append(r["pp_tps"])
            elif r["status"] in _FAIL:
                b["fail"] += 1
        if len(buckets) > 1:
            out[axis] = {v: {"completed": b["ok"], "failed": b["fail"],
                             "tg_best": max(b["tg"], default=None), "tg_median": median_iqr(b["tg"])["median"],
                             "pp_best": max(b["pp"], default=None)} for v, b in sorted(buckets.items())}
    return out


def failure_boundaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Axis values at which a (kv, depth) slice flips from completing to failing — OOM / abort cliffs."""

    cliffs = []
    for axis in ("ngl", "batch", "ubatch"):
        slices: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            other = tuple((a, r.get(a)) for a in ("ngl", "batch", "ubatch", "flash_attn", "n_cpu_moe") if a != axis)
            slices[(r["kv_type"], r["context_depth"], other)].append(r)
        for (kv, depth, other), group in slices.items():
            group = sorted((g for g in group if g.get(axis) is not None), key=lambda g: g[axis])
            for lo, hi in zip(group, group[1:]):
                if lo["capacity_ok"] and hi["status"] in _FAIL:
                    cliffs.append({"axis": axis, "kv": kv, "depth": depth, "fixed": dict(other),
                                   "last_ok": lo[axis], "first_fail": hi[axis], "failure_status": hi["status"]})
    return cliffs


def promising(rows: list[dict[str, Any]], top_n: int = 3) -> list[dict[str, Any]]:
    """Pareto rows (pp↑ tg↑ vram↓, or pp↑ tg↑ when VRAM is unmeasured) plus the best few by tg and by pp, per depth."""

    done = [r for r in rows if r["capacity_ok"]]
    minimize = ["vram_peak_mb"] if all(r["vram_peak_mb"] is not None for r in done) and done else []
    fronts = pareto_by_group(done, "context_depth", maximize=["pp_tps", "tg_tps"], minimize=minimize)
    chosen: dict[str, dict[str, Any]] = {}
    for depth, front in fronts.items():
        pool = [r for r in done if r["context_depth"] == depth]
        extra = sorted(pool, key=lambda r: -r["tg_tps"])[:top_n] + sorted(pool, key=lambda r: -r["pp_tps"])[:top_n]
        for r in list(front) + extra:
            chosen[r["case_key"]] = r
    return list(chosen.values())


def next_space(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Value sets seen among promising rows — the narrowed search space handed to Optuna."""

    cols = {"ngl": "ngl", "batch": "batch", "ubatch": "ubatch", "flash_attn": "flash_attn", "kv": "kv_type",
            "n_cpu_moe": "n_cpu_moe"}
    space: dict[str, list[Any]] = {}
    for axis, column in cols.items():
        values = sorted({r[column] for r in rows if r.get(column) is not None}, key=str)
        if values:
            space[axis] = values
    return space


def run_grid(ctx: Context, capacity_path: Path, out_dir: Path, log_root: Path, *, space: dict[str, list[Any]],
             depths: Optional[Sequence[int]], kv_filter: Optional[Sequence[str]], include_capacity_only: bool,
             base: BenchPoint, max_cases: Optional[int], resume: bool) -> list[dict[str, Any]]:
    capacity = load_capacity(capacity_path)
    targets = targets_from_capacity(capacity, depths, kv_filter, include_capacity_only)
    if not targets:
        raise SystemExit("[FATAL] capacity sweep left no surviving (kv, depth) to grid over. "
                         "Use --include-capacity-only to allow merely-completed points.")
    space = {**default_space(), **space}
    cases = build_cases(space, targets, base, max_cases)
    out_dir.mkdir(parents=True, exist_ok=True)
    definition = {"stage": "grid", **ctx.identity(), "capacity_fingerprint": capacity.get("benchmark_fingerprint"),
                  "cases": [c.key() for c in cases]}
    ck = Checkpoint(out_dir / "checkpoint.json", definition, resume, "grid")
    measurer = ctx.measurer(out_dir, log_root)
    failed_cfg: set[tuple[str, str]] = {(r["kv_type"], BenchPoint(**{**_point_from_row(r), "depth": 0}).config_key())
                                        for r in ck.rows if r["status"] in _FAIL}
    for point in cases:
        key = point.key()
        if key in ck.done:
            continue
        cfg_id = (point.kv, point.config_key())
        if cfg_id in failed_cfg:
            row = measurer.base_row("grid", point)
            row.update(status=BenchStatus.SKIPPED_AFTER_FAIL.value, success=False, capacity_ok=False,
                       practical_candidate=False, error="", practical_reason="skipped_after_fail",
                       skip_reason="shallower depth failed for this configuration", telemetry_status="not_run")
        else:
            row = measurer.measure("grid", point)
            if row["status"] in _FAIL:
                failed_cfg.add(cfg_id)
        ck.add(key, row, len(cases), "grid")

    rows = ck.rows
    best = promising(rows)
    write_rows_csv(out_dir / "grid.csv", rows)
    write_json(out_dir / "grid.json", {
        "stage": "grid", "benchmark_fingerprint": ck.fingerprint, "capacity": str(capacity_path),
        "targets": [{"kv": kv, "depth": d} for kv, d in targets], "space": space,
        "n_cases": len(cases), "elapsed_seconds": ck.elapsed(),
        "axis_effects": axis_effects(rows), "failure_boundaries": failure_boundaries(rows),
        "promising": [{k: r[k] for k in ("case_key", "context_depth", "kv_type", "ngl", "batch", "ubatch",
                                         "flash_attn", "n_cpu_moe", "pp_tps", "tg_tps", "vram_peak_mb")} for r in best],
        "next_space": next_space(best),
    })
    return rows


def _point_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """Rebuild BenchPoint kwargs from a stored row (used to restore failure state after --resume)."""

    return {"kv": row["kv_type"], "depth": row["context_depth"], "ngl": row["ngl"], "batch": row["batch"],
            "ubatch": row["ubatch"], "flash_attn": row["flash_attn"], "n_cpu_moe": row.get("n_cpu_moe"),
            "prompt": row["prompt_tokens"], "ngen": row["generated_tokens"]}
