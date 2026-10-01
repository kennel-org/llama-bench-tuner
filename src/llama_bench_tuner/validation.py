"""Phase 3 — repeat the best candidates and keep only what is stable (``llama-tune-pipeline validate``).

A single best Optuna trial is never trusted. The top candidates (from the Optuna Pareto sets and
the Grid's promising rows) are re-measured ``reps`` times as *separate processes* at the target
depths, and summarised with median / IQR / CV. A depth is only ``practical`` when every repeat
completed, throughput was stable and the capacity criteria hold.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

from .capacity import PracticalCriteria, evaluate_practical
from .measure import BenchPoint, Measurer, estimate_ttft_s
from .stage_common import Checkpoint, Context, load_capacity, median_iqr, read_json, write_json, write_rows_csv
from .status import BenchStatus

CONFIG_AXES = ("kv", "ngl", "batch", "ubatch", "flash_attn", "n_cpu_moe")
_AXIS_PARAM = {"kv": "kv", "ngl": "ngl", "batch": "batch", "ubatch": "ubatch",
               "flash_attn": "flash_attn", "n_cpu_moe": "n_cpu_moe"}


def point_from_params(params: dict[str, Any], space: dict[str, list[Any]], base: dict[str, Any]) -> BenchPoint:
    """Rebuild a full point: suggested params win, otherwise the single fixed value of the study space."""

    values: dict[str, Any] = {}
    for axis in CONFIG_AXES:
        if _AXIS_PARAM[axis] in params:
            values[axis] = params[_AXIS_PARAM[axis]]
        elif space.get(axis):
            values[axis] = space[axis][0]
        else:
            values[axis] = base[axis]
    return BenchPoint(**{**base, **values, "depth": 0})


def collect_candidates(run_root: Path, top_k: int) -> list[dict[str, Any]]:
    """Gather configs from every optuna.json and grid.json under ``run_root``, rank, keep ``top_k``."""

    found: dict[str, dict[str, Any]] = {}
    for path in sorted((run_root / "optuna").glob("*/optuna.json")):
        study = read_json(path, "optuna.json")
        base = study.get("base") or BenchPoint().to_dict()
        for item in study.get("best", []):
            point = point_from_params(item["params"], study.get("space", {}), base)
            attrs = item.get("attrs", {})
            found.setdefault(point.config_key(), {"point": point, "source": f"optuna@{study['depth']}",
                                                  "tg": attrs.get("tg_tps"), "pp": attrs.get("pp_tps"),
                                                  "vram": attrs.get("vram_peak_mb")})
    grid_path = run_root / "grid" / "grid.json"
    if grid_path.exists():
        grid = read_json(grid_path, "grid.json")
        base = BenchPoint().to_dict()
        for item in grid.get("promising", []):
            point = BenchPoint(**{**base, "kv": item["kv_type"], "ngl": item["ngl"], "batch": item["batch"],
                                  "ubatch": item["ubatch"], "flash_attn": item["flash_attn"],
                                  "n_cpu_moe": item.get("n_cpu_moe"), "depth": 0})
            found.setdefault(point.config_key(), {"point": point, "source": f"grid@{item['context_depth']}",
                                                  "tg": item.get("tg_tps"), "pp": item.get("pp_tps"),
                                                  "vram": item.get("vram_peak_mb")})
    cands = list(found.values())
    if not cands:
        raise SystemExit("[FATAL] no candidates: run the grid and/or optuna stage first")

    # Throughput at different depths is not comparable (tg falls with depth), so candidates are ranked
    # only against others from the same source (e.g. optuna@32768) and then interleaved round-robin:
    # every target depth contributes its best configs instead of the shallowest study winning by default.
    groups: dict[str, list[dict[str, Any]]] = {}
    for c in cands:
        groups.setdefault(c["source"], []).append(c)
    for members in groups.values():
        ranks = [_rank(members, "tg", True), _rank(members, "pp", True), _rank(members, "vram", False)]
        for c in members:
            positions = [r[c["point"].config_key()] for r in ranks if c["point"].config_key() in r]
            c["rank"] = sum(positions) / len(positions) if positions else 1e9
        members.sort(key=lambda c: c["rank"])
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    order = sorted(groups)  # deterministic
    index = 0
    while len(chosen) < top_k and any(index < len(groups[g]) for g in order):
        for g in order:
            if index < len(groups[g]):
                c = groups[g][index]
                key = c["point"].config_key()
                if key not in seen and len(chosen) < top_k:
                    seen.add(key)
                    chosen.append(c)
        index += 1
    return chosen


def _rank(members: list[dict[str, Any]], key: str, reverse: bool) -> dict[str, int]:
    have = sorted((c for c in members if c[key] is not None), key=lambda c: c[key], reverse=reverse)
    return {c["point"].config_key(): i for i, c in enumerate(have)}


def summarise(runs: list[dict[str, Any]], depth: int, reps: int, d0_pp: Optional[float],
              d0_tg: Optional[float], crit: PracticalCriteria, cv_limit: float,
              vram_total: Optional[int]) -> dict[str, Any]:
    ok = [r for r in runs if r["capacity_ok"]]
    tg, pp = median_iqr([r["tg_tps"] for r in ok]), median_iqr([r["pp_tps"] for r in ok])
    vram = [r["vram_peak_mb"] for r in ok if r["vram_peak_mb"] is not None]
    ram = [r["ram_peak_mb"] for r in ok if r["ram_peak_mb"] is not None]
    wall = median_iqr([r["wall_time_s"] for r in ok])
    stable = len(ok) == reps and (tg["cv"] is None or tg["cv"] <= cv_limit)
    practical, reason = evaluate_practical(
        completed=stable, tg=tg["median"], pp=pp["median"], tg0=d0_tg, pp0=d0_pp,
        wall_s=wall["median"] or 0.0, vram_peak=max(vram) if vram else None, vram_total=vram_total, crit=crit)
    ttft = estimate_ttft_s(depth, d0_pp, pp["median"]) if depth else None
    statuses = sorted({r["status"] for r in runs})
    return {
        "n_runs": len(runs), "n_ok": len(ok), "statuses": statuses, "stable": stable, "cv_limit": cv_limit,
        "tg": tg, "pp": pp, "wall_s": wall, "ttft_est_s": ttft,
        "vram_peak_mb": max(vram) if vram else None, "ram_peak_mb": max(ram) if ram else None,
        "temp_max_c": max((r["temp_max_c"] for r in ok if r["temp_max_c"] is not None), default=None),
        "sm_clock_min_mhz": min((r["sm_clock_min_mhz"] for r in ok if r["sm_clock_min_mhz"] is not None), default=None),
        "throttle_reasons": sorted({r["throttle_reasons"] for r in ok if r["throttle_reasons"]}),
        "practical": practical, "practical_reason": reason,
    }


def run_validation(ctx: Context, run_root: Path, capacity_path: Path, out_dir: Path, log_root: Path, *,
                   depths: Sequence[int], reps: int, top_k: int, criteria: PracticalCriteria, cv_limit: float,
                   soak_ngen: int, base: BenchPoint, resume: bool) -> dict[str, Any]:
    capacity = load_capacity(capacity_path)
    cands = collect_candidates(run_root, top_k)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap_max = {kv: info["capacity_max_depth"] for kv, info in capacity["per_kv"].items()}

    plan: list[tuple[dict[str, Any], int, int]] = []  # (candidate, depth, rep)
    skipped: list[dict[str, Any]] = []
    for cand in cands:
        kv = cand["point"].kv
        for depth in [0, *depths]:
            if depth and (cap_max.get(kv) is None or depth > cap_max[kv]):
                skipped.append({"config": cand["point"].config_key(), "depth": depth,
                                "reason": f"beyond capacity boundary of kv={kv} ({cap_max.get(kv)})"})
                continue
            plan.extend((cand, depth, rep) for rep in range(1, reps + 1))
    definition = {"stage": "validation", **ctx.identity(), "reps": reps, "depths": list(depths),
                  "capacity_fingerprint": capacity.get("benchmark_fingerprint"), "criteria": criteria.__dict__,
                  "plan": [f"{c['point'].config_key()}|d={d}|r={r}" for c, d, r in plan]}
    ck = Checkpoint(out_dir / "checkpoint.json", definition, resume, "validation")
    measurer = ctx.measurer(out_dir, log_root)
    for cand, depth, rep in plan:
        point = cand["point"].replace(depth=depth, prompt=base.prompt, ngen=base.ngen, reps=1)
        key = f"{cand['point'].config_key()}|d={depth}|r={rep}"
        if key in ck.done:
            continue
        row = measurer.measure("validation", point, tag=f"r{rep}")
        row.update(step=f"rep{rep}")
        ck.add(key, row, len(plan), "validation")

    summaries = []
    vram_total = ctx.gpu.memory_total_mib if ctx.gpu else None
    for cand in cands:
        cfg_key = cand["point"].config_key()
        mine = [r for r in ck.rows if r["case_key"] and _config_of(r) == cfg_key]
        by_depth: dict[int, list[dict[str, Any]]] = {}
        for r in mine:
            by_depth.setdefault(r["context_depth"], []).append(r)
        d0 = summarise(by_depth.get(0, []), 0, reps, None, None, criteria, cv_limit, vram_total)
        per_depth = {"0": d0}
        for depth in depths:
            if depth in by_depth:
                per_depth[str(depth)] = summarise(by_depth[depth], depth, reps, d0["pp"]["median"],
                                                  d0["tg"]["median"], criteria, cv_limit, vram_total)
        soak = None
        practical_depths = [int(d) for d, s in per_depth.items() if int(d) and s["practical"]]
        if soak_ngen and practical_depths:
            deepest = max(practical_depths)
            srow = measurer.measure("validation", cand["point"].replace(depth=deepest, ngen=soak_ngen, reps=1),
                                    tag="soak")
            srow.update(step="soak")
            ck.rows.append(srow)
            tg_med = per_depth[str(deepest)]["tg"]["median"]
            soak = {"depth": deepest, "ngen": soak_ngen, "status": srow["status"], "tg_tps": srow["tg_tps"],
                    "tg_vs_median": (srow["tg_tps"] / tg_med) if srow["tg_tps"] and tg_med else None,
                    "temp_max_c": srow["temp_max_c"], "sm_clock_min_mhz": srow["sm_clock_min_mhz"],
                    "throttle_reasons": srow["throttle_reasons"]}
        summaries.append({"config": cand["point"].to_dict(), "config_key": cfg_key, "source": cand["source"],
                          "depths": per_depth, "soak": soak})
    write_rows_csv(out_dir / "validation.csv", ck.rows)
    result = {"stage": "validation", "reps": reps, "depths": list(depths), "top_k": top_k,
              "criteria": criteria.__dict__, "cv_limit": cv_limit, "benchmark_fingerprint": ck.fingerprint,
              "runtime_commits": sorted({r["runtime_commit"] for r in ck.rows if r.get("runtime_commit")}),
              "skipped": skipped, "candidates": summaries, "elapsed_seconds": ck.elapsed()}
    write_json(out_dir / "validation.json", result)
    return result


def _config_of(row: dict[str, Any]) -> str:
    return BenchPoint(kv=row["kv_type"], ngl=row["ngl"], batch=row["batch"], ubatch=row["ubatch"],
                      flash_attn=row["flash_attn"], n_cpu_moe=row.get("n_cpu_moe")).config_key()
