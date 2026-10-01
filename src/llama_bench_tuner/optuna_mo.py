"""Phase 2 — Optuna over the narrowed space (``llama-tune-pipeline optuna``).

Two modes:

* ``pareto`` (default): multi-objective NSGA-II over tg↑ pp↑ ttft↓ vram↓ wall↓. Optuna does
  not support ``trial.report()`` / ``should_prune()`` in multi-objective studies, so pruning is a
  ``PrunePolicy`` that raises ``TrialPruned`` after the cheap stage.
* ``score:<fast|balanced|long>``: single-objective use-case score with genuine
  ``trial.report`` + ``MedianPruner`` on the cheap stage.

Every trial measures twice: a *cheap* stage (depth 0, pp512/tg64) used only for pruning and for
the TTFT estimate, then the *final* stage at the study's target depth. Cheap metrics are stored
as ``cheap_*`` user attrs and as ``step=cheap`` rows and are never mixed into final metrics.
One study per target depth, so 32K / 64K / 128K Pareto sets stay separate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import optuna

from .measure import BenchPoint, Measurer, estimate_ttft_s, unsupported_reason
from .stage_common import Context, read_json, write_json, write_rows_csv
from .status import BenchStatus

OBJECTIVES = {
    "tg": ("tg_tps", "maximize"),
    "pp": ("pp_tps", "maximize"),
    "ttft": ("ttft_est_s", "minimize"),
    "vram": ("vram_peak_mb", "minimize"),
    "wall": ("wall_time_s", "minimize"),
}
DEFAULT_OBJECTIVES = ("tg", "pp", "ttft", "vram", "wall")

# (tg, pp, ttft-penalty, vram-penalty) weights for the single-objective use-case scores.
SCORE_WEIGHTS = {
    "fast": (0.5, 0.2, 0.3, 0.0),
    "balanced": (0.35, 0.3, 0.1, 0.25),
    "long": (0.7, 0.3, 0.0, 0.0),
}


@dataclass(frozen=True)
class PrunePolicy:
    """Cheap-stage pruning shared by both modes (mode ``score`` additionally uses MedianPruner)."""

    min_completed: int = 4       # never prune before this many cheap results exist
    tg_ratio: float = 0.6        # prune if cheap tg < ratio x best cheap tg so far

    def reason(self, cheap: dict[str, Any], seen_tg: Sequence[float]) -> Optional[str]:
        if not cheap["capacity_ok"]:
            return f"cheap stage {cheap['status']}"
        tg = cheap["tg_tps"]
        if tg is not None and len(seen_tg) >= self.min_completed and tg < self.tg_ratio * max(seen_tg):
            return f"cheap tg {tg:.1f} < {self.tg_ratio:g} x best {max(seen_tg):.1f}"
        return None


def active_objectives(requested: Sequence[str], telemetry_available: bool) -> list[str]:
    """Drop VRAM when no telemetry exists — an unmeasured objective must not steer the search."""

    unknown = [o for o in requested if o not in OBJECTIVES]
    if unknown:
        raise SystemExit(f"[FATAL] unknown objective(s) {unknown}; choose from {sorted(OBJECTIVES)}")
    chosen = [o for o in requested if o != "vram" or telemetry_available]
    if len(chosen) < 2:
        raise SystemExit("[FATAL] multi-objective mode needs at least two usable objectives")
    return chosen


def use_case_score(profile: str, row: dict[str, Any], refs: dict[str, float]) -> float:
    w_tg, w_pp, w_ttft, w_vram = SCORE_WEIGHTS[profile]
    total = row.get("vram_total_mb") or 0
    score = w_tg * (row["tg_tps"] or 0) / refs.get("tg", 1.0) + w_pp * (row["pp_tps"] or 0) / refs.get("pp", 1.0)
    if w_ttft and row.get("ttft_est_s"):
        score -= w_ttft * row["ttft_est_s"] / refs.get("ttft", row["ttft_est_s"])
    if w_vram and total and row.get("vram_peak_mb"):
        score -= w_vram * row["vram_peak_mb"] / total
    return score


def _choices(space: dict[str, list[Any]], name: str, default: Any) -> list[Any]:
    values = space.get(name) or [default]
    return list(values)


def sample_point(trial: optuna.trial.Trial, space: dict[str, list[Any]], base: BenchPoint, depth: int) -> BenchPoint:
    def pick(param: str, axis: str, default: Any) -> Any:
        values = _choices(space, axis, default)
        return values[0] if len(values) == 1 else trial.suggest_categorical(param, values)

    return base.replace(
        depth=depth, kv=pick("kv", "kv", base.kv), ngl=pick("ngl", "ngl", base.ngl),
        batch=pick("batch", "batch", base.batch), ubatch=pick("ubatch", "ubatch", base.ubatch),
        flash_attn=pick("flash_attn", "flash_attn", base.flash_attn),
        n_cpu_moe=pick("n_cpu_moe", "n_cpu_moe", base.n_cpu_moe),
    )


def _row_with(row: dict[str, Any], **extra: Any) -> dict[str, Any]:
    row.update(extra)
    return row


def run_study(ctx: Context, measurer: Measurer, out_dir: Path, *, space: dict[str, list[Any]], base: BenchPoint,
              depth: int, n_trials: int, mode: str, objectives: Sequence[str], policy: PrunePolicy,
              seed: Optional[int], population: Optional[int], refs: dict[str, float],
              study_name: str) -> dict[str, Any]:
    """Run (or resume) one study at one target depth; returns the summary dict written to optuna.json."""

    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "trial_rows.jsonl"
    storage = f"sqlite:///{(out_dir / 'study.db').resolve()}"
    multi = mode == "pareto"
    objs = active_objectives(objectives, ctx.backend is not None) if multi else []
    if multi:
        sampler = optuna.samplers.NSGAIISampler(seed=seed, population_size=population or max(4, min(16, n_trials)))
        study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True, sampler=sampler,
                                    directions=[OBJECTIVES[o][1] for o in objs])
        study.set_user_attr("objectives", objs)
    else:
        profile = mode.split(":", 1)[1]
        if profile not in SCORE_WEIGHTS:
            raise SystemExit(f"[FATAL] unknown score profile '{profile}'; choose from {sorted(SCORE_WEIGHTS)}")
        study = optuna.create_study(
            study_name=study_name, storage=storage, load_if_exists=True, direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=policy.min_completed, n_warmup_steps=0))
    seen_tg: list[float] = [t.user_attrs["cheap_tg"] for t in study.trials
                            if t.user_attrs.get("cheap_tg") is not None]

    def log_row(row: dict[str, Any]) -> None:
        with rows_path.open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def objective(trial: optuna.trial.Trial):
        point = sample_point(trial, space, base, depth)
        bad = unsupported_reason(ctx.caps, point)
        if bad:
            trial.set_user_attr("status", BenchStatus.UNSUPPORTED.value)
            trial.set_user_attr("error", bad)
            raise optuna.TrialPruned()
        # ---- cheap stage: depth 0 (pruning metric + pp0 for the TTFT estimate) ----
        cheap = _row_with(measurer.measure("optuna", point.replace(depth=0), tag=f"t{trial.number}c"),
                          trial_number=trial.number, step="cheap")
        log_row(cheap)
        trial.set_user_attr("cheap_tg", cheap["tg_tps"])
        trial.set_user_attr("cheap_pp", cheap["pp_tps"])
        trial.set_user_attr("cheap_status", cheap["status"])
        reason = policy.reason(cheap, seen_tg)
        if cheap["tg_tps"] is not None:
            seen_tg.append(cheap["tg_tps"])
        if not multi:
            trial.report(cheap["tg_tps"] or 0.0, 0)  # cheap metric: compared with other trials' step 0 only
            if reason is None and trial.should_prune():
                reason = "MedianPruner on cheap tg"
        if reason:
            trial.set_user_attr("status", "pruned")
            trial.set_user_attr("prune_reason", reason)
            raise optuna.TrialPruned()
        # ---- final stage at the study depth ----
        final = cheap if depth == 0 else measurer.measure("optuna", point, tag=f"t{trial.number}f")
        ttft = estimate_ttft_s(depth, cheap["pp_tps"], final["pp_tps"])
        final = _row_with(final, trial_number=trial.number, step="final", ttft_ms=(ttft * 1000 if ttft else None),
                          ttft_kind="estimated_from_pp" if ttft else None)
        final["ttft_est_s"] = ttft
        log_row(final)
        for key in ("status", "error"):
            trial.set_user_attr(key, final[key])
        for key in ("pp_tps", "tg_tps", "vram_peak_mb", "ram_peak_mb", "wall_time_s", "ttft_est_s"):
            trial.set_user_attr(key, final.get(key))
        if not final["capacity_ok"]:
            raise optuna.TrialPruned()  # failures never become "best"; the row above records why
        if multi:
            values = [final.get(OBJECTIVES[o][0]) for o in objs]
            if any(v is None for v in values):
                raise optuna.TrialPruned()  # an unmeasured objective cannot be ranked
            return [float(v) for v in values]
        return use_case_score(mode.split(":", 1)[1], final, refs)

    # Only COMPLETE and PRUNED trials count: a trial that died (GPU busy, Ctrl-C) must be re-run on resume.
    counted = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    remaining = max(0, n_trials - len([t for t in study.trials if t.state in counted]))
    if remaining:
        study.optimize(objective, n_trials=remaining)

    best = study.best_trials if multi else ([study.best_trial] if any(
        t.state == optuna.trial.TrialState.COMPLETE for t in study.trials) else [])
    states: dict[str, int] = {}
    for t in study.trials:
        states[t.state.name] = states.get(t.state.name, 0) + 1
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()] if rows_path.exists() else []
    write_rows_csv(out_dir / "optuna_rows.csv", rows)
    summary = {
        "stage": "optuna", "mode": mode, "depth": depth, "study": study_name, "objectives": objs or [mode],
        "base": base.to_dict(), "space": space,
        "directions": [OBJECTIVES[o][1] for o in objs] if multi else ["maximize"],
        "trial_states": states, "n_trials": len(study.trials),
        "prune_policy": {"min_completed": policy.min_completed, "tg_ratio": policy.tg_ratio,
                         "note": "cheap-stage (depth 0) metrics are used for pruning only"},
        "best": [{"number": t.number, "params": t.params, "values": t.values if multi else [t.value],
                  "attrs": {k: t.user_attrs.get(k) for k in ("pp_tps", "tg_tps", "vram_peak_mb", "ram_peak_mb",
                                                              "wall_time_s", "ttft_est_s", "cheap_pp", "cheap_tg")}}
                 for t in best],
    }
    write_json(out_dir / "optuna.json", summary)
    return summary


def refs_from_grid(grid_path: Optional[Path], depth: int) -> dict[str, float]:
    """Normalisers for the single-objective scores: best Grid tg/pp at this depth (1.0 if unavailable)."""

    if grid_path is None or not grid_path.exists():
        return {}
    promising = read_json(grid_path, "grid.json").get("promising", [])
    at_depth = [p for p in promising if p["context_depth"] == depth]
    refs: dict[str, float] = {}
    if at_depth:
        refs["tg"] = max(p["tg_tps"] for p in at_depth)
        refs["pp"] = max(p["pp_tps"] for p in at_depth)
    return refs
