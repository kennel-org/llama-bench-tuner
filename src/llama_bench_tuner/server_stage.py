"""Pipeline stages that need a real ``llama-server``: ``server`` (measured TTFT, speculation compare)
and ``soak`` (long decode with GPU time series).

Both start the server from a *profile's own command*, so what is measured is exactly what the profile
tells people to run. A profile is only annotated (never re-chosen): measured values are added next to the
``llama-bench`` estimates, and a speculation command is added only as an alternative."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from .measure import GpuBusyError
from .schema import pipeline_row
from .server_adapter import (ServerSession, command_with, measure_sizes, server_row_fields, soak, spec_args)
from .stage_common import Context, read_json, write_json, write_rows_csv
from .measure import model_size_bytes

PROFILE_NAMES = ("fast", "balanced", "long")


def _profiles(run_root: Path, only: Optional[Sequence[str]]) -> list[tuple[Path, dict[str, Any]]]:
    found = []
    for path in sorted((run_root / "profiles").glob("*.json")):
        data = read_json(path, "profile")
        if data.get("available") and (not only or data["profile"] in only):
            found.append((path, data))
    if not found:
        raise SystemExit("[FATAL] no available profiles found; run the profile stage first")
    return found


def _sizes_for(profile: dict[str, Any], requested: Sequence[int], max_tokens: int) -> list[int]:
    limit = profile.get("server_context", profile["context"]) - max_tokens - 96
    sizes = sorted({s for s in requested if 256 <= s <= limit})
    return sizes or [max(256, min(4096, limit))]


def _gate(ctx: Context) -> None:
    if ctx.gate is not None:
        verdict = ctx.gate()
        if not verdict.ok:
            raise GpuBusyError(verdict)


def _session(ctx: Context, argv: list[str], log: Path, startup_timeout: float, series: bool = False) -> ServerSession:
    return ServerSession(argv=argv, backend=ctx.backend, gpu_index=ctx.gpu_index, log_path=log,
                         startup_timeout=startup_timeout, record_series=series,
                         poll_interval=2.0 if series else 1.0)


def run_server_stage(ctx: Context, run_root: Path, out_dir: Path, log_root: Path, *, sizes: Sequence[int],
                     reps: int, max_tokens: int, spec_compare: bool, spec_type: str, spec_draft_n_max: int,
                     startup_timeout: float, only: Optional[Sequence[str]] = None) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {"stage": "server", "reps": reps, "max_tokens": max_tokens, "profiles": {}}
    for path, profile in _profiles(run_root, only):
        name = profile["profile"]
        base_cmd = profile["commands"]["llama_server"]
        server_bin = Path(base_cmd.split()[0])
        use_sizes = _sizes_for(profile, sizes, max_tokens)
        entry: dict[str, Any] = {"sizes": use_sizes}

        _gate(ctx)
        with _session(ctx, command_with(base_cmd, []), log_root / f"{name}.base.log", startup_timeout) as s:
            commit = (s.props().get("build_info") or None)
            base = measure_sizes(s, use_sizes, reps, max_tokens)
            peaks = s.stop()
        entry.update(baseline=base, runtime_build_info=commit,
                     vram_peak_mb=peaks.vram_peak_mib, ram_peak_mb=peaks.ram_peak_mib)
        for res in base:
            row = pipeline_row(stage="server", benchmark_kind="server", case_key=f"{name}|size={res['target_tokens']}",
                               model=str(ctx.model), model_size_bytes=model_size_bytes(ctx.model),
                               context_depth=res["target_tokens"], prompt_tokens=res["target_tokens"],
                               generated_tokens=max_tokens, status="success" if res["n_ok"] == res["reps"] else "failed",
                               success=res["n_ok"] == res["reps"], gpu_index=ctx.gpu_index, runtime_commit=commit,
                               speculation=None, **server_row_fields(res, peaks))
            rows.append(row)

        spec_entry: dict[str, Any] = {"requested": spec_compare}
        if spec_compare:
            extra = spec_args(server_bin, spec_type, spec_draft_n_max)
            if extra is None:
                spec_entry.update(status="unsupported", reason=f"llama-server --help does not list --spec-type {spec_type}")
            else:
                _gate(ctx)
                try:
                    with _session(ctx, command_with(base_cmd, extra), log_root / f"{name}.spec.log", startup_timeout) as s:
                        spec = measure_sizes(s, use_sizes, reps, max_tokens)
                        speaks = s.stop()
                    drafted = [r["draft_n"] for res in spec for r in res["runs"] if r["draft_n"]]
                    per_size = []
                    for b, sp in zip(base, spec):
                        tb, ts = b["tg_tps"]["median"], sp["tg_tps"]["median"]
                        per_size.append({"target_tokens": b["target_tokens"], "tg_base": tb, "tg_spec": ts,
                                         "tg_speedup": (ts / tb) if tb and ts else None,
                                         "pp_ratio": (sp["pp_tps"]["median"] / b["pp_tps"]["median"])
                                         if b["pp_tps"]["median"] and sp["pp_tps"]["median"] else None,
                                         "ttft_base_s": b["ttft_s"]["median"], "ttft_spec_s": sp["ttft_s"]["median"],
                                         "draft_acceptance": sp["draft_acceptance"]["median"]})
                    active = bool(drafted)
                    spec_entry.update(status="active" if active else "inactive", spec_type=spec_type,
                                      draft_n_max=spec_draft_n_max, sizes=per_size, vram_peak_mb=speaks.vram_peak_mib,
                                      note=None if active else "server reported no drafted tokens; the model/GGUF "
                                                               "probably has no usable MTP head")
                    for res in spec:
                        rows.append(pipeline_row(
                            stage="server", benchmark_kind="server", case_key=f"{name}|spec|size={res['target_tokens']}",
                            model=str(ctx.model), context_depth=res["target_tokens"], prompt_tokens=res["target_tokens"],
                            generated_tokens=max_tokens, status="success", success=True, gpu_index=ctx.gpu_index,
                            speculation=f"{spec_type}:{spec_draft_n_max}", **server_row_fields(res, speaks)))
                except RuntimeError as exc:  # server could not start with speculation (e.g. no MTP tensors)
                    spec_entry.update(status="failed_to_start", reason=str(exc)[:300])
        entry["speculation"] = spec_entry
        report["profiles"][name] = entry

        # annotate the profile (never re-choose it)
        profile["measured_server"] = {
            "ttft_kind": "measured", "reps": reps, "max_tokens": max_tokens, "runtime_build_info": commit,
            "sizes": [{"target_tokens": r["target_tokens"], "ttft_s": r["ttft_s"]["median"],
                       "pp_tps": r["pp_tps"]["median"], "tg_tps": r["tg_tps"]["median"],
                       "total_s": r["total_s"]["median"]} for r in base],
            "vram_peak_mb": peaks.vram_peak_mib, "ram_peak_mb": peaks.ram_peak_mib,
        }
        profile["mtp_speculation"] = spec_entry
        if spec_entry.get("status") == "active":
            gains = [x["tg_speedup"] for x in spec_entry["sizes"] if x["tg_speedup"]]
            if gains and min(gains) >= 1.10:
                profile["commands"]["llama_server_mtp"] = " ".join(command_with(base_cmd, spec_args(server_bin, spec_type, spec_draft_n_max) or []))
        write_json(path, profile)
    write_rows_csv(out_dir / "server_rows.csv", rows, extra_fields=())
    write_json(out_dir / "server_results.json", report)
    return report


def run_soak_stage(ctx: Context, run_root: Path, out_dir: Path, log_root: Path, *, profile_name: str,
                   minutes: float, window_s: float, startup_timeout: float) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    path, profile = _profiles(run_root, [profile_name])[0]
    _gate(ctx)
    cmd = command_with(profile["commands"]["llama_server"], [])
    with _session(ctx, cmd, log_root / f"{profile_name}.soak.log", startup_timeout, series=True) as s:
        result = soak(s, minutes * 60.0, window_s)
        peaks = s.stop()
        series = list(s.monitor.series) if s.monitor else []
    result.update(profile=profile_name, command=profile["commands"]["llama_server"],
                  vram_peak_mb=peaks.vram_peak_mib, temp_max_c=peaks.temp_max_c,
                  sm_clock_min_mhz=peaks.sm_clock_min_mhz, power_max_w=peaks.power_max_w, gpu_index=ctx.gpu_index)
    write_json(out_dir / f"soak_{profile_name}.json", result)
    write_json(out_dir / f"soak_{profile_name}_gpu_series.json", series)
    profile["soak_server"] = {k: result[k] for k in ("duration_s", "requests", "tg_first_tps", "tg_last_tps",
                                                      "temp_max_c", "sm_clock_min_mhz", "power_max_w", "verdict")}
    write_json(path, profile)
    return result
