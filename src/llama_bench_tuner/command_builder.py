"""Stable llama-bench command construction and placement metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .parsing import BenchCsvRow


@dataclass(frozen=True)
class LlamaBenchCommand:
    """Arguments shared by the legacy Grid and Optuna runners.

    This intentionally represents only options already emitted by the runners.
    New llama.cpp tuning axes belong to later phases and must not alter legacy
    command lines by default.
    """

    llama_bench: Path
    model: Path
    threads: int | None
    ngl: int
    batch: int
    ubatch: int
    prompt: int
    ngen: int
    mmap: int | None
    flash_attn: int | None
    nkvo: int | None = None
    split_mode: str | None = None
    output_format: str = "csv"
    verbose: bool = False
    option_order: tuple[str, ...] = ("flash_attn", "nkvo", "split_mode")
    depth: int | None = None
    cache_type_k: str | None = None
    cache_type_v: str | None = None
    repetitions: int | None = None
    n_cpu_moe: int | None = None
    """Pipeline-only options. ``None`` emits nothing, so a spec that does not set
    them produces exactly the historical command line."""
    legacy_mmap_flag: bool = True
    """Whether the target binary still accepts ``-mmp <0|1>``.

    Newer llama.cpp builds removed ``-mmp`` in favor of ``-lm/--load-mode
    <auto|none|mmap|mlock|mmap+mlock|dio>``. Defaults to True so existing
    (older) binaries keep their exact historical command line; callers must
    opt in via a capability probe to get the ``--load-mode`` translation.
    """


def build_llama_bench_command(spec: LlamaBenchCommand) -> list[str]:
    """Build the legacy llama-bench invocation in its established argument order."""

    command = [str(spec.llama_bench), "-m", str(spec.model)]
    if spec.threads is not None:
        command.extend(["-t", str(spec.threads)])
    command += [
        "-ngl", str(spec.ngl),
        "-b", str(spec.batch),
        "-ub", str(spec.ubatch),
        "-p", str(spec.prompt),
        "-n", str(spec.ngen),
    ]
    if spec.mmap is None:
        pass  # pipeline runs keep llama-bench's default load mode
    elif spec.legacy_mmap_flag:
        command.extend(["-mmp", str(spec.mmap)])
    else:
        command.extend(["-lm", "mmap" if spec.mmap else "none"])
    command.extend(["-o", spec.output_format])
    if spec.verbose:
        command.append("-v")
    optional = {
        "flash_attn": ("-fa", str(spec.flash_attn)) if spec.flash_attn is not None else None,
        "nkvo": ("-nkvo", str(spec.nkvo)) if spec.nkvo is not None else None,
        "split_mode": ("-sm", spec.split_mode) if spec.split_mode else None,
    }
    for name in spec.option_order:
        if name not in optional:
            raise ValueError(f"Unknown llama-bench option order key: {name}")
        option = optional[name]
        if option:
            command.extend(option)
    # Pipeline-only options come last so legacy argument order is untouched.
    if spec.depth is not None:
        command.extend(["-d", str(spec.depth)])
    if spec.cache_type_k is not None:
        command.extend(["-ctk", spec.cache_type_k])
    if spec.cache_type_v is not None:
        command.extend(["-ctv", spec.cache_type_v])
    if spec.n_cpu_moe is not None:
        command.extend(["-ncmoe", str(spec.n_cpu_moe)])
    if spec.repetitions is not None:
        command.extend(["-r", str(spec.repetitions)])
    return command


def requested_placement_json(spec: LlamaBenchCommand) -> str:
    """Serialize requested placement without claiming it was realized at runtime."""

    return json.dumps(
        {
            "n_gpu_layers": spec.ngl,
            "split_mode": spec.split_mode,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def requested_offload_json(spec: LlamaBenchCommand) -> str:
    """Serialize offload-related requests separately from the native-reported state."""

    return json.dumps(
        {"no_kv_offload": spec.nkvo}, separators=(",", ":"), sort_keys=True
    )


_NATIVE_REPORTED_PLACEMENT_FIELDS = (
    "n_gpu_layers",
    "split_mode",
    "main_gpu",
    "devices",
    "tensor_split",
    "tensor_buft_overrides",
)

_NATIVE_REPORTED_OFFLOAD_FIELDS = ("n_gpu_layers", "n_cpu_moe", "no_kv_offload")


def native_reported_placement_json(rows: Iterable[BenchCsvRow]) -> str | None:
    """Return placement fields llama-bench's own CSV self-reported, if any.

    These values are the benchmark binary's own account of its configuration,
    not an independent runtime measurement (e.g. nvidia-smi telemetry). They
    deliberately stay separate from the requested placement, which is only the
    CLI argument, not evidence of what was actually realized. True observed
    (telemetry-measured) placement is reserved for a later phase.
    """

    native_reported: list[Mapping[str, str]] = []
    for row in rows:
        item = {
            name: row.values[name]
            for name in _NATIVE_REPORTED_PLACEMENT_FIELDS
            if row.values.get(name) not in (None, "")
        }
        if item and item not in native_reported:
            native_reported.append(item)
    if not native_reported:
        return None
    value: Mapping[str, str] | list[Mapping[str, str]]
    value = native_reported[0] if len(native_reported) == 1 else native_reported
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def native_reported_offload_json(rows: Iterable[BenchCsvRow]) -> str | None:
    """Return offload fields llama-bench's own CSV self-reported, if any.

    See :func:`native_reported_placement_json`: this is the benchmark's own
    self-report, not independently measured telemetry.
    """

    native_reported: list[Mapping[str, str]] = []
    for row in rows:
        item = {
            name: row.values[name]
            for name in _NATIVE_REPORTED_OFFLOAD_FIELDS
            if row.values.get(name) not in (None, "")
        }
        if item and item not in native_reported:
            native_reported.append(item)
    if not native_reported:
        return None
    value: Mapping[str, str] | list[Mapping[str, str]]
    value = native_reported[0] if len(native_reported) == 1 else native_reported
    return json.dumps(value, separators=(",", ":"), sort_keys=True)
