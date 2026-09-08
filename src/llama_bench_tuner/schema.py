"""Common result fields shared by llama-bench-tuner runners."""

from __future__ import annotations

import os
import platform
import socket
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1

# Keep the legacy fields first in CSV output; these fields are appended by the
# runner so existing consumers continue to work.
COMMON_RESULT_FIELDS = [
    "schema_version",
    "host",
    "gpu",
    "gpu_connection",
    "platform",
    "python",
    "backend",
    "llama_bench",
    "status",
    "model",
    "model_size_bytes",
    "quant",
    "context",
    "context_depth",
    "prompt_tokens",
    "generated_tokens",
    "batch",
    "ubatch",
    "kv_type",
    "mtp",
    "speculation",
    "moe_mode",
    "requested_offload",
    "requested_placement",
    "native_reported_offload",
    "native_reported_placement",
    "pp_tps",
    "tg_tps",
    "ttft_ms",
    "wall_time_s",
    "vram_peak_mb",
    "ram_peak_mb",
    "success",
    "error",
    "skip_reason",
]


def environment_metadata() -> dict[str, str]:
    """Return non-secret host labels supplied by the environment.

    Hardware labels are intentionally opt-in. The runner must not guess GPU
    identity or connection topology from a device index.
    """

    return {
        "host": os.environ.get("BENCH_HOST", socket.gethostname()),
        "gpu": os.environ.get("BENCH_GPU", ""),
        "gpu_connection": os.environ.get("BENCH_GPU_CONNECTION", ""),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }


def enrich_result(result: dict[str, Any], *, model: Path, llama_bench: Path,
                  prompt: int, ngen: int, backend: str = "llama.cpp/llama-bench",
                  requested_offload: str | None = None,
                  requested_placement: str | None = None,
                  native_reported_offload: str | None = None,
                  native_reported_placement: str | None = None) -> dict[str, Any]:
    """Add stable common-schema fields without removing legacy fields."""

    model_size = model.stat().st_size if model.exists() else None
    enriched = {field: None for field in COMMON_RESULT_FIELDS}
    enriched.update(result)
    enriched.update({
        "schema_version": SCHEMA_VERSION,
        **environment_metadata(),
        "backend": backend,
        "llama_bench": str(llama_bench),
        "status": result.get("status", ""),
        "model": str(model),
        "model_size_bytes": model_size,
        "prompt_tokens": prompt,
        "generated_tokens": ngen,
        "batch": result.get("b"),
        "ubatch": result.get("ub"),
        "pp_tps": result.get("prefill_tps", 0.0),
        "tg_tps": result.get("decode_tps", 0.0),
        "success": result.get("ok", False),
        "wall_time_s": result.get("elapsed_sec", 0.0),
        "error": result.get("error", ""),
        "requested_offload": requested_offload,
        "requested_placement": requested_placement,
        "native_reported_offload": native_reported_offload,
        "native_reported_placement": native_reported_placement,
    })
    return enriched


def csv_fields(legacy_fields: list[str]) -> list[str]:
    """Return legacy fields followed by unique common-schema fields."""

    return legacy_fields + [field for field in COMMON_RESULT_FIELDS if field not in legacy_fields]
