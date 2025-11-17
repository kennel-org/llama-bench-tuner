from __future__ import annotations

import csv
from typing import Iterable, Optional, Tuple


TokenSpeeds = Tuple[Optional[float], Optional[float]]


def extract_tps_from_csv(lines: Iterable[str]) -> TokenSpeeds:
    """Return (prefill_tps, decode_tps) parsed from a llama-bench CSV dump."""
    materialized = list(lines)
    if not materialized:
        return None, None

    header_idx = _find_header_index(materialized)
    if header_idx is None:
        return None, None

    reader = csv.DictReader(materialized[header_idx:])
    prefill_tps: Optional[float] = None
    decode_tps: Optional[float] = None

    for row in reader:
        tps = _coerce_float(
            row.get("t/s")
            or row.get("tps")
            or row.get("tok/s")
            or row.get("avg_ts")
        )
        if tps is None and row.get("avg_ns"):
            avg_ns = _coerce_float(row["avg_ns"])
            if avg_ns:
                tps = 1e9 / avg_ns
        if tps is None:
            continue

        phase = (row.get("type") or row.get("Type") or row.get("phase") or "").lower()
        if phase:
            if "pp" in phase or "prompt" in phase:
                prefill_tps = max(prefill_tps or 0.0, tps)
            if "tg" in phase or "decode" in phase or "gen" in phase:
                decode_tps = max(decode_tps or 0.0, tps)
            continue

        # Fallback: rely on token counters
        n_prompt = _coerce_int(row.get("n_prompt"))
        n_gen = _coerce_int(row.get("n_gen"))
        if n_prompt:
            prefill_tps = max(prefill_tps or 0.0, tps)
        if n_gen:
            decode_tps = max(decode_tps or 0.0, tps)

    return prefill_tps, decode_tps


def _find_header_index(lines: Iterable[str]) -> Optional[int]:
    for idx, line in enumerate(lines):
        if "," not in line:
            continue
        lower = line.lower()
        if any(key in lower for key in ("t/s", "tok/s", "avg_ts", "avg_ns")):
            return idx
    return None


def _coerce_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Optional[str]) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0
