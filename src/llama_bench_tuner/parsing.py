from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Tuple


TokenSpeeds = Tuple[Optional[float], Optional[float]]


@dataclass(frozen=True)
class BenchCsvRow:
    """One native llama-bench CSV row, kept intact for later case expansion."""

    values: Mapping[str, str]
    tps: Optional[float]
    n_prompt: int
    n_gen: int
    n_depth: int
    phase: str


def parse_bench_csv(lines: Iterable[str]) -> list[BenchCsvRow]:
    """Parse native rows without collapsing pp/tg or benchmark conditions."""

    materialized = list(lines)
    header_idx = _find_header_index(materialized)
    if header_idx is None:
        return []

    parsed: list[BenchCsvRow] = []
    for row in csv.DictReader(materialized[header_idx:]):
        if not row:
            continue
        values = {key: value for key, value in row.items() if key is not None}
        tps = _coerce_float(
            values.get("t/s")
            or values.get("tps")
            or values.get("tok/s")
            or values.get("avg_ts")
        )
        if tps is None and values.get("avg_ns"):
            avg_ns = _coerce_float(values["avg_ns"])
            if avg_ns:
                tps = 1e9 / avg_ns
        n_prompt = _coerce_int(values.get("n_prompt"))
        n_gen = _coerce_int(values.get("n_gen"))
        parsed.append(BenchCsvRow(
            values=values,
            tps=tps,
            n_prompt=n_prompt,
            n_gen=n_gen,
            n_depth=_coerce_int(values.get("n_depth")),
            phase=_phase(values, n_prompt=n_prompt, n_gen=n_gen),
        ))
    return parsed


def extract_tps_from_csv(lines: Iterable[str]) -> TokenSpeeds:
    """Return legacy max(prefill), max(decode) values from parsed native rows."""
    return extract_tps_from_rows(parse_bench_csv(lines))


def extract_tps_from_rows(rows: Iterable[BenchCsvRow]) -> TokenSpeeds:
    """Retain the legacy aggregate while callers can preserve each source row."""
    prefill_tps: Optional[float] = None
    decode_tps: Optional[float] = None
    for row in rows:
        tps = row.tps
        if tps is None:
            continue
        if row.phase in ("prefill", "combined"):
            prefill_tps = max(prefill_tps or 0.0, tps)
        if row.phase in ("decode", "combined"):
            decode_tps = max(decode_tps or 0.0, tps)

    return prefill_tps, decode_tps


def _phase(values: Mapping[str, str], *, n_prompt: int, n_gen: int) -> str:
    label = (values.get("type") or values.get("Type") or values.get("phase") or "").lower()
    if "pp" in label or "prompt" in label:
        return "prefill"
    if "tg" in label or "decode" in label or "gen" in label:
        return "decode"
    if n_prompt and n_gen:
        return "combined"
    if n_prompt:
        return "prefill"
    if n_gen:
        return "decode"
    return "unknown"


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
