"""Non-dominated (Pareto) filtering for benchmark rows."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence


def _value(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def dominates(a: Mapping[str, Any], b: Mapping[str, Any], *, maximize: Sequence[str],
              minimize: Sequence[str]) -> bool:
    """True if ``a`` is at least as good as ``b`` everywhere and strictly better somewhere.

    A missing metric never dominates and never counts as better, so rows with
    unmeasured values cannot silently win."""

    strictly = False
    for key in maximize:
        va, vb = _value(a, key), _value(b, key)
        if va is None or vb is None:
            return False
        if va < vb:
            return False
        strictly |= va > vb
    for key in minimize:
        va, vb = _value(a, key), _value(b, key)
        if va is None or vb is None:
            return False
        if va > vb:
            return False
        strictly |= va < vb
    return strictly


def pareto_front(rows: Iterable[Mapping[str, Any]], *, maximize: Sequence[str],
                 minimize: Sequence[str]) -> list[Mapping[str, Any]]:
    """Return rows not dominated by any other row (input order preserved)."""

    items = list(rows)
    return [
        r for i, r in enumerate(items)
        if not any(dominates(o, r, maximize=maximize, minimize=minimize)
                   for j, o in enumerate(items) if j != i)
    ]


def pareto_by_group(rows: Iterable[Mapping[str, Any]], group_key: str, *, maximize: Sequence[str],
                    minimize: Sequence[str]) -> dict[Any, list[Mapping[str, Any]]]:
    """One frontier per group value (e.g. per context depth)."""

    groups: dict[Any, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row.get(group_key), []).append(row)
    return {key: pareto_front(group, maximize=maximize, minimize=minimize)
            for key, group in groups.items()}
