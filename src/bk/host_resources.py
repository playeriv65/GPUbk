"""Declared host budgets for cooperative admission in the laboratory fork."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from .config import Config
from .models import BookingError, STATUS_ACTIVE
from .timeparse import parse_iso


def normalize_host_budget(
    config: Config, cpus: Optional[int], memory_mb: Optional[int]
) -> dict:
    values = {}
    for key, value, capacity, default in (
        ("host_cpus", cpus, config.host_cpu_capacity, config.default_host_cpus),
        (
            "host_memory_mb",
            memory_mb,
            config.host_memory_capacity_mb,
            config.default_host_memory_mb,
        ),
    ):
        if value is None:
            value = default if capacity else 0
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BookingError(f"{key} must be a nonnegative integer")
        if capacity and (value < 1 or value > capacity):
            raise BookingError(
                f"{key} must be between 1 and configured capacity {capacity}"
            )
        if value:
            values[key] = value
    return values


def host_budget_conflict(
    ledger: dict, config: Config, start: datetime, end: datetime, budget: dict
) -> Optional[str]:
    neighbours = [
        r
        for r in ledger.get("reservations", [])
        if r.get("status") == STATUS_ACTIVE
        and parse_iso(r["start_at"]) < end
        and parse_iso(r["end_at"]) > start
    ]
    boundaries = {start, *[max(start, parse_iso(r["start_at"])) for r in neighbours]}
    for point in boundaries:
        overlapping = [
            r
            for r in neighbours
            if parse_iso(r["start_at"]) <= point < parse_iso(r["end_at"])
        ]
        for field, capacity, default in (
            ("host_cpus", config.host_cpu_capacity, config.default_host_cpus),
            (
                "host_memory_mb",
                config.host_memory_capacity_mb,
                config.default_host_memory_mb,
            ),
        ):
            if not capacity:
                continue
            total = budget.get(field, default) + sum(
                r.get(field, default) for r in overlapping
            )
            if total > capacity:
                return f"{field} budget conflict: {total} exceeds configured capacity {capacity}"
    return None
