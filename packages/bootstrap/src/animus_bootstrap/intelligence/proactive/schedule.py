"""Schedule parsing for proactive checks — pure Python, no external deps."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


@dataclass
class ScheduleResult:
    """When the next check should fire."""

    next_fire: datetime
    interval_seconds: float | None = None  # For interval-based schedules


_INTERVAL_RE = re.compile(r"^every\s+(\d+)\s*([smhd])$", re.IGNORECASE)

_UNIT_MULTIPLIERS: dict[str, float] = {
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
}


class ScheduleParser:
    """Parses cron expressions and interval strings."""

    @staticmethod
    def parse_interval(spec: str) -> float:
        """Parse 'every Nm', 'every Nh', 'every Ns' to seconds.

        Supports: 'every 30m', 'every 2h', 'every 60s', 'every 1d'

        Raises:
            ValueError: If the spec is not a valid interval string.
        """
        match = _INTERVAL_RE.match(spec.strip())
        if not match:
            msg = f"Invalid interval spec: {spec!r}. Expected 'every <N><s|m|h|d>'."
            raise ValueError(msg)
        value = int(match.group(1))
        unit = match.group(2).lower()
        return value * _UNIT_MULTIPLIERS[unit]

    @staticmethod
    def parse_cron(expression: str) -> dict[str, list[int]]:
        """Parse a 5-field cron expression into component dict.

        Returns: {minute, hour, day, month, weekday} with parsed values as lists.
        Supports: *, specific numbers, ranges (1-5), steps (*/15), lists (1,3,5)

        Raises:
            ValueError: If the expression does not have exactly 5 fields.
        """
        fields = expression.strip().split()
        if len(fields) != 5:
            msg = f"Cron expression must have 5 fields, got {len(fields)}: {expression!r}"
            raise ValueError(msg)

        names = ("minute", "hour", "day", "month", "weekday")
        ranges = (
            (0, 59),
            (0, 23),
            (1, 31),
            (1, 12),
            (0, 6),
        )

        result: dict[str, list[int]] = {}
        for name, field_str, (lo, hi) in zip(names, fields, ranges, strict=True):
            result[name] = _parse_cron_field(field_str, lo, hi)
        return result

    @staticmethod
    def next_cron_fire(expression: str, after: datetime | None = None) -> datetime:
        """Calculate next fire time for a cron expression.

        Simple implementation: iterates minute-by-minute from 'after' until match.
        Cap search at 48 hours to prevent infinite loops.

        Raises:
            ValueError: If no matching time found within 48 hours.
        """
        parsed = ScheduleParser.parse_cron(expression)
        if after is None:
            after = datetime.now(UTC)

        # Start from the next full minute
        candidate = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
        cap = after + timedelta(hours=48)

        while candidate <= cap:
            if (
                candidate.minute in parsed["minute"]
                and candidate.hour in parsed["hour"]
                and candidate.day in parsed["day"]
                and candidate.month in parsed["month"]
                and candidate.weekday() in parsed["weekday"]
            ):
                return candidate
            candidate += timedelta(minutes=1)

        msg = f"No matching cron time found within 48h for: {expression!r}"
        raise ValueError(msg)

    @staticmethod
    def is_interval(spec: str) -> bool:
        """Check if spec is an interval ('every ...') vs cron."""
        return spec.strip().lower().startswith("every ")


def _parse_cron_field(field: str, lo: int, hi: int) -> list[int]:
    """Parse a single cron field into a list of matching integers."""
    values: set[int] = set()

    for part in field.split(","):
        part = part.strip()
        if "/" in part:
            # Step: */15 or 1-30/5
            base, step_str = part.split("/", 1)
            step = int(step_str)
            if base == "*":
                start, end = lo, hi
            elif "-" in base:
                start_str, end_str = base.split("-", 1)
                start, end = int(start_str), int(end_str)
            else:
                start, end = int(base), hi
            values.update(range(start, end + 1, step))
        elif "-" in part:
            # Range: 1-5
            start_str, end_str = part.split("-", 1)
            start, end = int(start_str), int(end_str)
            values.update(range(start, end + 1))
        elif part == "*":
            values.update(range(lo, hi + 1))
        else:
            values.add(int(part))

    return sorted(values)
