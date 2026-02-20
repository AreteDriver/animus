"""Trace context propagation via HTTP headers.

Implements W3C Trace Context specification:
- traceparent: Required header with trace/span IDs
- tracestate: Optional vendor-specific data

See: https://www.w3.org/TR/trace-context/
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# W3C Trace Context headers
TRACEPARENT_HEADER = "traceparent"
TRACESTATE_HEADER = "tracestate"

# Also support common alternative names
TRACEPARENT_ALIASES = ["traceparent", "x-trace-id", "x-request-id"]
TRACESTATE_ALIASES = ["tracestate", "x-trace-state"]

# traceparent format: {version}-{trace_id}-{span_id}-{flags}
# Example: 00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01
TRACEPARENT_PATTERN = re.compile(r"^([0-9a-f]{2})-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$")


@dataclass
class PropagatedContext:
    """Extracted trace context from HTTP headers."""

    trace_id: str
    parent_span_id: str
    sampled: bool = True
    tracestate: str | None = None


def parse_traceparent(header: str) -> tuple[str, str, str, str] | None:
    """Parse a traceparent header.

    Args:
        header: traceparent header value

    Returns:
        Tuple of (version, trace_id, span_id, flags) or None if invalid
    """
    header = header.strip().lower()
    match = TRACEPARENT_PATTERN.match(header)
    if not match:
        return None

    version, trace_id, span_id, flags = match.groups()

    # Validate version (currently only version 00 is defined)
    if version != "00":
        # Future versions may have different formats
        # For now, still try to extract trace_id and span_id
        pass

    # Validate trace_id is not all zeros
    if trace_id == "0" * 32:
        return None

    # Validate span_id is not all zeros
    if span_id == "0" * 16:
        return None

    return version, trace_id, span_id, flags


def extract_trace_context(
    headers: dict[str, str],
    case_insensitive: bool = True,
) -> PropagatedContext | None:
    """Extract trace context from HTTP headers.

    Args:
        headers: HTTP headers dict
        case_insensitive: If True, match headers case-insensitively

    Returns:
        PropagatedContext if valid traceparent found, None otherwise
    """
    # Normalize headers if case-insensitive
    if case_insensitive:
        headers = {k.lower(): v for k, v in headers.items()}

    # Find traceparent header
    traceparent = None
    for alias in TRACEPARENT_ALIASES:
        key = alias.lower() if case_insensitive else alias
        if key in headers:
            traceparent = headers[key]
            break

    if not traceparent:
        return None

    # Parse traceparent
    parsed = parse_traceparent(traceparent)
    if not parsed:
        return None

    version, trace_id, span_id, flags = parsed

    # Check sampled flag (last bit of flags)
    sampled = (int(flags, 16) & 0x01) == 0x01

    # Get tracestate if present
    tracestate = None
    for alias in TRACESTATE_ALIASES:
        key = alias.lower() if case_insensitive else alias
        if key in headers:
            tracestate = headers[key]
            break

    return PropagatedContext(
        trace_id=trace_id,
        parent_span_id=span_id,
        sampled=sampled,
        tracestate=tracestate,
    )


def format_traceparent(
    trace_id: str,
    span_id: str,
    sampled: bool = True,
    version: str = "00",
) -> str:
    """Format a traceparent header value.

    Args:
        trace_id: 32 hex char trace ID
        span_id: 16 hex char span ID
        sampled: Whether trace is sampled
        version: Trace context version (default "00")

    Returns:
        Formatted traceparent header value
    """
    flags = "01" if sampled else "00"
    return f"{version}-{trace_id}-{span_id}-{flags}"


def inject_trace_headers(
    trace_id: str,
    span_id: str,
    sampled: bool = True,
    tracestate: str | None = None,
) -> dict[str, str]:
    """Create trace context headers for outgoing requests.

    Args:
        trace_id: Trace ID
        span_id: Current span ID
        sampled: Whether trace is sampled
        tracestate: Optional tracestate value

    Returns:
        Dict of headers to add to request
    """
    headers = {
        TRACEPARENT_HEADER: format_traceparent(trace_id, span_id, sampled),
    }

    if tracestate:
        headers[TRACESTATE_HEADER] = tracestate

    return headers


def add_gorgon_tracestate(
    existing: str | None,
    workflow_id: str | None = None,
    step_id: str | None = None,
) -> str:
    """Add Gorgon-specific data to tracestate.

    Tracestate format: vendor=value,vendor2=value2
    We use 'gorgon' as our vendor key.

    Args:
        existing: Existing tracestate value
        workflow_id: Current workflow ID
        step_id: Current step ID

    Returns:
        Updated tracestate value
    """
    # Build Gorgon state
    gorgon_parts = []
    if workflow_id:
        gorgon_parts.append(f"wf:{workflow_id}")
    if step_id:
        gorgon_parts.append(f"st:{step_id}")

    if not gorgon_parts:
        return existing or ""

    gorgon_value = ";".join(gorgon_parts)
    gorgon_entry = f"gorgon={gorgon_value}"

    if not existing:
        return gorgon_entry

    # Prepend to existing (most recent first per spec)
    # Remove any existing gorgon entry
    parts = [p.strip() for p in existing.split(",") if not p.strip().startswith("gorgon=")]
    parts.insert(0, gorgon_entry)

    return ",".join(parts)


def parse_gorgon_tracestate(tracestate: str) -> dict[str, str]:
    """Parse Gorgon-specific data from tracestate.

    Args:
        tracestate: Full tracestate header value

    Returns:
        Dict with 'workflow_id' and 'step_id' if present
    """
    result = {}

    for part in tracestate.split(","):
        part = part.strip()
        if part.startswith("gorgon="):
            value = part[7:]  # Remove 'gorgon='
            for item in value.split(";"):
                if item.startswith("wf:"):
                    result["workflow_id"] = item[3:]
                elif item.startswith("st:"):
                    result["step_id"] = item[3:]
            break

    return result
