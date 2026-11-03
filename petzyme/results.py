"""Common result envelopes; no scientific interpretation or persistence."""
from __future__ import annotations

import math

from .contracts import Evidence, Observation, Source, Status, T, ToolResult


def outcome(status: Status, result: T | None, provenance: list[Source],
            limitations: list[str], evidence: Evidence | None,
            message: str | None = None) -> ToolResult[T]:
    """Build the standard ToolResult envelope every scientific tool returns.

    Purpose: guarantees a uniform output shape (status, result, provenance,
        limitations, evidence_category, message) so callers, the agent loop,
        and the UI can handle every tool identically. Copies the limitations
        list so later mutation cannot leak across results.
    Args:
        status: 'SUCCESS', 'NO_DATA', 'PARTIAL', or 'ERROR'.
        result: typed payload, or None when there is nothing to return.
        provenance: source records attributing the payload.
        limitations: scientific caveats the caller must retain.
        evidence: 'MEASURED', 'COMPUTED', 'SIMULATED', 'HYPOTHETICAL', or None.
        message: short human-readable note; None when unneeded.
    Returns:
        ToolResult[T] dict with exactly those six keys.
    Example:
        >>> out = outcome('NO_DATA', None, [], ['Units unknown.'], 'MEASURED')
        >>> (out['status'], out['result'])
        ('NO_DATA', None)
    Flow: Receives typed status, result and provenance from parse_nrel(), get_pazy_metadata() or compare_identity(); returns ToolResult[T] as their final JSON-compatible output."""
    return {'status': status, 'result': result, 'provenance': provenance,
            'limitations': limitations.copy(), 'evidence_category': evidence, 'message': message}


def sources_from_observations(observations: list[Observation]) -> list[Source]:
    """Collect the distinct dataset-level sources behind a list of observations.

    Purpose: builds the provenance list for filtered views without
        duplicating the shared dataset source. Individual observations keep
        their own cell locators; this helper keeps only one copy per dataset.
    Returns:
        list[Source] with duplicates removed, preserving first-seen order.
    Flow: Receives list[Observation] from parse_nrel(); returns list[Source] for filter_observations() and summarize_condition()."""
    # Keep the dataset locator here; individual observations retain cell locators.
    sources: list[Source] = []
    for observation in observations:
        source = observation['source']
        if source not in sources:
            sources.append(source)
    return sources


def finite_number(value: object) -> bool:
    """Check for a finite int or float without coercion.

    Purpose: single numeric gate so NaN, infinities, numeric strings, and
        bools are rejected uniformly wherever temperatures, pH values, or
        tool arguments are validated.
    Returns:
        bool; True only for finite int/float values.
    Example:
        >>> finite_number(6.5), finite_number(float('nan')), finite_number('6.5')
        (True, False, False)
    Flow: Receives external value object; returns bool for filter_observations(), valid_condition() and _valid() numeric validation."""
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False
