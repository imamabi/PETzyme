"""Planning feasibility from declared requirements and unverified researcher constraints."""
from __future__ import annotations
from typing import get_args, get_type_hints
from .advisor_types import ExperimentCandidate, FeasibilityAssessment, ResearcherConstraints, Resource
from .contracts import ToolResult
from .results import outcome


def valid_constraints(value: ResearcherConstraints | None) -> bool:
    """Flow: Receives external optional ResearcherConstraints; returns bool for assess_experiment_feasibility() and run_investigation().

    Validate structured researcher constraints (None means unconstrained).

    Purpose: accepts only the typed ResearcherConstraints vocabulary
        (resource lists from the Resource enum, boolean flags, enum
        strings) so free-text instructions can never smuggle in
        requirements or override planning logic.
    Returns:
        bool; True for None or a fully well-typed constraints dict."""
    if value is None: return True
    if not isinstance(value, dict) or not value.keys() <= ResearcherConstraints.__annotations__.keys(): return False
    hints = get_type_hints(ResearcherConstraints)
    for field, supplied in value.items():
        if field in ('available_resources', 'unavailable_resources'):
            if (not isinstance(supplied, list) or len(supplied) > len(get_args(Resource))
                    or any(not isinstance(v, str) or v not in get_args(Resource) for v in supplied)):
                return False
        elif field == 'allow_new_measurements':
            if type(supplied) is not bool: return False
        elif not isinstance(supplied, str) or supplied not in get_args(hints[field]): return False
    return True


def assess_experiment_feasibility(action: ExperimentCandidate,
                                  optional_constraints: ResearcherConstraints | None = None) -> ToolResult[FeasibilityAssessment]:
    """Flow: Receives ExperimentCandidate from generate_experiment_candidates() and external optional ResearcherConstraints; returns ToolResult[FeasibilityAssessment] for recommend_next_experiments().

    Assess planning feasibility of one candidate against declared constraints.

    Purpose: reconciles the action's required resources with the
        researcher's available/unavailable lists and flags (budget,
        costs, turnaround, new measurements). Reports blockers and
        unknowns; never infers inventory, costs, timing, or success.
    Args:
        action: ExperimentCandidate from generate_experiment_candidates().
        optional_constraints: validated ResearcherConstraints or None.
    Returns:
        ToolResult[FeasibilityAssessment] with status
            CONFLICTING_CONSTRAINTS, BLOCKED, or UNKNOWN (UNKNOWN is the
            honest default — feasibility is unassessed, not feasible),
            plus 'blockers', 'unknown_requirements', and 'basis'.
    Statuses:
        SUCCESS normally; ERROR on invalid constraints."""
    sources = [s for basis in action['evidence_basis'] for s in basis['sources']]
    limits = ['Resource and constraint assertions are researcher-supplied, not independently authenticated.',
              'No cost estimate, equipment availability, timing or experimental success is inferred.']
    if not valid_constraints(optional_constraints):
        return outcome('ERROR', None, sources, limits, 'COMPUTED', 'Invalid structured researcher constraints; free-text instructions are not accepted.')
    constraints = optional_constraints or {}
    available = set(constraints.get('available_resources', []))
    unavailable = set(constraints.get('unavailable_resources', []))
    conflicts = sorted(available & unavailable)
    blockers = [f'Resource reported unavailable: {r}' for r in sorted(set(action['required_resources']) & unavailable)]
    unknown = [f'Availability of {r}' for r in action['required_resources'] if r not in available | unavailable]
    if constraints.get('budget', 'UNKNOWN') == 'UNKNOWN': unknown.append('Budget availability')
    if constraints.get('budget') == 'UNAVAILABLE': blockers.append('Budget reported unavailable; no cost or free execution is assumed.')
    if constraints.get('costs', 'UNKNOWN') == 'UNKNOWN': unknown.append('Documented costs and budget sufficiency')
    else:
        # Stating that costs exist does not supply costs or prove coverage by a budget.
        unknown.append('Cost amounts and budget sufficiency require documented reconciliation')
    if constraints.get('turnaround', 'UNKNOWN') == 'UNKNOWN': unknown.append('Turnaround time and researcher deadline')
    if constraints.get('turnaround') == 'UNACCEPTABLE': blockers.append('Turnaround reported unacceptable.')
    if action['category'] == 'PROPOSED_EXPERIMENT' and constraints.get('allow_new_measurements') is False:
        blockers.append('New measurements prohibited by supplied constraints.')
    status = 'CONFLICTING_CONSTRAINTS' if conflicts else 'BLOCKED' if blockers else 'UNKNOWN'
    result: FeasibilityAssessment = {'action_id': action['action_id'], 'status': status,
        'blockers': [f'Conflicting availability: {r}' for r in conflicts] + blockers,
        'unknown_requirements': unknown,
        'basis': 'Only action requirements and explicitly supplied constraints; no inferred laboratory inventory or cost estimates.'}
    return outcome('SUCCESS', result, sources, limits, 'COMPUTED')
