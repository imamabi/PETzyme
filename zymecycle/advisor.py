"""Deterministic dependency ordering and composition of the Experiment Advisor units."""
from __future__ import annotations

from .advisor_candidates import ADVISOR_LIMITS, generate_experiment_candidates
from .advisor_feasibility import assess_experiment_feasibility, valid_constraints
from .advisor_types import (ExperimentCandidate, ExperimentPlan, FeasibilityAssessment,
                            Recommendation, ResearcherConstraints)
from .assessment_types import ConditionInvestigation
from .contracts import Source, ToolResult
from .protein_types import SimilaritySearch
from .results import outcome


def prioritize_experiments(candidates: list[ExperimentCandidate],
                           feasibility: list[FeasibilityAssessment]) -> ToolResult[list[Recommendation]]:
    """Flow: Receives list[ExperimentCandidate] from generate_experiment_candidates() and list[FeasibilityAssessment] from assess_experiment_feasibility(); returns ToolResult[list[Recommendation]] for recommend_next_experiments().

    Order candidates topologically with a deterministic qualitative ranking.

    Purpose: emits prerequisites before dependents (cycle = error), then
        prefers EVIDENCE_VERIFICATION over PROPOSED_EXPERIMENT, unblocked
        over blocked/conflicting, RESOLVES_BLOCKER information value, and
        finally code plus stable ID as tiebreakers. Priority is an
        ordinal, never an information-gain estimate.
    Args:
        candidates: unique-action list from generate_experiment_candidates().
        feasibility: exactly one FeasibilityAssessment per candidate.
    Returns:
        ToolResult[list[Recommendation]]; each recommendation is its
            candidate plus 'feasibility', 'priority' (1-based), and
            'ranking_rationale'.
    Statuses:
        SUCCESS normally; ERROR on conflicting duplicates, missing or
            extra assessments, unknown dependencies, or cycles."""
    sources: list[Source] = []
    for candidate in candidates:
        for basis in candidate['evidence_basis']:
            for source in basis['sources']:
                if source not in sources: sources.append(source)
    try:
        actions = {}
        for action in candidates:
            key = action['action_id']
            if key in actions and actions[key] != action: raise ValueError('Conflicting duplicate action')
            actions[key] = action
        assessments = {f['action_id']: f for f in feasibility}
        if len(assessments) != len(feasibility) or set(actions) != set(assessments):
            raise ValueError('One feasibility result per unique candidate is required')
        if any(not set(c['prerequisites']) <= actions.keys() for c in actions.values()):
            raise ValueError('Unknown dependency')
        ordered: list[Recommendation] = []
        completed: set[str] = set()
        while len(completed) < len(actions):
            ready = [c for key, c in actions.items() if key not in completed and set(c['prerequisites']) <= completed]
            if not ready: raise ValueError('Cyclic action prerequisites')

            # Verification before experiments, unblocked before blocked, blockers resolved first, then stable order.
            def key(c: ExperimentCandidate) -> tuple:
                f = assessments[c['action_id']]
                return (c['category'] != 'EVIDENCE_VERIFICATION',
                        f['status'] in ('BLOCKED', 'CONFLICTING_CONSTRAINTS'),
                        c['information_value'] != 'RESOLVES_BLOCKER', c['code'], c['action_id'])

            selected = min(ready, key=key)
            assessment = assessments[selected['action_id']]
            rationale = ('Prerequisites precede dependent actions. Evidence blockers precede proposed measurements; '
                         'among ready actions in the same category, actions with no stated blocker precede blocked/conflicting actions. '
                         f"Qualitative information value: {selected['information_value']}; feasibility: {assessment['status']}. "
                         'Ties use code and stable source-bound ID; priority is an ordinal, not an information-gain estimate.')
            ordered.append({**selected, 'feasibility': assessment, 'priority': len(ordered) + 1,
                            'ranking_rationale': rationale})
            completed.add(selected['action_id'])
        return outcome('SUCCESS', ordered, sources, ADVISOR_LIMITS, 'COMPUTED')
    except (KeyError, TypeError, ValueError):
        return outcome('ERROR', None, sources, ADVISOR_LIMITS, 'COMPUTED',
                       'Candidates require consistent IDs, an acyclic complete dependency graph and one feasibility result each.')


def recommend_next_experiments(investigation: ConditionInvestigation | SimilaritySearch,
                               max_recommendations: int = 10,
                               optional_constraints: ResearcherConstraints | None = None) -> ToolResult[ExperimentPlan]:
    """Flow: Receives ConditionInvestigation from investigate_conditions() or SimilaritySearch from find_similar_nrel_proteins() and external bounds/constraints; returns ToolResult[ExperimentPlan] for project_evidence() or final researcher review.

    Recommend next experiments: generate, assess feasibility, prioritize.

    Purpose: composes the three advisor units into one HYPOTHETICAL plan.
        Candidates are generated, each gets a feasibility assessment,
        then all are ordered; the caller-visible prefix is cut at
        max_recommendations with omission counts recorded.
    Args:
        investigation: ConditionInvestigation or SimilaritySearch.
        max_recommendations: 1..10 recommendations to return.
        optional_constraints: validated ResearcherConstraints or None.
    Returns:
        ToolResult[ExperimentPlan] with 'recommendations',
            'candidate_count', 'omitted_count', 'omitted_categories',
            'basis_kind', 'scientific_status' (always PROPOSED_ONLY),
            and 'limitations'.
    Statuses:
        SUCCESS with candidates; NO_DATA when none apply; ERROR on bad
            bounds/constraints or any failed sub-step.
    Example:
        >>> inv = investigate_conditions(obs, a, b, enzyme_identifier='ESM050')['result']
        >>> out = recommend_next_experiments(inv, 3)
        >>> (out['status'], out['result']['recommendations'][0]['code'])
        ('SUCCESS', 'VERIFY_DEFINITION_AND_UNITS')
    Limitations:
        The envelope is HYPOTHETICAL even though its basis cites MEASURED
            and COMPUTED evidence; nothing here was performed."""
    if type(max_recommendations) is not int or not 1 <= max_recommendations <= 10 or not valid_constraints(optional_constraints):
        return outcome('ERROR', None, [], ADVISOR_LIMITS, 'HYPOTHETICAL',
                       'Use max_recommendations 1–10 and structured researcher constraints only.')
    generated = generate_experiment_candidates(investigation)
    if generated['result'] is None:
        return outcome('ERROR', None, generated['provenance'], ADVISOR_LIMITS, 'HYPOTHETICAL', generated['message'])
    candidates = generated['result']
    assessments = [assess_experiment_feasibility(c, optional_constraints) for c in candidates]
    if any(f['status'] == 'ERROR' for f in assessments):
        return outcome('ERROR', None, generated['provenance'], ADVISOR_LIMITS, 'HYPOTHETICAL', 'Feasibility assessment failed.')
    ordered = prioritize_experiments(candidates, [f['result'] for f in assessments])
    if ordered['result'] is None:
        return outcome('ERROR', None, generated['provenance'], ADVISOR_LIMITS, 'HYPOTHETICAL', ordered['message'])
    ranked = ordered['result']
    omitted = ranked[max_recommendations:]
    limits = ADVISOR_LIMITS.copy()
    if omitted: limits.append(f'{len(omitted)} lower-priority actions omitted by the requested bound; a short prefix may contain verification only.')
    if not ranked: limits.append('No additional evidence-grounded action is identified within the supplied investigation scope.')
    plan: ExperimentPlan = {'recommendations': ranked[:max_recommendations], 'candidate_count': len(ranked),
        'omitted_count': len(omitted), 'omitted_categories': sorted({c['category'] for c in omitted}),
        'basis_kind': 'CONDITION_INVESTIGATION' if 'conditions' in investigation else 'SEQUENCE_REPRESENTATION',
        'scientific_status': 'PROPOSED_ONLY', 'limitations': limits}
    return outcome('SUCCESS' if ranked else 'NO_DATA', plan, ordered['provenance'], limits, 'HYPOTHETICAL')
