"""Conservative comparison of conditions and documented measurement definitions."""
from __future__ import annotations

from decimal import Decimal, DecimalException, localcontext

from .assessment_types import (
    ComparabilityAssessment,
    MetadataRecord,
    NumericalComparison,
    VaryingVariable,
)
from .contracts import Observation, ToolResult
from .evidence import assess_evidence
from .experimental import (
    LIMITATIONS,
    PROTOCOL_FIELDS,
    details_for,
    metadata_index,
    source_key,
    unique_observations,
)
from .results import outcome


def check_comparability(observations: list[Observation], metadata: list[MetadataRecord] | None = None,
                        varying_variable: VaryingVariable | None = None) -> ToolResult[ComparabilityAssessment]:
    """Classify whether observations support a quantitative comparison.

    Purpose: compares condition fields, source identity, measurement
        definitions/units, experimental groups, and protocol fields across
        the supplied cells. Any difference outside the single permitted
        varying variable blocks comparison; any missing field caps it at
        LIMITED_COMPARISON.
    Args:
        observations: list from parse_nrel()/filter_observations().
        metadata: optional documented MetadataRecords (see assess_evidence).
        varying_variable: None, 'temperature_c', or 'ph'. Powder vs film
            can never be the varying variable.
    Returns:
        ToolResult[ComparabilityAssessment] with 'category' (one of
            COMPARABLE_WITHIN_DOCUMENTED_GROUP, LIMITED_COMPARISON,
            NOT_COMPARABLE, INSUFFICIENT_INFORMATION),
            'differing_fields', 'blocking_differences', 'missing_fields',
            and the 'quantitative_comparison_supported' /
            'descriptive_within_column_supported' flags.
    Statuses:
        SUCCESS/NO_DATA mirror assess_evidence; ERROR on invalid metadata
            or varying variable.
    Example:
        >>> obs = parse_nrel(load_nrel()['result'])['result']['observations']
        >>> out = check_comparability(
        ...     filter_observations(obs, enzyme_identifier='ESM050',
        ...         ph=6.5, pet_form='CRYSTALLINE_POWDER')['result'],
        ...     varying_variable='temperature_c')
        >>> out['result']['category']
        'LIMITED_COMPARISON'
    Limitations:
        The NREL CSV alone always yields LIMITED_COMPARISON at best
            (units/protocols unverified); SUCCESS never means "comparable".
    Flow: Receives list[Observation] from filter_observations() and external documented metadata; returns ToolResult[ComparabilityAssessment] for investigate_conditions() or final comparison review."""
    assessment = assess_evidence(observations, metadata)
    if assessment['status'] == 'ERROR' or varying_variable not in (None, 'temperature_c', 'ph'):
        return outcome('ERROR', None, assessment['provenance'], LIMITATIONS, 'COMPUTED', 'Invalid comparison evidence or varying variable')
    selected, index = unique_observations(observations), metadata_index(metadata)
    details = [details_for(o, index) for o in selected]
    differences: list[str] = []
    missing: list[str] = []
    for field in ('temperature_c', 'ph', 'pet_form'):
        if len({o[field] for o in selected}) > 1:
            differences.append(field)
    if len({source_key(o) for o in selected}) > 1:
        differences.append('source')
    if any(not all(source_key(o)) for o in selected):
        missing.append('source')
    for field in ('measurement_definition', 'measurement_unit', 'experimental_group', 'protocol_id'):
        values = [d.get(field) for d in details]
        if any(not value for value in values):
            missing.append(field)
        if len({value for value in values if value}) > 1:
            differences.append(field)
    if any(d.get('units_verified') is not True for d in details):
        missing.append('verified_units')
    for field in PROTOCOL_FIELDS:
        values = [d.get('protocol', {}).get(field) for d in details]
        if any(not value for value in values):
            missing.append(f'protocol.{field}')
        if len({value for value in values if value}) > 1:
            differences.append(f'protocol.{field}')
    # Isolating one variable is meaningless when the two sides come from different protein rows.
    if varying_variable in differences and len({(*source_key(o), o['source_row']) for o in selected}) > 1:
        differences.append('protein_record')
    blockers = [field for field in differences if field != varying_variable]
    enough = len(selected) >= 2 and 'source' not in missing
    if not enough:
        category = 'INSUFFICIENT_INFORMATION'
    elif blockers:
        category = 'NOT_COMPARABLE'
    elif missing:
        category = 'LIMITED_COMPARISON'
    else:
        category = 'COMPARABLE_WITHIN_DOCUMENTED_GROUP'
    same_column = (enough and not any(field in differences for field in ('temperature_c', 'ph', 'pet_form', 'source'))
                   and len({o['source_column'] for o in selected}) == 1 and not blockers)
    result: ComparabilityAssessment = {
        'category': category, 'observation_count': len(selected), 'differing_fields': differences,
        'blocking_differences': blockers, 'missing_fields': missing, 'varying_variable': varying_variable,
        'descriptive_within_column_supported': same_column,
        'quantitative_comparison_supported': category == 'COMPARABLE_WITHIN_DOCUMENTED_GROUP',
        'issues': assessment['result']['issues'],
    }
    if result['quantitative_comparison_supported'] and len(selected) == 2:
        try:
            a, b = (Decimal(o['original_text']) for o in selected)
            if a.is_finite() and b.is_finite():
                with localcontext() as ctx:
                    ctx.prec = 28
                    difference = b - a
                    ratio = b / a if a else None
                    percentage = difference / a * 100 if a else None
                numeric: NumericalComparison = {
                    'original_a': selected[0]['original_text'], 'original_b': selected[1]['original_text'],
                    'difference_b_minus_a': str(difference), 'ratio_b_over_a': str(ratio) if ratio is not None else None,
                    'percent_change_from_a': str(percentage) if percentage is not None else None,
                    'unit': details[0]['measurement_unit'], 'evidence_category': 'COMPUTED',
                    'metadata_basis': 'CALLER_SUPPLIED_SOURCE_BOUND_NOT_INDEPENDENTLY_AUTHENTICATED',
                }
                result['numerical_comparison'] = numeric
        except (DecimalException, ValueError, KeyError, TypeError):
            pass
    limitations = [*LIMITATIONS,
                   'Within-column descriptive ordering is not verified unit equivalence or enzyme performance prediction.',
                   'An explicitly varied temperature or pH may differ within a documented controlled group; PET form is never exempted.',
                   'Comparable measurements alone do not establish causal attribution or statistical significance.']
    if 'numerical_comparison' in result:
        limitations.append('Numerical differences are descriptive; caller-supplied metadata has not been independently authenticated, and no significance or causal effect is established.')
    return outcome('SUCCESS' if enough else 'NO_DATA', result, assessment['provenance'], limitations, 'COMPUTED')
