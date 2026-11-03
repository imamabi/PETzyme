"""Exact condition investigation, without interpolation, ranking or causal inference."""
from __future__ import annotations

from .assessment_types import ConditionInvestigation, MetadataRecord, VaryingVariable
from .comparability import check_comparability
from .contracts import Condition, Observation, ToolResult
from .evidence import assess_evidence
from .experimental import (LIMITATIONS, provenance, source_key, unique_observations, valid_condition)
from .observations import filter_observations
from .results import outcome


def investigate_conditions(observations: list[Observation], condition_a: Condition, condition_b: Condition,
                           enzyme_identifier: str | None = None, selected_record: Observation | None = None,
                           metadata: list[MetadataRecord] | None = None) -> ToolResult[ConditionInvestigation]:
    """Investigate one enzyme at two exact conditions, with support flags.

    Purpose: retrieves the exact observations for condition_a and
        condition_b (OBSERVED or MISSING each), identifies changed
        variables, and runs comparability plus evidence assessments. The
        single varying variable is temperature or pH only when exactly one
        of those changed; a PET-form change is never a varying variable.
    Args:
        observations: list from parse_nrel().
        condition_a: first Condition (temperature_c, ph, pet_form).
        condition_b: second Condition.
        enzyme_identifier: exact label; supply exactly one of this or
            selected_record. Fails when the label maps to multiple rows.
        selected_record: one Observation pinning the source row; used when
            the label is ambiguous.
        metadata: optional documented MetadataRecords.
    Returns:
        ToolResult[ConditionInvestigation] with 'conditions' (per-side
            observations/status), 'changed_variables', 'comparability',
            'evidence', 'direct_comparison_supported',
            'causal_attribution_supported' (always False), 'uncertainties',
            and 'unsupported' (explicit non-conclusions).
    Statuses:
        SUCCESS when both sides observed; PARTIAL when some evidence
            exists; NO_DATA when neither side has observations; ERROR on
            bad inputs, ambiguous identity, or conflicting cells.
    Example:
        >>> obs = parse_nrel(load_nrel()['result'])['result']['observations']
        >>> a = {'temperature_c': 40, 'ph': 6.5, 'pet_form': 'CRYSTALLINE_POWDER'}
        >>> b = {'temperature_c': 60, 'ph': 6.5, 'pet_form': 'CRYSTALLINE_POWDER'}
        >>> out = investigate_conditions(obs, a, b, enzyme_identifier='ESM050')
        >>> (out['status'], out['result']['direct_comparison_supported'])
        ('SUCCESS', False)
    Limitations:
        Missing sides are never interpolated; no causal effect is ever
            established by this descriptive assessment.
    Flow: Receives list[Observation] from parse_nrel(), an external enzyme or selected record and two Conditions; returns ToolResult[ConditionInvestigation] for identify_evidence_gaps() or recommend_next_experiments()."""
    if ((enzyme_identifier is None) == (selected_record is None) or
            not valid_condition(condition_a) or not valid_condition(condition_b) or
            (enzyme_identifier is not None and not enzyme_identifier)):
        return outcome('ERROR', None, provenance(observations, metadata), LIMITATIONS, 'COMPUTED',
                       'Supply exactly one nonempty enzyme identifier or selected record, and two valid exact conditions')
    try:
        available = unique_observations(observations)
    except ValueError:
        return outcome('ERROR', None, provenance(observations, metadata), LIMITATIONS, 'COMPUTED', 'Conflicting observation cells')
    if selected_record is not None:
        # Pin to the exact source row: same label on another row/version must not leak into this comparison.
        scoped = [o for o in available if source_key(o) == source_key(selected_record) and
                  o['source_row'] == selected_record['source_row']]
        if any(o['enzyme_identifier'] != selected_record['enzyme_identifier'] for o in scoped):
            return outcome('ERROR', None, provenance(scoped, metadata), LIMITATIONS, 'COMPUTED',
                           'Selected record identifier conflicts with the source row')
        identifier = selected_record['enzyme_identifier']
    else:
        scoped = filter_observations(available, enzyme_identifier=enzyme_identifier)['result'] or []
        identifier = enzyme_identifier
        if len({(*source_key(o), o['source_row']) for o in scoped}) > 1:
            return outcome('ERROR', None, provenance(scoped, metadata), LIMITATIONS, 'COMPUTED',
                           'Enzyme identifier maps to multiple source records; supply selected_record')
    conditions = []
    for condition in (condition_a, condition_b):
        exact = filter_observations(scoped, **condition)
        matched = exact['result'] or []
        conditions.append({'condition': condition, 'observations': matched,
                           'status': 'OBSERVED' if matched else 'MISSING'})
    selected = unique_observations([o for entry in conditions for o in entry['observations']])
    changed = [field for field in ('temperature_c', 'ph', 'pet_form') if condition_a[field] != condition_b[field]]
    # Only a lone temperature-or-pH change may vary; powder/film or multi-field changes never isolate one factor.
    varying: VaryingVariable | None = changed[0] if len(changed) == 1 and changed[0] in ('temperature_c', 'ph') else None
    comparison = check_comparability(selected, metadata, varying_variable=varying)
    evidence = assess_evidence(selected, metadata, [condition_a, condition_b])
    sources = provenance(scoped or available, metadata)
    if selected_record is not None and selected_record['source'] not in sources:
        sources.append(selected_record['source'])
    if comparison['result'] is None or evidence['result'] is None:
        return outcome('ERROR', None, sources, LIMITATIONS, 'COMPUTED', 'Comparison metadata is invalid')
    observed_both = all(entry['status'] == 'OBSERVED' for entry in conditions)
    direct = observed_both and comparison['result']['quantitative_comparison_supported']
    unsupported = ['No causal effect is established by this descriptive assessment.']
    if not observed_both:
        unsupported.append('Missing exact-condition measurements cannot be interpolated, predicted or treated as zero.')
    if not direct:
        unsupported.append('A direct quantitative comparison is unsupported by the supplied evidence.')
    record = scoped[0] if scoped else selected_record
    record_identity = ({key: record[key] for key in ('enzyme_identifier', 'source_row', 'csv_sha256', 'source')}
                       if record is not None else None)
    result: ConditionInvestigation = {
        'record_identity': record_identity,
        'enzyme_identifier': identifier, 'conditions': conditions, 'changed_variables': changed,
        'comparability': comparison['result'], 'evidence': evidence['result'],
        'direct_comparison_supported': direct, 'causal_attribution_supported': False,
        'uncertainties': [issue['explanation'] for issue in evidence['result']['issues']],
        'unsupported': unsupported, 'source_provenance': sources,
    }
    return outcome('SUCCESS' if observed_both else 'PARTIAL' if selected else 'NO_DATA',
                   result, sources, LIMITATIONS, 'COMPUTED')
