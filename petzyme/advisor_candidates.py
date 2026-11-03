"""Evidence-linked action generation; measurement interpretation stays in existing instruments."""
from __future__ import annotations

import hashlib
import json
from .advisor_types import EvidenceBasis, ExperimentCandidate
from .assessment_types import ConditionInvestigation
from .contracts import Source, ToolResult
from .experimental import cell_locator, unique_observations, valid_condition
from .gaps import identify_evidence_gaps
from .protein_types import SimilaritySearch
from .results import outcome

ADVISOR_LIMITS = [
    'Recommendations are HYPOTHETICAL proposals, not measurements, simulations or predictions of success.',
    'Missing metadata does not establish that a procedure was absent; source verification may resolve a gap.',
    'No response, cost, replicate count, reagent concentration or incubation duration is proposed.',
    'Units and measurement definitions require source reconciliation; no conversion or superiority claim is made.',
    'Feasibility concerns planning requirements, not biological success; supplied constraints are unverified data.',
]
ASSAY_RESOURCES = ['ASSAY_PROTOCOL', 'ENZYME_MATERIAL', 'PET_MATERIAL',
                   'TEMPERATURE_CONTROL', 'PH_CONTROL', 'RESPONSE_MEASUREMENT', 'LAB_ACCESS']
CONTROLS = ['enzyme identity', 'enzyme preparation', 'PET form, crystallinity, source and batch',
            'temperature', 'pH and buffer', 'enzyme and substrate loadings',
            'incubation duration', 'response definition, units, detection and normalization']
MEASUREMENTS = ['Record exact conditions and a source-linked response definition with verified units.',
                'Design independent replication and uncertainty reporting; determine the detection limit.',
                'Define substrate-only blanks and a relevant reference control before measurement.']


def generate_experiment_candidates(
        investigation: ConditionInvestigation | SimilaritySearch) -> ToolResult[list[ExperimentCandidate]]:
    """Flow: Receives ConditionInvestigation from investigate_conditions() or SimilaritySearch from find_similar_nrel_proteins(); returns ToolResult[list[ExperimentCandidate]] for recommend_next_experiments().

    Generate HYPOTHETICAL verification/measurement candidates from evidence.

    Purpose: converts an investigation (or similarity search) into
        evidence-linked actions: EVIDENCE_VERIFICATION steps that resolve
        the gaps from identify_evidence_gaps, plus PROPOSED_EXPERIMENT
        actions (missing-condition measurements, controlled comparison,
        independent replication) with explicit controls, varied
        variables, and prerequisites. Each action gets a stable
        source-bound ID (code + SHA-256 prefix) so duplicates collapse.
    Args:
        investigation: ConditionInvestigation from investigate_conditions()
            or SimilaritySearch from find_similar_nrel_proteins().
    Returns:
        ToolResult[list[ExperimentCandidate]]; each candidate carries its
            code, category, evidence basis, proposed action, required
            measurements, target conditions, prerequisites, resources,
            and information value.
    Statuses:
        SUCCESS normally; ERROR when the input is inconsistent (wrong
            condition count, mismatched evidence, ambiguous records).
    Limitations:
        Proposes no response values, costs, counts, concentrations, or
            durations; missing metadata never proves a procedure absent."""
    candidates: list[ExperimentCandidate] = []
    sources: list[Source] = []
    objective = ''

    def add(code: str, action: str, basis: list[EvidenceBasis], *, experiment: bool = False,
            conditions: list | None = None, dependencies: list[str] | None = None,
            varied: list[str] | None = None, information: str | None = None) -> str:
        targets = conditions or []
        identity = json.dumps([code, targets, basis], sort_keys=True, allow_nan=False)
        action_id = code + ':' + hashlib.sha256(identity.encode()).hexdigest()[:12]
        if any(c['action_id'] == action_id for c in candidates): return action_id
        candidates.append({
            'action_id': action_id, 'code': code,
            'category': 'PROPOSED_EXPERIMENT' if experiment else 'EVIDENCE_VERIFICATION',
            'evidence_category': 'HYPOTHETICAL',
            'objective': objective,
            'evidence_basis': basis, 'proposed_action': action,
            'required_measurements': MEASUREMENTS.copy() if experiment else [action],
            'target_conditions': targets,
            'controlled_variables': [v for v in CONTROLS if not any(x in v for x in (varied or []))] if experiment else [],
            'varied_variables': varied or [],
            'expected_information': information or 'Whether source-linked documentation resolves this stated evidence gap.',
            'prerequisites': dependencies or [],
            'required_resources': ASSAY_RESOURCES.copy() if experiment else ['SOURCE_DOCUMENTATION'],
            'information_value': 'DIRECT_QUESTION' if experiment else 'RESOLVES_BLOCKER',
            'limitations': ADVISOR_LIMITS.copy() + (['The proposed comparison cannot establish superiority before measurements and uncertainty analysis.'] if experiment else []),
        })
        return action_id

    try:
        if 'conditions' in investigation:
            conditions = investigation['conditions']
            if len(conditions) != 2 or any(not valid_condition(c['condition']) for c in conditions):
                raise ValueError('Expected two valid investigated conditions')
            if any((c['status'] == 'OBSERVED') != bool(c['observations']) or
                   any(any(o[field] != c['condition'][field] for field in ('temperature_c', 'ph', 'pet_form'))
                       for o in c['observations']) for c in conditions):
                raise ValueError('Inconsistent exact-condition evidence')
            sources = investigation['source_provenance']
            if not sources or not investigation['enzyme_identifier']:
                raise ValueError('Exact source-backed enzyme identity is required')
            observations = unique_observations([o for c in conditions for o in c['observations']])
            record = investigation.get('record_identity')
            if not observations and (record is None or record['enzyme_identifier'] != investigation['enzyme_identifier']):
                raise ValueError('Missing responses require a known source record identity')
            if record and any(any(o[k] != record[k] for k in ('enzyme_identifier', 'source_row', 'csv_sha256', 'source')) for o in observations):
                raise ValueError('Record identity must agree with observed cells')
            if (any(o['enzyme_identifier'] != investigation['enzyme_identifier'] for o in observations)
                    or len({(o['csv_sha256'], o['source_row']) for o in observations}) > 1):
                raise ValueError('Ambiguous protein records are not an advisor input')
            objective = (f"Establish evidence needed to evaluate {investigation['enzyme_identifier']} at "
                         f"{[c['condition'] for c in conditions]}; retain separate PET-form branches where applicable.")
            basis: list[EvidenceBasis] = []
            for observation in observations:
                source = dict(observation['source'], locator=cell_locator(observation))
                basis.append({'category': 'MEASURED', 'statement':
                              f"{observation['enzyme_identifier']}: recorded response {observation['original_text']}; units {observation['unit_status']}.",
                              'sources': [source], 'csv_sha256': observation['csv_sha256']})
            basis.append({'category': 'COMPUTED', 'statement':
                          f"{investigation['enzyme_identifier']}: exact-condition availability {[(c['condition'], c['status']) for c in conditions]}; "
                          f"comparability {investigation['comparability']['category']}; direct comparison {investigation['direct_comparison_supported']}.",
                          'sources': ([dict(record['source'], locator=f"{record['source']['locator']}#record={record['source_row']}")]
                                      if record else sources),
                          'csv_sha256': observations[0]['csv_sha256'] if observations else record['csv_sha256']})
            gaps = identify_evidence_gaps(investigation)['result']
            prerequisites = []
            if not observations:
                prerequisites.append(add('DEFINE_ASSAY_CONTEXT',
                    'Establish the intended source-linked measurement definition, units, assay protocol and enzyme identity before obtaining new responses; no exact response is available.', basis))
            for gap in gaps:
                if gap['kind'] == 'VERIFICATION':
                    text = gap['requirement']
                    if gap['code'] == 'SEPARATE_PET_FORMS':
                        text += ' Confirm the intended PET form or retain separate form-specific assay branches; do not assume an unspecified form.'
                    prerequisites.append(add(gap['code'], text, basis))
            if not investigation['direct_comparison_supported'] and observations:
                prerequisites.append(add('REASSESS_COMPARABILITY',
                    'After source verification, rerun investigate_conditions() with documented cell-linked metadata and decide whether a matched comparison remains necessary.',
                    basis, dependencies=prerequisites.copy()))
            # Distinct substrates remain independent branches, including when the question left form unspecified.
            different_forms = len({c['condition']['pet_form'] for c in conditions}) > 1
            if different_forms and not any(c['code'] == 'SEPARATE_PET_FORMS' for c in candidates):
                prerequisites.append(add('SEPARATE_PET_FORMS',
                    'Confirm the intended PET form or retain separate form-specific assay branches; powder and film must not be pooled or treated as equivalent.', basis))
            missing_targets = []
            for gap in gaps:
                if gap['kind'] == 'MEASUREMENT_REQUIREMENT':
                    for target in gap['target_conditions']:
                        if target not in missing_targets:
                            missing_targets.append(target)
                            add('MEASURE_EXACT_CONDITION',
                                f"Propose an exact-condition measurement for {investigation['enzyme_identifier']} at {target}, conditional on confirming that substrate branch and documenting the assay.",
                                basis, experiment=True, conditions=[target], dependencies=prerequisites.copy(),
                                information='An observed response at this currently unmeasured exact condition, with uncertainty; no response value or success is predicted.')
            if (all(c['status'] == 'OBSERVED' for c in conditions) and not investigation['direct_comparison_supported']
                    and not different_forms and len({json.dumps(c['condition'], sort_keys=True) for c in conditions}) == 2):
                varied = [{'temperature_c': 'temperature', 'ph': 'pH', 'pet_form': 'PET form'}[v] for v in investigation['changed_variables']]
                add('CONTROLLED_COMPARISON',
                    f"If verification cannot establish an adequate comparison, propose matched assays for {investigation['enzyme_identifier']} at the requested conditions. "
                    'If more than one variable differs, design separate comparisons varying one factor at a time; do not attribute differences causally from the current records.',
                    basis, experiment=True, conditions=[c['condition'] for c in conditions],
                    dependencies=prerequisites.copy(), varied=varied,
                    information='Whether matched, defined response measurements and uncertainty support a descriptive comparison; no direction, effect size or causal result is predicted.')
            issue_codes = {i['code'] for i in investigation['evidence']['issues']}
            if observations and issue_codes & {'REPLICATES_UNREPORTED', 'UNCERTAINTY_UNREPORTED', 'ZERO_WITHOUT_DETECTION_LIMIT'}:
                add('INDEPENDENT_REPLICATION',
                    'If documentation leaves replication or uncertainty unresolved, plan independent replication of the existing exact-condition observations; integrate with proposed matched assays to avoid duplicate work. Keep multiple targets in separate form/condition-specific branches; do not pool results.',
                    basis, experiment=True,
                    conditions=[c['condition'] for c in conditions if c['observations']], dependencies=prerequisites.copy(),
                    information='Reproducibility, uncertainty and detection-limit interpretation of the observed response; no replicate count or outcome is assumed.')
        else:
            # This branch accepts the existing bounded representation-search contract, never a sequence or model-authored record.
            query = investigation['query_identifier']
            neighbors = investigation['neighbors']
            if not query or not neighbors: raise ValueError('A real query and at least one returned neighbor are required')
            objective = (f'Evaluate measured PET-response comparability of a selected returned representation neighbor and {query}, '
                         'under an explicitly chosen matched assay; representation similarity does not answer this experimental question.')
            basis = []
            for neighbor in neighbors:
                for source in neighbor['provenance']:
                    if source not in sources: sources.append(source)
                basis.append({'category': 'COMPUTED', 'statement':
                              f"{neighbor['identifier']} (record {neighbor['source_row']}) versus {query} (record {investigation['query_source_row']}): "
                              f"ESM-2 cosine {neighbor['cosine_similarity']}; model {investigation['model']}; bounded coverage {investigation['coverage']}.",
                              'sources': [dict(s, locator=f"{s['locator']}#record={row}&column=sequence")
                                          for s in neighbor['provenance'] if s['name'] == 'NREL'
                                          for row in (investigation['query_source_row'], neighbor['source_row'])]
                                         + [s for s in neighbor['provenance'] if s['name'] != 'NREL'],
                              'csv_sha256': investigation['csv_sha256']})
            if not sources or any(not any(s['name'] == 'NREL' for s in b['sources']) for b in basis):
                raise ValueError('NREL source locators are required')
            prep = add('DEFINE_MATCHED_SEQUENCE_ASSAY',
                f"Verify source-linked identities and choose among the returned neighbors for a matched comparison with {query}. "
                'Document candidate selection, PET form/crystallinity/source, temperature, pH, response definition/units and protocol; these assay choices are not supplied by representation similarity.', basis)
            add('TEST_REPRESENTATION_HYPOTHESIS',
                f"After documenting candidate choice and assay context, propose matched measurements of the selected returned protein and {query}, including reference and substrate-only controls; select on research rationale, not assumed activity.",
                basis, experiment=True, dependencies=[prep], varied=['enzyme identity'],
                information='Measured PET-response evidence and uncertainty for the selected protein and reference under matched conditions; whether any comparison is supported remains unknown.')
            for c in candidates:
                c['limitations'].append('Embedding cosine is a COMPUTED representation observation, not sequence identity, structural similarity, catalytic activity or functional equivalence; no superior candidate or mutation is proposed.')
        return outcome('SUCCESS', candidates, sources, ADVISOR_LIMITS, 'HYPOTHETICAL')
    except (KeyError, TypeError, ValueError, OverflowError):
        return outcome('ERROR', None, sources, ADVISOR_LIMITS, 'HYPOTHETICAL',
                       'Supply an unambiguous source-backed condition investigation or nonempty bounded similarity search.')
