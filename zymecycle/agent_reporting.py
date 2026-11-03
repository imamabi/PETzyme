"""Grounding/completion validation and deterministic, source-traceable report rendering."""
from __future__ import annotations

import re

from .agent_request import broad_pet_exploration, required_tool_steps, sequence_advice_only
from .agent_scope import assess_agent_scope
from .agent_types import AgentState, Fact, ResearchReport, ResearchRequest, ToolEvent, Validation
from .contracts import DISCOVERY_SCOPE_LIMIT
from .pet_accumulation import requested_condition_coverage
from .scientific_scope import PLANNING_CONSTRAINTS_ISSUE
from .source_coverage import source_receipts


def _lineage_valid(ref: str, state: AgentState, request: ResearchRequest) -> bool:
    """Reject prefiltered or leading-only subsets masquerading as the requested evidence universe."""
    origin = next((event for event in state.events if ref in event['refs'].values()), None)
    if origin is None: return False
    if origin['name'] == 'parse_nrel': return True
    if origin['name'] != 'filter_observations': return False
    args = origin['arguments']
    constraints = {'enzyme_identifier': request['enzyme_identifiers'], 'temperature_c': request['temperatures'],
                   'ph': request['ph_values'], 'pet_form': request['pet_forms']}
    for field, values in constraints.items():
        if args.get(field) is not None and (len(values) != 1 or args[field] != values[0]): return False
    return _lineage_valid(args['observations_ref'], state, request)


def _covers(event: dict, request: ResearchRequest, state: AgentState) -> bool:
    """Check requested condition coverage using executed arguments, not model assertions."""
    args = event['arguments']
    if not request['temperatures'] or not request['ph_values']: return False
    conditions = [args] if event['name'] == 'summarize_condition' else [args['condition_a'], args['condition_b']]
    for field, expected in [('temperature_c', request['temperatures']), ('ph', request['ph_values']), ('pet_form', request['pet_forms'])]:
        if expected and {condition[field] for condition in conditions} != set(expected): return False
    if not request['pet_forms'] and {c['pet_form'] for c in conditions} != {'CRYSTALLINE_POWDER', 'AMORPHOUS_FILM'}:
        return False
    if request['enzyme_identifiers'] and event['name'] == 'investigate_conditions':
        actual = state.results[event['call_id']]['result']['enzyme_identifier']
        if actual not in request['enzyme_identifiers']: return False
    return _lineage_valid(args['observations_ref'], state, request)


def required_source_fact_ids(request: ResearchRequest, events: list[ToolEvent], facts: list[Fact]) -> list[str]:
    """List the EVIDENCE/PROPOSAL fact IDs the final selection must include.

    Purpose: builds the cumulative checklist across all executed
        required evidence steps (summaries, investigations, PAZy detail,
        identity, similarity, advisor, multi-study). The model must
        select every listed ID — same_as aliases share content, not
        selection obligations — while LIMITATION/UNANSWERED facts are
        retained automatically even when omitted.
    Returns:
        list[str] of required canonical fact IDs in execution order.
    Flow: Receives ResearchRequest from interpret_request() and executed events/facts from execute_tool_call(); returns list[str] for run_investigation() model context and validate_investigation()."""
    required = required_tool_steps(request)
    by_id = {fact['id']: fact for fact in facts}
    selection_tools = {'summarize_condition', 'investigate_conditions', 'get_pazy_metadata',
        'get_associated_literature', 'compare_identity', 'find_similar_nrel_proteins',
        'recommend_next_experiments', 'investigate_pet_evidence', 'advise_next_experiment', 'discover_pet_enzymes'}
    if any(event['name'] == 'check_comparability' and event.get('actor') == 'APPLICATION_COVERAGE'
           for event in events):
        required.add('check_comparability')
        selection_tools.add('check_comparability')
    if 'exploratory_planning' in request['objectives'] or request['sources']==['NREL'] and 'multi_study' in request['objectives']:
        selection_tools.update(('filter_observations', 'assess_evidence'))
        # The executed composition's own source/proposal facts are mandatory too;
        # its validation below authenticates coverage instead of inventing steps.
        required.add('advise_next_experiment')
    if broad_pet_exploration(request):
        # Optional scientific tools still create mandatory source/proposal facts.
        required.update(('discover_pet_enzymes', 'investigate_pet_evidence', 'advise_next_experiment'))
    return [identifier for event in events
            if event['status'] != 'ERROR' and event['name'] in required & selection_tools
            for identifier in event['fact_ids'] if by_id[identifier]['kind'] in ('EVIDENCE', 'PROPOSAL')]


def validate_investigation(request: ResearchRequest, state: AgentState,
                           selection: dict | None, stop_reason: str | None = None) -> Validation:
    """Validate grounding, scope coverage, and selection completeness of an investigation.

    Purpose: independent Python check over what actually executed — never
        over model assertions. Verifies the final selection contract
        (existing IDs only, at most 100), required-step completion,
        condition/identifier/scope coverage with valid evidence lineage,
        advisor/gap linkage to the requested evidence, and inclusion of
        every required source fact ID. LIMITATION/UNANSWERED facts are
        force-retained so required caveats cannot be hidden.
    Args:
        request: ResearchRequest from interpret_request().
        state: executed AgentState with events, facts, and results.
        selection: model output dict (must be exactly {'fact_ids': [...]})
            or None when the loop stopped early.
        stop_reason: loop stop message or None.
    Returns:
        Validation with 'grounded', 'complete' (no issues at all),
            deduplicated 'issues', 'selected_fact_ids' (model selection
            plus mandatory caveats), and 'selection_counts'
            (required/selected/omitted).
    Flow: Receives ResearchRequest from interpret_request() and executed AgentState from run_investigation(); returns Validation for format_research_report()."""
    issues: list[str] = []
    facts = {fact['id']: fact for fact in state.facts}
    selected = selection.get('fact_ids') if isinstance(selection, dict) and set(selection) == {'fact_ids'} else None
    grounded = (isinstance(selected, list) and bool(selected) and len(selected) <= 100 and
                all(isinstance(identifier, str) and identifier in facts for identifier in selected))
    if not grounded:
        issues.append('Model final output was absent or failed the canonical fact-reference contract; free-form claims were withheld.')
        selected = []
    scientific_scope = assess_agent_scope(request, state, selected)
    issues.extend(scientific_scope['issues'])
    if stop_reason: issues.append(stop_reason)
    good = [event for event in state.events if event['status'] != 'ERROR']
    names = {event['name'] for event in good}
    if any(event['status'] == 'ERROR' for event in state.events):
        issues.append('One or more tool calls failed or were blocked; no substitute evidence was generated.')
    required = required_tool_steps(request)
    goals = request['objectives']
    multi = 'multi_study' in goals
    # A native advisor performs real child evidence calls. Recognize only that
    # executed capability, never unrelated operations such as discovery.
    if any(e['name'] == 'advise_next_experiment' and any(
            child['result'] is not None and child['status'] != 'ERROR'
            for child in state.results[e['call_id']]['result']['evidence_results']) for e in good):
        names.add('investigate_pet_evidence')
    if (broad_pet_exploration(request) and not request['enzyme_identifiers']
            and 'comparison' not in goals
            and any(e['name'] == 'discover_pet_enzymes' for e in good)):
        # A real scoped screen queries the pinned sources and supplies endpoint,
        # protocol and duplicate provenance; it is not a cross-source comparison.
        names.add('investigate_pet_evidence')
    coverage_advice = [state.results[e['call_id']]['result'] for e in good
                       if e['name'] == 'advise_next_experiment'
                       and state.results[e['call_id']]['result']['plan_result']['result']['basis_kind'] == 'EXPERIMENTAL_COVERAGE']
    if coverage_advice and 'exploratory_planning' in goals:
        # Native composition performs authenticated loading/parsing/assessment
        # and scoped retrieval. This is equivalent coverage, not extra tool calls.
        names.update(required & {'load_nrel', 'parse_nrel', 'filter_observations',
                                 'assess_evidence', 'investigate_pet_evidence'})
    if 'exploratory_planning' in goals:
        issues.append(PLANNING_CONSTRAINTS_ISSUE)
        if not coverage_advice and request['sources'] != ['PAZy']:
            filters = [e for e in good if e['name'] == 'filter_observations']
            if not filters or any(
                e['arguments'].get('enzyme_identifier') not in request['enzyme_identifiers']
                or not _lineage_valid(e['arguments']['observations_ref'], state, request)
                or any(e['arguments'].get(k) is not None and e['arguments'][k] not in allowed
                       for k, allowed in [('temperature_c', request['temperatures']), ('ph', request['ph_values']), ('pet_form', request['pet_forms'])])
                for e in filters
            ):
                issues.append('Exploratory planning evidence was not faithfully filtered to the requested enzyme and explicit conditions.')
    if 'unsupported' in goals:
        issues.append('Prediction, mutation design and simulation are unsupported in Build 03; no outcome is generated.')
    missing = required - names
    if missing: issues.append('Required evidence steps not completed: ' + ', '.join(sorted(missing)))
    if set(goals) & {'multi_study', 'enzyme_discovery', 'exploratory_planning'}:
        pending = [r['source'] for r in source_receipts(request,state) if r['status'] in ('PENDING','UNAVAILABLE')]
        if pending:
            issues.append('Relevant source coverage incomplete: ' + ', '.join(pending))
    matching = []
    sequence_matching = []
    for event in good:
        if event['name'] == 'discover_pet_enzymes':
            value = state.results[event['call_id']]['result']
            group = value['group'] if value else None
            if not group:
                issues.append('No documented comparable discovery group was retrieved.')
            elif any(expected and expected != [group[field]] for expected, field in
                     ((request['temperatures'], 'temperature_c'), (request['ph_values'], 'ph'),
                      (request['pet_forms'], 'pet_form'))):
                issues.append('Discovery group does not match the requested exact assay filters.')
            issues.append(DISCOVERY_SCOPE_LIMIT)
        if event['name'] == 'advise_next_experiment':
            value = state.results[event['call_id']]['result']
            args = event['arguments']
            conditions = [args.get('condition_a'), args.get('condition_b')]
            exploratory = value['plan_result']['result']['basis_kind'] == 'EXPERIMENTAL_COVERAGE'
            if exploratory:
                if any(conditions) or (len(request['temperatures'])>=2 and request['ph_values'] and request['pet_forms']):
                    issues.append('Open coverage advice cannot replace explicitly requested exact conditions.')
                if request['enzyme_identifiers'] and args['enzyme_identifier'] not in request['enzyme_identifiers']:
                    issues.append('Coverage advice did not investigate the requested enzyme.')
            if (not exploratory and (len(request['enzyme_identifiers']) != 1 or args['enzyme_identifier'] != request['enzyme_identifiers'][0]
                    or not request['temperatures'] or not request['ph_values']
                    or any(set(c[field] for c in conditions) != set(expected) for field, expected in
                        [('temperature_c', request['temperatures']), ('ph', request['ph_values']),
                         ('pet_form', request['pet_forms'] or ['CRYSTALLINE_POWDER', 'AMORPHOUS_FILM'])]))):
                issues.append('Integrated advice did not cover the requested exact enzyme and condition scope.')
            if event['status'] == 'PARTIAL':
                issues.append('Integrated advice is partial: '+', '.join(value['partial_reasons']))
        if event['name'] == 'investigate_pet_evidence':
            value = state.results[event['call_id']]['result']
            query = event['arguments'].get('query') or {}
            observations = value['nrel']['observations']
            names = {o['enzyme_identifier'] for o in observations} | {query.get('protein_name')}
            if not set(request['enzyme_identifiers']) <= names:
                issues.append('Multi-study selection did not cover the requested exact protein identifier.')
            explicit_selectors = set()
            question = request['question'].casefold()
            doi = query.get('publication_doi')
            if doi and re.search(r'(?<![\w/.])(?:https://doi\.org/)?'+re.escape(doi.casefold())+r'(?=$|[\s,;!?]|[.)](?:$|\s))',question):
                explicit_selectors.add('publication_doi')
            for field in ('parent_index','assay_index','measurement_index','protein_id'):
                label = field.replace('_', r'\s*')
                if field in query and re.search(r'\b'+label+r'\s*[:=#]?\s*'+str(query[field])+r'\b',question):
                    explicit_selectors.add(field)
            if any(k not in {'protein_name','temperature_c','ph','form',*explicit_selectors} for k in query):
                issues.append('Additional source/protocol filters narrow this investigation; full question coverage is not authenticated.')
            if event['status'] == 'PARTIAL':
                issues.append('Multi-study investigation is partial: '+', '.join(value['partial_reasons']))
        if event['name'] == 'find_similar_nrel_proteins':
            args = event['arguments']
            origin = next((e for e in state.events if args['table_ref'] in e['refs'].values()), None)
            if (len(request['enzyme_identifiers']) != 1 or args['enzyme_identifier'] != request['enzyme_identifiers'][0]
                    or origin is None or origin['name'] != 'load_nrel'):
                issues.append('Sequence search did not cover the requested exact identifier and full-source table lineage.')
            else:
                sequence_matching.append(event['call_id'])
            if event['status'] == 'PARTIAL':
                issues.append('Some eligible sequence candidates failed; the bounded search is partial.')
            if any(cue in request['question'].lower() for cue in ('all nrel', 'entire dataset', 'whole dataset')):
                issues.append('Only a bounded candidate window was searched; the requested whole-dataset ranking is unsupported.')
        if event['name'] in ('summarize_condition', 'investigate_conditions'):
            if _covers(event, request, state): matching.append(event['call_id'])
            else: issues.append('An executed query used a mismatched condition, identifier or restricted evidence subset.')
    if multi and 'NREL' in request['sources']:
        retrieved = []
        for event in good:
            value = state.results[event['call_id']]['result']
            if event['name'] == 'investigate_pet_evidence': retrieved.extend(value['nrel']['observations'])
            if event['name'] == 'filter_observations':
                retrieved.extend(value)
            if event['name'] == 'advise_next_experiment':
                retrieved.extend(o for item in value['evidence_results'] if item['result'] for o in item['result']['nrel']['observations'])
        if not requested_condition_coverage(request,retrieved)['complete']:
            issues.append('Multi-study selection did not cover the requested exact conditions with actual NREL cells across executed calls.')
    if multi and request['sources'] == ['PAZy'] and any(request[k] for k in ('temperatures', 'ph_values', 'pet_forms')):
        # Nominal-condition scope projection only: these are never NREL observations
        # and cannot establish biological identity, measurement equivalence or activity.
        pazy_conditions = []
        for event in good:
            if event['name'] != 'investigate_pet_evidence':
                continue
            value = state.results[event['call_id']]['result']
            for card in value['pazy']['records']:
                material = card['context']['material']
                form = {'crystalline powder': 'CRYSTALLINE_POWDER', 'amorphous film': 'AMORPHOUS_FILM'}.get(
                    str(material.get('morphology')) + ' ' + str(material.get('form')))
                pazy_conditions.append({'enzyme_identifier': value['query'].get('protein_name', card['protein_name']),
                    'temperature_c': card['context']['conditions'].get('temperature_c'),
                    'ph': card['context']['conditions'].get('ph'), 'pet_form': form})
        if not requested_condition_coverage(request, pazy_conditions, basis='PAZY_ASSAYS')['complete']:
            issues.append('PAZy-only selection did not cover every requested nominal condition with executed assay records.')
        if 'experiment_advice' in goals:
            issues.append('PAZy-only exact-condition advice is not implemented; retrieved assay evidence is not a completed experiment plan.')
    if required & {'summarize_condition', 'investigate_conditions'} and not matching:
        issues.append('No scientific query covered the requested conditions and evidence scope.')
    if 'assess_evidence' in required and not coverage_advice and not any(event['name'] == 'assess_evidence' and
            _lineage_valid(event['arguments']['observations_ref'], state, request) for event in good):
        issues.append('Reliability assessment did not cover the requested evidence scope.')
    if 'identify_evidence_gaps' in required and not any(
        event['name'] == 'identify_evidence_gaps' and event['arguments']['investigation_ref'].split(':')[0] in matching for event in good):
        issues.append('Evidence gaps were not linked to the requested investigation.')
    if 'recommend_next_experiments' in required:
        allowed_origins = sequence_matching if sequence_advice_only(request) else matching
        advisor_events = [event for event in good if event['name'] == 'recommend_next_experiments']
        if not advisor_events or any(event['arguments']['investigation_ref'].split(':')[0] not in allowed_origins for event in advisor_events):
            issues.append('Experiment advice was not linked to the requested source-backed evidence scope.')
    # The same exact ID set is published to the model; no required fact is filled in automatically.
    required_ids = set(required_source_fact_ids(request, state.events, state.facts))
    omitted_ids = required_ids - set(selected)
    if omitted_ids:
        issues.append('Final selection omitted required source evidence facts.')
    for event in good:
        if event['name'] == 'get_associated_literature' and event['status'] == 'PARTIAL':
            issues.append('PAZy reference retrieval is bounded/partial; omitted or unresolved references remain explicit.')
    # Limitations and unanswered conclusions cannot be hidden by model selection.
    mandatory = [fact['id'] for fact in state.facts if fact['kind'] in ('LIMITATION', 'UNANSWERED')]
    return {'grounded': bool(grounded), 'complete': not issues,
            'scientific_scope': scientific_scope,
            'issues': list(dict.fromkeys(issues)), 'selected_fact_ids': list(dict.fromkeys([*selected, *mandatory])),
            'selection_counts': {'required':len(required_ids), 'selected':len(required_ids)-len(omitted_ids),
                                 'omitted':len(omitted_ids)}}


def format_research_report(report: ResearchReport, *, public: bool = True) -> str:
    """Render a source-traceable text report from selected canonical facts.

    Purpose: deterministic rendering of the validated selection in four
        sections (Observed evidence; Proposed investigations; Limitations;
        Unanswered/unsupported), each fact with its ID, evidence
        category, text, and source. Duplicate texts print once; generic
        limitations are suppressed when specific COMPUTED ones exist.
    Args:
        report: ResearchReport from run_investigation().
        public: True (default) withholds ephemeral PAZy-derived facts
            and prints only aggregate reference counts; False includes
            them for transient in-memory use only — never files or logs.
    Returns:
        Multiline report string ending with validation/incomplete notes.
    Limitations:
        Returned reports involving live PAZy data are marked
            FORBIDDEN_PAZY_DERIVED; never save or log those objects.
    Flow: Receives ResearchReport and Validation from run_investigation(); returns str as a final source-traceable report, with PAZy records redacted by default."""
    lines = [f"PETzyme — investigation {report['status']}", report['question'],
             f"Backend: {report['provider_mode']}; model requests {report['model_requests']}; tool attempts {report['tool_attempts']}."]
    chosen = set(report['validation']['selected_fact_ids'])
    scope = report['validation'].get('scientific_scope')
    if scope and scope['background_only']:
        lines.append('\nScientific scope — background only; not direct support for the requested comparison.')
        lines.extend(scope['explanations'])
    specific_limits = any(f['id'] in chosen and f['kind'] == 'LIMITATION' and f['evidence_category'] == 'COMPUTED' for f in report['facts'])
    sections = [('EVIDENCE', 'Background evidence only' if scope and scope['issues'] else 'Observed evidence')]
    if any(f['kind'] == 'PROPOSAL' for f in report['facts']):
        sections.append(('PROPOSAL', 'Proposed investigations (not measurements)'))
    sections.extend([('LIMITATION', 'Limitations'), ('UNANSWERED', 'Unanswered / unsupported')])
    for kind, title in sections:
        lines.append('\n' + title)
        shown = 0
        printed: set[str] = set()
        for fact in report['facts']:
            if fact['id'] not in chosen or fact['kind'] != kind or (public and fact['ephemeral']): continue
            if kind == 'LIMITATION' and specific_limits and fact['evidence_category'] is None and not fact['ephemeral']: continue
            if fact['text'] in printed: continue
            printed.add(fact['text'])
            source = '; '.join(f"{s['name']}: {s['url']}" for s in fact['provenance'])
            lines.append(f"- [{fact['id']}] [{fact['evidence_category'] or 'METADATA'}] {fact['text']} Source: {source or 'execution status only'}")
            shown += 1
        if not shown: lines.append('- No validated public facts in this section.')
    if public and report['persistence'] == 'FORBIDDEN_PAZY_DERIVED':
        lines.append('\nPAZy record values remain in the transient report object and are withheld from public output.')
        for event in report['trace']:
            if event['name'] == 'get_associated_literature':
                lines.append(f"- PAZy reference status {event['status']}; counts {event['preview'].get('reference_counts')}. Source: https://www.pazy.eu/")
    for issue in report['validation']['issues']: lines.append('- Incomplete/validation: ' + issue)
    return '\n'.join(lines) + '\n'
