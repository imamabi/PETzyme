"""Contract/adversarial tests; no paid inference or semantic entailment heuristic."""
from copy import deepcopy
import json
import pytest
from zymecycle.agent import run_investigation
from zymecycle.communication import SECTIONS, validate_explanation
from zymecycle.ui_adapter import EXAMPLES, OfflineWorkflow


def draft_from_facts(facts):
    """Construct a synthetic response to the documented extractive contract, not a model acceptance."""
    result = {key: [] for key in SECTIONS}
    def add(section, fact, field, text):
        result[section].append({'fact_id': fact['id'], 'field': field, 'text': text})
    proposals = []
    dois = set()
    for f in facts:
        if f['kind'] == 'EVIDENCE' and f['text'].startswith('PAZy publication-derived assay:'):
            summary = json.loads(f['text'].split('PAZy publication-derived assay: ', 1)[1]).get('measurement_summary')
            if summary:
                add('evidence', f, 'measurement_summary', summary)
        if f['kind'] == 'EVIDENCE' and not f['text'].startswith(('PAZy publication-derived assay:',
                'Multi-study aggregate coverage:', 'Cross-source comparability:')):
            add('evidence', f, 'statement', f['text'])
        if f['kind'] == 'UNANSWERED' or (f['kind'] == 'LIMITATION' and len(result['uncertainty']) < 3):
            add('uncertainty', f, 'statement', f['text'])
        if f['kind'] == 'PROPOSAL':
            proposals.append((f, json.loads(f['text'].split('PROPOSED (not observed): ', 1)[1])))
        for s in f.get('sources', f.get('provenance', [])):
            if s['doi'] and s['doi'] not in dois:
                add('publications', f, 'publication', s['doi']); dois.add(s['doi'])
    if proposals:
        first, value = min(proposals, key=lambda p: p[1]['priority'])
        add('next_action', first, 'proposed_action', value['proposed_action'])
        add('rationale', first, 'expected_information', value['expected_information'])
        for f, p in proposals:
            if p['category'] == 'PROPOSED_EXPERIMENT':
                for field in ('controlled_variables', 'required_measurements'):
                    add('controls', f, field, '; '.join(p[field]))
    return result


@pytest.fixture(scope='module')
def report():
    question = EXAMPLES['Integrated experiment advisor']
    return run_investigation(question, OfflineWorkflow(question))['result']


def test_real_research_advice_source_values_proposals_and_citations(report):
    result = validate_explanation(draft_from_facts(report['facts']), report)
    assert result['status'] == 'VERIFIED_EXTRACTIVE'
    assert report['status'] == 'PARTIAL'
    assert result['persistence'] == 'TRANSIENT'
    measured = [s for s in result['sections']['evidence'] if s['evidence_category'] == 'MEASURED']
    assert len(measured) == 4
    assert sum(s['field'] == 'measurement_summary' for s in measured) == 2
    assert '0.08499999999999999' in measured[0]['text'] and '0.13' in measured[1]['text']
    assert all(s['fact_id'] and s['sources'] for group in result['sections'].values() for s in group)
    assert result['sections']['next_action'][0]['evidence_category'] == 'HYPOTHETICAL'
    assert result['sections']['next_action'][0]['proposal_context']['priority'] == 1
    assert all(s['proposal_context']['feasibility']['status'] == 'UNKNOWN' for s in result['sections']['controls'])
    assert {s['text'] for s in result['sections']['publications']} == {'10.5281/zenodo.15417757','10.1021/acscatal.5c03460'}


@pytest.mark.parametrize('text', ['60°C proves superior PET activity.', 'Temperature causes improved performance.',
    'The response is 130 µmol/mg.', 'Use 5 replicates at 70°C.', 'not measured activity'.replace('not ', '')])
def test_citation_does_not_validate_unsupported_paraphrases(report, text):
    draft = draft_from_facts(report['facts'])
    draft['evidence'][0]['text'] = text
    result = validate_explanation(draft, report)
    assert result['status'] == 'DETERMINISTIC_FALLBACK'
    assert result['reason_code'] == 'UNVERIFIED_WORDING_OR_ROLE'
    assert result['sections'] == {} and text not in json.dumps(result)


@pytest.mark.parametrize('mode', ['missing_id', 'fake_id', 'missing_text', 'extra_claim', 'category_swap', 'wrong_section'])
def test_malformed_or_missing_citations_and_role_laundering(report, mode):
    draft = draft_from_facts(report['facts']); first = draft['evidence'][0]
    if mode == 'missing_id': first.pop('fact_id')
    if mode == 'fake_id': first['fact_id'] = 'invented'
    if mode == 'missing_text': first.pop('text')
    if mode == 'extra_claim': first['conclusion'] = 'Superior'
    if mode == 'category_swap': first['evidence_category'] = 'MEASURED'
    if mode == 'wrong_section': draft['next_action'] = [first]
    assert validate_explanation(draft, report)['status'] == 'DETERMINISTIC_FALLBACK'


@pytest.mark.parametrize('section,reason', [('evidence','EVIDENCE_COVERAGE_INCOMPLETE'),
    ('uncertainty','UNCERTAINTY_OMITTED'),('next_action','PRIORITY_NOT_PRESERVED'),
    ('rationale','RATIONALE_OMITTED'),('controls','CONTROLS_OMITTED'),('publications','PUBLICATIONS_OMITTED')])
def test_incomplete_communication_falls_back_without_losing_deterministic_answer(report, section, reason):
    draft = draft_from_facts(report['facts']); draft[section] = []
    result = validate_explanation(draft, report)
    assert result['status'] == 'DETERMINISTIC_FALLBACK' and result['reason_code'] == reason
    assert report['synthesis']['status'] == 'VALIDATED_FACT_SELECTION'


def test_no_draft_bad_scientific_selection_or_wrong_scope_falls_back(report):
    assert validate_explanation(None, report)['reason_code'] == 'NOT_REQUESTED_OR_ABSENT'
    modified = deepcopy(report); modified['synthesis']['status'] = 'INCOMPLETE'
    assert validate_explanation(draft_from_facts(report['facts']), modified)['reason_code'] == 'SCIENTIFIC_GROUNDING_INCOMPLETE'
    modified = deepcopy(report); modified['validation']['issues'].append('Wrong scope')
    assert validate_explanation(draft_from_facts(report['facts']), modified)['reason_code'] == 'SCIENTIFIC_SCOPE_INCOMPLETE'


def test_publication_fabrication_raw_record_and_duplicate_rejected(report):
    draft = draft_from_facts(report['facts']); draft['publications'][0]['text'] = '10.0000/invented'
    assert validate_explanation(draft, report)['status'] == 'DETERMINISTIC_FALLBACK'
    draft = draft_from_facts(report['facts'])
    raw = next(f for f in report['facts'] if f['text'].startswith('PAZy publication-derived assay:'))
    draft['evidence'].append({'fact_id':raw['id'], 'field':'statement', 'text':raw['text']})
    assert validate_explanation(draft, report)['status'] == 'DETERMINISTIC_FALLBACK'
    draft = draft_from_facts(report['facts']); draft['evidence'].append(draft['evidence'][0])
    assert validate_explanation(draft, report)['reason_code'] == 'DUPLICATE_STATEMENT'


def test_unit_conversion_and_negation_change_rejected_with_genuine_citation(report):
    for mutation in ('value', 'negation'):
        draft = draft_from_facts(report['facts'])
        if mutation == 'value':
            entry = next(i for i in draft['evidence'] if '0.13' in i['text'])
            entry['text'] = entry['text'].replace('0.13','130 µmol/mg')
        else:
            entry = next(i for i in draft['uncertainty'] if 'unsupported' in i['text'])
            entry['text'] = entry['text'].replace('unsupported','supported')
        assert validate_explanation(draft, report)['reason_code'] == 'UNVERIFIED_WORDING_OR_ROLE'


def test_size_bound_and_unselected_citation_fail_closed(report):
    draft = draft_from_facts(report['facts']); draft['evidence'][0]['text'] = 'x' * 65537
    assert validate_explanation(draft, report)['reason_code'] == 'EXPLANATION_TOO_LARGE'
    copied = deepcopy(report)
    draft = draft_from_facts(report['facts'])
    first = draft['evidence'][0]['fact_id']
    copied['validation']['selected_fact_ids'].remove(first)
    assert validate_explanation(draft, copied)['reason_code'] == 'MISSING_OR_INVALID_CITATION'


def test_unanswered_claim_cannot_be_omitted_behind_generic_caveats(report):
    draft = draft_from_facts(report['facts'])
    unanswered = next(f['id'] for f in report['facts'] if f['kind']=='UNANSWERED')
    draft['uncertainty'] = [i for i in draft['uncertainty'] if i['fact_id'] != unanswered]
    assert validate_explanation(draft, report)['reason_code'] == 'UNCERTAINTY_COVERAGE_INCOMPLETE'
