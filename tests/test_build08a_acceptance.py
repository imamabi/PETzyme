"""Real local evidence acceptance; fixed test decisions do not test live Muse reasoning."""
import json

import pytest
from streamlit.testing.v1 import AppTest

from zymecycle.agent import run_investigation
from zymecycle.agent_reporting import format_research_report
from zymecycle.agent_request import interpret_request, select_tool_catalog
from zymecycle.ui_adapter import EXAMPLES, ROOT


QUESTIONS = {
    'A': 'What PET-degradation evidence exists for LCC-ICCG across NREL and PAZy?',
    'B': 'Which PAZy experiments cover conditions or publications outside the NREL study?',
    'C': 'Can ESM050 be concluded to perform better at 60°C than 40°C using the available evidence?',
    'D': 'What important evidence gaps should a researcher resolve before comparing PET hydrolases across studies?',
}
PLANS = {
    'A': [('investigate_pet_evidence', {'query': {'protein_name': 'LCC-ICCG'}, 'limit': 3})],
    'B': [('investigate_pet_evidence', {'scope': 'OUTSIDE_NREL_PUBLICATION', 'limit': 3})],
    'D': [('investigate_pet_evidence', {'limit': 3})],
    # No pH, PET form or exact experimental group was specified by question C.
    'C': [('load_nrel', {}), ('parse_nrel', {'table_ref': 'v1:table'}),
          ('filter_observations', {'observations_ref': 'v2:observations', 'enzyme_identifier': 'ESM050'}),
          ('assess_evidence', {'observations_ref': 'v3:observations'}),
          ('check_comparability', {'observations_ref': 'v3:observations'})],
}


class AcceptanceDecisions:
    mode = 'DETERMINISTIC_ACCEPTANCE_NO_API'

    def __init__(self, case: str) -> None:
        """Flow: Receives external acceptance case str; initializes fixed test decisions for complete()."""
        self.plan = PLANS[case]
        self.inputs = []

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        """Flow: Receives messages/tools from run_investigation(); returns assistant dict with fixed test calls or canonical fact IDs for run_investigation()."""
        self.inputs.append(json.loads(json.dumps(messages)))
        replies = [json.loads(m['content']) for m in messages if m['role'] == 'tool']
        index = len(replies)
        if index < len(self.plan):
            name, arguments = self.plan[index]
            return {'role': 'assistant', 'content': None, 'tool_calls': [{
                'id': f'v{index + 1}', 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(arguments)}}]}
        return {'role': 'assistant', 'content': json.dumps({
            'fact_ids': [fact['id'] for reply in replies for fact in reply['facts']]})}


@pytest.mark.parametrize('case', ['A', 'B', 'C', 'D'])
def test_requested_investigations_real_local_dispatch(case):
    backend = AcceptanceDecisions(case)
    report = run_investigation(QUESTIONS[case], backend)['result']
    assert report['status'] == 'PARTIAL' and report['validation']['grounded']
    assert [(e['name'], e['status']) for e in report['trace']] == [
        (name, 'PARTIAL' if name == 'investigate_pet_evidence' else 'SUCCESS')
        for name, _ in PLANS[case]]
    assert report['model_requests'] == len(PLANS[case]) + 1 <= 6
    assert report['tool_attempts'] == len(PLANS[case]) <= 8
    outbound = json.dumps(backend.inputs, allow_nan=False)
    assert all(key not in outbound for key in (
        'amino_acid_sequence', 'normalized_sequence', 'sequence_sha256',
        '"rows"', '"vector"', '"embedding"', '/Users/'))
    assert max(len(json.dumps(m).encode()) for m in backend.inputs) < 262144
    if case == 'C':
        assert report['persistence'] == 'NREL_ONLY'
        assert report['trace'][2]['preview'] == {
            'observation_count': 10, 'returned_observations': 5, 'omitted_observations': 5}
        assert '5 returned, 5 omitted' in format_research_report(report)
        assert any('Required evidence steps not completed' in issue for issue in report['validation']['issues'])
        assert any('UNITS_UNVERIFIED' in f['text'] for f in report['facts'])
        assert any('quantitative support: False' in f['text'] for f in report['facts'])
        assert all(not ({'ph', 'pet_form'} & set(e['arguments'])) for e in report['trace'])
        return
    p = report['trace'][0]['preview']
    assert not p['quantitative_superiority_supported'] and not p['causal_claim_supported']
    assert p['identity']['verified'] == 0 and not p['identity']['historical_checks_reused']
    assert p['units']['csv_units'].startswith('UNRESOLVED original declaration (UNKNOWN/UNVERIFIED)')
    assert 'INFERRED' in p['units']['csv_units'] and 'COMPUTED' in p['units']['csv_units']
    assert all(o['unit_status'] == 'UNKNOWN/UNVERIFIED' for o in p['nrel']['observations'])
    assert len(p['pazy']['records']) <= 3 and len(p['comparisons']) <= 8
    assert all(len(c['measurements']) <= 3 for c in p['pazy']['records'])
    assert [g['priority'] for g in p['gaps']] == [1, 2, 3, 4]
    assert report['persistence'] == 'FORBIDDEN_PAZY_DERIVED'
    public = format_research_report(report)
    assert 'PAZy publication-derived assay:' not in public
    assert all(c['locator'] not in public for c in p['pazy']['records'])
    if case == 'A':
        assert p['nrel']['total_matches'] == 11 and p['nrel']['returned'] == 3
        assert p['pazy']['total_matches'] == 0 and 'not establish absence' in p['answer']
    elif case == 'B':
        assert p['pazy']['total_matches'] == 4169 and p['pazy']['returned'] == 3
        assert p['coverage']['outside_publications'] == 90
        assert all(c['publication_doi'].casefold() != '10.1021/acscatal.5c03460' for c in p['pazy']['records'])
    else:
        assert p['coverage']['comparison_categories'] == {
            'INSUFFICIENT_INFORMATION': 771, 'NOT_COMPARABLE': 470,
            'VERIFIED_COMPARABLE': 0, 'LIMITED_COMPARISON': 0}


def test_cross_study_gap_question_exposes_existing_local_tool_only():
    request = interpret_request(QUESTIONS['D'])
    assert {'gaps', 'multi_study'} <= set(request['objectives'])
    tools = select_tool_catalog(request)
    assert {t['function']['name'] for t in tools} == {'investigate_pet_evidence', 'advise_next_experiment', 'discover_pet_enzymes'}
    assert all(not {'verification', 'metadata', 'csv_path', 'request_window'} & t['function']['parameters']['properties'].keys() for t in tools)


def test_multistudy_answer_is_outside_collapsing_progress_panel():
    at = AppTest.from_file(ROOT / 'tests/legacy_streamlit_app.py', default_timeout=30).run()
    at.button(key='example_4').click().run()
    at.button(key='investigate').click().run()
    assert not at.exception and [s.value for s in at.subheader] == ['Research findings', 'Multi-Study Evidence']
    assert not at.get('status')[0].get('subheader')
    assert not at.get('status')[0].get('info')
    assert 'view' not in at.session_state
    at.run()
    assert not at.subheader and 'view' not in at.session_state


@pytest.mark.parametrize('index', range(4))
def test_original_four_demonstrations_real_local_apptest(index, monkeypatch):
    def forbidden(*a, **k):
        pytest.fail('Offline acceptance attempted provider inference or Keychain launch')
    monkeypatch.setattr('zymecycle.meta.MetaModelClient.complete', forbidden)
    monkeypatch.setattr('zymecycle.ui_live.subprocess.Popen', forbidden)
    at = AppTest.from_file(ROOT / 'tests/legacy_streamlit_app.py', default_timeout=60).run()
    at.button(key=f'example_{index}').click().run()
    at.button(key='investigate').click().run()
    assert not at.exception
    view = at.session_state['view']
    assert view['question'] == list(EXAMPLES.values())[index]
    assert view['workflow_status'] == 'COMPLETE' and view['mode'] == 'DETERMINISTIC_OFFLINE'
    if index == 0:
        assert [o['original_text'] for o in view['observations']] == ['0.08499999999999999', '0.13']
        assert not view['comparison']['direct_comparison_supported']
    elif index == 1:
        assert not view['observations'] and view['gaps']
    elif index == 2:
        assert view['similarity']['model']['checkpoint'] == 'facebook/esm2_t12_35M_UR50D'
        assert view['similarity']['coverage']['searched_candidates'] == 8
    else:
        assert all(r['evidence_category'] == 'HYPOTHETICAL' for r in view['plan']['recommendations'])
    at.run()
    assert at.session_state['view']['tool_attempts'] == view['tool_attempts']
