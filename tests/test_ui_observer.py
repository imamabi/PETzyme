import json
import pytest
from test_agent import csv_path
from scripts.demo_build03 import QUESTIONS, ScenarioMeta
import httpx
from petzyme.agent import run_investigation
from petzyme.meta import MetaModelClient
from petzyme.agent_evidence import public_tool_update
from petzyme.results import outcome


def run(csv_path, observer=None):
    backend = MetaModelClient(transport=httpx.MockTransport(ScenarioMeta('B').respond))
    return run_investigation(QUESTIONS['B'], backend, csv_path=csv_path, on_tool_event=observer)


def test_observer_detached_and_does_not_change_science_or_decisions(csv_path):
    baseline = run(csv_path)
    updates = []
    def observe(update):
        updates.append(json.loads(json.dumps(update)))
        update['data'].clear()
        update['limitations'].append('observer-only mutation')
    observed = run(csv_path, observe)
    assert observed == baseline
    assert [u['name'] for u in updates] == ['load_nrel', 'parse_nrel', 'investigate_conditions']
    assert [u['tool_attempts'] for u in updates] == [1, 2, 3]
    assert [u['model_requests'] for u in updates] == [1, 2, 3]
    rows = updates[-1]['data']['observations']
    assert rows and rows[0]['original_text'] == '0.085'
    assert 'record=' in rows[0]['source_locator']
    assert 'ACDEF'*20 not in json.dumps(updates)
    assert 'rows' not in updates[0]['data']


def test_observer_failure_cannot_interrupt_inference_workflow(csv_path):
    def broken(update): raise RuntimeError('PRIVATE_ERROR_DO_NOT_LOG')
    result = run(csv_path, broken)
    assert result['status'] == 'SUCCESS'
    assert result['result']['tool_attempts'] == 3 and result['result']['model_requests'] == 4
    assert any('observer failed' in l for l in result['result']['limitations'])
    assert 'PRIVATE_ERROR' not in json.dumps(result)


@pytest.mark.parametrize('name', ['search_pazy_pet', 'get_pazy_metadata', 'get_associated_literature', 'compare_identity'])
def test_public_observer_never_receives_pazy_records(name):
    result = outcome('SUCCESS', {'id': 123, 'name': 'PRIVATE_RECORD', 'sequence': 'ACDEF'*20}, [], ['PRIVATE_TEXT'], None)
    result['reference_counts'] = {'total': 5, 'returned': 2, 'omitted': 3, 'unresolved_returned': 0}
    event = {'call_id': 'c1', 'name': name, 'status': 'SUCCESS', 'ephemeral': True}
    update = public_tool_update(event, result, 2, 1)
    text = json.dumps(update)
    assert 'PRIVATE_' not in text and '123' not in text and 'ACDEF' not in text
    assert update['data']['reference_counts']['omitted'] == 3
