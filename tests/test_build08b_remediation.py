"""Offline mock transport and real local evidence; never a live Muse acceptance claim."""
import json

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from petzyme.agent import run_investigation
from petzyme.agent_evidence import model_evidence
from petzyme.agent_reporting import format_research_report
from petzyme.meta import MAX_COMPLETION_TOKENS, CompletionLimitError, MetaModelClient
from petzyme.ui_adapter import ROOT, investigate_with_backend

QUESTION = ('What does the available NREL and PAZy experimental evidence tell us about ESM050 '
            'at 40°C versus 60°C, pH 6.5, using crystalline PET powder? '
            'What conclusions are supported, and what remains uncertain?')


class TemperatureScenario:
    def __init__(self, *, finish: str = 'stop', final: str | None = None,
                 temperatures: tuple[int, ...] = (40,60)) -> None:
        """Flow: Receives external mock scenario configuration; initializes fixed offline responses for respond()."""
        self.finish, self.final, self.temperatures = finish, final, temperatures
        self.requests, self.sizes = [], []

    def respond(self, request: httpx.Request) -> httpx.Response:
        """Flow: Receives httpx.Request from MetaModelClient.complete() mock transport; returns synthetic httpx.Response without network access."""
        payload=json.loads(request.content)
        self.requests.append(payload);self.sizes.append(len(request.content))
        replies=[json.loads(m['content']) for m in payload['messages'] if m['role']=='tool']
        index=len(replies)
        if index<len(self.temperatures):
            arguments={'query':{'protein_name':'ESM050','temperature_c':self.temperatures[index],
                                'ph':6.5,'form':'powder'},'limit':2 if index==2 else 3}
            message={'role':'assistant','content':None,'tool_calls':[{'id':f'r{index}',
                'type':'function','function':{'name':'investigate_pet_evidence','arguments':json.dumps(arguments)}}]}
            finish='tool_calls'
        else:
            message={'role':'assistant','content':self.final or json.dumps({'fact_ids':[
                f['id'] for r in replies for f in r['facts']]})}
            finish=self.finish
        return httpx.Response(200,json={'model':'muse-spark-1.3','choices':[{'message':message,'finish_reason':finish}]})


def client(scenario: TemperatureScenario) -> MetaModelClient:
    """Flow: Receives TemperatureScenario from offline acceptance tests; returns mocked MetaModelClient for run_investigation() or investigate_with_backend()."""
    return MetaModelClient(transport=httpx.MockTransport(scenario.respond))


def test_two_temperatures_grounded_with_combined_actual_coverage():
    scenario=TemperatureScenario()
    report=run_investigation(QUESTION,client(scenario))['result']
    assert report['status']=='PARTIAL' and report['validation']['grounded']
    assert report['synthesis']['status']=='VALIDATED_FACT_SELECTION'
    assert not any('requested exact conditions' in i for i in report['validation']['issues'])
    assert report['model_requests']==3 and report['tool_attempts']==2
    assert all(e['status']=='PARTIAL' for e in report['trace'])
    view=investigate_with_backend(QUESTION,client(TemperatureScenario()))['result']
    value=view['multi_study'];a=value['accumulation']
    assert a['condition_coverage']['complete'] and a['condition_coverage']['covered']==2
    assert [o['original_text'] for o in value['nrel']['observations']]==['0.08499999999999999','0.13']
    assert all(c['conditions']['covered']==1 for c in a['calls'])
    assert not value['quantitative_superiority_supported'] and not value['causal_claim_supported']
    assert 'PAZy publication-derived assay:' not in format_research_report(report)
    outbound=json.dumps(scenario.requests)
    assert all(s not in outbound for s in ('amino_acid_sequence','sequence_sha256','"rows"','"vector"','/Users/'))


def test_real_duplicate_retrieval_has_no_double_count():
    scenario=TemperatureScenario(temperatures=(40,60,40))
    view=investigate_with_backend(QUESTION,client(scenario))['result']
    a=view['multi_study']['accumulation']
    assert len(scenario.requests)==4 and a['run_count']==3
    assert a['unique_nrel_retrieved']==a['unique_pazy_assays_retrieved']==2
    assert a['unique_pazy_measurements_retrieved']==6
    assert a['unique_comparisons_retrieved']==6


def test_missing_actual_temperature_cannot_be_filled_from_arguments_or_history():
    view=investigate_with_backend(QUESTION,client(TemperatureScenario(temperatures=(50,60))))['result']
    c=view['multi_study']['accumulation']['condition_coverage']
    assert not c['complete'] and c['covered']==1
    assert c['missing_conditions'][0]['temperature_c']==40
    assert any('requested exact conditions' in i for i in view['issues'])


def test_length_limit_stops_without_retry_and_preserves_evidence():
    scenario=TemperatureScenario(finish='length',final='PRIVATE_PARTIAL_PROVIDER_TEXT')
    view=investigate_with_backend(QUESTION,client(scenario))['result']
    assert len(scenario.requests)==3 and view['workflow_status']=='PARTIAL'
    assert view['synthesis']['status']=='INCOMPLETE'
    assert view['synthesis']['reason_code']=='COMPLETION_LENGTH_LIMIT'
    assert view['multi_study']['accumulation']['condition_coverage']['complete']
    assert 'PRIVATE_PARTIAL_PROVIDER_TEXT' not in json.dumps(view)
    assert all(r['max_completion_tokens']==MAX_COMPLETION_TOKENS==2048 for r in scenario.requests)


def test_length_error_is_typed_without_accepting_partial_text():
    scenario=TemperatureScenario(finish='length',temperatures=(),final='PRIVATE_BODY')
    with pytest.raises(CompletionLimitError) as error: client(scenario).complete([],[])
    assert error.value.code=='COMPLETION_LENGTH_LIMIT' and 'PRIVATE_BODY' not in str(error.value)


@pytest.mark.parametrize('final',['Temperature causes superior activity and units are verified',
                                '{"fact_ids":["invented"]}'])
def test_unsupported_final_claims_remain_unaccepted(final):
    report=run_investigation(QUESTION,client(TemperatureScenario(final=final)))['result']
    assert not report['validation']['grounded'] and report['synthesis']['status']=='INCOMPLETE'
    assert final not in format_research_report(report)


def test_equivalent_mock_requests_have_smaller_context_without_budget_change(monkeypatch):
    import petzyme.agent as agent
    old=TemperatureScenario()
    with monkeypatch.context() as patch:
        patch.setattr(agent,'compact_model_evidence',lambda e,f,m:model_evidence(e,f))
        before=run_investigation(QUESTION,client(old))
    new=TemperatureScenario()
    after=run_investigation(QUESTION,client(new))
    assert len(old.requests)==len(new.requests)==3
    assert old.sizes[0]==new.sizes[0] and sum(new.sizes)<sum(old.sizes)*.75
    assert before['result']['facts']==after['result']['facts']
    assert before['result']['validation']==after['result']['validation']
    assert [r['max_completion_tokens'] for r in old.requests]==[r['max_completion_tokens'] for r in new.requests]


@pytest.mark.parametrize('finish,invalid',[('stop',False),('length',True)])
def test_apptest_combined_values_failure_status_and_transience(monkeypatch,finish,invalid):
    from petzyme import ui_adapter
    scenario=TemperatureScenario(finish=finish)
    monkeypatch.setattr(ui_adapter,'run_offline',lambda q,**kw:investigate_with_backend(q,client(scenario),**kw))
    at=AppTest.from_file(ROOT/'tests/legacy_streamlit_app.py',default_timeout=30).run()
    at.text_area(key='question').input(QUESTION)
    at.button(key='investigate').click().run()
    assert not at.exception and len(scenario.requests)==3
    text=' '.join(t.value for t in at.text)
    assert '0.08499999999999999' in text and '0.13' in text
    assert 'Actual NREL requested-condition targets retrieved: 2 / 2; complete: True' in text
    assert any('PARTIAL' in c.value for c in at.caption)
    assert any('not live Muse acceptance' in c.value for c in at.caption)
    assert any(e.value=='Live Muse synthesis incomplete.' for e in at.error) is invalid
    if invalid:
        assert 'completion-token limit' in ' '.join(m.value for m in at.markdown)
    assert not at.json and not at.dataframe and not at.get('download_button')
    assert 'view' not in at.session_state
    at.run()
    assert len(scenario.requests)==3 and not at.text and 'view' not in at.session_state
