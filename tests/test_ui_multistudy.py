from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP=Path(__file__).resolve().parents[1]/'tests/legacy_streamlit_app.py'


def test_optional_preview_does_not_change_required_worker_fields():
    from zymecycle.ui_adapter import ResearchView
    assert 'multi_study' not in ResearchView.__required_keys__
    assert 'multi_study' in ResearchView.__optional_keys__


@pytest.mark.parametrize('index',[4,5,6])
def test_real_local_multistudy_ui_is_transient_and_offline(index,monkeypatch):
    from zymecycle import ui_adapter
    calls=[];original=ui_adapter.run_investigation
    def run(*a,**k):
        calls.append(1)
        return original(*a,**k)
    def forbidden(*a,**k): pytest.fail('Offline multi-study display attempted Meta/Keychain')
    monkeypatch.setattr(ui_adapter,'run_investigation',run)
    monkeypatch.setattr('zymecycle.meta.MetaModelClient.complete',forbidden)
    monkeypatch.setattr('zymecycle.ui_live.subprocess.Popen',forbidden)
    at=AppTest.from_file(APP,default_timeout=30).run()
    at.button(key=f'example_{index}').click().run()
    assert not calls
    at.button(key='investigate').click().run()
    assert not at.exception and len(calls)==1
    assert any('Multi-Study Evidence' in s.value for s in at.subheader)
    assert any('PARTIAL' in c.value for c in at.caption)
    workflow = next(c.value for c in at.caption if c.value.startswith('Workflow '))
    assert 'backend attempts ' in workflow and 'scientific attempts ' in workflow
    assert ' / ' not in workflow  # Automatic execution has no per-turn attempt quota.
    assert any('live Muse requests: 0' in c.value for c in at.caption)
    assert not at.dataframe and not at.json and not at.get('download_button')
    headings=[m.value for m in at.markdown if m.value.startswith('####')]
    assert headings[:5] == ['#### What has been tested?', '#### What were the results? · MEASURED / COMPUTED',
                           '#### What is interesting or missing? · COMPUTED', '#### Suggested next step', '#### What remains uncertain']
    assert headings[5:]==['#### Source coverage receipts','#### Source-specific experimental evidence','#### Missing evidence and condition coverage','#### Conditions, units and measurement definitions',
        '#### Publication references and source locators','#### Verified and unresolved overlaps',
        '#### Comparability conclusions','#### Critical evidence gaps and verification recommendations']
    assert 'view' not in at.session_state
    assert 'matching_parent_records' not in str(at.session_state)
    text=' '.join(t.value for t in at.text)
    if index==6:
        assert '0.08499999999999999' in text and '0.13' in text and 'UNKNOWN/UNVERIFIED' in text
        assert 'never independent experimental replication' in text
    if index==5:
        assert '4169' in text and 'PUBLICATION-DERIVED' in text
    at.run()
    assert len(calls)==1 and not at.text and 'view' not in at.session_state


def test_narrow_doi_ui_displays_only_bounded_projection_and_truthful_gap(monkeypatch):
    from test_agent_pet import LocalPetScenario
    from zymecycle import ui_adapter
    question=('Inspect multi-study PAZy evidence for ESM050 at 40°C, pH 6.5 using PET powder, '
              'publication DOI 10.1021/acscatal.5c03460, measurement index 0.')
    scenario=LocalPetScenario({'query':{'protein_name':'ESM050','publication_doi':'10.1021/acscatal.5c03460',
        'temperature_c':40,'ph':6.5,'form':'powder','measurement_index':0}})
    # Inject only local test decisions; the application receives the real projection/agent result.
    monkeypatch.setattr(ui_adapter,'run_offline',lambda question,**kwargs:ui_adapter.investigate_with_backend(question,scenario,**kwargs))
    at=AppTest.from_file(APP,default_timeout=30).run()
    at.text_area(key='question').input(question)
    at.button(key='investigate').click().run()
    assert not at.exception and len(scenario.inputs)==2
    text=' '.join(t.value for t in at.text)
    assert '1 / 1 assays' in text and 'Truncated: False' in text
    assert 'Authenticate the requested publication/protocol filters' in text
    assert 'Inspect omitted matching' not in text and 'Inspect omitted NREL' not in text
    assert not at.json and not at.dataframe and not at.get('download_button')
    assert 'view' not in at.session_state and 'matching_parent_records' not in str(at.session_state)
    at.run()
    assert len(scenario.inputs)==2 and not at.text
