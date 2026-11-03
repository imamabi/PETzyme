"""Opt-in REAL pretrained smoke test; never replaced by synthetic inference."""
import json
import os
import pytest
from scripts.demo_build04 import real_demonstrations


@pytest.mark.real_model
@pytest.mark.skipif(os.environ.get('ZYMECYCLE_REAL_ESM')!='1',reason='Enable explicitly for local pretrained smoke: ZYMECYCLE_REAL_ESM=1')
def test_real_pretrained_esm2_and_agent(capsys):
    result=real_demonstrations()
    assert result['execution']=='REAL_PRETRAINED_ESM2_LOCAL'
    assert result['model']['checkpoint']=='facebook/esm2_t12_35M_UR50D'
    assert result['model']['revision']=='6fbf070e65b0b7291e7bbcd451118c216cff79d8'
    assert result['query']=='LCC-ICCG' and result['embedding_shape']==[480] and result['finite']
    assert result['repeat_max_absolute_difference']<=result['reproducibility_absolute_tolerance']
    assert result['identical_sequence_cosine']==pytest.approx(1.,abs=1e-6)
    assert result['cpu_validation']['status']=='SUCCESS' and result['cpu_validation']['device']=='cpu'
    assert result['cpu_validation']['finite']
    assert result['cpu_validation']['cosine_to_default_device']==pytest.approx(1.,abs=1e-6)
    change=result['controlled_modification']
    assert -1<=change['cosine_similarity']<=1 and change['max_absolute_vector_difference']>0
    ranking=result['ranking']
    assert ranking['status']=='SUCCESS' and ranking['result']['neighbors']
    assert all(n['identifier']!='LCC-ICCG' for n in ranking['result']['neighbors'])
    assert ranking['result']['coverage']['candidate_window_records']==8
    assert ranking['result']['coverage']['searched_candidates']<=8
    agent=result['agent_integration']
    assert agent['status']=='SUCCESS' and agent['complete'] and agent['grounded']
    assert agent['mode']=='MOCK_META' and agent['mock_meta_requests']==3
    assert [e['name'] for e in agent['tool_sequence']]==['load_nrel','find_similar_nrel_proteins']
    # Emit only safe scalar/summary results for the Build 04 report.
    with capsys.disabled():
        summary={k:v for k,v in result.items() if k not in ('ranking','agent_integration')}
        summary['ranking']={k:ranking['result'][k] for k in ('coverage','exclusions','devices_used','csv_sha256')}
        summary['ranking']['neighbors']=[{k:n[k] for k in ('identifier','source_row','cosine_similarity')} for n in ranking['result']['neighbors']]
        summary['agent_integration']={k:v for k,v in agent.items() if k!='public_report'}
        print('REAL_ESM2_SMOKE_RESULT='+json.dumps(summary,allow_nan=False))
